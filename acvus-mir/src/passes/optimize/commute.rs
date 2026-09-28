//! Along each token chain, two calls adjacent on it, the later consuming
//! the token the earlier yields and nothing else consuming it, are offered
//! to `swap` when trading their places shortens the block's critical path.
//! Only `swap` decides whether the declarations let them trade places
//! (effects §1); it refuses the rest.
//!
//! The measure is the block's critical path: the longest def-use path
//! among its instructions and its terminator, order tokens included, each
//! counting one. Before offering a swap, the block's graph is taken as it
//! would stand after it: the later call consumes what the earlier consumed,
//! the earlier consumes what the later yields, and every other use of the
//! later call's yield takes the earlier call's. A swap is offered only when
//! that graph's critical path is strictly shorter. Every path the swap
//! changes passes one of the two calls, so a shorter critical path of the
//! block is a shorter one through the chain. A swap that would put a call
//! after a value it reads leaves a cycle in the graph, has no measure, and
//! is not offered.
//!
//! Each block is walked in rounds, offering each adjacent pair in block
//! order, until a round swaps nothing. Every swap made strictly lowers the
//! measure, a natural number no larger than the block's instructions plus
//! one, so a block takes at most that many swaps and one round more.
//!
//! Form: F2, loops canonical, in and out; it changes no block.
//!
//! Cost: per block of n instructions and e operands, O(n + e) per pair
//! measured, O(n) pairs per round, and O(n) rounds: O(n² · (n + e)).
//!
//! Meaning: two calls whose declared effects commute take effect in either
//! order (effects §1), and `swap` rethreads their tokens so that every
//! token is still defined before its uses (effects §3); no other
//! instruction moves. No type changes.

use rustc_hash::FxHashMap;

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotSwap, Editor, Effect};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Module, OrderUse, Site};
use crate::ir::OrderEdge;
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Commute;

impl BodyTransform for Commute {
    const NAME: &'static str = "commute";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let blocks: Vec<BlockId> = module.body(body).blocks().map(|(block, _)| block).collect();
        let mut facts = EditFacts::new(am, module, body);
        let mut editor = Editor::<Effect>::new(module, body, &mut facts);
        for block in blocks {
            while commute_round(&mut editor, block) {}
        }
        editor.finish()
    }
}

/// Offers every adjacent pair of `block`, in block order, whose swap
/// shortens the block's critical path; whether any swap was made.
fn commute_round(editor: &mut Editor<'_, Effect>, block: BlockId) -> bool {
    let mut swapped = false;
    let mut index = 0;
    while let Some(earlier) = editor.inst_at(block, index) {
        index += 1;
        let Some(pair) = ChainedPair::at(editor.body(), block, earlier.index()) else {
            continue;
        };
        let graph = BlockGraph::of(editor.body(), block);
        let now = graph.critical_path(None).expect("a body defines every value before its uses, so its graph has no cycle");
        let shorter = graph.critical_path(Some(&pair)).is_some_and(|after| after < now);
        if !shorter {
            continue;
        }
        let later = editor.inst_at(block, pair.later).expect("the later call stands in the block");
        match editor.swap(earlier, later) {
            Ok(()) => swapped = true,
            Err(CannotSwap::NotAdjacent | CannotSwap::Ordered | CannotSwap::DependsOn(_)) => {}
        }
    }
    swapped
}

/// Two calls of one block adjacent on a token chain, by index: `later`
/// consumes the token `earlier` yields, and nothing else does.
struct ChainedPair {
    earlier: usize,
    later: usize,
    earlier_edge: OrderEdge,
    later_edge: OrderEdge,
}

impl ChainedPair {
    /// The pair whose earlier call stands at `index` of `block`, where one
    /// is. Cost: O(1).
    fn at(body: &Body, block: BlockId, index: usize) -> Option<ChainedPair> {
        let Some(OrderUse::Edge(earlier_edge)) = body.block(block).insts()[index].order() else {
            return None;
        };
        let mut users = body.users(earlier_edge.after);
        let (Some(Site::Inst(at, later)), None) = (users.next(), users.next()) else {
            return None;
        };
        if at != block {
            return None;
        }
        let Some(OrderUse::Edge(later_edge)) = body.block(block).insts()[later].order() else {
            return None;
        };
        (later_edge.before == earlier_edge.after).then_some(ChainedPair { earlier: index, later, earlier_edge, later_edge })
    }

    /// The operand `value` of the instruction or terminator `node` stands
    /// for once the pair has traded places.
    fn after_swap(&self, node: Node, value: ValueId) -> ValueId {
        match node {
            Node::Inst(inst) if inst == self.later && value == self.earlier_edge.after => self.earlier_edge.before,
            Node::Inst(inst) if inst == self.earlier && value == self.earlier_edge.before => self.later_edge.after,
            Node::Inst(inst) if inst == self.earlier || inst == self.later => value,
            Node::Inst(_) | Node::Terminator if value == self.later_edge.after => self.earlier_edge.after,
            Node::Inst(_) | Node::Terminator => value,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Node {
    Inst(usize),
    Terminator,
}

/// A block's instructions and terminator with the values each reads, and
/// the instruction of the block that defines each value it defines.
struct BlockGraph {
    operands: Vec<Vec<ValueId>>,
    defined_at: FxHashMap<ValueId, usize>,
}

impl BlockGraph {
    /// The terminator is the last node. Cost: O(instructions + operands).
    fn of(body: &Body, block: BlockId) -> BlockGraph {
        let record = body.block(block);
        let mut defined_at = FxHashMap::default();
        let mut operands = Vec::with_capacity(record.insts().len() + 1);
        for (index, inst) in record.insts().iter().enumerate() {
            let defined = inst.result().into_iter().chain(inst.order().and_then(OrderUse::yielded));
            defined_at.extend(defined.map(|value| (value, index)));
            operands.push(inst.operands().into_vec());
        }
        operands.push(record.terminator().operands().into_vec());
        BlockGraph { operands, defined_at }
    }

    fn node(&self, index: usize) -> Node {
        match index + 1 == self.operands.len() {
            true => Node::Terminator,
            false => Node::Inst(index),
        }
    }

    /// The longest def-use path, each node counting one, with `swapped`
    /// traded where given; none where the graph then has a cycle. Kahn's
    /// order over the nodes. Cost: O(nodes + operands).
    fn critical_path(&self, swapped: Option<&ChainedPair>) -> Option<usize> {
        let count = self.operands.len();
        let mut definers: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); count];
        for (at, operands) in self.operands.iter().enumerate() {
            let node = self.node(at);
            for operand in operands {
                let value = swapped.map_or(*operand, |pair| pair.after_swap(node, *operand));
                if let Some(definer) = self.defined_at.get(&value) {
                    definers[at].push(*definer);
                    users[*definer].push(at);
                }
            }
        }
        let mut waiting: Vec<usize> = definers.iter().map(Vec::len).collect();
        let mut depth = vec![0; count];
        let mut pending: Vec<usize> = (0..count).filter(|node| waiting[*node] == 0).collect();
        let mut done = 0;
        while let Some(node) = pending.pop() {
            done += 1;
            depth[node] = 1 + definers[node].iter().map(|definer| depth[*definer]).fold(0, usize::max);
            for user in &users[node] {
                waiting[*user] -= 1;
                if waiting[*user] == 0 {
                    pending.push(*user);
                }
            }
        }
        (done == count).then(|| depth.into_iter().fold(0, usize::max))
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::Span;

    use super::*;
    use crate::ir::cfg::{Inst, Terminator};
    use crate::passes::optimize::tests::{Fixture, assert_whole, direct_call, int_const, ret, wrapping_add};
    use crate::ty::Ty;

    /// A body `two_calls` built, and the results of its two calls.
    struct Built {
        id: crate::ir::cfg::BodyId,
        p: ValueId,
        q: ValueId,
    }

    /// ```text
    /// a = 1; b = a + a; c = b + b; d = c + c; p = A(d); q = B();
    /// r = q + q; s = r + r; u = s + s; v = u + p; return v
    /// ```
    /// with `A` typed `with_argument` and `B` typed `without`. The critical
    /// path runs a..d, A, B, r..v and the return: 11. With B first it is
    /// a..d, A, v and the return: 7.
    fn two_calls(f: &mut Fixture, with_argument: Ty, without: Ty) -> Built {
        let (int, name) = (f.int, f.name("f"));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push(int_const(1), int, Span::ZERO);
        let b = builder.push(wrapping_add(a, a), int, Span::ZERO);
        let c = builder.push(wrapping_add(b, b), int, Span::ZERO);
        let d = builder.push(wrapping_add(c, c), int, Span::ZERO);
        let p = builder.push_effect(direct_call(name, with_argument, vec![d]), int, Span::ZERO);
        let q = builder.push_effect(direct_call(name, without, Vec::new()), int, Span::ZERO);
        let r = builder.push(wrapping_add(q, q), int, Span::ZERO);
        let s = builder.push(wrapping_add(r, r), int, Span::ZERO);
        let u = builder.push(wrapping_add(s, s), int, Span::ZERO);
        let v = builder.push(wrapping_add(u, p), int, Span::ZERO);
        builder.terminate(ret(v));
        Built { id: builder.seal().expect("the body has no open state"), p, q }
    }

    fn edge(inst: &Inst) -> OrderEdge {
        match inst.order() {
            Some(OrderUse::Edge(edge)) => edge,
            other => panic!("{:?} is on the chain, not {other:?}", inst.op()),
        }
    }

    #[test]
    fn commutative_calls_trade_places_when_that_shortens_the_critical_path() {
        let mut f = Fixture::new();
        let (with_argument, without) = (f.commuting_unary, f.commuting);
        let Built { id, p, q } = two_calls(&mut f, with_argument, without);
        let entry = f.module.body(id).entry();
        let o0 = f.module.body(id).order_param().expect("the body is ordered");

        let changes = f.run::<Commute>(id);

        let body = f.module.body(id);
        let insts = body.block(entry).insts();
        assert_eq!((insts[4].result(), insts[5].result()), (Some(q), Some(p)), "B now comes first");
        let (first, second) = (edge(&insts[4]), edge(&insts[5]));
        assert_eq!(first.before, o0);
        assert_eq!(second.before, first.after);
        assert!(matches!(body.block(entry).terminator(), Terminator::Return { order: Some(order), .. } if *order == second.after));
        assert!(changes.touched(crate::ir::cfg::Aspect::OrderTokens));
        assert_whole(&f.module, id);
    }

    #[test]
    fn calls_not_declared_commutative_keep_their_order() {
        let mut f = Fixture::new();
        let (with_argument, without) = (f.opaque_unary, f.opaque);
        let Built { id, .. } = two_calls(&mut f, with_argument, without);
        let before = f.snapshot(id);

        f.run::<Commute>(id);

        assert_eq!(f.snapshot(id), before);
    }

    /// `p = A(); q = B(); return p + q`: either order gives the same
    /// critical path, so the calls stay.
    #[test]
    fn commutative_calls_stay_where_a_swap_shortens_nothing() {
        let mut f = Fixture::new();
        let (int, commuting, name) = (f.int, f.commuting, f.name("f"));
        let mut builder = f.module.build_body(true, int);
        let p = builder.push_effect(direct_call(name, commuting, Vec::new()), int, Span::ZERO);
        let q = builder.push_effect(direct_call(name, commuting, Vec::new()), int, Span::ZERO);
        let sum = builder.push(wrapping_add(p, q), int, Span::ZERO);
        builder.terminate(ret(sum));
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<Commute>(id);

        assert_eq!(f.snapshot(id), before);
    }
}
