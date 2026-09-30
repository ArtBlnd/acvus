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
//! after a value it reads is not offered: `swap` refuses it
//! (`CannotSwap::DependsOn`).
//!
//! Each block is walked in rounds, offering each adjacent pair in block
//! order, until a round swaps nothing. Every swap made strictly lowers the
//! measure, a natural number no larger than the block's instructions plus
//! one, so a block takes at most that many swaps and one round more.
//!
//! A round starts from each node's earliest time (ASAP: the longest path
//! ending at it) and latest time (ALAP: the longest path starting at it).
//! A pair is judged from them alone. The later call reads nothing defined
//! from the earlier one on, so after the swap no path through a neighbour
//! of the two passes either of them, and the times of those neighbours
//! stand: the longest path through the swapped pair is their new times'
//! sum, from the neighbours' times. Every other path is unchanged, and the
//! critical path shortens exactly when the new paths through the pair are
//! shorter and no critical path avoids both calls. That holds where one of
//! them is critical and the only critical node of its time, since every
//! critical path has one node of each time, and fails where neither is
//! critical. In the remaining case, both calls' times held by other
//! critical nodes too, the pair's graph is measured whole.
//!
//! Form: F2, loops canonical, in and out; it changes no block.
//!
//! Cost: per block of n instructions and e operands, O(n + e) per round for
//! the times and O(the pair's operands) per pair judged, plus O(n + e) for
//! a pair whose times other critical nodes share; a swap made strictly
//! lowers the measure, so every node's slack changes, and the times are
//! taken again, O(n + e), at most once per swap, which the measure bounds.
//!
//! Meaning: two calls whose declared effects commute take effect in either
//! order (effects §1), and `swap` rethreads their tokens so that every
//! token is still defined before its uses (effects §3); no other
//! instruction moves. No type changes.

use rustc_hash::FxHashMap;

use crate::ir::ValueId;
use crate::ir::cfg::edit::{CannotSwap, Editor, Effect};
use crate::ir::cfg::{BlockId, Body, BodyId, Changes, InstId, Module, OrderUse, Site};
use crate::ir::OrderEdge;
use crate::passes::analysis::AnalysisManager;
use crate::passes::edit_facts::EditFacts;
use crate::passes::optimize::BodyTransform;

pub struct Commute;

impl BodyTransform for Commute {
    const NAME: &'static str = "commute";

    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes {
        let blocks: Vec<BlockId> = module.body(body).blocks().map(|(block, _)| block).collect();
        let mut facts = EditFacts::new(am, body);
        let mut editor = Editor::<Effect>::new(module, body, &mut facts);
        for block in blocks {
            while commute_round(&mut editor, block) {}
        }
        editor.finish()
    }
}

/// Offers every adjacent pair of `block`, in block order, whose swap
/// shortens the block's critical path; whether any swap was made. After a
/// swap the walk looks again at the earlier call, which the later one now
/// stands before.
fn commute_round(editor: &mut Editor<'_, Effect>, block: BlockId) -> bool {
    let mut swapped = false;
    let mut times: Option<Times> = None;
    let mut next = editor.first_inst(block);
    while let Some(earlier) = next {
        next = editor.next_inst(earlier);
        let Some(pair) = ChainedPair::at(editor.body(), block, earlier.id()) else {
            continue;
        };
        let times = times.get_or_insert_with(|| Times::of(editor.body(), block));
        if !times.shortened_by(editor.body(), block, &pair) {
            continue;
        }
        let later = editor.inst_ref(pair.later);
        match editor.swap(earlier, later) {
            Ok(()) => {
                swapped = true;
                *times = Times::of(editor.body(), block);
                next = Some(earlier);
            }
            Err(CannotSwap::NotAdjacent | CannotSwap::Ordered | CannotSwap::DependsOn(_)) => {}
        }
    }
    swapped
}

/// Two calls of one block adjacent on a token chain: `later` consumes the
/// token `earlier` yields, and nothing else does.
struct ChainedPair {
    earlier: InstId,
    later: InstId,
    earlier_edge: OrderEdge,
    later_edge: OrderEdge,
}

impl ChainedPair {
    /// The pair whose earlier call is `earlier` of `block`, where one is.
    /// Cost: O(1).
    fn at(body: &Body, block: BlockId, earlier: InstId) -> Option<ChainedPair> {
        let Some(OrderUse::Edge(earlier_edge)) = body.inst(earlier).order() else {
            return None;
        };
        let mut users = body.users(earlier_edge.after);
        let (Some(Site::Inst(later)), None) = (users.next(), users.next()) else {
            return None;
        };
        if body.block_of(later) != block {
            return None;
        }
        let Some(OrderUse::Edge(later_edge)) = body.inst(later).order() else {
            return None;
        };
        (later_edge.before == earlier_edge.after).then_some(ChainedPair { earlier, later, earlier_edge, later_edge })
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
    Inst(InstId),
    Terminator,
}

/// A block's instructions and terminator with the values each reads, and
/// the node of the block that defines each value it defines. The
/// terminator is the last node.
struct BlockGraph {
    nodes: Vec<Node>,
    operands: Vec<Vec<ValueId>>,
    defined_at: FxHashMap<ValueId, usize>,
    index_of: FxHashMap<InstId, usize>,
}

impl BlockGraph {
    /// Cost: O(instructions + operands).
    fn of(body: &Body, block: BlockId) -> BlockGraph {
        let record = body.block(block);
        let mut defined_at = FxHashMap::default();
        let mut index_of = FxHashMap::default();
        let mut nodes = Vec::with_capacity(record.insts().len() + 1);
        let mut operands = Vec::with_capacity(record.insts().len() + 1);
        for (index, inst) in record.insts().enumerate() {
            let defined = inst.result().into_iter().chain(inst.order().and_then(OrderUse::yielded));
            defined_at.extend(defined.map(|value| (value, index)));
            index_of.insert(inst.id(), index);
            nodes.push(Node::Inst(inst.id()));
            operands.push(inst.operands().into_vec());
        }
        nodes.push(Node::Terminator);
        operands.push(record.terminator().operands().into_vec());
        BlockGraph { nodes, operands, defined_at, index_of }
    }

    /// The nodes that define what node `at` reads, with `swapped` traded
    /// where given.
    fn definers<'g>(&'g self, at: usize, swapped: Option<&'g ChainedPair>) -> impl Iterator<Item = usize> + 'g {
        let node = self.nodes[at];
        self.operands[at].iter().filter_map(move |operand| {
            let value = swapped.map_or(*operand, |pair| pair.after_swap(node, *operand));
            self.defined_at.get(&value).copied()
        })
    }

    /// The longest def-use path, each node counting one, with `swapped`
    /// traded where given; none where the graph then has a cycle. Kahn's
    /// order over the nodes. Cost: O(nodes + operands).
    fn critical_path(&self, swapped: Option<&ChainedPair>) -> Option<usize> {
        let count = self.nodes.len();
        let mut definers: Vec<Vec<usize>> = vec![Vec::new(); count];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); count];
        for at in 0..count {
            for definer in self.definers(at, swapped) {
                definers[at].push(definer);
                users[definer].push(at);
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

/// A block's graph with each node's earliest and latest times: the longest
/// path ending at it and the longest starting at it, each node counting
/// one; the critical path; and how many critical nodes each earliest time
/// has.
struct Times {
    graph: BlockGraph,
    earliest: Vec<usize>,
    latest: Vec<usize>,
    critical_path: usize,
    critical_at: Vec<usize>,
}

impl Times {
    /// A node's definers stand before it in the block, so one pass in block
    /// order and one against it take the times. Cost: O(instructions +
    /// operands).
    fn of(body: &Body, block: BlockId) -> Times {
        let graph = BlockGraph::of(body, block);
        let count = graph.nodes.len();
        let mut earliest = vec![0; count];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); count];
        for at in 0..count {
            let mut longest = 0;
            for definer in graph.definers(at, None) {
                longest = longest.max(earliest[definer]);
                users[definer].push(at);
            }
            earliest[at] = 1 + longest;
        }
        let mut latest = vec![0; count];
        for at in (0..count).rev() {
            latest[at] = 1 + users[at].iter().map(|user| latest[*user]).fold(0, usize::max);
        }
        let critical_path = earliest.iter().copied().fold(0, usize::max);
        let mut critical_at = vec![0; critical_path + 1];
        for at in 0..count {
            if earliest[at] + latest[at] - 1 == critical_path {
                critical_at[earliest[at]] += 1;
            }
        }
        Times { graph, earliest, latest, critical_path, critical_at }
    }

    fn critical(&self, at: usize) -> bool {
        self.earliest[at] + self.latest[at] - 1 == self.critical_path
    }

    /// Whether every critical path passes the node at `at`: it is
    /// critical, and no other critical node has its earliest time.
    fn on_every_critical_path(&self, at: usize) -> bool {
        self.critical(at) && self.critical_at[self.earliest[at]] == 1
    }

    /// Whether trading `pair`'s places would strictly shorten the block's
    /// critical path. Cost: O(the pair's operands and users), and O(nodes +
    /// operands) where the times do not decide it.
    fn shortened_by(&self, body: &Body, block: BlockId, pair: &ChainedPair) -> bool {
        let graph = &self.graph;
        let (earlier, later) = (graph.index_of[&pair.earlier], graph.index_of[&pair.later]);
        let reads_since_earlier = graph.operands[later]
            .iter()
            .filter(|operand| **operand != pair.later_edge.before)
            .filter_map(|operand| graph.defined_at.get(operand))
            .any(|definer| *definer >= earlier);
        if reads_since_earlier || later < earlier {
            return false;
        }
        if !self.critical(earlier) && !self.critical(later) {
            return false;
        }
        let through_pair = self.through_swapped(body, block, pair, earlier, later);
        if through_pair >= self.critical_path {
            return false;
        }
        if self.on_every_critical_path(earlier) || self.on_every_critical_path(later) {
            return true;
        }
        graph.critical_path(Some(pair)).is_some_and(|after| after < self.critical_path)
    }

    /// The longest path through either call once the two have traded
    /// places, from their neighbours' times, which the trade leaves as
    /// they are. A value defined outside the block, or read only outside
    /// it, adds no node to a path of the block.
    fn through_swapped(&self, body: &Body, block: BlockId, pair: &ChainedPair, earlier: usize, later: usize) -> usize {
        let graph = &self.graph;
        let longest_before = |values: &mut dyn Iterator<Item = ValueId>| -> usize {
            values.filter_map(|value| graph.defined_at.get(&value)).map(|definer| self.earliest[*definer]).fold(0, usize::max)
        };
        let data = |at: usize, token: ValueId| graph.operands[at].iter().copied().filter(move |operand| *operand != token);
        let later_earliest = 1 + longest_before(
            &mut data(later, pair.later_edge.before).chain(std::iter::once(pair.earlier_edge.before)),
        );
        let earlier_earliest = 1 + longest_before(&mut data(earlier, pair.earlier_edge.before)).max(later_earliest);
        let longest_after = |value: ValueId| -> usize {
            body.users(value)
                .filter_map(|site| match site {
                    Site::Inst(user) => graph.index_of.get(&user).copied(),
                    Site::Term(term) => (term == block).then_some(graph.nodes.len() - 1),
                })
                .map(|user| self.latest[user])
                .fold(0, usize::max)
        };
        let result_after = |inst: InstId| body.inst(inst).result().into_iter().map(longest_after).fold(0, usize::max);
        let earlier_latest = 1 + result_after(pair.earlier).max(longest_after(pair.later_edge.after));
        let later_latest = 1 + result_after(pair.later).max(earlier_latest);
        (later_earliest + later_latest - 1).max(earlier_earliest + earlier_latest - 1)
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
        builder.terminate(ret(v), acvus_ast::Span::ZERO);
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
        let insts: Vec<_> = body.block(entry).insts().collect();
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
        builder.terminate(ret(sum), acvus_ast::Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let before = f.snapshot(id);

        f.run::<Commute>(id);

        assert_eq!(f.snapshot(id), before);
    }
}
