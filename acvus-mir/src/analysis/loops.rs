//! The natural loops of a body, and what "loop-invariant" means.
//!
//! There is no loop analysis below this file: a loop is a back edge of the
//! dominator tree and nothing else. An edge `tail -> head` whose head
//! dominates its tail names one, and its body is the head together with
//! every block that reaches the tail without passing the head.
//!
//! Both `optimize::code_motion`, which asks only how deep a block sits, and
//! `optimize::lsr`, which asks for the header, the latches and the body,
//! read the definition from here, so the two passes cannot drift apart on
//! what a loop is or on which values one iteration shares with the next.
//!
//! A retreating edge whose target does not dominate its source is
//! irreducible control flow, which has no natural loop. The lowering emits
//! `while`, `while let` and `for` and nothing else, so [`back_edges`] aborts
//! rather than guess a shape the front end cannot produce.
//!
//! A `for` loop has a back edge like the other two -- its latch jumps to its
//! header -- so it is a natural loop here without a word added. What its
//! terminator gives on top of that is the header without the search and the
//! induction variable without a pattern match: [`for_headers`].

use crate::analysis::domtree::DomTree;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ForSource, InstKind, ValueId};
use acvus_ast::Literal;
use rustc_hash::FxHashMap;

pub struct BackEdge {
    pub tail: BlockIdx,
    pub head: BlockIdx,
}

struct DfsFrame {
    block: BlockIdx,
    next_succ: usize,
}

/// # Panics
/// On irreducible control flow: a retreating edge whose head does not
/// dominate its tail.
pub fn back_edges(cfg: &CfgBody, domtree: &DomTree) -> Vec<BackEdge> {
    let n = cfg.blocks.len();
    if n == 0 {
        return Vec::new();
    }

    let mut edges = Vec::new();
    let mut visited = vec![false; n];
    let mut on_path = vec![false; n];
    let mut stack = vec![DfsFrame {
        block: BlockIdx(0),
        next_succ: 0,
    }];
    visited[0] = true;
    on_path[0] = true;

    while let Some(frame) = stack.last_mut() {
        let block = frame.block;
        let succs = cfg.successors(block);
        let Some(&succ) = succs.get(frame.next_succ) else {
            on_path[block.0] = false;
            stack.pop();
            continue;
        };
        frame.next_succ += 1;

        if on_path[succ.0] {
            assert!(
                domtree.dominates(succ, block),
                "irreducible control flow: the edge from block {} back to block {} \
                 has a head that does not dominate its tail, and the lowering emits \
                 only structured loops",
                block.0,
                succ.0
            );
            edges.push(BackEdge {
                tail: block,
                head: succ,
            });
            continue;
        }

        if !visited[succ.0] {
            visited[succ.0] = true;
            on_path[succ.0] = true;
            stack.push(DfsFrame {
                block: succ,
                next_succ: 0,
            });
        }
    }

    edges
}

pub struct NaturalLoop {
    pub header: BlockIdx,
    pub latches: Vec<BlockIdx>,
    body: Vec<bool>,
}

impl NaturalLoop {
    pub fn contains(&self, block: BlockIdx) -> bool {
        self.body[block.0]
    }

    pub fn block_count(&self) -> usize {
        self.body.iter().filter(|inside| **inside).count()
    }

    pub fn blocks(&self) -> impl Iterator<Item = BlockIdx> + '_ {
        self.body
            .iter()
            .enumerate()
            .filter(|(_, inside)| **inside)
            .map(|(b, _)| BlockIdx(b))
    }
}

/// Why a value is the same on every iteration of a loop, and what it takes
/// to read it above the header.
///
/// `code_motion` lifts an instruction only to a block of no greater loop
/// depth, so what its hoist leaves above a header is [`Invariant::Outside`]
/// and nothing else. A word `Const` is the one thing the hoist cannot lift
/// out of a loop and that is invariant all the same: control equivalence
/// holds it inside the body, because a loop's body does not post-dominate
/// its preheader, while the instruction reads nothing and writes the same
/// word every time it runs. `lsr` needs that case — a literal step `1` and
/// a literal factor `3` both stand inside the body — so the term carries
/// it, with the price named: the constant is re-emitted above the header
/// rather than moved. Which literals are words and which build a heap value
/// is the same split `code_motion::hoistable` makes on `Const`.
#[derive(Clone)]
pub enum Invariant {
    Outside,
    Word(Literal),
}

/// The one answer to "is this value the same on every iteration", built
/// once per body and asked per loop.
pub struct Invariants {
    def_block: FxHashMap<ValueId, BlockIdx>,
    words: FxHashMap<ValueId, Literal>,
}

impl Invariants {
    pub fn of(cfg: &CfgBody) -> Self {
        let words = cfg
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(|inst| match &inst.kind {
                InstKind::Const { dst, value } => match value {
                    Literal::String(_) | Literal::List(_) => None,
                    word => Some((*dst, word.clone())),
                },
                _ => None,
            })
            .collect();
        Self {
            def_block: def_blocks(cfg),
            words,
        }
    }

    /// # Panics
    /// If `value` has no definition in this body.
    pub fn def_block(&self, value: ValueId) -> BlockIdx {
        *self
            .def_block
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is used but never defined"))
    }

    pub fn at(&self, loop_: &NaturalLoop, value: ValueId) -> Option<Invariant> {
        if !loop_.contains(self.def_block(value)) {
            return Some(Invariant::Outside);
        }
        self.words.get(&value).cloned().map(Invariant::Word)
    }
}

/// The loop headers a terminator names, with the traversal each one is
/// (RFC-0057 rule 3). A `for` header is a loop header by what ends it, so
/// a reader that wants the loop and its induction variable asks the
/// terminator rather than searching for a back edge.
pub fn for_headers(cfg: &CfgBody) -> FxHashMap<BlockIdx, ForSource> {
    cfg.blocks
        .iter()
        .enumerate()
        .filter_map(|(bi, block)| match &block.terminator {
            Terminator::For { source, .. } => Some((BlockIdx(bi), *source)),
            _ => None,
        })
        .collect()
}

pub fn natural_loops_innermost_first(cfg: &CfgBody, domtree: &DomTree) -> Vec<NaturalLoop> {
    let n = cfg.blocks.len();
    let preds = cfg.predecessors();
    let mut by_header: FxHashMap<usize, NaturalLoop> = FxHashMap::default();

    for edge in back_edges(cfg, domtree) {
        let loop_ = by_header.entry(edge.head.0).or_insert_with(|| {
            let mut body = vec![false; n];
            body[edge.head.0] = true;
            NaturalLoop {
                header: edge.head,
                latches: Vec::new(),
                body,
            }
        });
        loop_.latches.push(edge.tail);

        let mut stack = Vec::new();
        if !loop_.body[edge.tail.0] {
            loop_.body[edge.tail.0] = true;
            stack.push(edge.tail);
        }
        while let Some(b) = stack.pop() {
            for &p in preds.get(&b).into_iter().flatten() {
                if !loop_.body[p.0] {
                    loop_.body[p.0] = true;
                    stack.push(p);
                }
            }
        }
    }

    let mut loops: Vec<NaturalLoop> = by_header.into_values().collect();
    loops.sort_by_key(|l| (l.block_count(), l.header.0));
    loops
}

pub struct LoopDepth {
    per_block: Vec<usize>,
}

impl LoopDepth {
    pub fn of(cfg: &CfgBody, domtree: &DomTree) -> Self {
        let mut per_block = vec![0usize; cfg.blocks.len()];
        for loop_ in natural_loops_innermost_first(cfg, domtree) {
            for b in loop_.blocks() {
                per_block[b.0] += 1;
            }
        }
        Self { per_block }
    }

    pub fn at(&self, block: BlockIdx) -> usize {
        self.per_block[block.0]
    }
}

pub fn def_blocks(cfg: &CfgBody) -> FxHashMap<ValueId, BlockIdx> {
    let mut def_block = FxHashMap::default();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let idx = BlockIdx(bi);
        for &p in &block.params {
            def_block.insert(p, idx);
        }
        for inst in &block.insts {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                def_block.insert(d, idx);
            }
        }
    }
    for v in cfg.entry_defs() {
        def_block.entry(v).or_insert(BlockIdx(0));
    }
    def_block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn label(n: u32) -> InstKind {
        InstKind::BlockLabel {
            label: Label(n),
            params: vec![],
        }
    }

    fn cfg_of(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..32 {
            factory.next();
        }
        promote(MirBody {
            demoted_diamonds: Default::default(),
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    const OUTER_HEADER: u32 = 0;
    const INNER_HEADER: u32 = 1;
    const OUTER_LATCH: u32 = 2;
    const AFTER: u32 = 3;

    fn nested_loops() -> CfgBody {
        cfg_of(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Jump {
                label: Label(OUTER_HEADER),
                args: vec![],
            },
            label(OUTER_HEADER),
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(INNER_HEADER),
                then_args: vec![],
                else_label: Label(AFTER),
                else_args: vec![],
            },
            label(INNER_HEADER),
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(INNER_HEADER),
                then_args: vec![],
                else_label: Label(OUTER_LATCH),
                else_args: vec![],
            },
            label(OUTER_LATCH),
            InstKind::Jump {
                label: Label(OUTER_HEADER),
                args: vec![],
            },
            label(AFTER),
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ])
    }

    #[test]
    fn the_header_is_in_its_own_body() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let loops = natural_loops_innermost_first(&cfg, &domtree);
        assert_eq!(loops.len(), 2);
        for loop_ in &loops {
            assert!(
                loop_.contains(loop_.header),
                "the loop headed at block {} does not run its own header",
                loop_.header.0
            );
        }
    }

    #[test]
    fn an_inner_loop_comes_before_the_outer_one() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let loops = natural_loops_innermost_first(&cfg, &domtree);
        let [inner, outer] = &loops[..] else {
            panic!("two nested loops, got {}", loops.len());
        };
        assert!(
            outer.contains(inner.header) && !inner.contains(outer.header),
            "the first loop is the one the second contains"
        );
    }

    #[test]
    fn a_value_defined_above_the_header_is_invariant() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let invariants = Invariants::of(&cfg);
        for loop_ in natural_loops_innermost_first(&cfg, &domtree) {
            assert!(matches!(
                invariants.at(&loop_, v(0)),
                Some(Invariant::Outside)
            ));
        }
    }
}
