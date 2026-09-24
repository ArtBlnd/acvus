//! CFG - basic-block-based IR representation.
//!
//! The single CFG representation used by all analysis and optimization passes.
//! MirBody (flat Vec<Inst>) is promoted to CfgBody where each basic block owns
//! its instructions, eliminating index arithmetic entirely.
//!
//! Lifecycle: MirBody -> promote -> CfgBody -> (passes) -> demote -> MirBody

use acvus_utils::{Astr, LocalFactory};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::ir::{
    DebugInfo, ExitTrip, ForSource, Inst, InstKind, Label, MirBody, Stages, SwitchKey, ValueId,
};
use crate::ty::{Task, Ty};

// -- BlockIdx ------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockIdx(pub usize);

// -- Block ---------------------------------------------------------

/// Sentinel label for the entry block (no BlockLabel instruction in MirBody).
pub const ENTRY_LABEL: Label = Label(u32::MAX);

/// A basic block that owns its instructions.
#[derive(Debug, Clone)]
pub struct Block {
    /// Unique block label. Entry block uses `ENTRY_LABEL`.
    pub label: Label,
    /// SSA block parameters (PHI equivalent).
    pub params: Vec<ValueId>,
    /// Body instructions (no control flow).
    pub insts: Vec<Inst>,
    /// Block terminator (control flow).
    pub terminator: Terminator,
}

// -- Terminator ----------------------------------------------------

/// Block terminator - extracted from InstKind control flow variants.
#[derive(Debug, Clone)]
pub enum Terminator {
    Jump {
        label: Label,
        args: Vec<ValueId>,
    },
    JumpIf {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
    },
    /// See [`crate::ir::InstKind::Diamond`] for what `join` obliges of a
    /// pass that rewrites labels.
    Diamond {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
        join: Label,
    },
    /// One dispatch (RFC-0051): the value is read once and the block leaves
    /// through the arm whose key it holds. `default` is the edge a key
    /// outside `arms` takes, and it is present exactly when the `match` had
    /// a catch-all -- a `Switch` without one is exhaustive over `arms`,
    /// which is what `validate`'s exhaustiveness pass decides.
    Switch {
        tag: ValueId,
        arms: Vec<(SwitchKey, Label, Vec<ValueId>)>,
        default: Option<(Label, Vec<ValueId>)>,
    },
    /// One traversal (RFC-0057). The terminator is the loop's condition: no
    /// instruction writes the comparison, the element read or the advance.
    /// It fills the body block's parameters itself -- the element and the
    /// counter, which `ForSource::supplied_params` counts -- and the body
    /// reads the carried values as the header's parameters. The body is a
    /// chain of `stages` (RFC-0089 rule 1). Where `exit_trip` is `Defined` it
    /// fills the exit block's leading parameter with the trip count
    /// (RFC-0057 rule 9), and `exit_args` follow it.
    For {
        source: ForSource,
        stages: Stages,
        exit: Label,
        exit_trip: ExitTrip,
        exit_args: Vec<ValueId>,
    },
    Return {
        value: ValueId,
        order: Option<ValueId>,
        span: acvus_ast::Span,
    },
    /// The block does not continue: it ended in a `!` (RFC-0038).
    Diverge,
    /// Implicit fallthrough to next block.
    Fallthrough,
}

// -- CfgBody -------------------------------------------------------

/// Basic-block-based IR body for analysis and optimization passes.
#[derive(Debug, Clone)]
pub struct CfgBody {
    pub blocks: Vec<Block>,
    pub label_to_block: FxHashMap<Label, BlockIdx>,
    pub val_types: FxHashMap<ValueId, Ty>,
    /// Function parameters: (name, register).
    pub params: Vec<(Astr, ValueId)>,
    /// Captured variables: (name, register).
    pub captures: Vec<(Astr, ValueId)>,
    /// The `Order` the body takes first when its effect is not Pure.
    pub order_param: Option<ValueId>,
    /// The join of the tasks of everything the body does (RFC-0046).
    pub task: Task,
    pub debug: DebugInfo,
    pub val_factory: LocalFactory<ValueId>,
    /// See [`crate::ir::MirBody::demoted_diamonds`], which this is carried to
    /// and from.
    pub demoted_diamonds: FxHashSet<Label>,
}

impl CfgBody {
    /// Successors of a block.
    pub fn successors(&self, idx: BlockIdx) -> SmallVec<[BlockIdx; 2]> {
        let block = &self.blocks[idx.0];
        let mut succs = SmallVec::new();
        match &block.terminator {
            Terminator::Jump { label, .. } => {
                if let Some(&bi) = self.label_to_block.get(label) {
                    succs.push(bi);
                }
            }
            Terminator::JumpIf {
                then_label,
                else_label,
                ..
            }
            | Terminator::Diamond {
                then_label,
                else_label,
                ..
            } => {
                if let Some(&bi) = self.label_to_block.get(then_label) {
                    succs.push(bi);
                }
                if let Some(&bi) = self.label_to_block.get(else_label) {
                    succs.push(bi);
                }
            }
            Terminator::Switch { arms, default, .. } => {
                let edges = arms
                    .iter()
                    .map(|(_, label, _)| label)
                    .chain(default.iter().map(|(label, _)| label));
                for label in edges {
                    if let Some(&bi) = self.label_to_block.get(label) {
                        succs.push(bi);
                    }
                }
            }
            Terminator::For { stages, exit, .. } => {
                for label in [stages.body(), *exit] {
                    if let Some(&bi) = self.label_to_block.get(&label) {
                        succs.push(bi);
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < self.blocks.len() {
                    succs.push(BlockIdx(next));
                }
            }
            Terminator::Return { .. } | Terminator::Diverge => {}
        }
        succs
    }

    /// Build a predecessors map from the CFG.
    /// The values defined before block 0 runs: params, captures, and the
    /// entry `Order`.
    pub fn entry_defs(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.params
            .iter()
            .chain(self.captures.iter())
            .map(|(_, v)| *v)
            .chain(self.order_param)
    }

    pub fn predecessors(&self) -> FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>> {
        let mut preds: FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>> = FxHashMap::default();
        for i in 0..self.blocks.len() {
            let idx = BlockIdx(i);
            for succ in self.successors(idx) {
                preds.entry(succ).or_default().push(idx);
            }
        }
        preds
    }
}

// -- Promote: MirBody -> CfgBody -----------------------------------

/// RFC-0063 rule 1 admits a `Diamond` only where the arms rejoin, so a
/// pass that dissolves the join calls this: what is left is a `JumpIf`.
pub fn demote_diamond(cfg: &mut CfgBody, at: BlockIdx) {
    let block = &mut cfg.blocks[at.0];
    let Terminator::Diamond {
        cond,
        then_label,
        then_args,
        else_label,
        else_args,
        ..
    } = &mut block.terminator
    else {
        return;
    };
    let demoted = Terminator::JumpIf {
        cond: *cond,
        then_label: *then_label,
        then_args: std::mem::take(then_args),
        else_label: *else_label,
        else_args: std::mem::take(else_args),
    };
    block.terminator = demoted;
    let demoted_at = block.label;
    cfg.demoted_diamonds.insert(demoted_at);
}

pub fn promote(body: MirBody) -> CfgBody {
    let mut blocks: Vec<Block> = Vec::new();
    let mut current_label: Label = ENTRY_LABEL;
    let mut current_params: Vec<ValueId> = Vec::new();
    let mut current_insts: Vec<Inst> = Vec::new();

    let mut label_to_block: FxHashMap<Label, BlockIdx> = FxHashMap::default();
    label_to_block.insert(ENTRY_LABEL, BlockIdx(0));

    // Synthetic label counter for blocks without BlockLabel (should be rare -
    // the lowerer emits BlockLabel for all non-entry blocks).
    let mut next_label = body.label_count;

    for inst in body.insts {
        match &inst.kind {
            InstKind::BlockLabel { label, params } => {
                // Flush previous block.
                let terminator = extract_terminator(&mut current_insts);
                blocks.push(Block {
                    label: current_label,
                    params: current_params,
                    insts: std::mem::take(&mut current_insts),
                    terminator,
                });

                // Start new block.
                label_to_block.insert(*label, BlockIdx(blocks.len()));
                current_label = *label;
                current_params = params.clone();
            }
            _ => {
                current_insts.push(inst);
            }
        }
    }

    // Flush last block.
    let terminator = extract_terminator(&mut current_insts);
    blocks.push(Block {
        label: current_label,
        params: current_params,
        insts: current_insts,
        terminator,
    });

    // Invariant: every block has a unique label.
    // Entry block (bi=0) has ENTRY_LABEL. All others should have a label from
    // BlockLabel. If any non-entry block has ENTRY_LABEL, it means MirBody had
    // a block boundary without BlockLabel - assign a synthetic label.
    for bi in 1..blocks.len() {
        if blocks[bi].label == ENTRY_LABEL {
            let label = Label(next_label);
            next_label += 1;
            label_to_block.insert(label, BlockIdx(bi));
            blocks[bi].label = label;
        }
    }

    // Debug: verify uniqueness.
    debug_assert!(
        {
            let mut seen = FxHashSet::default();
            blocks.iter().all(|b| seen.insert(b.label))
        },
        "duplicate block labels after promote"
    );

    prune_unreachable(CfgBody {
        blocks,
        label_to_block,
        val_types: body.val_types,
        params: body.params,
        captures: body.captures,
        order_param: body.order_param,
        task: body.task,
        debug: body.debug,
        val_factory: body.val_factory,
        demoted_diamonds: body.demoted_diamonds,
    })
}

/// Which blocks a path from the entry reaches.
pub fn reachable(cfg: &CfgBody) -> Vec<bool> {
    let mut seen = vec![false; cfg.blocks.len()];
    let mut work = vec![BlockIdx(0)];
    while let Some(at) = work.pop() {
        if std::mem::replace(&mut seen[at.0], true) {
            continue;
        }
        work.extend(cfg.successors(at));
    }
    seen
}

/// Drop the blocks `alive` marks dead: the code the source wrote after an
/// expression typed `!` (RFC-0038), and the arms a pass decided against. A
/// pass that walks predecessors never sees a block without any.
pub fn prune(cfg: &mut CfgBody, alive: &[bool]) {
    if alive.iter().all(|alive| *alive) {
        return;
    }
    let mut bi = 0;
    cfg.blocks.retain(|_| {
        let keep = alive[bi];
        bi += 1;
        keep
    });
    cfg.label_to_block = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, block)| (block.label, BlockIdx(bi)))
        .collect();
}

fn prune_unreachable(mut cfg: CfgBody) -> CfgBody {
    let alive = reachable(&cfg);
    prune(&mut cfg, &alive);
    cfg
}

/// Extract the last control-flow instruction as a Terminator.
fn extract_terminator(insts: &mut Vec<Inst>) -> Terminator {
    if let Some(last) = insts.last() {
        match &last.kind {
            InstKind::Jump { label, args } => {
                let term = Terminator::Jump {
                    label: *label,
                    args: args.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let term = Terminator::JumpIf {
                    cond: *cond,
                    then_label: *then_label,
                    then_args: then_args.clone(),
                    else_label: *else_label,
                    else_args: else_args.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                join,
            } => {
                let term = Terminator::Diamond {
                    cond: *cond,
                    then_label: *then_label,
                    then_args: then_args.clone(),
                    else_label: *else_label,
                    else_args: else_args.clone(),
                    join: *join,
                };
                insts.pop();
                return term;
            }
            InstKind::Switch { tag, arms, default } => {
                let term = Terminator::Switch {
                    tag: *tag,
                    arms: arms.clone(),
                    default: default.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                let term = Terminator::For {
                    source: *source,
                    stages: stages.clone(),
                    exit: *exit,
                    exit_trip: *exit_trip,
                    exit_args: exit_args.clone(),
                };
                insts.pop();
                return term;
            }
            InstKind::Return { value, order } => {
                let term = Terminator::Return {
                    value: *value,
                    order: *order,
                    span: last.span,
                };
                insts.pop();
                return term;
            }
            InstKind::Diverge => {
                insts.pop();
                return Terminator::Diverge;
            }
            _ => {}
        }
    }
    Terminator::Fallthrough
}

// -- Demote: CfgBody -> MirBody ------------------------------------

pub fn demote(cfg: CfgBody) -> MirBody {
    let emitted: FxHashSet<Label> = cfg.blocks.iter().map(|block| block.label).collect();
    let demoted_diamonds = cfg
        .demoted_diamonds
        .intersection(&emitted)
        .copied()
        .collect();

    let mut insts: Vec<Inst> = Vec::new();

    for block in cfg.blocks {
        // Emit BlockLabel for non-entry blocks.
        if block.label != ENTRY_LABEL {
            insts.push(Inst {
                span: acvus_ast::Span::ZERO,
                kind: InstKind::BlockLabel {
                    label: block.label,
                    params: block.params,
                },
            });
        }

        // Emit body instructions.
        insts.extend(block.insts);

        // Emit terminator as instruction.
        match block.terminator {
            Terminator::Jump { label, args } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Jump { label, args },
                });
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::JumpIf {
                        cond,
                        then_label,
                        then_args,
                        else_label,
                        else_args,
                    },
                });
            }
            Terminator::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                join,
            } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Diamond {
                        cond,
                        then_label,
                        then_args,
                        else_label,
                        else_args,
                        join,
                    },
                });
            }
            Terminator::Switch { tag, arms, default } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Switch { tag, arms, default },
                });
            }
            Terminator::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::For {
                        source,
                        stages,
                        exit,
                        exit_trip,
                        exit_args,
                    },
                });
            }
            Terminator::Return { value, order, span } => {
                insts.push(Inst {
                    span,
                    kind: InstKind::Return { value, order },
                });
            }
            Terminator::Diverge => {
                insts.push(Inst {
                    span: acvus_ast::Span::ZERO,
                    kind: InstKind::Diverge,
                });
            }
            Terminator::Fallthrough => {
                // No instruction - implicit fallthrough.
            }
        }
    }

    // Filter out Nops.
    insts.retain(|inst| !matches!(inst.kind, InstKind::Nop));

    // Compute label_count: next available label = max(label.0) + 1.
    // Excludes ENTRY_LABEL (sentinel u32::MAX).
    let label_count = insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::BlockLabel { label, .. } => Some(label.0 + 1),
            InstKind::Jump { label, .. } => Some(label.0 + 1),
            InstKind::JumpIf {
                then_label,
                else_label,
                ..
            } => Some(then_label.0.max(else_label.0) + 1),
            InstKind::Diamond {
                then_label,
                else_label,
                join,
                ..
            } => Some(then_label.0.max(else_label.0).max(join.0) + 1),
            InstKind::Switch { arms, default, .. } => arms
                .iter()
                .map(|(_, label, _)| label.0)
                .chain(default.iter().map(|(label, _)| label.0))
                .max()
                .map(|l| l + 1),
            InstKind::For { stages, exit, .. } => stages
                .entries()
                .map(|label| label.0)
                .chain([exit.0])
                .max()
                .map(|l| l + 1),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    MirBody {
        insts,
        val_types: cfg.val_types,
        params: cfg.params,
        captures: cfg.captures,
        order_param: cfg.order_param,
        task: cfg.task,
        debug: cfg.debug,
        val_factory: cfg.val_factory,
        label_count,
        demoted_diamonds,
    }
}

// -- Tests ---------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::LocalIdOps;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_body(insts: Vec<InstKind>) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        MirBody {
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
            task: Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    #[test]
    fn roundtrip_simple() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(42),
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);

        let original_len = body.insts.len();
        let cfg = promote(body);
        assert_eq!(cfg.blocks.len(), 1);
        assert_eq!(cfg.blocks[0].insts.len(), 1); // Const
        assert!(matches!(
            cfg.blocks[0].terminator,
            Terminator::Return { .. }
        ));

        let body = demote(cfg);
        assert_eq!(body.insts.len(), original_len);
    }

    #[test]
    fn roundtrip_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![v(1)],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Const {
                dst: v(2),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![v(2)],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![v(3)],
            },
            InstKind::Return {
                value: v(3),
                order: None,
            },
        ]);

        let cfg = promote(body);
        assert_eq!(cfg.blocks.len(), 4);
        assert!(matches!(
            cfg.blocks[0].terminator,
            Terminator::JumpIf { .. }
        ));
        assert!(matches!(cfg.blocks[1].terminator, Terminator::Jump { .. }));
        assert!(matches!(cfg.blocks[2].terminator, Terminator::Jump { .. }));
        assert!(matches!(
            cfg.blocks[3].terminator,
            Terminator::Return { .. }
        ));

        let body = demote(cfg);
        assert_eq!(body.insts.len(), 10);
    }

    #[test]
    fn label_to_block_mapping() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);

        let cfg = promote(body);
        assert_eq!(cfg.label_to_block[&Label(0)], BlockIdx(1));
    }

    #[test]
    fn nop_filtered_on_demote() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Nop,
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);

        let cfg = promote(body);
        let body = demote(cfg);
        assert!(!body.insts.iter().any(|i| matches!(i.kind, InstKind::Nop)));
    }

    #[test]
    fn successors_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);

        let cfg = promote(body);
        // B0 -> B1, B2
        let b0_succs = cfg.successors(BlockIdx(0));
        assert_eq!(b0_succs.len(), 2);
        // B1 -> B3, B2 -> B3
        let b1_succs = cfg.successors(BlockIdx(1));
        assert_eq!(b1_succs.len(), 1);
        assert_eq!(b1_succs[0], BlockIdx(3));
        // B3 has no successors (Return)
        let b3_succs = cfg.successors(BlockIdx(3));
        assert!(b3_succs.is_empty());
    }

    #[test]
    fn predecessors_diamond() {
        let body = make_body(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(0),
                then_args: vec![],
                else_label: Label(1),
                else_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![],
            },
            InstKind::Jump {
                label: Label(2),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            },
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ]);

        let cfg = promote(body);
        let preds = cfg.predecessors();
        // B3 (merge) has two predecessors: B1, B2
        let b3_preds = preds.get(&BlockIdx(3)).unwrap();
        assert_eq!(b3_preds.len(), 2);
    }
}
