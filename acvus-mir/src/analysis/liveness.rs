//! Liveness analysis - backward dataflow over the CFG.
//!
//! Computes which ValueIds are live at each program point.
//! A ValueId is live at a point if there exists a path from that point
//! to a use of the ValueId without an intervening redefinition.
//!
//! Result: `LivenessResult` provides `is_live_in(block, val)` / `is_live_out(block, val)`.

use rustc_hash::FxHashSet;

use crate::analysis::dataflow::{
    DataflowAnalysis, DataflowState, backward_analysis, value_propagate_backward,
};
use crate::analysis::domain::SemiLattice;
use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, ValueId};

// -- Domain ----------------------------------------------------------

/// Liveness domain: a ValueId is either Live or Dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Liveness {
    Dead,
    Live,
}

impl SemiLattice for Liveness {
    fn bottom() -> Self {
        Liveness::Dead
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        match (*self, *other) {
            (Liveness::Live, _) => false,
            (Liveness::Dead, Liveness::Live) => {
                *self = Liveness::Live;
                true
            }
            (Liveness::Dead, Liveness::Dead) => false,
        }
    }
}

// -- Transfer function -----------------------------------------------

/// Backward liveness: kill defs, gen uses.
struct LivenessAnalysis;

impl DataflowAnalysis for LivenessAnalysis {
    type Key = ValueId;
    type Domain = Liveness;

    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<ValueId, Liveness>) {
        // Kill: defs make the value dead (before this point).
        for d in inst_info::defs(&inst.kind) {
            state.values.remove(&d);
        }
        // Gen: uses make the value live (before this point).
        for u in inst_info::uses(&inst.kind) {
            state.set(u, Liveness::Live);
        }
    }

    fn terminator_uses(&self, term: &Terminator, state: &mut DataflowState<ValueId, Liveness>) {
        match term {
            Terminator::Return(val) => state.set(*val, Liveness::Live),
            Terminator::JumpIf { cond, .. } => state.set(*cond, Liveness::Live),
            _ => {}
        }
    }

    fn propagate_forward(
        &self,
        _source_exit: &DataflowState<ValueId, Liveness>,
        _params: &[ValueId],
        _args: &[ValueId],
        _target_entry: &mut DataflowState<ValueId, Liveness>,
    ) -> bool {
        unreachable!("liveness is backward-only")
    }

    fn propagate_backward(
        &self,
        succ_entry: &DataflowState<ValueId, Liveness>,
        succ_params: &[ValueId],
        term_args: &[ValueId],
        exit_state: &mut DataflowState<ValueId, Liveness>,
    ) {
        value_propagate_backward(succ_entry, succ_params, term_args, exit_state);
    }
}

// -- Result ----------------------------------------------------------

/// Per-block liveness: live-in set (values live at block entry).
pub struct LivenessResult {
    /// live_in[block_idx] = set of ValueIds live at the start of the block.
    pub live_in: Vec<FxHashSet<ValueId>>,
    /// live_out[block_idx] = set of ValueIds live at the end of the block.
    pub live_out: Vec<FxHashSet<ValueId>>,
}

impl LivenessResult {
    /// Is `val` live at the entry of block `block`?
    pub fn is_live_in(&self, block: BlockIdx, val: ValueId) -> bool {
        self.live_in
            .get(block.0)
            .is_some_and(|set| set.contains(&val))
    }

    /// Is `val` live at the exit of block `block`?
    pub fn is_live_out(&self, block: BlockIdx, val: ValueId) -> bool {
        self.live_out
            .get(block.0)
            .is_some_and(|set| set.contains(&val))
    }
}

/// Run liveness analysis on a CfgBody.
pub fn analyze(cfg: &CfgBody) -> LivenessResult {
    if cfg.blocks.is_empty() {
        return LivenessResult {
            live_in: vec![],
            live_out: vec![],
        };
    }

    let result = backward_analysis(cfg, &LivenessAnalysis);

    let live_in = result.block_entry.iter().map(extract_live_set).collect();
    let live_out = result.block_exit.iter().map(extract_live_set).collect();

    LivenessResult { live_in, live_out }
}

/// Extract the set of live ValueIds from a DataflowState.
fn extract_live_set(state: &DataflowState<ValueId, Liveness>) -> FxHashSet<ValueId> {
    state
        .values
        .iter()
        .filter_map(|(&vid, &liveness)| match liveness {
            Liveness::Live => Some(vid),
            Liveness::Dead => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{LocalFactory, LocalIdOps};
    use rustc_hash::FxHashMap;

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn make_cfg(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..20 {
            factory.next();
        }
        promote(MirBody {
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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    // -- Single block ------------------------------------------------

    #[test]
    fn simple_linear() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(1),
            },
            InstKind::Return(v(2)),
        ]);

        let result = analyze(&cfg);
        // v(0), v(1) are defined in block 0 -> not live-in.
        assert!(!result.is_live_in(BlockIdx(0), v(0)));
        assert!(!result.is_live_in(BlockIdx(0), v(1)));
    }

    #[test]
    fn dead_value_not_live() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Int(2),
            },
            InstKind::Return(v(1)),
        ]);

        let result = analyze(&cfg);
        assert!(!result.is_live_out(BlockIdx(0), v(0)));
    }

    // -- Multi-block ------------------------------------------------

    #[test]
    fn branch_both_arms_use_value() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Const {
                dst: v(1),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::JumpIf {
                cond: v(1),
                then_label: Label(0),
                then_args: vec![v(0)],
                else_label: Label(1),
                else_args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(2)],
                merge_of: None,
            },
            InstKind::Return(v(2)),
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(3)],
                merge_of: None,
            },
            InstKind::Return(v(3)),
        ]);

        let result = analyze(&cfg);
        assert!(result.is_live_out(BlockIdx(0), v(0)));
        assert!(result.is_live_out(BlockIdx(0), v(1)));
    }

    #[test]
    fn value_live_across_blocks() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(1),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![],
                merge_of: None,
            },
            InstKind::BinOp {
                dst: v(1),
                op: acvus_ast::BinOp::Add,
                left: v(0),
                right: v(0),
            },
            InstKind::Return(v(1)),
        ]);

        let result = analyze(&cfg);
        assert!(result.is_live_out(BlockIdx(0), v(0)));
        assert!(result.is_live_in(BlockIdx(1), v(0)));
    }

    // -- Loop -------------------------------------------------------

    #[test]
    fn loop_keeps_value_live() {
        let cfg = make_cfg(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Int(0),
            },
            InstKind::Jump {
                label: Label(0),
                args: vec![v(0)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(1)],
                merge_of: None,
            },
            InstKind::BinOp {
                dst: v(2),
                op: acvus_ast::BinOp::Add,
                left: v(1),
                right: v(1),
            },
            InstKind::BinOp {
                dst: v(3),
                op: acvus_ast::BinOp::Lt,
                left: v(2),
                right: v(5),
            },
            InstKind::JumpIf {
                cond: v(3),
                then_label: Label(0),
                then_args: vec![v(2)],
                else_label: Label(1),
                else_args: vec![v(2)],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(4)],
                merge_of: None,
            },
            InstKind::Return(v(4)),
        ]);

        let result = analyze(&cfg);
        assert!(result.is_live_in(BlockIdx(1), v(1)));
        assert!(result.is_live_out(BlockIdx(1), v(2)));
    }
}
