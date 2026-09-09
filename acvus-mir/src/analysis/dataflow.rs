use std::collections::VecDeque;
use std::hash::Hash;

use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, ValueId};
use rustc_hash::FxHashMap;

use crate::analysis::domain::SemiLattice;

// ── DataflowState ──────────────────────────────────────────────────

/// Key-generic dataflow state. Maps keys to lattice domains.
#[derive(Debug, Clone, PartialEq)]
pub struct DataflowState<K: Eq + Hash + Copy, D: SemiLattice> {
    pub values: FxHashMap<K, D>,
}

impl<K: Eq + Hash + Copy, D: SemiLattice> Default for DataflowState<K, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Eq + Hash + Copy, D: SemiLattice> DataflowState<K, D> {
    pub fn new() -> Self {
        DataflowState {
            values: FxHashMap::default(),
        }
    }

    pub fn get(&self, key: K) -> D {
        self.values.get(&key).cloned().unwrap_or_else(D::bottom)
    }

    pub fn set(&mut self, key: K, domain: D) {
        self.values.insert(key, domain);
    }

    /// Join another state into this one. Returns true if anything changed.
    pub fn join_from(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (key, domain) in &other.values {
            let entry = self.values.entry(*key).or_insert_with(D::bottom);
            if entry.join_mut(domain) {
                changed = true;
            }
        }
        changed
    }
}

// ── DataflowAnalysis trait ─────────────────────────────────────────

/// Dataflow analysis definition, generic over key and domain types.
///
/// The engine handles worklist iteration and fixpoint detection.
/// Implementors define transfer functions and edge propagation.
///
/// Terminator handling is split into two methods:
///
/// - `terminator_uses`: gen what the terminator reads. In backward analysis,
///   this contributes to block_exit (values that must be live at block exit).
///   E.g., Return's value, JumpIf's condition.
///
/// In forward analysis, both are called sequentially after body instructions.
pub trait DataflowAnalysis {
    type Key: Eq + Hash + Copy;
    type Domain: SemiLattice;

    /// Transfer through a single instruction.
    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<Self::Key, Self::Domain>);

    /// Gen what the terminator uses (reads).
    ///
    /// Backward: called before block_exit snapshot. Values gen'd here appear in block_exit.
    /// Forward: called after body instructions, before block_exit.
    fn terminator_uses(
        &self,
        _term: &Terminator,
        _state: &mut DataflowState<Self::Key, Self::Domain>,
    ) {
    }


    /// (Forward only) Evaluate branch condition for dead-branch pruning.
    fn eval_branch_cond(
        &self,
        _exit_state: &DataflowState<Self::Key, Self::Domain>,
        _cond: &ValueId,
    ) -> Option<bool> {
        None
    }

    /// Propagate state across a forward edge.
    /// `params`/`args`: target block params and jump args for param→arg mapping.
    fn propagate_forward(
        &self,
        source_exit: &DataflowState<Self::Key, Self::Domain>,
        params: &[ValueId],
        args: &[ValueId],
        target_entry: &mut DataflowState<Self::Key, Self::Domain>,
    ) -> bool;

    /// Propagate state across a backward edge.
    /// `succ_params`/`term_args`: successor params and this terminator's args.
    fn propagate_backward(
        &self,
        succ_entry: &DataflowState<Self::Key, Self::Domain>,
        succ_params: &[ValueId],
        term_args: &[ValueId],
        exit_state: &mut DataflowState<Self::Key, Self::Domain>,
    );
}

// ── DataflowResult ─────────────────────────────────────────────────

pub struct DataflowResult<K: Eq + Hash + Copy, D: SemiLattice> {
    pub block_entry: Vec<DataflowState<K, D>>,
    pub block_exit: Vec<DataflowState<K, D>>,
}

// ── Forward analysis ───────────────────────────────────────────────

pub fn forward_analysis<A: DataflowAnalysis>(
    cfg: &CfgBody,
    analysis: &A,
    initial: DataflowState<A::Key, A::Domain>,
) -> DataflowResult<A::Key, A::Domain> {
    let n = cfg.blocks.len();
    let mut block_entry: Vec<_> = (0..n).map(|_| DataflowState::new()).collect();
    let mut block_exit: Vec<_> = (0..n).map(|_| DataflowState::new()).collect();

    if n == 0 {
        return DataflowResult {
            block_entry,
            block_exit,
        };
    }

    block_entry[0] = initial;
    let mut worklist = VecDeque::new();
    worklist.push_back(BlockIdx(0));

    while let Some(idx) = worklist.pop_front() {
        let block = &cfg.blocks[idx.0];
        let mut state = block_entry[idx.0].clone();

        // Transfer: instructions → terminator uses → terminator defs.
        for inst in &block.insts {
            analysis.transfer_inst(inst, &mut state);
        }
        analysis.terminator_uses(&block.terminator, &mut state);

        block_exit[idx.0] = state;

        // Propagate to successors.
        propagate_to_successors(
            cfg,
            idx,
            &block.terminator,
            &block_exit[idx.0],
            analysis,
            &mut block_entry,
            &mut worklist,
        );
    }

    DataflowResult {
        block_entry,
        block_exit,
    }
}

// ── Backward analysis ──────────────────────────────────────────────

pub fn backward_analysis<A: DataflowAnalysis>(
    cfg: &CfgBody,
    analysis: &A,
) -> DataflowResult<A::Key, A::Domain> {
    let n = cfg.blocks.len();
    let mut block_entry: Vec<_> = (0..n).map(|_| DataflowState::new()).collect();
    let mut block_exit: Vec<_> = (0..n).map(|_| DataflowState::new()).collect();

    if n == 0 {
        return DataflowResult {
            block_entry,
            block_exit,
        };
    }

    let preds = cfg.predecessors();

    let mut worklist = VecDeque::new();
    for i in (0..n).rev() {
        worklist.push_back(BlockIdx(i));
    }

    while let Some(idx) = worklist.pop_front() {
        let block = &cfg.blocks[idx.0];

        // 1. Collect what successors need.
        let mut exit_state = DataflowState::new();
        propagate_from_successors(
            cfg,
            idx,
            &block.terminator,
            analysis,
            &block_entry,
            &mut exit_state,
        );

        // 2. Gen terminator uses → included in block_exit.
        analysis.terminator_uses(&block.terminator, &mut exit_state);

        // 3. Snapshot as block_exit.
        block_exit[idx.0] = exit_state;

        // 4. Walk instructions backward from the exit state.
        let mut state = block_exit[idx.0].clone();
        for inst in block.insts.iter().rev() {
            analysis.transfer_inst(inst, &mut state);
        }

        // 5. Update block_entry; enqueue predecessors if changed.
        if state != block_entry[idx.0] {
            block_entry[idx.0] = state;
            if let Some(pred_list) = preds.get(&idx) {
                for &pred in pred_list {
                    worklist.push_back(pred);
                }
            }
        }
    }

    DataflowResult {
        block_entry,
        block_exit,
    }
}

// ── Edge propagation helpers ───────────────────────────────────────

/// Forward: propagate block_exit to each successor via the terminator's edges.
fn propagate_to_successors<A: DataflowAnalysis>(
    cfg: &CfgBody,
    idx: BlockIdx,
    term: &Terminator,
    exit_state: &DataflowState<A::Key, A::Domain>,
    analysis: &A,
    block_entry: &mut [DataflowState<A::Key, A::Domain>],
    worklist: &mut VecDeque<BlockIdx>,
) {
    let n = block_entry.len();

    match term {
        Terminator::Jump { label, args } => {
            if let Some(&t) = cfg.label_to_block.get(label)
                && analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    args,
                    &mut block_entry[t.0],
                )
            {
                worklist.push_back(t);
            }
        }
        Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } => {
            let definite = analysis.eval_branch_cond(exit_state, cond);
            if definite != Some(false)
                && let Some(&t) = cfg.label_to_block.get(then_label)
                && analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    then_args,
                    &mut block_entry[t.0],
                )
            {
                worklist.push_back(t);
            }
            if definite != Some(true)
                && let Some(&t) = cfg.label_to_block.get(else_label)
                && analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    else_args,
                    &mut block_entry[t.0],
                )
            {
                worklist.push_back(t);
            }
        }
        Terminator::Fallthrough => {
            let next = idx.0 + 1;
            if next < n && block_entry[next].join_from(exit_state) {
                worklist.push_back(BlockIdx(next));
            }
        }
        Terminator::Return(_) => {}
    }
}

/// Backward: join successor entries into exit_state via the terminator's edges.
fn propagate_from_successors<A: DataflowAnalysis>(
    cfg: &CfgBody,
    idx: BlockIdx,
    term: &Terminator,
    analysis: &A,
    block_entry: &[DataflowState<A::Key, A::Domain>],
    exit_state: &mut DataflowState<A::Key, A::Domain>,
) {
    let n = block_entry.len();

    match term {
        Terminator::Jump { label, args } => {
            if let Some(&t) = cfg.label_to_block.get(label) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    args,
                    exit_state,
                );
            }
        }
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => {
            if let Some(&t) = cfg.label_to_block.get(then_label) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    then_args,
                    exit_state,
                );
            }
            if let Some(&t) = cfg.label_to_block.get(else_label) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    else_args,
                    exit_state,
                );
            }
        }
        Terminator::Fallthrough => {
            let next = idx.0 + 1;
            if next < n {
                exit_state.join_from(&block_entry[next]);
            }
        }
        Terminator::Return(_) => {}
    }
}

// ── ValueId propagation helpers ────────────────────────────────────

/// Standard forward propagation for ValueId-keyed analyses:
/// map args → params, then join flow-through values.
pub fn value_propagate_forward<D: SemiLattice>(
    source_exit: &DataflowState<ValueId, D>,
    params: &[ValueId],
    args: &[ValueId],
    target_entry: &mut DataflowState<ValueId, D>,
) -> bool {
    let mut changed = false;

    for (param, arg) in params.iter().zip(args.iter()) {
        let arg_val = source_exit.get(*arg);
        let entry = target_entry.values.entry(*param).or_insert_with(D::bottom);
        if entry.join_mut(&arg_val) {
            changed = true;
        }
    }

    if target_entry.join_from(source_exit) {
        changed = true;
    }

    changed
}

/// Standard backward propagation for ValueId-keyed analyses:
/// map live params → args, then join flow-through values.
pub fn value_propagate_backward<D: SemiLattice>(
    succ_entry: &DataflowState<ValueId, D>,
    succ_params: &[ValueId],
    term_args: &[ValueId],
    exit_state: &mut DataflowState<ValueId, D>,
) {
    for (param, arg) in succ_params.iter().zip(term_args.iter()) {
        let param_val = succ_entry.get(*param);
        if param_val != D::bottom() {
            let entry = exit_state.values.entry(*arg).or_insert_with(D::bottom);
            entry.join_mut(&param_val);
        }
    }

    let param_set: rustc_hash::FxHashSet<ValueId> = succ_params.iter().copied().collect();
    for (val, domain) in &succ_entry.values {
        if !param_set.contains(val) {
            let entry = exit_state.values.entry(*val).or_insert_with(D::bottom);
            entry.join_mut(domain);
        }
    }
}
