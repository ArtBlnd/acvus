use std::collections::VecDeque;
use std::hash::Hash;

use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, Label, ValueId};
use rustc_hash::FxHashMap;

use crate::analysis::domain::SemiLattice;

// -- DataflowState --------------------------------------------------

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

// -- DataflowAnalysis trait -----------------------------------------

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

    /// Propagate state across a forward edge. `params` is the target
    /// block's whole parameter list and `args` fills it from `first`: a
    /// `For`'s body edge starts after the parameters the terminator fills
    /// itself (RFC-0057), and every other edge starts at zero.
    fn propagate_forward(
        &self,
        source_exit: &DataflowState<Self::Key, Self::Domain>,
        params: &[ValueId],
        first: usize,
        args: &[ValueId],
        target_entry: &mut DataflowState<Self::Key, Self::Domain>,
    ) -> bool;

    /// Propagate state across a backward edge, with `succ_params` and
    /// `first` as they are on the forward one.
    fn propagate_backward(
        &self,
        succ_entry: &DataflowState<Self::Key, Self::Domain>,
        succ_params: &[ValueId],
        first: usize,
        term_args: &[ValueId],
        exit_state: &mut DataflowState<Self::Key, Self::Domain>,
    );
}

// -- DataflowResult -------------------------------------------------

pub struct DataflowResult<K: Eq + Hash + Copy, D: SemiLattice> {
    pub block_entry: Vec<DataflowState<K, D>>,
    pub block_exit: Vec<DataflowState<K, D>>,
}

// -- Forward analysis -----------------------------------------------

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
    let mut worklist = Worklist::new(n);
    worklist.push(BlockIdx(0));
    let mut visited = vec![false; n];

    while let Some(idx) = worklist.pop() {
        visited[idx.0] = true;
        let block = &cfg.blocks[idx.0];
        let mut state = block_entry[idx.0].clone();

        // Transfer: instructions -> terminator uses -> terminator defs.
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
            &visited,
            &mut worklist,
        );
    }

    DataflowResult {
        block_entry,
        block_exit,
    }
}

/// The forward worklist: a block is queued at most once at a time. A block
/// pushed while already queued is visited once, on the state every push
/// joined into its entry, so the number of visits is bounded by the
/// changes to its entry rather than by the paths that reach it.
struct Worklist {
    queue: VecDeque<BlockIdx>,
    queued: Vec<bool>,
}

impl Worklist {
    fn new(n: usize) -> Self {
        Worklist {
            queue: VecDeque::new(),
            queued: vec![false; n],
        }
    }

    fn push(&mut self, idx: BlockIdx) {
        if !self.queued[idx.0] {
            self.queued[idx.0] = true;
            self.queue.push_back(idx);
        }
    }

    fn pop(&mut self) -> Option<BlockIdx> {
        let idx = self.queue.pop_front()?;
        self.queued[idx.0] = false;
        Some(idx)
    }
}

// -- Backward analysis ----------------------------------------------

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

        // 2. Gen terminator uses -> included in block_exit.
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

// -- Edge propagation helpers ---------------------------------------

/// One branch of a `JumpIf`, with whether the condition can take it.
struct Edge<'a> {
    taken: bool,
    label: crate::ir::Label,
    args: &'a [ValueId],
}

/// Forward: propagate block_exit to each successor via the terminator's edges.
fn propagate_to_successors<A: DataflowAnalysis>(
    cfg: &CfgBody,
    idx: BlockIdx,
    term: &Terminator,
    exit_state: &DataflowState<A::Key, A::Domain>,
    analysis: &A,
    block_entry: &mut [DataflowState<A::Key, A::Domain>],
    visited: &[bool],
    worklist: &mut Worklist,
) {
    let n = block_entry.len();

    match term {
        Terminator::Jump { label, args } => {
            if let Some(&t) = cfg.label_to_block.get(label) {
                let changed = analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    0,
                    args,
                    &mut block_entry[t.0],
                );
                if changed || !visited[t.0] {
                    worklist.push(t);
                }
            }
        }
        Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        }
        | Terminator::Diamond {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => {
            let definite = analysis.eval_branch_cond(exit_state, cond);
            let edges = [
                Edge {
                    taken: definite != Some(false),
                    label: *then_label,
                    args: then_args,
                },
                Edge {
                    taken: definite != Some(true),
                    label: *else_label,
                    args: else_args,
                },
            ];
            for Edge { taken, label, args } in edges {
                if taken && let Some(&t) = cfg.label_to_block.get(&label) {
                    let changed = analysis.propagate_forward(
                        exit_state,
                        &cfg.blocks[t.0].params,
                        0,
                        args,
                        &mut block_entry[t.0],
                    );
                    if changed || !visited[t.0] {
                        worklist.push(t);
                    }
                }
            }
        }
        Terminator::While {
            cond,
            stages,
            exit,
            exit_args,
        } => {
            let definite = analysis.eval_branch_cond(exit_state, cond);
            let edges = [
                Edge {
                    taken: definite != Some(false),
                    label: stages.body(),
                    args: &[],
                },
                Edge {
                    taken: definite != Some(true),
                    label: *exit,
                    args: exit_args,
                },
            ];
            for Edge { taken, label, args } in edges {
                if taken && let Some(&t) = cfg.label_to_block.get(&label) {
                    let changed = analysis.propagate_forward(
                        exit_state,
                        &cfg.blocks[t.0].params,
                        0,
                        args,
                        &mut block_entry[t.0],
                    );
                    if changed || !visited[t.0] {
                        worklist.push(t);
                    }
                }
            }
        }
        // Both edges of a `For` are taken on some path: the source decides
        // which, and no analysis here reads a source. The body's parameters
        // are the terminator's own (RFC-0089 rule 1), and so is the exit's
        // first where the edge defines the trip count (RFC-0057 rule 9).
        Terminator::For {
            source,
            stages,
            exit,
            exit_trip,
            exit_args,
        } => {
            if let Some(&t) = cfg.label_to_block.get(&stages.body()) {
                let changed = analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    source.supplied_params(),
                    &[],
                    &mut block_entry[t.0],
                );
                if changed || !visited[t.0] {
                    worklist.push(t);
                }
            }
            if let Some(&t) = cfg.label_to_block.get(&exit) {
                let changed = analysis.propagate_forward(
                    exit_state,
                    &cfg.blocks[t.0].params,
                    exit_trip.supplied_params(),
                    exit_args,
                    &mut block_entry[t.0],
                );
                if changed || !visited[t.0] {
                    worklist.push(t);
                }
            }
        }
        // Every arm of a `Switch` is taken on some path: the tag decides
        // which, and no analysis here reads a tag.
        Terminator::Switch { arms, default, .. } => {
            let edges = arms
                .iter()
                .map(|(_, label, args)| (label, args))
                .chain(default.iter().map(|(label, args)| (label, args)));
            for (label, args) in edges {
                if let Some(&t) = cfg.label_to_block.get(label) {
                    let changed = analysis.propagate_forward(
                        exit_state,
                        &cfg.blocks[t.0].params,
                        0,
                        args,
                        &mut block_entry[t.0],
                    );
                    if changed || !visited[t.0] {
                        worklist.push(t);
                    }
                }
            }
        }
        Terminator::Fallthrough => {
            let next = idx.0 + 1;
            if next < n && (block_entry[next].join_from(exit_state) || !visited[next]) {
                worklist.push(BlockIdx(next));
            }
        }
        Terminator::Return { .. } | Terminator::Diverge => {}
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
                    0,
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
        }
        | Terminator::Diamond {
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
                    0,
                    then_args,
                    exit_state,
                );
            }
            if let Some(&t) = cfg.label_to_block.get(else_label) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    0,
                    else_args,
                    exit_state,
                );
            }
        }
        Terminator::While {
            stages,
            exit,
            exit_args,
            ..
        } => {
            let edges: [(Label, &[ValueId]); 2] = [(stages.body(), &[]), (*exit, exit_args)];
            for (label, args) in edges {
                if let Some(&t) = cfg.label_to_block.get(&label) {
                    analysis.propagate_backward(
                        &block_entry[t.0],
                        &cfg.blocks[t.0].params,
                        0,
                        args,
                        exit_state,
                    );
                }
            }
        }
        Terminator::Switch { arms, default, .. } => {
            let edges = arms
                .iter()
                .map(|(_, label, args)| (label, args))
                .chain(default.iter().map(|(label, args)| (label, args)));
            for (label, args) in edges {
                if let Some(&t) = cfg.label_to_block.get(label) {
                    analysis.propagate_backward(
                        &block_entry[t.0],
                        &cfg.blocks[t.0].params,
                        0,
                        args,
                        exit_state,
                    );
                }
            }
        }
        Terminator::For {
            source,
            stages,
            exit,
            exit_trip,
            exit_args,
        } => {
            if let Some(&t) = cfg.label_to_block.get(&stages.body()) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    source.supplied_params(),
                    &[],
                    exit_state,
                );
            }
            if let Some(&t) = cfg.label_to_block.get(&exit) {
                analysis.propagate_backward(
                    &block_entry[t.0],
                    &cfg.blocks[t.0].params,
                    exit_trip.supplied_params(),
                    exit_args,
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
        Terminator::Return { .. } | Terminator::Diverge => {}
    }
}

// -- ValueId propagation helpers ------------------------------------

/// Standard backward propagation for ValueId-keyed analyses:
/// map live params -> args, then join flow-through values.
pub fn value_propagate_backward<D: SemiLattice>(
    succ_entry: &DataflowState<ValueId, D>,
    succ_params: &[ValueId],
    first: usize,
    term_args: &[ValueId],
    exit_state: &mut DataflowState<ValueId, D>,
) {
    for (param, arg) in succ_params.iter().skip(first).zip(term_args.iter()) {
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
