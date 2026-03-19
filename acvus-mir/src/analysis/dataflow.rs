use std::collections::VecDeque;

use crate::ir::{Inst, ValueId};
use rustc_hash::FxHashMap;

use crate::analysis::cfg::{BlockIdx, Cfg, Terminator};
use crate::analysis::domain::SemiLattice;

/// Trait for domains that can produce definite boolean results.
/// Used by the dataflow engine to determine branch pruning.
pub trait BooleanDomain {
    fn as_definite_bool(&self) -> Option<bool>;
}

#[derive(Debug, Clone)]
pub struct DataflowState<D: SemiLattice> {
    pub values: FxHashMap<ValueId, D>,
}

impl<D: SemiLattice> DataflowState<D> {
    pub fn new() -> Self {
        DataflowState {
            values: FxHashMap::default(),
        }
    }

    pub fn get(&self, val: ValueId) -> D {
        self.values.get(&val).cloned().unwrap_or_else(D::bottom)
    }

    pub fn set(&mut self, val: ValueId, domain: D) {
        self.values.insert(val, domain);
    }

    /// Join another state into this one. Returns true if anything changed.
    pub fn join_from(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (val, domain) in &other.values {
            let entry = self.values.entry(*val).or_insert_with(D::bottom);
            if entry.join_mut(domain) {
                changed = true;
            }
        }
        changed
    }
}

pub trait TransferFunction<D: SemiLattice> {
    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<D>);
}

pub struct DataflowResult<D: SemiLattice> {
    pub block_entry: Vec<DataflowState<D>>,
    pub block_exit: Vec<DataflowState<D>>,
    pub visited: Vec<bool>,
}

pub fn forward_analysis<D, T>(
    cfg: &Cfg,
    insts: &[Inst],
    transfer: &T,
    initial: DataflowState<D>,
) -> DataflowResult<D>
where
    D: SemiLattice + BooleanDomain,
    T: TransferFunction<D>,
{
    let n = cfg.blocks.len();
    let mut block_entry: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut block_exit: Vec<DataflowState<D>> = (0..n).map(|_| DataflowState::new()).collect();
    let mut visited = vec![false; n];

    if n == 0 {
        return DataflowResult {
            block_entry,
            block_exit,
            visited,
        };
    }

    block_entry[0] = initial;
    let mut worklist = VecDeque::new();
    worklist.push_back(BlockIdx(0));

    while let Some(idx) = worklist.pop_front() {
        visited[idx.0] = true;
        let block = &cfg.blocks[idx.0];

        let mut state = block_entry[idx.0].clone();

        for &inst_idx in &block.inst_indices {
            transfer.transfer_inst(&insts[inst_idx], &mut state);
        }

        block_exit[idx.0] = state;

        match &block.terminator {
            Terminator::Jump { target, args } => {
                if let Some(&target_idx) = cfg.label_to_block.get(target) {
                    let changed = propagate_to_successor(
                        &block_exit[idx.0],
                        &cfg.blocks[target_idx.0].params,
                        args,
                        &mut block_entry[target_idx.0],
                    );
                    if changed {
                        worklist.push_back(target_idx);
                    }
                }
            }
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            } => {
                let cond_val = block_exit[idx.0].get(*cond);
                let definite = cond_val.as_definite_bool();

                if definite != Some(false) {
                    if let Some(&target_idx) = cfg.label_to_block.get(then_label) {
                        let changed = propagate_to_successor(
                            &block_exit[idx.0],
                            &cfg.blocks[target_idx.0].params,
                            then_args,
                            &mut block_entry[target_idx.0],
                        );
                        if changed {
                            worklist.push_back(target_idx);
                        }
                    }
                }

                if definite != Some(true) {
                    if let Some(&target_idx) = cfg.label_to_block.get(else_label) {
                        let changed = propagate_to_successor(
                            &block_exit[idx.0],
                            &cfg.blocks[target_idx.0].params,
                            else_args,
                            &mut block_entry[target_idx.0],
                        );
                        if changed {
                            worklist.push_back(target_idx);
                        }
                    }
                }
            }
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < n {
                    let changed = block_entry[next].join_from(&block_exit[idx.0]);
                    if changed {
                        worklist.push_back(BlockIdx(next));
                    }
                }
            }
            Terminator::Return => {}
        }
    }

    DataflowResult {
        block_entry,
        block_exit,
        visited,
    }
}

fn propagate_to_successor<D: SemiLattice>(
    source_exit: &DataflowState<D>,
    target_params: &[ValueId],
    args: &[ValueId],
    target_entry: &mut DataflowState<D>,
) -> bool {
    let mut changed = false;

    for (param, arg) in target_params.iter().zip(args.iter()) {
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
