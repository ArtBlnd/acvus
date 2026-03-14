use acvus_mir::ir::{Inst, InstKind, Label, ValueId};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockIdx(pub usize);

#[derive(Debug, Clone)]
pub enum Terminator {
    Jump {
        target: Label,
        args: Vec<ValueId>,
    },
    JumpIf {
        cond: ValueId,
        then_label: Label,
        then_args: Vec<ValueId>,
        else_label: Label,
        else_args: Vec<ValueId>,
    },
    Return,
    Fallthrough,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: Option<Label>,
    pub params: Vec<ValueId>,
    pub inst_indices: Vec<usize>,
    pub terminator: Terminator,
    pub merge_of: Option<Label>,
}

#[derive(Debug, Clone)]
pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub label_to_block: FxHashMap<Label, BlockIdx>,
}

impl Cfg {
    pub fn build(insts: &[Inst]) -> Self {
        let mut blocks = Vec::new();
        let mut label: Option<Label> = None;
        let mut params: Vec<ValueId> = Vec::new();
        let mut inst_indices = Vec::new();
        let mut current_merge_of: Option<Label> = None;

        for (i, inst) in insts.iter().enumerate() {
            match &inst.kind {
                InstKind::BlockLabel {
                    label: l,
                    params: p,
                    merge_of,
                } => {
                    if !inst_indices.is_empty() || label.is_some() {
                        blocks.push(BasicBlock {
                            label: label.take(),
                            params: std::mem::take(&mut params),
                            inst_indices: std::mem::take(&mut inst_indices),
                            terminator: Terminator::Fallthrough,
                            merge_of: current_merge_of.take(),
                        });
                    }
                    label = Some(*l);
                    params = p.clone();
                    current_merge_of = *merge_of;
                }
                InstKind::Jump {
                    label: target,
                    args,
                } => {
                    blocks.push(BasicBlock {
                        label: label.take(),
                        params: std::mem::take(&mut params),
                        inst_indices: std::mem::take(&mut inst_indices),
                        terminator: Terminator::Jump {
                            target: *target,
                            args: args.clone(),
                        },
                        merge_of: current_merge_of.take(),
                    });
                }
                InstKind::JumpIf {
                    cond,
                    then_label,
                    then_args,
                    else_label,
                    else_args,
                } => {
                    blocks.push(BasicBlock {
                        label: label.take(),
                        params: std::mem::take(&mut params),
                        inst_indices: std::mem::take(&mut inst_indices),
                        terminator: Terminator::JumpIf {
                            cond: *cond,
                            then_label: *then_label,
                            then_args: then_args.clone(),
                            else_label: *else_label,
                            else_args: else_args.clone(),
                        },
                        merge_of: current_merge_of.take(),
                    });
                }
                InstKind::Return(_) => {
                    blocks.push(BasicBlock {
                        label: label.take(),
                        params: std::mem::take(&mut params),
                        inst_indices: std::mem::take(&mut inst_indices),
                        terminator: Terminator::Return,
                        merge_of: current_merge_of.take(),
                    });
                }
                _ => {
                    inst_indices.push(i);
                }
            }
        }

        if !inst_indices.is_empty() || label.is_some() {
            blocks.push(BasicBlock {
                label,
                params,
                inst_indices,
                terminator: Terminator::Fallthrough,
                merge_of: current_merge_of,
            });
        }

        let label_to_block = blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.label.map(|l| (l, BlockIdx(i))))
            .collect();

        Cfg {
            blocks,
            label_to_block,
        }
    }

    pub fn successors(&self, idx: BlockIdx) -> SmallVec<[BlockIdx; 2]> {
        let block = &self.blocks[idx.0];
        let mut succs = SmallVec::new();
        match &block.terminator {
            Terminator::Jump { target, .. } => {
                if let Some(&bi) = self.label_to_block.get(target) {
                    succs.push(bi);
                }
            }
            Terminator::JumpIf {
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
            Terminator::Fallthrough => {
                let next = idx.0 + 1;
                if next < self.blocks.len() {
                    succs.push(BlockIdx(next));
                }
            }
            Terminator::Return => {}
        }
        succs
    }
}
