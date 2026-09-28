//! The CFG. Fields are private to `ir::cfg`: reading is by accessor, writing
//! is through `Builder` and `Rewriter` only.

use acvus_ast::Span;
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use super::op::Op;
use crate::ir::{DebugInfo, ExitTrip, ForSource, SwitchKey, ValueId};
use crate::ty::{Concrete, Flows, Ty, Types};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

/// A compiled module: its bodies, the entry's result type and flows, and
/// the one table every type of the module is an id of.
pub struct Module {
    bodies: Vec<Body>,
    entry: BodyId,
    ret: Ty,
    flows: Flows,
    types: Types<Concrete>,
}

/// What a value is: its type, and where it is defined.
#[derive(Debug, Clone, Copy)]
pub(super) struct ValueDef {
    pub ty: Ty,
    pub def: Def,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Def {
    Param(usize),
    Capture(usize),
    OrderParam,
    BlockParam(BlockId, usize),
    Inst(BlockId, usize),
}

pub struct Body {
    blocks: Vec<Block>,
    entry: BlockId,
    params: Vec<(Astr, ValueId)>,
    captures: Vec<(Astr, ValueId)>,
    order_param: Option<ValueId>,
    values: FxHashMap<ValueId, ValueDef>,
    debug: DebugInfo,
}

pub struct Block {
    params: Vec<ValueId>,
    insts: Vec<(Op, Span)>,
    terminator: Terminator,
}

/// A loop's stage chain (effects: loop deps): the entry block of each stage,
/// in body order; the first is the body block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stages {
    body: BlockId,
    rest: Vec<BlockId>,
}

/// Control flow. `Diamond` and `For` hold structure on purpose (D44, D45):
/// the core keeps their structural blocks in step with every edit, and a
/// `Diamond` is made only by the Builder for an `if` region.
#[derive(Debug, Clone)]
pub enum Terminator {
    Jump { to: BlockId, args: Vec<ValueId> },
    JumpIf { cond: ValueId, then_to: BlockId, then_args: Vec<ValueId>, else_to: BlockId, else_args: Vec<ValueId> },
    Diamond { cond: ValueId, then_to: BlockId, then_args: Vec<ValueId>, else_to: BlockId, else_args: Vec<ValueId>, join: BlockId },
    Switch { tag: ValueId, arms: Vec<(SwitchKey, BlockId, Vec<ValueId>)>, default: Option<(BlockId, Vec<ValueId>)> },
    For { source: ForSource, stages: Stages, exit: BlockId, exit_trip: ExitTrip, exit_args: Vec<ValueId> },
    While { cond: ValueId, stages: Stages, exit: BlockId, exit_args: Vec<ValueId> },
    Return { value: ValueId, order: Option<ValueId>, span: Span },
    Diverge,
}

impl Module {
    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[id.0 as usize]
    }
    pub fn entry(&self) -> BodyId {
        self.entry
    }
    pub fn ret(&self) -> Ty {
        self.ret
    }
    pub fn flows(&self) -> &Flows {
        &self.flows
    }
    pub fn types(&self) -> &Types<Concrete> {
        &self.types
    }
}

impl Body {
    pub fn entry(&self) -> BlockId {
        self.entry
    }
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }
    pub fn blocks(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        self.blocks.iter().enumerate().map(|(i, b)| (BlockId(i as u32), b))
    }
    /// # Panics
    /// `value` is not a value of this body: every value has a recorded type.
    pub fn ty_of(&self, value: ValueId) -> Ty {
        self.values.get(&value).expect("every IR value has a recorded type").ty
    }
    pub fn params(&self) -> &[(Astr, ValueId)] {
        &self.params
    }
    pub fn captures(&self) -> &[(Astr, ValueId)] {
        &self.captures
    }
    pub fn order_param(&self) -> Option<ValueId> {
        self.order_param
    }
    pub fn debug(&self) -> &DebugInfo {
        &self.debug
    }
}

impl Block {
    pub fn params(&self) -> &[ValueId] {
        &self.params
    }
    pub fn insts(&self) -> &[(Op, Span)] {
        &self.insts
    }
    pub fn terminator(&self) -> &Terminator {
        &self.terminator
    }
}
