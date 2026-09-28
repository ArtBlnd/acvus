//! The CFG. Fields are private to `ir::cfg`: reading is by accessor, writing
//! is through `Builder` and `Rewriter` only.

use acvus_ast::Span;
use acvus_utils::{Astr, LocalFactory};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::op::Op;
use crate::ir::{DebugInfo, ExitTrip, ForSource, OrderEdge, SwitchKey, ValueId};
use crate::ty::{Concrete, Flows, Ty, Types};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyId(pub(super) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub(super) u32);

impl BlockId {
    pub(super) fn of_index(index: usize) -> BlockId {
        BlockId(u32::try_from(index).expect("a body holds at most u32::MAX blocks"))
    }

    pub(super) fn index(self) -> usize {
        self.0 as usize
    }
}

/// A compiled module: its bodies, the entry's result type and flows, and
/// the one table every type of the module is an id of.
pub struct Module {
    pub(super) bodies: Vec<Body>,
    pub(super) entry: Option<BodyId>,
    pub(super) ret: Ty,
    pub(super) flows: Flows,
    pub(super) types: Types<Concrete>,
}

/// What a value is: its type, where it is defined, and whether it is an
/// order token, which only the core makes.
#[derive(Debug, Clone, Copy)]
pub(super) struct ValueDef {
    pub ty: Ty,
    pub def: Def,
    pub token: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Def {
    Param(usize),
    Capture(usize),
    OrderParam,
    Slot(usize),
    BlockParam(BlockId, usize),
    /// A parameter the Builder adds to a block for the order token, whose
    /// position is fixed at `seal`, after the caller's parameters.
    UnplacedParam(BlockId),
    Inst(BlockId, usize),
}

/// Where a use stands: an instruction, by block and position, or a block's
/// terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Site {
    Inst(BlockId, usize),
    Term(BlockId),
}

/// The sites that use one value, each with how many of its operands do.
#[derive(Debug, Clone, Default)]
pub(super) struct UseSites {
    counts: FxHashMap<Site, u32>,
}

impl UseSites {
    pub(super) fn add(&mut self, site: Site, count: u32) {
        *self.counts.entry(site).or_insert(0) += count;
    }

    /// # Panics
    /// `site` does not use the value.
    pub(super) fn remove_one(&mut self, site: Site) {
        let count = self.counts.get_mut(&site).expect("a use being removed was recorded");
        *count -= 1;
        if *count == 0 {
            self.counts.remove(&site);
        }
    }

    /// Moves every use recorded at `from` to `to`.
    pub(super) fn move_site(&mut self, from: Site, to: Site) {
        if let Some(count) = self.counts.remove(&from) {
            self.add(to, count);
        }
    }

    pub(super) fn total(&self) -> usize {
        self.counts.values().map(|n| *n as usize).sum()
    }

    pub(super) fn into_sites(self) -> impl Iterator<Item = (Site, u32)> {
        self.counts.into_iter()
    }
}

pub struct Body {
    pub(super) blocks: Vec<Block>,
    pub(super) entry: BlockId,
    pub(super) params: Vec<(Astr, ValueId)>,
    pub(super) captures: Vec<(Astr, ValueId)>,
    pub(super) order_param: Option<ValueId>,
    pub(super) slots: Vec<ValueId>,
    pub(super) values: FxHashMap<ValueId, ValueDef>,
    /// Every value's uses; a value has an entry from its birth.
    pub(super) uses: FxHashMap<ValueId, UseSites>,
    pub(super) value_ids: LocalFactory<ValueId>,
    pub(super) debug: DebugInfo,
}

pub struct Block {
    pub(super) params: Vec<ValueId>,
    pub(super) insts: Vec<Inst>,
    pub(super) terminator: Terminator,
}

/// An instruction: its operation, the value it yields, its place on the
/// order chain, and its span. The core makes the result and the tokens; no
/// caller names them.
#[derive(Debug, Clone)]
pub struct Inst {
    pub(super) op: Op,
    pub(super) result: Option<ValueId>,
    pub(super) order: Option<OrderUse>,
    pub(super) span: Span,
}

/// How an instruction stands on the order chain (effects §3, §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderUse {
    /// An effectful call: it consumes one token and yields the next.
    Edge(OrderEdge),
    /// A split call's start: it consumes the token.
    Start(ValueId),
    /// A split call's taking: it yields the token that follows the call.
    Take(ValueId),
}

impl OrderUse {
    pub fn consumed(self) -> Option<ValueId> {
        match self {
            OrderUse::Edge(edge) => Some(edge.before),
            OrderUse::Start(before) => Some(before),
            OrderUse::Take(_) => None,
        }
    }

    pub fn yielded(self) -> Option<ValueId> {
        match self {
            OrderUse::Edge(edge) => Some(edge.after),
            OrderUse::Take(after) => Some(after),
            OrderUse::Start(_) => None,
        }
    }
}

impl Inst {
    pub fn op(&self) -> &Op {
        &self.op
    }
    pub fn result(&self) -> Option<ValueId> {
        self.result
    }
    pub fn order(&self) -> Option<OrderUse> {
        self.order
    }
    pub fn span(&self) -> Span {
        self.span
    }

    /// Every value the instruction defines: its result and the token it
    /// yields.
    pub(super) fn defs(&self) -> SmallVec<[ValueId; 2]> {
        self.result.into_iter().chain(self.order.and_then(OrderUse::yielded)).collect()
    }

    /// Every value the instruction reads, once per operand: its operation's
    /// operands and then the token it consumes. Cost: O(its operands).
    pub fn operands(&self) -> SmallVec<[ValueId; 4]> {
        let mut operands = SmallVec::new();
        self.op.for_each_operand(|value| operands.push(value));
        operands.extend(self.order.and_then(OrderUse::consumed));
        operands
    }

    pub(super) fn for_each_operand_mut(&mut self, mut on_use: impl FnMut(&mut ValueId)) {
        self.op.for_each_operand_mut(&mut on_use);
        match &mut self.order {
            Some(OrderUse::Edge(OrderEdge { before, .. })) | Some(OrderUse::Start(before)) => on_use(before),
            Some(OrderUse::Take(_)) | None => {}
        }
    }

    /// Whether the instruction consumes, yields or joins order tokens.
    pub(super) fn is_on_order_chain(&self) -> bool {
        self.order.is_some() || matches!(self.op, Op::Merge { .. })
    }
}

/// A loop's stage chain (effects: loop deps): the entry block of each stage,
/// in body order; the first is the body block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stages {
    body: BlockId,
    rest: Vec<BlockId>,
}

impl Stages {
    /// The chain a loop is lowered with: one stage, the body.
    pub fn lowered(body: BlockId) -> Stages {
        Stages { body, rest: Vec::new() }
    }

    pub fn body(&self) -> BlockId {
        self.body
    }

    pub fn entries(&self) -> impl Iterator<Item = BlockId> + '_ {
        std::iter::once(self.body).chain(self.rest.iter().copied())
    }

    fn entries_mut(&mut self) -> impl Iterator<Item = &mut BlockId> {
        std::iter::once(&mut self.body).chain(self.rest.iter_mut())
    }
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

/// One edge out of a terminator: the target's leading `supplied`
/// parameters are filled by the terminator itself, and `args` fill the rest.
#[derive(Debug, Clone, Copy)]
pub struct Edge<'t> {
    pub to: BlockId,
    pub supplied: usize,
    pub args: &'t [ValueId],
}

impl Terminator {
    /// Every edge, in the terminator's own order; a `Diamond`'s join and a
    /// loop's later stages are not edges. Cost: O(its edges).
    pub fn edges(&self) -> SmallVec<[Edge<'_>; 2]> {
        fn plain(to: BlockId, args: &[ValueId]) -> Edge<'_> {
            Edge { to, supplied: 0, args }
        }
        match self {
            Terminator::Jump { to, args } => smallvec::smallvec![plain(*to, args)],
            Terminator::JumpIf { then_to, then_args, else_to, else_args, .. }
            | Terminator::Diamond { then_to, then_args, else_to, else_args, .. } => {
                smallvec::smallvec![plain(*then_to, then_args), plain(*else_to, else_args)]
            }
            Terminator::Switch { arms, default, .. } => arms
                .iter()
                .map(|(_, to, args)| plain(*to, args))
                .chain(default.iter().map(|(to, args)| plain(*to, args)))
                .collect(),
            Terminator::For { source, stages, exit, exit_trip, exit_args } => smallvec::smallvec![
                Edge { to: stages.body(), supplied: source.supplied_params(), args: &[] },
                Edge { to: *exit, supplied: exit_trip.supplied_params(), args: exit_args },
            ],
            Terminator::While { stages, exit, exit_args, .. } => {
                smallvec::smallvec![plain(stages.body(), &[]), plain(*exit, exit_args)]
            }
            Terminator::Return { .. } | Terminator::Diverge => SmallVec::new(),
        }
    }

    /// Every edge's target, how many leading parameters the terminator
    /// supplies, and the edge's argument list where it has one; a loop's
    /// edge into its body has none.
    pub(super) fn for_each_edge_mut(
        &mut self,
        mut on_edge: impl FnMut(BlockId, usize, Option<&mut Vec<ValueId>>),
    ) {
        match self {
            Terminator::Jump { to, args } => on_edge(*to, 0, Some(args)),
            Terminator::JumpIf { then_to, then_args, else_to, else_args, .. }
            | Terminator::Diamond { then_to, then_args, else_to, else_args, .. } => {
                on_edge(*then_to, 0, Some(then_args));
                on_edge(*else_to, 0, Some(else_args));
            }
            Terminator::Switch { arms, default, .. } => {
                for (_, to, args) in arms {
                    on_edge(*to, 0, Some(args));
                }
                if let Some((to, args)) = default {
                    on_edge(*to, 0, Some(args));
                }
            }
            Terminator::For { source, stages, exit, exit_trip, exit_args } => {
                on_edge(stages.body(), source.supplied_params(), None);
                on_edge(*exit, exit_trip.supplied_params(), Some(exit_args));
            }
            Terminator::While { stages, exit, exit_args, .. } => {
                on_edge(stages.body(), 0, None);
                on_edge(*exit, 0, Some(exit_args));
            }
            Terminator::Return { .. } | Terminator::Diverge => {}
        }
    }

    /// Every block the terminator names: its edges' targets and its
    /// structural blocks (a `Diamond`'s join, every stage entry).
    pub(super) fn for_each_block_mut(&mut self, mut on_block: impl FnMut(&mut BlockId)) {
        match self {
            Terminator::Jump { to, .. } => on_block(to),
            Terminator::JumpIf { then_to, else_to, .. } => {
                on_block(then_to);
                on_block(else_to);
            }
            Terminator::Diamond { then_to, else_to, join, .. } => {
                on_block(then_to);
                on_block(else_to);
                on_block(join);
            }
            Terminator::Switch { arms, default, .. } => {
                for (_, to, _) in arms {
                    on_block(to);
                }
                if let Some((to, _)) = default {
                    on_block(to);
                }
            }
            Terminator::For { stages, exit, .. } | Terminator::While { stages, exit, .. } => {
                stages.entries_mut().for_each(&mut on_block);
                on_block(exit);
            }
            Terminator::Return { .. } | Terminator::Diverge => {}
        }
    }

    pub(super) fn for_each_block(&self, mut on_block: impl FnMut(BlockId)) {
        match self {
            Terminator::Diamond { join, .. } => on_block(*join),
            Terminator::For { stages, .. } | Terminator::While { stages, .. } => {
                stages.entries().skip(1).for_each(&mut on_block);
            }
            Terminator::Jump { .. }
            | Terminator::JumpIf { .. }
            | Terminator::Switch { .. }
            | Terminator::Return { .. }
            | Terminator::Diverge => {}
        }
        for edge in self.edges() {
            on_block(edge.to);
        }
    }

    /// Every value the terminator reads, once per operand, in the order
    /// `for_each_operand` walks them. Cost: O(its operands).
    pub fn operands(&self) -> SmallVec<[ValueId; 4]> {
        let mut operands = SmallVec::new();
        self.for_each_operand(|value| operands.push(value));
        operands
    }

    pub(super) fn for_each_operand(&self, mut on_use: impl FnMut(ValueId)) {
        match self {
            Terminator::Jump { args, .. } => args.iter().copied().for_each(on_use),
            Terminator::JumpIf { cond, then_args, else_args, .. }
            | Terminator::Diamond { cond, then_args, else_args, .. } => {
                on_use(*cond);
                then_args.iter().chain(else_args).copied().for_each(on_use);
            }
            Terminator::Switch { tag, arms, default } => {
                on_use(*tag);
                for (_, _, args) in arms {
                    args.iter().copied().for_each(&mut on_use);
                }
                if let Some((_, args)) = default {
                    args.iter().copied().for_each(&mut on_use);
                }
            }
            Terminator::For { source, exit_args, .. } => {
                source.uses().into_iter().for_each(&mut on_use);
                exit_args.iter().copied().for_each(on_use);
            }
            Terminator::While { cond, exit_args, .. } => {
                on_use(*cond);
                exit_args.iter().copied().for_each(on_use);
            }
            Terminator::Return { value, order, .. } => {
                on_use(*value);
                order.iter().copied().for_each(on_use);
            }
            Terminator::Diverge => {}
        }
    }

    pub(super) fn for_each_operand_mut(&mut self, mut on_use: impl FnMut(&mut ValueId)) {
        match self {
            Terminator::Jump { args, .. } => args.iter_mut().for_each(on_use),
            Terminator::JumpIf { cond, then_args, else_args, .. }
            | Terminator::Diamond { cond, then_args, else_args, .. } => {
                on_use(cond);
                then_args.iter_mut().chain(else_args).for_each(on_use);
            }
            Terminator::Switch { tag, arms, default } => {
                on_use(tag);
                for (_, _, args) in arms {
                    args.iter_mut().for_each(&mut on_use);
                }
                if let Some((_, args)) = default {
                    args.iter_mut().for_each(&mut on_use);
                }
            }
            Terminator::For { source, exit_args, .. } => {
                source.for_each_use(&mut on_use);
                exit_args.iter_mut().for_each(on_use);
            }
            Terminator::While { cond, exit_args, .. } => {
                on_use(cond);
                exit_args.iter_mut().for_each(on_use);
            }
            Terminator::Return { value, order, .. } => {
                on_use(value);
                order.iter_mut().for_each(on_use);
            }
            Terminator::Diverge => {}
        }
    }
}

impl Module {
    /// A module over the type table typeck's writeback filled. It has no
    /// body, and so no entry, until `set_entry`.
    ///
    /// Cost: O(1).
    pub fn new(types: Types<Concrete>, ret: Ty, flows: Flows) -> Module {
        Module {
            bodies: Vec::new(),
            entry: None,
            ret,
            flows,
            types,
        }
    }

    /// A Builder for a new body of this module; its `seal` adds the body.
    /// An `ordered` body, one whose effect is not pure, takes an order
    /// parameter, which the Builder makes and threads (effects §3).
    ///
    /// Cost: O(1).
    pub fn build_body(&mut self, ordered: bool) -> super::Builder<'_> {
        super::Builder::new(self, ordered)
    }

    /// A Rewriter over one body of this module.
    ///
    /// Cost: O(1).
    ///
    /// # Panics
    /// `body` is not a body of this module.
    pub fn rewrite(&mut self, body: BodyId) -> super::Rewriter<'_> {
        self.assert_body(body);
        super::Rewriter::new(self, body)
    }

    /// Cost: O(1).
    ///
    /// # Panics
    /// `body` is not a body of this module.
    pub fn set_entry(&mut self, body: BodyId) {
        self.assert_body(body);
        self.entry = Some(body);
    }

    pub fn body(&self, id: BodyId) -> &Body {
        &self.bodies[id.0 as usize]
    }
    /// # Panics
    /// No entry was set.
    pub fn entry(&self) -> BodyId {
        self.entry.expect("the module's entry was set before it is read")
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

    fn assert_body(&self, body: BodyId) {
        assert!(
            (body.0 as usize) < self.bodies.len(),
            "{body:?} is not a body of this module"
        );
    }
}

impl Body {
    pub fn entry(&self) -> BlockId {
        self.entry
    }
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.index()]
    }
    pub fn blocks(&self) -> impl Iterator<Item = (BlockId, &Block)> {
        self.blocks.iter().enumerate().map(|(i, b)| (BlockId::of_index(i), b))
    }
    /// # Panics
    /// `value` is not a value of this body: every value has a recorded type.
    pub fn ty_of(&self, value: ValueId) -> Ty {
        self.values.get(&value).expect("every IR value has a recorded type").ty
    }
    /// How many operands, over every instruction and terminator, are
    /// `value`. Cost: O(sites that use it).
    ///
    /// # Panics
    /// `value` is not a value of this body.
    pub fn use_count(&self, value: ValueId) -> usize {
        self.uses.get(&value).expect("every IR value has a use record").total()
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
    /// The body's storage slots, which `RefTarget::Var` names.
    pub fn slots(&self) -> &[ValueId] {
        &self.slots
    }
    pub fn debug(&self) -> &DebugInfo {
        &self.debug
    }

    /// Whether `value` is an order token: the order parameter, a token an
    /// instruction yields, a token parameter the Builder placed, or a
    /// `Merge`'s result. Cost: O(1).
    ///
    /// # Panics
    /// `value` is not a value of this body.
    pub fn is_token(&self, value: ValueId) -> bool {
        self.values
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
            .token
    }

    /// Every value of the body, in no stated order. Cost: O(values).
    pub fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.values.keys().copied()
    }

    /// Every site that reads `value`, each once however many of its
    /// operands do. Cost: O(those sites).
    ///
    /// # Panics
    /// `value` is not a value of this body.
    pub fn users(&self, value: ValueId) -> impl Iterator<Item = Site> + '_ {
        self.uses
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is not a value of this body"))
            .counts
            .keys()
            .copied()
    }

    /// Every block's predecessors, each once; every block of the body has an
    /// entry, empty where no edge enters it. Cost: O(blocks + edges).
    pub fn predecessors(&self) -> FxHashMap<BlockId, SmallVec<[BlockId; 2]>> {
        let mut preds: FxHashMap<BlockId, SmallVec<[BlockId; 2]>> =
            self.blocks().map(|(id, _)| (id, SmallVec::new())).collect();
        for (from, block) in self.blocks() {
            for edge in block.terminator.edges() {
                let into = preds.get_mut(&edge.to).expect("an edge names a block of its body");
                if !into.contains(&from) {
                    into.push(from);
                }
            }
        }
        preds
    }

    /// The loop headers: the targets of the back edges of a depth-first walk
    /// of the edges from the entry, a back edge being one into a block still
    /// on the walk's path. A block the entry does not reach is none. Cost:
    /// O(blocks + edges).
    pub fn loop_headers(&self) -> FxHashSet<BlockId> {
        let mut headers = FxHashSet::default();
        let mut visited = vec![false; self.blocks.len()];
        let mut on_path = vec![false; self.blocks.len()];
        let mut path: Vec<(BlockId, usize)> = vec![(self.entry, 0)];
        visited[self.entry.index()] = true;
        on_path[self.entry.index()] = true;
        while let Some((block, next)) = path.last_mut() {
            let edges = self.blocks[block.index()].terminator.edges();
            let Some(edge) = edges.get(*next) else {
                on_path[block.index()] = false;
                path.pop();
                continue;
            };
            *next += 1;
            let to = edge.to;
            if on_path[to.index()] {
                headers.insert(to);
            } else if !visited[to.index()] {
                visited[to.index()] = true;
                on_path[to.index()] = true;
                path.push((to, 0));
            }
        }
        headers
    }
}

impl Block {
    pub fn params(&self) -> &[ValueId] {
        &self.params
    }
    pub fn insts(&self) -> &[Inst] {
        &self.insts
    }
    pub fn terminator(&self) -> &Terminator {
        &self.terminator
    }
}
