//! What a `For` costs (RFC-0066 rule 8): one scalar per loop, read from a
//! backend's static table, and the one compare on the trip count that
//! chooses between running in place and splitting.
//!
//! This reads `analysis::loop_deps` for which stages are free and which
//! cycles are `Disjoint`, and computes no stage fact itself. No MIR pass reads a cost, and none is measured at
//! run time; the reader it is for is a lowerer that splits (RFC-0066 rule
//! 10), which no backend has yet.
//!
//! A work that leaves `u64` stays at `u64::MAX`. That can only lower `W`,
//! and a lower `W` only raises the threshold, which keeps the loop in
//! place: the direction rule 8 asks every wrong estimate to err in.

use acvus_ast::Literal;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loop_deps::{Control, Head, LoopDeps, Order, Placement, StageBlocks};
use crate::analysis::loops::{Invariants, LoopId, LoopKind, LoopNest, Term, Trip};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{BinOp, Callee, ForSource, InstKind, ValueId};
use crate::laws::LawTable;
use crate::ty::{Task, Ty};

/// A backend's weights in ticks, one row per operation family RFC-0066
/// rule 8 lists, and the `k` its split compares against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostTable {
    pub arithmetic: u64,
    pub compare: u64,
    pub load: u64,
    pub store: u64,
    pub allocation: u64,
    pub local_call: u64,
    pub extern_call: u64,
    /// An extern call whose settled task is `Heavy` (RFC-0046).
    pub heavy: u64,
    pub spawn: u64,
    pub merge: u64,
    /// One synchronous executor call that runs a chunk, which is how a
    /// `Sync` body splits (RFC-0092).
    pub chunk_dispatch: u64,
    /// One element a chunk hands from a stage to the next.
    pub buffered_element: u64,
    pub k: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InPlace {
    /// No stage is free or holds only `Disjoint` cycles.
    NoStageRunsApart,
    NoWork,
    /// A pull loop: no count is known on entry (RFC-0089 rule 1).
    CountUnknown,
}

/// What the trip count `n` a split compares is (RFC-0089 rule 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TripCount {
    /// Every exit is the header's: the loop runs `n` iterations.
    Exact,
    /// The loop can leave early, from its body: it runs at most `n`
    /// iterations, and work a split spends past the exit is discarded.
    Bound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCost {
    /// The loop splits when its trip count `n` satisfies `n > threshold`.
    Split {
        work: u64,
        overhead: u128,
        threshold: u128,
        trips: TripCount,
    },
    InPlace(InPlace),
}

#[derive(Clone, Copy)]
struct LoopExit {
    from: BlockIdx,
    to: BlockIdx,
}

/// The weight after a block no edge leaves: the iteration's run ends there.
const PATH_ENDS: u64 = 0;

/// A threshold no trip count exceeds. A trip count is a `u64`, and where
/// `K · O` leaves `u128` the true threshold is past `u64` as well, since `W`
/// is a `u64`.
const BEYOND_EVERY_TRIP_COUNT: u128 = u128::MAX;

/// Rule 8's least trip count for an inner loop where that is unknown, here
/// no constant `u64`, or where the loop can leave early, from its body.
const UNKNOWN: u64 = 0;

pub struct Costs<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    table: &'a CostTable,
    nest: LoopNest,
    /// The atoms a constant trip count reads. A value some other
    /// definition also writes, as a reused register is, is none of them.
    integers: FxHashMap<ValueId, i128>,
}

impl<'a> Costs<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &'a LawTable, table: &'a CostTable) -> Self {
        let nest = LoopNest::of(cfg, &DomTree::build(cfg), &Invariants::of(cfg));
        let mut definitions: FxHashMap<ValueId, usize> = FxHashMap::default();
        let defined = cfg.blocks.iter().flat_map(|block| {
            block
                .params
                .iter()
                .copied()
                .chain(block.insts.iter().flat_map(|inst| inst_info::defs(&inst.kind)))
        });
        for value in defined {
            *definitions.entry(value).or_default() += 1;
        }
        let integers = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.kind {
                InstKind::Const { dst, value } if definitions.get(dst) == Some(&1) => {
                    integer(value).map(|integer| (*dst, integer))
                }
                _ => None,
            })
            .collect();
        Self {
            cfg,
            laws,
            table,
            nest,
            integers,
        }
    }

    pub fn trip(&self, header: BlockIdx) -> Option<&Trip> {
        self.nest
            .by_header(header)
            .map(|id| &self.nest.get(id).trip)
    }

    /// `W` sums the stages that run apart: the free ones, and those whose
    /// every cycle is `Disjoint`, whose token is absent (RFC-0092).
    /// A stage holding an `AnyOrder` or `InOrder` cycle, or a cycle that
    /// crosses into another stage, runs in its order and is not counted.
    /// With no stage that runs apart the loop runs in place. A loop whose
    /// control is chained can leave early, so its trip count is a bound.
    pub fn of_loop(&self, deps: &LoopDeps) -> LoopCost {
        if let Head::Pull = deps.membership.head() {
            return LoopCost::InPlace(InPlace::CountUnknown);
        }
        let stages = deps.membership.stages();
        let judged = deps.judge(self.cfg, self.laws);
        let runs_apart = |stage: usize| {
            deps.cycles
                .iter()
                .zip(&judged)
                .all(|(cycle, judged)| match &cycle.placement {
                    Placement::Stage(at) => *at != stage || judged.order == Order::Disjoint,
                    Placement::Crosses(across) => !across.contains(&stage),
                })
        };
        if !(0..stages.len()).any(runs_apart) {
            return LoopCost::InPlace(InPlace::NoStageRunsApart);
        }
        let work = stages
            .iter()
            .enumerate()
            .filter(|(stage, _)| runs_apart(*stage))
            .map(|(_, blocks)| self.stage_work(blocks))
            .fold(0, u64::saturating_add);
        if work == 0 {
            return LoopCost::InPlace(InPlace::NoWork);
        }
        let overhead = self.overhead(stages);
        let threshold = match u128::from(self.table.k).checked_mul(overhead) {
            Some(bound) => bound.div_ceil(u128::from(work)),
            None => BEYOND_EVERY_TRIP_COUNT,
        };
        let trips = match deps.control {
            Control::Upfront => TripCount::Exact,
            Control::Chained { .. } => TripCount::Bound,
        };
        LoopCost::Split {
            work,
            overhead,
            threshold,
            trips,
        }
    }

    /// Rule 8's `O`: one chunk's dispatch, which is a spawn where the body
    /// suspends and otherwise one synchronous executor call (RFC-0066 rule
    /// 10); one merge; and one buffered element per boundary between two
    /// stages.
    fn overhead(&self, stages: &[StageBlocks]) -> u128 {
        let dispatch = match self.cfg.task {
            Task::Sync => self.table.chunk_dispatch,
            Task::Async | Task::Heavy => self.table.spawn,
        };
        stages
            .windows(2)
            .map(|_| self.table.buffered_element)
            .chain([dispatch, self.table.merge])
            .map(u128::from)
            .sum()
    }

    fn stage_work(&self, stage: &StageBlocks) -> u64 {
        let region: FxHashSet<BlockIdx> = stage.blocks.iter().copied().collect();
        self.least_weight_from(stage.entry_block, &region, &mut FxHashMap::default())
    }

    /// The least weight from `block` to an edge out of `region`. `region`
    /// holds no header of the loop it lies in, so every edge back to that
    /// header leaves it; a loop inside it is entered only at its header, the
    /// body being reducible, and is weighed there whole.
    fn least_weight_from(
        &self,
        block: BlockIdx,
        region: &FxHashSet<BlockIdx>,
        memo: &mut FxHashMap<BlockIdx, u64>,
    ) -> u64 {
        if !region.contains(&block) {
            return PATH_ENDS;
        }
        if let Some(found) = memo.get(&block) {
            return *found;
        }
        let found = match self.nest.by_header(block) {
            Some(inner) => self.inner_loop_weight(inner, region, memo),
            None => {
                let rest = self
                    .cfg
                    .successors(block)
                    .into_iter()
                    .map(|succ| self.least_weight_from(succ, region, memo))
                    .min()
                    .unwrap_or(PATH_ENDS);
                self.block_weight(block).saturating_add(rest)
            }
        };
        memo.insert(block, found);
        found
    }

    /// A loop inside `region`, from its header to the first block after
    /// it: the header on each of `t + 1` visits, `t` iterations at their
    /// least, and the lightest way on from where it leaves. `t` is the
    /// least trip count, which rule 8 counts as zero for a loop that can
    /// leave early, from its body.
    fn inner_loop_weight(
        &self,
        inner: LoopId,
        region: &FxHashSet<BlockIdx>,
        memo: &mut FxHashMap<BlockIdx, u64>,
    ) -> u64 {
        let found = self.nest.get(inner);
        let header = found.natural.header;
        let exits: Vec<LoopExit> = found
            .natural
            .blocks()
            .flat_map(|from| {
                self.cfg
                    .successors(from)
                    .into_iter()
                    .filter(|to| !found.natural.contains(*to))
                    .map(move |to| LoopExit { from, to })
            })
            .collect();
        let trips = match exits.iter().all(|exit| exit.from == header) {
            true => self.least_trip(inner),
            false => UNKNOWN,
        };
        let iteration = match trips {
            0 => 0,
            _ => {
                let body: FxHashSet<BlockIdx> = found
                    .natural
                    .blocks()
                    .filter(|block| *block != header)
                    .collect();
                let mut within = FxHashMap::default();
                self.cfg
                    .successors(header)
                    .into_iter()
                    .filter(|succ| body.contains(succ))
                    .map(|succ| self.least_weight_from(succ, &body, &mut within))
                    .min()
                    .unwrap_or(PATH_ENDS)
            }
        };
        let after = exits
            .iter()
            .map(|exit| self.least_weight_from(exit.to, region, memo))
            .min()
            .unwrap_or(PATH_ENDS);
        self.block_weight(header)
            .saturating_mul(trips.saturating_add(1))
            .saturating_add(iteration.saturating_mul(trips))
            .saturating_add(after)
    }

    fn least_trip(&self, inner: LoopId) -> u64 {
        let found = self.nest.get(inner);
        let (LoopKind::For { .. }, Trip::Known(term)) = (found.kind, &found.trip) else {
            return UNKNOWN;
        };
        match self.constant(term).map(u64::try_from) {
            Some(Ok(count)) => count,
            Some(Err(_)) | None => UNKNOWN,
        }
    }

    fn constant(&self, term: &Term) -> Option<i128> {
        match term {
            Term::Const(literal) => integer(literal),
            Term::Value(value) => self.integers.get(value).copied(),
            Term::Len(_) | Term::LenOnEntry(_) => None,
            Term::Add(a, b) => self.constant(a)?.checked_add(self.constant(b)?),
            Term::Sub(a, b) => self.constant(a)?.checked_sub(self.constant(b)?),
            Term::Mul(a, b) => self.constant(a)?.checked_mul(self.constant(b)?),
            Term::Max(a, b) => Some(self.constant(a)?.max(self.constant(b)?)),
        }
    }

    fn block_weight(&self, block: BlockIdx) -> u64 {
        let at = &self.cfg.blocks[block.0];
        at.insts
            .iter()
            .map(|inst| self.inst_weight(&inst.kind))
            .fold(self.terminator_weight(&at.terminator), u64::saturating_add)
    }

    fn terminator_weight(&self, terminator: &Terminator) -> u64 {
        match terminator {
            Terminator::For { source, .. } => self.step_weight(*source),
            Terminator::Switch { .. } => self.table.compare,
            // A two-way branch or a jump has no row, and rule 8 weighs an
            // operation with no row as nothing; the compare that produced a
            // branch's `Bool` was weighed at its row.
            Terminator::JumpIf { .. } | Terminator::Diamond { .. } | Terminator::While { .. } => 0,
            Terminator::Jump { .. }
            | Terminator::Return { .. }
            | Terminator::Diverge
            | Terminator::Fallthrough => 0,
        }
    }

    /// A backend runs a `for`'s advance and its compare with the bound as
    /// one operation, as the interpreter's `For` op does, so the step weighs
    /// one compare, and a slice or an array source's step also reads the
    /// element.
    fn step_weight(&self, source: ForSource) -> u64 {
        match source {
            ForSource::Range { .. } => self.table.compare,
            ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => {
                self.table.compare.saturating_add(self.table.load)
            }
        }
    }

    fn inst_weight(&self, kind: &InstKind) -> u64 {
        let table = self.table;
        match kind {
            InstKind::BinOp { op, .. } => match op {
                BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                    table.compare
                }
                BinOp::Add(_)
                | BinOp::Sub(_)
                | BinOp::Mul(_)
                | BinOp::Div
                | BinOp::And
                | BinOp::Or
                | BinOp::Xor
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::Shl(_)
                | BinOp::Shr(_)
                | BinOp::Mod
                | BinOp::Min
                | BinOp::Max => table.arithmetic,
            },
            InstKind::UnaryOp { .. }
            | InstKind::Cast { .. }
            | InstKind::Check { .. }
            | InstKind::CheckSteps { .. } => table.arithmetic,
            InstKind::StringEq { .. }
            | InstKind::StructuralEq { .. }
            | InstKind::TestLiteral { .. }
            | InstKind::TestObjectKey { .. }
            | InstKind::TestVariant { .. } => table.compare,
            InstKind::Ref { .. }
            | InstKind::Take { .. }
            | InstKind::AsSlice { .. }
            | InstKind::Index { .. }
            | InstKind::Fetch { .. }
            | InstKind::FieldGet { .. }
            | InstKind::LoadFunction { .. }
            | InstKind::TupleIndex { .. }
            | InstKind::ArrayIndex { .. }
            | InstKind::ObjectGet { .. }
            | InstKind::UnwrapVariant { .. } => table.load,
            InstKind::Assign { .. }
            | InstKind::IndexSet { .. }
            | InstKind::Commit { .. }
            | InstKind::StringAppend { .. }
            | InstKind::ArrayPush { .. }
            | InstKind::FieldSet { .. } => table.store,
            InstKind::StringConcat { .. }
            | InstKind::StringClone { .. }
            | InstKind::StructuralClone { .. }
            | InstKind::ArrayBegin { .. }
            | InstKind::MakeObject { .. }
            | InstKind::MakeTuple { .. }
            | InstKind::MakeClosure { .. }
            | InstKind::MakeVariant { .. } => table.allocation,
            // A word, a `ConstStr` (the prepared code owns the text, the
            // constant being its pointer and length, RFC-0062 rule 2), an
            // `Eval` (its call is weighed at its `Spawn`), a `Drop` and a
            // marker have no row, and rule 8 weighs an operation with no row
            // as nothing.
            InstKind::Const { value, .. } => match value {
                Literal::String(_) | Literal::List(_) => table.allocation,
                Literal::Int(_)
                | Literal::IntOf(_)
                | Literal::Float(_)
                | Literal::Char(_)
                | Literal::Bytes(_)
                | Literal::Bool(_)
                | Literal::Unit => 0,
            },
            InstKind::ConstStr { .. } => 0,
            InstKind::FunctionCall {
                callee, callee_ty, ..
            } => self.call_weight(callee, callee_ty),
            InstKind::Spawn {
                callee, callee_ty, ..
            } => table
                .spawn
                .saturating_add(self.call_weight(callee, callee_ty)),
            InstKind::Eval { .. } => 0,
            InstKind::Merge { .. } => table.merge,
            InstKind::Drop { .. } => 0,
            InstKind::BlockLabel { .. }
            | InstKind::Undef { .. }
            | InstKind::Nop
            | InstKind::Poison { .. } => 0,
            InstKind::Switch { .. } => table.compare,
            InstKind::For { source, .. } => self.step_weight(*source),
            InstKind::JumpIf { .. } | InstKind::Diamond { .. } | InstKind::While { .. } => 0,
            InstKind::Jump { .. } | InstKind::Return { .. } | InstKind::Diverge => 0,
        }
    }

    /// An extern weighs what the instance the call names states, and
    /// otherwise its family's row by the call's settled task (RFC-0046
    /// rule 9).
    fn call_weight(&self, callee: &Callee, callee_ty: &Ty) -> u64 {
        match callee {
            Callee::Direct(_) | Callee::Indirect(_) => self.table.local_call,
            Callee::Extern { .. } => match self.laws.cost_of(callee) {
                Some(stated) => stated,
                None => {
                    let Some(effect) = callee_ty.effect() else {
                        panic!("a call's callee type is a function type, and {callee_ty:?} is not")
                    };
                    match effect.task {
                        Task::Heavy => self.table.heavy,
                        Task::Sync | Task::Async => self.table.extern_call,
                    }
                }
            },
        }
    }
}

fn integer(literal: &Literal) -> Option<i128> {
    match literal {
        Literal::Int(value) => Some(*value),
        Literal::IntOf(suffixed) => Some(suffixed.value),
        Literal::String(_)
        | Literal::Float(_)
        | Literal::Char(_)
        | Literal::Bytes(_)
        | Literal::Bool(_)
        | Literal::List(_)
        | Literal::Unit => None,
    }
}
