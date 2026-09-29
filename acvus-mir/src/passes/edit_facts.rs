//! The editor's facts over the analysis manager (passes-analyses.md): one
//! adapter serves every scope. `refresh` only drops what the last edit made
//! stale, in the manager and in the adapter's own tables. A query reads the
//! body it is lent, and computes a fact on its first request after the
//! fact was dropped: an analysis through the manager, and an entry of the
//! adapter's own tables for the one instruction or pair asked about (D47).
//!
//! Each own table is kept while the aspects it is made from are unchanged:
//! - `named`, per instruction, the places it names, from `Loans` and the
//!   instruction;
//! - `loop_between`, per pair of blocks, from `LoopTree`;
//! - `commuting`, per instruction, from its operation and its callee type;
//! - `unobserved`, per instruction, from `Loans`, `IntervalValues`,
//!   `AffineValues`, the instruction and the uses of its result.
//!
//! A `Position::End(block)` is the point after every instruction of the
//! block and before its terminator.

use std::collections::BTreeSet;
use std::rc::Rc;

use rustc_hash::{FxHashMap, FxHashSet};

use smallvec::SmallVec;

use crate::ir::cfg::edit::{
    Asked, ContentFacts, DischargeFacts, DominanceFacts, Facts, InstRef, LoanFacts, OrderFacts, Position, Unobserved,
};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Changes, Definition, InstId, Op, Site, Terminator};
use crate::ir::{Checked, IndexBound, Intrinsic, PathSeg, RefTarget, ValueId, fn_effect};
use crate::passes::analysis::{
    AffineValues, Analysis, AnalysisManager, ContentLiveness, Dominators, Escapes, IntervalValues, LoopTree, Loans,
    Point, PostDominators, Storage, TouchedPlace, ValueBounds,
};
use crate::passes::domain::{Bound, Interval, JoinSemiLattice};
use crate::ty::{CastTy, Concrete, Ty, Types, positions};

pub struct EditFacts<'a, 'd> {
    am: &'a mut AnalysisManager<'d>,
    body: BodyId,
    /// The places each instruction's operands and result name, and those
    /// it reads and writes.
    named: FxHashMap<InstId, BTreeSet<TouchedPlace>>,
    /// `LoopTree::loop_between` by its two blocks.
    loop_between: FxHashMap<(BlockId, BlockId), bool>,
    /// Whether each instruction is a call, or a call's start or taking,
    /// whose callee's declared effect commutes.
    commuting: FxHashMap<InstId, bool>,
    /// The fact that makes each instruction unobservable, where one is
    /// proven.
    unobserved: FxHashMap<InstId, Option<Unobserved>>,
}

impl<'a, 'd> EditFacts<'a, 'd> {
    /// Facts for editing `body`; none is computed before a query asks for
    /// it. Cost: O(1).
    pub fn new(am: &'a mut AnalysisManager<'d>, body: BodyId) -> Self {
        EditFacts {
            am,
            body,
            named: FxHashMap::default(),
            loop_between: FxHashMap::default(),
            commuting: FxHashMap::default(),
            unobserved: FxHashMap::default(),
        }
    }

    /// `A`'s facts for the body asked about, from the manager, which keeps
    /// them until an edit changes an aspect `A` reads. Cost: O(1) when
    /// kept, else `A`'s own.
    ///
    /// # Panics
    /// `asked` is about another body than these facts'.
    fn get<A>(&mut self, asked: Asked<'_>) -> Rc<A::Facts>
    where
        A: Analysis,
    {
        assert_eq!(asked.body_id(), self.body, "the facts answer for {:?}, not {:?}", self.body, asked.body_id());
        self.am.get::<A>(asked.module(), self.body)
    }

    fn loop_between(&mut self, asked: Asked<'_>, from: BlockId, to: BlockId) -> bool {
        if let Some(between) = self.loop_between.get(&(from, to)) {
            return *between;
        }
        let between = self.get::<LoopTree>(asked).loop_between(from, to);
        self.loop_between.insert((from, to), between);
        between
    }

    fn named(&mut self, asked: Asked<'_>, inst: InstRef) -> &BTreeSet<TouchedPlace> {
        if !self.named.contains_key(&inst.id()) {
            let loans = self.get::<Loans>(asked);
            let places = named_places(asked.body(), &loans, inst.id());
            self.named.insert(inst.id(), places);
        }
        &self.named[&inst.id()]
    }
}

/// Whether `aspects` holds an aspect `changes` touched.
fn stale(aspects: &[Aspect], changes: &Changes) -> bool {
    aspects.iter().any(|aspect| changes.touched(*aspect))
}

impl Facts for EditFacts<'_, '_> {
    /// Cost: O(kept analyses · aspects), and O(1) per own table.
    ///
    /// # Panics
    /// `body` is not the body these facts answer for.
    fn refresh(&mut self, body: BodyId, changes: &Changes) {
        assert_eq!(body, self.body, "the facts answer for {:?}, not {body:?}", self.body);
        self.am.invalidate(body, changes);
        if stale(Loans::READS, changes) {
            self.named.clear();
        }
        if stale(LoopTree::READS, changes) {
            self.loop_between.clear();
        }
        if changes.touched(Aspect::Instructions) || changes.touched(Aspect::Types) {
            self.commuting.clear();
        }
        if stale(Loans::READS, changes) || stale(IntervalValues::READS, changes) || stale(AffineValues::READS, changes) {
            self.unobserved.clear();
        }
    }
}

impl DominanceFacts for EditFacts<'_, '_> {
    /// Cost: O(1), and `Dominators`' own when an edit of the CFG's shape
    /// dropped it.
    fn dominates(&mut self, asked: Asked<'_>, value: ValueId, at: Position) -> bool {
        let body = asked.body();
        let defined = match body.definition(value) {
            Definition::Param(_) | Definition::Capture(_) | Definition::OrderParam | Definition::Slot(_) => return true,
            Definition::BlockParam(block, _) => Point::start(body, block),
            Definition::Inst(inst) => Point::after(body, inst),
        };
        self.get::<Dominators>(asked).point_dominates(body, defined, point(at))
    }

    /// Cost: as `dominates`.
    fn position_dominates(&mut self, asked: Asked<'_>, at: Position, site: Site) -> bool {
        let read = match site {
            Site::Inst(inst) => Point::Before(inst),
            Site::Term(block) => Point::End(block),
        };
        self.get::<Dominators>(asked).point_dominates(asked.body(), point(at), read)
    }

    /// Cost: O(1) for a pair of blocks asked before under the same loop
    /// tree, and otherwise O(blocks + edges), for the loops between the
    /// two; and the analyses' own when an edit of the CFG's shape dropped
    /// them.
    fn control_equivalent(&mut self, asked: Asked<'_>, a: Position, b: Position) -> bool {
        let body = asked.body();
        let (a, b) = (point(a), point(b));
        let dominators = self.get::<Dominators>(asked);
        let post_dominators = self.get::<PostDominators>(asked);
        let mut in_order = |earlier: Point, later: Point| {
            dominators.point_dominates(body, earlier, later)
                && post_dominators.point_post_dominates(body, later, earlier)
                && !self.loop_between(asked, earlier.block(body), later.block(body))
        };
        in_order(a, b) || in_order(b, a)
    }
}

impl LoanFacts for EditFacts<'_, '_> {
    /// Two places conflict where they may overlap: a call's declared
    /// access names the elements it touches. Cost: O(places the two touch,
    /// squared), once `Loans` is current.
    fn conflicts(&mut self, asked: Asked<'_>, inst: InstRef, write: InstRef) -> bool {
        let loans = self.get::<Loans>(asked);
        let written = loans.written_places(Site::Inst(write.id()));
        if written.is_empty() {
            return false;
        }
        let body = asked.body();
        let named = self.named(asked, inst);
        written.iter().any(|written| named.iter().any(|place| written.may_overlap(place, body)))
    }

    /// An instruction that names no place, sees no context beyond the
    /// program, and whose operands and result are no slot and hold no
    /// position (ownership §5), names no storage: `Loans` gives every such
    /// value no loan and such an instruction no read or write. A call may
    /// see contexts, which its effect states (context §1). Only another
    /// instruction needs `Loans`. Cost: O(its
    /// operands · their types' size), and `Loans`' own when an edit dropped
    /// it and the instruction is of the other kind.
    fn names_storage(&mut self, asked: Asked<'_>, inst: InstRef) -> bool {
        let body = asked.body();
        let record = body.inst(inst.id());
        let names_a_place = matches!(
            record.op(),
            Op::Take { .. }
                | Op::Assign { .. }
                | Op::Ref { .. }
                | Op::FunctionCall { .. }
                | Op::TakeEffect { .. }
                | Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, .. }
        );
        let types = asked.module().types();
        let decls = self.am.decls();
        let holds_nothing = |value: ValueId| {
            !matches!(body.definition(value), Definition::Slot(_)) && positions(types, decls, body.ty_of(value)) == 0
        };
        if !names_a_place && record.operands().into_iter().chain(record.result()).all(holds_nothing) {
            return false;
        }
        !self.named(asked, inst).is_empty()
    }

    /// Cost: O(1), and `Escapes`' own when an edit dropped it.
    fn escapes(&mut self, asked: Asked<'_>, slot: ValueId) -> Option<Site> {
        self.get::<Escapes>(asked).escape(slot)
    }
}

impl ContentFacts for EditFacts<'_, '_> {
    /// Read from the content liveness. Cost: O(1), and its own when an edit
    /// dropped it.
    fn content_read_after(&mut self, asked: Asked<'_>, place: RefTarget, at: Position) -> bool {
        let content = self.get::<ContentLiveness>(asked);
        let live = match at {
            Position::Before(inst) => content.live_before(inst.id()),
            Position::End(block) => content.live_before_terminator(block),
        };
        live.contains(&by_name(place))
    }

    /// Read from the content liveness. Cost: O(the place's units), and its
    /// own when an edit dropped it.
    fn part_read_after(&mut self, asked: Asked<'_>, place: RefTarget, path: &[PathSeg], at: Position) -> bool {
        let content = self.get::<ContentLiveness>(asked);
        let place = by_name(place);
        match at {
            Position::Before(inst) => content.part_live_before(inst.id(), place, path),
            Position::End(block) => content.part_live_before_terminator(block, place, path),
        }
    }
}

/// The storage `place`, a place by name, is.
///
/// # Panics
/// `place` is reached through a reference.
fn by_name(place: RefTarget) -> Storage {
    Storage::by_name(&place).unwrap_or_else(|| panic!("{place:?} is not a place by name"))
}

impl OrderFacts for EditFacts<'_, '_> {
    /// Both are calls, or a call's start or taking (effects §6), whose
    /// callee's declared effect commutes (effects §1). Cost: O(1).
    fn may_reorder(&mut self, asked: Asked<'_>, a: InstRef, b: InstRef) -> bool {
        let mut commutes = |inst: InstRef| {
            *self.commuting.entry(inst.id()).or_insert_with(|| commutes(asked.body(), asked.module().types(), inst.id()))
        };
        commutes(a) && commutes(b)
    }
}

impl DischargeFacts for EditFacts<'_, '_> {
    /// A check that never fires and a store never read, proven for `inst`
    /// alone. Cost: O(1) when
    /// kept; else `Proven::unobserved`'s, and the analyses' own when an
    /// edit dropped them.
    fn discharged(&mut self, asked: Asked<'_>, inst: InstRef) -> Option<Unobserved> {
        if let Some(proven) = self.unobserved.get(&inst.id()) {
            return *proven;
        }
        let loans = self.get::<Loans>(asked);
        let intervals = self.get::<IntervalValues>(asked);
        let affine = self.get::<AffineValues>(asked);
        let body = asked.body();
        let types = asked.module().types();
        let decls = self.am.decls();
        let bounds = ValueBounds { intervals: &intervals, affine: &affine, decls, body, types };
        let proven = Proven { body, types, loans: &loans, bounds };
        let fact = proven.unobserved(inst.id());
        self.unobserved.insert(inst.id(), fact);
        fact
    }
}

/// The places the instruction `inst` of `body` names: what it reads and
/// writes, the storages its result's loans name, and, unless its callee's
/// declared access states every place it touches, the storages its
/// operands' loans name and a slot it names directly. Cost: O(operands ·
/// positions · loans).
fn named_places(body: &Body, loans: &Loans, inst: InstId) -> BTreeSet<TouchedPlace> {
    let at = Site::Inst(inst);
    let record = body.inst(inst);
    let mut places: BTreeSet<TouchedPlace> = loans.read_places(at).iter().chain(loans.written_places(at)).copied().collect();
    let operands = match loans.by_access(at) {
        true => SmallVec::new(),
        false => record.operands(),
    };
    for value in operands.into_iter().chain(record.result()) {
        match body.definition(value) {
            Definition::Slot(_) => {
                places.insert(TouchedPlace::whole(Storage::Slot(value)));
            }
            Definition::Param(_)
            | Definition::Capture(_)
            | Definition::OrderParam
            | Definition::BlockParam(..)
            | Definition::Inst(..) => {
                places.extend(loans.all_of(value).into_iter().map(|loan| TouchedPlace::whole(loan.storage)));
            }
        }
    }
    places
}

fn point(at: Position) -> Point {
    match at {
        Position::Before(inst) => Point::Before(inst.id()),
        Position::End(block) => Point::End(block),
    }
}

/// What the facts prove an instruction of `body` unobservable from.
struct Proven<'f> {
    body: &'f Body,
    types: &'f Types<Concrete>,
    loans: &'f Loans,
    bounds: ValueBounds<'f>,
}

impl Proven<'_> {
    /// Whether a fact proves `inst` unobservable (D55, G1):
    /// - a check whose condition the intervals show holds on every run
    ///   that reaches it;
    /// - a store whose storage no read reaches before a write replaces it
    ///   or the body ends, per the loans. A store on the order chain, a
    ///   volatile context's, is an effect and never proven: nothing merges
    ///   or removes it (context §3).
    ///
    /// Cost: O(the relations of a slice's impl) for a check, and O(storages
    /// it writes · the body's sites and edges) for a store.
    fn unobserved(&self, inst: InstId) -> Option<Unobserved> {
        let record = self.body.inst(inst);
        match record.op() {
            Op::Check { op, left, right } => self.check_holds(*op, *left, *right).then_some(Unobserved::NeverFires),
            Op::CheckSteps { from, step, count } => self.steps_hold(*from, *step, *count).then_some(Unobserved::NeverFires),
            Op::Index { slice, index, bound: IndexBound::Checked, .. }
            | Op::IndexSet { slice, index, bound: IndexBound::Checked, .. } => {
                self.bounds.index_in_bounds(*slice, *index).then_some(Unobserved::NeverFires)
            }
            Op::Assign { .. } if record.order().is_none() => {
                store_never_read(self.body, self.loans, inst).then_some(Unobserved::NeverRead)
            }
            _ => None,
        }
    }

    /// Whether `left op right`, over the integers, fits the width the two
    /// share on every run that reaches the check.
    fn check_holds(&self, op: Checked, left: ValueId, right: ValueId) -> bool {
        let (left_at, right_at) = (self.bounds.value_interval(left), self.bounds.value_interval(right));
        let exact = match op {
            Checked::Add => left_at.sum(&right_at),
            Checked::Sub => left_at.difference(&right_at),
            Checked::Mul => left_at.product(&right_at),
        };
        exact.le(&self.width_of(left))
    }

    /// Whether `from + count·step`, over the integers, fits the width `from`
    /// and `step` share on every run that reaches the check; the value is
    /// monotone in the steps, so every step fits where the last one does.
    fn steps_hold(&self, from: ValueId, step: ValueId, count: ValueId) -> bool {
        let [from_at, step_at, count_at] = [from, step, count].map(|value| self.bounds.value_interval(value));
        from_at.sum(&count_at.product(&step_at)).le(&self.width_of(from))
    }

    /// Every integer of `value`'s width.
    ///
    /// # Panics
    /// `value` is not of an integer type: a check's operands are.
    fn width_of(&self, value: ValueId) -> Interval {
        match CastTy::of_kind(self.types.kind(self.body.ty_of(value))) {
            Some(CastTy::Int(width)) => Interval::Range { lo: Bound::Finite(width.min()), hi: Bound::Finite(width.max()) },
            other => panic!("a check's operand {value:?} is of an integer type, not {other:?}"),
        }
    }
}

/// The storage and path an `Assign` names by name: a slot, a by-value
/// parameter's own storage or a context; `None` for one through a
/// reference.
fn direct_place(op: &Op) -> Option<(Storage, &[PathSeg])> {
    let Op::Assign { target, path, .. } = op else {
        return None;
    };
    Storage::by_name(target).map(|storage| (storage, path.as_slice()))
}

/// Whether no read of any storage the store `store` writes is reached from
/// just after it, before a write that replaces what it stored or the body's
/// end. A write replaces it where it assigns a place that holds the stored
/// one: a prefix of its path in the same storage, or the whole storage
/// where the store went through a reference. The body's end ends its own
/// slots and parameters' storage, and not an outside storage, which the
/// caller reads after the return; a run that dies owes nothing (effects
/// §5).
fn store_never_read(body: &Body, loans: &Loans, store: InstId) -> bool {
    let stored_path = direct_place(body.inst(store).op()).map(|(_, path)| path);
    let replaces = |site_op: &Op, storage: Storage| {
        let Some((written, path)) = direct_place(site_op) else {
            return false;
        };
        let covers = match stored_path {
            Some(stored) => stored.starts_with(path),
            None => path.is_empty(),
        };
        written == storage && covers
    };
    loans.writes(Site::Inst(store)).iter().all(|storage| {
        let mut visited: FxHashSet<BlockId> = FxHashSet::default();
        let mut pending: Vec<(BlockId, Option<InstId>)> = vec![(body.block_of(store), body.next_in_block(store))];
        while let Some((at, from)) = pending.pop() {
            let mut replaced = false;
            let mut next = from;
            while let Some(inst) = next {
                if loans.reads(Site::Inst(inst)).contains(storage) {
                    return false;
                }
                if replaces(body.inst(inst).op(), *storage) {
                    replaced = true;
                    break;
                }
                next = body.next_in_block(inst);
            }
            if replaced {
                continue;
            }
            if loans.reads(Site::Term(at)).contains(storage) {
                return false;
            }
            match body.block(at).terminator() {
                Terminator::Return { .. } if matches!(storage, Storage::Outside { .. }) => return false,
                terminator => {
                    for edge in terminator.edges() {
                        if visited.insert(edge.to) {
                            pending.push((edge.to, body.first_inst(edge.to)));
                        }
                    }
                }
            }
        }
        true
    })
}

/// Whether `inst` is a call, a start or a taking whose callee's declared
/// effect commutes (effects §1, §6). A taking's callee is its start's.
/// Cost: O(1).
fn commutes(body: &Body, types: &Types<Concrete>, inst: InstId) -> bool {
    let callee_ty = match body.inst(inst).op() {
        Op::FunctionCall { callee_ty, .. } | Op::Spawn { callee_ty, .. } => *callee_ty,
        Op::TakeEffect { src } => started_callee(body, *src),
        _ => return false,
    };
    let Some(effect) = fn_effect(types, callee_ty) else {
        panic!("a call's callee type is a function type")
    };
    effect.commutes()
}

/// The callee type of the start whose handle is `handle`.
///
/// # Panics
/// `handle` is not a `Spawn`'s result: a taking takes a start's handle
/// (effects §6).
fn started_callee(body: &Body, handle: ValueId) -> Ty {
    let Definition::Inst(start) = body.definition(handle) else {
        panic!("a taking's handle {handle:?} is a start's result (effects §6)")
    };
    match body.inst(start).op() {
        Op::Spawn { callee_ty, .. } => *callee_ty,
        other => panic!("a taking's handle is a start's result (effects §6), not {other:?}'s"),
    }
}

#[cfg(test)]
mod tests {
    use acvus_ast::{Literal, Span};
    use acvus_utils::{Interner, QualifiedRef};

    use super::*;
    use crate::decl::{ContextDecl, Core, Declarations, LoopMarkers, RawDeclarations};
    use crate::ir::cfg::edit::{Asked, CannotDischarge, CannotMove, Discharge, Editor, Motion, NoFacts, Remove, Value};
    use crate::ir::cfg::{Inst, Module, Terminator};
    use crate::ir::{BinOp, Callee, Overflow, RefTarget};
    use crate::ty::{
        Calls, Contexts, Cost, Effect, EffectTerm, FlowTerm, Flows, Holds, IntTy, Mutability, Reissue, SplitCost,
        TyKind, TypeArg,
    };

    struct Fixture {
        module: Module,
        interner: Interner,
        int: Ty,
        bool: Ty,
        /// Opaque, and declared to commute with other calls.
        commuting: Ty,
        /// Opaque, commuting, and splittable.
        commuting_splittable: Ty,
        /// Opaque and not commuting.
        opaque: Ty,
        /// `&i64`.
        ref_int: Ty,
    }

    fn fixture() -> Fixture {
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let bool = types.intern(TyKind::Bool);
        let mut call_ty = |effect: Effect| {
            types.intern(TyKind::Fn {
                params: Box::new([]),
                ret: int,
                effect: EffectTerm::Known(effect),
                flows: FlowTerm::Known(Flows::none()),
                calls: Calls::Many,
                holds: Holds::Nothing,
                values: (),
            })
        };
        let opaque_effect = |commutes: bool, split: SplitCost| {
            Effect::new(Reissue::Opaque, commutes, Contexts::new(), Contexts::new(), split)
        };
        let commuting = call_ty(opaque_effect(true, SplitCost::Unavailable));
        let commuting_splittable = call_ty(opaque_effect(true, SplitCost::WithCost(Cost::stated(1))));
        let opaque = call_ty(Effect::opaque());
        let ref_int = types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(int)));
        Fixture {
            module: Module::new(types),
            interner: Interner::new(),
            int,
            bool,
            commuting,
            commuting_splittable,
            opaque,
            ref_int,
        }
    }

    /// Declarations of two contexts of `i64`: `page`, held in a slot, and
    /// `clock`, volatile.
    fn declarations(interner: &Interner) -> Declarations {
        let mut types = Types::<crate::ty::Poly>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let context = |name: &str, volatile: bool| (QualifiedRef::root(interner.intern(name)), ContextDecl { ty: int, volatile });
        Declarations::verify(RawDeclarations {
            types,
            functions: Vec::new(),
            lang_impls: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: vec![context("page", false), context("clock", true)],
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("two contexts under two names verify")
    }

    fn constant(n: i128) -> Op {
        Op::Const { value: Literal::Int(n) }
    }

    fn add(left: ValueId, right: ValueId) -> Op {
        Op::BinOp { op: BinOp::Add(Overflow::Trap), left, right }
    }

    fn call(f: &Fixture, callee_ty: Ty) -> Op {
        Op::FunctionCall { callee: Callee::Direct(QualifiedRef::root(f.interner.intern("f"))), callee_ty, args: Vec::new() }
    }

    /// Every instruction of `body`, in block order, as the editor names them.
    fn inst_refs(module: &mut Module, body: BodyId) -> Vec<InstRef> {
        let mut no_facts = NoFacts;
        let mut editor = Editor::<Remove>::new(module, body, &mut no_facts);
        let mut cursor = editor.cursor();
        std::iter::from_fn(|| cursor.next()).collect()
    }

    struct DiamondBody {
        id: BodyId,
        entry: BlockId,
        then_block: BlockId,
        else_block: BlockId,
        join: BlockId,
        c: ValueId,
        x: ValueId,
        t: ValueId,
        p: ValueId,
    }

    /// ```text
    /// entry(c): x = 0; diamond c { then } else { else } join
    /// then:     t = x + x; jump join(t)
    /// else:     jump join(x)
    /// join(p):  return p
    /// ```
    fn diamond_body(f: &mut Fixture) -> DiamondBody {
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let entry = builder.current_block();
        let x = builder.push(constant(0), int, Span::ZERO);
        let [then_block, else_block] = [(); 2].map(|()| builder.new_block(&[]));
        let join = builder.new_block(&[int]);
        let p = builder.block_params(join)[0];
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        let t = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: vec![t] }, acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: vec![x] }, acvus_ast::Span::ZERO);
        builder.switch_to(join);
        builder.terminate(Terminator::Return { value: p, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        DiamondBody { id, entry, then_block, else_block, join, c, x, t, p }
    }

    #[test]
    fn dominates_reads_where_each_value_is_defined() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, d.id);

        assert!(facts.dominates(Asked::new(&f.module, d.id), d.c, Position::Before(x_at)), "a parameter is defined before the entry runs");
        assert!(!facts.dominates(Asked::new(&f.module, d.id), d.x, Position::Before(x_at)), "a result is not defined before its instruction");
        assert!(facts.dominates(Asked::new(&f.module, d.id), d.x, Position::End(d.entry)));
        assert!(facts.dominates(Asked::new(&f.module, d.id), d.x, Position::Before(t_at)));
        assert!(facts.dominates(Asked::new(&f.module, d.id), d.x, Position::End(d.join)));
        assert!(facts.dominates(Asked::new(&f.module, d.id), d.t, Position::End(d.then_block)));
        assert!(!facts.dominates(Asked::new(&f.module, d.id), d.t, Position::End(d.join)), "the else arm reaches the join without t");
        assert!(!facts.dominates(Asked::new(&f.module, d.id), d.t, Position::End(d.else_block)));
        assert!(facts.dominates(Asked::new(&f.module, d.id), d.p, Position::End(d.join)), "a block parameter is defined at its block's start");
        assert!(!facts.dominates(Asked::new(&f.module, d.id), d.p, Position::End(d.then_block)));
    }

    #[test]
    fn position_dominates_orders_a_block_and_includes_the_site_just_after() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, d.id);

        let t_site = Site::Inst(t_at.id());
        assert!(facts.position_dominates(Asked::new(&f.module, d.id), Position::Before(t_at), t_site), "standing just before the site");
        assert!(!facts.position_dominates(Asked::new(&f.module, d.id), Position::End(d.then_block), t_site));
        assert!(facts.position_dominates(Asked::new(&f.module, d.id), Position::End(d.then_block), Site::Term(d.then_block)));
        assert!(facts.position_dominates(Asked::new(&f.module, d.id), Position::Before(x_at), Site::Term(d.join)));
        assert!(facts.position_dominates(Asked::new(&f.module, d.id), Position::End(d.entry), t_site));
        assert!(!facts.position_dominates(Asked::new(&f.module, d.id), Position::End(d.then_block), Site::Term(d.join)));
    }

    #[test]
    fn control_equivalence_needs_dominance_one_way_and_post_dominance_the_other() {
        let mut f = fixture();
        let d = diamond_body(&mut f);
        let [x_at, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, d.id);

        assert!(facts.control_equivalent(Asked::new(&f.module, d.id), Position::Before(x_at), Position::End(d.join)));
        assert!(facts.control_equivalent(Asked::new(&f.module, d.id), Position::End(d.join), Position::Before(x_at)), "symmetric");
        assert!(facts.control_equivalent(Asked::new(&f.module, d.id), Position::Before(t_at), Position::End(d.then_block)));
        assert!(!facts.control_equivalent(Asked::new(&f.module, d.id), Position::Before(x_at), Position::Before(t_at)), "t runs on one arm");
        assert!(!facts.control_equivalent(Asked::new(&f.module, d.id), Position::End(d.then_block), Position::End(d.else_block)));
    }

    /// `a = commuting(); b = commuting(); c = opaque(); d = a + b; return d`,
    /// in an ordered body.
    #[test]
    fn may_reorder_holds_between_two_calls_that_commute() {
        let mut f = fixture();
        let (int, commuting, opaque) = (f.int, f.commuting, f.opaque);
        let (commuting_call, opaque_call) = (call(&f, commuting), call(&f, opaque));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push_effect(commuting_call.clone(), int, Span::ZERO);
        let b = builder.push_effect(commuting_call, int, Span::ZERO);
        builder.push_effect(opaque_call, int, Span::ZERO);
        let d = builder.push(add(a, b), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: d, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [a_at, b_at, c_at, d_at] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has four instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        assert!(facts.may_reorder(Asked::new(&f.module, id), a_at, b_at));
        assert!(facts.may_reorder(Asked::new(&f.module, id), b_at, a_at));
        assert!(!facts.may_reorder(Asked::new(&f.module, id), a_at, c_at), "the opaque call does not commute");
        assert!(!facts.may_reorder(Asked::new(&f.module, id), c_at, b_at));
        assert!(!facts.may_reorder(Asked::new(&f.module, id), a_at, d_at), "an addition is not a call");
    }

    /// An instruction inserted through `Editor<Value>` is known to the next
    /// check the editor makes: replacing `t` by it passes only if the refresh
    /// after the insertion saw where it stands.
    #[test]
    fn refresh_after_an_insertion_through_the_value_editor_reflects_it() {
        let mut f = fixture();
        let int = f.int;
        let d = diamond_body(&mut f);
        let [_, t_at] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, d.id);

        let one = {
            let mut editor = Editor::<Value>::new(&mut f.module, d.id, &mut facts);
            let one = editor.insert(Position::Before(t_at), constant(1), int, Span::ZERO);
            editor.replace_uses(d.t, one).expect("the inserted value dominates every use of t");
            editor.finish();
            one
        };
        assert!(facts.dominates(Asked::new(&f.module, d.id), one, Position::End(d.then_block)));
        assert!(!facts.dominates(Asked::new(&f.module, d.id), one, Position::End(d.join)));
        let [_, one_at, t_now] = inst_refs(&mut f.module, d.id)[..] else {
            panic!("the body has three instructions")
        };
        let body = f.module.body(d.id);
        assert_eq!((body.block_of(one_at.id()), body.index_in_block(one_at.id())), (d.then_block, 0));
        assert!(!facts.dominates(Asked::new(&f.module, d.id), d.t, Position::Before(t_now)), "t moved down by one");
        assert!(facts.dominates(Asked::new(&f.module, d.id), one, Position::Before(t_now)));
    }

    /// A split through `Editor<Effect>`: the start and the taking of a
    /// commuting call are known to commute after the refresh, at the places
    /// the split gave them.
    #[test]
    fn refresh_after_a_split_through_the_effect_editor_reflects_it() {
        let mut f = fixture();
        let (int, splittable, commuting) = (f.int, f.commuting_splittable, f.commuting);
        let (split_call, other_call) = (call(&f, splittable), call(&f, commuting));
        let mut builder = f.module.build_body(true, int);
        let a = builder.push_effect(split_call, int, Span::ZERO);
        builder.push_effect(other_call, int, Span::ZERO);
        builder.terminate(Terminator::Return { value: a, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [a_at, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        {
            let mut editor = Editor::<crate::ir::cfg::edit::Effect>::new(&mut f.module, id, &mut facts);
            editor.split_call(a_at).expect("the callee lets the call be split");
            editor.finish();
        }
        let refs = inst_refs(&mut f.module, id);
        let insts: Vec<&Inst> = f.module.body(id).blocks().flat_map(|(_, block)| block.insts()).collect();
        let start = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::Spawn { .. }));
        let take = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::TakeEffect { .. }));
        let other = refs.iter().zip(&insts).find(|(_, inst)| matches!(inst.op(), Op::FunctionCall { .. }));
        let (Some((start, _)), Some((take, _)), Some((other, _))) = (start, take, other) else {
            panic!("the split left a start, a taking and the other call")
        };
        assert!(facts.may_reorder(Asked::new(&f.module, id), *start, *other));
        assert!(facts.may_reorder(Asked::new(&f.module, id), *take, *other));
        assert!(facts.may_reorder(Asked::new(&f.module, id), *start, *take));
    }

    fn assign(slot: ValueId, value: ValueId) -> Op {
        Op::Assign { target: RefTarget::Var(slot), path: Vec::new(), value, restores: false }
    }

    fn take(slot: ValueId) -> Op {
        Op::Take { target: RefTarget::Var(slot), path: Vec::new(), taken_out: false }
    }

    /// ```text
    /// entry(c): x = 0; jump header
    /// header:   while c { body } else exit
    /// body:     jump header
    /// exit:     y = x + x; return y
    /// ```
    /// The entry dominates the exit and the exit post-dominates the entry,
    /// but the loop between them may never end.
    #[test]
    fn control_equivalence_is_refused_across_a_loop() {
        let mut f = fixture();
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let entry = builder.current_block();
        let x = builder.push(constant(0), int, Span::ZERO);
        let [header, loop_body, exit] = [(); 3].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(header);
        builder.terminate(Terminator::While {
            cond: c,
            stages: crate::ir::cfg::Stages::lowered(loop_body),
            exit,
            exit_args: Vec::new(),
        }, acvus_ast::Span::ZERO);
        builder.switch_to(loop_body);
        builder.terminate(Terminator::Jump { to: header, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(exit);
        let y = builder.push(add(x, x), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: y, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [x_at, y_at] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has two instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        assert!(facts.dominates(Asked::new(&f.module, id), x, Position::Before(y_at)));
        assert!(!facts.control_equivalent(Asked::new(&f.module, id), Position::End(entry), Position::Before(y_at)), "a loop lies between");
        assert!(!facts.control_equivalent(Asked::new(&f.module, id), Position::Before(y_at), Position::Before(x_at)));
        assert!(!facts.control_equivalent(Asked::new(&f.module, id), Position::End(header), Position::End(loop_body)));
        assert!(facts.control_equivalent(Asked::new(&f.module, id), Position::Before(x_at), Position::End(entry)), "one block, no loop");
        assert!(facts.control_equivalent(Asked::new(&f.module, id), Position::Before(y_at), Position::End(exit)));
    }

    struct StorageBody {
        id: BodyId,
        x: ValueId,
        y: ValueId,
        /// The load of `y`.
        u: ValueId,
    }

    /// ```text
    /// x = 1; y = 2; r = &x; t = *r; u = y; return t
    /// ```
    /// with `x`, `y` slots and the stores and loads whole.
    fn storage_body(f: &mut Fixture) -> StorageBody {
        let int = f.int;
        let ref_int = f.ref_int;
        let mut builder = f.module.build_body(false, int);
        let x = builder.slot(int, Span::ZERO);
        let y = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(x, one), Span::ZERO);
        builder.push_unit(assign(y, one), Span::ZERO);
        let r = builder.push(Op::Ref { target: RefTarget::Var(x), path: Vec::new(), mutability: Mutability::Shared }, ref_int, Span::ZERO);
        let t = builder.push(Op::Take { target: RefTarget::Through(r), path: Vec::new(), taken_out: false }, int, Span::ZERO);
        let u = builder.push(take(y), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        StorageBody { id, x, y, u }
    }

    #[test]
    fn conflicts_intersects_what_a_write_writes_with_what_an_instruction_names() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [_, assign_x, assign_y, make_ref, read_through, take_y] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, s.id);

        assert!(facts.conflicts(Asked::new(&f.module, s.id), read_through, assign_x), "the read through r reads x");
        assert!(facts.conflicts(Asked::new(&f.module, s.id), make_ref, assign_x), "r names x");
        assert!(!facts.conflicts(Asked::new(&f.module, s.id), read_through, assign_y));
        assert!(facts.conflicts(Asked::new(&f.module, s.id), take_y, assign_y));
        assert!(!facts.conflicts(Asked::new(&f.module, s.id), take_y, assign_x));
        assert!(!facts.conflicts(Asked::new(&f.module, s.id), assign_x, read_through), "a read writes nothing");
    }

    #[test]
    fn escapes_names_the_first_use_a_promotion_cannot_rewrite() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [.., make_ref, _, _] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, s.id);

        assert_eq!(facts.escapes(Asked::new(&f.module, s.id), s.x), Some(Site::Inst(make_ref.id())));
        assert_eq!(facts.escapes(Asked::new(&f.module, s.id), s.y), None);
    }

    /// A move through `Editor<Motion>` above the store to `y` is refused,
    /// since the load of `y` names what it writes; a move above the reads of
    /// `x` is made, and the refresh after it knows the load's new place.
    #[test]
    fn refresh_after_a_move_through_the_motion_editor_reflects_it() {
        let mut f = fixture();
        let s = storage_body(&mut f);
        let [_, _, assign_y, make_ref, _, take_y] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, s.id);

        {
            let mut editor = Editor::<Motion>::new(&mut f.module, s.id, &mut facts);
            assert_eq!(
                editor.move_to(take_y, Position::Before(assign_y)),
                Err(CannotMove::CrossesConflictingWrite(assign_y))
            );
            editor.move_to(take_y, Position::Before(make_ref)).expect("the load crosses only reads of x");
            editor.finish();
        }
        let [_, _, assign_y, moved, make_ref, _] = inst_refs(&mut f.module, s.id)[..] else {
            panic!("the body has six instructions")
        };
        assert!(matches!(f.module.body(s.id).inst(moved.id()).op(), Op::Take { .. }));
        assert!(facts.dominates(Asked::new(&f.module, s.id), s.u, Position::Before(make_ref)), "the load now stands before the Ref");
        assert!(!facts.dominates(Asked::new(&f.module, s.id), s.u, Position::Before(moved)));
        assert!(facts.conflicts(Asked::new(&f.module, s.id), moved, assign_y), "the load's storage is known at its new place");
    }

    fn assign_context(context: QualifiedRef, value: ValueId) -> Op {
        Op::Assign { target: RefTarget::Context(context), path: Vec::new(), value, restores: false }
    }

    /// ```text
    /// one = 1; s = one; s = one; t = s; check(one + one); return t
    /// ```
    /// with `s` a slot: the first store is replaced before any read, and the
    /// second is read.
    #[test]
    fn a_store_replaced_before_any_read_is_never_read() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(false, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        let t = builder.push(take(s), int, Span::ZERO);
        builder.push_unit(Op::Check { op: crate::ir::Checked::Add, left: one, right: one }, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [_, first, second, _, check] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has five instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        assert_eq!(facts.discharged(Asked::new(&f.module, id), first), Some(Unobserved::NeverRead));
        assert_eq!(facts.discharged(Asked::new(&f.module, id), second), None, "the load of s reads it");
        assert_eq!(facts.discharged(Asked::new(&f.module, id), check), Some(Unobserved::NeverFires), "1 + 1 fits an i64");
    }

    /// `s = 1; if c { t = s } else {}; return 0`: the store is read on one
    /// arm, and a store no path reads before the end is never read.
    #[test]
    fn a_store_read_on_one_path_is_read_and_one_left_at_the_end_is_not() {
        let mut f = fixture();
        let (nc, int, bool) = (f.interner.intern("c"), f.int, f.bool);
        let mut builder = f.module.build_body(false, int);
        let c = builder.param(nc, bool, acvus_ast::Span::ZERO);
        let (s, unread) = (builder.slot(int, Span::ZERO), builder.slot(int, Span::ZERO));
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(unread, one), Span::ZERO);
        let [then_block, else_block, join] = [(); 3].map(|()| builder.new_block(&[]));
        builder.terminate(Terminator::Diamond {
            cond: c,
            then_to: then_block,
            then_args: Vec::new(),
            else_to: else_block,
            else_args: Vec::new(),
            join,
        }, acvus_ast::Span::ZERO);
        builder.switch_to(then_block);
        builder.push(take(s), int, Span::ZERO);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(else_block);
        builder.terminate(Terminator::Jump { to: join, args: Vec::new() }, acvus_ast::Span::ZERO);
        builder.switch_to(join);
        builder.terminate(Terminator::Return { value: one, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [_, read_store, unread_store, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has four instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        assert_eq!(facts.discharged(Asked::new(&f.module, id), read_store), None);
        assert_eq!(facts.discharged(Asked::new(&f.module, id), unread_store), Some(Unobserved::NeverRead));
    }

    /// ```text
    /// one = 1; @page = one; @page = one; @clock = one; @clock = one;
    /// return one
    /// ```
    /// with `clock` volatile: a store into a context is judged as any
    /// store, the return reading every context (context §1), and a volatile
    /// context's stores stand on the order chain and are never proven
    /// (context §3).
    #[test]
    fn a_context_store_is_read_at_the_return_and_a_volatile_one_is_never_proven() {
        let mut f = fixture();
        let int = f.int;
        let (page, clock) = (QualifiedRef::root(f.interner.intern("page")), QualifiedRef::root(f.interner.intern("clock")));
        let mut builder = f.module.build_body(true, int);
        builder.context(page, int, false);
        builder.context(clock, int, true);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign_context(page, one), Span::ZERO);
        builder.push_unit(assign_context(page, one), Span::ZERO);
        builder.push_effect_unit(assign_context(clock, one), Span::ZERO);
        builder.push_effect_unit(assign_context(clock, one), Span::ZERO);
        builder.terminate(Terminator::Return { value: one, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [_, overwritten, returned, volatile_overwritten, volatile_returned] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has five instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        assert_eq!(
            facts.discharged(Asked::new(&f.module, id), overwritten),
            Some(Unobserved::NeverRead),
            "@page is assigned whole again before any read"
        );
        assert_eq!(facts.discharged(Asked::new(&f.module, id), returned), None, "the return reads @page");
        assert_eq!(facts.discharged(Asked::new(&f.module, id), volatile_overwritten), None, "a volatile store is never removed");
        assert_eq!(facts.discharged(Asked::new(&f.module, id), volatile_returned), None, "a volatile store is never removed");
    }

    /// A volatile context's store nothing reads again stays: `discharge`
    /// refuses it.
    #[test]
    fn a_volatile_store_is_not_discharged() {
        let mut f = fixture();
        let int = f.int;
        let clock = QualifiedRef::root(f.interner.intern("clock"));
        let mut builder = f.module.build_body(true, int);
        builder.context(clock, int, true);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_effect_unit(assign_context(clock, one), Span::ZERO);
        builder.push_effect_unit(assign_context(clock, one), Span::ZERO);
        builder.terminate(Terminator::Return { value: one, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [_, store, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has three instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);
        let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
        assert_eq!(editor.discharge(store), Err(CannotDischarge::NotProven));
        editor.finish();
        assert_eq!(f.module.body(id).blocks().flat_map(|(_, block)| block.insts()).count(), 3);
    }

    /// Through `Editor<Discharge>` over `EditFacts`: a dead store goes on
    /// the fact the facts prove, and the facts refuse the store that is
    /// read.
    #[test]
    fn discharge_through_the_editor_removes_what_the_facts_prove_unobserved() {
        let mut f = fixture();
        let int = f.int;
        let mut builder = f.module.build_body(true, int);
        let s = builder.slot(int, Span::ZERO);
        let one = builder.push(constant(1), int, Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        builder.push_unit(assign(s, one), Span::ZERO);
        let u = builder.push(take(s), int, Span::ZERO);
        builder.terminate(Terminator::Return { value: u, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [_, dead, live, _] = inst_refs(&mut f.module, id)[..] else {
            panic!("the body has four instructions")
        };
        let decls = declarations(&f.interner);
        let mut am = AnalysisManager::new(&decls);
        let mut facts = EditFacts::new(&mut am, id);

        {
            let mut editor = Editor::<Discharge>::new(&mut f.module, id, &mut facts);
            assert_eq!(editor.discharge(live), Err(CannotDischarge::NotProven));
            assert_eq!(editor.discharge(dead), Ok(()));
            editor.finish();
        }
        let ops: Vec<&Op> = f.module.body(id).blocks().flat_map(|(_, block)| block.insts()).map(Inst::op).collect();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops.iter().filter(|op| matches!(op, Op::Assign { .. })).count(), 1);
    }

    /// Declarations of five externs, the first four stating their access:
    /// `put(m: &mut Map, key: i64, v: i64)` stores `v` into `m[key]`,
    /// `lookup(m: &mut Map, key: i64) -> i64` reads `m[key]`,
    /// `set(s: &mut [i64], other: &mut i64, i: u64, v: i64)` stores `v`
    /// into `s[i]` and touches nothing of `other`,
    /// `peek(s: &mut [i64], i: u64) -> i64` reads `s[i]`, and
    /// `opaque(s: &mut [i64], other: &mut i64, i: u64)` states none.
    struct AccessDecls {
        decls: Declarations,
        map: QualifiedRef,
        put: QualifiedRef,
        lookup: QualifiedRef,
        set: QualifiedRef,
        peek: QualifiedRef,
        opaque: QualifiedRef,
    }

    fn access_declarations(interner: &Interner) -> AccessDecls {
        use crate::decl::declared::{Access, Element, Mode, Place, Stored, Touch};
        use crate::decl::{DeclaredFacts, GenericImplSig, Impls, Scheme, UserTypeDecl};

        let name = |name: &str| QualifiedRef::root(interner.intern(name));
        let map = name("Map");
        let mut types = Types::<crate::ty::Poly>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let index = types.intern(TyKind::Int(IntTy::U64));
        let unit = types.intern(TyKind::Unit);
        let map_ty = types.intern(TyKind::UserDefined {
            id: map,
            type_args: Box::new([]),
            effect: None,
            identity: None,
        });
        let mut_map = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(map_ty)));
        let slice = types.intern(TyKind::Slice(int));
        let mut_slice = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(slice)));
        let mut_int = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
        let mut declare = |id: QualifiedRef, params: &[crate::ty::TyId<crate::ty::Poly>], ret, touches: Option<Vec<Touch>>| {
            let ty = types.intern(TyKind::Fn {
                params: params.into(),
                ret,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(Flows::Every),
                calls: Calls::Many,
                holds: Holds::Nothing,
                values: (),
            });
            let access = touches.map(|touches| Access::Touches(touches.into())).into_iter().collect();
            let facts = DeclaredFacts { access, ..DeclaredFacts::default() };
            let scheme = Scheme {
                ty,
                param_names: (0..params.len()).map(|at| interner.intern(&format!("p{at}"))).collect(),
                bounds: Box::new([]),
                impls: Some(Impls { concrete: Box::new([]), generic: Some(GenericImplSig { facts }) }),
                requires: Box::new([]),
                facts: DeclaredFacts::default(),
            };
            (id, scheme)
        };
        let at = |element: Element| Place { param: 0, element: Some(element) };
        let (put, lookup, set, peek, opaque) = (name("put"), name("lookup"), name("set"), name("peek"), name("opaque"));
        let functions = vec![
            declare(put, &[mut_map, int, int], unit, Some(vec![Touch { place: at(Element::Key(1)), mode: Mode::Store(Stored::Value(2)) }])),
            declare(lookup, &[mut_map, int], int, Some(vec![Touch { place: at(Element::Key(1)), mode: Mode::Read }])),
            declare(set, &[mut_slice, mut_int, index, int], unit, Some(vec![Touch { place: at(Element::Index(2)), mode: Mode::Store(Stored::Value(3)) }])),
            declare(peek, &[mut_slice, index], int, Some(vec![Touch { place: at(Element::Index(1)), mode: Mode::Read }])),
            declare(opaque, &[mut_slice, mut_int, index], unit, None),
        ];
        let decls = Declarations::verify(RawDeclarations {
            types,
            functions,
            lang_impls: Vec::new(),
            user_types: vec![UserTypeDecl {
                id: map,
                type_params: Box::new([]),
                specializes: Box::new([]),
                carries_effect: false,
                carries_identity: false,
                lifetimes: 0,
                kind: crate::decl::UserTypeKind::Opaque,
            }],
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            entries: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .expect("the five externs verify");
        AccessDecls { decls, map, put, lookup, set, peek, opaque }
    }

    /// ```text
    /// entry(m: &mut Map, s: &mut [i64], other: &mut i64, j: u64):
    ///   zero = 0u64; one = 1u64; v = 5; k1 = 1; k2 = 2
    ///   put(m, k1, v); lookup(m, k1); lookup(m, k2)
    ///   set(s, other, zero, v); peek(s, one); peek(s, zero); peek(s, j)
    ///   opaque(s, other, zero); t = *other; return t
    /// ```
    /// A call with an access conflicts only where the places it touches
    /// may overlap: the same key, the same index or an index not known to
    /// differ; a call without one touches all its arguments reach.
    #[test]
    fn a_call_touching_one_element_conflicts_only_where_its_element_may_be_touched() {
        let interner = Interner::new();
        let a = access_declarations(&interner);
        let mut types = Types::<Concrete>::new();
        let int = types.intern(TyKind::Int(IntTy::I64));
        let index = types.intern(TyKind::Int(IntTy::U64));
        let unit = types.intern(TyKind::Unit);
        let map_ty = types.intern(TyKind::UserDefined {
            id: a.map,
            type_args: Box::new([]),
            effect: None,
            identity: None,
        });
        let mut_map = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(map_ty)));
        let slice = types.intern(TyKind::Slice(int));
        let mut_slice = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(slice)));
        let mut_int = types.intern(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(int)));
        let mut callee_ty = |params: &[Ty], ret| {
            types.intern(TyKind::Fn {
                params: params.into(),
                ret,
                effect: EffectTerm::Known(Effect::pure()),
                flows: FlowTerm::Known(Flows::Every),
                calls: Calls::Many,
                holds: Holds::Nothing,
                values: (),
            })
        };
        let put_ty = callee_ty(&[mut_map, int, int], unit);
        let lookup_ty = callee_ty(&[mut_map, int], int);
        let set_ty = callee_ty(&[mut_slice, mut_int, index, int], unit);
        let peek_ty = callee_ty(&[mut_slice, index], int);
        let opaque_ty = callee_ty(&[mut_slice, mut_int, index], unit);
        let mut module = Module::new(types);
        let mut builder = module.build_body(false, int);
        let param = |builder: &mut crate::ir::cfg::Builder<'_>, name: &str, ty| builder.param(interner.intern(name), ty, Span::ZERO);
        let m = param(&mut builder, "m", mut_map);
        let s = param(&mut builder, "s", mut_slice);
        let other = param(&mut builder, "other", mut_int);
        let j = param(&mut builder, "j", index);
        let zero = builder.push(constant(0), index, Span::ZERO);
        let one = builder.push(constant(1), index, Span::ZERO);
        let v = builder.push(constant(5), int, Span::ZERO);
        let k1 = builder.push(constant(1), int, Span::ZERO);
        let k2 = builder.push(constant(2), int, Span::ZERO);
        let call = |id, callee_ty, args| Op::FunctionCall { callee: Callee::Extern { id, r#impl: Some(0), required: Vec::new() }, callee_ty, args };
        builder.push(call(a.put, put_ty, vec![m, k1, v]), unit, Span::ZERO);
        builder.push(call(a.lookup, lookup_ty, vec![m, k1]), int, Span::ZERO);
        builder.push(call(a.lookup, lookup_ty, vec![m, k2]), int, Span::ZERO);
        builder.push(call(a.set, set_ty, vec![s, other, zero, v]), unit, Span::ZERO);
        builder.push(call(a.peek, peek_ty, vec![s, one]), int, Span::ZERO);
        builder.push(call(a.peek, peek_ty, vec![s, zero]), int, Span::ZERO);
        builder.push(call(a.peek, peek_ty, vec![s, j]), int, Span::ZERO);
        builder.push(call(a.opaque, opaque_ty, vec![s, other, zero]), unit, Span::ZERO);
        let t = builder.push(Op::Take { target: RefTarget::Through(other), path: Vec::new(), taken_out: false }, int, Span::ZERO);
        builder.terminate(Terminator::Return { value: t, order: None, span: Span::ZERO }, Span::ZERO);
        let id = builder.seal().expect("the body has no open state");
        let [.., put, same_key, other_key, set, peek_one, peek_zero, peek_j, opaque, read_other] =
            inst_refs(&mut module, id)[..]
        else {
            panic!("the body has fourteen instructions")
        };
        let mut am = AnalysisManager::new(&a.decls);
        let mut facts = EditFacts::new(&mut am, id);
        let mut conflicts = |inst, write| facts.conflicts(Asked::new(&module, id), inst, write);

        assert!(conflicts(same_key, put), "lookup(m, k1) reads the entry put(m, k1, v) stores");
        assert!(conflicts(other_key, put), "k2 may name k1's entry: Loans does not evaluate eq");
        assert!(!conflicts(read_other, put), "put touches only m[k1]");
        assert!(conflicts(peek_zero, set), "s[zero] is the element set stores");
        assert!(!conflicts(peek_one, set), "s[1] and s[0] are two elements");
        assert!(conflicts(peek_j, set), "j may be 0");
        assert!(!conflicts(read_other, set), "set is given other, and touches only s[zero]");
        assert!(conflicts(read_other, opaque), "without an access, a call writes through every &mut it is given");
        assert!(conflicts(peek_one, opaque));
    }
}
