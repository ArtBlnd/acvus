//! B, the typing rule (rfc_compiler: types): terms under solving and the
//! relations between them. It knows no syntax. C's label is on the terms it
//! relates (`ty::Label`); it asks nothing of the program.
//!
//! A variable is bound at most once to a term, or rebound only: to a larger
//! term of one head under `join` where the label allows growth; up the cast
//! tree under `enter`; or to the more opaque of two equal shapes under
//! `eq` (types §1, §6). Knowledge only grows.

mod bound;
mod canonical;
mod cells;
mod closed;
mod copy;
mod freeze;
mod import;
mod relate;
mod scheme;
#[cfg(test)]
mod tests;

use crate::ty::{
    Calls, CallsTerm, Effect, CallsVarId, EffectTerm, EffectVarId, FlowTerm, FlowVarId, Holds, HoldsTerm, HoldsVarId,
    IdentityId, IdentityTerm, IdentityVarId, Infer as InferPhase, IntTy, LenTerm, LenVarId, Mark, MarkTerm, TyId, TyKind, TyVarId,
    Types, FnValues, ValuesVarId,
};

use rustc_hash::{FxHashMap, FxHashSet};

use cells::{Cells, End as CellEnd, FlowRefused, ValuesTerm};

pub use cells::{Exceeds, FlowId};
pub use canonical::{Canonical, CanonicalTable, VarStates};
pub use closed::{Closed, OpenSources};
pub use copy::{AtInstance, CopiedFlow, Reached, VarMarks, VarsOf};
pub use freeze::{FreezeMemo, Unsettled};
pub use import::Imported;
pub use relate::{Coerced, CoercedPart, Entered, MapMismatch, Mismatch, MismatchTerms, ShapeCoercion, Solving};
pub use scheme::{Instantiated, PairAt, RequiredAt};

pub type Term = TyId<InferPhase>;

/// What a type variable is while solving.
#[derive(Debug, Clone, PartialEq)]
pub enum VarState {
    Unbound(Bound),
    Bound(Term),
}

/// A variable's bound, instantiated (extern-fn §1, types §5).
#[derive(Debug, Clone, PartialEq)]
pub enum Bound {
    Any,
    /// An integer type among `among`, signed only where `signed`. A literal
    /// (`literal`) is one whose width a use narrows; at a stall it is `i64`
    /// if `among` has it, the one width if one remains, and refused
    /// otherwise (types §5).
    Integer { signed: bool, among: Box<[IntTy]>, literal: bool },
    Settled,
}

/// How deep in body instantiations a variable or an identity was made
/// (types §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(pub u32);

impl Level {
    /// Outside every body: a context's identities, those a declaration
    /// writes, and those another unit's store made.
    pub const OUTSIDE: Level = Level(0);
    /// A unit's own body.
    pub const UNIT: Level = Level(1);
}

/// The solving state of one checked unit.
pub struct Infer {
    types: Types<InferPhase>,
    ty_vars: Vec<VarState>,
    /// Every variable a reference's target is, which types §4 keeps from
    /// being bound to a reference.
    reference_targets: FxHashSet<TyVarId>,
    /// Every variable that never stands for a marked type (types §8): a
    /// declaration's type variable, and an extern-defined type's argument.
    /// Binding one raises every mark in what it stands for, as a part or
    /// itself; widening where a marked type may stand is how a
    /// specialization comes to change a meaning (`vector<bool>`), so this
    /// stays the narrow rule.
    unmarked: FxHashSet<TyVarId>,
    effect_vars: Cells<EffectTerm<InferPhase>>,
    /// The marks of extern-defined types (types §8), joined as effects are.
    mark_vars: Cells<MarkTerm<InferPhase>>,
    len_vars: Cells<LenTerm<InferPhase>>,
    identity_vars: Cells<IdentityTerm<InferPhase>>,
    /// The body instantiation now being checked (types §3): a unit's body,
    /// a lambda's, and a loop's iteration each raise it by one. Every type
    /// and identity variable, and every identity made, is of the level it
    /// was made at.
    level: Level,
    /// Each type variable's level, by its number. A variable bound into a
    /// term of a lower level takes that lower level, as does every variable
    /// the term holds (level lowering).
    ty_levels: Vec<Level>,
    /// Each identity variable's level, by its number.
    identity_levels: Vec<Level>,
    /// The level each identity was made at. An identity never enters a
    /// variable of a lower level (types §3): it would escape the body
    /// instantiation it belongs to.
    made_at: FxHashMap<IdentityId, Level>,
    flow_vars: Cells<FlowTerm<InferPhase>>,
    holds_vars: Cells<HoldsTerm>,
    calls_vars: Cells<CallsTerm>,
    values_vars: Cells<ValuesTerm>,
    /// Each flow of effects refused since typeck last took them.
    effects_refused: Vec<EffectRefused>,
}

/// A flow `into ⊒ from` of effects that would grow `into`, fixed by `eq`,
/// past its value (D94, D95).
#[derive(Debug, Clone, PartialEq)]
pub struct EffectRefused {
    pub flow: FlowId,
    pub into: EffectTerm<InferPhase>,
    pub exceeds: Exceeds<Effect>,
}

/// Where a term's chain of bound variables ends: at an unbound variable,
/// or at a node, reached through `holder`, the last variable on the way.
#[derive(Debug, Clone, Copy)]
enum End {
    Unbound(TyVarId, Term),
    Node { holder: Option<(TyVarId, Term)>, node: Term },
}

impl Infer {
    pub fn new() -> Self {
        Self {
            types: Types::new(),
            ty_vars: Vec::new(),
            reference_targets: FxHashSet::default(),
            unmarked: FxHashSet::default(),
            effect_vars: Cells::new(),
            mark_vars: Cells::new(),
            len_vars: Cells::new(),
            identity_vars: Cells::new(),
            level: Level::UNIT,
            ty_levels: Vec::new(),
            identity_levels: Vec::new(),
            made_at: FxHashMap::default(),
            flow_vars: Cells::new(),
            holds_vars: Cells::new(),
            calls_vars: Cells::new(),
            values_vars: Cells::new(),
            effects_refused: Vec::new(),
        }
    }

    pub fn types(&self) -> &Types<InferPhase> {
        &self.types
    }

    /// A term of a written shape. A structural object or enum the program
    /// writes carries `Label::Here`; one instantiated from a
    /// declaration carries `Label::Elsewhere` (`scheme`). No type is `!` by
    /// holding a part of type `!` (types §4). A reference's target is never
    /// a reference (types §4): a variable it targets is marked so, and is
    /// never bound to one.
    ///
    /// # Panics
    /// `kind` is a reference whose target is a reference, through bound
    /// variables or not. Every caller builds a reference over a target it
    /// already holds to types §4.
    pub fn intern(&mut self, kind: TyKind<InferPhase>) -> Term {
        if let TyKind::UserDefined { type_args, .. } = &kind {
            for arg in type_args.iter() {
                if let End::Unbound(var, _) = self.end(*arg) {
                    self.unmarked.insert(var);
                }
            }
        }
        if let TyKind::Ref(_, target) = &kind {
            match self.end(*target) {
                End::Unbound(var, _) => {
                    self.reference_targets.insert(var);
                }
                End::Node { node, .. } => assert!(
                    !matches!(self.types.kind(node), TyKind::Ref(..)),
                    "a reference's target is never a reference (types §4)"
                ),
            }
        }
        self.types.intern(kind)
    }

    pub fn fresh(&mut self, bound: Bound) -> Term {
        let var = TyVarId(u32::try_from(self.ty_vars.len()).expect("Infer::fresh: u32::MAX variables"));
        self.ty_vars.push(VarState::Unbound(bound));
        self.ty_levels.push(self.level);
        self.types.intern(TyKind::Var(var))
    }

    /// The body instantiation now being checked (types §3).
    pub fn level(&self) -> Level {
        self.level
    }

    /// Checks from here at `level`, returning the level it replaces: a
    /// body instantiation entered or left, or a decision fired at the
    /// level it was made at.
    pub fn set_level(&mut self, level: Level) -> Level {
        std::mem::replace(&mut self.level, level)
    }

    /// A fresh identity, made at `at` (types §3): only the compiler makes
    /// one.
    pub fn make_identity(&mut self, at: Level) -> IdentityId {
        let identity = IdentityId::alloc();
        self.made_at.insert(identity, at);
        identity
    }

    /// The level `identity` was made at.
    ///
    /// # Panics
    /// `identity` was neither made here nor met in a written type or an
    /// imported store, each of which records its level.
    fn made_at(&self, identity: IdentityId) -> Level {
        *self
            .made_at
            .get(&identity)
            .unwrap_or_else(|| panic!("{identity:?} is made by `make_identity`, or met written or imported with its level"))
    }

    /// A mark a value an extern fn gives marked starts at: marked, and
    /// raised by each use that takes it unmarked (types §8).
    pub fn fresh_mark(&mut self) -> MarkTerm<InferPhase> {
        let var = self.mark_vars.fresh("Infer::fresh_mark: u32::MAX variables");
        let term = MarkTerm::Var(var);
        self.mark_vars
            .join(&term, &MarkTerm::Known(Mark::Marked), Mark::join)
            .expect("a fresh mark is fixed at nothing");
        term
    }

    pub fn fresh_effect(&mut self) -> EffectVarId {
        self.effect_vars.fresh("Infer::fresh_effect: u32::MAX variables")
    }

    pub fn fresh_len(&mut self) -> LenVarId {
        self.len_vars.fresh("Infer::fresh_len: u32::MAX variables")
    }

    pub fn fresh_identity(&mut self) -> IdentityVarId {
        let var = self.identity_vars.fresh("Infer::fresh_identity: u32::MAX variables");
        self.identity_levels.push(self.level);
        var
    }

    pub fn fresh_flow(&mut self) -> FlowVarId {
        self.flow_vars.fresh("Infer::fresh_flow: u32::MAX variables")
    }

    pub fn fresh_holds(&mut self) -> HoldsVarId {
        self.holds_vars.fresh("Infer::fresh_holds: u32::MAX variables")
    }

    pub fn fresh_calls(&mut self) -> CallsVarId {
        self.calls_vars.fresh("Infer::fresh_calls: u32::MAX variables")
    }

    /// A variable for which function values a function type may be (types
    /// §2), unbound: what a use's own form of a function type asks.
    pub fn fresh_values(&mut self) -> ValuesVarId {
        self.values_vars.fresh("Infer::fresh_values: u32::MAX variables")
    }

    /// A variable for which function values a function type may be, at
    /// `values` (types §1-2): a type the program writes, `Of` its own
    /// values, grows where it is transparent; `Any`, a type a declaration
    /// or the host writes, never grows.
    pub fn fresh_values_of(&mut self, values: FnValues) -> ValuesVarId {
        let var = self.fresh_values();
        let at = ValuesTerm::Var(var);
        match values {
            FnValues::Any => {
                self.values_vars
                    .unify(&at, &ValuesTerm::Known(FnValues::Any))
                    .expect("a fresh variable is unbound");
            }
            FnValues::Of(_) => {
                self.values_vars
                    .join(&at, &ValuesTerm::Known(values), FnValues::join)
                    .expect("a fresh variable is not fixed");
            }
        }
        var
    }

    /// Which function values `values` is known to be so far; `None` while
    /// nothing has decided it.
    pub fn values(&self, values: ValuesVarId) -> Option<FnValues> {
        match self.values_vars.end(&ValuesTerm::Var(values)) {
            CellEnd::Known(known) | CellEnd::Var { value: Some(known), .. } => Some(known),
            CellEnd::Var { value: None, .. } => None,
        }
    }

    /// What `holds` is known to be so far; `None` while nothing has
    /// decided it.
    pub fn holds(&self, holds: HoldsTerm) -> Option<Holds> {
        match self.holds_vars.end(&holds) {
            CellEnd::Known(value) | CellEnd::Var { value: Some(value), .. } => Some(value),
            CellEnd::Var { value: None, .. } => None,
        }
    }

    /// What `calls` is known to be so far; `None` while nothing has
    /// decided it.
    pub fn calls(&self, calls: CallsTerm) -> Option<Calls> {
        match self.calls_vars.end(&calls) {
            CellEnd::Known(value) | CellEnd::Var { value: Some(value), .. } => Some(value),
            CellEnd::Var { value: None, .. } => None,
        }
    }

    /// Joins `value` into the variable `holds` (types §2): what settling a
    /// lambda's captures found they hold, seen by every copy of its type.
    /// A `holds` `eq` fixed does not grow (D94).
    pub fn join_holds(&mut self, holds: HoldsVarId, value: Holds) -> Result<(), Exceeds<Holds>> {
        self.holds_vars
            .join(&HoldsTerm::Var(holds), &HoldsTerm::Known(value), |a, b| a.join(*b))
            .map(drop)
    }

    /// Joins `value` into the variable `calls` (types §2), as `join_holds`.
    pub fn join_calls(&mut self, calls: CallsVarId, value: Calls) -> Result<(), Exceeds<Calls>> {
        self.calls_vars
            .join(&CallsTerm::Var(calls), &CallsTerm::Known(value), |a, b| a.join(*b))
            .map(drop)
    }

    /// effects §1: `into ⊒ from`, one way. A body's effect is the join of
    /// its calls' effects: what a call's effect is, or later rises to,
    /// flows into the body's, and nothing flows back.
    ///
    /// Propagation ends: an effect only rises, and its lattice's height is
    /// 4 + 2c for c the contexts the program and its declarations name
    /// (reissue twice, commuting once, splitting once, each context read
    /// and written once), so each variable is followed at most that many
    /// times plus once per link or flow added.
    pub fn flows_into(&mut self, from: &EffectTerm<InferPhase>, into: &EffectTerm<InferPhase>) -> FlowId {
        let (flow, refused) = self.effect_vars.flows_into(from, into, &Effect::join);
        self.record_effects_refused(refused);
        flow
    }

    /// Every flow of effects refused since the last call.
    pub fn take_effects_refused(&mut self) -> Vec<EffectRefused> {
        std::mem::take(&mut self.effects_refused)
    }

    /// Follows every effect flow from a variable that rose.
    pub(super) fn propagate_effects(&mut self) {
        let refused = self.effect_vars.propagate(&Effect::join);
        self.record_effects_refused(refused);
    }

    fn record_effects_refused(&mut self, refused: Vec<FlowRefused<EffectTerm<InferPhase>>>) {
        self.effects_refused.extend(refused.into_iter().map(|FlowRefused { flow, into, exceeds }| EffectRefused {
            flow,
            into,
            exceeds,
        }));
    }

    /// Whether `a` and `b` are one effect variable, linked.
    pub fn one_effect_variable(&self, a: &EffectTerm<InferPhase>, b: &EffectTerm<InferPhase>) -> bool {
        match (self.effect_vars.end(a), self.effect_vars.end(b)) {
            (CellEnd::Var { var: x, .. }, CellEnd::Var { var: y, .. }) => x == y,
            _ => false,
        }
    }

    /// `ty` with every bound variable at its root followed; a term whose
    /// root is an unbound variable is that variable.
    pub fn shallow(&self, ty: Term) -> Term {
        let mut at = ty;
        while let TyKind::Var(var) = self.types.kind(at)
            && let VarState::Bound(to) = &self.ty_vars[var.0 as usize]
        {
            at = *to;
        }
        at
    }

    /// Whether `term` may still become a transparent structured type: it
    /// is one, or it is an unbound variable whose bound admits one, as an
    /// integer literal's does not.
    pub fn may_be_structured(&self, term: Term) -> bool {
        match self.end(term) {
            End::Unbound(var, _) => match &self.ty_vars[var.0 as usize] {
                VarState::Unbound(bound) => !matches!(bound, Bound::Integer { .. }),
                VarState::Bound(_) => unreachable!("an end is an unbound variable"),
            },
            End::Node { node, .. } => crate::ty::structural::structure(&self.types, node).is_some(),
        }
    }

    fn end(&self, term: Term) -> End {
        let mut at = term;
        let mut holder = None;
        while let TyKind::Var(var) = self.types.kind(at) {
            match &self.ty_vars[var.0 as usize] {
                VarState::Unbound(_) => return End::Unbound(*var, at),
                VarState::Bound(to) => {
                    holder = Some((*var, at));
                    at = *to;
                }
            }
        }
        End::Node { holder, node: at }
    }

    /// types §5: an effect nothing constrains is opaque. Every such
    /// variable still open takes its default; whether any did. Solving
    /// calls this only at a stall.
    pub fn default_effects(&mut self) -> bool {
        let effects = self.effect_vars.default_unbound(&crate::ty::Effect::opaque());
        self.propagate_effects();
        effects
    }

    /// types §5: a function type's flows nothing decides state every flow,
    /// as a declaration's default does (extern-fn §4). Every flow set still
    /// open takes it, but those `claimed`, which a function value's claim
    /// decides (ownership §5); whether any did. Solving calls this only at a stall.
    pub(crate) fn default_unclaimed_flows(&mut self, claimed: &[FlowTerm<InferPhase>]) -> bool {
        let kept: Vec<FlowVarId> = claimed.iter().filter_map(|flows| self.flow_vars.unbound_end(flows)).collect();
        self.flow_vars.default_unbound_but(&crate::ty::Flows::Every, &kept)
    }

    /// types §5: a function value's loans nothing decides are none, and how
    /// often it may be called is many. Every such variable still open takes
    /// it, but those of `lambdas`, which settling a lambda's captures binds
    /// (functions §2-3); whether any did. Solving calls this only at a
    /// stall.
    pub(crate) fn default_unsettled_holds_and_calls(&mut self, lambdas: &[(HoldsVarId, CallsVarId)]) -> bool {
        let kept_holds: Vec<HoldsVarId> = lambdas
            .iter()
            .filter_map(|(holds, _)| self.holds_vars.unbound_end(&HoldsTerm::Var(*holds)))
            .collect();
        let kept_calls: Vec<CallsVarId> = lambdas
            .iter()
            .filter_map(|(_, calls)| self.calls_vars.unbound_end(&CallsTerm::Var(*calls)))
            .collect();
        let holds = self.holds_vars.default_unbound_but(&Holds::Nothing, &kept_holds);
        let calls = self.calls_vars.default_unbound_but(&Calls::Many, &kept_calls);
        holds || calls
    }

    /// The unbound variables, with their bounds: what defaults read at a
    /// stall (types §5).
    pub fn unbound(&self) -> impl Iterator<Item = (TyVarId, &Bound)> {
        self.ty_vars.iter().enumerate().filter_map(|(index, state)| match state {
            VarState::Unbound(bound) => Some((
                TyVarId(u32::try_from(index).expect("Infer::unbound: indices are u32 by fresh")),
                bound,
            )),
            VarState::Bound(_) => None,
        })
    }
}

impl crate::ty::Resolve<InferPhase> for Infer {
    fn types(&self) -> &Types<InferPhase> {
        &self.types
    }

    fn resolve(&self, ty: Term) -> Term {
        self.shallow(ty)
    }
}

impl Default for Infer {
    fn default() -> Self {
        Self::new()
    }
}
