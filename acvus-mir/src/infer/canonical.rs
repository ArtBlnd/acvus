//! A store's terms copied into a table another store's copies share: each
//! bound variable followed to its value, and each variable of every kind
//! renumbered by the order the copy first reaches it, with its state beside
//! it. Two sets of terms are one but for the names of their variables where
//! each side, copied in the same order, gives the same terms and the same
//! states (typeck::Interface::same).

use rustc_hash::FxHashMap;

use super::cells::{Cells, End as CellEnd, ValuesTerm, VarTerm};
use super::{Bound, Infer, Term};
use crate::ty::{
    CallsTerm, EffectTerm, EffectVarId, FlowTerm, HoldsTerm, IdentityTerm, IdentityVarId, Infer as InferPhase,
    LenTerm, LenVarId, TyKind, TyVarId, Types, ValuesVarId,
};

/// The table both sides' copies are interned in, so that two copies are
/// equal exactly where their shapes are.
pub struct CanonicalTable {
    types: Types<InferPhase>,
}

impl CanonicalTable {
    pub fn new() -> CanonicalTable {
        CanonicalTable { types: Types::new() }
    }
}

impl Default for CanonicalTable {
    fn default() -> Self {
        Self::new()
    }
}

/// One store's terms being copied into a `CanonicalTable`.
pub struct Canonical<'c> {
    from: &'c Infer,
    into: &'c mut Types<InferPhase>,
    memo: FxHashMap<Term, Term>,
    types: FxHashMap<TyVarId, TyVarId>,
    effects: Renumbered<EffectTerm<InferPhase>>,
    lengths: Renumbered<LenTerm<InferPhase>>,
    identities: Renumbered<IdentityTerm<InferPhase>>,
    flows: Renumbered<FlowTerm<InferPhase>>,
    holds: Renumbered<HoldsTerm>,
    calls: Renumbered<CallsTerm>,
    values: Renumbered<ValuesTerm>,
    states: VarStates,
}

/// The state of each variable a copy reached, by its number there: a type
/// variable's bound and whether it is a reference's target (types §4);
/// any other kind's value so far and whether `eq` fixed it; and the effect
/// flows into or out of a reached effect variable (effects §1).
#[derive(Debug, PartialEq)]
pub struct VarStates {
    types: Vec<TyVarState>,
    effects: Vec<CellState<crate::ty::Effect>>,
    lengths: Vec<CellState<usize>>,
    identities: Vec<CellState<crate::ty::IdentityId>>,
    flows: Vec<CellState<crate::ty::Flows>>,
    holds: Vec<CellState<crate::ty::Holds>>,
    calls: Vec<CellState<crate::ty::Calls>>,
    values: Vec<CellState<crate::ty::FnValues>>,
    effect_flows: Vec<EffectFlow>,
}

#[derive(Debug, PartialEq)]
struct TyVarState {
    bound: Bound,
    reference_target: bool,
}

/// `into ⊒ from` (effects §1).
#[derive(Debug, PartialEq)]
struct EffectFlow {
    from: EffectVarId,
    into: EffectTerm<InferPhase>,
}

#[derive(Debug, PartialEq)]
struct CellState<V> {
    value: Option<V>,
    fixed: bool,
}

/// One kind's variables, each by the number the copy gave it.
struct Renumbered<T: VarTerm> {
    numbers: FxHashMap<T::Var, T::Var>,
    order: Vec<T::Var>,
}

impl<T: VarTerm> Renumbered<T>
where
    T::Var: std::hash::Hash,
{
    fn new() -> Self {
        Renumbered { numbers: FxHashMap::default(), order: Vec::new() }
    }

    /// `term` copied: a known value as it is, a variable its chain's end
    /// renumbered, its state pushed onto `states` where it is reached for
    /// the first time.
    fn term(&mut self, cells: &Cells<T>, term: &T, states: &mut Vec<CellState<T::Value>>) -> T {
        match cells.end(term) {
            CellEnd::Known(value) => T::of_value(value),
            CellEnd::Var { var, value } => T::of_var(self.var(cells, var, value, states)),
        }
    }

    fn var(&mut self, cells: &Cells<T>, var: T::Var, value: Option<T::Value>, states: &mut Vec<CellState<T::Value>>) -> T::Var {
        if let Some(number) = self.numbers.get(&var) {
            return *number;
        }
        let number = T::var_of_index(u32::try_from(self.order.len()).expect("Canonical: u32::MAX variables"));
        self.numbers.insert(var, number);
        self.order.push(var);
        states.push(CellState { value, fixed: cells.is_fixed(var) });
        number
    }
}

impl<'c> Canonical<'c> {
    pub fn new(from: &'c Infer, table: &'c mut CanonicalTable) -> Canonical<'c> {
        Canonical {
            from,
            into: &mut table.types,
            memo: FxHashMap::default(),
            types: FxHashMap::default(),
            effects: Renumbered::new(),
            lengths: Renumbered::new(),
            identities: Renumbered::new(),
            flows: Renumbered::new(),
            holds: Renumbered::new(),
            calls: Renumbered::new(),
            values: Renumbered::new(),
            states: VarStates {
                types: Vec::new(),
                effects: Vec::new(),
                lengths: Vec::new(),
                identities: Vec::new(),
                flows: Vec::new(),
                holds: Vec::new(),
                calls: Vec::new(),
                values: Vec::new(),
                effect_flows: Vec::new(),
            },
        }
    }

    /// `term` copied, its variables renumbered in the order this walk
    /// reaches them: a node's types before its other kinds' variables.
    pub fn term(&mut self, term: Term) -> Term {
        acvus_utils::grow(|| self.term_level(term))
    }

    fn term_level(&mut self, term: Term) -> Term {
        let term = self.from.shallow(term);
        if let Some(done) = self.memo.get(&term) {
            return *done;
        }
        let from = self.from;
        let kind = match from.types.kind(term) {
            TyKind::Var(var) => TyKind::Var(self.ty_var(*var)),
            kind => {
                let children = kind.with_children(&mut |child| self.term(child));
                self.slots(children)
            }
        };
        let copied = self.into.intern(kind);
        self.memo.insert(term, copied);
        copied
    }

    /// `kind`'s variables of every kind but types copied.
    fn slots(&mut self, kind: TyKind<InferPhase>) -> TyKind<InferPhase> {
        let from = self.from;
        match kind {
            TyKind::Array(element, len, label) => {
                TyKind::Array(element, self.lengths.term(&from.len_vars, &len, &mut self.states.lengths), label)
            }
            TyKind::Fn { params, ret, effect, flows, calls, holds, values } => TyKind::Fn {
                params,
                ret,
                effect: self.effect(&effect),
                flows: self.flows.term(&from.flow_vars, &flows, &mut self.states.flows),
                calls: self.calls.term(&from.calls_vars, &calls, &mut self.states.calls),
                holds: self.holds.term(&from.holds_vars, &holds, &mut self.states.holds),
                values: self.values_var(values),
            },
            TyKind::UserDefined { id, type_args, effect, identity, marked } => TyKind::UserDefined {
                id,
                type_args,
                effect: effect.as_ref().map(|effect| self.effect(effect)),
                identity: identity
                    .as_ref()
                    .map(|identity| self.identities.term(&from.identity_vars, identity, &mut self.states.identities)),
                marked,
            },
            other => other,
        }
    }

    pub fn effect(&mut self, effect: &EffectTerm<InferPhase>) -> EffectTerm<InferPhase> {
        self.effects.term(&self.from.effect_vars, effect, &mut self.states.effects)
    }

    pub fn ty_var(&mut self, var: TyVarId) -> TyVarId {
        if let Some(number) = self.types.get(&var) {
            return *number;
        }
        let number = TyVarId(u32::try_from(self.states.types.len()).expect("Canonical: u32::MAX type variables"));
        let super::VarState::Unbound(bound) = &self.from.ty_vars[var.0 as usize] else {
            panic!("Canonical::ty_var: a variable reached through `shallow` is unbound")
        };
        self.types.insert(var, number);
        self.states.types.push(TyVarState { bound: bound.clone(), reference_target: self.from.reference_targets.contains(&var) });
        number
    }

    pub fn effect_var(&mut self, var: EffectVarId) -> EffectVarId {
        cell_var(&mut self.effects, &self.from.effect_vars, var, &mut self.states.effects)
    }

    pub fn length_var(&mut self, var: LenVarId) -> LenVarId {
        cell_var(&mut self.lengths, &self.from.len_vars, var, &mut self.states.lengths)
    }

    pub fn identity_var(&mut self, var: IdentityVarId) -> IdentityVarId {
        cell_var(&mut self.identities, &self.from.identity_vars, var, &mut self.states.identities)
    }

    pub fn values_var(&mut self, var: ValuesVarId) -> ValuesVarId {
        cell_var(&mut self.values, &self.from.values_vars, var, &mut self.states.values)
    }

    /// The states of every variable reached, once each effect flow into or
    /// out of a reached effect variable is copied too, reaching the
    /// variables at its other end.
    pub fn finish(mut self) -> VarStates {
        let mut at = 0;
        while at < self.effects.order.len() {
            let var = self.effects.order[at];
            for (from, into, _) in self.from.effect_vars.flows_touching(&|touched| touched == var) {
                let from = self.effect_var(from);
                let into = self.effect(&into);
                self.states.effect_flows.push(EffectFlow { from, into });
            }
            at += 1;
        }
        self.states
    }
}

/// A variable of a slot that holds one, its chain's end renumbered.
fn cell_var<T: VarTerm>(renumbered: &mut Renumbered<T>, cells: &Cells<T>, var: T::Var, states: &mut Vec<CellState<T::Value>>) -> T::Var
where
    T::Var: std::hash::Hash,
{
    let CellEnd::Var { var, value } = cells.end(&T::of_var(var)) else {
        panic!("Canonical: a variable's chain ends at a variable")
    };
    renumbered.var(cells, var, value, states)
}
