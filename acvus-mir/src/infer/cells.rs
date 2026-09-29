//! The variables whose value is a term of their own kind: an effect, a
//! length, an identity, a mark, a set of flows, what a function value's
//! captures hold, and how often it may be called. Each is unbound, known, or
//! made one with another variable of its kind, so that two variables made
//! one stay one whatever either learns later.

use crate::ty::{
    Calls, CallsTerm, CallsVarId, Effect, EffectTerm, EffectVarId, FlowTerm, FlowVarId, Flows, Holds, HoldsTerm,
    HoldsVarId, IdentityId, IdentityTerm, IdentityVarId, Infer as InferPhase, LenTerm, LenVarId, ReprVarId,
};

use super::{Mark, Unsettled};

/// A term of one variable kind: a known value or a variable.
pub(super) trait VarTerm: Clone {
    type Var: Copy + Eq;
    type Value: Clone + PartialEq;

    fn var_of_index(index: u32) -> Self::Var;
    fn index(var: Self::Var) -> usize;
    fn of_var(var: Self::Var) -> Self;
    fn of_value(value: Self::Value) -> Self;
    fn read(&self) -> Read<'_, Self>;
    fn unsettled(var: Self::Var) -> Unsettled;
}

pub(super) enum Read<'a, T: VarTerm> {
    Known(&'a T::Value),
    Var(T::Var),
}

/// An open `#` as a term: decided, or its variable (types §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MarkTerm {
    Known(Mark),
    Var(ReprVarId),
}

/// Where a term's chain of variables ends.
pub(super) enum End<T: VarTerm> {
    Known(T::Value),
    Var { var: T::Var, value: Option<T::Value> },
}

impl<T: VarTerm> End<T> {
    fn term(&self) -> T {
        match self {
            End::Known(value) => T::of_value(value.clone()),
            End::Var { var, .. } => T::of_var(*var),
        }
    }
}

/// Two known values that differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Unequal;

/// A join that would grow a variable past the value `eq` made it equal
/// (D94): equality is equality, so the variable stays `fixed`, and what
/// asked for `needs` is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exceeds<V> {
    pub fixed: V,
    pub needs: V,
}

/// A one-way constraint `into ⊒ from`, by the number `flows_into` gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FlowId(pub u32);

/// A flow that would grow a fixed `into` past its value (D94).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FlowRefused<T: VarTerm> {
    pub flow: FlowId,
    pub into: T,
    pub exceeds: Exceeds<T::Value>,
}

/// Each variable's term; at the end of each chain, whether `eq` made its
/// value equal to a known one, which no join may then grow; and the terms
/// each variable flows into, one way (`flows_into`). A variable whose
/// value rose since its flows were last followed is `rose`.
pub(super) struct Cells<T: VarTerm> {
    cells: Vec<Option<T>>,
    fixed: Vec<bool>,
    flows: Vec<Vec<(T, FlowId)>>,
    rose: Vec<T::Var>,
    next_flow: u32,
}

impl<T: VarTerm> Cells<T> {
    pub(super) fn new() -> Self {
        Self {
            cells: Vec::new(),
            fixed: Vec::new(),
            flows: Vec::new(),
            rose: Vec::new(),
            next_flow: 0,
        }
    }

    /// # Panics
    /// `u32::MAX` variables of this kind exist; `exhausted` says which.
    pub(super) fn fresh(&mut self, exhausted: &'static str) -> T::Var {
        let var = T::var_of_index(u32::try_from(self.cells.len()).expect(exhausted));
        self.cells.push(None);
        self.fixed.push(false);
        self.flows.push(Vec::new());
        var
    }

    /// Links `from`, the end of a chain, to `to`: the variable `to` ends at
    /// takes `from`'s flows, and follows them again.
    fn link(&mut self, from: T::Var, to: T::Var) {
        self.cells[T::index(from)] = Some(T::of_var(to));
        self.fixed[T::index(to)] |= self.fixed[T::index(from)];
        let moved = std::mem::take(&mut self.flows[T::index(from)]);
        self.flows[T::index(to)].extend(moved);
        self.mark_rose(to);
    }

    /// Queues `var`, the end of a chain, for `propagate`, where it flows
    /// anywhere.
    fn mark_rose(&mut self, var: T::Var) {
        if !self.flows[T::index(var)].is_empty() {
            self.rose.push(var);
        }
    }

    /// Sets the value of `var`, the end of a chain.
    fn set(&mut self, var: T::Var, value: T::Value) {
        self.cells[T::index(var)] = Some(T::of_value(value));
        self.mark_rose(var);
    }

    /// Records `into ⊒ from` (the lattice's subtyping constraint) and
    /// raises `into` to what `from` is so far; `from` rising later raises
    /// it again (`propagate`). Nothing flows back into `from`.
    pub(super) fn flows_into(
        &mut self,
        from: &T,
        into: &T,
        join: &impl Fn(&T::Value, &T::Value) -> T::Value,
    ) -> (FlowId, Vec<FlowRefused<T>>) {
        let flow = FlowId(self.next_flow);
        self.next_flow = self.next_flow.checked_add(1).expect("Cells::flows_into: u32::MAX flows");
        match self.end(from) {
            End::Known(value) => {
                let refused = match self.raise(into, &value, join) {
                    Ok(()) => Vec::new(),
                    Err(exceeds) => vec![FlowRefused { flow, into: into.clone(), exceeds }],
                };
                (flow, refused)
            }
            End::Var { var, .. } => {
                self.flows[T::index(var)].push((into.clone(), flow));
                self.rose.push(var);
                (flow, self.propagate(join))
            }
        }
    }

    /// Follows the flows of every variable that rose, until none rises:
    /// each flow raises its `into` to its source's value. A variable rises
    /// only up its lattice, so each is queued at most once per rise plus
    /// once per link or flow added: with h the lattice's height, O((v · h
    /// + links + flows) · the flows out of each).
    pub(super) fn propagate(&mut self, join: &impl Fn(&T::Value, &T::Value) -> T::Value) -> Vec<FlowRefused<T>> {
        let mut refused = Vec::new();
        while let Some(var) = self.rose.pop() {
            let End::Var { var: root, value: Some(value) } = self.end(&T::of_var(var)) else {
                continue;
            };
            let flows = self.flows[T::index(root)].clone();
            for (into, flow) in flows {
                if let Err(exceeds) = self.raise(&into, &value, join) {
                    refused.push(FlowRefused { flow, into, exceeds });
                }
            }
        }
        refused
    }

    /// Raises `into` to at least `value`: a variable joins it in, a fixed
    /// one only where that does not grow it, and a known value must be it
    /// already.
    fn raise(
        &mut self,
        into: &T,
        value: &T::Value,
        join: &impl Fn(&T::Value, &T::Value) -> T::Value,
    ) -> Result<(), Exceeds<T::Value>> {
        match self.end(into) {
            End::Known(fixed) => {
                let needs = join(&fixed, value);
                match needs == fixed {
                    true => Ok(()),
                    false => Err(Exceeds { fixed, needs }),
                }
            }
            End::Var { var, value: had } => {
                let joined = match &had {
                    Some(had) => join(had, value),
                    None => value.clone(),
                };
                self.keep_fixed(var, had.as_ref(), &joined)?;
                if had.as_ref() != Some(&joined) {
                    self.set(var, joined);
                }
                Ok(())
            }
        }
    }

    pub(super) fn end(&self, term: &T) -> End<T> {
        let mut var = match term.read() {
            Read::Known(value) => return End::Known(value.clone()),
            Read::Var(var) => var,
        };
        loop {
            match &self.cells[T::index(var)] {
                None => return End::Var { var, value: None },
                Some(to) => match to.read() {
                    Read::Known(value) => {
                        return End::Var {
                            var,
                            value: Some(value.clone()),
                        };
                    }
                    Read::Var(next) => var = next,
                },
            }
        }
    }

    /// Every variable still unbound takes `value`; whether any did. A
    /// variable made one with another is bound through it and is left.
    pub(super) fn default_unbound(&mut self, value: &T::Value) -> bool {
        let mut any = false;
        for index in 0..self.cells.len() {
            if self.cells[index].is_none() {
                self.set(T::var_of_index(u32::try_from(index).expect("indices are u32 by fresh")), value.clone());
                any = true;
            }
        }
        any
    }

    /// Every variable still unbound, but those `kept` names, takes `value`;
    /// whether any did. A kept variable is one some other rule decides.
    pub(super) fn default_unbound_but(&mut self, value: &T::Value, kept: &[T::Var]) -> bool {
        let mut any = false;
        for index in 0..self.cells.len() {
            if self.cells[index].is_none() && !kept.iter().any(|var| T::index(*var) == index) {
                self.set(T::var_of_index(u32::try_from(index).expect("indices are u32 by fresh")), value.clone());
                any = true;
            }
        }
        any
    }

    /// The unbound variable `term`'s chain ends at, `None` where it ends at
    /// a known value.
    pub(super) fn unbound_end(&self, term: &T) -> Option<T::Var> {
        match self.end(term) {
            End::Var { var, value: None } => Some(var),
            End::Known(_) | End::Var { value: Some(_), .. } => None,
        }
    }

    /// Makes `a` and `b` one (types §4): an unbound variable is bound, two
    /// variables are linked, two known values must be equal. A variable
    /// made one with a known value is fixed at it (D94).
    pub(super) fn unify(&mut self, a: &T, b: &T) -> Result<T, Unequal> {
        match (self.end(a), self.end(b)) {
            (End::Known(x), End::Known(y)) => (x == y).then(|| T::of_value(x)).ok_or(Unequal),
            (End::Var { var, value: None }, other) | (other, End::Var { var, value: None }) => {
                if let End::Var { var: same, .. } = other
                    && same == var
                {
                    return Ok(T::of_var(var));
                }
                let term = other.term();
                match &other {
                    End::Known(value) => {
                        self.set(var, value.clone());
                        self.fixed[T::index(var)] = true;
                    }
                    End::Var { var: to, .. } => self.link(var, *to),
                }
                Ok(term)
            }
            (End::Var { var, value: Some(x) }, End::Known(y)) | (End::Known(y), End::Var { var, value: Some(x) }) => {
                if x != y {
                    return Err(Unequal);
                }
                self.fixed[T::index(var)] = true;
                Ok(T::of_var(var))
            }
            (End::Var { var: v, value: Some(x) }, End::Var { var: w, value: Some(y) }) => {
                if v == w {
                    return Ok(T::of_var(v));
                }
                if x != y {
                    return Err(Unequal);
                }
                self.link(w, v);
                Ok(T::of_var(v))
            }
        }
    }

    /// The join of `a` and `b` (types §2): a variable's value becomes the
    /// join of what was joined into it, and two variables are linked. A
    /// fixed variable's value does not grow (D94): a join that would grow
    /// it is `Exceeds`, and changes nothing.
    pub(super) fn join(
        &mut self,
        a: &T,
        b: &T,
        join: impl Fn(&T::Value, &T::Value) -> T::Value,
    ) -> Result<T, Exceeds<T::Value>> {
        match (self.end(a), self.end(b)) {
            (End::Known(x), End::Known(y)) => Ok(T::of_value(join(&x, &y))),
            (End::Var { var, value }, End::Known(y)) | (End::Known(y), End::Var { var, value }) => {
                let joined = match &value {
                    Some(x) => join(x, &y),
                    None => y,
                };
                self.keep_fixed(var, value.as_ref(), &joined)?;
                self.set(var, joined);
                Ok(T::of_var(var))
            }
            (End::Var { var: v, value: x }, End::Var { var: w, value: y }) => {
                if v == w {
                    return Ok(T::of_var(v));
                }
                let joined = match (&x, &y) {
                    (Some(x), Some(y)) => Some(join(x, y)),
                    (Some(one), None) | (None, Some(one)) => Some(one.clone()),
                    (None, None) => None,
                };
                if let Some(joined) = &joined {
                    self.keep_fixed(v, x.as_ref(), joined)?;
                    self.keep_fixed(w, y.as_ref(), joined)?;
                }
                if let Some(joined) = joined {
                    self.set(v, joined);
                }
                self.link(w, v);
                Ok(T::of_var(v))
            }
        }
    }

    /// How many variables of this kind were made so far.
    pub(super) fn count(&self) -> u32 {
        u32::try_from(self.cells.len()).expect("indices are u32 by fresh")
    }

    /// A fresh variable in the state of each of `vars`, each the end of its
    /// chain: its value so far and whether `eq` fixed it. What flows into
    /// or out of them is copied by `flows_touching`.
    pub(super) fn duplicate(&mut self, vars: &[T::Var], exhausted: &'static str) -> Vec<T::Var> {
        vars.iter()
            .map(|var| {
                let copy = self.fresh(exhausted);
                self.cells[T::index(copy)] = self.cells[T::index(*var)].clone();
                self.fixed[T::index(copy)] = self.fixed[T::index(*var)];
                copy
            })
            .collect()
    }

    /// Every flow out of or into a variable `touches` admits, as `(from,
    /// into, id)`, `from` the end its flows are kept at.
    pub(super) fn flows_touching(&self, touches: &dyn Fn(T::Var) -> bool) -> Vec<(T::Var, T, FlowId)> {
        let mut out = Vec::new();
        for (index, flows) in self.flows.iter().enumerate() {
            let from = T::var_of_index(u32::try_from(index).expect("indices are u32 by fresh"));
            for (into, flow) in flows {
                let into_touched = match self.end(into) {
                    End::Var { var, .. } => touches(var),
                    End::Known(_) => false,
                };
                if touches(from) || into_touched {
                    out.push((from, into.clone(), *flow));
                }
            }
        }
        out
    }

    /// Refuses `joined` where `var`, the end of its chain, is fixed at a
    /// `value` it differs from.
    fn keep_fixed(&self, var: T::Var, value: Option<&T::Value>, joined: &T::Value) -> Result<(), Exceeds<T::Value>> {
        match value {
            Some(fixed) if self.fixed[T::index(var)] && fixed != joined => Err(Exceeds {
                fixed: fixed.clone(),
                needs: joined.clone(),
            }),
            _ => Ok(()),
        }
    }
}

impl VarTerm for EffectTerm<InferPhase> {
    type Var = EffectVarId;
    type Value = Effect;

    fn var_of_index(index: u32) -> EffectVarId {
        EffectVarId(index)
    }
    fn index(var: EffectVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: EffectVarId) -> Unsettled {
        Unsettled::Effect(var)
    }
    fn of_var(var: EffectVarId) -> Self {
        EffectTerm::Var(var)
    }
    fn of_value(value: Effect) -> Self {
        EffectTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            EffectTerm::Known(value) => Read::Known(value),
            EffectTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for FlowTerm<InferPhase> {
    type Var = FlowVarId;
    type Value = Flows;

    fn var_of_index(index: u32) -> FlowVarId {
        FlowVarId(index)
    }
    fn index(var: FlowVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: FlowVarId) -> Unsettled {
        Unsettled::Flow(var)
    }
    fn of_var(var: FlowVarId) -> Self {
        FlowTerm::Var(var)
    }
    fn of_value(value: Flows) -> Self {
        FlowTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            FlowTerm::Known(value) => Read::Known(value),
            FlowTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for LenTerm<InferPhase> {
    type Var = LenVarId;
    type Value = usize;

    fn var_of_index(index: u32) -> LenVarId {
        LenVarId(index)
    }
    fn index(var: LenVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: LenVarId) -> Unsettled {
        Unsettled::Len(var)
    }
    fn of_var(var: LenVarId) -> Self {
        LenTerm::Var(var)
    }
    fn of_value(value: usize) -> Self {
        LenTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            LenTerm::Known(value) => Read::Known(value),
            LenTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for IdentityTerm<InferPhase> {
    type Var = IdentityVarId;
    type Value = IdentityId;

    fn var_of_index(index: u32) -> IdentityVarId {
        IdentityVarId(index)
    }
    fn index(var: IdentityVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: IdentityVarId) -> Unsettled {
        Unsettled::Identity(var)
    }
    fn of_var(var: IdentityVarId) -> Self {
        IdentityTerm::Var(var)
    }
    fn of_value(value: IdentityId) -> Self {
        IdentityTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            IdentityTerm::Known(value) => Read::Known(value),
            IdentityTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for MarkTerm {
    type Var = ReprVarId;
    type Value = Mark;

    fn var_of_index(index: u32) -> ReprVarId {
        ReprVarId(index)
    }
    fn index(var: ReprVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: ReprVarId) -> Unsettled {
        Unsettled::Mark(var)
    }
    fn of_var(var: ReprVarId) -> Self {
        MarkTerm::Var(var)
    }
    fn of_value(value: Mark) -> Self {
        MarkTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            MarkTerm::Known(value) => Read::Known(value),
            MarkTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for HoldsTerm {
    type Var = HoldsVarId;
    type Value = Holds;

    fn var_of_index(index: u32) -> HoldsVarId {
        HoldsVarId(index)
    }
    fn index(var: HoldsVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: HoldsVarId) -> Unsettled {
        Unsettled::Holds(var)
    }
    fn of_var(var: HoldsVarId) -> Self {
        HoldsTerm::Var(var)
    }
    fn of_value(value: Holds) -> Self {
        HoldsTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            HoldsTerm::Known(value) => Read::Known(value),
            HoldsTerm::Var(var) => Read::Var(*var),
        }
    }
}

impl VarTerm for CallsTerm {
    type Var = CallsVarId;
    type Value = Calls;

    fn var_of_index(index: u32) -> CallsVarId {
        CallsVarId(index)
    }
    fn index(var: CallsVarId) -> usize {
        var.0 as usize
    }
    fn unsettled(var: CallsVarId) -> Unsettled {
        Unsettled::Calls(var)
    }
    fn of_var(var: CallsVarId) -> Self {
        CallsTerm::Var(var)
    }
    fn of_value(value: Calls) -> Self {
        CallsTerm::Known(value)
    }
    fn read(&self) -> Read<'_, Self> {
        match self {
            CallsTerm::Known(value) => Read::Known(value),
            CallsTerm::Var(var) => Read::Var(*var),
        }
    }
}
