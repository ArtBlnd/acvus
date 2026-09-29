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

/// Each variable's term, and at the end of each chain whether `eq` made
/// its value equal to a known one, which no join may then grow.
pub(super) struct Cells<T: VarTerm> {
    cells: Vec<Option<T>>,
    fixed: Vec<bool>,
}

impl<T: VarTerm> Cells<T> {
    pub(super) fn new() -> Self {
        Self {
            cells: Vec::new(),
            fixed: Vec::new(),
        }
    }

    /// # Panics
    /// `u32::MAX` variables of this kind exist; `exhausted` says which.
    pub(super) fn fresh(&mut self, exhausted: &'static str) -> T::Var {
        let var = T::var_of_index(u32::try_from(self.cells.len()).expect(exhausted));
        self.cells.push(None);
        self.fixed.push(false);
        var
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
        for cell in self.cells.iter_mut().filter(|cell| cell.is_none()) {
            *cell = Some(T::of_value(value.clone()));
            any = true;
        }
        any
    }

    /// Every variable still unbound, but those `kept` names, takes `value`;
    /// whether any did. A kept variable is one some other rule decides.
    pub(super) fn default_unbound_but(&mut self, value: &T::Value, kept: &[T::Var]) -> bool {
        let mut any = false;
        for (index, cell) in self.cells.iter_mut().enumerate() {
            if cell.is_none() && !kept.iter().any(|var| T::index(*var) == index) {
                *cell = Some(T::of_value(value.clone()));
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
                self.cells[T::index(var)] = Some(term.clone());
                if let End::Known(_) = other {
                    self.fixed[T::index(var)] = true;
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
                self.cells[T::index(w)] = Some(T::of_var(v));
                self.fixed[T::index(v)] |= self.fixed[T::index(w)];
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
                self.cells[T::index(var)] = Some(T::of_value(joined));
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
                self.cells[T::index(v)] = joined.map(T::of_value);
                self.cells[T::index(w)] = Some(T::of_var(v));
                self.fixed[T::index(v)] |= self.fixed[T::index(w)];
                Ok(T::of_var(v))
            }
        }
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
