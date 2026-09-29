//! A term's variables, read and copied: what generalizing a binding's type
//! and instantiating it at a use need (functions §2). Every kind a type
//! states is generalized: types, lengths, identities, `#` marks, effects
//! with the flows among them, and which function values a function type
//! may be. What a function value's captures
//! hold, how often it may be called and its flows are decided where it is
//! written, and are not copied.

use rustc_hash::{FxHashMap, FxHashSet};

use super::cells::{Cells, End as CellEnd, MarkTerm, ValuesTerm, VarTerm};
use super::{FlowId, Infer, Term, VarState};
use crate::ty::{
    EffectArg, EffectTerm, EffectVarId, IdentityTerm, IdentityVarId, Infer as InferPhase, Label, LenTerm, LenVarId, ObjectTy,
    ReprVarId, TyKind, TyVarId, TypeArg, ValuesVarId,
};

/// How many variables of each kind were made so far: one made later has an
/// index at least its kind's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VarMarks {
    pub types: u32,
    pub effects: u32,
    pub lengths: u32,
    pub identities: u32,
    pub marks: u32,
    pub values: u32,
}

/// The variables of every kind some terms reach: each type variable on the
/// way, bound or not, and each other kind's variable at the end of its
/// chain.
#[derive(Debug, Clone, Default)]
pub struct Reached {
    pub types: FxHashSet<TyVarId>,
    pub effects: FxHashSet<EffectVarId>,
    pub lengths: FxHashSet<LenVarId>,
    pub identities: FxHashSet<IdentityVarId>,
    pub marks: FxHashSet<ReprVarId>,
    pub values: FxHashSet<ValuesVarId>,
}

/// Generalized variables of every kind, each by position.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VarsOf {
    pub types: Box<[TyVarId]>,
    pub effects: Box<[EffectVarId]>,
    pub lengths: Box<[LenVarId]>,
    pub identities: Box<[IdentityVarId]>,
    pub marks: Box<[ReprVarId]>,
    pub values: Box<[ValuesVarId]>,
}

/// Each generalized variable's term at one instance.
#[derive(Debug, Clone, Default)]
pub struct AtInstance {
    pub types: FxHashMap<TyVarId, Term>,
    pub effects: FxHashMap<EffectVarId, EffectVarId>,
    pub lengths: FxHashMap<LenVarId, LenVarId>,
    pub identities: FxHashMap<IdentityVarId, IdentityVarId>,
    pub marks: FxHashMap<ReprVarId, ReprVarId>,
    pub values: FxHashMap<ValuesVarId, ValuesVarId>,
}

/// An effect flow copied at an instance: the flow it copies, and the copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopiedFlow {
    pub of: FlowId,
    pub copy: FlowId,
}

impl Infer {
    pub fn var_marks(&self) -> VarMarks {
        VarMarks {
            types: u32::try_from(self.ty_vars.len()).expect("Infer::var_marks: u32::MAX variables"),
            effects: self.effect_vars.count(),
            lengths: self.len_vars.count(),
            identities: self.identity_vars.count(),
            marks: self.repr_vars.count(),
            values: self.values_vars.count(),
        }
    }

    /// The term that is the variable `var`.
    pub fn var_term(&mut self, var: TyVarId) -> Term {
        self.types.intern(TyKind::Var(var))
    }

    /// Whether `var` is unbound.
    pub fn is_unbound(&self, var: TyVarId) -> bool {
        matches!(self.ty_vars[var.0 as usize], VarState::Unbound(_))
    }

    /// Whether a length, identity or mark variable is unbound: the end of
    /// its chain with no value.
    pub fn is_open_length(&self, var: LenVarId) -> bool {
        self.len_vars.unbound_end(&LenTerm::Var(var)) == Some(var)
    }

    pub fn is_open_identity(&self, var: IdentityVarId) -> bool {
        self.identity_vars.unbound_end(&IdentityTerm::Var(var)) == Some(var)
    }

    pub fn is_open_mark(&self, var: ReprVarId) -> bool {
        self.repr_vars.unbound_end(&MarkTerm::Var(var)) == Some(var)
    }

    /// The variables of every kind `types` and `effects` reach.
    pub fn reached(&self, types: impl IntoIterator<Item = Term>, effects: impl IntoIterator<Item = EffectTerm<InferPhase>>) -> Reached {
        let mut reached = Reached::default();
        for effect in effects {
            reach_cell(&self.effect_vars, &effect, &mut reached.effects);
        }
        let mut seen = FxHashSet::default();
        let mut stack: Vec<Term> = types.into_iter().collect();
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            let kind = self.types.kind(at);
            match kind {
                TyKind::Var(var) => {
                    reached.types.insert(*var);
                    if let VarState::Bound(to) = &self.ty_vars[var.0 as usize] {
                        stack.push(*to);
                    }
                }
                TyKind::Array(_, len, _) => reach_cell(&self.len_vars, len, &mut reached.lengths),
                TyKind::Fn { effect, values, .. } => {
                    reach_cell(&self.effect_vars, effect, &mut reached.effects);
                    reach_cell(&self.values_vars, &ValuesTerm::Var(*values), &mut reached.values);
                }
                TyKind::UserDefined { type_args, effect_args, identity_args, .. } => {
                    for arg in effect_args.iter() {
                        let (EffectArg::Uniform(effect) | EffectArg::Specialized(effect)) = arg;
                        reach_cell(&self.effect_vars, effect, &mut reached.effects);
                    }
                    for identity in identity_args.iter() {
                        reach_cell(&self.identity_vars, identity, &mut reached.identities);
                    }
                    for arg in type_args.iter() {
                        if let TypeArg::Open(mark, _) = arg {
                            reach_cell(&self.repr_vars, &MarkTerm::Var(*mark), &mut reached.marks);
                        }
                    }
                }
                TyKind::Ref(_, TypeArg::Open(mark, _)) => reach_cell(&self.repr_vars, &MarkTerm::Var(*mark), &mut reached.marks),
                _ => {}
            }
            kind.for_each_child(|child| stack.push(child));
        }
        reached
    }

    /// Fresh variables of every kind for `vars` at a new instance: a type
    /// variable of the same bound (types §5), an open length, identity or
    /// mark, an effect in the same state, with every flow out of or into
    /// one of the effects copied onto its copy, so that an instance's
    /// effect rises only by its own flows (effects §1), and the function
    /// values a function type may be as they are, which a use then grows as
    /// its own and not the binding's (types §1-2).
    pub fn instance_at(&mut self, vars: &VarsOf) -> (AtInstance, Vec<CopiedFlow>) {
        let mut at = AtInstance::default();
        for var in vars.types.iter() {
            let VarState::Unbound(bound) = self.ty_vars[var.0 as usize].clone() else {
                panic!("instance_at: a generalized type variable is unbound")
            };
            let fresh = self.fresh(bound);
            at.types.insert(*var, fresh);
        }
        for var in vars.lengths.iter() {
            let fresh = self.fresh_len();
            at.lengths.insert(*var, fresh);
        }
        for var in vars.identities.iter() {
            let fresh = self.fresh_identity();
            at.identities.insert(*var, fresh);
        }
        for var in vars.marks.iter() {
            let fresh = self.fresh_repr();
            at.marks.insert(*var, fresh);
        }
        let copies = self.values_vars.duplicate(&vars.values, "Infer::instance_at: u32::MAX values variables");
        at.values.extend(vars.values.iter().copied().zip(copies));
        let copies = self.effect_vars.duplicate(&vars.effects, "Infer::instance_at: u32::MAX effect variables");
        at.effects.extend(vars.effects.iter().copied().zip(copies));
        let generalized: FxHashSet<EffectVarId> = vars.effects.iter().copied().collect();
        let flows = self.effect_vars.flows_touching(&|var| generalized.contains(&var));
        let mut copied = Vec::with_capacity(flows.len());
        for (from, into, of) in flows {
            let from = EffectTerm::Var(at.effects.get(&from).copied().unwrap_or(from));
            let into = self.copy_effect(&into, &at);
            let copy = self.flows_into(&from, &into);
            copied.push(CopiedFlow { of, copy });
        }
        (at, copied)
    }

    /// `effect` at an instance: a generalized variable its copy.
    pub fn copy_effect(&self, effect: &EffectTerm<InferPhase>, at: &AtInstance) -> EffectTerm<InferPhase> {
        match self.effect_vars.end(effect) {
            CellEnd::Var { var, .. } if at.effects.contains_key(&var) => EffectTerm::Var(at.effects[&var]),
            _ => effect.clone(),
        }
    }

    fn copy_len(&self, len: &LenTerm<InferPhase>, at: &AtInstance) -> LenTerm<InferPhase> {
        match self.len_vars.end(len) {
            CellEnd::Var { var, .. } if at.lengths.contains_key(&var) => LenTerm::Var(at.lengths[&var]),
            _ => *len,
        }
    }

    fn copy_identity(&self, identity: &IdentityTerm<InferPhase>, at: &AtInstance) -> IdentityTerm<InferPhase> {
        match self.identity_vars.end(identity) {
            CellEnd::Var { var, .. } if at.identities.contains_key(&var) => IdentityTerm::Var(at.identities[&var]),
            _ => *identity,
        }
    }

    /// `values` at an instance: a generalized variable its copy.
    pub fn copy_values(&self, values: ValuesVarId, at: &AtInstance) -> ValuesVarId {
        match self.values_vars.end(&ValuesTerm::Var(values)) {
            CellEnd::Var { var, .. } if at.values.contains_key(&var) => at.values[&var],
            _ => values,
        }
    }

    fn copy_mark(&self, mark: ReprVarId, at: &AtInstance) -> ReprVarId {
        match self.repr_vars.end(&MarkTerm::Var(mark)) {
            CellEnd::Var { var, .. } if at.marks.contains_key(&var) => at.marks[&var],
            _ => mark,
        }
    }

    /// `term` at an instance: each generalized variable of every kind its
    /// copy there, bound variables followed. A node holding none of them is
    /// `term` itself; every node rebuilt around one is written elsewhere,
    /// since a use does not grow what the value wrote (types §1).
    pub fn copy_at(&mut self, term: Term, at: &AtInstance, memo: &mut FxHashMap<Term, Term>) -> Term {
        acvus_utils::grow(|| self.copy_level(term, at, memo))
    }

    fn copy_level(&mut self, term: Term, at: &AtInstance, memo: &mut FxHashMap<Term, Term>) -> Term {
        if let Some(done) = memo.get(&term) {
            return *done;
        }
        let kind = self.types.kind(term).clone();
        let copied = match kind {
            TyKind::Var(var) => match (at.types.get(&var), &self.ty_vars[var.0 as usize]) {
                (Some(fresh), _) => *fresh,
                (None, VarState::Bound(to)) => {
                    let to = *to;
                    let copied = self.copy_at(to, at, memo);
                    if copied == to { term } else { copied }
                }
                (None, VarState::Unbound(_)) => term,
            },
            kind => {
                let rebuilt = self.copy_kind(&kind, at, memo);
                match rebuilt == kind {
                    true => term,
                    false => self.intern(relabelled(rebuilt)),
                }
            }
        };
        memo.insert(term, copied);
        copied
    }

    fn copy_kind(&mut self, kind: &TyKind<InferPhase>, at: &AtInstance, memo: &mut FxHashMap<Term, Term>) -> TyKind<InferPhase> {
        let copy_arg = |infer: &mut Infer, arg: &TypeArg<InferPhase>, memo: &mut FxHashMap<Term, Term>| match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(infer.copy_at(*ty, at, memo)),
            TypeArg::Specialized(ty) => TypeArg::Specialized(infer.copy_at(*ty, at, memo)),
            TypeArg::Open(mark, ty) => TypeArg::Open(infer.copy_mark(*mark, at), infer.copy_at(*ty, at, memo)),
        };
        match kind {
            TyKind::Handle(inner) => TyKind::Handle(self.copy_at(*inner, at, memo)),
            TyKind::Array(element, len, label) => TyKind::Array(self.copy_at(*element, at, memo), self.copy_len(len, at), *label),
            TyKind::Slice(element) => TyKind::Slice(self.copy_at(*element, at, memo)),
            TyKind::Option(payload, label) => TyKind::Option(self.copy_at(*payload, at, memo), *label),
            TyKind::Result(ok, err, label) => {
                let ok = self.copy_at(*ok, at, memo);
                TyKind::Result(ok, self.copy_at(*err, at, memo), *label)
            }
            TyKind::Tuple(items, label) => TyKind::Tuple(items.iter().map(|item| self.copy_at(*item, at, memo)).collect(), *label),
            TyKind::Object(ObjectTy::Structural(fields, label)) => TyKind::Object(ObjectTy::Structural(
                fields.iter().map(|(name, field)| (*name, self.copy_at(*field, at, memo))).collect(),
                *label,
            )),
            TyKind::Object(ObjectTy::Declared(id, fields)) => TyKind::Object(ObjectTy::Declared(
                *id,
                fields.iter().map(|(name, field)| (*name, self.copy_at(*field, at, memo))).collect(),
            )),
            TyKind::Enum { name, variants, label } => TyKind::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| (*tag, payload.map(|payload| self.copy_at(payload, at, memo)))).collect(),
                label: *label,
            },
            TyKind::Fn { params, ret, effect, flows, calls, holds, values } => TyKind::Fn {
                params: params.iter().map(|param| self.copy_at(*param, at, memo)).collect(),
                ret: self.copy_at(*ret, at, memo),
                effect: self.copy_effect(effect, at),
                flows: flows.clone(),
                calls: *calls,
                holds: *holds,
                values: self.copy_values(*values, at),
            },
            TyKind::UserDefined { id, type_args, effect_args, identity_args } => TyKind::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|arg| copy_arg(self, arg, memo)).collect(),
                effect_args: effect_args
                    .iter()
                    .map(|arg| match arg {
                        EffectArg::Uniform(effect) => EffectArg::Uniform(self.copy_effect(effect, at)),
                        EffectArg::Specialized(effect) => EffectArg::Specialized(self.copy_effect(effect, at)),
                    })
                    .collect(),
                identity_args: identity_args.iter().map(|identity| self.copy_identity(identity, at)).collect(),
            },
            TyKind::Ref(mutability, target) => TyKind::Ref(*mutability, copy_arg(self, target, memo)),
            leaf => leaf.clone(),
        }
    }
}

/// A rebuilt node's transparent structured kinds written elsewhere.
fn relabelled(kind: TyKind<InferPhase>) -> TyKind<InferPhase> {
    match kind {
        TyKind::Array(element, len, _) => TyKind::Array(element, len, Label::Elsewhere),
        TyKind::Option(payload, _) => TyKind::Option(payload, Label::Elsewhere),
        TyKind::Result(ok, err, _) => TyKind::Result(ok, err, Label::Elsewhere),
        TyKind::Tuple(items, _) => TyKind::Tuple(items, Label::Elsewhere),
        TyKind::Object(ObjectTy::Structural(fields, _)) => TyKind::Object(ObjectTy::Structural(fields, Label::Elsewhere)),
        TyKind::Enum { name, variants, .. } => TyKind::Enum { name, variants, label: Label::Elsewhere },
        other => other,
    }
}

/// The end of `term`'s chain, where it is a variable, into `into`.
fn reach_cell<T: VarTerm>(cells: &Cells<T>, term: &T, into: &mut FxHashSet<T::Var>)
where
    T::Var: std::hash::Hash,
{
    if let CellEnd::Var { var, .. } = cells.end(term) {
        into.insert(var);
    }
}
