//! A term's variables, read and copied: what generalizing a binding's type
//! and instantiating it at a use need (functions §2).

use rustc_hash::{FxHashMap, FxHashSet};

use super::{Infer, Term, VarState};
use crate::ty::{Label, ObjectTy, TyKind, TyVarId, TypeArg};

impl Infer {
    /// How many type variables were made so far: a variable made later has
    /// an id at least this.
    pub fn ty_var_mark(&self) -> u32 {
        u32::try_from(self.ty_vars.len()).expect("Infer::ty_var_mark: u32::MAX variables")
    }

    /// The term that is the variable `var`.
    pub fn var_term(&mut self, var: TyVarId) -> Term {
        self.types.intern(TyKind::Var(var))
    }

    /// Every type variable reachable from `from`, bound variables followed
    /// and themselves included.
    pub fn vars_reachable(&self, from: impl IntoIterator<Item = Term>) -> FxHashSet<TyVarId> {
        let mut vars = FxHashSet::default();
        let mut seen = FxHashSet::default();
        let mut stack: Vec<Term> = from.into_iter().collect();
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            match self.types.kind(at) {
                TyKind::Var(var) => {
                    vars.insert(*var);
                    if let VarState::Bound(to) = &self.ty_vars[var.0 as usize] {
                        stack.push(*to);
                    }
                }
                kind => kind.for_each_child(|child| stack.push(child)),
            }
        }
        vars
    }

    /// Whether `var` is unbound.
    pub fn is_unbound(&self, var: TyVarId) -> bool {
        matches!(self.ty_vars[var.0 as usize], VarState::Unbound(_))
    }

    /// A fresh variable of `var`'s bound: a generalized variable keeps its
    /// bound, answered at each use (types §5).
    ///
    /// # Panics
    /// `var` is bound: only an unbound variable is generalized.
    pub fn fresh_as(&mut self, var: TyVarId) -> Term {
        let VarState::Unbound(bound) = self.ty_vars[var.0 as usize].clone() else {
            panic!("fresh_as: a generalized variable is unbound")
        };
        self.fresh(bound)
    }

    /// `term` with each variable of `at` replaced by its term there, bound
    /// variables followed. A node holding none of them is `term` itself;
    /// every node rebuilt around one is written elsewhere, since a use does
    /// not grow what the value wrote (types §1).
    pub fn copy_at(&mut self, term: Term, at: &FxHashMap<TyVarId, Term>, memo: &mut FxHashMap<Term, Term>) -> Term {
        acvus_utils::grow(|| self.copy_level(term, at, memo))
    }

    fn copy_level(&mut self, term: Term, at: &FxHashMap<TyVarId, Term>, memo: &mut FxHashMap<Term, Term>) -> Term {
        if let Some(done) = memo.get(&term) {
            return *done;
        }
        let kind = self.types.kind(term).clone();
        let copied = match kind {
            TyKind::Var(var) => match (at.get(&var), &self.ty_vars[var.0 as usize]) {
                (Some(fresh), _) => *fresh,
                (None, VarState::Bound(to)) => {
                    let to = *to;
                    let copied = self.copy_at(to, at, memo);
                    if copied == to { term } else { copied }
                }
                (None, VarState::Unbound(_)) => term,
            },
            kind => {
                let mut changed = false;
                let mut copy = |infer: &mut Infer, child: Term| {
                    let copied = infer.copy_at(child, at, memo);
                    changed |= copied != child;
                    copied
                };
                let copy_arg = |infer: &mut Infer, arg: TypeArg<crate::ty::Infer>, copy: &mut dyn FnMut(&mut Infer, Term) -> Term| match arg {
                    TypeArg::Uniform(ty) => TypeArg::Uniform(copy(infer, ty)),
                    TypeArg::Specialized(ty) => TypeArg::Specialized(copy(infer, ty)),
                    TypeArg::Open(mark, ty) => TypeArg::Open(mark, copy(infer, ty)),
                };
                let rebuilt = match kind {
                    TyKind::Handle(inner) => TyKind::Handle(copy(self, inner)),
                    TyKind::Array(element, len, _) => TyKind::Array(copy(self, element), len, Label::Elsewhere),
                    TyKind::Slice(element) => TyKind::Slice(copy(self, element)),
                    TyKind::Option(payload, _) => TyKind::Option(copy(self, payload), Label::Elsewhere),
                    TyKind::Result(ok, err, _) => {
                        let ok = copy(self, ok);
                        TyKind::Result(ok, copy(self, err), Label::Elsewhere)
                    }
                    TyKind::Tuple(items, _) => {
                        TyKind::Tuple(items.iter().map(|item| copy(self, *item)).collect(), Label::Elsewhere)
                    }
                    TyKind::Object(ObjectTy::Structural(fields, _)) => TyKind::Object(ObjectTy::Structural(
                        fields.iter().map(|(name, field)| (*name, copy(self, *field))).collect(),
                        Label::Elsewhere,
                    )),
                    TyKind::Object(ObjectTy::Declared(id, fields)) => {
                        TyKind::Object(ObjectTy::Declared(id, fields.iter().map(|(name, field)| (*name, copy(self, *field))).collect()))
                    }
                    TyKind::Enum { name, variants, .. } => TyKind::Enum {
                        name,
                        variants: variants.iter().map(|(tag, payload)| (*tag, payload.map(|payload| copy(self, payload)))).collect(),
                        label: Label::Elsewhere,
                    },
                    TyKind::Fn { params, ret, effect, flows, calls, holds } => TyKind::Fn {
                        params: params.iter().map(|param| copy(self, *param)).collect(),
                        ret: copy(self, ret),
                        effect,
                        flows,
                        calls,
                        holds,
                    },
                    TyKind::UserDefined { id, type_args, effect_args, identity_args } => TyKind::UserDefined {
                        id,
                        type_args: type_args.iter().map(|arg| copy_arg(self, arg.clone(), &mut copy)).collect(),
                        effect_args,
                        identity_args,
                    },
                    TyKind::Ref(mutability, target) => TyKind::Ref(mutability, copy_arg(self, target, &mut copy)),
                    leaf => leaf,
                };
                match changed {
                    true => self.intern(rebuilt),
                    false => term,
                }
            }
        };
        memo.insert(term, copied);
        copied
    }
}
