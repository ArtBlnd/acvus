//! Type inference solver — manages type inference variables.
//!
//! Core types: `Solver`, `TypeBound`, `FreezeError`.
//! The solver is purely internal to type inference; graph-level types
//! use `PolyTy` (Solver-independent) or concrete `Ty`.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::graph::types::QualifiedRef;
use crate::ty::{
    CastRule, Effect, EffectConflict, EffectTerm, EffectVarId, IdentityId, Infer, InferTy,
    ParamTerm, Polarity, PolyTy, Ty, TyTerm, TypeRegistry,
    TypeBoundId,
};

// ── Solver types ────────────────────────────────────────────────────

/// Capability that a type may or may not possess.
/// Used in `TypeBound::Unresolved` to constrain what a type variable can resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Capability {
    Cloneable = 0,
    // Future: Equatable, Hashable, ...
}

/// State of a type inference variable in the solver.
#[derive(Debug, Clone)]
pub enum TypeBound {
    /// Resolved to a (possibly partially-known) type.
    /// Inner `InferTy` may still contain `Var` references to other bounds.
    Resolved(InferTy),
    /// Not yet resolved — constrained by capabilities and allowed types.
    /// `caps`: required capabilities (e.g. Cloneable).
    /// `allowed`: concrete types this variable may resolve to (lazy, checked at freeze).
    ///   Empty = unconstrained.
    Unresolved {
        caps: u8, // bitset over Capability
        allowed: Vec<Ty>,
    },
    /// Union-find forwarding pointer.
    Forward(TypeBoundId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EffectBound {
    /// `lower <= var <= upper` on the chain.
    Range { lower: Effect, upper: Effect },
    Bound(Effect),
    Forward(EffectVarId),
}

impl EffectBound {
    fn free() -> Self {
        EffectBound::Range { lower: Effect::Pure, upper: Effect::Opaque }
    }
}

// ── Solver ──────────────────────────────────────────────────────────

/// Snapshot for solver rollback during overload resolution.
pub struct SolverSnapshot {
    ty_bounds: Vec<TypeBound>,
    effect_vars: Vec<EffectBound>,
    identity_factory: acvus_utils::LocalFactory<IdentityId>,
}

/// Pure type inference solver.
///
/// Manages type inference variables. All constraint/resolution
/// state lives here — no inference state leaks into `Ty`.
pub struct Solver {
    pub(crate) ty_bounds: Vec<TypeBound>,
    pub(crate) effect_vars: Vec<EffectBound>,
    pub(crate) identity_factory: acvus_utils::LocalFactory<IdentityId>,
}

impl Solver {
    pub fn new() -> Self {
        Self {
            ty_bounds: Vec::new(),
            effect_vars: Vec::new(),
            identity_factory: acvus_utils::LocalFactory::new(),
        }
    }

    // ── Effect variables ────────────────────────────────────────────

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Infer> {
        let id = EffectVarId(self.effect_vars.len() as u32);
        self.effect_vars.push(EffectBound::free());
        EffectTerm::Var(id)
    }

    fn alloc_effect_var(effect_vars: &mut Vec<EffectBound>) -> EffectVarId {
        let id = EffectVarId(effect_vars.len() as u32);
        effect_vars.push(EffectBound::free());
        id
    }

    pub fn find_effect_root(&self, id: EffectVarId) -> EffectVarId {
        match &self.effect_vars[id.0 as usize] {
            EffectBound::Forward(next) => self.find_effect_root(*next),
            _ => id,
        }
    }

    pub fn resolve_effect(&self, term: &EffectTerm<Infer>) -> EffectTerm<Infer> {
        match term {
            EffectTerm::Known(e) => EffectTerm::Known(*e),
            EffectTerm::Var(id) => {
                let root = self.find_effect_root(*id);
                match &self.effect_vars[root.0 as usize] {
                    EffectBound::Bound(e) => EffectTerm::Known(*e),
                    EffectBound::Range { .. } => EffectTerm::Var(root),
                    EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
                }
            }
        }
    }

    fn range_of(&self, root: EffectVarId) -> (Effect, Effect) {
        match &self.effect_vars[root.0 as usize] {
            EffectBound::Range { lower, upper } => (*lower, *upper),
            EffectBound::Bound(e) => (*e, *e),
            EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
        }
    }

    /// RFC-0008: an effect variable that inference left open freezes to the
    /// top of what its constraints allow, Opaque when nothing constrained it.
    pub fn freeze_effect(&self, term: &EffectTerm<Infer>) -> Effect {
        match self.resolve_effect(term) {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(root) => self.range_of(root).1,
        }
    }

    /// The join of everything constrained below the variable. A function's
    /// effect is defined as this join over its body.
    pub fn effect_lower_bound(&self, term: &EffectTerm<Infer>) -> Effect {
        match self.resolve_effect(term) {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(root) => self.range_of(root).0,
        }
    }

    pub fn bind_effect(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        if effect < lower {
            return Err(EffectConflict { required: lower, allowed: effect });
        }
        if effect > upper {
            return Err(EffectConflict { required: effect, allowed: upper });
        }
        self.effect_vars[root.0 as usize] = EffectBound::Bound(effect);
        Ok(())
    }

    fn raise_lower(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        let lower = lower.join(effect);
        if lower > upper {
            return Err(EffectConflict { required: lower, allowed: upper });
        }
        self.effect_vars[root.0 as usize] = EffectBound::Range { lower, upper };
        Ok(())
    }

    fn lower_upper(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        let upper = upper.min(effect);
        if lower > upper {
            return Err(EffectConflict { required: lower, allowed: upper });
        }
        self.effect_vars[root.0 as usize] = EffectBound::Range { lower, upper };
        Ok(())
    }

    fn forward_effect(&mut self, from: EffectVarId, to: EffectVarId) -> Result<(), EffectConflict> {
        let from_root = self.find_effect_root(from);
        let to_root = self.find_effect_root(to);
        if from_root == to_root {
            return Ok(());
        }
        let (la, ua) = self.range_of(from_root);
        let (lb, ub) = self.range_of(to_root);
        let lower = la.join(lb);
        let upper = ua.min(ub);
        if lower > upper {
            return Err(EffectConflict { required: lower, allowed: upper });
        }
        self.effect_vars[to_root.0 as usize] = EffectBound::Range { lower, upper };
        self.effect_vars[from_root.0 as usize] = EffectBound::Forward(to_root);
        Ok(())
    }

    pub fn unify_effect(
        &mut self,
        a: &EffectTerm<Infer>,
        b: &EffectTerm<Infer>,
        pol: Polarity,
    ) -> Result<(), EffectConflict> {
        let a = self.resolve_effect(a);
        let b = self.resolve_effect(b);
        match (a, b) {
            (EffectTerm::Known(ea), EffectTerm::Known(eb)) => {
                let (required, allowed) = match pol {
                    Polarity::Invariant => (ea.join(eb), ea.min(eb)),
                    Polarity::Covariant => (ea, eb),
                    Polarity::Contravariant => (eb, ea),
                };
                if required <= allowed {
                    Ok(())
                } else {
                    Err(EffectConflict { required, allowed })
                }
            }
            (EffectTerm::Var(va), EffectTerm::Known(eb)) => match pol {
                Polarity::Invariant => self.bind_effect(va, eb),
                Polarity::Covariant => self.lower_upper(va, eb),
                Polarity::Contravariant => self.raise_lower(va, eb),
            },
            (EffectTerm::Known(ea), EffectTerm::Var(vb)) => match pol {
                Polarity::Invariant => self.bind_effect(vb, ea),
                Polarity::Covariant => self.raise_lower(vb, ea),
                Polarity::Contravariant => self.lower_upper(vb, ea),
            },
            (EffectTerm::Var(va), EffectTerm::Var(vb)) => self.forward_effect(va, vb),
        }
    }

    pub fn lub_effect(
        &mut self,
        a: &EffectTerm<Infer>,
        b: &EffectTerm<Infer>,
    ) -> Result<EffectTerm<Infer>, EffectConflict> {
        let a = self.resolve_effect(a);
        let b = self.resolve_effect(b);
        match (a, b) {
            (EffectTerm::Known(ea), EffectTerm::Known(eb)) => Ok(EffectTerm::Known(ea.join(eb))),
            (EffectTerm::Var(v), EffectTerm::Known(e)) | (EffectTerm::Known(e), EffectTerm::Var(v)) => {
                self.raise_lower(v, e)?;
                Ok(EffectTerm::Var(self.find_effect_root(v)))
            }
            (EffectTerm::Var(va), EffectTerm::Var(vb)) => {
                self.forward_effect(va, vb)?;
                Ok(EffectTerm::Var(self.find_effect_root(vb)))
            }
        }
    }

    /// Allocate a fresh unconstrained type variable.
    pub fn fresh_ty_var(&mut self) -> InferTy {
        let id = TypeBoundId(self.ty_bounds.len() as u32);
        self.ty_bounds.push(TypeBound::Unresolved {
            caps: 0,
            allowed: Vec::new(),
        });
        TyTerm::Var(id)
    }

    pub fn alloc_identity(&mut self) -> InferTy {
        TyTerm::Identity(self.identity_factory.next())
    }

    /// Take a snapshot for later rollback.
    pub fn snapshot(&self) -> SolverSnapshot {
        SolverSnapshot {
            ty_bounds: self.ty_bounds.clone(),
            effect_vars: self.effect_vars.clone(),
            identity_factory: self.identity_factory.clone(),
        }
    }

    /// Rollback to a snapshot: restore entire state.
    pub fn rollback(&mut self, snap: SolverSnapshot) {
        self.ty_bounds = snap.ty_bounds;
        self.effect_vars = snap.effect_vars;
        self.identity_factory = snap.identity_factory;
    }

    // ── Resolution ──────────────────────────────────────────────────

    /// Follow forwarding pointers to find the root bound for a type variable.
    pub fn find_ty_root(&self, id: TypeBoundId) -> TypeBoundId {
        match &self.ty_bounds[id.0 as usize] {
            TypeBound::Forward(next) => self.find_ty_root(*next),
            _ => id,
        }
    }

    /// Freeze an InferTy into a concrete Ty.
    /// Returns Err if any unresolved variable remains.
    pub fn freeze_ty(&self, ty: &InferTy) -> Result<Ty, FreezeError> {
        ty.try_map(
            &mut |id: TypeBoundId| {
                let root = self.find_ty_root(id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved(inner) => self.freeze_ty(inner),
                    TypeBound::Unresolved { .. } => Err(FreezeError::UnresolvedType(root)),
                    TypeBound::Forward(_) => unreachable!("find_ty_root should resolve forwards"),
                }
            },
            &mut |id| Ok(TyTerm::Identity(id)),
            &mut |id: EffectVarId| Ok(EffectTerm::Known(self.freeze_effect(&EffectTerm::Var(id)))),
        )
    }

    // ── Resolve ─────────────────────────────────────────────────────

    /// Shallow-resolve: follow Var chains but don't recurse into structure.
    pub fn shallow_resolve_ty(&self, ty: &InferTy) -> InferTy {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved(inner) => self.shallow_resolve_ty(inner),
                    _ => TyTerm::Var(root),
                }
            }
            other => other.clone(),
        }
    }

    /// Deep-resolve: follow Var chains and recurse into structure.
    pub fn resolve_ty(&self, ty: &InferTy) -> InferTy {
        ty.map(
            &mut |id: TypeBoundId| {
                let root = self.find_ty_root(id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved(inner) => self.resolve_ty(inner),
                    _ => TyTerm::Var(root),
                }
            },
            &mut |id| TyTerm::Identity(id),
            &mut |id: EffectVarId| self.resolve_effect(&EffectTerm::Var(id)),
        )
    }

    // ── Occurs check ────────────────────────────────────────────────

    /// Returns true if the type variable `id` appears in `ty`.
    fn occurs_in(&self, id: TypeBoundId, ty: &InferTy) -> bool {
        match ty {
            TyTerm::Var(other_id) => {
                let root = self.find_ty_root(*other_id);
                if root == id {
                    return true;
                }
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved(inner) => self.occurs_in(id, inner),
                    _ => false,
                }
            }
            TyTerm::List(inner) | TyTerm::Option(inner) | TyTerm::Ref(inner, _) => {
                self.occurs_in(id, inner)
            }
            TyTerm::Tuple(elems) => elems.iter().any(|e| self.occurs_in(id, e)),
            TyTerm::Object(fields) => fields.values().any(|v| self.occurs_in(id, v)),
            TyTerm::Fn { params, ret, captures, .. } => {
                params.iter().any(|p| self.occurs_in(id, &p.ty))
                    || self.occurs_in(id, ret)
                    || captures.iter().any(|c| self.occurs_in(id, c))
            }
            TyTerm::Enum { variants, .. } => variants
                .values()
                .any(|p| p.as_ref().is_some_and(|ty| self.occurs_in(id, ty))),
            TyTerm::UserDefined { type_args, .. } => {
                type_args.iter().any(|t| self.occurs_in(id, t))
            }
            TyTerm::Handle(inner) => self.occurs_in(id, inner),
            _ => false,
        }
    }

    /// Find the leaf Var in an InferTy — the deepest Var in a binding chain
    /// that is bound to a concrete (non-Var) type. Returns None if not a Var.
    pub fn find_leaf_var(&self, ty: &InferTy) -> Option<TypeBoundId> {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved(inner) => match inner {
                        TyTerm::Var(_) => self.find_leaf_var(inner),
                        _ => Some(root),
                    },
                    _ => Some(root),
                }
            }
            _ => None,
        }
    }

    // ── Bind helpers ────────────────────────────────────────────────

    /// Bind a type variable to a resolved InferTy.
    pub fn bind_ty(&mut self, id: TypeBoundId, ty: InferTy) {
        let root = self.find_ty_root(id);
        self.ty_bounds[root.0 as usize] = TypeBound::Resolved(ty);
    }

    /// Bind a type variable to forward to another.
    fn forward_ty(&mut self, from: TypeBoundId, to: TypeBoundId) {
        let from_root = self.find_ty_root(from);
        let to_root = self.find_ty_root(to);
        if from_root != to_root {
            self.ty_bounds[from_root.0 as usize] = TypeBound::Forward(to_root);
        }
    }

    // ── Type unification ────────────────────────────────────────────

    /// Unify two InferTy with polarity-based subtyping.
    /// Returns `Ok(Some(fn_ref))` when an ExternCast coercion was used.
    pub fn unify_ty(
        &mut self,
        a: &InferTy,
        b: &InferTy,
        pol: Polarity,
        registry: &TypeRegistry,
    ) -> Result<Option<QualifiedRef>, (InferTy, InferTy)> {
        let orig_a = a;
        let orig_b = b;
        let a = self.shallow_resolve_ty(a);
        let b = self.shallow_resolve_ty(b);

        match (&a, &b) {
            (TyTerm::Error(_), _) | (_, TyTerm::Error(_)) => Ok(None),

            (TyTerm::Int, TyTerm::Int)
            | (TyTerm::Float, TyTerm::Float)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Byte, TyTerm::Byte) => Ok(None),

            (
                TyTerm::UserDefined { id: id_a, type_args: ta_args, effect_args: ea_args },
                TyTerm::UserDefined { id: id_b, type_args: tb_args, effect_args: eb_args },
            ) if id_a == id_b => {
                assert_eq!(ta_args.len(), tb_args.len());
                assert_eq!(ea_args.len(), eb_args.len());
                let snap = self.snapshot();
                let type_ok = ta_args.iter().zip(tb_args.iter())
                    .all(|(a, b)| self.unify_ty(a, b, Polarity::Invariant, registry).is_ok());
                let effect_ok = ea_args.iter().zip(eb_args.iter())
                    .all(|(a, b)| self.unify_effect(a, b, pol).is_ok());
                if type_ok && effect_ok {
                    Ok(None)
                } else {
                    self.rollback(snap);
                    self.lub_or_err_infer(pol, orig_a, orig_b, &a, &b, registry)
                }
            }

            (
                TyTerm::Enum { name: na, variants: va },
                TyTerm::Enum { name: nb, variants: vb },
            ) => {
                if na != nb {
                    return Err((a, b));
                }
                for (tag, payload_a) in va {
                    if let Some(payload_b) = vb.get(tag) {
                        match (payload_a, payload_b) {
                            (None, None) => {}
                            (Some(ty_a), Some(ty_b)) => { self.unify_ty(ty_a, ty_b, pol, registry)?; }
                            _ => return Err((a.clone(), b.clone())),
                        }
                    }
                }
                let needs_merge = va.len() != vb.len() || va.keys().any(|k| !vb.contains_key(k));
                if needs_merge {
                    let mut merged: FxHashMap<Astr, Option<Box<InferTy>>> = va.clone();
                    for (tag, payload) in vb {
                        merged.entry(*tag).or_insert_with(|| payload.clone());
                    }
                    let merged_ty = TyTerm::Enum { name: *na, variants: merged };
                    if let Some(leaf) = self.find_leaf_var(orig_a) {
                        self.bind_ty(leaf, merged_ty.clone());
                    }
                    if let Some(leaf) = self.find_leaf_var(orig_b) {
                        self.bind_ty(leaf, merged_ty);
                    }
                }
                Ok(None)
            }

            // Var + anything
            (TyTerm::Var(id), other) | (other, TyTerm::Var(id)) => {
                if let TyTerm::Var(id2) = other
                    && id == id2
                {
                    return Ok(None);
                }
                if self.occurs_in(*id, other) {
                    return Err((a.clone(), b.clone()));
                }
                match other {
                    TyTerm::Var(id2) => {
                        // Var + Var: check allowed intersection, then forward.
                        let root1 = self.find_ty_root(*id);
                        let root2 = self.find_ty_root(*id2);
                        let (allowed1, caps1) = match &self.ty_bounds[root1.0 as usize] {
                            TypeBound::Unresolved { allowed, caps } => (allowed.clone(), *caps),
                            _ => (vec![], 0),
                        };
                        let (allowed2, caps2) = match &self.ty_bounds[root2.0 as usize] {
                            TypeBound::Unresolved { allowed, caps } => (allowed.clone(), *caps),
                            _ => (vec![], 0),
                        };
                        // Intersect allowed (if both non-empty).
                        let merged_allowed = if allowed1.is_empty() {
                            allowed2
                        } else if allowed2.is_empty() {
                            allowed1
                        } else {
                            let inter: Vec<Ty> = allowed1.into_iter()
                                .filter(|t| allowed2.contains(t))
                                .collect();
                            if inter.is_empty() {
                                return Err((a.clone(), b.clone()));
                            }
                            inter
                        };
                        let merged_caps = caps1 | caps2;
                        // Forward id → id2, update id2's bounds.
                        self.ty_bounds[root2.0 as usize] = TypeBound::Unresolved {
                            caps: merged_caps,
                            allowed: merged_allowed,
                        };
                        self.forward_ty(root1, root2);
                    }
                    TyTerm::Error(_) => {
                        self.bind_ty(*id, other.clone());
                    }
                    _ => {
                        // Var + concrete: check allowed constraint.
                        let root = self.find_ty_root(*id);
                        if let TypeBound::Unresolved { ref allowed, .. } = self.ty_bounds[root.0 as usize] {
                            if !allowed.is_empty() {
                                // Check if concrete type matches any allowed type.
                                // We freeze `other` to compare against allowed (Vec<Ty>).
                                if let Ok(concrete) = self.freeze_ty(other) {
                                    if !allowed.contains(&concrete) {
                                        return Err((a.clone(), b.clone()));
                                    }
                                }
                                // If freeze fails (still has vars), defer check.
                            }
                        }
                        self.bind_ty(*id, other.clone());
                    }
                }
                Ok(None)
            }

            (TyTerm::Tuple(ea), TyTerm::Tuple(eb)) => {
                if ea.len() != eb.len() {
                    return Err((a.clone(), b.clone()));
                }
                for (ta, tb) in ea.iter().zip(eb.iter()) {
                    self.unify_ty(ta, tb, pol, registry)?;
                }
                Ok(None)
            }

            (TyTerm::List(a), TyTerm::List(b)) => self.unify_ty(a, b, Polarity::Invariant, registry),

            (TyTerm::Identity(a), TyTerm::Identity(b)) => {
                if a == b { Ok(None) } else { Err((TyTerm::Identity(*a), TyTerm::Identity(*b))) }
            }

            (TyTerm::Option(a), TyTerm::Option(b)) => self.unify_ty(a, b, Polarity::Invariant, registry),

            (TyTerm::Object(fa), TyTerm::Object(fb)) => {
                let snap = self.snapshot();
                let mut field_mismatch = false;
                for (key, ty_a) in fa {
                    if let Some(ty_b) = fb.get(key)
                        && self.unify_ty(ty_a, ty_b, pol, registry).is_err()
                    {
                        field_mismatch = true;
                        break;
                    }
                }
                if field_mismatch && pol != Polarity::Invariant {
                    self.rollback(snap);
                    let mut merged = FxHashMap::default();
                    for (key, ty_a) in fa {
                        if let Some(ty_b) = fb.get(key) {
                            let fsnap = self.snapshot();
                            if self.unify_ty(ty_a, ty_b, pol, registry).is_ok() {
                                merged.insert(*key, self.resolve_ty(ty_a));
                            } else {
                                self.rollback(fsnap);
                                let lub = self.try_lub_infer(ty_a, ty_b, registry)
                                    .ok_or_else(|| (a.clone(), b.clone()))?;
                                merged.insert(*key, lub);
                            }
                        } else {
                            merged.insert(*key, self.resolve_ty(ty_a));
                        }
                    }
                    for (key, ty_b) in fb {
                        if !fa.contains_key(key) {
                            merged.insert(*key, self.resolve_ty(ty_b));
                        }
                    }
                    let merged_ty = TyTerm::Object(merged);
                    if let Some(leaf) = self.find_leaf_var(orig_a) {
                        self.bind_ty(leaf, merged_ty.clone());
                    }
                    if let Some(leaf) = self.find_leaf_var(orig_b) {
                        self.bind_ty(leaf, merged_ty);
                    }
                    return Ok(None);
                } else if field_mismatch {
                    return Err((a.clone(), b.clone()));
                }

                let a_only = fa.keys().any(|k| !fb.contains_key(k));
                let b_only = fb.keys().any(|k| !fa.contains_key(k));
                if !a_only && !b_only {
                    return Ok(None);
                }

                let leaf_a = self.find_leaf_var(orig_a);
                let leaf_b = self.find_leaf_var(orig_b);
                if leaf_a.is_none() && leaf_b.is_none() {
                    return Err((a.clone(), b.clone()));
                }

                let mut merged = FxHashMap::default();
                for (k, v) in fa {
                    merged.insert(*k, self.resolve_ty(v));
                }
                for (k, v) in fb {
                    merged.entry(*k).or_insert_with(|| self.resolve_ty(v));
                }
                let merged_ty = TyTerm::Object(merged);
                if let Some(var) = leaf_a {
                    self.bind_ty(var, merged_ty.clone());
                }
                if let Some(var) = leaf_b {
                    self.bind_ty(var, merged_ty);
                }
                Ok(None)
            }

            (
                TyTerm::Fn { params: pa, ret: ra, effect: ea, .. },
                TyTerm::Fn { params: pb, ret: rb, effect: eb, .. },
            ) => {
                if pa.len() != pb.len() {
                    return Err((a.clone(), b.clone()));
                }
                let param_pol = pol.flip();
                for (ta, tb) in pa.iter().zip(pb.iter()) {
                    self.unify_ty(&ta.ty, &tb.ty, param_pol, registry)?;
                }
                self.unify_ty(ra, rb, pol, registry)?;
                if self.unify_effect(ea, eb, pol).is_err() {
                    return Err((a.clone(), b.clone()));
                }
                Ok(None)
            }

            // Cross-type coercion
            _ => {
                if pol != Polarity::Invariant {
                    let (sub, sup) = match pol {
                        Polarity::Covariant => (&a, &b),
                        Polarity::Contravariant => (&b, &a),
                        Polarity::Invariant => unreachable!(),
                    };
                    if let Ok(maybe_fn) = self.try_coerce_infer(sub, sup, registry) {
                        return Ok(maybe_fn);
                    }
                }
                Err((a, b))
            }
        }
    }

    // ── LUB ─────────────────────────────────────────────────────────

    fn try_lub_infer(&mut self, a: &InferTy, b: &InferTy, registry: &TypeRegistry) -> Option<InferTy> {
        match (a, b) {
            (
                TyTerm::Fn { params: pa, ret: ra, effect: ea, .. },
                TyTerm::Fn { params: pb, ret: rb, effect: eb, .. },
            ) => {
                if pa.len() != pb.len() { return None; }
                for (a, b) in pa.iter().zip(pb.iter()) {
                    self.unify_ty(&a.ty, &b.ty, Polarity::Invariant, registry).ok()?;
                }
                self.unify_ty(ra, rb, Polarity::Invariant, registry).ok()?;
                let effect = self.lub_effect(ea, eb).ok()?;
                Some(TyTerm::Fn {
                    params: pa.iter().map(|p| ParamTerm::new(p.name, self.resolve_ty(&p.ty))).collect(),
                    ret: Box::new(self.resolve_ty(ra)),
                    captures: vec![],
                    effect,
                })
            }
            (
                TyTerm::UserDefined { id: id_a, type_args: ta_a, effect_args: ea_a },
                TyTerm::UserDefined { id: id_b, type_args: ta_b, effect_args: ea_b },
            ) if id_a == id_b => {
                assert_eq!(ta_a.len(), ta_b.len());
                let snap = self.snapshot();
                let type_args_ok = ta_a.iter().zip(ta_b.iter())
                    .all(|(a, b)| self.unify_ty(a, b, Polarity::Invariant, registry).is_ok());
                if type_args_ok {
                    let mut effect_args = Vec::with_capacity(ea_a.len());
                    for (a, b) in ea_a.iter().zip(ea_b.iter()) {
                        effect_args.push(self.lub_effect(a, b).ok()?);
                    }
                    Some(TyTerm::UserDefined {
                        id: *id_a,
                        type_args: ta_a.iter().map(|t| self.resolve_ty(t)).collect(),
                        effect_args,
                    })
                } else {
                    self.rollback(snap);
                    self.try_lub_via_cast_rules_infer(*id_a, a, b, registry)
                }
            }
            _ => None,
        }
    }

    fn try_lub_via_cast_rules_infer(
        &mut self,
        from_id: QualifiedRef,
        a: &InferTy,
        b: &InferTy,
        registry: &TypeRegistry,
    ) -> Option<InferTy> {
        let rules = registry.rules_from(from_id).to_vec();
        if rules.is_empty() { return None; }

        for rule in &rules {
            let snap = self.snapshot();
            let (inst_from_a, inst_to_a) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let a_ok = self.unify_ty(a, &inst_from_a, Polarity::Invariant, registry).is_ok();
            let target_a = if a_ok { Some(self.resolve_ty(&inst_to_a)) } else { None };
            self.rollback(snap);
            let target_a = target_a?;

            let snap = self.snapshot();
            let (inst_from_b, inst_to_b) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let b_ok = self.unify_ty(b, &inst_from_b, Polarity::Invariant, registry).is_ok();
            let target_b = if b_ok { Some(self.resolve_ty(&inst_to_b)) } else { None };
            self.rollback(snap);
            let target_b = target_b?;

            let snap = self.snapshot();
            if self.unify_ty(&target_a, &target_b, Polarity::Covariant, registry).is_ok() {
                let result = self.resolve_ty(&target_a);
                self.rollback(snap);
                return Some(result);
            }
            self.rollback(snap);
        }
        None
    }

    fn lub_or_err_infer(
        &mut self,
        pol: Polarity,
        orig_a: &InferTy,
        orig_b: &InferTy,
        a: &InferTy,
        b: &InferTy,
        registry: &TypeRegistry,
    ) -> Result<Option<QualifiedRef>, (InferTy, InferTy)> {
        if pol == Polarity::Invariant {
            return Err((a.clone(), b.clone()));
        }
        let leaf_a = self.find_leaf_var(orig_a);
        let leaf_b = self.find_leaf_var(orig_b);
        if leaf_a.is_none() && leaf_b.is_none() {
            return Err((a.clone(), b.clone()));
        }
        let lub = self.try_lub_infer(a, b, registry).ok_or_else(|| (a.clone(), b.clone()))?;
        if let Some(leaf) = leaf_a {
            self.bind_ty(leaf, lub.clone());
        }
        if let Some(leaf) = leaf_b {
            self.bind_ty(leaf, lub);
        }
        Ok(None)
    }

    // ── Coercion ────────────────────────────────────────────────────

    /// Try subtype coercion on InferTy. Returns Ok(Some(qref)) if ExternCast used.
    fn try_coerce_infer(
        &mut self,
        sub: &InferTy,
        sup: &InferTy,
        registry: &TypeRegistry,
    ) -> Result<Option<QualifiedRef>, ()> {
        match (sub, sup) {
            (TyTerm::UserDefined { id, .. }, _) => {
                let rules = registry.rules_from(*id).to_vec();
                self.try_extern_cast_rules_infer(&rules, sub, sup, registry)
            }
            (_, TyTerm::UserDefined { id, .. }) => {
                let rules = registry.rules_to(*id).to_vec();
                self.try_extern_cast_rules_infer(&rules, sub, sup, registry)
            }
            _ => Err(()),
        }
    }

    fn try_extern_cast_rules_infer(
        &mut self,
        rules: &[CastRule],
        sub: &InferTy,
        sup: &InferTy,
        registry: &TypeRegistry,
    ) -> Result<Option<QualifiedRef>, ()> {
        if rules.is_empty() { return Err(()); }

        let mut matched_idx = None;
        for (i, rule) in rules.iter().enumerate() {
            let snap = self.snapshot();
            let (inst_from, inst_to) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let ok = self.unify_ty(sub, &inst_from, Polarity::Invariant, registry).is_ok()
                && self.unify_ty(&inst_to, sup, Polarity::Invariant, registry).is_ok();
            self.rollback(snap);
            if ok {
                if matched_idx.is_some() { return Err(()); }
                matched_idx = Some(i);
            }
        }

        let idx = matched_idx.ok_or(())?;
        let rule = &rules[idx];
        let fn_ref = rule.fn_ref;
        let (inst_from, inst_to) = self.instantiate_poly_pair(&rule.from, &rule.to);
        self.unify_ty(sub, &inst_from, Polarity::Invariant, registry).map_err(|_| ())?;
        self.unify_ty(&inst_to, sup, Polarity::Invariant, registry).map_err(|_| ())?;
        Ok(Some(fn_ref))
    }

    // ── Instantiate ─────────────────────────────────────────────────

    /// Instantiate a polymorphic InferTy: replace all Var
    /// and Identity with fresh values.
    pub fn instantiate_infer(&mut self, ty: &InferTy) -> InferTy {
        let mut var_map: FxHashMap<TypeBoundId, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        let mut effect_map: FxHashMap<EffectVarId, EffectVarId> = FxHashMap::default();
        self.instantiate_infer_inner(ty, &mut var_map, &mut fresh_map, &mut effect_map)
    }

    /// Instantiate two InferTy sharing the same variable mappings.
    pub fn instantiate_pair_infer(&mut self, a: &InferTy, b: &InferTy) -> (InferTy, InferTy) {
        let mut var_map: FxHashMap<TypeBoundId, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        let mut effect_map: FxHashMap<EffectVarId, EffectVarId> = FxHashMap::default();
        let ia = self.instantiate_infer_inner(a, &mut var_map, &mut fresh_map, &mut effect_map);
        let ib = self.instantiate_infer_inner(b, &mut var_map, &mut fresh_map, &mut effect_map);
        (ia, ib)
    }

    fn instantiate_infer_inner(
        &mut self,
        ty: &InferTy,
        var_map: &mut FxHashMap<TypeBoundId, TypeBoundId>,
        fresh_map: &mut FxHashMap<IdentityId, IdentityId>,
        effect_map: &mut FxHashMap<EffectVarId, EffectVarId>,
    ) -> InferTy {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                // Clone bound data before mutating to avoid borrow conflict.
                let bound = self.ty_bounds[root.0 as usize].clone();
                match bound {
                    TypeBound::Resolved(inner) => {
                        self.instantiate_infer_inner(&inner, var_map, fresh_map, effect_map)
                    }
                    TypeBound::Unresolved { caps, allowed } => {
                        let new_id = *var_map.entry(root).or_insert_with(|| {
                            let fresh = TypeBoundId(self.ty_bounds.len() as u32);
                            self.ty_bounds.push(TypeBound::Unresolved {
                                caps,
                                allowed,
                            });
                            fresh
                        });
                        TyTerm::Var(new_id)
                    }
                    TypeBound::Forward(_) => unreachable!(),
                }
            }
            TyTerm::List(inner) => TyTerm::List(Box::new(
                self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map),
            )),
            TyTerm::Identity(id) => {
                let new_id = *fresh_map
                    .entry(*id)
                    .or_insert_with(|| self.identity_factory.next());
                TyTerm::Identity(new_id)
            }
            TyTerm::Option(inner) => TyTerm::Option(Box::new(
                self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map),
            )),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems.iter().map(|e| self.instantiate_infer_inner(e, var_map, fresh_map, effect_map)).collect(),
            ),
            TyTerm::Object(fields) => TyTerm::Object(
                fields.iter().map(|(k, v)| (*k, self.instantiate_infer_inner(v, var_map, fresh_map, effect_map))).collect(),
            ),
            TyTerm::Fn { params, ret, captures, effect } => TyTerm::Fn {
                params: params.iter().map(|p| ParamTerm::new(
                    p.name,
                    self.instantiate_infer_inner(&p.ty, var_map, fresh_map, effect_map),
                )).collect(),
                ret: Box::new(self.instantiate_infer_inner(ret, var_map, fresh_map, effect_map)),
                captures: captures.iter().map(|c| self.instantiate_infer_inner(c, var_map, fresh_map, effect_map)).collect(),
                effect: self.instantiate_effect_inner(effect, effect_map),
            },
            TyTerm::Enum { name, variants } => TyTerm::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| {
                    (*tag, payload.as_ref().map(|ty| Box::new(
                        self.instantiate_infer_inner(ty, var_map, fresh_map, effect_map),
                    )))
                }).collect(),
            },
            TyTerm::UserDefined { id, type_args, effect_args } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| self.instantiate_infer_inner(t, var_map, fresh_map, effect_map)).collect(),
                effect_args: effect_args.iter().map(|e| self.instantiate_effect_inner(e, effect_map)).collect(),
            },
            TyTerm::Handle(inner) => TyTerm::Handle(
                Box::new(self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map)),
            ),
            TyTerm::Ref(inner, volatile) => TyTerm::Ref(
                Box::new(self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map)),
                *volatile,
            ),
            other => other.clone(),
        }
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

/// Error when freezing an InferTy that still contains unresolved variables.
#[derive(Debug)]
pub enum FreezeError {
    UnresolvedType(TypeBoundId),
}

// ── Poly → Infer instantiation (in Solver) ──────────────────────────

impl Solver {
    /// Instantiate a PolyTy template into InferTy, replacing each positional
    /// placeholder with a fresh Solver variable.
    fn instantiate_effect_inner(
        &mut self,
        term: &EffectTerm<Infer>,
        effect_map: &mut FxHashMap<EffectVarId, EffectVarId>,
    ) -> EffectTerm<Infer> {
        match self.resolve_effect(term) {
            EffectTerm::Known(e) => EffectTerm::Known(e),
            EffectTerm::Var(root) => {
                let fresh = *effect_map
                    .entry(root)
                    .or_insert_with(|| Self::alloc_effect_var(&mut self.effect_vars));
                EffectTerm::Var(fresh)
            }
        }
    }

    pub fn instantiate_poly(&mut self, ty: &PolyTy) -> InferTy {
        let mut var_map: FxHashMap<u32, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, EffectVarId> = FxHashMap::default();
        let ty_bounds = &mut self.ty_bounds;
        let effect_vars = &mut self.effect_vars;
        let identity_factory = &mut self.identity_factory;
        ty.map(
            &mut |id: u32| {
                let bound_id = *var_map.entry(id).or_insert_with(|| {
                    let fresh = TypeBoundId(ty_bounds.len() as u32);
                    ty_bounds.push(TypeBound::Unresolved { caps: 0, allowed: Vec::new() });
                    fresh
                });
                TyTerm::Var(bound_id)
            },
            &mut |id| {
                let new_id = *fresh_map
                    .entry(id)
                    .or_insert_with(|| identity_factory.next());
                TyTerm::Identity(new_id)
            },
            &mut |id: u32| {
                let var = *effect_map
                    .entry(id)
                    .or_insert_with(|| Self::alloc_effect_var(effect_vars));
                EffectTerm::Var(var)
            },
        )
    }

    /// Instantiate two PolyTy templates sharing the same variable mappings.
    /// Used for CastRule where `from` and `to` share placeholder variables.
    pub fn instantiate_poly_pair(&mut self, a: &PolyTy, b: &PolyTy) -> (InferTy, InferTy) {
        let mut var_map: FxHashMap<u32, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, EffectVarId> = FxHashMap::default();
        let ty_bounds = &mut self.ty_bounds;
        let effect_vars = &mut self.effect_vars;
        let identity_factory = &mut self.identity_factory;
        let mut on_effect = |id: u32| {
            let var = *effect_map
                .entry(id)
                .or_insert_with(|| Self::alloc_effect_var(effect_vars));
            EffectTerm::Var(var)
        };
        let mut on_var = |id: u32| {
            let bound_id = *var_map.entry(id).or_insert_with(|| {
                let fresh = TypeBoundId(ty_bounds.len() as u32);
                ty_bounds.push(TypeBound::Unresolved { caps: 0, allowed: Vec::new() });
                fresh
            });
            TyTerm::Var(bound_id)
        };
        let mut on_identity = |id: IdentityId| {
            let new_id = *fresh_map
                .entry(id)
                .or_insert_with(|| identity_factory.next());
            TyTerm::Identity(new_id)
        };
        let ia = a.map(&mut on_var, &mut on_identity, &mut on_effect);
        let ib = b.map(&mut on_var, &mut on_identity, &mut on_effect);
        (ia, ib)
    }
}
