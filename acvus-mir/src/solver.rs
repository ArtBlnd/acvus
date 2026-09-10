//! Type inference solver - manages type inference variables.
//!
//! Core types: `Solver`, `TypeBound`, `FreezeError`.
//! The solver is purely internal to type inference; graph-level types
//! use `PolyTy` (Solver-independent) or concrete `Ty`.

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::graph::types::QualifiedRef;
use crate::ty::{
    CastRule, Effect, EffectConflict, EffectTerm, EffectVarId, IdentityId, IdentityTerm,
    IdentityVarId, Infer, InferTy, LenTerm, LenVarId, ParamTerm, Polarity, PolyTy, Scheme, Ty,
    TyTerm, TyVarBound, TypeBoundId, TypeRegistry,
};
use acvus_utils::LocalIdOps;

// -- Solver types ----------------------------------------------------

/// State of a type inference variable in the solver. The declared bound
/// travels with the variable and is verified when it freezes.
#[derive(Debug, Clone)]
pub enum TypeBound {
    /// Resolved to a (possibly partially-known) type.
    /// Inner `InferTy` may still contain `Var` references to other bounds.
    Resolved {
        ty: InferTy,
        bound: TyVarBound,
    },
    Unresolved {
        bound: TyVarBound,
    },
    /// Union-find forwarding pointer.
    Forward(TypeBoundId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EffectBound {
    /// `lower <= var <= upper` on the chain.
    Range {
        lower: Effect,
        upper: Effect,
    },
    Bound(Effect),
    Forward(EffectVarId),
}

impl EffectBound {
    fn free() -> Self {
        EffectBound::Range {
            lower: Effect::PURE,
            upper: Effect::OPAQUE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LenBound {
    Unbound,
    Bound(usize),
    Forward(LenVarId),
}

// -- Solver ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum IdentityBound {
    /// Not yet tied to a source. Frozen, it becomes a source of its own.
    Unbound,
    Bound(IdentityId),
    Forward(IdentityVarId),
}

/// Snapshot for solver rollback during overload resolution.
pub struct SolverSnapshot {
    ty_bounds: Vec<TypeBound>,
    effect_vars: Vec<EffectBound>,
    len_vars: Vec<LenBound>,
    identity_vars: Vec<IdentityBound>,
}

/// The sources of one compilation. A source number names one identity
/// for every solver of the compilation, so a frozen type may pass from
/// one solver to another and still name the source it was frozen with.
/// The compilation owns it and lends it to one solver at a time.
#[derive(Debug)]
pub struct Sources(acvus_utils::LocalFactory<IdentityId>);

impl Sources {
    pub fn new() -> Self {
        Self(acvus_utils::LocalFactory::new())
    }

    /// Mint a source no solver of this compilation has minted before.
    pub fn next(&mut self) -> IdentityId {
        self.0.next()
    }
}

/// Pure type inference solver.
///
/// Manages type inference variables. All constraint/resolution
/// state lives here - no inference state leaks into `Ty`.
pub struct Solver<'src> {
    pub(crate) ty_bounds: Vec<TypeBound>,
    pub(crate) effect_vars: Vec<EffectBound>,
    pub(crate) len_vars: Vec<LenBound>,
    pub(crate) identity_vars: Vec<IdentityBound>,
    /// Mints a new source for every identity a declaration introduces;
    /// lent by the compilation for this solver's lifetime.
    sources: &'src mut Sources,
}

impl<'src> Solver<'src> {
    pub fn new(sources: &'src mut Sources) -> Self {
        Self {
            ty_bounds: Vec::new(),
            effect_vars: Vec::new(),
            len_vars: Vec::new(),
            identity_vars: Vec::new(),
            sources,
        }
    }

    // -- Identity variables ------------------------------------------

    pub fn fresh_identity_var(&mut self) -> IdentityTerm<Infer> {
        IdentityTerm::Var(Self::alloc_identity_var(&mut self.identity_vars))
    }

    fn alloc_identity_var(identity_vars: &mut Vec<IdentityBound>) -> IdentityVarId {
        let id = IdentityVarId(identity_vars.len() as u32);
        identity_vars.push(IdentityBound::Unbound);
        id
    }

    pub fn find_identity_root(&self, id: IdentityVarId) -> IdentityVarId {
        match &self.identity_vars[id.0 as usize] {
            IdentityBound::Forward(next) => self.find_identity_root(*next),
            _ => id,
        }
    }

    pub fn resolve_identity(&self, term: &IdentityTerm<Infer>) -> IdentityTerm<Infer> {
        match term {
            IdentityTerm::Known(id) => IdentityTerm::Known(*id),
            IdentityTerm::Var(v) => {
                let root = self.find_identity_root(*v);
                match &self.identity_vars[root.0 as usize] {
                    IdentityBound::Bound(id) => IdentityTerm::Known(*id),
                    IdentityBound::Unbound => IdentityTerm::Var(root),
                    IdentityBound::Forward(_) => {
                        unreachable!("find_identity_root resolves forwards")
                    }
                }
            }
        }
    }

    /// Identities are invariant: two values share one only when they came
    /// from one source.
    pub fn unify_identity(
        &mut self,
        a: &IdentityTerm<Infer>,
        b: &IdentityTerm<Infer>,
    ) -> Result<(), (IdentityId, IdentityId)> {
        let a = self.resolve_identity(a);
        let b = self.resolve_identity(b);
        match (a, b) {
            (IdentityTerm::Known(x), IdentityTerm::Known(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err((x, y))
                }
            }
            (IdentityTerm::Var(v), IdentityTerm::Known(id))
            | (IdentityTerm::Known(id), IdentityTerm::Var(v)) => {
                self.identity_vars[v.0 as usize] = IdentityBound::Bound(id);
                Ok(())
            }
            (IdentityTerm::Var(x), IdentityTerm::Var(y)) => {
                if x != y {
                    self.identity_vars[x.0 as usize] = IdentityBound::Forward(y);
                }
                Ok(())
            }
        }
    }

    pub fn freeze_identity(&self, term: &IdentityTerm<Infer>) -> Result<IdentityId, FreezeError> {
        match self.resolve_identity(term) {
            IdentityTerm::Known(id) => Ok(id),
            IdentityTerm::Var(root) => Err(FreezeError::UnresolvedIdentity(root)),
        }
    }

    /// Tie every identity nothing has tied to a source to a source of its
    /// own. Run once a body is checked, before its types freeze.
    pub fn settle_identities(&mut self) {
        for index in 0..self.identity_vars.len() {
            if self.identity_vars[index] == IdentityBound::Unbound {
                self.identity_vars[index] = IdentityBound::Bound(self.sources.next());
            }
        }
    }

    /// The identity variables of a polymorphic type that occur in its
    /// parameters. Instantiated, these bind to the arguments' sources;
    /// every other identity variable is a new source at each instantiation.
    fn identity_vars_bound_by_params(ty: &PolyTy) -> rustc_hash::FxHashSet<u32> {
        let mut found = rustc_hash::FxHashSet::default();
        if let TyTerm::Fn { params, .. } = ty {
            for p in params {
                p.ty.map(
                    &mut |v: u32| TyTerm::<crate::ty::Poly>::Var(v),
                    &mut |v: u32| {
                        found.insert(v);
                        IdentityTerm::<crate::ty::Poly>::Var(v)
                    },
                    &mut |v: u32| EffectTerm::<crate::ty::Poly>::Var(v),
                    &mut |v: u32| LenTerm::<crate::ty::Poly>::Var(v),
                );
            }
        }
        found
    }

    // -- Length variables --------------------------------------------

    pub fn fresh_len_var(&mut self) -> LenTerm<Infer> {
        LenTerm::Var(Self::alloc_len_var(&mut self.len_vars))
    }

    fn alloc_len_var(len_vars: &mut Vec<LenBound>) -> LenVarId {
        let id = LenVarId(len_vars.len() as u32);
        len_vars.push(LenBound::Unbound);
        id
    }

    pub fn find_len_root(&self, id: LenVarId) -> LenVarId {
        match &self.len_vars[id.0 as usize] {
            LenBound::Forward(next) => self.find_len_root(*next),
            _ => id,
        }
    }

    pub fn resolve_len(&self, term: &LenTerm<Infer>) -> LenTerm<Infer> {
        match term {
            LenTerm::Known(n) => LenTerm::Known(*n),
            LenTerm::Var(id) => {
                let root = self.find_len_root(*id);
                match &self.len_vars[root.0 as usize] {
                    LenBound::Bound(n) => LenTerm::Known(*n),
                    LenBound::Unbound => LenTerm::Var(root),
                    LenBound::Forward(_) => unreachable!("find_len_root resolves forwards"),
                }
            }
        }
    }

    pub fn unify_len(
        &mut self,
        a: &LenTerm<Infer>,
        b: &LenTerm<Infer>,
    ) -> Result<(), (usize, usize)> {
        let a = self.resolve_len(a);
        let b = self.resolve_len(b);
        match (a, b) {
            (LenTerm::Known(na), LenTerm::Known(nb)) => {
                if na == nb {
                    Ok(())
                } else {
                    Err((na, nb))
                }
            }
            (LenTerm::Var(v), LenTerm::Known(n)) | (LenTerm::Known(n), LenTerm::Var(v)) => {
                self.len_vars[v.0 as usize] = LenBound::Bound(n);
                Ok(())
            }
            (LenTerm::Var(va), LenTerm::Var(vb)) => {
                if va != vb {
                    self.len_vars[va.0 as usize] = LenBound::Forward(vb);
                }
                Ok(())
            }
        }
    }

    // -- Effect variables --------------------------------------------

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

    /// An effect variable inference left open freezes to the join of what
    /// is constrained below it, Pure when nothing is (RFC-0014). A body and
    /// the calls inside it share variables, so one reading serves both.
    pub fn freeze_effect(&self, term: &EffectTerm<Infer>) -> Effect {
        match self.resolve_effect(term) {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(root) => self.range_of(root).0,
        }
    }

    pub fn bind_effect(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        if !lower.at_most(effect) {
            return Err(EffectConflict {
                required: lower,
                allowed: effect,
            });
        }
        if !effect.at_most(upper) {
            return Err(EffectConflict {
                required: effect,
                allowed: upper,
            });
        }
        self.effect_vars[root.0 as usize] = EffectBound::Bound(effect);
        Ok(())
    }

    fn raise_lower(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        let lower = lower.join(effect);
        if !lower.at_most(upper) {
            return Err(EffectConflict {
                required: lower,
                allowed: upper,
            });
        }
        self.effect_vars[root.0 as usize] = EffectBound::Range { lower, upper };
        Ok(())
    }

    fn lower_upper(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let (lower, upper) = self.range_of(root);
        let upper = upper.meet(effect);
        if !lower.at_most(upper) {
            return Err(EffectConflict {
                required: lower,
                allowed: upper,
            });
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
        let upper = ua.meet(ub);
        if !lower.at_most(upper) {
            return Err(EffectConflict {
                required: lower,
                allowed: upper,
            });
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
                    Polarity::Invariant => (ea.join(eb), ea.meet(eb)),
                    Polarity::Covariant => (ea, eb),
                    Polarity::Contravariant => (eb, ea),
                };
                if required.at_most(allowed) {
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
            (EffectTerm::Var(v), EffectTerm::Known(e))
            | (EffectTerm::Known(e), EffectTerm::Var(v)) => {
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
            bound: TyVarBound::Any,
        });
        TyTerm::Var(id)
    }

    /// Take a snapshot for later rollback.
    pub fn snapshot(&self) -> SolverSnapshot {
        SolverSnapshot {
            ty_bounds: self.ty_bounds.clone(),
            effect_vars: self.effect_vars.clone(),
            len_vars: self.len_vars.clone(),
            identity_vars: self.identity_vars.clone(),
        }
    }

    /// Rollback to a snapshot: restore entire state.
    pub fn rollback(&mut self, snap: SolverSnapshot) {
        self.ty_bounds = snap.ty_bounds;
        self.effect_vars = snap.effect_vars;
        self.len_vars = snap.len_vars;
        self.identity_vars = snap.identity_vars;
    }

    // -- Resolution --------------------------------------------------

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
                    TypeBound::Resolved { ty: inner, bound } => {
                        let frozen = self.freeze_ty(inner)?;
                        if bound.admits(&frozen) {
                            Ok(frozen)
                        } else {
                            Err(FreezeError::OutOfBound {
                                var: root,
                                ty: frozen,
                                bound: bound.clone(),
                            })
                        }
                    }
                    TypeBound::Unresolved { .. } => Err(FreezeError::UnresolvedType(root)),
                    TypeBound::Forward(_) => unreachable!("find_ty_root should resolve forwards"),
                }
            },
            &mut |id: IdentityVarId| {
                self.freeze_identity(&IdentityTerm::Var(id))
                    .map(IdentityTerm::Known)
            },
            &mut |id: EffectVarId| Ok(EffectTerm::Known(self.freeze_effect(&EffectTerm::Var(id)))),
            &mut |id: LenVarId| match self.resolve_len(&LenTerm::Var(id)) {
                LenTerm::Known(n) => Ok(LenTerm::Known(n)),
                LenTerm::Var(root) => Err(FreezeError::UnresolvedLen(root)),
            },
        )
    }

    // -- Resolve -----------------------------------------------------

    /// Shallow-resolve: follow Var chains but don't recurse into structure.
    pub fn shallow_resolve_ty(&self, ty: &InferTy) -> InferTy {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, .. } => self.shallow_resolve_ty(inner),
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
                    TypeBound::Resolved { ty: inner, .. } => self.resolve_ty(inner),
                    _ => TyTerm::Var(root),
                }
            },
            &mut |id: IdentityVarId| self.resolve_identity(&IdentityTerm::Var(id)),
            &mut |id: EffectVarId| self.resolve_effect(&EffectTerm::Var(id)),
            &mut |id: LenVarId| self.resolve_len(&LenTerm::Var(id)),
        )
    }

    // -- Occurs check ------------------------------------------------

    /// Returns true if the type variable `id` appears in `ty`.
    fn occurs_in(&self, id: TypeBoundId, ty: &InferTy) -> bool {
        match ty {
            TyTerm::Var(other_id) => {
                let root = self.find_ty_root(*other_id);
                if root == id {
                    return true;
                }
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, .. } => self.occurs_in(id, inner),
                    _ => false,
                }
            }
            TyTerm::Array(inner, _) | TyTerm::Option(inner) | TyTerm::Ref(inner) => {
                self.occurs_in(id, inner)
            }
            TyTerm::Tuple(elems) => elems.iter().any(|e| self.occurs_in(id, e)),
            TyTerm::Object(fields) => fields.values().any(|v| self.occurs_in(id, v)),
            TyTerm::Fn {
                params,
                ret,
                captures,
                ..
            } => {
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

    /// Find the leaf Var in an InferTy - the deepest Var in a binding chain
    /// that is bound to a concrete (non-Var) type. Returns None if not a Var.
    pub fn find_leaf_var(&self, ty: &InferTy) -> Option<TypeBoundId> {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, .. } => match inner {
                        TyTerm::Var(_) => self.find_leaf_var(inner),
                        _ => Some(root),
                    },
                    _ => Some(root),
                }
            }
            _ => None,
        }
    }

    // -- Bind helpers ------------------------------------------------

    /// Bind a type variable to a resolved InferTy.
    pub fn bind_ty(&mut self, id: TypeBoundId, ty: InferTy) {
        let root = self.find_ty_root(id);
        let bound = self.bound_of(root);
        self.ty_bounds[root.0 as usize] = TypeBound::Resolved { ty, bound };
    }

    /// The declared bound on a variable's root.
    fn bound_of(&self, root: TypeBoundId) -> TyVarBound {
        match &self.ty_bounds[root.0 as usize] {
            TypeBound::Resolved { bound, .. } | TypeBound::Unresolved { bound } => bound.clone(),
            TypeBound::Forward(_) => unreachable!("bound_of takes a root"),
        }
    }

    /// Bind a type variable to forward to another.
    fn forward_ty(&mut self, from: TypeBoundId, to: TypeBoundId) {
        let from_root = self.find_ty_root(from);
        let to_root = self.find_ty_root(to);
        if from_root != to_root {
            self.ty_bounds[from_root.0 as usize] = TypeBound::Forward(to_root);
        }
    }

    // -- Type unification --------------------------------------------

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
                TyTerm::UserDefined {
                    id: id_a,
                    type_args: ta_args,
                    effect_args: ea_args,
                    identity_args: ia_args,
                },
                TyTerm::UserDefined {
                    id: id_b,
                    type_args: tb_args,
                    effect_args: eb_args,
                    identity_args: ib_args,
                },
            ) if id_a == id_b => {
                assert_eq!(ta_args.len(), tb_args.len());
                assert_eq!(ea_args.len(), eb_args.len());
                assert_eq!(ia_args.len(), ib_args.len());
                let snap = self.snapshot();
                let type_ok = ta_args
                    .iter()
                    .zip(tb_args.iter())
                    .all(|(a, b)| self.unify_ty(a, b, Polarity::Invariant, registry).is_ok());
                let effect_ok = ea_args
                    .iter()
                    .zip(eb_args.iter())
                    .all(|(a, b)| self.unify_effect(a, b, pol).is_ok());
                let identity_ok = ia_args
                    .iter()
                    .zip(ib_args.iter())
                    .all(|(a, b)| self.unify_identity(a, b).is_ok());
                if type_ok && effect_ok && identity_ok {
                    Ok(None)
                } else {
                    self.rollback(snap);
                    self.lub_or_err_infer(pol, orig_a, orig_b, &a, &b, registry)
                }
            }

            (
                TyTerm::Enum {
                    name: na,
                    variants: va,
                },
                TyTerm::Enum {
                    name: nb,
                    variants: vb,
                },
            ) => {
                if na != nb {
                    return Err((a, b));
                }
                for (tag, payload_a) in va {
                    if let Some(payload_b) = vb.get(tag) {
                        match (payload_a, payload_b) {
                            (None, None) => {}
                            (Some(ty_a), Some(ty_b)) => {
                                self.unify_ty(ty_a, ty_b, pol, registry)?;
                            }
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
                    let merged_ty = TyTerm::Enum {
                        name: *na,
                        variants: merged,
                    };
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
                        let root1 = self.find_ty_root(*id);
                        let root2 = self.find_ty_root(*id2);
                        let Some(merged) = self.bound_of(root1).meet(&self.bound_of(root2)) else {
                            return Err((a.clone(), b.clone()));
                        };
                        self.ty_bounds[root2.0 as usize] = TypeBound::Unresolved { bound: merged };
                        self.forward_ty(root1, root2);
                    }
                    TyTerm::Error(_) => {
                        self.bind_ty(*id, other.clone());
                    }
                    _ => {
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

            (TyTerm::Array(ea, la), TyTerm::Array(eb, lb)) => {
                if self.unify_len(la, lb).is_err() {
                    return Err((a.clone(), b.clone()));
                }
                self.unify_ty(ea, eb, Polarity::Invariant, registry)
            }

            (TyTerm::Option(a), TyTerm::Option(b)) => {
                self.unify_ty(a, b, Polarity::Invariant, registry)
            }

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
                                let lub = self
                                    .try_lub_infer(ty_a, ty_b, registry)
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
                TyTerm::Fn {
                    params: pa,
                    ret: ra,
                    effect: ea,
                    ..
                },
                TyTerm::Fn {
                    params: pb,
                    ret: rb,
                    effect: eb,
                    ..
                },
            ) => {
                if pa.len() != pb.len() || pa.iter().zip(pb).any(|(x, y)| x.mode != y.mode) {
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

    // -- LUB ---------------------------------------------------------

    fn try_lub_infer(
        &mut self,
        a: &InferTy,
        b: &InferTy,
        registry: &TypeRegistry,
    ) -> Option<InferTy> {
        match (a, b) {
            (
                TyTerm::Fn {
                    params: pa,
                    ret: ra,
                    effect: ea,
                    ..
                },
                TyTerm::Fn {
                    params: pb,
                    ret: rb,
                    effect: eb,
                    ..
                },
            ) => {
                if pa.len() != pb.len() {
                    return None;
                }
                for (a, b) in pa.iter().zip(pb.iter()) {
                    self.unify_ty(&a.ty, &b.ty, Polarity::Invariant, registry)
                        .ok()?;
                }
                self.unify_ty(ra, rb, Polarity::Invariant, registry).ok()?;
                let effect = self.lub_effect(ea, eb).ok()?;
                Some(TyTerm::Fn {
                    params: pa
                        .iter()
                        .map(|p| p.retyped(self.resolve_ty(&p.ty)))
                        .collect(),
                    ret: Box::new(self.resolve_ty(ra)),
                    captures: vec![],
                    effect,
                })
            }
            (
                TyTerm::UserDefined {
                    id: id_a,
                    type_args: ta_a,
                    effect_args: ea_a,
                    identity_args: ia_a,
                },
                TyTerm::UserDefined {
                    id: id_b,
                    type_args: ta_b,
                    effect_args: ea_b,
                    identity_args: ia_b,
                },
            ) if id_a == id_b => {
                assert_eq!(ta_a.len(), ta_b.len());
                let snap = self.snapshot();
                let type_args_ok = ta_a
                    .iter()
                    .zip(ta_b.iter())
                    .all(|(a, b)| self.unify_ty(a, b, Polarity::Invariant, registry).is_ok());
                let identity_ok = ia_a
                    .iter()
                    .zip(ia_b.iter())
                    .all(|(a, b)| self.unify_identity(a, b).is_ok());
                if type_args_ok && identity_ok {
                    let mut effect_args = Vec::with_capacity(ea_a.len());
                    for (a, b) in ea_a.iter().zip(ea_b.iter()) {
                        effect_args.push(self.lub_effect(a, b).ok()?);
                    }
                    Some(TyTerm::UserDefined {
                        id: *id_a,
                        type_args: ta_a.iter().map(|t| self.resolve_ty(t)).collect(),
                        effect_args,
                        identity_args: ia_a.iter().map(|i| self.resolve_identity(i)).collect(),
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
        if rules.is_empty() {
            return None;
        }

        for rule in &rules {
            let snap = self.snapshot();
            let (inst_from_a, inst_to_a) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let a_ok = self
                .unify_ty(a, &inst_from_a, Polarity::Invariant, registry)
                .is_ok();
            let target_a = if a_ok {
                Some(self.resolve_ty(&inst_to_a))
            } else {
                None
            };
            self.rollback(snap);
            let target_a = target_a?;

            let snap = self.snapshot();
            let (inst_from_b, inst_to_b) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let b_ok = self
                .unify_ty(b, &inst_from_b, Polarity::Invariant, registry)
                .is_ok();
            let target_b = if b_ok {
                Some(self.resolve_ty(&inst_to_b))
            } else {
                None
            };
            self.rollback(snap);
            let target_b = target_b?;

            let snap = self.snapshot();
            if self
                .unify_ty(&target_a, &target_b, Polarity::Covariant, registry)
                .is_ok()
            {
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
        let lub = self
            .try_lub_infer(a, b, registry)
            .ok_or_else(|| (a.clone(), b.clone()))?;
        if let Some(leaf) = leaf_a {
            self.bind_ty(leaf, lub.clone());
        }
        if let Some(leaf) = leaf_b {
            self.bind_ty(leaf, lub);
        }
        Ok(None)
    }

    // -- Coercion ----------------------------------------------------

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
        if rules.is_empty() {
            return Err(());
        }

        let mut matched_idx = None;
        for (i, rule) in rules.iter().enumerate() {
            let snap = self.snapshot();
            let (inst_from, inst_to) = self.instantiate_poly_pair(&rule.from, &rule.to);
            let ok = self
                .unify_ty(sub, &inst_from, Polarity::Invariant, registry)
                .is_ok()
                && self
                    .unify_ty(&inst_to, sup, Polarity::Invariant, registry)
                    .is_ok();
            self.rollback(snap);
            if ok {
                if matched_idx.is_some() {
                    return Err(());
                }
                matched_idx = Some(i);
            }
        }

        let idx = matched_idx.ok_or(())?;
        let rule = &rules[idx];
        let fn_ref = rule.fn_ref;
        let (inst_from, inst_to) = self.instantiate_poly_pair(&rule.from, &rule.to);
        self.unify_ty(sub, &inst_from, Polarity::Invariant, registry)
            .map_err(|_| ())?;
        self.unify_ty(&inst_to, sup, Polarity::Invariant, registry)
            .map_err(|_| ())?;
        Ok(Some(fn_ref))
    }

    // -- Instantiate -------------------------------------------------

    /// Instantiate a polymorphic InferTy: replace all Var
    /// and Identity with fresh values.
    pub fn instantiate_infer(&mut self, ty: &InferTy) -> InferTy {
        let mut var_map: FxHashMap<TypeBoundId, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityVarId, IdentityVarId> = FxHashMap::default();
        let mut effect_map: FxHashMap<EffectVarId, EffectVarId> = FxHashMap::default();
        self.instantiate_infer_inner(ty, &mut var_map, &mut fresh_map, &mut effect_map)
    }

    /// Instantiate two InferTy sharing the same variable mappings.
    pub fn instantiate_pair_infer(&mut self, a: &InferTy, b: &InferTy) -> (InferTy, InferTy) {
        let mut var_map: FxHashMap<TypeBoundId, TypeBoundId> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityVarId, IdentityVarId> = FxHashMap::default();
        let mut effect_map: FxHashMap<EffectVarId, EffectVarId> = FxHashMap::default();
        let ia = self.instantiate_infer_inner(a, &mut var_map, &mut fresh_map, &mut effect_map);
        let ib = self.instantiate_infer_inner(b, &mut var_map, &mut fresh_map, &mut effect_map);
        (ia, ib)
    }

    fn instantiate_infer_inner(
        &mut self,
        ty: &InferTy,
        var_map: &mut FxHashMap<TypeBoundId, TypeBoundId>,
        fresh_map: &mut FxHashMap<IdentityVarId, IdentityVarId>,
        effect_map: &mut FxHashMap<EffectVarId, EffectVarId>,
    ) -> InferTy {
        match ty {
            TyTerm::Var(id) => {
                let root = self.find_ty_root(*id);
                // Clone bound data before mutating to avoid borrow conflict.
                let bound = self.ty_bounds[root.0 as usize].clone();
                match bound {
                    TypeBound::Resolved { ty: inner, .. } => {
                        self.instantiate_infer_inner(&inner, var_map, fresh_map, effect_map)
                    }
                    TypeBound::Unresolved { bound } => {
                        let new_id = *var_map.entry(root).or_insert_with(|| {
                            let fresh = TypeBoundId(self.ty_bounds.len() as u32);
                            self.ty_bounds.push(TypeBound::Unresolved { bound });
                            fresh
                        });
                        TyTerm::Var(new_id)
                    }
                    TypeBound::Forward(_) => unreachable!(),
                }
            }
            TyTerm::Array(inner, len) => TyTerm::Array(
                Box::new(self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map)),
                self.resolve_len(len),
            ),
            TyTerm::Option(inner) => TyTerm::Option(Box::new(
                self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map),
            )),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems
                    .iter()
                    .map(|e| self.instantiate_infer_inner(e, var_map, fresh_map, effect_map))
                    .collect(),
            ),
            TyTerm::Object(fields) => TyTerm::Object(
                fields
                    .iter()
                    .map(|(k, v)| {
                        (
                            *k,
                            self.instantiate_infer_inner(v, var_map, fresh_map, effect_map),
                        )
                    })
                    .collect(),
            ),
            TyTerm::Fn {
                params,
                ret,
                captures,
                effect,
            } => TyTerm::Fn {
                params: params
                    .iter()
                    .map(|p| {
                        p.retyped(
                            self.instantiate_infer_inner(&p.ty, var_map, fresh_map, effect_map),
                        )
                    })
                    .collect(),
                ret: Box::new(self.instantiate_infer_inner(ret, var_map, fresh_map, effect_map)),
                captures: captures
                    .iter()
                    .map(|c| self.instantiate_infer_inner(c, var_map, fresh_map, effect_map))
                    .collect(),
                effect: self.instantiate_effect_inner(effect, effect_map),
            },
            TyTerm::Enum { name, variants } => {
                TyTerm::Enum {
                    name: *name,
                    variants: variants
                        .iter()
                        .map(|(tag, payload)| {
                            (
                                *tag,
                                payload.as_ref().map(|ty| {
                                    Box::new(self.instantiate_infer_inner(
                                        ty, var_map, fresh_map, effect_map,
                                    ))
                                }),
                            )
                        })
                        .collect(),
                }
            }
            TyTerm::UserDefined {
                id,
                type_args,
                effect_args,
                identity_args,
            } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| self.instantiate_infer_inner(t, var_map, fresh_map, effect_map))
                    .collect(),
                identity_args: identity_args
                    .iter()
                    .map(|i| match self.resolve_identity(i) {
                        IdentityTerm::Known(id) => IdentityTerm::Known(id),
                        IdentityTerm::Var(root) => {
                            IdentityTerm::Var(*fresh_map.entry(root).or_insert_with(|| {
                                Self::alloc_identity_var(&mut self.identity_vars)
                            }))
                        }
                    })
                    .collect(),
                effect_args: effect_args
                    .iter()
                    .map(|e| self.instantiate_effect_inner(e, effect_map))
                    .collect(),
            },
            TyTerm::Handle(inner) => TyTerm::Handle(Box::new(
                self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map),
            )),
            TyTerm::Ref(inner) => TyTerm::Ref(Box::new(
                self.instantiate_infer_inner(inner, var_map, fresh_map, effect_map),
            )),
            other => other.clone(),
        }
    }
}

/// Error when freezing an InferTy that still contains unresolved variables.
#[derive(Debug)]
pub enum FreezeError {
    UnresolvedType(TypeBoundId),
    UnresolvedLen(LenVarId),
    UnresolvedIdentity(IdentityVarId),
    /// The variable resolved to a type its declaration does not admit.
    OutOfBound {
        var: TypeBoundId,
        ty: Ty,
        bound: TyVarBound,
    },
}

/// A scheme instantiated into the solver: the type, and the variables that
/// carry a declared bound so the caller can verify them where it chose to.
pub struct Instantiated {
    pub ty: InferTy,
    pub bounded: Vec<TypeBoundId>,
}

// -- Poly -> Infer instantiation (in Solver) --------------------------

impl Solver<'_> {
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
        self.instantiate_scheme(&Scheme::unbounded(ty.clone())).ty
    }

    pub fn instantiate_scheme(&mut self, scheme: &Scheme) -> Instantiated {
        let mut bounded: Vec<TypeBoundId> = Vec::new();
        let ty = self.instantiate_with(&scheme.ty, |var, fresh| {
            let bound = scheme.bound_of(var);
            if bound != TyVarBound::Any {
                bounded.push(fresh);
            }
            bound
        });
        Instantiated { ty, bounded }
    }

    fn instantiate_with(
        &mut self,
        ty: &PolyTy,
        mut bound_for: impl FnMut(u32, TypeBoundId) -> TyVarBound,
    ) -> InferTy {
        let mut var_map: FxHashMap<u32, TypeBoundId> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, EffectVarId> = FxHashMap::default();
        let mut len_map: FxHashMap<u32, LenVarId> = FxHashMap::default();
        let ty_bounds = &mut self.ty_bounds;
        let effect_vars = &mut self.effect_vars;
        let len_vars = &mut self.len_vars;
        let identity_vars = &mut self.identity_vars;
        let sources = &mut *self.sources;
        let from_params = Self::identity_vars_bound_by_params(ty);
        let mut identity_map: FxHashMap<u32, IdentityTerm<Infer>> = FxHashMap::default();
        ty.map(
            &mut |id: u32| {
                let bound_id = *var_map.entry(id).or_insert_with(|| {
                    let fresh = TypeBoundId(ty_bounds.len() as u32);
                    ty_bounds.push(TypeBound::Unresolved {
                        bound: TyVarBound::Any,
                    });
                    let bound = bound_for(id, fresh);
                    ty_bounds[fresh.0 as usize] = TypeBound::Unresolved { bound };
                    fresh
                });
                TyTerm::Var(bound_id)
            },
            &mut |id: u32| {
                *identity_map.entry(id).or_insert_with(|| {
                    if from_params.contains(&id) {
                        IdentityTerm::Var(Self::alloc_identity_var(identity_vars))
                    } else {
                        IdentityTerm::Known(sources.next())
                    }
                })
            },
            &mut |id: u32| {
                let var = *effect_map
                    .entry(id)
                    .or_insert_with(|| Self::alloc_effect_var(effect_vars));
                EffectTerm::Var(var)
            },
            &mut |id: u32| {
                let var = *len_map
                    .entry(id)
                    .or_insert_with(|| Self::alloc_len_var(len_vars));
                LenTerm::Var(var)
            },
        )
    }

    /// Instantiate two PolyTy templates sharing the same variable mappings.
    /// Used for CastRule where `from` and `to` share placeholder variables.
    pub fn instantiate_poly_pair(&mut self, a: &PolyTy, b: &PolyTy) -> (InferTy, InferTy) {
        let mut var_map: FxHashMap<u32, TypeBoundId> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, EffectVarId> = FxHashMap::default();
        let mut len_map: FxHashMap<u32, LenVarId> = FxHashMap::default();
        let ty_bounds = &mut self.ty_bounds;
        let effect_vars = &mut self.effect_vars;
        let len_vars = &mut self.len_vars;
        let identity_vars = &mut self.identity_vars;
        let sources = &mut *self.sources;
        let mut identity_map: FxHashMap<u32, IdentityTerm<Infer>> = FxHashMap::default();
        let mut on_len = |id: u32| {
            let var = *len_map
                .entry(id)
                .or_insert_with(|| Self::alloc_len_var(len_vars));
            LenTerm::Var(var)
        };
        let mut on_effect = |id: u32| {
            let var = *effect_map
                .entry(id)
                .or_insert_with(|| Self::alloc_effect_var(effect_vars));
            EffectTerm::Var(var)
        };
        let mut on_var = |id: u32| {
            let bound_id = *var_map.entry(id).or_insert_with(|| {
                let fresh = TypeBoundId(ty_bounds.len() as u32);
                ty_bounds.push(TypeBound::Unresolved {
                    bound: TyVarBound::Any,
                });
                fresh
            });
            TyTerm::Var(bound_id)
        };
        let mut on_identity_a = |id: u32| {
            *identity_map
                .entry(id)
                .or_insert_with(|| IdentityTerm::Var(Self::alloc_identity_var(identity_vars)))
        };
        let ia = a.map(&mut on_var, &mut on_identity_a, &mut on_effect, &mut on_len);
        let mut on_identity_b = |id: u32| {
            *identity_map
                .entry(id)
                .or_insert_with(|| IdentityTerm::Known(sources.next()))
        };
        let ib = b.map(&mut on_var, &mut on_identity_b, &mut on_effect, &mut on_len);
        (ia, ib)
    }
}
