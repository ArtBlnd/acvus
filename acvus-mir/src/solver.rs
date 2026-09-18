//! The type solver: equality now, decisions later (scratchpad/tobe/solver.md).
//!
//! `Terms` is the union-find over type, effect, length, identity, and
//! representation variables, and `join` on it is the one unification: the
//! join of the type lattice, taken where it is asked. A `Decision` is a
//! position with more than one admissible answer; it shrinks as the terms
//! resolve and settles when one answer remains. `Solver` owns both, the
//! registry that says which slots specialize and which conversions exist,
//! and the compilation's sources. A body is checked (`fresh`, `unify`,
//! `decide`), then solved once (`solve`), then frozen.

use std::convert::Infallible;

use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::types::QualifiedRef;
use crate::ir::Intrinsic;
use crate::ty::{
    CastRule, Concrete, Effect, EffectConflict, EffectTerm, EffectVarId, IdentityId, IdentityTerm,
    IdentityVarId, Infer, InferTy, IntTy, LenTerm, LenVarId, Mutability, ParamTerm, Phase, Poly,
    PolyTy, Repr, ReprVarId, Scheme, Task, Ty, TyTerm, TyVarBound, TypeArg, TypeBoundId,
    TypeRegistry, could_match_pattern, matches_pattern,
};

// -- Variable states --------------------------------------------------

/// State of a type variable. The declared bound travels with the variable
/// and is verified when it freezes (`TyVarBound::admits`).
#[derive(Debug, Clone)]
pub enum TypeBound {
    Resolved { ty: InferTy, bound: TyVarBound },
    Unresolved { bound: TyVarBound },
    Forward(TypeBoundId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EffectBound {
    /// `lower <= var <= upper` on the reissue chain (RFC-0017).
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
            upper: Effect::TOP,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LenBound {
    Unbound,
    Bound(usize),
    Forward(LenVarId),
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdentityBound {
    /// Not yet tied to a source. Solved, it becomes a source of its own.
    Unbound,
    Bound(IdentityId),
    Forward(IdentityVarId),
}

/// A representation variable's state (hash-types.md, R2): open until a
/// decision fixes it; solved open, it is `Uniform`.
#[derive(Debug, Clone, PartialEq)]
pub enum ReprBound {
    Unbound(ReprOwner),
    Bound(Repr<Concrete>),
    Forward(ReprVarId),
}

/// What binds an open representation variable. A signature's `ρ` is bound
/// by a decision — the instance decision of the signature it belongs to, a
/// conversion decision writing its answer — or by the default at `solve`;
/// a value flowing into its slot leaves it open, and the flow is the
/// conversion decision at that argument (R4). A local `ρ`, minted by the
/// checker for a shape it named, is bound by any join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReprOwner {
    Signature,
    Local,
}

/// The sources of one compilation. A source number names one identity
/// for every solver of the compilation, so a frozen type may pass from
/// one solver to another and still name the source it was frozen with.
#[derive(Debug, Default)]
pub struct Sources(acvus_utils::LocalFactory<IdentityId>);

impl Sources {
    pub fn new() -> Self {
        Self(acvus_utils::LocalFactory::new())
    }

    pub fn next(&mut self) -> IdentityId {
        self.0.next()
    }
}

// -- Mismatch ----------------------------------------------------------

/// A conflict whose one disagreement is the task names the task, so the
/// checker can report RFC-0046's refusal instead of two printed types.
fn task_reason(conflict: &EffectConflict) -> MismatchReason {
    match conflict.required.task > conflict.allowed.task {
        true => MismatchReason::TaskTooHigh {
            required: conflict.allowed.task,
            found: conflict.required.task,
        },
        false => MismatchReason::NoJoin,
    }
}

/// Two types that did not join: what was asked and what arrived, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub expected: InferTy,
    pub got: InferTy,
    pub reason: MismatchReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchReason {
    /// The lattice has no join for the two.
    NoJoin,
    /// The join is a union neither side is, and no variable names either
    /// side, so the union has no home.
    UnionWithoutHome,
    /// A slot's representation is a signature's open variable, which a
    /// flow does not bind: the decision that owns it answers later.
    ReprOpen(ReprVarId),
    /// The value's task is above the one the position fixes (RFC-0046).
    TaskTooHigh { required: Task, found: Task },
}

/// How two effects are related by a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectRelation {
    /// One effect.
    Equal,
    /// The first is at most the second on the reissue chain.
    AtMost,
}

/// Where a join is taken: at a value, where `!` is the bottom, or as the
/// argument of a type constructor, where every type is invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Position {
    Value,
    Argument,
}

/// How a structural type's members relate to its values: an `Object` is a
/// product whose fields are initialized per path; an `Enum` is a sum, so a
/// type lacking a variant the value may carry does not hold it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Structure {
    Product,
    Sum,
}

/// What the join is for: a value flowing into a type, a pattern tested
/// against a source, or a decision writing its answer. A pattern may name
/// fewer members than its source has without anything recording the
/// union; a value may not lack what the type it flows into has. A
/// signature's representation variable is bound only by a decision's
/// join (`ReprOwner`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinKind {
    Flow,
    Pattern,
    Decision,
}

// -- Terms ---------------------------------------------------------------

/// The union-find over every kind of variable, and the join on it.
#[derive(Debug, Clone)]
struct Terms {
    ty_bounds: Vec<TypeBound>,
    effect_vars: Vec<EffectBound>,
    /// `a <= b` (RFC-0017).
    effect_below: Vec<(EffectVarId, EffectVarId)>,
    len_vars: Vec<LenBound>,
    identity_vars: Vec<IdentityBound>,
    repr_vars: Vec<ReprBound>,
}

impl Terms {
    fn new() -> Self {
        Self {
            ty_bounds: Vec::new(),
            effect_vars: Vec::new(),
            effect_below: Vec::new(),
            len_vars: Vec::new(),
            identity_vars: Vec::new(),
            repr_vars: Vec::new(),
        }
    }

    // -- Type variables ----------------------------------------------

    fn alloc_ty_var(&mut self, bound: TyVarBound) -> TypeBoundId {
        alloc_ty_var(&mut self.ty_bounds, bound)
    }

    fn find_ty_root(&self, id: TypeBoundId) -> TypeBoundId {
        match &self.ty_bounds[id.0 as usize] {
            TypeBound::Forward(next) => self.find_ty_root(*next),
            _ => id,
        }
    }

    fn bound_of(&self, root: TypeBoundId) -> TyVarBound {
        match &self.ty_bounds[root.0 as usize] {
            TypeBound::Resolved { bound, .. } | TypeBound::Unresolved { bound } => bound.clone(),
            TypeBound::Forward(_) => unreachable!("bound_of takes a root"),
        }
    }

    fn bind_ty(&mut self, id: TypeBoundId, ty: InferTy) -> Result<(), Cyclic> {
        let root = self.find_ty_root(id);
        if self.occurs_in(root, &ty) {
            return Err(Cyclic);
        }
        let bound = self.bound_of(root);
        self.ty_bounds[root.0 as usize] = TypeBound::Resolved { ty, bound };
        Ok(())
    }

    fn forward_ty(&mut self, from: TypeBoundId, to: TypeBoundId) -> Result<(), Cyclic> {
        let from_root = self.find_ty_root(from);
        let to_root = self.find_ty_root(to);
        if from_root == to_root {
            return Ok(());
        }
        if self.occurs_in(from_root, &TyTerm::Var(to_root)) {
            return Err(Cyclic);
        }
        self.ty_bounds[from_root.0 as usize] = TypeBound::Forward(to_root);
        Ok(())
    }

    /// The root variable a type is a name of, if it is a variable.
    fn root_var(&self, ty: &InferTy) -> Option<TypeBoundId> {
        match ty {
            TyTerm::Var(id) => Some(self.find_ty_root(*id)),
            _ => None,
        }
    }

    fn shallow_resolve_ty(&self, ty: &InferTy) -> InferTy {
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

    fn resolve_ty(&self, ty: &InferTy) -> InferTy {
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
            &mut |id: ReprVarId| self.resolve_repr(Repr::Var(id)),
        )
    }

    fn occurs_in(&self, id: TypeBoundId, ty: &InferTy) -> bool {
        match ty {
            TyTerm::Var(other) => {
                let root = self.find_ty_root(*other);
                if root == id {
                    return true;
                }
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, .. } => self.occurs_in(id, inner),
                    _ => false,
                }
            }
            TyTerm::Array(inner, _) | TyTerm::Option(inner) | TyTerm::Handle(inner) => {
                self.occurs_in(id, inner)
            }
            TyTerm::Ref(_, inner) => self.occurs_in(id, &inner.ty),
            TyTerm::Result(ok, err) => self.occurs_in(id, ok) || self.occurs_in(id, err),
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
                type_args.iter().any(|t| self.occurs_in(id, &t.ty))
            }
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::String
            | TyTerm::Bool
            | TyTerm::Unit
            | TyTerm::Never
            | TyTerm::Order
            | TyTerm::Error(_) => false,
        }
    }

    // -- Identity variables ------------------------------------------

    fn alloc_identity_var(&mut self) -> IdentityVarId {
        alloc_identity_var(&mut self.identity_vars)
    }

    fn find_identity_root(&self, id: IdentityVarId) -> IdentityVarId {
        match &self.identity_vars[id.0 as usize] {
            IdentityBound::Forward(next) => self.find_identity_root(*next),
            _ => id,
        }
    }

    fn resolve_identity(&self, term: &IdentityTerm<Infer>) -> IdentityTerm<Infer> {
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
    fn unify_identity(
        &mut self,
        a: &IdentityTerm<Infer>,
        b: &IdentityTerm<Infer>,
    ) -> Result<(), (IdentityId, IdentityId)> {
        match (self.resolve_identity(a), self.resolve_identity(b)) {
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

    // -- Length variables --------------------------------------------

    fn alloc_len_var(&mut self) -> LenVarId {
        alloc_len_var(&mut self.len_vars)
    }

    fn find_len_root(&self, id: LenVarId) -> LenVarId {
        match &self.len_vars[id.0 as usize] {
            LenBound::Forward(next) => self.find_len_root(*next),
            _ => id,
        }
    }

    fn resolve_len(&self, term: &LenTerm<Infer>) -> LenTerm<Infer> {
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

    fn unify_len(&mut self, a: &LenTerm<Infer>, b: &LenTerm<Infer>) -> Result<(), (usize, usize)> {
        match (self.resolve_len(a), self.resolve_len(b)) {
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

    fn alloc_effect_var(&mut self) -> EffectVarId {
        alloc_effect_var(&mut self.effect_vars)
    }

    fn find_effect_root(&self, id: EffectVarId) -> EffectVarId {
        match &self.effect_vars[id.0 as usize] {
            EffectBound::Forward(next) => self.find_effect_root(*next),
            _ => id,
        }
    }

    fn resolve_effect(&self, term: &EffectTerm<Infer>) -> EffectTerm<Infer> {
        match term {
            EffectTerm::Known(e) => EffectTerm::Known(e.clone()),
            EffectTerm::Var(id) => {
                let root = self.find_effect_root(*id);
                match &self.effect_vars[root.0 as usize] {
                    EffectBound::Bound(e) => EffectTerm::Known(e.clone()),
                    EffectBound::Range { .. } => EffectTerm::Var(root),
                    EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
                }
            }
        }
    }

    /// The interval of a root: its own, with the lower raised by every
    /// variable placed below it.
    fn range_of(&self, root: EffectVarId) -> Interval {
        let Interval {
            lower: own_lower,
            upper,
        } = self.own_range(root);
        let mut lower = own_lower;
        let mut seen: Vec<EffectVarId> = vec![root];
        let mut stack: Vec<EffectVarId> = vec![root];
        while let Some(above) = stack.pop() {
            for (b, a) in &self.effect_below {
                if self.find_effect_root(*a) != above {
                    continue;
                }
                let b = self.find_effect_root(*b);
                if seen.contains(&b) {
                    continue;
                }
                seen.push(b);
                stack.push(b);
                lower = lower.join(&self.own_range(b).lower);
            }
        }
        Interval { lower, upper }
    }

    fn own_range(&self, root: EffectVarId) -> Interval {
        match &self.effect_vars[root.0 as usize] {
            EffectBound::Range { lower, upper } => Interval {
                lower: lower.clone(),
                upper: upper.clone(),
            },
            EffectBound::Bound(e) => Interval {
                lower: e.clone(),
                upper: e.clone(),
            },
            EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
        }
    }

    /// An open effect is the least element of its interval: the join of
    /// what the body requires (RFC-0014, solver.md R3).
    fn freeze_effect(&self, term: &EffectTerm<Infer>) -> Effect {
        match self.resolve_effect(term) {
            EffectTerm::Known(e) => e,
            EffectTerm::Var(root) => self.range_of(root).lower,
        }
    }

    fn bind_effect(&mut self, id: EffectVarId, effect: Effect) -> Result<(), EffectConflict> {
        let root = self.find_effect_root(id);
        let Interval { lower, upper } = self.range_of(root);
        if !lower.at_most(&effect) || !lower.touches_at_most(&effect) {
            return Err(EffectConflict {
                required: lower,
                allowed: effect,
            });
        }
        if !effect.at_most(&upper) {
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
        let Interval { lower, upper } = self.range_of(root);
        let lower = lower.join(&effect);
        if !lower.at_most(&upper) {
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
        let Interval { lower, upper } = self.range_of(root);
        let upper = upper.meet(&effect);
        if !lower.at_most(&upper) {
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
        let from_range = self.range_of(from_root);
        let to_range = self.range_of(to_root);
        let lower = from_range.lower.join(&to_range.lower);
        let upper = from_range.upper.meet(&to_range.upper);
        if !lower.at_most(&upper) {
            return Err(EffectConflict {
                required: lower,
                allowed: upper,
            });
        }
        self.effect_vars[to_root.0 as usize] = EffectBound::Range { lower, upper };
        self.effect_vars[from_root.0 as usize] = EffectBound::Forward(to_root);
        Ok(())
    }

    fn effect_at_least(
        &mut self,
        below: EffectVarId,
        above: EffectVarId,
    ) -> Result<(), EffectConflict> {
        let below = self.find_effect_root(below);
        let above = self.find_effect_root(above);
        if below == above {
            return Ok(());
        }
        let lower = self.range_of(below).lower;
        let upper = self.range_of(above).upper;
        if !lower.at_most(&upper) {
            return Err(EffectConflict {
                required: lower,
                allowed: upper,
            });
        }
        self.effect_below.push((below, above));
        Ok(())
    }

    fn unify_effect(
        &mut self,
        a: &EffectTerm<Infer>,
        b: &EffectTerm<Infer>,
        relation: EffectRelation,
    ) -> Result<(), EffectConflict> {
        match (self.resolve_effect(a), self.resolve_effect(b)) {
            (EffectTerm::Known(ea), EffectTerm::Known(eb)) => {
                let (required, allowed, same_contexts) = match relation {
                    EffectRelation::Equal => (
                        ea.join(&eb),
                        ea.meet(&eb),
                        ea.reads == eb.reads && ea.writes == eb.writes,
                    ),
                    EffectRelation::AtMost => (ea, eb, true),
                };
                if same_contexts && required.at_most(&allowed) {
                    Ok(())
                } else {
                    Err(EffectConflict { required, allowed })
                }
            }
            (EffectTerm::Var(va), EffectTerm::Known(eb)) => match relation {
                EffectRelation::Equal => self.bind_effect(va, eb),
                EffectRelation::AtMost => self.lower_upper(va, eb),
            },
            (EffectTerm::Known(ea), EffectTerm::Var(vb)) => match relation {
                EffectRelation::Equal => self.bind_effect(vb, ea),
                EffectRelation::AtMost => self.raise_lower(vb, ea),
            },
            (EffectTerm::Var(va), EffectTerm::Var(vb)) => match relation {
                EffectRelation::Equal => self.forward_effect(va, vb),
                EffectRelation::AtMost => self.effect_at_least(va, vb),
            },
        }
    }

    // -- Representation variables ------------------------------------

    fn alloc_repr_var(&mut self, owner: ReprOwner) -> ReprVarId {
        alloc_repr_var(&mut self.repr_vars, owner)
    }

    fn find_repr_root(&self, id: ReprVarId) -> ReprVarId {
        match &self.repr_vars[id.0 as usize] {
            ReprBound::Forward(next) => self.find_repr_root(*next),
            _ => id,
        }
    }

    fn resolve_repr(&self, repr: Repr<Infer>) -> Repr<Infer> {
        let Repr::Var(id) = repr else {
            return repr;
        };
        let root = self.find_repr_root(id);
        match &self.repr_vars[root.0 as usize] {
            ReprBound::Bound(fixed) => lift_repr(*fixed),
            ReprBound::Unbound(_) => Repr::Var(root),
            ReprBound::Forward(_) => unreachable!("find_repr_root resolves forwards"),
        }
    }

    fn repr_owner(&self, root: ReprVarId) -> ReprOwner {
        match &self.repr_vars[root.0 as usize] {
            ReprBound::Unbound(owner) => *owner,
            ReprBound::Bound(_) | ReprBound::Forward(_) => {
                unreachable!("repr_owner takes an open root")
            }
        }
    }

    /// Two representations at one specializing position (hash-types.md,
    /// R3). Two fixed ones must agree. An open one takes the other when
    /// its owner lets this join bind it: a local variable at any join, a
    /// signature's variable at a decision's join. A local variable meeting
    /// a signature's forwards to it, so the local name follows the
    /// decision; two signatures' variables are two decisions' names, and
    /// only a decision's join — a conversion answered identity — makes
    /// them one.
    fn unify_repr(
        &mut self,
        a: Repr<Infer>,
        b: Repr<Infer>,
        kind: JoinKind,
    ) -> Result<(), MismatchReason> {
        let binds = |terms: &Self, v: ReprVarId| {
            kind == JoinKind::Decision || terms.repr_owner(v) == ReprOwner::Local
        };
        match (self.resolve_repr(a), self.resolve_repr(b)) {
            (Repr::Var(x), Repr::Var(y)) if x == y => Ok(()),
            (Repr::Var(x), Repr::Var(y)) => {
                let (from, to) = match (self.repr_owner(x), self.repr_owner(y)) {
                    (ReprOwner::Local, _) => (x, y),
                    (ReprOwner::Signature, ReprOwner::Local) => (y, x),
                    (ReprOwner::Signature, ReprOwner::Signature) => {
                        if kind != JoinKind::Decision {
                            return Err(MismatchReason::ReprOpen(x));
                        }
                        (x, y)
                    }
                };
                self.repr_vars[from.0 as usize] = ReprBound::Forward(to);
                Ok(())
            }
            (Repr::Var(v), fixed) | (fixed, Repr::Var(v)) => {
                if !binds(self, v) {
                    return Err(MismatchReason::ReprOpen(v));
                }
                let fixed = match fixed {
                    Repr::Uniform => Repr::Uniform,
                    Repr::Specialized => Repr::Specialized,
                    Repr::Var(_) => unreachable!("two variables are matched above"),
                };
                self.repr_vars[v.0 as usize] = ReprBound::Bound(fixed);
                Ok(())
            }
            (Repr::Uniform, Repr::Uniform) | (Repr::Specialized, Repr::Specialized) => Ok(()),
            (Repr::Uniform, Repr::Specialized) | (Repr::Specialized, Repr::Uniform) => {
                Err(MismatchReason::NoJoin)
            }
        }
    }

    // -- Join ----------------------------------------------------------

    /// The join of two types (solver.md, R1), `a` the value and `b` the
    /// type it flows into. Written to the root variable of a side that is a
    /// variable; `Mismatch` where no join exists. A closure's effect is the
    /// one directional component: the value's is at most what the position
    /// allows (RFC-0017). At the top of a decision's own join `a` is not a
    /// value flowing into `b` but a call settling on the signature it took,
    /// and there the effect runs both ways: the caller runs what the callee
    /// does (RFC-0046).
    fn join(
        &mut self,
        a: &InferTy,
        b: &InferTy,
        position: Position,
        kind: JoinKind,
        registry: &TypeRegistry,
    ) -> Result<(), Mismatch> {
        let a_root = self.root_var(a);
        let b_root = self.root_var(b);
        if let (Some(x), Some(y)) = (a_root, b_root)
            && x == y
        {
            return Ok(());
        }
        let no_join = |terms: &Self| Mismatch {
            expected: terms.resolve_ty(a),
            got: terms.resolve_ty(b),
            reason: MismatchReason::NoJoin,
        };
        if position == Position::Value {
            let a_never = matches!(self.shallow_resolve_ty(a), TyTerm::Never);
            let b_never = matches!(self.shallow_resolve_ty(b), TyTerm::Never);
            match (a_never, b_never) {
                (true, true) => return Ok(()),
                (true, false) => {
                    return self
                        .yield_bottom(a_root, b, b_root)
                        .map_err(|Cyclic| no_join(self));
                }
                (false, true) => {
                    return self
                        .yield_bottom(b_root, a, a_root)
                        .map_err(|Cyclic| no_join(self));
                }
                (false, false) => {}
            }
        }
        let unbound = |terms: &Self, root: Option<TypeBoundId>| {
            root.filter(|r| matches!(terms.ty_bounds[r.0 as usize], TypeBound::Unresolved { .. }))
        };
        if let Some(v) = unbound(self, a_root) {
            return self
                .take_other(v, b, b_root)
                .map_err(|NoJoin| no_join(self));
        }
        if let Some(v) = unbound(self, b_root) {
            return self
                .take_other(v, a, a_root)
                .map_err(|NoJoin| no_join(self));
        }

        let ra = self.shallow_resolve_ty(a);
        let rb = self.shallow_resolve_ty(b);
        let mismatch_for = |terms: &Self, reason: MismatchReason| Mismatch {
            expected: terms.resolve_ty(&ra),
            got: terms.resolve_ty(&rb),
            reason,
        };
        let mismatch = |terms: &Self| mismatch_for(terms, MismatchReason::NoJoin);

        match (&ra, &rb) {
            (TyTerm::Error(_), _) | (_, TyTerm::Error(_)) => Ok(()),
            (TyTerm::Var(_), _) | (_, TyTerm::Var(_)) => {
                unreachable!("an unbound variable took the other side above")
            }

            (TyTerm::Never, TyTerm::Never) => Ok(()),

            (TyTerm::Int(ka), TyTerm::Int(kb)) if ka == kb => Ok(()),
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Order, TyTerm::Order) => Ok(()),

            (TyTerm::Object(fa), TyTerm::Object(fb)) => {
                for (key, ty_a) in fa {
                    if let Some(ty_b) = fb.get(key) {
                        self.join(ty_a, ty_b, Position::Argument, kind, registry)?;
                    }
                }
                let a_only = fa.keys().any(|k| !fb.contains_key(k));
                let b_only = fb.keys().any(|k| !fa.contains_key(k));
                if !a_only && !b_only {
                    return Ok(());
                }
                let mut merged: FxHashMap<Astr, InferTy> = fa.clone();
                for (k, v) in fb {
                    merged.entry(*k).or_insert_with(|| v.clone());
                }
                self.write_union(
                    UnionSide {
                        root: a_root,
                        grows: b_only,
                    },
                    UnionSide {
                        root: b_root,
                        grows: a_only,
                    },
                    TyTerm::Object(merged),
                    Structure::Product,
                    kind,
                    mismatch,
                )
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
                    return Err(mismatch(self));
                }
                for (tag, payload_a) in va {
                    if let Some(payload_b) = vb.get(tag) {
                        match (payload_a, payload_b) {
                            (None, None) => {}
                            (Some(ty_a), Some(ty_b)) => {
                                self.join(ty_a, ty_b, Position::Argument, kind, registry)?;
                            }
                            (None, Some(_)) | (Some(_), None) => return Err(mismatch(self)),
                        }
                    }
                }
                let a_only = va.keys().any(|k| !vb.contains_key(k));
                let b_only = vb.keys().any(|k| !va.contains_key(k));
                if !a_only && !b_only {
                    return Ok(());
                }
                let mut merged: FxHashMap<Astr, Option<Box<InferTy>>> = va.clone();
                for (tag, payload) in vb {
                    merged.entry(*tag).or_insert_with(|| payload.clone());
                }
                self.write_union(
                    UnionSide {
                        root: a_root,
                        grows: b_only,
                    },
                    UnionSide {
                        root: b_root,
                        grows: a_only,
                    },
                    TyTerm::Enum {
                        name: *na,
                        variants: merged,
                    },
                    Structure::Sum,
                    kind,
                    mismatch,
                )
            }

            (TyTerm::Array(ea, la), TyTerm::Array(eb, lb)) => {
                if self.unify_len(la, lb).is_err() {
                    return Err(mismatch(self));
                }
                self.join(ea, eb, Position::Argument, kind, registry)
            }
            (TyTerm::Option(ia), TyTerm::Option(ib)) | (TyTerm::Handle(ia), TyTerm::Handle(ib)) => {
                self.join(ia, ib, Position::Argument, kind, registry)
            }
            (TyTerm::Result(ta, ea), TyTerm::Result(tb, eb)) => {
                self.join(ta, tb, Position::Argument, kind, registry)?;
                self.join(ea, eb, Position::Argument, kind, registry)
            }
            (TyTerm::Tuple(ea), TyTerm::Tuple(eb)) => {
                if ea.len() != eb.len() {
                    return Err(mismatch(self));
                }
                for (ta, tb) in ea.iter().zip(eb.iter()) {
                    self.join(ta, tb, Position::Argument, kind, registry)?;
                }
                Ok(())
            }
            (TyTerm::Ref(ma, ia), TyTerm::Ref(mb, ib)) => {
                if ma != mb {
                    return Err(mismatch(self));
                }
                self.join(&ia.ty, &ib.ty, Position::Argument, kind, registry)?;
                self.unify_repr(ia.repr, ib.repr, kind)
                    .map_err(|reason| mismatch_for(self, reason))
            }
            (
                TyTerm::Fn {
                    params: pa,
                    ret: ret_a,
                    effect: ea,
                    ..
                },
                TyTerm::Fn {
                    params: pb,
                    ret: ret_b,
                    effect: eb,
                    ..
                },
            ) => {
                if pa.len() != pb.len() {
                    return Err(mismatch(self));
                }
                for (ta, tb) in pa.iter().zip(pb.iter()) {
                    self.join(&ta.ty, &tb.ty, Position::Argument, kind, registry)?;
                }
                self.join(ret_a, ret_b, Position::Argument, kind, registry)?;
                if let Err(conflict) = self.unify_effect(ea, eb, EffectRelation::AtMost) {
                    return Err(mismatch_for(self, task_reason(&conflict)));
                }
                Ok(())
            }
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
            ) => {
                if id_a != id_b {
                    return Err(mismatch(self));
                }
                assert_eq!(ta_args.len(), tb_args.len());
                assert_eq!(ea_args.len(), eb_args.len());
                assert_eq!(ia_args.len(), ib_args.len());
                for (x, y) in ta_args.iter().zip(tb_args.iter()) {
                    self.join(&x.ty, &y.ty, Position::Argument, kind, registry)?;
                }
                for (index, (x, y)) in ta_args.iter().zip(tb_args.iter()).enumerate() {
                    if registry.specializes(*id_a, index) {
                        self.unify_repr(x.repr, y.repr, kind)
                            .map_err(|reason| mismatch_for(self, reason))?;
                    } else {
                        debug_assert!(
                            matches!(x.repr, Repr::Uniform) && matches!(y.repr, Repr::Uniform),
                            "a slot that does not specialize is uniform by instantiation"
                        );
                    }
                }
                for (x, y) in ea_args.iter().zip(eb_args.iter()) {
                    if self.unify_effect(x, y, EffectRelation::Equal).is_err() {
                        return Err(mismatch(self));
                    }
                }
                for (x, y) in ia_args.iter().zip(ib_args.iter()) {
                    if self.unify_identity(x, y).is_err() {
                        return Err(mismatch(self));
                    }
                }
                Ok(())
            }

            _ => Err(mismatch(self)),
        }
    }

    /// A variable nothing has bound takes the other side: another
    /// variable by forwarding to its root, so the two stay one name for
    /// whatever the root comes to hold; a term by binding.
    fn take_other(
        &mut self,
        var: TypeBoundId,
        other: &InferTy,
        other_root: Option<TypeBoundId>,
    ) -> Result<(), NoJoin> {
        let Some(root) = other_root else {
            let term = self.shallow_resolve_ty(other);
            if matches!(term, TyTerm::Error(_)) {
                return Ok(());
            }
            if integer_bound_refuses(&self.bound_of(var), &term) {
                return Err(NoJoin);
            }
            return self.bind_ty(var, term).map_err(|Cyclic| NoJoin);
        };
        let merged = self
            .bound_of(var)
            .meet(&self.bound_of(root))
            .ok_or(NoJoin)?;
        let held = self.shallow_resolve_ty(&TyTerm::Var(root));
        if integer_bound_refuses(&merged, &held) {
            return Err(NoJoin);
        }
        self.forward_ty(var, root).map_err(|Cyclic| NoJoin)?;
        if let TypeBound::Unresolved { bound } = &mut self.ty_bounds[root.0 as usize] {
            *bound = merged;
        }
        Ok(())
    }

    /// `! ⊔ T = T` at a value position (RFC-0038): the variable that named
    /// `!` now names `T`, by forwarding to `T`'s root where it has one so
    /// the two stay one name, else by binding to `T`.
    fn yield_bottom(
        &mut self,
        never_root: Option<TypeBoundId>,
        other: &InferTy,
        other_root: Option<TypeBoundId>,
    ) -> Result<(), Cyclic> {
        let Some(root) = never_root else {
            return Ok(());
        };
        match other_root {
            Some(target) if self.find_ty_root(target) != root => {
                let bound = self.bound_of(root);
                let held = std::mem::replace(
                    &mut self.ty_bounds[root.0 as usize],
                    TypeBound::Unresolved { bound },
                );
                let forwarded = self.forward_ty(root, target);
                if forwarded.is_err() {
                    self.ty_bounds[root.0 as usize] = held;
                }
                forwarded
            }
            Some(_) => Ok(()),
            None => self.bind_ty(root, other.clone()),
        }
    }

    /// The union of two structural types: a side that lacks members the
    /// other has grows into the union, which its variable names now. A
    /// side that must grow and has no variable cannot: whether that is a
    /// mismatch follows the structure. An object's fields are initialized
    /// per path and the definite-assignment check (validate/init_check.rs)
    /// rejects a read of one that is not, so a value lacking a field is
    /// not the checker's error. A ground enum type lacking a variant the
    /// value may carry has no such check, and is. A ground pattern source
    /// lacking a member the pattern names is.
    fn write_union(
        &mut self,
        a: UnionSide,
        b: UnionSide,
        union: InferTy,
        structure: Structure,
        kind: JoinKind,
        mismatch: impl Fn(&Self) -> Mismatch,
    ) -> Result<(), Mismatch> {
        let must_grow_without_home = |side: &UnionSide| side.grows && side.root.is_none();
        let unsound = match (kind, structure) {
            (JoinKind::Pattern, _) => must_grow_without_home(&a),
            (JoinKind::Flow | JoinKind::Decision, Structure::Product) => false,
            (JoinKind::Flow | JoinKind::Decision, Structure::Sum) => must_grow_without_home(&b),
        };
        if unsound {
            return Err(Mismatch {
                reason: MismatchReason::UnionWithoutHome,
                ..mismatch(self)
            });
        }
        if let (Some(ra), true) = (a.root, a.grows) {
            self.bind_ty(ra, union.clone())
                .map_err(|Cyclic| mismatch(self))?;
        }
        if let (Some(rb), true) = (b.root, b.grows)
            && !a
                .root
                .is_some_and(|ra| self.find_ty_root(ra) == self.find_ty_root(rb))
        {
            self.bind_ty(rb, union).map_err(|Cyclic| mismatch(self))?;
        }
        Ok(())
    }
}

/// `lower <= var <= upper` on the reissue chain (RFC-0017).
#[derive(Debug, Clone, PartialEq)]
struct Interval {
    lower: Effect,
    upper: Effect,
}

impl Terms {
    /// An instance's signature with every placeholder a fresh variable and
    /// every identity open, for joining with a call type whose sources are
    /// minted. An open representation is `Uniform`: a concrete instance
    /// names its representations, and a variable is a generic body's
    /// (hash-types.md R3).
    fn instantiate_open(&mut self, ty: &PolyTy, registry: &TypeRegistry) -> InferTy {
        let mut maps = PolyMaps::default();
        let Terms {
            ty_bounds,
            effect_vars,
            len_vars,
            identity_vars,
            ..
        } = self;
        let instance = ty.map(
            &mut |id: u32| {
                TyTerm::Var(
                    *maps
                        .ty
                        .entry(id)
                        .or_insert_with(|| alloc_ty_var(ty_bounds, TyVarBound::Any)),
                )
            },
            &mut |id: u32| {
                *maps
                    .identity
                    .entry(id)
                    .or_insert_with(|| IdentityTerm::Var(alloc_identity_var(identity_vars)))
            },
            &mut |id: u32| {
                EffectTerm::Var(
                    *maps
                        .effect
                        .entry(id)
                        .or_insert_with(|| alloc_effect_var(effect_vars)),
                )
            },
            &mut |id: u32| {
                LenTerm::Var(
                    *maps
                        .len
                        .entry(id)
                        .or_insert_with(|| alloc_len_var(len_vars)),
                )
            },
            &mut |_: u32| Repr::Uniform,
        );
        uniform_slots(instance, registry)
    }

    /// Whether the call type would join an instance's signature, on a copy
    /// of the terms: the solver's own lattice says what "could match" means.
    fn would_take(&self, call: &InferTy, signature: &PolyTy, registry: &TypeRegistry) -> bool {
        let mut trial = self.clone();
        let instance = trial.instantiate_open(signature, registry);
        trial
            .join(
                call,
                &instance,
                Position::Value,
                JoinKind::Decision,
                registry,
            )
            .is_ok()
    }
}

/// Two types with no join, before the types themselves are attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NoJoin;

/// A store refused because the variable it writes occurs in what would be
/// written: `T = &&T` has no finite term, and every reader of a bound walks
/// it to the leaves. The refusing caller reports it as its own no-join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cyclic;

/// One side of a structural union: the variable that names it, if any,
/// and whether it lacks members the other side has.
#[derive(Debug, Clone, Copy)]
struct UnionSide {
    root: Option<TypeBoundId>,
    grows: bool,
}

/// A `OneOf` bound is passed over here, and that is a decision rather
/// than an omission. Its shapes are patterns a term still open can grow
/// into, so judging one against a half-built term would refuse types the
/// solve was going to reach. Those bounds are verified instead by
/// `freeze_ty_with`, once the term is whole, at the sites `typeck`
/// registered. Integer widths are ground and have no such growth, so they
/// are decided the moment a variable would come to name a term.
fn integer_bound_refuses(bound: &TyVarBound, term: &InferTy) -> bool {
    let TyVarBound::Integer { signed, among } = bound else {
        return false;
    };
    if matches!(term, TyTerm::Var(_) | TyTerm::Error(_)) {
        return false;
    }
    !matches!(term, TyTerm::Int(k) if among.contains(k) && (!*signed || k.signed()))
}

fn alloc_ty_var(ty_bounds: &mut Vec<TypeBound>, bound: TyVarBound) -> TypeBoundId {
    let id = TypeBoundId(ty_bounds.len() as u32);
    ty_bounds.push(TypeBound::Unresolved { bound });
    id
}

fn alloc_identity_var(identity_vars: &mut Vec<IdentityBound>) -> IdentityVarId {
    let id = IdentityVarId(identity_vars.len() as u32);
    identity_vars.push(IdentityBound::Unbound);
    id
}

fn alloc_len_var(len_vars: &mut Vec<LenBound>) -> LenVarId {
    let id = LenVarId(len_vars.len() as u32);
    len_vars.push(LenBound::Unbound);
    id
}

fn alloc_effect_var(effect_vars: &mut Vec<EffectBound>) -> EffectVarId {
    let id = EffectVarId(effect_vars.len() as u32);
    effect_vars.push(EffectBound::free());
    id
}

fn alloc_repr_var(repr_vars: &mut Vec<ReprBound>, owner: ReprOwner) -> ReprVarId {
    let id = ReprVarId(repr_vars.len() as u32);
    repr_vars.push(ReprBound::Unbound(owner));
    id
}

fn lift_repr(repr: Repr<Concrete>) -> Repr<Infer> {
    repr.map(&mut |v: Infallible| match v {})
}

// -- Decisions -----------------------------------------------------------

/// Index into `Solver::decisions`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecisionId(pub u32);

/// An instance a call may run: one of the Extern function's handlers, by
/// its number (RFC-0040), or an instruction of the language (RFC-0020).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceKind {
    Extern(usize),
    Intrinsic(Intrinsic),
}

/// A concrete instance an instance decision may still settle on.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub instance: InstanceKind,
    pub ty: PolyTy,
    pub admits: Task,
}

/// The generic instance of a function: the uniform one, whose signature a
/// call type with a specialized slot does not match (hash-types.md, R3).
#[derive(Debug, Clone)]
pub struct GenericInstance {
    pub instance: usize,
    pub ty: PolyTy,
}

/// A position with more than one admissible answer (solver.md, R2).
#[derive(Debug, Clone)]
pub enum Decision {
    /// Which instance of an Extern function a call runs (RFC-0027,
    /// RFC-0040).
    Instance {
        call: InferTy,
        candidates: Vec<Candidate>,
        generic: Option<GenericInstance>,
    },
    /// Which conversion takes an argument to its parameter (RFC-0023):
    /// identity where the two join, else a declared cast.
    Conversion { from: InferTy, to: InferTy },
    /// Which signature a call of an overloaded bare name is a call of
    /// (RFC-0043). The checker opened one conversion decision at each
    /// argument some option takes by conversion, whose `to` is the call's
    /// parameter there, so that decision settles once this one has.
    Signature {
        name: Astr,
        call: CallShape,
        options: Vec<SignatureOption>,
        body_effect: EffectTerm<Infer>,
    },
    /// Since no `&&T` exists (RFC-0029), what a reference to a place names
    /// depends on whether the place itself holds a reference. The checker
    /// opens this only where the head of `of` is still a variable; a known
    /// head it answers on the spot, through `Solver::lend`.
    Lend {
        of: InferTy,
        referent: InferTy,
        mutability: Mutability,
        kind: LendKind,
    },
    /// How a pattern reads its scrutinee (RFC-0024). The checker opens
    /// this only where the head of `scrutinee` is still a variable; a
    /// known head it answers on the spot, through `Solver::match_mode`.
    /// The pattern is checked against `referent` — what the scrutinee
    /// names under `Through`, the scrutinee itself under `Value` — so
    /// settling the mode is what joins the two and what gives every
    /// binding under the pattern its type.
    Match {
        scrutinee: InferTy,
        referent: InferTy,
        bindings: Vec<MatchBinding>,
    },
}

/// One name a pattern binds while its mode is still open: `binding` is
/// the name's type, `part` the part of the referent the name stands for.
#[derive(Debug, Clone)]
pub struct MatchBinding {
    pub binding: InferTy,
    pub part: InferTy,
}

/// Rust's default binding modes (RFC-0024): a non-reference pattern
/// against a reference scrutinee reads through the reference, and every
/// name it binds is a reference into the referent; against anything else
/// it reads the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    Value,
    Through,
}

/// What a scrutinee's known head gives the pattern: the mode it is read
/// in, and the type the pattern's referent is then joined with.
#[derive(Debug, Clone)]
pub struct MatchReads {
    pub mode: MatchMode,
    pub names: InferTy,
}

#[derive(Debug, Clone)]
pub enum MatchOutcome {
    HeadOpen,
    Reads(MatchReads),
}

/// A reference found under the place is reborrowed (RFC-0029) for a
/// `Borrow` and refused (RFC-0018) for a `Capture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LendKind {
    Borrow,
    Capture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lend {
    Reference,
    Reborrow,
}

#[derive(Debug, Clone)]
pub enum LendOutcome {
    HeadOpen,
    Names {
        referent: InferTy,
        lend: Lend,
    },
    Refused {
        refusal: LendRefusal,
        referent: InferTy,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LendRefusal {
    MutableBorrowOfShared,
    ReferenceCaptured,
}

/// RFC-0030.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverMode {
    Lent(Mutability),
    Value,
}

/// RFC-0043.
#[derive(Debug, Clone)]
pub enum SignatureCandidate {
    Named { qref: QualifiedRef, scheme: Scheme },
    Local { ty: InferTy },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureName {
    Named(QualifiedRef),
    Local,
}

impl SignatureCandidate {
    pub fn name(&self) -> SignatureName {
        match self {
            Self::Named { qref, .. } => SignatureName::Named(*qref),
            Self::Local { .. } => SignatureName::Local,
        }
    }

    fn local_params(ty: &InferTy) -> Option<&[ParamTerm<Infer>]> {
        match ty {
            TyTerm::Fn { params, .. } => Some(params),
            _ => None,
        }
    }

    /// RFC-0043.
    pub fn arity(&self) -> Option<usize> {
        match self {
            Self::Named { scheme, .. } => Some(scheme.params().len()),
            Self::Local { ty } => Self::local_params(ty).map(<[_]>::len),
        }
    }

    pub fn param_name(&self, index: usize) -> Option<Astr> {
        match self {
            Self::Named { scheme, .. } => scheme.params().get(index).map(|p| p.name),
            Self::Local { ty } => Self::local_params(ty)?.get(index).map(|p| p.name),
        }
    }

    /// A binding's parameter is an inference type, not a shape a scheme
    /// could name, so it bounds the call's parameter by nothing.
    pub fn param_bound(&self, index: usize) -> TyVarBound {
        match self {
            Self::Named { scheme, .. } => match scheme.params().get(index) {
                Some(param) => scheme.param_bound(&param.ty),
                None => TyVarBound::Any,
            },
            Self::Local { .. } => TyVarBound::Any,
        }
    }
}

/// A candidate of a signature decision, with how it takes the call's
/// arguments (RFC-0043).
#[derive(Debug, Clone)]
pub struct SignatureOption {
    pub candidate: SignatureCandidate,
    /// The arguments this candidate takes through one declared
    /// conversion; every other argument it takes directly.
    pub converted: Vec<ConvertedArgument>,
}

impl SignatureOption {
    pub fn taking_every_argument_directly(candidate: SignatureCandidate) -> Self {
        Self {
            candidate,
            converted: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConvertedArgument {
    pub index: usize,
    pub ty: InferTy,
}

/// A call of an overloaded bare name as the call site wrote it (RFC-0043).
/// There is no effect here, and that is a decision rather than an omission.
/// A call runs what the instance it settles on runs, so before the
/// signature decision settles there is no effect to hold; a field holding
/// one would be a second term for one effect, and the settle joins the two
/// under RFC-0017's demotion, which leaves the call's free to freeze below
/// the instance's. That is how `find` and `last` over a suspending pipeline
/// took their asynchronous instance while the call froze to `Pure`,
/// carrying neither the task nor the contexts.
/// `a_calls_effect_is_the_term_of_its_instance` is the test that fails when
/// an effect of the call's own comes back.
#[derive(Debug, Clone)]
pub struct CallShape {
    pub params: Vec<ParamTerm<Infer>>,
    pub ret: InferTy,
}

impl CallShape {
    fn at_effect(&self, effect: EffectTerm<Infer>) -> InferTy {
        TyTerm::Fn {
            params: self.params.clone(),
            ret: Box::new(self.ret.clone()),
            captures: vec![],
            effect,
        }
    }
}

/// `body_effect` travels with the call for the same reason the effect is
/// absent from `CallShape`: the effect the enclosing body must be raised by
/// is the instance's, unknown until the decision settles, so the decision
/// raises it and the checker cannot.
pub struct UndecidedCall {
    pub name: Astr,
    pub call: CallShape,
    pub options: Vec<SignatureOption>,
    pub body_effect: EffectTerm<Infer>,
}

/// How one candidate takes one argument (RFC-0043).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Direct,
    Converted,
    Refused,
}

impl Decision {
    pub fn conversion(from: &InferTy, to: &InferTy) -> Self {
        Decision::Conversion {
            from: from.clone(),
            to: to.clone(),
        }
    }
}

/// A decision's settled answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Instance(InstanceKind),
    Conversion(Conversion),
    Signature {
        settled: SettledSignature,
        callee_ty: InferTy,
    },
    Lend(Lend),
    Match(MatchMode),
}

/// `bounded` is verified by the checker when the body freezes, as an
/// `Instantiated`'s is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettledSignature {
    Named {
        qref: QualifiedRef,
        instance: Option<InstanceChoice>,
        bounded: Vec<TypeBoundId>,
    },
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    Identity,
    Cast(QualifiedRef),
    /// The lowering does not cast the reference value: it runs `cast` on
    /// the place the argument borrows before the call and `back` on the
    /// same place after it, so the callee's writes land in the caller's
    /// storage (scratchpad/tobe/reference-cast-in-place.md).
    ThroughRef {
        mutability: Mutability,
        cast: QualifiedRef,
        back: QualifiedRef,
    },
}

/// Why a decision did not settle.
#[derive(Debug, Clone)]
pub enum Unsettled {
    /// No instance's signature matches the call type.
    NoInstance {
        decision: DecisionId,
        call: InferTy,
    },
    /// The one remaining instance's signature does not join the call type.
    InstanceMismatch {
        decision: DecisionId,
        expected: InferTy,
        got: InferTy,
    },
    /// Several instances still match a call type nothing narrows further.
    AmbiguousInstance {
        decision: DecisionId,
        call: InferTy,
    },
    /// A function value's task is above the one the position fixes
    /// (RFC-0046), which no conversion can lower.
    TaskTooHigh {
        decision: DecisionId,
        required: Task,
        found: Task,
    },
    /// The two do not join and no declared conversion takes one to the other.
    NoConversion {
        decision: DecisionId,
        from: InferTy,
        to: InferTy,
    },
    /// More than one declared conversion takes one to the other.
    AmbiguousConversion {
        decision: DecisionId,
        from: InferTy,
        to: InferTy,
        rules: Vec<QualifiedRef>,
    },
    /// One declared conversion could take one to the other, but a type
    /// inside them stayed open to the end.
    ConversionOpen {
        decision: DecisionId,
        from: InferTy,
        to: InferTy,
        rule: QualifiedRef,
    },
    /// The answer is `Conversion::ThroughRef`, but the argument at the
    /// decision's site is a reference value and not a borrow of a place,
    /// so there is no storage to hold the callee's representation. The
    /// solver does not see expressions; the checker raises this at the site
    /// when it reads the answer.
    ConversionNeedsPlace {
        decision: DecisionId,
        from: InferTy,
        to: InferTy,
    },
    /// RFC-0043.
    NoSignature {
        decision: DecisionId,
        name: Astr,
        call: CallShape,
    },
    AmbiguousSignature {
        decision: DecisionId,
        name: Astr,
        candidates: Vec<SignatureName>,
    },
    /// The settled instance's effect is above what the body the call runs
    /// in allows (RFC-0046).
    EffectExceeded {
        decision: DecisionId,
        conflict: EffectConflict,
    },
    /// RFC-0018.
    ReferenceCaptured {
        decision: DecisionId,
    },
    MutableBorrowOfShared {
        decision: DecisionId,
    },
    /// The place's settled type contradicts what the borrow's users
    /// required of it.
    LendMismatch {
        decision: DecisionId,
        expected: InferTy,
        got: InferTy,
    },
    /// The scrutinee's settled head contradicts what the pattern or one of
    /// its bindings required: the pattern's shape against the referent, or
    /// a binding read as a reference where the mode settled on a value.
    MatchMismatch {
        decision: DecisionId,
        expected: InferTy,
        got: InferTy,
    },
}

impl Unsettled {
    pub fn decision(&self) -> DecisionId {
        match self {
            Unsettled::NoInstance { decision, .. }
            | Unsettled::InstanceMismatch { decision, .. }
            | Unsettled::AmbiguousInstance { decision, .. }
            | Unsettled::TaskTooHigh { decision, .. }
            | Unsettled::NoConversion { decision, .. }
            | Unsettled::AmbiguousConversion { decision, .. }
            | Unsettled::ConversionOpen { decision, .. }
            | Unsettled::ConversionNeedsPlace { decision, .. }
            | Unsettled::NoSignature { decision, .. }
            | Unsettled::AmbiguousSignature { decision, .. }
            | Unsettled::EffectExceeded { decision, .. }
            | Unsettled::ReferenceCaptured { decision }
            | Unsettled::MutableBorrowOfShared { decision }
            | Unsettled::LendMismatch { decision, .. }
            | Unsettled::MatchMismatch { decision, .. } => *decision,
        }
    }
}

#[derive(Debug, Clone)]
enum DecisionState {
    Open,
    Settled(Answer),
    Failed,
}

#[derive(Debug, Clone)]
struct DecisionSlot {
    decision: Decision,
    state: DecisionState,
}

/// What one settlement round did to a decision.
enum Progress {
    Unchanged,
    Narrowed,
    Settled(Answer),
    Failed(Unsettled),
}

// -- Solver ---------------------------------------------------------------

pub struct Solver<'src> {
    terms: Terms,
    decisions: Vec<DecisionSlot>,
    /// Mints a new source for every identity a declaration introduces;
    /// lent by the compilation for this solver's lifetime.
    sources: &'src mut Sources,
    /// The user-defined types and cast rules of the compilation: which
    /// slots specialize (hash-types.md R1) and which conversions exist.
    registry: &'src TypeRegistry,
}

impl<'src> Solver<'src> {
    pub fn new(sources: &'src mut Sources, registry: &'src TypeRegistry) -> Self {
        Self {
            terms: Terms::new(),
            decisions: Vec::new(),
            sources,
            registry,
        }
    }

    pub fn registry(&self) -> &'src TypeRegistry {
        self.registry
    }

    // -- Fresh variables ---------------------------------------------

    pub fn fresh_ty_var(&mut self) -> InferTy {
        self.fresh_var_with(TyVarBound::Any)
    }

    pub fn fresh_var_with(&mut self, bound: TyVarBound) -> InferTy {
        TyTerm::Var(self.terms.alloc_ty_var(bound))
    }

    /// A fresh variable for an integer literal: its bound is the set of
    /// widths still admissible (RFC-0037), `i64` where nothing narrows it.
    pub fn fresh_int_var(&mut self) -> InferTy {
        TyTerm::Var(self.terms.alloc_ty_var(TyVarBound::Integer {
            signed: false,
            among: IntTy::ALL.to_vec(),
        }))
    }

    /// A negated integer literal is a signed one (RFC-0037): its variable
    /// loses the unsigned widths. Whether the variable was a literal's.
    pub fn require_signed(&mut self, var: TypeBoundId) -> bool {
        let root = self.terms.find_ty_root(var);
        if let TypeBound::Unresolved {
            bound: TyVarBound::Integer { signed, among },
        } = &mut self.terms.ty_bounds[root.0 as usize]
        {
            *signed = true;
            among.retain(|k| k.signed());
            return true;
        }
        false
    }

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Infer> {
        EffectTerm::Var(self.terms.alloc_effect_var())
    }

    pub fn decide_signature(&mut self, call: UndecidedCall) -> DecisionId {
        let UndecidedCall {
            name,
            call,
            options,
            body_effect,
        } = call;
        self.decide(Decision::Signature {
            name,
            call,
            options,
            body_effect,
        })
    }

    pub fn settled_callee_ty(&self, id: DecisionId) -> Option<InferTy> {
        let DecisionState::Settled(Answer::Signature { callee_ty, .. }) =
            &self.decisions[id.0 as usize].state
        else {
            return None;
        };
        Some(self.terms.resolve_ty(callee_ty))
    }

    pub fn fresh_len_var(&mut self) -> LenTerm<Infer> {
        LenTerm::Var(self.terms.alloc_len_var())
    }

    pub fn fresh_identity_var(&mut self) -> IdentityTerm<Infer> {
        IdentityTerm::Var(self.terms.alloc_identity_var())
    }

    pub fn fresh_repr_var(&mut self) -> Repr<Infer> {
        Repr::Var(self.terms.alloc_repr_var(ReprOwner::Local))
    }

    // -- Unify -------------------------------------------------------

    /// Two types that must be one type, at a value position (solver.md R1):
    /// `a` is the value, `b` the type it flows into.
    pub fn unify(&mut self, a: &InferTy, b: &InferTy) -> Result<(), Mismatch> {
        self.terms
            .join(a, b, Position::Value, JoinKind::Flow, self.registry)
    }

    /// A pattern tested against its source: the source's structural type
    /// grows by what the pattern names; a pattern naming fewer members
    /// than a ground source is within it.
    pub fn unify_pattern(&mut self, source: &InferTy, pattern: &InferTy) -> Result<(), Mismatch> {
        self.terms.join(
            source,
            pattern,
            Position::Value,
            JoinKind::Pattern,
            self.registry,
        )
    }

    /// The join a decision writes as its answer: a signature's
    /// representation variable takes the answer here (`ReprOwner`).
    fn settle_join(&mut self, a: &InferTy, b: &InferTy) -> Result<(), Mismatch> {
        self.terms
            .join(a, b, Position::Value, JoinKind::Decision, self.registry)
    }

    /// Whether `settle_join(a, b)` would succeed, on a copy of the terms.
    fn trial_settle_join(&self, a: &InferTy, b: &InferTy) -> Result<(), Mismatch> {
        let mut trial = self.terms.clone();
        trial.join(a, b, Position::Value, JoinKind::Decision, self.registry)
    }

    /// Whether `unify(a, b)` would succeed, on a copy of the terms.
    pub fn would_unify(&self, a: &InferTy, b: &InferTy) -> bool {
        let mut trial = self.terms.clone();
        trial
            .join(a, b, Position::Value, JoinKind::Flow, self.registry)
            .is_ok()
    }

    pub fn unify_effect(
        &mut self,
        a: &EffectTerm<Infer>,
        b: &EffectTerm<Infer>,
        relation: EffectRelation,
    ) -> Result<(), EffectConflict> {
        self.terms.unify_effect(a, b, relation)
    }

    pub fn unify_identity(
        &mut self,
        a: &IdentityTerm<Infer>,
        b: &IdentityTerm<Infer>,
    ) -> Result<(), (IdentityId, IdentityId)> {
        self.terms.unify_identity(a, b)
    }

    pub fn unify_len(
        &mut self,
        a: &LenTerm<Infer>,
        b: &LenTerm<Infer>,
    ) -> Result<(), (usize, usize)> {
        self.terms.unify_len(a, b)
    }

    // -- Read ----------------------------------------------------------

    pub fn resolve_ty(&self, ty: &InferTy) -> InferTy {
        self.terms.resolve_ty(ty)
    }

    pub fn shallow_resolve_ty(&self, ty: &InferTy) -> InferTy {
        self.terms.shallow_resolve_ty(ty)
    }

    pub fn resolve_effect(&self, term: &EffectTerm<Infer>) -> EffectTerm<Infer> {
        self.terms.resolve_effect(term)
    }

    pub fn resolve_len(&self, term: &LenTerm<Infer>) -> LenTerm<Infer> {
        self.terms.resolve_len(term)
    }

    pub fn resolve_identity(&self, term: &IdentityTerm<Infer>) -> IdentityTerm<Infer> {
        self.terms.resolve_identity(term)
    }

    pub fn find_ty_root(&self, id: TypeBoundId) -> TypeBoundId {
        self.terms.find_ty_root(id)
    }

    /// The declared bound a type variable's root carries.
    pub fn bound_of_var(&self, id: TypeBoundId) -> TyVarBound {
        self.terms.bound_of(self.terms.find_ty_root(id))
    }

    // -- Decide --------------------------------------------------------

    pub fn decide(&mut self, decision: Decision) -> DecisionId {
        self.decisions.push(DecisionSlot {
            decision,
            state: DecisionState::Open,
        });
        DecisionId((self.decisions.len() - 1) as u32)
    }

    /// The answer of a settled decision; `None` while it is open or after
    /// it failed.
    pub fn answer(&self, id: DecisionId) -> Option<Answer> {
        match &self.decisions[id.0 as usize].state {
            DecisionState::Settled(answer) => Some(answer.clone()),
            DecisionState::Open | DecisionState::Failed => None,
        }
    }

    // -- Settle and solve ----------------------------------------------

    /// Run every open decision against the terms until none moves. What
    /// remains open stays open: nothing is defaulted here.
    pub fn settle(&mut self) -> Vec<Unsettled> {
        let mut failures = Vec::new();
        let mut progressed = true;
        while progressed {
            progressed = false;
            for index in 0..self.decisions.len() {
                let id = DecisionId(index as u32);
                if !matches!(self.decisions[index].state, DecisionState::Open) {
                    continue;
                }
                match self.step(id) {
                    Progress::Unchanged => {}
                    Progress::Narrowed => progressed = true,
                    Progress::Settled(answer) => {
                        self.decisions[index].state = DecisionState::Settled(answer);
                        progressed = true;
                    }
                    Progress::Failed(why) => {
                        self.decisions[index].state = DecisionState::Failed;
                        failures.push(why);
                        progressed = true;
                    }
                }
            }
        }
        failures
    }

    /// Settle, then close every decision still open by its least element
    /// (solver.md R3): a width is `i64`, a representation `Uniform`, a lend
    /// a reference, an identity a source of its own; then settle again, and
    /// report every decision that neither settled nor could take a least
    /// element.
    pub fn solve(&mut self) -> Vec<Unsettled> {
        let mut failures = self.settle();
        for index in 0..self.terms.ty_bounds.len() {
            let TypeBound::Unresolved { bound } = &self.terms.ty_bounds[index] else {
                continue;
            };
            let Some(width) = bound.integer_default() else {
                continue;
            };
            self.terms.ty_bounds[index] = TypeBound::Resolved {
                ty: TyTerm::Int(width),
                bound: bound.clone(),
            };
        }
        for index in 0..self.terms.repr_vars.len() {
            if matches!(self.terms.repr_vars[index], ReprBound::Unbound(_)) {
                self.terms.repr_vars[index] = ReprBound::Bound(Repr::Uniform);
            }
        }
        // A pattern's mode closes before a lend does: a binding closed to
        // a value is what a lend of that name then lends, and a lend
        // closed first would name a referent the pattern had not yet
        // settled, which is how a `&&T` would be formed (RFC-0029).
        self.close_matches_by_least_element(&mut failures);
        failures.extend(self.settle());
        self.close_lends_by_least_element(&mut failures);
        failures.extend(self.settle());
        self.close_instances_by_task();
        failures.extend(self.settle());
        for index in 0..self.decisions.len() {
            let id = DecisionId(index as u32);
            if !matches!(self.decisions[index].state, DecisionState::Open) {
                continue;
            }
            let why = match &self.decisions[index].decision {
                Decision::Instance { call, .. } => Unsettled::AmbiguousInstance {
                    decision: id,
                    call: self.terms.resolve_ty(call),
                },
                Decision::Conversion { from, to } if self.awaits_signature(to) => continue,
                Decision::Conversion { from, to } => {
                    let rules = self.conversion_rules(from, to);
                    let from = self.terms.resolve_ty(from);
                    let to = self.terms.resolve_ty(to);
                    match rules.as_slice() {
                        [] => Unsettled::NoConversion {
                            decision: id,
                            from,
                            to,
                        },
                        [rule] => Unsettled::ConversionOpen {
                            decision: id,
                            from,
                            to,
                            rule: rule.fn_ref,
                        },
                        _ => Unsettled::AmbiguousConversion {
                            decision: id,
                            from,
                            to,
                            rules: rules.into_iter().map(|r| r.fn_ref).collect(),
                        },
                    }
                }
                Decision::Signature { name, options, .. } => Unsettled::AmbiguousSignature {
                    decision: id,
                    name: *name,
                    candidates: options.iter().map(|o| o.candidate.name()).collect(),
                },
                Decision::Lend { .. } => {
                    unreachable!("close_lends_by_least_element leaves no lend open")
                }
                Decision::Match { .. } => {
                    unreachable!("close_matches_by_least_element leaves no match open")
                }
            };
            self.decisions[index].state = DecisionState::Failed;
            failures.push(why);
        }
        for index in 0..self.terms.identity_vars.len() {
            if self.terms.identity_vars[index] == IdentityBound::Unbound {
                self.terms.identity_vars[index] = IdentityBound::Bound(self.sources.next());
            }
        }
        failures
    }

    /// A reborrow needs the place to hold a reference (RFC-0029), and a
    /// variable nothing made a reference is not one. This close is also
    /// what carries a parameter's type back to a place whose type the
    /// program states nowhere else, so an open lend is never an error.
    fn close_lends_by_least_element(&mut self, failures: &mut Vec<Unsettled>) {
        for index in 0..self.decisions.len() {
            let Decision::Lend { of, referent, .. } = &self.decisions[index].decision else {
                continue;
            };
            if !matches!(self.decisions[index].state, DecisionState::Open) {
                continue;
            }
            let of = of.clone();
            let referent = referent.clone();
            self.decisions[index].state = match self.settle_join(&of, &referent) {
                Ok(()) => DecisionState::Settled(Answer::Lend(Lend::Reference)),
                Err(Mismatch { expected, got, .. }) => {
                    failures.push(Unsettled::LendMismatch {
                        decision: DecisionId(index as u32),
                        expected,
                        got,
                    });
                    DecisionState::Failed
                }
            };
        }
    }

    fn step(&mut self, id: DecisionId) -> Progress {
        let decision = self.decisions[id.0 as usize].decision.clone();
        match decision {
            Decision::Instance {
                call,
                candidates,
                generic,
            } => self.step_instance(id, &call, candidates, generic),
            Decision::Conversion { from, to } => self.step_conversion(id, &from, &to),
            Decision::Signature {
                name,
                call,
                options,
                body_effect,
            } => self.step_signature(id, name, &call, options, &body_effect),
            Decision::Lend {
                of,
                referent,
                mutability,
                kind,
            } => self.step_lend(id, &of, &referent, mutability, kind),
            Decision::Match {
                scrutinee,
                referent,
                bindings,
            } => self.step_match(id, &scrutinee, &referent, bindings),
        }
    }

    /// How a pattern reads a scrutinee of type `of`, once its head is
    /// known. `check_pattern` reads this directly for a head it already
    /// knows, so the decision only carries the case where it does not.
    pub fn match_mode(&self, of: &InferTy) -> MatchOutcome {
        match self.terms.shallow_resolve_ty(of) {
            TyTerm::Var(_) => MatchOutcome::HeadOpen,
            TyTerm::Ref(_, inner) => MatchOutcome::Reads(MatchReads {
                mode: MatchMode::Through,
                names: inner.ty,
            }),
            head => MatchOutcome::Reads(MatchReads {
                mode: MatchMode::Value,
                names: head,
            }),
        }
    }

    fn step_match(
        &mut self,
        id: DecisionId,
        scrutinee: &InferTy,
        referent: &InferTy,
        bindings: Vec<MatchBinding>,
    ) -> Progress {
        let MatchOutcome::Reads(reads) = self.match_mode(scrutinee) else {
            return Progress::Unchanged;
        };
        let mode = reads.mode;
        match self.settle_match(referent, reads, bindings) {
            Ok(()) => Progress::Settled(Answer::Match(mode)),
            Err(Mismatch { expected, got, .. }) => Progress::Failed(Unsettled::MatchMismatch {
                decision: id,
                expected,
                got,
            }),
        }
    }

    /// The pattern was checked against `referent`, so settling its mode
    /// joins that with what the scrutinee names and gives every binding
    /// the part it stands for. A binding under `Through` is a shared
    /// reference to its part whatever the scrutinee's own mutability, as
    /// a binding under a head already known to be a reference is
    /// (`check_pattern`).
    fn settle_match(
        &mut self,
        referent: &InferTy,
        reads: MatchReads,
        bindings: Vec<MatchBinding>,
    ) -> Result<(), Mismatch> {
        self.settle_join(referent, &reads.names)?;
        for MatchBinding { binding, part } in bindings {
            let bound = match reads.mode {
                MatchMode::Value => part,
                MatchMode::Through => {
                    TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(part)))
                }
            };
            self.settle_join(&binding, &bound)?;
        }
        Ok(())
    }

    /// A pattern reads through a reference only where its scrutinee is
    /// one, and a scrutinee nothing made a reference is not one: the least
    /// element is `Value`. This close is also what carries the pattern's
    /// shape back to a scrutinee whose type the program states nowhere
    /// else, so an open match is never an error.
    fn close_matches_by_least_element(&mut self, failures: &mut Vec<Unsettled>) {
        for index in 0..self.decisions.len() {
            let Decision::Match {
                scrutinee,
                referent,
                bindings,
            } = &self.decisions[index].decision
            else {
                continue;
            };
            if !matches!(self.decisions[index].state, DecisionState::Open) {
                continue;
            }
            let reads = MatchReads {
                mode: MatchMode::Value,
                names: scrutinee.clone(),
            };
            let referent = referent.clone();
            let bindings = bindings.clone();
            self.decisions[index].state = match self.settle_match(&referent, reads, bindings) {
                Ok(()) => DecisionState::Settled(Answer::Match(MatchMode::Value)),
                Err(Mismatch { expected, got, .. }) => {
                    failures.push(Unsettled::MatchMismatch {
                        decision: DecisionId(index as u32),
                        expected,
                        got,
                    });
                    DecisionState::Failed
                }
            };
        }
    }

    /// What lending a place of type `of` yields, once its head is known.
    /// `check_borrow` reads this directly for a head it already knows, so
    /// the decision only carries the case where it does not.
    pub fn lend(&self, of: &InferTy, mutability: Mutability, kind: LendKind) -> LendOutcome {
        match self.terms.shallow_resolve_ty(of) {
            TyTerm::Var(_) => LendOutcome::HeadOpen,
            TyTerm::Ref(_, inner) if kind == LendKind::Capture => LendOutcome::Refused {
                refusal: LendRefusal::ReferenceCaptured,
                referent: inner.ty,
            },
            TyTerm::Ref(Mutability::Shared, inner) if mutability == Mutability::Mut => {
                LendOutcome::Refused {
                    refusal: LendRefusal::MutableBorrowOfShared,
                    referent: inner.ty,
                }
            }
            TyTerm::Ref(_, inner) => LendOutcome::Names {
                referent: inner.ty,
                lend: Lend::Reborrow,
            },
            head => LendOutcome::Names {
                referent: head,
                lend: Lend::Reference,
            },
        }
    }

    fn step_lend(
        &mut self,
        id: DecisionId,
        of: &InferTy,
        referent: &InferTy,
        mutability: Mutability,
        kind: LendKind,
    ) -> Progress {
        match self.lend(of, mutability, kind) {
            LendOutcome::HeadOpen => Progress::Unchanged,
            LendOutcome::Refused {
                refusal: LendRefusal::ReferenceCaptured,
                ..
            } => Progress::Failed(Unsettled::ReferenceCaptured { decision: id }),
            LendOutcome::Refused {
                refusal: LendRefusal::MutableBorrowOfShared,
                ..
            } => Progress::Failed(Unsettled::MutableBorrowOfShared { decision: id }),
            LendOutcome::Names {
                referent: names,
                lend,
            } => match self.settle_join(referent, &names) {
                Ok(()) => Progress::Settled(Answer::Lend(lend)),
                Err(Mismatch { expected, got, .. }) => Progress::Failed(Unsettled::LendMismatch {
                    decision: id,
                    expected,
                    got,
                }),
            },
        }
    }

    /// A signature decision narrows to the candidates the call would still
    /// take (RFC-0043) and settles when one remains: the call takes that
    /// instance's effect, and the function type it thereby has is joined
    /// with the instance.
    fn step_signature(
        &mut self,
        id: DecisionId,
        name: Astr,
        call: &CallShape,
        options: Vec<SignatureOption>,
        body_effect: &EffectTerm<Infer>,
    ) -> Progress {
        let remaining: Vec<SignatureOption> = options
            .iter()
            .filter(|option| self.takes_signature(call, option))
            .cloned()
            .collect();
        let narrowed = remaining.len() != options.len();
        if narrowed
            && let Decision::Signature { options, .. } = &mut self.decisions[id.0 as usize].decision
        {
            *options = remaining.clone();
        }
        match remaining.as_slice() {
            [] => Progress::Failed(Unsettled::NoSignature {
                decision: id,
                name,
                call: self.resolve_shape(call),
            }),
            [only] => {
                let (ty, settled) = match &only.candidate {
                    SignatureCandidate::Named { qref, scheme } => {
                        let Instantiated {
                            ty,
                            bounded,
                            instance,
                        } = self.instantiate_scheme(scheme);
                        (
                            ty,
                            SettledSignature::Named {
                                qref: *qref,
                                instance,
                                bounded,
                            },
                        )
                    }
                    SignatureCandidate::Local { ty } => (ty.clone(), SettledSignature::Local),
                };
                let effect = self.effect_of_instance(&ty);
                if let Err(conflict) =
                    self.terms
                        .unify_effect(&effect, body_effect, EffectRelation::AtMost)
                {
                    return Progress::Failed(Unsettled::EffectExceeded {
                        decision: id,
                        conflict,
                    });
                }
                match self.settle_join(&call.at_effect(effect), &ty) {
                    Ok(()) => Progress::Settled(Answer::Signature {
                        settled,
                        callee_ty: ty,
                    }),
                    Err(Mismatch { expected, got, .. }) => {
                        Progress::Failed(Unsettled::InstanceMismatch {
                            decision: id,
                            expected,
                            got,
                        })
                    }
                }
            }
            _ if narrowed => Progress::Narrowed,
            _ => Progress::Unchanged,
        }
    }

    fn resolve_shape(&self, call: &CallShape) -> CallShape {
        CallShape {
            params: call
                .params
                .iter()
                .map(|param| ParamTerm::new(param.name, self.terms.resolve_ty(&param.ty)))
                .collect(),
            ret: self.terms.resolve_ty(&call.ret),
        }
    }

    /// A local binding used as a signature (RFC-0043) can still be an open
    /// variable with no function head, so there is no declared effect to
    /// read. The variable made here is that binding's own: `settle_join`
    /// below gives an unbound instance this very call type, which is how
    /// the binding comes to carry this term.
    fn effect_of_instance(&mut self, instance: &InferTy) -> EffectTerm<Infer> {
        match self.terms.shallow_resolve_ty(instance) {
            TyTerm::Fn { effect, .. } => effect,
            _ => self.fresh_effect_var(),
        }
    }

    fn takes_signature(&self, call: &CallShape, option: &SignatureOption) -> bool {
        let converted = option.converted.as_slice();
        let mut trial = self.terms.clone();
        let call_ty = call.at_effect(EffectTerm::Var(trial.alloc_effect_var()));
        let scheme = match &option.candidate {
            SignatureCandidate::Named { scheme, .. } => scheme,
            SignatureCandidate::Local { ty } => {
                return trial
                    .join(&call_ty, ty, Position::Value, JoinKind::Flow, self.registry)
                    .is_ok();
            }
        };
        let instance = trial.instantiate_open(&scheme.ty, self.registry);
        if trial
            .join(
                &call_ty,
                &instance,
                Position::Value,
                JoinKind::Decision,
                self.registry,
            )
            .is_err()
        {
            return false;
        }
        let bounds_meet =
            call.params
                .iter()
                .zip(scheme.params())
                .all(
                    |(param, declared)| match self.terms.shallow_resolve_ty(&param.ty) {
                        TyTerm::Var(var) => self
                            .bound_of_var(var)
                            .meet(&scheme.param_bound(&declared.ty))
                            .is_some(),
                        _ => true,
                    },
                );
        if !bounds_meet {
            return false;
        }
        let TyTerm::Fn {
            params: instance_params,
            ..
        } = &instance
        else {
            unreachable!("a candidate's scheme is a function type")
        };
        converted.iter().all(|argument| {
            let param = &instance_params[argument.index].ty;
            converts(&trial, self.registry, &argument.ty, param)
                || trial
                    .join(
                        &argument.ty,
                        param,
                        Position::Value,
                        JoinKind::Flow,
                        self.registry,
                    )
                    .is_ok()
        })
    }

    /// RFC-0030, RFC-0043.
    pub fn receiver_mode(&self, candidate: &SignatureCandidate) -> ReceiverMode {
        let mutability = match candidate {
            SignatureCandidate::Named { scheme, .. } => {
                match scheme.params().first().map(|p| &p.ty) {
                    Some(TyTerm::Ref(mutability, _)) => Some(*mutability),
                    _ => None,
                }
            }
            SignatureCandidate::Local { ty } => match SignatureCandidate::local_params(ty)
                .and_then(<[_]>::first)
                .map(|param| self.terms.shallow_resolve_ty(&param.ty))
            {
                Some(TyTerm::Ref(mutability, _)) => Some(mutability),
                _ => None,
            },
        };
        mutability.map_or(ReceiverMode::Value, ReceiverMode::Lent)
    }

    /// How the candidate takes the argument at `index` (RFC-0043), by one
    /// declared rule (RFC-0023) where it converts. An argument still a
    /// variable is admitted as it is where the two bounds intersect: a
    /// conversion is admitted from a resolved head only.
    pub fn admits(&self, candidate: &SignatureCandidate, index: usize, arg: &InferTy) -> Admission {
        let bound = candidate.param_bound(index);
        if let TyTerm::Var(var) = self.terms.shallow_resolve_ty(arg) {
            return match self.bound_of_var(var).meet(&bound) {
                Some(_) => Admission::Direct,
                None => Admission::Refused,
            };
        }
        let shapes = match bound {
            TyVarBound::Any => return Admission::Direct,
            TyVarBound::OneOf(shapes) => shapes,
            TyVarBound::Integer { among, .. } => among.into_iter().map(TyTerm::Int).collect(),
        };
        if self.term_within_shapes(arg, &shapes) {
            return Admission::Direct;
        }
        let converts = shapes.iter().any(|shape| {
            let mut trial = self.terms.clone();
            let to = trial.instantiate_open(shape, self.registry);
            converts(&trial, self.registry, arg, &to)
        });
        if converts {
            Admission::Converted
        } else {
            Admission::Refused
        }
    }

    /// `take_other` binds a `OneOf`-bounded variable to any term and leaves
    /// the bound to `freeze_ty_with`, so a decision that would bind one
    /// tests the term against the shapes here first.
    fn term_within_shapes(&self, term: &InferTy, shapes: &[PolyTy]) -> bool {
        let term = self.terms.resolve_ty(term);
        matches!(term, TyTerm::Var(_))
            || shapes.iter().any(|shape| could_match_pattern(&term, shape))
    }

    /// A conversion into such a parameter answers identity only by binding
    /// it, which would decide the signature by the argument's shape
    /// (RFC-0043).
    fn awaits_signature(&self, to: &InferTy) -> bool {
        let TyTerm::Var(var) = self.terms.shallow_resolve_ty(to) else {
            return false;
        };
        self.decisions
            .iter()
            .filter(|decision| matches!(decision.state, DecisionState::Open))
            .any(|decision| {
                let Decision::Signature { call, .. } = &decision.decision else {
                    return false;
                };
                call.params.iter().any(|param| {
                    matches!(self.terms.shallow_resolve_ty(&param.ty), TyTerm::Var(v) if v == var)
                })
            })
    }

    fn identity_within_bounds(&self, from: &InferTy, to: &InferTy) -> bool {
        [(from, to), (to, from)]
            .into_iter()
            .all(|(var_side, term_side)| {
                let TyTerm::Var(var) = self.terms.shallow_resolve_ty(var_side) else {
                    return true;
                };
                let TyVarBound::OneOf(shapes) = self.bound_of_var(var) else {
                    return true;
                };
                self.term_within_shapes(term_side, &shapes)
            })
    }

    /// An instance decision narrows to the signatures the call type would
    /// still join, and settles when one remains. With a generic instance
    /// present, a concrete one is taken only once the call type joins it
    /// with no variable left open in the join: an open variable may still
    /// be a demand for the generic one (hash-types.md R3).
    /// Of the instances whose declared ceiling admits the call's task, the
    /// one with the lowest ceiling (RFC-0046). The task read is the one the
    /// type freezes to - the join of the lower bounds - which is the task
    /// the program runs with. A list whose ceilings agree is left alone, so
    /// a declaration that names no task is decided by its types as before.
    fn tightest_admitting(&self, call: &InferTy, candidates: Vec<Candidate>) -> Vec<Candidate> {
        let one_ceiling = candidates
            .iter()
            .map(|c| c.admits)
            .min()
            .is_some_and(|min| candidates.iter().all(|c| c.admits == min));
        if one_ceiling {
            return candidates;
        }
        let TyTerm::Fn { effect, .. } = call else {
            return candidates;
        };
        let task = self.terms.freeze_effect(effect).task;
        let admitting: Vec<Candidate> = candidates
            .into_iter()
            .filter(|c| task <= c.admits)
            .collect();
        let Some(tightest) = admitting.iter().map(|c| c.admits).min() else {
            return admitting;
        };
        admitting
            .into_iter()
            .filter(|c| c.admits == tightest)
            .collect()
    }

    /// Every instance decision the types left tied, narrowed to the
    /// instances its task admits (RFC-0046). It runs where the other least
    /// elements are taken, once `settle` has stalled, because the task a
    /// call runs with is the join over its arguments and the last of those
    /// arrives when the argument's own decision settles.
    fn close_instances_by_task(&mut self) {
        for index in 0..self.decisions.len() {
            if !matches!(self.decisions[index].state, DecisionState::Open) {
                continue;
            }
            let Decision::Instance {
                call, candidates, ..
            } = &self.decisions[index].decision
            else {
                continue;
            };
            let call = self.terms.resolve_ty(call);
            let narrowed = self.tightest_admitting(&call, candidates.clone());
            if let Decision::Instance { candidates, .. } = &mut self.decisions[index].decision {
                *candidates = narrowed;
            }
        }
    }

    fn step_instance(
        &mut self,
        id: DecisionId,
        call: &InferTy,
        candidates: Vec<Candidate>,
        generic: Option<GenericInstance>,
    ) -> Progress {
        let ty = self.terms.resolve_ty(call);
        let remaining: Vec<Candidate> = candidates
            .iter()
            .filter(|c| self.terms.would_take(call, &c.ty, self.registry))
            .cloned()
            .collect();
        let generic = generic.filter(|g| self.terms.would_take(call, &g.ty, self.registry));
        let narrowed = remaining.len() != candidates.len();
        if narrowed
            && let Decision::Instance { candidates, .. } =
                &mut self.decisions[id.0 as usize].decision
        {
            *candidates = remaining.clone();
        }
        let progress_without_answer = if narrowed {
            Progress::Narrowed
        } else {
            Progress::Unchanged
        };
        match (remaining.as_slice(), generic) {
            ([], None) => Progress::Failed(Unsettled::NoInstance {
                decision: id,
                call: ty,
            }),
            ([], Some(generic)) => {
                let instance = self.instantiate_open(&generic.ty);
                match self.settle_join(call, &instance) {
                    Ok(()) => {
                        Progress::Settled(Answer::Instance(InstanceKind::Extern(generic.instance)))
                    }
                    Err(Mismatch { expected, got, .. }) => {
                        Progress::Failed(Unsettled::InstanceMismatch {
                            decision: id,
                            expected,
                            got,
                        })
                    }
                }
            }
            ([only], generic) if generic.is_none() || matches_pattern(&ty, &only.ty) => {
                let instance = self.instantiate_open(&only.ty);
                match self.settle_join(call, &instance) {
                    Ok(()) => Progress::Settled(Answer::Instance(only.instance)),
                    Err(Mismatch { expected, got, .. }) => {
                        Progress::Failed(Unsettled::InstanceMismatch {
                            decision: id,
                            expected,
                            got,
                        })
                    }
                }
            }
            _ => progress_without_answer,
        }
    }

    /// A conversion is identity where the two join, the join written as
    /// the answer; else the one declared cast, once both sides resolve. A
    /// pair of references is answered through the rule for what they name,
    /// and needs the rule back as well.
    fn step_conversion(&mut self, id: DecisionId, from: &InferTy, to: &InferTy) -> Progress {
        if self.awaits_signature(to) {
            return Progress::Unchanged;
        }
        let joined = self.trial_settle_join(from, to);
        if self.identity_within_bounds(from, to) && joined.is_ok() {
            self.settle_join(from, to)
                .expect("the trial join checked this on a copy of the terms");
            return Progress::Settled(Answer::Conversion(Conversion::Identity));
        }
        if let Err(Mismatch {
            reason: MismatchReason::TaskTooHigh { required, found },
            ..
        }) = joined
        {
            return Progress::Failed(Unsettled::TaskTooHigh {
                decision: id,
                required,
                found,
            });
        }
        let from_r = self.terms.shallow_resolve_ty(from);
        let to_r = self.terms.shallow_resolve_ty(to);
        if matches!(from_r, TyTerm::Var(_)) || matches!(to_r, TyTerm::Var(_)) {
            return Progress::Unchanged;
        }
        let rules = self.conversion_rules(from, to);
        let [rule] = rules.as_slice() else {
            if rules.is_empty() {
                return Progress::Failed(Unsettled::NoConversion {
                    decision: id,
                    from: self.terms.resolve_ty(from),
                    to: self.terms.resolve_ty(to),
                });
            }
            return Progress::Unchanged;
        };
        let rule = rule.clone();
        let (inst_from, inst_to) = self.instantiate_poly_pair(&rule.from, &rule.to);
        let Some(references) = ReferencePair::of(&from_r, &to_r) else {
            return self.settle_cast(
                id,
                from,
                to,
                &inst_from,
                &inst_to,
                Conversion::Cast(rule.fn_ref),
            );
        };
        let back = self.conversion_rules(&references.to.ty, &references.from.ty);
        let [back] = back.as_slice() else {
            if back.is_empty() {
                return Progress::Failed(Unsettled::NoConversion {
                    decision: id,
                    from: self.terms.resolve_ty(&references.to.ty),
                    to: self.terms.resolve_ty(&references.from.ty),
                });
            }
            return Progress::Unchanged;
        };
        let answer = Conversion::ThroughRef {
            mutability: references.mutability,
            cast: rule.fn_ref,
            back: back.fn_ref,
        };
        let inst_from = references.from_side(inst_from);
        let inst_to = references.to_side(inst_to);
        self.settle_cast(id, from, to, &inst_from, &inst_to, answer)
    }

    /// The rule's two sides, instantiated, joined with the decision's two
    /// sides; the decision settles on `answer`.
    fn settle_cast(
        &mut self,
        id: DecisionId,
        from: &InferTy,
        to: &InferTy,
        inst_from: &InferTy,
        inst_to: &InferTy,
        answer: Conversion,
    ) -> Progress {
        match self
            .settle_join(from, inst_from)
            .and_then(|()| self.settle_join(inst_to, to))
        {
            Ok(()) => Progress::Settled(Answer::Conversion(answer)),
            Err(Mismatch { expected, got, .. }) => Progress::Failed(Unsettled::NoConversion {
                decision: id,
                from: expected,
                to: got,
            }),
        }
    }

    fn conversion_rules(&self, from: &InferTy, to: &InferTy) -> Vec<CastRule> {
        conversion_rules(&self.terms, self.registry, from, to)
    }

    // -- Freeze --------------------------------------------------------

    /// A type after `solve`: every variable nothing constrained is `!`.
    pub fn close_ty(&self, ty: &InferTy) -> Result<Ty, FreezeError> {
        self.freeze_ty_with(ty, Open::Never)
    }

    /// A type as far as it is known: a variable still open is an error,
    /// for a message rendered before `solve`.
    pub fn freeze_ty(&self, ty: &InferTy) -> Result<Ty, FreezeError> {
        self.freeze_ty_with(ty, Open::Refuse)
    }

    /// The type a report shows: as written, every variable nothing
    /// resolved closed to `!`, whatever bound it carries (RFC-0043).
    pub fn written_ty(&self, ty: &InferTy) -> Result<Ty, FreezeError> {
        self.freeze_ty_with(ty, Open::AsWritten)
    }

    fn freeze_ty_with(&self, ty: &InferTy, open: Open) -> Result<Ty, FreezeError> {
        ty.try_map(
            &mut |id: TypeBoundId| {
                let root = self.terms.find_ty_root(id);
                match &self.terms.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, bound } => {
                        let frozen = self.freeze_ty_with(inner, open)?;
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
                    TypeBound::Unresolved { bound } => match (bound.integer_default(), bound, open)
                    {
                        (Some(k), _, _) => Ok(Ty::Int(k)),
                        (None, TyVarBound::Any, Open::Never) | (None, _, Open::AsWritten) => {
                            Ok(Ty::Never)
                        }
                        (None, _, _) => Err(FreezeError::UnresolvedType(root)),
                    },
                    TypeBound::Forward(_) => unreachable!("find_ty_root resolves forwards"),
                }
            },
            &mut |id: IdentityVarId| {
                self.freeze_identity(&IdentityTerm::Var(id))
                    .map(IdentityTerm::Known)
            },
            &mut |id: EffectVarId| Ok(EffectTerm::Known(self.freeze_effect(&EffectTerm::Var(id)))),
            &mut |id: LenVarId| match self.terms.resolve_len(&LenTerm::Var(id)) {
                LenTerm::Known(n) => Ok(LenTerm::Known(n)),
                LenTerm::Var(root) => Err(FreezeError::UnresolvedLen(root)),
            },
            &mut |id: ReprVarId| match self.terms.resolve_repr(Repr::Var(id)) {
                Repr::Uniform => Ok(Repr::Uniform),
                Repr::Specialized => Ok(Repr::Specialized),
                Repr::Var(root) => Err(FreezeError::UnresolvedRepr(root)),
            },
        )
    }

    pub fn freeze_effect(&self, term: &EffectTerm<Infer>) -> Effect {
        self.terms.freeze_effect(term)
    }

    pub fn freeze_identity(&self, term: &IdentityTerm<Infer>) -> Result<IdentityId, FreezeError> {
        match self.terms.resolve_identity(term) {
            IdentityTerm::Known(id) => Ok(id),
            IdentityTerm::Var(root) => Err(FreezeError::UnresolvedIdentity(root)),
        }
    }

    // -- Instantiate -----------------------------------------------------

    /// A polymorphic type with fresh variables for every placeholder;
    /// identities are declared (RFC-0012).
    pub fn instantiate_poly(&mut self, ty: &PolyTy) -> InferTy {
        self.instantiate_scheme(&Scheme::unbounded(ty.clone())).ty
    }

    /// A polymorphic type with every identity a variable: for unifying
    /// with a type whose sources are already minted.
    fn instantiate_open(&mut self, ty: &PolyTy) -> InferTy {
        self.terms.instantiate_open(ty, self.registry)
    }

    pub fn fresh_shape(&mut self, pattern: &PolyTy) -> InferTy {
        self.instantiate_open(pattern)
    }

    pub fn instantiate_scheme(&mut self, scheme: &Scheme) -> Instantiated {
        self.instantiate_scheme_with(scheme, Vec::new())
    }

    /// A scheme with instances the compiler adds to the declared ones
    /// (RFC-0020): its own instructions for a shared signature.
    pub fn instantiate_scheme_with(
        &mut self,
        scheme: &Scheme,
        compiler_instances: Vec<Candidate>,
    ) -> Instantiated {
        let mut bounded: Vec<TypeBoundId> = Vec::new();
        let fixed_generic = scheme.instances.as_ref().is_some_and(|instances| {
            instances.concrete.is_empty() && !instances.generic && compiler_instances.is_empty()
        });
        let reprs = if fixed_generic {
            Reprs::Uniform
        } else {
            Reprs::Open
        };
        let ty = self.instantiate_with(
            &scheme.ty,
            |var, fresh| {
                let bound = scheme.bound_of(var);
                if bound != TyVarBound::Any {
                    bounded.push(fresh);
                }
                bound
            },
            Identities::Declared,
            reprs,
        );
        let instance = scheme.instances.as_ref().map(|instances| {
            if fixed_generic {
                return InstanceChoice::Fixed(instances.generic_index());
            }
            let id = self.decide(Decision::Instance {
                call: ty.clone(),
                candidates: instances
                    .concrete
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(instance, sig)| Candidate {
                        instance: InstanceKind::Extern(instance),
                        ty: sig.ty,
                        admits: sig.admits,
                    })
                    .chain(compiler_instances)
                    .collect(),
                generic: instances.generic.then(|| GenericInstance {
                    instance: instances.generic_index(),
                    ty: scheme.ty.clone(),
                }),
            });
            InstanceChoice::Decided(id)
        });
        Instantiated {
            ty,
            bounded,
            instance,
        }
    }

    fn instantiate_with(
        &mut self,
        ty: &PolyTy,
        mut bound_for: impl FnMut(u32, TypeBoundId) -> TyVarBound,
        identities: Identities,
        reprs: Reprs,
    ) -> InferTy {
        let mut maps = PolyMaps::default();
        let from_params = identity_vars_bound_by_params(ty);
        let Terms {
            ty_bounds,
            effect_vars,
            len_vars,
            identity_vars,
            repr_vars,
            ..
        } = &mut self.terms;
        let sources = &mut *self.sources;
        let instance = ty.map(
            &mut |id: u32| {
                let var = *maps.ty.entry(id).or_insert_with(|| {
                    let fresh = alloc_ty_var(ty_bounds, TyVarBound::Any);
                    let bound = bound_for(id, fresh);
                    ty_bounds[fresh.0 as usize] = TypeBound::Unresolved { bound };
                    fresh
                });
                TyTerm::Var(var)
            },
            &mut |id: u32| {
                *maps.identity.entry(id).or_insert_with(|| {
                    let open = match identities {
                        Identities::Declared => from_params.contains(&id),
                        Identities::Open => true,
                    };
                    if open {
                        IdentityTerm::Var(alloc_identity_var(identity_vars))
                    } else {
                        IdentityTerm::Known(sources.next())
                    }
                })
            },
            &mut |id: u32| {
                EffectTerm::Var(
                    *maps
                        .effect
                        .entry(id)
                        .or_insert_with(|| alloc_effect_var(effect_vars)),
                )
            },
            &mut |id: u32| {
                LenTerm::Var(
                    *maps
                        .len
                        .entry(id)
                        .or_insert_with(|| alloc_len_var(len_vars)),
                )
            },
            &mut |id: u32| match reprs {
                Reprs::Uniform => Repr::Uniform,
                Reprs::Open => Repr::Var(
                    *maps
                        .repr
                        .entry(id)
                        .or_insert_with(|| alloc_repr_var(repr_vars, ReprOwner::Signature)),
                ),
            },
        );
        uniform_slots(instance, self.registry)
    }

    /// Two polymorphic types sharing one set of placeholders, as a cast
    /// rule's `from` and `to` (RFC-0023): identities of `from` are open,
    /// those of `to` are new sources.
    pub fn instantiate_poly_pair(&mut self, a: &PolyTy, b: &PolyTy) -> (InferTy, InferTy) {
        let mut maps = PolyMaps::default();
        let Terms {
            ty_bounds,
            effect_vars,
            len_vars,
            identity_vars,
            repr_vars,
            ..
        } = &mut self.terms;
        let sources = &mut *self.sources;
        let mut on_var = |id: u32| {
            TyTerm::Var(
                *maps
                    .ty
                    .entry(id)
                    .or_insert_with(|| alloc_ty_var(ty_bounds, TyVarBound::Any)),
            )
        };
        let mut on_effect = |id: u32| {
            EffectTerm::Var(
                *maps
                    .effect
                    .entry(id)
                    .or_insert_with(|| alloc_effect_var(effect_vars)),
            )
        };
        let mut on_len = |id: u32| {
            LenTerm::Var(
                *maps
                    .len
                    .entry(id)
                    .or_insert_with(|| alloc_len_var(len_vars)),
            )
        };
        let mut on_repr = |id: u32| {
            Repr::Var(
                *maps
                    .repr
                    .entry(id)
                    .or_insert_with(|| alloc_repr_var(repr_vars, ReprOwner::Signature)),
            )
        };
        let mut on_identity_a = |id: u32| {
            *maps
                .identity
                .entry(id)
                .or_insert_with(|| IdentityTerm::Var(alloc_identity_var(identity_vars)))
        };
        let ia = a.map(
            &mut on_var,
            &mut on_identity_a,
            &mut on_effect,
            &mut on_len,
            &mut on_repr,
        );
        let mut on_identity_b = |id: u32| {
            *maps
                .identity
                .entry(id)
                .or_insert_with(|| IdentityTerm::Known(sources.next()))
        };
        let ib = b.map(
            &mut on_var,
            &mut on_identity_b,
            &mut on_effect,
            &mut on_len,
            &mut on_repr,
        );
        (
            uniform_slots(ia, self.registry),
            uniform_slots(ib, self.registry),
        )
    }

    /// An `InferTy` with every open variable renamed fresh: a let-bound
    /// type used again.
    pub fn instantiate_infer(&mut self, ty: &InferTy) -> InferTy {
        let resolved = self.terms.resolve_ty(ty);
        let mut maps = InferMaps::default();
        let Terms {
            ty_bounds,
            effect_vars,
            len_vars,
            identity_vars,
            repr_vars,
            ..
        } = &mut self.terms;
        resolved.map(
            &mut |root: TypeBoundId| {
                let bound = match &ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { bound, .. } | TypeBound::Unresolved { bound } => {
                        bound.clone()
                    }
                    TypeBound::Forward(_) => unreachable!("resolve_ty yields roots"),
                };
                TyTerm::Var(
                    *maps
                        .ty
                        .entry(root)
                        .or_insert_with(|| alloc_ty_var(ty_bounds, bound)),
                )
            },
            &mut |root: IdentityVarId| {
                IdentityTerm::Var(
                    *maps
                        .identity
                        .entry(root)
                        .or_insert_with(|| alloc_identity_var(identity_vars)),
                )
            },
            &mut |root: EffectVarId| {
                EffectTerm::Var(
                    *maps
                        .effect
                        .entry(root)
                        .or_insert_with(|| alloc_effect_var(effect_vars)),
                )
            },
            &mut |root: LenVarId| {
                LenTerm::Var(
                    *maps
                        .len
                        .entry(root)
                        .or_insert_with(|| alloc_len_var(len_vars)),
                )
            },
            &mut |root: ReprVarId| {
                let owner = match &repr_vars[root.0 as usize] {
                    ReprBound::Unbound(owner) => *owner,
                    ReprBound::Bound(_) | ReprBound::Forward(_) => {
                        unreachable!("resolve_ty yields open roots")
                    }
                };
                Repr::Var(
                    *maps
                        .repr
                        .entry(root)
                        .or_insert_with(|| alloc_repr_var(repr_vars, owner)),
                )
            },
        )
    }
}

/// The placeholders one instantiation of a `PolyTy` has renamed so far.
#[derive(Default)]
struct PolyMaps {
    ty: FxHashMap<u32, TypeBoundId>,
    identity: FxHashMap<u32, IdentityTerm<Infer>>,
    effect: FxHashMap<u32, EffectVarId>,
    len: FxHashMap<u32, LenVarId>,
    repr: FxHashMap<u32, ReprVarId>,
}

/// The variables one re-instantiation of an `InferTy` has renamed so far.
#[derive(Default)]
struct InferMaps {
    ty: FxHashMap<TypeBoundId, TypeBoundId>,
    identity: FxHashMap<IdentityVarId, IdentityVarId>,
    effect: FxHashMap<EffectVarId, EffectVarId>,
    len: FxHashMap<LenVarId, LenVarId>,
    repr: FxHashMap<ReprVarId, ReprVarId>,
}

/// The identity variables of a polymorphic type that occur in its
/// parameters. Instantiated, these bind to the arguments' sources; every
/// other identity variable is a new source at each instantiation.
fn identity_vars_bound_by_params(ty: &PolyTy) -> FxHashSet<u32> {
    let mut found = FxHashSet::default();
    if let TyTerm::Fn { params, .. } = ty {
        for p in params {
            p.ty.map(
                &mut |v: u32| TyTerm::<Poly>::Var(v),
                &mut |v: u32| {
                    found.insert(v);
                    IdentityTerm::<Poly>::Var(v)
                },
                &mut |v: u32| EffectTerm::<Poly>::Var(v),
                &mut |v: u32| LenTerm::<Poly>::Var(v),
                &mut |v: u32| Repr::<Poly>::Var(v),
            );
        }
    }
    found
}

/// The declared conversions that could still take `from` to `to`; for
/// a pair of references of one mutability, those that take what they
/// name.
fn conversion_rules(
    terms: &Terms,
    registry: &TypeRegistry,
    from: &InferTy,
    to: &InferTy,
) -> Vec<CastRule> {
    let from_r = terms.resolve_ty(from);
    let to_r = terms.resolve_ty(to);
    if let Some(references) = ReferencePair::of(&from_r, &to_r) {
        return conversion_rules(terms, registry, &references.from.ty, &references.to.ty);
    }
    let shape = conversion_shape(&from_r, &to_r);
    let from_rules = match &from_r {
        TyTerm::UserDefined { id, .. } => registry.rules_from(*id),
        _ => &[],
    };
    let to_rules = match &to_r {
        TyTerm::UserDefined { id, .. } => registry.rules_to(*id),
        _ => &[],
    };
    let mut rules: Vec<CastRule> = Vec::new();
    for rule in from_rules.iter().chain(to_rules.iter()) {
        if rules.contains(rule) {
            continue;
        }
        if could_match_pattern(&shape, &conversion_shape(&rule.from, &rule.to)) {
            rules.push(rule.clone());
        }
    }
    rules
}

/// One declared rule (RFC-0023) takes `from` to `to`; through a reference,
/// the value is cast back when the call ends (RFC-0041).
fn converts(terms: &Terms, registry: &TypeRegistry, from: &InferTy, to: &InferTy) -> bool {
    let from_r = terms.resolve_ty(from);
    let to_r = terms.resolve_ty(to);
    let Some(references) = ReferencePair::of(&from_r, &to_r) else {
        return !conversion_rules(terms, registry, from, to).is_empty();
    };
    [
        (&references.from.ty, &references.to.ty),
        (&references.to.ty, &references.from.ty),
    ]
    .into_iter()
    .all(|(from, to)| !conversion_rules(terms, registry, from, to).is_empty())
}

/// `(from, to)` as one type, so one pattern match covers both sides of a
/// cast rule with the rule's placeholders shared.
fn conversion_shape<V>(from: &TyTerm<V>, to: &TyTerm<V>) -> TyTerm<V>
where
    V: Phase,
{
    TyTerm::Tuple(vec![from.clone(), to.clone()])
}

/// The two sides of a conversion decision when both are references of one
/// mutability: the decision is answered through what they name.
pub(crate) struct ReferencePair<'a> {
    pub(crate) mutability: Mutability,
    pub(crate) from: &'a TypeArg<Infer>,
    pub(crate) to: &'a TypeArg<Infer>,
}

impl<'a> ReferencePair<'a> {
    pub(crate) fn of(from: &'a InferTy, to: &'a InferTy) -> Option<Self> {
        let (TyTerm::Ref(from_m, from), TyTerm::Ref(to_m, to)) = (from, to) else {
            return None;
        };
        (from_m == to_m).then(|| Self {
            mutability: *from_m,
            from,
            to,
        })
    }

    /// The `from` reference with what it names replaced by `ty`.
    fn from_side(&self, ty: InferTy) -> InferTy {
        TyTerm::Ref(self.mutability, Box::new(TypeArg::new(self.from.repr, ty)))
    }

    /// The `to` reference with what it names replaced by `ty`.
    fn to_side(&self, ty: InferTy) -> InferTy {
        TyTerm::Ref(self.mutability, Box::new(TypeArg::new(self.to.repr, ty)))
    }
}

/// A signature enters the solver with the slots that do not specialize
/// (hash-types.md R1) set `Uniform`: their representation is not a fact
/// of the type.
fn uniform_slots(ty: InferTy, registry: &TypeRegistry) -> InferTy {
    fn arg(a: TypeArg<Infer>, specializing: bool, registry: &TypeRegistry) -> TypeArg<Infer> {
        TypeArg {
            repr: if specializing { a.repr } else { Repr::Uniform },
            ty: uniform_slots(a.ty, registry),
        }
    }
    match ty {
        TyTerm::UserDefined {
            id,
            type_args,
            effect_args,
            identity_args,
        } => TyTerm::UserDefined {
            id,
            type_args: type_args
                .into_iter()
                .enumerate()
                .map(|(index, a)| arg(a, registry.specializes(id, index), registry))
                .collect(),
            effect_args,
            identity_args,
        },
        TyTerm::Ref(m, inner) => TyTerm::Ref(m, Box::new(arg(*inner, true, registry))),
        TyTerm::Array(inner, len) => TyTerm::Array(Box::new(uniform_slots(*inner, registry)), len),
        TyTerm::Option(inner) => TyTerm::Option(Box::new(uniform_slots(*inner, registry))),
        TyTerm::Result(ok, err) => TyTerm::Result(
            Box::new(uniform_slots(*ok, registry)),
            Box::new(uniform_slots(*err, registry)),
        ),
        TyTerm::Tuple(elems) => TyTerm::Tuple(
            elems
                .into_iter()
                .map(|e| uniform_slots(e, registry))
                .collect(),
        ),
        TyTerm::Object(fields) => TyTerm::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k, uniform_slots(v, registry)))
                .collect(),
        ),
        TyTerm::Fn {
            params,
            ret,
            captures,
            effect,
        } => TyTerm::Fn {
            params: params
                .into_iter()
                .map(|p| {
                    let ty = uniform_slots(p.ty.clone(), registry);
                    p.retyped(ty)
                })
                .collect(),
            ret: Box::new(uniform_slots(*ret, registry)),
            captures: captures
                .into_iter()
                .map(|c| uniform_slots(c, registry))
                .collect(),
            effect,
        },
        TyTerm::Enum { name, variants } => TyTerm::Enum {
            name,
            variants: variants
                .into_iter()
                .map(|(k, v)| (k, v.map(|t| Box::new(uniform_slots(*t, registry)))))
                .collect(),
        },
        TyTerm::Handle(inner) => TyTerm::Handle(Box::new(uniform_slots(*inner, registry))),
        leaf @ (TyTerm::Int(_)
        | TyTerm::Float
        | TyTerm::String
        | TyTerm::Bool
        | TyTerm::Unit
        | TyTerm::Never
        | TyTerm::Order
        | TyTerm::Error(_)
        | TyTerm::Var(_)) => leaf,
    }
}

/// What freezing does with a type variable nothing constrained.
#[derive(Clone, Copy)]
enum Open {
    /// Before `solve`: the variable stays open, and the caller reports it.
    Refuse,
    /// After `solve`: the variable is `!` (RFC-0038). A variable whose
    /// bound names shapes is refused: no shape of it is the program's.
    Never,
    /// In a report: every variable nothing resolved is `!` (RFC-0043).
    AsWritten,
}

/// Error when freezing an InferTy that still contains unresolved variables.
#[derive(Debug)]
pub enum FreezeError {
    UnresolvedType(TypeBoundId),
    UnresolvedLen(LenVarId),
    UnresolvedIdentity(IdentityVarId),
    UnresolvedRepr(ReprVarId),
    /// The variable resolved to a type its declaration does not admit.
    OutOfBound {
        var: TypeBoundId,
        ty: Ty,
        bound: TyVarBound,
    },
}

/// A scheme instantiated into the solver; the caller verifies the bounded
/// variables where it chose to.
pub struct Instantiated {
    pub ty: InferTy,
    pub bounded: Vec<TypeBoundId>,
    /// `Some` for an Extern function.
    pub instance: Option<InstanceChoice>,
}

/// Which instance of an Extern function a call runs: fixed at the
/// instantiation when the function has one instance and no generic one,
/// or a decision the solver settles as the call's type resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceChoice {
    Fixed(usize),
    Decided(DecisionId),
}

/// What a representation variable of a declaration becomes when the
/// declaration is instantiated.
#[derive(Clone, Copy)]
enum Reprs {
    /// A signature's variable, bound by its instance decision or the
    /// default.
    Open,
    /// `Uniform`: the function's one instance is the generic one, and the
    /// generic instance is the uniform one (hash-types.md R3).
    Uniform,
}

/// What an identity variable of a declaration becomes when the declaration
/// is instantiated.
#[derive(Clone, Copy)]
enum Identities {
    /// A parameter's identity is a variable the argument fixes; any other
    /// is a new source (RFC-0012).
    Declared,
    /// Every identity is a variable: for unifying with a type whose
    /// sources are already minted.
    Open,
}
