//! The type solver: equality now, decisions later (RFC-0042).
//!
//! `Terms` is the union-find over type, effect, length, identity, and
//! representation variables, and `join` on it is the one unification: the
//! join of the type lattice, taken where it is asked. A `Decision` is a
//! position with more than one admissible answer; it shrinks as the terms
//! resolve and settles when one answer remains. `Solver` owns both, the
//! registry that says which slots specialize and which conversions exist,
//! and the compilation's sources. A body is checked (`fresh`, `unify`,
//! `decide`), then solved once (`solve`), then frozen.

use std::rc::Rc;

use acvus_ast::Span;
use acvus_utils::Astr;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::types::QualifiedRef;
use crate::ir::Intrinsic;
use crate::pvec::PVec;
use crate::structural::{Component, StructuralSignature, components};
use crate::ty::{
    CastRule, Concrete, Effect, EffectConflict, EffectTerm, EffectVarBound, EffectVarId,
    ErrorToken, FieldSet, FlowTerm, FlowVarId, Flows, HeldTy, Home, IdentityId, IdentityTerm, IdentityVarId, Infer, InferTy,
    Instances, IntTy, LenTerm, LenVarId, Mutability, ObjectMeet, ObjectTy, ParamTerm, Phase, Poly,
    PolyTy, Repr, ReprVarId, RequirementSig, Scheme, Task, Ty, TyTerm, TyVarBound, TypeArg,
    TypeBoundId, TypeRegistry, View, Viewed, could_match_pattern, effect_bound_at, matches_pattern,
};

// -- Variable states --------------------------------------------------

/// State of a type variable. The declared bound travels with the variable
/// and is verified when it freezes (`TyVarBound::admits`).
#[derive(Debug, Clone)]
pub enum TypeBound {
    Resolved {
        ty: InferTy,
        bound: TyVarBound,
        growth: Growth,
    },
    Unresolved {
        bound: TyVarBound,
    },
    Forward(TypeBoundId),
}

/// RFC-0042 rule 1 and RFC-0050 rule 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Growth {
    Open,
    Fixed,
}

/// State of an effect variable. The declared bound travels with the
/// variable and is verified when it freezes (RFC-0011 rule 5); nothing
/// about the interval reads it.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectBound {
    /// `lower <= var <= upper` on the reissue chain (RFC-0013 rule 1).
    Range {
        lower: Effect,
        upper: Effect,
        bound: EffectVarBound,
    },
    Bound {
        effect: Effect,
        bound: EffectVarBound,
    },
    Forward(EffectVarId),
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
    Uniform,
    Specialized(HeldTy<Infer>),
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

/// Where a source began, as a reader would point at it: the expression
/// that minted the identity and the name the program gave the value.
/// Both are what a refusal over two sources prints, and neither is known
/// to the solver, so `Origin` is written by the checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin {
    pub span: Span,
    pub name: Option<Astr>,
}

/// The sources of one compilation. A source number names one identity
/// for every solver of the compilation, so a frozen type may pass from
/// one solver to another and still name the source it was frozen with.
///
/// An identity minted where no expression is in hand has no `Origin`, and
/// a refusal over it names the places it can name and no more.
///
/// `Clone` is for a check whose identities must not reach the compilation:
/// a completion probe mints into a copy.
#[derive(Debug, Default, Clone)]
pub struct Sources {
    ids: acvus_utils::LocalFactory<IdentityId>,
    origins: FxHashMap<IdentityId, Origin>,
}

impl Sources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next(&mut self) -> IdentityId {
        self.ids.next()
    }

    /// The first expression to name a source is the one that minted it:
    /// an instantiation that reuses the identity later is not its origin.
    pub fn begins_at(&mut self, id: IdentityId, span: Span) {
        self.origins
            .entry(id)
            .or_insert(Origin { span, name: None });
    }

    /// The first binding to hold a value of this source names it, as the
    /// program does: a later rebinding is another name for one source.
    pub fn named(&mut self, id: IdentityId, name: Astr) {
        if let Some(origin) = self.origins.get_mut(&id)
            && origin.name.is_none()
        {
            origin.name = Some(name);
        }
    }

    pub fn origin(&self, id: IdentityId) -> Option<Origin> {
        self.origins.get(&id).copied()
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
    /// A type the body did not lay out lacks `member`, which the other
    /// side of the join has.
    FixedLacks { member: Astr },
    /// A slot's representation is a signature's open variable, which a
    /// flow does not bind: the decision that owns it answers later.
    ReprOpen(ReprVarId),
    /// The value's task is above the one the position fixes (RFC-0046).
    TaskTooHigh { required: Task, found: Task },
    /// A field the struct `declared` names and the object lacks. A declared
    /// struct's field set is exact (RFC-0042).
    ObjectLacksDeclaredField { declared: Astr, field: Astr },
    /// A field the object has and the struct `declared` does not name.
    ObjectFieldNotDeclared { declared: Astr, field: Astr },
    /// The union of the two field sets is wider than `ObjectTy::MAX_FIELDS`.
    ObjectTooWide { fields: usize },
    /// The first side is a `&mut` and the second a `&`: a flow of the first
    /// into the second reborrows it shared (RFC-0029 rule 3).
    Weakens,
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
///
/// The solver tries a join on a copy of the terms once per open conversion,
/// so each table is a [`PVec`], whose copy costs what the trial writes. A
/// `Vec` copy cost every variable of the body per conversion, quadratic in
/// an array literal's elements.
struct Terms {
    ty_bounds: PVec<TypeBound>,
    effect_vars: PVec<EffectBound>,
    /// `a <= b` (RFC-0013 rule 1).
    effect_below: PVec<(EffectVarId, EffectVarId)>,
    len_vars: PVec<LenBound>,
    identity_vars: PVec<IdentityBound>,
    repr_vars: PVec<ReprBound>,
    flow_vars: PVec<FlowBound>,
}

/// A function type's flows during the solve (RFC-0079 rule 5). Two
/// function types that meet forward one variable to the other, and the
/// root holds the union of what both carried: a value of either type may
/// be called through the other.
#[derive(Debug, Clone)]
enum FlowBound {
    Root(Flows),
    Forward(FlowVarId),
}

/// Two function types met whose flows are both fixed and differ: neither
/// can take the other's.
#[derive(Debug, Clone, Copy)]
pub struct FixedFlowsDiffer;

/// Whether raising a function type's flows changed them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Raised {
    Grew,
    Held,
}

impl Terms {
    fn new() -> Self {
        Self {
            ty_bounds: PVec::new(),
            effect_vars: PVec::new(),
            effect_below: PVec::new(),
            len_vars: PVec::new(),
            identity_vars: PVec::new(),
            repr_vars: PVec::new(),
            flow_vars: PVec::new(),
        }
    }

    // -- Flow variables ----------------------------------------------

    fn alloc_flow_var(&mut self, seed: Flows) -> FlowVarId {
        alloc_flow_var(&mut self.flow_vars, seed)
    }

    fn find_flow_root(&self, id: FlowVarId) -> FlowVarId {
        match &self.flow_vars[id.0 as usize] {
            FlowBound::Forward(next) => self.find_flow_root(*next),
            FlowBound::Root(_) => id,
        }
    }

    fn flows_at(&self, id: FlowVarId) -> &Flows {
        match &self.flow_vars[self.find_flow_root(id).0 as usize] {
            FlowBound::Root(flows) => flows,
            FlowBound::Forward(_) => unreachable!("find_flow_root resolves forwards"),
        }
    }

    fn resolve_flows(&self, term: &FlowTerm<Infer>) -> FlowTerm<Infer> {
        match term {
            FlowTerm::Known(flows) => FlowTerm::Known(flows.clone()),
            FlowTerm::Var(id) => FlowTerm::Var(self.find_flow_root(*id)),
        }
    }

    fn freeze_flows(&self, term: &FlowTerm<Infer>) -> Flows {
        match term {
            FlowTerm::Known(flows) => flows.clone(),
            FlowTerm::Var(id) => self.flows_at(*id).clone(),
        }
    }

    fn raise_flows(&mut self, id: FlowVarId, flows: &Flows) -> Raised {
        let root = self.find_flow_root(id);
        let FlowBound::Root(held) = &self.flow_vars[root.0 as usize] else {
            unreachable!("find_flow_root resolves forwards")
        };
        let joined = held.join(flows);
        let raised = match joined == *held {
            true => Raised::Held,
            false => Raised::Grew,
        };
        self.flow_vars[root.0 as usize] = FlowBound::Root(joined);
        raised
    }

    /// Where two function types meet, their flows join by union.
    fn meet_flows(&mut self, a: &FlowTerm<Infer>, b: &FlowTerm<Infer>) -> Result<(), FixedFlowsDiffer> {
        match (a, b) {
            (FlowTerm::Var(x), FlowTerm::Var(y)) => {
                let (x, y) = (self.find_flow_root(*x), self.find_flow_root(*y));
                if x == y {
                    return Ok(());
                }
                let from = self.flows_at(x).clone();
                self.raise_flows(y, &from);
                self.flow_vars[x.0 as usize] = FlowBound::Forward(y);
                Ok(())
            }
            (FlowTerm::Var(var), FlowTerm::Known(flows))
            | (FlowTerm::Known(flows), FlowTerm::Var(var)) => {
                self.raise_flows(*var, flows);
                Ok(())
            }
            (FlowTerm::Known(a), FlowTerm::Known(b)) => match a == b {
                true => Ok(()),
                false => Err(FixedFlowsDiffer),
            },
        }
    }

    /// Every function type in `ty` given a flow variable of its own, seeded
    /// with the flows it states: an instantiated signature meets the types
    /// at its call, and a fixed term could not take theirs.
    fn open_flows(&mut self, ty: &mut InferTy) {
        if let TyTerm::Fn { flows, .. } = ty
            && let FlowTerm::Known(known) = flows
        {
            *flows = FlowTerm::Var(self.alloc_flow_var(known.clone()));
        }
        ty.rewrite_children(&mut |child| self.open_flows(child));
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

    fn bind_ty(&mut self, id: TypeBoundId, ty: InferTy, growth: Growth) -> Result<(), Cyclic> {
        let ty = self.stored(ty);
        let root = self.find_ty_root(id);
        if self.occurs_in(root, &ty) {
            return Err(Cyclic);
        }
        let bound = self.bound_of(root);
        self.ty_bounds[root.0 as usize] = TypeBound::Resolved { ty, bound, growth };
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
    /// The variable a type is a name of: a variable, or a structural term
    /// the solver resolved from one.
    fn root_var(&self, ty: &InferTy) -> Option<TypeBoundId> {
        match ty {
            TyTerm::Var(id) => Some(self.find_ty_root(*id)),
            other => other.home().map(|home| self.find_ty_root(home)),
        }
    }

    /// `ty` with every term that carries a home replaced by that home's
    /// variable, so a resolved copy is read as what it copies and never as
    /// the snapshot it holds.
    fn unstale(&self, ty: &InferTy) -> InferTy {
        let mut ty = ty.clone();
        self.unstale_in_place(&mut ty);
        ty
    }

    fn unstale_in_place(&self, ty: &mut InferTy) {
        if let Some(home) = ty.home() {
            *ty = TyTerm::Var(home);
            return;
        }
        ty.rewrite_children(&mut |child| self.unstale_in_place(child));
    }

    /// A term as a variable holds it: its own home is the variable it is
    /// stored at, and every term below it that carries one is that home's
    /// variable.
    fn stored(&self, term: InferTy) -> InferTy {
        let mut term = term;
        term.set_home(Home::NONE);
        term.rewrite_children(&mut |child| self.unstale_in_place(child));
        term
    }

    /// The term `root` is bound to, carrying `root` as its home.
    fn bound_term(&self, root: TypeBoundId, term: InferTy) -> InferTy {
        let mut term = term;
        term.set_home(Home::of(root));
        term
    }

    fn shallow_resolve_ty(&self, ty: &InferTy) -> InferTy {
        let Some(root) = self.root_var(ty) else {
            return ty.clone();
        };
        match &self.ty_bounds[root.0 as usize] {
            TypeBound::Resolved { ty: inner, .. } => match inner {
                TyTerm::Var(_) => self.shallow_resolve_ty(inner),
                term => self.bound_term(root, term.clone()),
            },
            _ => TyTerm::Var(root),
        }
    }

    fn resolve_ty(&self, ty: &InferTy) -> InferTy {
        self.unstale(ty).map(
            &mut |id: TypeBoundId| {
                let root = self.find_ty_root(id);
                match &self.ty_bounds[root.0 as usize] {
                    TypeBound::Resolved { ty: inner, .. } => {
                        self.bound_term(root, self.resolve_ty(inner))
                    }
                    _ => TyTerm::Var(root),
                }
            },
            &mut |id: IdentityVarId| self.resolve_identity(&IdentityTerm::Var(id)),
            &mut |id: EffectVarId| self.resolve_effect(&EffectTerm::Var(id)),
            &mut |id: LenVarId| self.resolve_len(&LenTerm::Var(id)),
            &mut |id: ReprVarId| self.resolve_repr(id),
            &mut |id: FlowVarId| self.resolve_flows(&FlowTerm::Var(id)),
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
            TyTerm::Ref(_, inner) => self.occurs_in(id, &inner.ty()),
            TyTerm::Slice(elem) => self.occurs_in(id, elem),
            TyTerm::Str => false,
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
                type_args.iter().any(|t| self.occurs_in(id, &t.ty()))
            }
            TyTerm::Int(_)
            | TyTerm::Float
            | TyTerm::Char
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
        alloc_effect_var(&mut self.effect_vars, EffectVarBound::Any)
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
                    EffectBound::Bound { effect, .. } => EffectTerm::Known(effect.clone()),
                    EffectBound::Range { .. } => EffectTerm::Var(root),
                    EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
                }
            }
        }
    }

    /// The interval of a root: its own, with the lower raised by every
    /// variable placed below it and the upper met with every variable
    /// placed above it, so a join at any variable of the relation is
    /// checked against an upper anywhere above it (RFC-0046 rule 7).
    fn range_of(&self, root: EffectVarId) -> Interval {
        let own = self.own_range(root);
        let lower = self
            .reachable(root, Toward::Below)
            .into_iter()
            .fold(own.lower, |lower, b| lower.join(&self.own_range(b).lower));
        let upper = self
            .reachable(root, Toward::Above)
            .into_iter()
            .fold(own.upper, |upper, a| upper.meet(&self.own_range(a).upper));
        Interval { lower, upper }
    }

    /// Every root related to `root` through `effect_below` in one
    /// direction, transitively, without `root` itself.
    fn reachable(&self, root: EffectVarId, toward: Toward) -> Vec<EffectVarId> {
        let mut seen: Vec<EffectVarId> = vec![root];
        let mut stack: Vec<EffectVarId> = vec![root];
        while let Some(at) = stack.pop() {
            for (b, a) in self.effect_below.iter() {
                let (from, to) = match toward {
                    Toward::Below => (a, b),
                    Toward::Above => (b, a),
                };
                if self.find_effect_root(*from) != at {
                    continue;
                }
                let to = self.find_effect_root(*to);
                if seen.contains(&to) {
                    continue;
                }
                seen.push(to);
                stack.push(to);
            }
        }
        seen.split_off(1)
    }

    fn own_range(&self, root: EffectVarId) -> Interval {
        match &self.effect_vars[root.0 as usize] {
            EffectBound::Range { lower, upper, .. } => Interval {
                lower: lower.clone(),
                upper: upper.clone(),
            },
            EffectBound::Bound { effect, .. } => Interval {
                lower: effect.clone(),
                upper: effect.clone(),
            },
            EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
        }
    }

    /// The declared bound a root carries: the meet of every variable
    /// forwarded to it.
    fn effect_var_bound(&self, root: EffectVarId) -> EffectVarBound {
        match &self.effect_vars[root.0 as usize] {
            EffectBound::Range { bound, .. } | EffectBound::Bound { bound, .. } => *bound,
            EffectBound::Forward(_) => unreachable!("find_effect_root resolves forwards"),
        }
    }

    /// An open effect is the least element of its interval: the join of
    /// what the body requires (RFC-0014, RFC-0042 rule 3).
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
        let bound = self.effect_var_bound(root);
        self.effect_vars[root.0 as usize] = EffectBound::Bound { effect, bound };
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
        let bound = self.effect_var_bound(root);
        self.effect_vars[root.0 as usize] = EffectBound::Range {
            lower,
            upper,
            bound,
        };
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
        let bound = self.effect_var_bound(root);
        self.effect_vars[root.0 as usize] = EffectBound::Range {
            lower,
            upper,
            bound,
        };
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
        let bound = self
            .effect_var_bound(from_root)
            .meet(self.effect_var_bound(to_root));
        self.effect_vars[to_root.0 as usize] = EffectBound::Range {
            lower,
            upper,
            bound,
        };
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

    fn resolve_repr(&self, id: ReprVarId) -> Repr<Infer> {
        let root = self.find_repr_root(id);
        match &self.repr_vars[root.0 as usize] {
            ReprBound::Uniform => Repr::Uniform,
            ReprBound::Specialized(held) => Repr::Specialized(self.resolve_held(held)),
            ReprBound::Unbound(_) => Repr::Var(root),
            ReprBound::Forward(_) => unreachable!("find_repr_root resolves forwards"),
        }
    }

    fn resolve_held(&self, held: &HeldTy<Infer>) -> HeldTy<Infer> {
        held.map(
            &mut |id: TypeBoundId| self.resolve_ty(&TyTerm::Var(id)),
            &mut |id: IdentityVarId| self.resolve_identity(&IdentityTerm::Var(id)),
            &mut |id: EffectVarId| self.resolve_effect(&EffectTerm::Var(id)),
            &mut |id: LenVarId| self.resolve_len(&LenTerm::Var(id)),
            &mut |id: ReprVarId| self.resolve_repr(id),
            &mut |id: FlowVarId| self.resolve_flows(&FlowTerm::Var(id)),
        )
    }

    fn resolve_arg(&self, arg: &TypeArg<Infer>) -> TypeArg<Infer> {
        match arg {
            TypeArg::Uniform(ty) => TypeArg::Uniform(self.resolve_ty(ty)),
            TypeArg::Open(v, ty) => self.resolve_repr(*v).at(self.resolve_ty(ty)),
            TypeArg::Specialized(held) => TypeArg::Specialized(self.resolve_held(held)),
        }
    }

    fn repr_owner(&self, root: ReprVarId) -> ReprOwner {
        match &self.repr_vars[root.0 as usize] {
            ReprBound::Unbound(owner) => *owner,
            ReprBound::Uniform | ReprBound::Specialized(_) | ReprBound::Forward(_) => {
                unreachable!("repr_owner takes an open root")
            }
        }
    }

    /// Two arguments at one specializing position, whose types have
    /// joined (hash-types.md, R3), met part by part. Two fixed
    /// representations must agree at every part. An open one takes the
    /// other, a `#` tree whole, when its owner lets this join bind it: a
    /// local variable at any join, a signature's variable at a decision's
    /// join. A local variable meeting a signature's forwards to it, so the
    /// local name follows the decision; two signatures' variables are two
    /// decisions' names, and only a decision's join — a conversion answered
    /// identity — makes them one.
    fn unify_repr(
        &mut self,
        a: &TypeArg<Infer>,
        b: &TypeArg<Infer>,
        kind: JoinKind,
    ) -> Result<(), MismatchReason> {
        let a = self.resolve_arg(a);
        let b = self.resolve_arg(b);
        self.meet_reprs(&a, &b, kind)
    }

    fn meet_reprs(
        &mut self,
        a: &TypeArg<Infer>,
        b: &TypeArg<Infer>,
        kind: JoinKind,
    ) -> Result<(), MismatchReason> {
        let binds = |terms: &Self, v: ReprVarId| {
            kind == JoinKind::Decision || terms.repr_owner(v) == ReprOwner::Local
        };
        match (a, b) {
            (TypeArg::Open(x, _), TypeArg::Open(y, _)) if x == y => Ok(()),
            (TypeArg::Open(x, _), TypeArg::Open(y, _)) => {
                let (from, to) = match (self.repr_owner(*x), self.repr_owner(*y)) {
                    (ReprOwner::Local, _) => (*x, *y),
                    (ReprOwner::Signature, ReprOwner::Local) => (*y, *x),
                    (ReprOwner::Signature, ReprOwner::Signature) => {
                        if kind != JoinKind::Decision {
                            return Err(MismatchReason::ReprOpen(*x));
                        }
                        (*x, *y)
                    }
                };
                self.repr_vars[from.0 as usize] = ReprBound::Forward(to);
                Ok(())
            }
            (TypeArg::Open(v, _), fixed) | (fixed, TypeArg::Open(v, _)) => {
                if !binds(self, *v) {
                    return Err(MismatchReason::ReprOpen(*v));
                }
                self.repr_vars[v.0 as usize] = match fixed {
                    TypeArg::Uniform(_) => ReprBound::Uniform,
                    TypeArg::Specialized(held) => ReprBound::Specialized(held.clone()),
                    TypeArg::Open(..) => unreachable!("two variables are matched above"),
                };
                Ok(())
            }
            (TypeArg::Uniform(_), TypeArg::Uniform(_)) => Ok(()),
            (TypeArg::Specialized(x), TypeArg::Specialized(y)) => self.meet_held(x, y, kind),
            (TypeArg::Uniform(_), TypeArg::Specialized(_))
            | (TypeArg::Specialized(_), TypeArg::Uniform(_)) => Err(MismatchReason::NoJoin),
        }
    }

    /// Two `#` nodes whose types have joined. A leaf's own arguments have
    /// met already, in the `UserDefined` arm of `join`.
    fn meet_held(
        &mut self,
        a: &HeldTy<Infer>,
        b: &HeldTy<Infer>,
        kind: JoinKind,
    ) -> Result<(), MismatchReason> {
        match (a, b) {
            (HeldTy::Tuple(xs), HeldTy::Tuple(ys)) if xs.len() == ys.len() => {
                for (x, y) in xs.iter().zip(ys) {
                    self.meet_reprs(x, y, kind)?;
                }
                Ok(())
            }
            (HeldTy::Option(x), HeldTy::Option(y))
            | (HeldTy::Array(x, _), HeldTy::Array(y, _))
            | (HeldTy::RustArray(x, _), HeldTy::RustArray(y, _)) => self.meet_reprs(x, y, kind),
            (HeldTy::Result(xo, xe), HeldTy::Result(yo, ye)) => {
                self.meet_reprs(xo, yo, kind)?;
                self.meet_reprs(xe, ye, kind)
            }
            (HeldTy::Leaf(_), HeldTy::Leaf(_)) => Ok(()),
            (HeldTy::Held(_), _) | (_, HeldTy::Held(_)) if a.is_full(false) && b.is_full(false) => {
                Ok(())
            }
            (
                HeldTy::Tuple(_)
                | HeldTy::Option(_)
                | HeldTy::Result(..)
                | HeldTy::Array(..)
                | HeldTy::RustArray(..)
                | HeldTy::Leaf(_)
                | HeldTy::Held(_),
                _,
            ) => Err(MismatchReason::NoJoin),
        }
    }

    // -- Join ----------------------------------------------------------

    /// The join of two types (RFC-0042 rule 1), `a` the value and `b` the
    /// type it flows into. Written to the root variable of a side that is a
    /// variable; `Mismatch` where no join exists. A closure's effect is the
    /// one directional component: the value's is at most what the position
    /// allows (RFC-0046 rule 7). At the top of a decision's own join `a` is not a
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
                .take_other(v, b, b_root, kind == JoinKind::Pattern)
                .map_err(|NoJoin| no_join(self));
        }
        if let Some(v) = unbound(self, b_root) {
            return self
                .take_other(v, a, a_root, false)
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
            (TyTerm::Error(token), other) | (other, TyTerm::Error(token)) => {
                self.poison(other, *token);
                Ok(())
            }
            (TyTerm::Var(_), _) | (_, TyTerm::Var(_)) => {
                unreachable!("an unbound variable took the other side above")
            }

            (TyTerm::Never, TyTerm::Never) => Ok(()),

            (TyTerm::Int(ka), TyTerm::Int(kb)) if ka == kb => Ok(()),
            (TyTerm::Float, TyTerm::Float)
            | (TyTerm::Char, TyTerm::Char)
            | (TyTerm::String, TyTerm::String)
            | (TyTerm::Str, TyTerm::Str)
            | (TyTerm::Bool, TyTerm::Bool)
            | (TyTerm::Unit, TyTerm::Unit)
            | (TyTerm::Order, TyTerm::Order) => Ok(()),

            (TyTerm::Object(oa), TyTerm::Object(ob)) => {
                for (key, ty_a) in oa {
                    if let Some(ty_b) = ob.get(key) {
                        self.join(ty_a, ty_b, Position::Argument, kind, registry)?;
                    }
                }
                match ObjectTy::meet(oa, ob) {
                    ObjectMeet::Joined { ty } => {
                        let pattern = kind == JoinKind::Pattern;
                        let a_side = self.union_side(a_root, &ra, ty.missing_from(oa), false);
                        let b_side = self.union_side(b_root, &rb, ty.missing_from(ob), pattern);
                        self.write_union(a_side, b_side, TyTerm::Object(ty), kind, mismatch)
                    }
                    ObjectMeet::Lacks { declared, field } => Err(mismatch_for(
                        self,
                        MismatchReason::ObjectLacksDeclaredField {
                            declared: declared.name,
                            field,
                        },
                    )),
                    ObjectMeet::Undeclared { declared, field } => Err(mismatch_for(
                        self,
                        MismatchReason::ObjectFieldNotDeclared {
                            declared: declared.name,
                            field,
                        },
                    )),
                    ObjectMeet::TooWide { fields } => {
                        Err(mismatch_for(self, MismatchReason::ObjectTooWide { fields }))
                    }
                    ObjectMeet::TwoDeclarations { .. } => Err(mismatch(self)),
                }
            }
            (
                TyTerm::Enum {
                    name: na,
                    variants: va,
                    ..
                },
                TyTerm::Enum {
                    name: nb,
                    variants: vb,
                    ..
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
                let missing =
                    |of: &FxHashMap<Astr, Option<Box<InferTy>>>,
                     from: &FxHashMap<Astr, Option<Box<InferTy>>>| {
                        of.keys()
                            .filter(|tag| !from.contains_key(tag))
                            .min()
                            .copied()
                    };
                let pattern = kind == JoinKind::Pattern;
                let a_side = self.union_side(a_root, &ra, missing(vb, va), false);
                let b_side = self.union_side(b_root, &rb, missing(va, vb), pattern);
                let mut merged: FxHashMap<Astr, Option<Box<InferTy>>> = va.clone();
                for (tag, payload) in vb {
                    merged.entry(*tag).or_insert_with(|| payload.clone());
                }
                self.write_union(
                    a_side,
                    b_side,
                    TyTerm::Enum {
                        name: *na,
                        variants: merged,
                        home: crate::ty::Home::NONE,
                    },
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
            (TyTerm::Option(ia), TyTerm::Option(ib))
            | (TyTerm::Handle(ia), TyTerm::Handle(ib))
            | (TyTerm::Slice(ia), TyTerm::Slice(ib)) => {
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
                    let reason = match (ma, mb) {
                        (Mutability::Mut, Mutability::Shared) => MismatchReason::Weakens,
                        _ => MismatchReason::NoJoin,
                    };
                    return Err(mismatch_for(self, reason));
                }
                self.join(&ia.ty(), &ib.ty(), Position::Argument, kind, registry)?;
                self.unify_repr(ia, ib, kind)
                    .map_err(|reason| mismatch_for(self, reason))
            }
            (
                TyTerm::Fn {
                    params: pa,
                    ret: ret_a,
                    effect: ea,
                    flows: fa,
                    ..
                },
                TyTerm::Fn {
                    params: pb,
                    ret: ret_b,
                    effect: eb,
                    flows: fb,
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
                if self.meet_flows(fa, fb).is_err() {
                    return Err(mismatch(self));
                }
                Ok(())
            }
            (
                TyTerm::UserDefined {
                    id: id_a,
                    type_args: ta_args,
                    effect_args: ea_args,
                    identity_args: ia_args,
                    region_params: ra,
                },
                TyTerm::UserDefined {
                    id: id_b,
                    type_args: tb_args,
                    effect_args: eb_args,
                    identity_args: ib_args,
                    region_params: rb,
                },
            ) => {
                if id_a != id_b {
                    return Err(mismatch(self));
                }
                assert_eq!(ra, rb, "one declaration states one count of region parameters");
                assert_eq!(ta_args.len(), tb_args.len());
                assert_eq!(ea_args.len(), eb_args.len());
                assert_eq!(ia_args.len(), ib_args.len());
                for (x, y) in ta_args.iter().zip(tb_args.iter()) {
                    self.join(&x.ty(), &y.ty(), Position::Argument, kind, registry)?;
                }
                for (index, (x, y)) in ta_args.iter().zip(tb_args.iter()).enumerate() {
                    if registry.specializes(*id_a, index) {
                        self.unify_repr(x, y, kind)
                            .map_err(|reason| mismatch_for(self, reason))?;
                    } else {
                        debug_assert!(
                            x.is_uniform() && y.is_uniform(),
                            "a slot that does not specialize is uniform by instantiation"
                        );
                    }
                }
                for (x, y) in ea_args.iter().zip(eb_args.iter()) {
                    if x.is_specialized() != y.is_specialized() {
                        return Err(mismatch(self));
                    }
                    if self
                        .unify_effect(x.effect(), y.effect(), EffectRelation::Equal)
                        .is_err()
                    {
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
        other_is_pattern: bool,
    ) -> Result<(), NoJoin> {
        let Some(root) = other_root else {
            let term = self.shallow_resolve_ty(other);
            if integer_bound_refuses(&self.bound_of(var), &term) {
                return Err(NoJoin);
            }
            let growth = Carrier::of_term(&term, other_is_pattern).growth();
            return self.bind_ty(var, term, growth).map_err(|Cyclic| NoJoin);
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
        match &mut self.ty_bounds[root.0 as usize] {
            TypeBound::Unresolved { bound } | TypeBound::Resolved { bound, .. } => *bound = merged,
            TypeBound::Forward(_) => unreachable!("find_ty_root yields a root"),
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
            None => self.bind_ty(root, other.clone(), Carrier::of_term(other, false).growth()),
        }
    }

    /// Writes the union of two structural types. In a flow or a decision
    /// the two sides become one variable; a pattern is a view of its
    /// source and is joined to it only where both are open.
    fn write_union(
        &mut self,
        a: UnionSide,
        b: UnionSide,
        union: InferTy,
        kind: JoinKind,
        mismatch: impl Fn(&Self) -> Mismatch,
    ) -> Result<(), Mismatch> {
        let checked: &[&UnionSide] = match kind {
            JoinKind::Pattern => &[&a],
            JoinKind::Flow | JoinKind::Decision => &[&a, &b],
        };
        for side in checked {
            if let (Carrier::Fixed, Some(member)) = (side.carrier, side.lacks) {
                return Err(Mismatch {
                    reason: MismatchReason::FixedLacks { member },
                    ..mismatch(self)
                });
            }
        }
        match kind {
            JoinKind::Flow | JoinKind::Decision => {
                let growth = if a.carrier == Carrier::Fixed || b.carrier == Carrier::Fixed {
                    Growth::Fixed
                } else {
                    Growth::Open
                };
                match (a.root, b.root) {
                    (Some(ra), Some(rb)) => self
                        .merge_ty(ra, rb, union, growth)
                        .map_err(|NoJoin| mismatch(self)),
                    (Some(root), None) | (None, Some(root)) => self
                        .bind_ty(root, union, growth)
                        .map_err(|Cyclic| mismatch(self)),
                    (None, None) => Ok(()),
                }
            }
            JoinKind::Pattern => match (a.open_root(), b.open_root()) {
                (Some(ra), Some(rb)) => self
                    .merge_ty(ra, rb, union, Growth::Open)
                    .map_err(|NoJoin| mismatch(self)),
                (ra, rb) => {
                    for root in [ra, rb].into_iter().flatten() {
                        self.bind_ty(root, union.clone(), Growth::Open)
                            .map_err(|Cyclic| mismatch(self))?;
                    }
                    Ok(())
                }
            },
        }
    }

    /// Poison rule 1 (docs/solver.md): every variable still open inside a
    /// type joined with poison is bound to it.
    fn poison(&mut self, ty: &InferTy, token: ErrorToken) {
        let mut open = Vec::new();
        collect_open_vars(&self.resolve_ty(ty), &mut open);
        for var in open {
            if matches!(self.ty_bounds[var.0 as usize], TypeBound::Unresolved { .. }) {
                self.bind_ty(var, TyTerm::Error(token), Growth::Fixed)
                    .expect("poison contains no variable");
            }
        }
    }

    /// Makes two resolved variables one variable naming `union`.
    fn merge_ty(
        &mut self,
        a: TypeBoundId,
        b: TypeBoundId,
        union: InferTy,
        growth: Growth,
    ) -> Result<(), NoJoin> {
        let union = self.stored(union);
        let (a, b) = (self.find_ty_root(a), self.find_ty_root(b));
        let bound = self.bound_of(a).meet(&self.bound_of(b)).ok_or(NoJoin)?;
        if a != b {
            self.forward_ty(b, a).map_err(|Cyclic| NoJoin)?;
        }
        if self.occurs_in(a, &union) {
            return Err(NoJoin);
        }
        self.ty_bounds[a.0 as usize] = TypeBound::Resolved {
            ty: union,
            bound,
            growth,
        };
        Ok(())
    }

    fn union_side(
        &self,
        root: Option<TypeBoundId>,
        term: &InferTy,
        lacks: Option<Astr>,
        pattern: bool,
    ) -> UnionSide {
        let carrier = match root {
            Some(root) => match self.growth_of(root) {
                Growth::Open => Carrier::Open,
                Growth::Fixed => Carrier::Fixed,
            },
            None => Carrier::of_term(term, pattern),
        };
        UnionSide {
            root,
            lacks,
            carrier,
        }
    }

    fn growth_of(&self, root: TypeBoundId) -> Growth {
        match &self.ty_bounds[self.find_ty_root(root).0 as usize] {
            TypeBound::Resolved { growth, .. } => *growth,
            TypeBound::Unresolved { .. } => Growth::Open,
            TypeBound::Forward(_) => unreachable!("find_ty_root yields a root"),
        }
    }
}

/// A direction along `Terms::effect_below`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Toward {
    Below,
    Above,
}

/// `lower <= var <= upper` on the reissue chain (RFC-0013 rule 1).
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
        self.instantiate_open_beside(ty, &[], &[], registry).ty
    }

    /// `ty` and `beside` at one set of fresh variables: a placeholder `ty`
    /// and a pattern in `beside` both name is one solver variable.
    fn instantiate_open_beside(
        &mut self,
        ty: &PolyTy,
        beside: &[&PolyTy],
        effect_bounds: &[EffectVarBound],
        registry: &TypeRegistry,
    ) -> OpenInstance {
        let mut maps = PolyMaps::default();
        let mut bounded_effects: Vec<EffectVarId> = Vec::new();
        let Terms {
            ty_bounds,
            effect_vars,
            len_vars,
            identity_vars,
            ..
        } = self;
        let mut instantiate = |poly: &PolyTy| {
            poly.map(
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
                    EffectTerm::Var(*maps.effect.entry(id).or_insert_with(|| {
                        let bound = effect_bound_at(effect_bounds, id);
                        let fresh = alloc_effect_var(effect_vars, bound);
                        if bound != EffectVarBound::Any {
                            bounded_effects.push(fresh);
                        }
                        fresh
                    }))
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
                &mut crate::ty::no_flow_var,
            )
        };
        let mut instance = instantiate(ty);
        let mut beside: Vec<InferTy> = beside.iter().map(|poly| instantiate(poly)).collect();
        self.open_flows(&mut instance);
        for poly in &mut beside {
            self.open_flows(poly);
        }
        OpenInstance {
            ty: uniform_slots(instance, registry),
            beside: beside
                .into_iter()
                .map(|poly| uniform_slots(poly, registry))
                .collect(),
            bounded_effects,
        }
    }

    /// Whether the call type would join an instance's signature, on a copy
    /// of the terms: the solver's own lattice says what "could match" means.
    fn would_take(
        &self,
        call: &InferTy,
        signature: &PolyTy,
        bound: EffectBoundedBy,
        registry: &TypeRegistry,
    ) -> bool {
        let mut trial = self.clone();
        let instance = trial.instantiate_open(signature, registry);
        let Flow { value, into } = bound.ordered(CallOfInstance {
            call,
            instance: &instance,
        });
        trial
            .join(value, into, Position::Value, JoinKind::Decision, registry)
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

#[derive(Debug, Clone, Copy)]
struct UnionSide {
    root: Option<TypeBoundId>,
    /// A member the other side has and this one does not.
    lacks: Option<Astr>,
    carrier: Carrier,
}

impl UnionSide {
    fn open_root(&self) -> Option<TypeBoundId> {
        self.root.filter(|_| self.carrier == Carrier::Open)
    }
}

/// What a side of a structural join can hold of the union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carrier {
    /// A variable that may gain members.
    Open,
    /// A layout the body did not choose.
    Fixed,
    /// A lower bound, which holds what it names and asks nothing of the
    /// rest: a read's `AtLeast` object, or a pattern.
    Bound,
}

impl Carrier {
    /// The carrier of a term no variable names.
    fn of_term(term: &InferTy, pattern: bool) -> Self {
        match term {
            _ if pattern => Carrier::Bound,
            TyTerm::Object(object) if object.field_set() == FieldSet::AtLeast => Carrier::Bound,
            _ => Carrier::Fixed,
        }
    }

    fn growth(self) -> Growth {
        match self {
            Carrier::Open | Carrier::Bound => Growth::Open,
            Carrier::Fixed => Growth::Fixed,
        }
    }
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

fn collect_open_vars(ty: &InferTy, out: &mut Vec<TypeBoundId>) {
    if let TyTerm::Var(var) = ty {
        out.push(*var);
    }
    for child in ty.children() {
        collect_open_vars(&child, out);
    }
}

fn alloc_ty_var(ty_bounds: &mut PVec<TypeBound>, bound: TyVarBound) -> TypeBoundId {
    let id = TypeBoundId(ty_bounds.len() as u32);
    ty_bounds.push(TypeBound::Unresolved { bound });
    id
}

fn alloc_identity_var(identity_vars: &mut PVec<IdentityBound>) -> IdentityVarId {
    let id = IdentityVarId(identity_vars.len() as u32);
    identity_vars.push(IdentityBound::Unbound);
    id
}

fn alloc_len_var(len_vars: &mut PVec<LenBound>) -> LenVarId {
    let id = LenVarId(len_vars.len() as u32);
    len_vars.push(LenBound::Unbound);
    id
}

/// A fresh effect variable spans the whole chain whatever its bound: a
/// floor asserted here would lift a `Sync` effect rather than refuse it
/// (RFC-0011 rule 5).
fn alloc_effect_var(effect_vars: &mut PVec<EffectBound>, bound: EffectVarBound) -> EffectVarId {
    let id = EffectVarId(effect_vars.len() as u32);
    effect_vars.push(EffectBound::Range {
        lower: Effect::PURE,
        upper: Effect::TOP,
        bound,
    });
    id
}

/// The flows a declared instance's type states; `None` where the type is no
/// function type, which no instance of a signature is.
fn declared_flows(ty: &PolyTy) -> Option<Flows> {
    match ty {
        TyTerm::Fn { flows, .. } => Some(flows.get().clone()),
        _ => None,
    }
}

fn alloc_flow_var(flow_vars: &mut PVec<FlowBound>, seed: Flows) -> FlowVarId {
    let id = FlowVarId(flow_vars.len() as u32);
    flow_vars.push(FlowBound::Root(seed));
    id
}

fn alloc_repr_var(repr_vars: &mut PVec<ReprBound>, owner: ReprOwner) -> ReprVarId {
    let id = ReprVarId(repr_vars.len() as u32);
    repr_vars.push(ReprBound::Unbound(owner));
    id
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
    /// The language's own instance at an operator over a word or text: the
    /// operator's instruction (RFC-0020).
    Operator,
    /// The language's own instance at a structural type (RFC-0020).
    Structural,
}

/// The compiler's instances of a shared signature at one use, and which of
/// the declared instances they keep from it.
pub struct CompilerInstances {
    pub candidates: Vec<Candidate>,
    pub withholds: Withholds,
    pub structural: Option<ComponentOffer>,
}

impl CompilerInstances {
    pub fn none() -> Self {
        Self {
            candidates: Vec::new(),
            withholds: Withholds::SameType,
            structural: None,
        }
    }
}

/// Obligation across modules: `typeck` fills `candidates` with the
/// language's word and text instances and the registry's instances at every
/// other type, since a component's decision offers this same set and
/// `Structural` again (RFC-0020).
#[derive(Debug)]
pub struct ComponentOffer {
    pub signature: QualifiedRef,
    pub shape: StructuralSignature,
    pub params: Vec<Astr>,
    pub candidates: Vec<Candidate>,
}

impl ComponentOffer {
    fn call_at(&self, component: &InferTy) -> InferTy {
        let taken = || {
            TyTerm::Ref(
                Mutability::Shared,
                Box::new(TypeArg::uniform(component.clone())),
            )
        };
        let ret = match self.shape {
            StructuralSignature::Eq => TyTerm::Bool,
            StructuralSignature::Clone => component.clone(),
        };
        TyTerm::Fn {
            params: self
                .params
                .iter()
                .map(|name| ParamTerm::new(*name, taken()))
                .collect(),
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::PURE.into(),
            flows: Flows::Every.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuralHead {
    Takes,
    Open,
    Refuses,
}

#[derive(Debug, Clone)]
pub struct SettledStructural {
    pub call: InferTy,
    pub offer: Rc<ComponentOffer>,
    pub children: Vec<(Component, DecisionId)>,
}

/// The declared instances a use does not reach beside the compiler's own.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Withholds {
    /// The ones at a type the compiler has its own instance for: a named
    /// call of `clone` on a `String` is the instruction.
    SameType,
    /// Every one at a language-owned type: an operator over a word or text
    /// is an instruction or nothing (RFC-0020).
    LanguageOwned,
}

/// Whether an instance's first parameter takes a scalar or text.
pub fn takes_a_language_owned_type(ty: &PolyTy) -> bool {
    let TyTerm::Fn { params, .. } = ty else {
        return false;
    };
    let Some(first) = params.first() else {
        return false;
    };
    let taken = match &first.ty {
        TyTerm::Ref(_, named) => &named.ty(),
        other => other,
    };
    taken.is_scalar() || matches!(taken, TyTerm::String | TyTerm::Str)
}

/// A concrete instance an instance decision may still settle on.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub instance: InstanceKind,
    pub ty: PolyTy,
    pub admits: Task,
    /// Written at this candidate's own variables, as `ty` is (RFC-0070 rule 3).
    pub requires: Vec<RequirementSig>,
    pub effect_bounds: Vec<EffectVarBound>,
}

/// Which side of an instance's join holds the effect the other stays
/// within: a call takes its instance's, a required instance stays within
/// its requirer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectBoundedBy {
    TheInstance,
    TheRequirer,
}

impl EffectBoundedBy {
    fn of(required: Option<QualifiedRef>) -> Self {
        match required {
            Some(_) => EffectBoundedBy::TheRequirer,
            None => EffectBoundedBy::TheInstance,
        }
    }

    fn ordered<'t>(self, ran: CallOfInstance<'t>) -> Flow<'t> {
        let CallOfInstance { call, instance } = ran;
        match self {
            EffectBoundedBy::TheInstance => Flow {
                value: call,
                into: instance,
            },
            EffectBoundedBy::TheRequirer => Flow {
                value: instance,
                into: call,
            },
        }
    }
}

/// A call type and the type of an instance it may run.
#[derive(Clone, Copy)]
struct CallOfInstance<'t> {
    call: &'t InferTy,
    instance: &'t InferTy,
}

/// The two sides of a join, in the order `Terms::join` reads them.
struct Flow<'t> {
    value: &'t InferTy,
    into: &'t InferTy,
}

/// A requirement's call type at `Requirement::calls`, the task of the
/// marker `acvus-extern-macro` read off the requirer's `Instance` parameter.
fn called_at(call: InferTy, calls: Task) -> InferTy {
    match call {
        TyTerm::Fn {
            params,
            ret,
            captures,
            effect: EffectTerm::Known(effect),
            flows,
        } => TyTerm::Fn {
            params,
            ret,
            captures,
            effect: EffectTerm::Known(effect.at_task(calls)),
            flows,
        },
        other => other,
    }
}

/// What still separates the candidates of an instance decision: their
/// types, or, once the call has joined the one type they share, only the
/// task each admits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceTie {
    Types,
    TaskOnly,
}

/// The generic instance of a function: the uniform one, whose signature a
/// call type with a specialized slot does not match (hash-types.md, R3).
#[derive(Debug, Clone)]
pub struct GenericInstance {
    pub instance: usize,
    pub ty: PolyTy,
}

/// A position with more than one admissible answer (RFC-0042 rule 2).
#[derive(Debug, Clone)]
pub enum Decision {
    /// Which instance of an Extern function a call runs (RFC-0019,
    /// RFC-0040).
    Instance {
        call: InferTy,
        candidates: Vec<Candidate>,
        generic: Option<GenericInstance>,
        /// The signature a requirement asked an instance of (RFC-0068 rule 5);
        /// a call's own decision is named by its site.
        required: Option<QualifiedRef>,
        tie: InstanceTie,
        structural: Option<Rc<ComponentOffer>>,
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
        awaiting_head: Vec<UnjoinedArgument>,
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
    },
    /// How a lambda's body reads a name it captured (RFC-0018). The
    /// checker opens this only where the head of `of` is still a
    /// variable; a known head it answers on the spot, through
    /// `Solver::capture_read`.
    Capture { of: InferTy, seen: InferTy },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lend {
    Reference,
    Reborrow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureRead {
    Word,
    Lent,
}

#[derive(Debug, Clone)]
pub enum CaptureOutcome {
    HeadOpen,
    Reads { read: CaptureRead, seen: InferTy },
}

#[derive(Debug, Clone)]
pub enum LendOutcome {
    HeadOpen,
    Names { referent: InferTy, lend: Lend },
    MutableBorrowOfShared { referent: InferTy },
}

/// RFC-0030.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverMode {
    Lent(Mutability),
    Value,
}

/// RFC-0043.
#[derive(Debug, Clone, PartialEq)]
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
    /// The arguments this candidate takes as a view of what the caller
    /// lent: a `&String` at a `&str` parameter (RFC-0062 rule 3).
    pub viewed: Vec<ConvertedArgument>,
}

impl SignatureOption {
    pub fn taking_every_argument_directly(candidate: SignatureCandidate) -> Self {
        Self {
            candidate,
            converted: Vec::new(),
            viewed: Vec::new(),
        }
    }
}

/// An argument the checker held rather than joined with the call's
/// parameter (RFC-0043 rule 2): its head, or for a receiver the mode it is
/// taken in (rule 6), is the settled candidate's to answer, so `admits`
/// answers it afresh at every step of the decision, at the type `seen_by`
/// says that candidate sees it as.
#[derive(Debug, Clone)]
pub struct UnjoinedArgument {
    pub index: usize,
    pub ty: InferTy,
    pub handed: Handed,
}

/// How the call hands a held argument to a candidate.
#[derive(Debug, Clone)]
pub enum Handed {
    AsIs,
    /// `ty` is the type of a receiver's place, handed in the mode the
    /// candidate's first parameter asks (RFC-0043 rule 6); `referent` is what
    /// a reference to the place names.
    Receiver {
        referent: InferTy,
    },
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
/// under RFC-0046 rule 7's demotion, which leaves the call's free to freeze below
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
    fn at_effect(&self, effect: EffectTerm<Infer>, flows: FlowTerm<Infer>) -> InferTy {
        TyTerm::Fn {
            params: self.params.clone(),
            ret: Box::new(self.ret.clone()),
            captures: vec![],
            effect,
            flows,
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
    pub awaiting_head: Vec<UnjoinedArgument>,
    pub body_effect: EffectTerm<Infer>,
}

/// How one candidate takes one argument (RFC-0043).
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    Direct,
    /// A `&mut` reaching a `&` parameter as `shared`: as direct as `Direct`
    /// (RFC-0029 rule 3), and not joined, since the argument keeps its
    /// `&mut`.
    Reborrowed {
        shared: InferTy,
    },
    Converted,
    /// The argument reaches the parameter as a view of the storage it
    /// lends, which the checker takes at the argument and the decision
    /// therefore cannot unify (RFC-0062 rule 3).
    Viewed(Viewed),
    Refused,
}

impl Admission {
    fn is_direct(&self) -> bool {
        matches!(self, Admission::Direct | Admission::Reborrowed { .. })
    }
}

/// RFC-0043 rule 1: a candidate that takes the argument only by conversion
/// or by view leaves the set where another takes it directly.
#[derive(Clone, Copy)]
pub struct Kept {
    direct: bool,
}

impl Kept {
    pub fn among<'a, A>(admissions: A) -> Self
    where
        A: IntoIterator<Item = &'a Admission>,
    {
        Self {
            direct: admissions.into_iter().any(Admission::is_direct),
        }
    }

    pub fn keeps(self, admission: &Admission) -> bool {
        match admission {
            Admission::Direct | Admission::Reborrowed { .. } => true,
            Admission::Converted | Admission::Viewed(_) => !self.direct,
            Admission::Refused => false,
        }
    }
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
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Instance(InstanceKind),
    Conversion(Conversion),
    /// `candidate` is the one the decision settled on, which an argument it
    /// held unjoined is admitted by again once the solve names its head
    /// (RFC-0043 rule 2).
    Signature {
        candidate: SignatureCandidate,
        settled: SettledSignature,
        callee_ty: InferTy,
    },
    Lend(Lend),
    Capture(CaptureRead),
    Match(MatchMode),
}

/// `bounded` and `bounded_effects` are verified by the checker when the
/// body freezes, as an `Instantiated`'s are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettledSignature {
    Named {
        qref: QualifiedRef,
        instance: Option<InstanceChoice>,
        bounded: Vec<TypeBoundId>,
        bounded_effects: Vec<EffectVarId>,
        requirements: Vec<RequiredDecision>,
    },
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    Identity,
    Cast(QualifiedRef),
    /// The lowering does not cast the reference value: it runs `cast` on
    /// the value the argument borrows, into a temporary the call borrows,
    /// and `back` on the temporary after it, stored back into the place, so
    /// the callee's writes land in the caller's storage (RFC-0041).
    ThroughRef {
        mutability: Mutability,
        cast: QualifiedRef,
        back: QualifiedRef,
    },
    /// A `&mut` reaches a `&` of what it names as the shared reborrow `&r`
    /// gives (RFC-0029 rule 3).
    Reborrow,
}

/// Why a decision did not settle.
#[derive(Debug, Clone)]
pub enum Unsettled {
    /// No instance's signature matches the call type; `instances` are the
    /// signatures the decision still held when the last was refused.
    NoInstance {
        decision: DecisionId,
        call: InferTy,
        instances: Vec<PolyTy>,
        required: Option<QualifiedRef>,
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
    pub fn types(&self) -> Vec<&InferTy> {
        match self {
            Unsettled::NoInstance { call, .. } | Unsettled::AmbiguousInstance { call, .. } => {
                vec![call]
            }
            Unsettled::InstanceMismatch { expected, got, .. }
            | Unsettled::LendMismatch { expected, got, .. }
            | Unsettled::MatchMismatch { expected, got, .. } => vec![expected, got],
            Unsettled::NoConversion { from, to, .. }
            | Unsettled::AmbiguousConversion { from, to, .. }
            | Unsettled::ConversionOpen { from, to, .. } => vec![from, to],
            Unsettled::TaskTooHigh { .. }
            | Unsettled::NoSignature { .. }
            | Unsettled::AmbiguousSignature { .. }
            | Unsettled::EffectExceeded { .. }
            | Unsettled::MutableBorrowOfShared { .. } => Vec::new(),
        }
    }

    pub fn decision(&self) -> DecisionId {
        match self {
            Unsettled::NoInstance { decision, .. }
            | Unsettled::InstanceMismatch { decision, .. }
            | Unsettled::AmbiguousInstance { decision, .. }
            | Unsettled::TaskTooHigh { decision, .. }
            | Unsettled::NoConversion { decision, .. }
            | Unsettled::AmbiguousConversion { decision, .. }
            | Unsettled::ConversionOpen { decision, .. }
            | Unsettled::NoSignature { decision, .. }
            | Unsettled::AmbiguousSignature { decision, .. }
            | Unsettled::EffectExceeded { decision, .. }
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
    /// The flows the instance an instance decision settled on declares
    /// (RFC-0096 rule 4), written where it settles; `None` while it is open
    /// or failed, and where it settled structurally, on no declared type.
    instance_flows: Option<Flows>,
}

/// Every decision of the solve, by id, and the ids of its signature
/// decisions, which `open` records.
///
/// `awaits_signature` asks, for each open conversion, whether an open
/// signature decision has the conversion's target as a parameter. Asking
/// it of every decision cost their number per conversion, quadratic in an
/// array literal's elements; asking it of the signatures costs the calls.
#[derive(Debug, Clone, Default)]
struct Decisions {
    slots: Vec<DecisionSlot>,
    signatures: Vec<DecisionId>,
}

impl Decisions {
    fn open(&mut self, decision: Decision) -> DecisionId {
        let id = DecisionId(self.slots.len() as u32);
        if let Decision::Signature { .. } = decision {
            self.signatures.push(id);
        }
        self.slots.push(DecisionSlot {
            decision,
            state: DecisionState::Open,
            instance_flows: None,
        });
        id
    }

    fn signatures(&self) -> impl Iterator<Item = &DecisionSlot> {
        self.signatures.iter().map(|id| &self.slots[id.0 as usize])
    }
}

impl std::ops::Deref for Decisions {
    type Target = [DecisionSlot];

    fn deref(&self) -> &[DecisionSlot] {
        &self.slots
    }
}

impl std::ops::DerefMut for Decisions {
    fn deref_mut(&mut self) -> &mut [DecisionSlot] {
        &mut self.slots
    }
}

/// What one settlement round did to a decision.
enum Progress {
    Unchanged,
    Narrowed,
    Settled(Answer),
    Failed(Unsettled),
    /// A signature decision an argument of poison reached (RFC-0043): the
    /// call is poison too and reports nothing of its own, since the refusal
    /// the poison came from is the one the reader is told.
    Poisoned,
}

// -- Solver ---------------------------------------------------------------

/// A source that began where a decision settled on an instance, for the
/// checker to place at that decision's site.
#[derive(Debug, Clone, Copy)]
pub struct BegunSource {
    pub decision: DecisionId,
    pub source: IdentityId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequiredDecision {
    pub signature: QualifiedRef,
    pub id: DecisionId,
}

#[derive(Debug, Clone, Copy)]
pub struct OpenedChild {
    pub parent: DecisionId,
    pub child: DecisionId,
    pub component: Option<Component>,
}

struct RequiredCall {
    requirement: RequirementSig,
    pattern: InferTy,
}

/// A bounded effect variable of the instance a decision settled on. The
/// solver only records it: the checker takes these and verifies each at
/// the decision's call when the variable freezes, as it verifies a named
/// call's (RFC-0011 rule 5), and a bound nobody takes is never checked.
#[derive(Debug, Clone, Copy)]
pub struct InstanceBound {
    pub decision: DecisionId,
    pub var: EffectVarId,
}

struct CandidateAt {
    ty: InferTy,
    required: Vec<RequiredCall>,
    bounded_effects: Vec<EffectVarId>,
}

pub struct Solver<'src> {
    terms: Terms,
    decisions: Decisions,
    begun_by_decisions: Vec<BegunSource>,
    children: FxHashMap<DecisionId, Vec<RequiredDecision>>,
    structural: FxHashMap<DecisionId, SettledStructural>,
    opened_children: Vec<OpenedChild>,
    instance_bounds: Vec<InstanceBound>,
    /// Mints a new source for every identity a declaration introduces;
    /// lent by the compilation for this solver's lifetime.
    sources: &'src mut Sources,
    /// The user-defined types and cast rules of the compilation: which
    /// slots specialize (hash-types.md R1) and which conversions exist.
    registry: &'src TypeRegistry,
    /// The instances of every shared signature, for the requirements a
    /// settled instance opens. They cannot be carried by the candidate:
    /// `clone` at `Vec<T>` requires `clone`, whose instances include it
    /// (RFC-0070 rule 3).
    signatures: &'src FxHashMap<QualifiedRef, Instances>,
}

/// What `Solver::snapshot` took.
#[derive(Clone)]
pub struct SolverSnapshot {
    terms: Terms,
    decisions: Decisions,
    begun_by_decisions: Vec<BegunSource>,
    children: FxHashMap<DecisionId, Vec<RequiredDecision>>,
    structural: FxHashMap<DecisionId, SettledStructural>,
    opened_children: Vec<OpenedChild>,
    instance_bounds: Vec<InstanceBound>,
    sources: Sources,
}

impl<'src> Solver<'src> {
    pub fn new(
        sources: &'src mut Sources,
        registry: &'src TypeRegistry,
        signatures: &'src FxHashMap<QualifiedRef, Instances>,
    ) -> Self {
        Self {
            terms: Terms::new(),
            decisions: Decisions::default(),
            begun_by_decisions: Vec::new(),
            children: FxHashMap::default(),
            structural: FxHashMap::default(),
            opened_children: Vec::new(),
            instance_bounds: Vec::new(),
            sources,
            registry,
            signatures,
        }
    }

    pub fn registry(&self) -> &'src TypeRegistry {
        self.registry
    }

    /// Everything the solve has recorded, to go back to: a component of
    /// the call graph is checked again from here when its members' flows
    /// grew (RFC-0079 rule 5), and the round before leaves nothing behind.
    pub fn snapshot(&self) -> SolverSnapshot {
        SolverSnapshot {
            terms: self.terms.clone(),
            decisions: self.decisions.clone(),
            begun_by_decisions: self.begun_by_decisions.clone(),
            children: self.children.clone(),
            structural: self.structural.clone(),
            opened_children: self.opened_children.clone(),
            instance_bounds: self.instance_bounds.clone(),
            sources: self.sources.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: SolverSnapshot) {
        let SolverSnapshot {
            terms,
            decisions,
            begun_by_decisions,
            children,
            structural,
            opened_children,
            instance_bounds,
            sources,
        } = snapshot;
        self.terms = terms;
        self.decisions = decisions;
        self.begun_by_decisions = begun_by_decisions;
        self.children = children;
        self.structural = structural;
        self.opened_children = opened_children;
        self.instance_bounds = instance_bounds;
        *self.sources = sources;
    }

    pub fn trial<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Solver<'_>) -> R,
    {
        let mut sources = self.sources.clone();
        let mut trial = Solver {
            terms: self.terms.clone(),
            decisions: self.decisions.clone(),
            begun_by_decisions: self.begun_by_decisions.clone(),
            children: self.children.clone(),
            structural: self.structural.clone(),
            opened_children: self.opened_children.clone(),
            instance_bounds: self.instance_bounds.clone(),
            sources: &mut sources,
            registry: self.registry,
            signatures: self.signatures,
        };
        f(&mut trial)
    }

    // -- Sources -----------------------------------------------------

    pub fn source_begins_at(&mut self, id: IdentityId, span: Span) {
        self.sources.begins_at(id, span);
    }

    pub fn name_source(&mut self, id: IdentityId, name: Astr) {
        self.sources.named(id, name);
    }

    pub fn source_origin(&self, id: IdentityId) -> Option<Origin> {
        self.sources.origin(id)
    }

    // -- Fresh variables ---------------------------------------------

    pub fn fresh_ty_var(&mut self) -> InferTy {
        self.fresh_var_with(TyVarBound::Any)
    }

    pub fn fresh_var_with(&mut self, bound: TyVarBound) -> InferTy {
        TyTerm::Var(self.terms.alloc_ty_var(bound))
    }

    /// The type of a value the body constructs: a variable, so that every
    /// name the value flows to shares whatever the type gains.
    pub fn construct(&mut self, term: InferTy) -> InferTy {
        let var = self.terms.alloc_ty_var(TyVarBound::Any);
        self.terms
            .bind_ty(var, term, Growth::Open)
            .expect("a fresh variable occurs in no term");
        TyTerm::Var(var)
    }

    /// A fresh variable for an integer literal: its bound is the set of
    /// widths still admissible (RFC-0037), `i64` where nothing narrows it.
    pub fn fresh_int_var(&mut self) -> InferTy {
        TyTerm::Var(self.terms.alloc_ty_var(TyVarBound::Integer {
            signed: false,
            among: IntTy::ALL.to_vec(),
        }))
    }

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Infer> {
        EffectTerm::Var(self.terms.alloc_effect_var())
    }

    /// The flows of a function type the checker writes, which the types it
    /// meets and the body it is inferred from add to.
    pub fn fresh_flow_var(&mut self) -> FlowTerm<Infer> {
        FlowTerm::Var(self.terms.alloc_flow_var(Flows::none()))
    }

    /// What a function type's flows are now.
    pub fn flows_of(&self, term: &FlowTerm<Infer>) -> Flows {
        self.terms.freeze_flows(term)
    }

    /// `flows` joined into the flows `term` names. A known term is a
    /// declaration's and never grows, so it already covers what it is met
    /// with; `Err` is a body whose flows it does not cover.
    pub fn raise_flows(&mut self, term: &FlowTerm<Infer>, flows: &Flows) -> Result<Raised, FixedFlowsDiffer> {
        match term {
            FlowTerm::Var(var) => Ok(self.terms.raise_flows(*var, flows)),
            FlowTerm::Known(known) => match known.covers(flows) {
                true => Ok(Raised::Held),
                false => Err(FixedFlowsDiffer),
            },
        }
    }

    pub fn decide_signature(&mut self, call: UndecidedCall) -> DecisionId {
        let UndecidedCall {
            name,
            call,
            options,
            awaiting_head,
            body_effect,
        } = call;
        self.decide(Decision::Signature {
            name,
            call,
            options,
            awaiting_head,
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

    pub fn fresh_repr_var(&mut self) -> ReprVarId {
        self.terms.alloc_repr_var(ReprOwner::Local)
    }

    // -- Unify -------------------------------------------------------

    /// Two types that must be one type, at a value position (RFC-0042 rule 1):
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
    /// A captured value flows into the type the body reads it at, so a
    /// captured function's effect is at most the one its callers see.
    fn capture_flows(&mut self, captured: &InferTy, seen: &InferTy) -> Result<(), Mismatch> {
        self.settle_join(captured, seen)
    }

    fn settle_join(&mut self, a: &InferTy, b: &InferTy) -> Result<(), Mismatch> {
        self.terms
            .join(a, b, Position::Value, JoinKind::Decision, self.registry)
    }

    /// The call joined with the instance it runs. A call takes its
    /// instance's effect, so the task `tightest_admitting` reads off the call
    /// is the instance's own; a required instance only stays within its
    /// requirer's.
    fn settle_instance(
        &mut self,
        id: DecisionId,
        ran: CallOfInstance<'_>,
        bound: EffectBoundedBy,
    ) -> Result<(), Unsettled> {
        let CallOfInstance { call, instance } = ran;
        let Flow { value, into } = bound.ordered(ran);
        self.settle_join(value, into)
            .map_err(
                |Mismatch { expected, got, .. }| Unsettled::InstanceMismatch {
                    decision: id,
                    expected,
                    got,
                },
            )?;
        let (
            EffectBoundedBy::TheInstance,
            TyTerm::Fn { effect: called, .. },
            TyTerm::Fn { effect: run, .. },
        ) = (bound, call, instance)
        else {
            return Ok(());
        };
        self.terms
            .unify_effect(called, run, EffectRelation::Equal)
            .map_err(|conflict| Unsettled::EffectExceeded {
                decision: id,
                conflict,
            })
    }

    /// Whether `settle_join(a, b)` would succeed, on a copy of the terms.
    fn trial_settle_join(&self, a: &InferTy, b: &InferTy) -> Result<(), Mismatch> {
        let mut trial = self.terms.clone();
        trial.join(a, b, Position::Value, JoinKind::Decision, self.registry)
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

    /// Poison rule 1 (docs/solver.md) for a type a refusal named: what it
    /// left open is poison from here on.
    pub fn poison(&mut self, ty: &InferTy) {
        self.terms.poison(ty, ErrorToken::new());
    }

    /// The variables still open in `ty`, by their roots.
    pub fn open_vars(&self, ty: &InferTy) -> Vec<TypeBoundId> {
        let mut open = Vec::new();
        collect_open_vars(&self.terms.resolve_ty(ty), &mut open);
        open
    }

    /// The declared bound a type variable's root carries.
    pub fn bound_of_var(&self, id: TypeBoundId) -> TyVarBound {
        self.terms.bound_of(self.terms.find_ty_root(id))
    }

    // -- Decide --------------------------------------------------------

    pub fn decisions_opened(&self) -> usize {
        self.decisions.len()
    }

    pub fn decide(&mut self, decision: Decision) -> DecisionId {
        self.decisions.open(decision)
    }

    /// The answer of a settled decision; `None` while it is open or after
    /// it failed.
    /// RFC-0096 rule 4: the flows the instance decision `id` settled on
    /// declares, which a call of a shared signature reads as it reads that
    /// instance's effect. `None` where `id` is open or failed, and where it
    /// settled structurally: the call's own type's flows, the signature's,
    /// stand there.
    pub fn settled_instance_flows(&self, id: DecisionId) -> Option<&Flows> {
        let slot = &self.decisions[id.0 as usize];
        match slot.state {
            DecisionState::Settled(Answer::Instance(_)) => slot.instance_flows.as_ref(),
            DecisionState::Settled(_) | DecisionState::Open | DecisionState::Failed => None,
        }
    }

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
                    Progress::Poisoned => {
                        self.decisions[index].state = DecisionState::Failed;
                        progressed = true;
                    }
                }
            }
            if !progressed {
                progressed = self.open_grown_components();
            }
        }
        failures
    }

    /// Settle, then close every decision still open by its least element
    /// (RFC-0042 rule 3): a width is `i64`, a text `String`, a representation `Uniform`, a lend
    /// a reference, a capture a word, an identity a source of its own;
    /// then settle again, and report every decision that neither settled
    /// nor could take a least element.
    ///
    /// `tags` are the types of a template's `{{ x }}`: one no use settled,
    /// or what a reference among them names, is a `String` once the lends
    /// are closed (RFC-0071 rule 3). Which of text or `core::display` a tag
    /// is, the checker decides once this returns.
    pub fn solve(&mut self, tags: &[InferTy]) -> Vec<Unsettled> {
        let mut failures = self.settle();
        // A pattern's mode closes first. Before a width or a text takes its
        // least element: no least element is a reference, so a head still
        // open here closes to `Value` in either order, and the join that
        // closing makes is what carries a type between the scrutinee and
        // its referent before either is defaulted apart. Before a lend: a
        // binding closed to a value is what a lend of that name then lends,
        // and a lend closed first would name a referent the pattern had not
        // yet settled, which is how a `&&T` would be formed (RFC-0029).
        self.close_matches_by_least_element(&mut failures);
        failures.extend(self.settle());
        for index in 0..self.terms.ty_bounds.len() {
            let TypeBound::Unresolved { bound } = &self.terms.ty_bounds[index] else {
                continue;
            };
            let ty = match bound.integer_default() {
                Some(width) => TyTerm::Int(width),
                None if bound.is_text() => TyTerm::String,
                None => continue,
            };
            self.terms.ty_bounds[index] = TypeBound::Resolved {
                ty,
                bound: bound.clone(),
                growth: Growth::Fixed,
            };
        }
        for index in 0..self.terms.repr_vars.len() {
            if matches!(self.terms.repr_vars[index], ReprBound::Unbound(_)) {
                self.terms.repr_vars[index] = ReprBound::Uniform;
            }
        }
        self.close_lends_by_least_element(&mut failures);
        failures.extend(self.settle());
        // After the lends: what a tag `&x` names is `x`'s type once the
        // lend is closed, and a width `x` took above is not a text.
        self.close_tags_as_text(tags);
        failures.extend(self.settle());
        self.close_captures_by_least_element(&mut failures);
        failures.extend(self.settle());
        self.close_instances_by_task();
        failures.extend(self.settle());
        self.close_components_by_least_element();
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
                Decision::Capture { .. } => {
                    unreachable!("close_captures_by_least_element leaves no capture open")
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
                required,
                tie,
                structural,
            } => self.step_instance(id, &call, candidates, generic, required, tie, structural),
            Decision::Conversion { from, to } => self.step_conversion(id, &from, &to),
            Decision::Signature {
                name,
                call,
                options,
                awaiting_head,
                body_effect,
            } => self.step_signature(id, name, &call, options, &awaiting_head, &body_effect),
            Decision::Lend {
                of,
                referent,
                mutability,
            } => self.step_lend(id, &of, &referent, mutability),
            Decision::Capture { of, seen } => self.step_capture(id, &of, &seen),
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
                names: inner.into_ty(),
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
    pub fn lend(&self, of: &InferTy, mutability: Mutability) -> LendOutcome {
        match self.terms.shallow_resolve_ty(of) {
            TyTerm::Var(_) => LendOutcome::HeadOpen,
            TyTerm::Ref(Mutability::Shared, inner) if mutability == Mutability::Mut => {
                LendOutcome::MutableBorrowOfShared {
                    referent: inner.into_ty(),
                }
            }
            TyTerm::Ref(_, inner) => LendOutcome::Names {
                referent: inner.into_ty(),
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
    ) -> Progress {
        match self.lend(of, mutability) {
            LendOutcome::HeadOpen => Progress::Unchanged,
            LendOutcome::MutableBorrowOfShared { .. } => {
                Progress::Failed(Unsettled::MutableBorrowOfShared { decision: id })
            }
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

    /// What a lambda's body reads a captured name of type `of` at, once
    /// that type says whether it is a word. `lookup_var` reads this
    /// directly for a type that already says, so the decision only carries
    /// the case where it does not.
    pub fn capture_read(&self, of: &InferTy) -> CaptureOutcome {
        match self.terms.shallow_resolve_ty(of) {
            TyTerm::Var(_) => CaptureOutcome::HeadOpen,
            TyTerm::Ref(..) => CaptureOutcome::Reads {
                read: CaptureRead::Word,
                seen: of.clone(),
            },
            _ => match self.resolve_ty(of).is_word() {
                Some(true) => CaptureOutcome::Reads {
                    read: CaptureRead::Word,
                    seen: of.clone(),
                },
                Some(false) => CaptureOutcome::Reads {
                    read: CaptureRead::Lent,
                    seen: TyTerm::Ref(Mutability::Shared, Box::new(TypeArg::uniform(of.clone()))),
                },
                None => CaptureOutcome::HeadOpen,
            },
        }
    }

    fn step_capture(&mut self, id: DecisionId, of: &InferTy, seen: &InferTy) -> Progress {
        let (read, reads) = match self.capture_read(of) {
            CaptureOutcome::HeadOpen => return Progress::Unchanged,
            CaptureOutcome::Reads { read, seen } => (read, seen),
        };
        match self.capture_flows(&reads, seen) {
            Ok(()) => Progress::Settled(Answer::Capture(read)),
            Err(Mismatch { expected, got, .. }) => Progress::Failed(Unsettled::LendMismatch {
                decision: id,
                expected,
                got,
            }),
        }
    }

    /// A capture whose type the program states nowhere else takes the
    /// least element of RFC-0018's two readings — the word — so the type
    /// the body read the name at is the type the name has.
    fn close_captures_by_least_element(&mut self, failures: &mut Vec<Unsettled>) {
        for index in 0..self.decisions.len() {
            let Decision::Capture { of, seen } = &self.decisions[index].decision else {
                continue;
            };
            if !matches!(self.decisions[index].state, DecisionState::Open) {
                continue;
            }
            let (of, seen) = (of.clone(), seen.clone());
            self.decisions[index].state = match self.capture_flows(&of, &seen) {
                Ok(()) => DecisionState::Settled(Answer::Capture(CaptureRead::Word)),
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
        awaiting_head: &[UnjoinedArgument],
        body_effect: &EffectTerm<Infer>,
    ) -> Progress {
        if self.takes_poison(call, &options, awaiting_head) {
            self.terms.poison(&call.ret, ErrorToken::new());
            return Progress::Poisoned;
        }
        let mut remaining: Vec<SignatureOption> = options
            .iter()
            .filter(|option| self.takes_signature(call, awaiting_head, option))
            .cloned()
            .collect();
        for argument in awaiting_head {
            remaining = self.admitted_again(remaining, argument);
        }
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
                call: self.resolve_shape(call, awaiting_head),
            }),
            [only] => {
                let (ty, settled) = match &only.candidate {
                    SignatureCandidate::Named { qref, scheme } => {
                        let Instantiated {
                            ty,
                            bounded,
                            bounded_effects,
                            instance,
                            requirements,
                        } = self.instantiate_scheme(scheme);
                        (
                            ty,
                            SettledSignature::Named {
                                qref: *qref,
                                instance,
                                bounded,
                                bounded_effects,
                                requirements,
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
                let flows = self.flows_of_instance(&ty);
                let candidate = only.candidate.clone();
                match self.settle_join(&call.at_effect(effect, flows), &ty) {
                    Ok(()) => {
                        if let Err(Mismatch { expected, got, .. }) =
                            self.join_unjoined(call, &candidate, awaiting_head)
                        {
                            return Progress::Failed(Unsettled::InstanceMismatch {
                                decision: id,
                                expected,
                                got,
                            });
                        }
                        Progress::Settled(Answer::Signature {
                            candidate,
                            settled,
                            callee_ty: ty,
                        })
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
            _ if narrowed => Progress::Narrowed,
            _ => Progress::Unchanged,
        }
    }

    /// Whether an argument of the call is poison (RFC-0078 rule 5): one the
    /// call's parameter joined, one the decision holds unjoined, or one a
    /// candidate takes by conversion or view. Such an argument is admitted
    /// everywhere, and the call is poison (RFC-0043).
    fn takes_poison(
        &self,
        call: &CallShape,
        options: &[SignatureOption],
        awaiting_head: &[UnjoinedArgument],
    ) -> bool {
        let poison = |ty: &InferTy| self.terms.resolve_ty(ty).mentions_error();
        call.params.iter().any(|param| poison(&param.ty))
            || awaiting_head.iter().any(|argument| poison(&argument.ty))
            || options.iter().any(|option| {
                option
                    .converted
                    .iter()
                    .chain(&option.viewed)
                    .any(|argument| poison(&argument.ty))
            })
    }

    /// Admission at an argument the decision held unjoined, asked again with
    /// whatever its head has resolved to, then rule 1 of RFC-0043: a
    /// candidate that takes the argument only by conversion or by view leaves
    /// the set where another takes it directly.
    fn admitted_again(
        &self,
        options: Vec<SignatureOption>,
        argument: &UnjoinedArgument,
    ) -> Vec<SignatureOption> {
        let admissions: Vec<Admission> = options
            .iter()
            .map(|option| {
                let seen = self.seen_by(&option.candidate, argument);
                self.admits(&option.candidate, argument.index, &seen)
            })
            .collect();
        let kept = Kept::among(&admissions);
        options
            .into_iter()
            .zip(admissions)
            .filter(|(_, admission)| kept.keeps(admission))
            .map(|(option, _)| option)
            .collect()
    }

    /// The settled candidate takes each unjoined argument as `admits` says,
    /// with the call's parameter `settle_join` has just bound to the
    /// candidate's own. A direct one joins that parameter here, and a
    /// reborrowed one joins it as the shared reborrow it reaches it as
    /// (RFC-0029 rule 3), the argument keeping its `&mut`; a viewed one is
    /// the checker's coercion at the argument and its type stays what the
    /// caller wrote (RFC-0043 rule 5). A converted one is the conversion
    /// decision the checker opens at the argument once this solve has
    /// settled (`settle_held_arguments`). An argument still waiting for its
    /// head is joined by none: rule 2 gives the head to the solve, not to
    /// the candidate the other arguments settled on.
    fn join_unjoined(
        &mut self,
        call: &CallShape,
        candidate: &SignatureCandidate,
        awaiting_head: &[UnjoinedArgument],
    ) -> Result<(), Mismatch> {
        for argument in awaiting_head {
            let seen = self.seen_by(candidate, argument);
            if self.admission_waits(candidate, argument.index, &seen) {
                continue;
            }
            let param = call.params[argument.index].ty.clone();
            match self.admits(candidate, argument.index, &seen) {
                Admission::Direct => self.settle_join(&seen, &param)?,
                Admission::Reborrowed { shared } => self.settle_join(&shared, &param)?,
                Admission::Converted | Admission::Viewed(_) | Admission::Refused => {}
            }
        }
        Ok(())
    }

    /// The call as its arguments asked it: an argument the decision held
    /// unjoined never became the call's parameter, so its own type stands
    /// there.
    fn resolve_shape(&self, call: &CallShape, awaiting_head: &[UnjoinedArgument]) -> CallShape {
        CallShape {
            params: call
                .params
                .iter()
                .enumerate()
                .map(|(index, param)| {
                    let asked = awaiting_head
                        .iter()
                        .find(|argument| argument.index == index)
                        .map_or(&param.ty, |argument| &argument.ty);
                    ParamTerm::new(param.name, self.terms.resolve_ty(asked))
                })
                .collect(),
            ret: self.terms.resolve_ty(&call.ret),
        }
    }

    /// A local binding used as a signature (RFC-0043) can still be an open
    /// variable with no function head, so there is no declared effect to
    /// read. The variable made here is that binding's own: `settle_join`
    /// below gives an unbound instance this very call type, which is how
    /// the binding comes to carry this term.
    fn flows_of_instance(&mut self, instance: &InferTy) -> FlowTerm<Infer> {
        match self.terms.shallow_resolve_ty(instance) {
            TyTerm::Fn { flows, .. } => flows,
            _ => FlowTerm::Var(self.terms.alloc_flow_var(Flows::none())),
        }
    }

    fn effect_of_instance(&mut self, instance: &InferTy) -> EffectTerm<Infer> {
        match self.terms.shallow_resolve_ty(instance) {
            TyTerm::Fn { effect, .. } => effect,
            _ => self.fresh_effect_var(),
        }
    }

    /// Whether the candidate would still take the call. An unjoined
    /// argument is tried with `JoinKind::Decision`, the kind `settle_join`
    /// runs if this candidate wins: a signature's open representation is
    /// undecided, not disagreed, and only a decision's join may name it.
    fn takes_signature(
        &self,
        call: &CallShape,
        awaiting_head: &[UnjoinedArgument],
        option: &SignatureOption,
    ) -> bool {
        let converted = option.converted.as_slice();
        let mut trial = self.terms.clone();
        let call_ty = call.at_effect(
            EffectTerm::Var(trial.alloc_effect_var()),
            FlowTerm::Var(trial.alloc_flow_var(Flows::none())),
        );
        let scheme = match &option.candidate {
            SignatureCandidate::Named { scheme, .. } => scheme,
            SignatureCandidate::Local { ty } => {
                if trial
                    .join(&call_ty, ty, Position::Value, JoinKind::Flow, self.registry)
                    .is_err()
                {
                    return false;
                }
                let local = trial.resolve_ty(ty);
                let Some(params) = SignatureCandidate::local_params(&local) else {
                    return true;
                };
                return awaiting_head.iter().all(|argument| {
                    let Some(param) = params.get(argument.index) else {
                        return true;
                    };
                    trial
                        .join(
                            &self.seen_by(&option.candidate, argument),
                            &param.ty,
                            Position::Value,
                            JoinKind::Flow,
                            self.registry,
                        )
                        .is_ok()
                });
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
                        _ => match scheme.param_bound(&declared.ty) {
                            TyVarBound::OneOf { shapes, .. } => {
                                self.term_within_shapes(&param.ty, &shapes)
                            }
                            TyVarBound::Any | TyVarBound::Integer { .. } => true,
                        },
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
        let takes_converted = converted.iter().all(|argument| {
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
        });
        let takes_viewed = option.viewed.iter().all(|argument| {
            self.views_as(
                &argument.ty,
                &trial.resolve_ty(&instance_params[argument.index].ty),
            )
        });
        let takes_unjoined = awaiting_head.iter().all(|argument| {
            let seen = self.seen_by(&option.candidate, argument);
            if self.admission_waits(&option.candidate, argument.index, &seen) {
                return true;
            }
            let param = &instance_params[argument.index].ty;
            match self.admits(&option.candidate, argument.index, &seen) {
                Admission::Refused => false,
                Admission::Viewed(_) => self.views_as(&seen, &trial.resolve_ty(param)),
                Admission::Reborrowed { .. } | Admission::Converted => {
                    converts(&trial, self.registry, &seen, param)
                }
                Admission::Direct => trial
                    .join(
                        &seen,
                        param,
                        Position::Value,
                        JoinKind::Decision,
                        self.registry,
                    )
                    .is_ok(),
            }
        });
        takes_converted && takes_viewed && takes_unjoined
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

    /// The mode a candidate takes a receiver place of type `owned` in, and
    /// the type that mode sees it as: the place itself by value, or a
    /// reference to `referent`, what a lend of the place names (RFC-0043
    /// rule 6). Every receiver the candidate is admitted at is seen here.
    pub fn receiver_seen_by(
        &self,
        candidate: &SignatureCandidate,
        owned: &InferTy,
        referent: &InferTy,
    ) -> (ReceiverMode, InferTy) {
        let mode = self.receiver_mode(candidate);
        let seen = match mode {
            ReceiverMode::Value => owned.clone(),
            ReceiverMode::Lent(mutability) => {
                TyTerm::Ref(mutability, Box::new(TypeArg::uniform(referent.clone())))
            }
        };
        (mode, seen)
    }

    pub fn seen_by(&self, candidate: &SignatureCandidate, argument: &UnjoinedArgument) -> InferTy {
        match &argument.handed {
            Handed::AsIs => argument.ty.clone(),
            Handed::Receiver { referent } => {
                self.receiver_seen_by(candidate, &argument.ty, referent).1
            }
        }
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
            TyVarBound::OneOf { shapes, .. } => shapes,
            TyVarBound::Integer { among, .. } => among.into_iter().map(TyTerm::Int).collect(),
        };
        if self.term_within_shapes(arg, &shapes) {
            return Admission::Direct;
        }
        if let Some(shared) = self.shared_reborrow(arg)
            && self.term_within_shapes(&shared, &shapes)
        {
            return Admission::Reborrowed { shared };
        }
        if let Some(viewed) = self.view_among(arg, &shapes) {
            return Admission::Viewed(viewed);
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

    /// Whether admission at this argument waits for a head (RFC-0043 rule
    /// 2): the head this candidate's admission reads is one the solve has
    /// not named, so the argument's own type is evidence for no admission
    /// but `Direct` yet. An argument that is such a head itself may still
    /// reach the parameter by a reborrow, a view or a conversion; one that
    /// lends such a storage is a reference already, and waits where this
    /// candidate takes a run of that storage rather than the storage itself.
    /// An integer literal's head is named, an integer; its width is a
    /// decision of its own (RFC-0037).
    pub fn admission_waits(
        &self,
        candidate: &SignatureCandidate,
        index: usize,
        arg: &InferTy,
    ) -> bool {
        let TyVarBound::OneOf { shapes, .. } = candidate.param_bound(index) else {
            return false;
        };
        match self.terms.shallow_resolve_ty(arg) {
            TyTerm::Var(var) => !matches!(self.bound_of_var(var), TyVarBound::Integer { .. }),
            _ => self.lends_an_unnamed_head(arg) && shapes.iter().any(borrows_a_view),
        }
    }

    /// RFC-0071 rule 3: a tag whose type no use settled, or what a tag's
    /// reference names where that is open, is a `String`.
    fn close_tags_as_text(&mut self, tags: &[InferTy]) {
        for tag in tags {
            let shown = match self.terms.shallow_resolve_ty(tag) {
                TyTerm::Ref(_, lent) => lent.ty().into_owned(),
                held => held,
            };
            let TyTerm::Var(var) = self.terms.shallow_resolve_ty(&shown) else {
                continue;
            };
            let root = self.terms.find_ty_root(var).0 as usize;
            let TypeBound::Unresolved { bound } = &self.terms.ty_bounds[root] else {
                continue;
            };
            if !bound.admits(&Ty::String) {
                continue;
            }
            self.terms.ty_bounds[root] = TypeBound::Resolved {
                ty: TyTerm::String,
                bound: bound.clone(),
                growth: Growth::Fixed,
            };
        }
    }

    /// The instances `scheme` lacked at a call of `args` (RFC-0043,
    /// RFC-0070): where the call's arguments join its parameters with every
    /// variable left unbounded, each requirement no instance of its
    /// signature takes at the types that join gives it, with the
    /// requirement's call type there, as written. Empty where an argument
    /// does not join even so, since the candidate then left by more than a
    /// requirement. The terms are put back as they were: the trial keeps
    /// nothing.
    pub fn lacked_requirements(
        &mut self,
        scheme: &Scheme,
        args: &[InferTy],
    ) -> Vec<(QualifiedRef, Ty)> {
        if scheme.requires.is_empty() {
            return Vec::new();
        }
        let kept = self.terms.clone();
        let lacked = self.lacked_on_the_terms(scheme, args);
        self.terms = kept;
        lacked
    }

    fn lacked_on_the_terms(
        &mut self,
        scheme: &Scheme,
        args: &[InferTy],
    ) -> Vec<(QualifiedRef, Ty)> {
        let patterns: Vec<&PolyTy> = scheme.requires.iter().map(|req| &req.pattern).collect();
        let OpenInstance { ty, beside, .. } = self.terms.instantiate_open_beside(
            &scheme.ty,
            &patterns,
            &scheme.effect_bounds,
            self.registry,
        );
        let TyTerm::Fn { params, .. } = &ty else {
            return Vec::new();
        };
        if params.len() != args.len() {
            return Vec::new();
        }
        for (arg, param) in args.iter().zip(params) {
            if self
                .terms
                .join(
                    arg,
                    &param.ty,
                    Position::Value,
                    JoinKind::Flow,
                    self.registry,
                )
                .is_err()
            {
                return Vec::new();
            }
        }
        scheme
            .requires
            .iter()
            .zip(beside)
            .filter_map(|(requirement, pattern)| {
                let call = called_at(pattern, requirement.calls);
                let bound = EffectBoundedBy::of(Some(requirement.signature));
                let taken = requirement.instances.generic.is_some()
                    || requirement
                        .instances
                        .concrete
                        .iter()
                        .filter(|sig| sig.task <= requirement.calls)
                        .any(|sig| self.terms.would_take(&call, &sig.ty, bound, self.registry));
                if taken {
                    return None;
                }
                Some((requirement.signature, self.written_ty(&call).ok()?))
            })
            .collect()
    }

    /// The candidate a signature decision settled on, which takes each
    /// argument the decision held unjoined as it admits that argument once
    /// the solve has named its head (RFC-0043 rule 2). `None` where the
    /// decision did not settle, which is reported at the call.
    pub fn settled_candidate(&self, id: DecisionId) -> Option<&SignatureCandidate> {
        match &self.decisions[id.0 as usize].state {
            DecisionState::Settled(Answer::Signature { candidate, .. }) => Some(candidate),
            _ => None,
        }
    }

    /// Whether the storage this argument lends is one the solve has not
    /// named (RFC-0043 rule 2). The head is the solve's to name, so no
    /// site reads the argument's own type as evidence about it, and every
    /// site that must not asks here rather than deriving it again.
    pub fn lends_an_unnamed_head(&self, arg: &InferTy) -> bool {
        let TyTerm::Ref(_, lent) = self.terms.shallow_resolve_ty(arg) else {
            return false;
        };
        matches!(self.terms.resolve_ty(&lent.ty()), TyTerm::Var(_))
    }

    /// RFC-0047 rule 6, RFC-0062 rule 3.
    pub fn lent_view(&self, arg: &InferTy) -> Option<Viewed> {
        let TyTerm::Ref(mutability, storage) = self.terms.resolve_ty(arg) else {
            return None;
        };
        let storage = self.terms.resolve_ty(&storage.ty());
        if matches!(storage, TyTerm::String) {
            return Some(Viewed {
                view: View::Str,
                mutability,
            });
        }
        let head = crate::ty::SliceableHead::of(&storage)?;
        [mutability, Mutability::Shared]
            .into_iter()
            .filter(|lent| mutability.reaches(*lent))
            .find(|lent| self.registry.lends_a_slice(head, *lent))
            .map(|mutability| Viewed {
                view: View::Slice,
                mutability,
            })
    }

    /// The `&` a `&mut` argument reaches a shared position as.
    fn shared_reborrow(&self, arg: &InferTy) -> Option<InferTy> {
        let TyTerm::Ref(Mutability::Mut, named) = self.terms.shallow_resolve_ty(arg) else {
            return None;
        };
        Some(TyTerm::Ref(Mutability::Shared, named))
    }

    fn views_as<V>(&self, arg: &InferTy, shape: &TyTerm<V>) -> bool
    where
        V: Phase,
    {
        let TyTerm::Ref(mutability, pointee) = shape else {
            return false;
        };
        self.lent_view(arg).is_some_and(|lent| {
            lent.mutability.reaches(*mutability) && View::of(&pointee.ty()) == Some(lent.view)
        })
    }

    /// The view a parameter of these shapes takes the argument as, where it
    /// takes it as one.
    fn view_among(&self, arg: &InferTy, shapes: &[PolyTy]) -> Option<Viewed> {
        shapes.iter().find_map(|shape| {
            let TyTerm::Ref(mutability, pointee) = shape else {
                return None;
            };
            let view = View::of(&pointee.ty())?;
            self.views_as(arg, shape).then_some(Viewed {
                view,
                mutability: *mutability,
            })
        })
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
            .signatures()
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
                let TyVarBound::OneOf { shapes, .. } = self.bound_of_var(var) else {
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
        required: Option<QualifiedRef>,
        tie: InstanceTie,
        structural: Option<Rc<ComponentOffer>>,
    ) -> Progress {
        let ty = self.terms.resolve_ty(call);
        let bound = EffectBoundedBy::of(required);
        let remaining: Vec<Candidate> = candidates
            .iter()
            .filter(|c| self.terms.would_take(call, &c.ty, bound, self.registry))
            .cloned()
            .collect();
        let generic = generic.filter(|g| self.terms.would_take(call, &g.ty, bound, self.registry));
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
        let head = match &structural {
            Some(_) => self.structural_head(call),
            None => StructuralHead::Refuses,
        };
        match (head, structural) {
            (StructuralHead::Takes, Some(offer)) if remaining.is_empty() && generic.is_none() => {
                return self.settle_structural(id, call, bound, &offer);
            }
            (StructuralHead::Takes | StructuralHead::Open, _) => return progress_without_answer,
            (StructuralHead::Refuses, _) => {}
        }
        match (remaining.as_slice(), generic) {
            ([], None) => Progress::Failed(Unsettled::NoInstance {
                decision: id,
                call: ty,
                instances: candidates.iter().map(|c| c.ty.clone()).collect(),
                required,
            }),
            ([], Some(generic)) => {
                let instance = self.instantiate_open(&generic.ty);
                match self.settle_instance(
                    id,
                    CallOfInstance {
                        call,
                        instance: &instance,
                    },
                    bound,
                ) {
                    Ok(()) => {
                        self.decisions[id.0 as usize].instance_flows = declared_flows(&generic.ty);
                        Progress::Settled(Answer::Instance(InstanceKind::Extern(generic.instance)))
                    }
                    Err(refused) => Progress::Failed(refused),
                }
            }
            ([only], generic) if generic.is_none() || matches_pattern(&ty, &only.ty) => {
                let CandidateAt {
                    ty: instance,
                    required,
                    bounded_effects,
                } = self.instantiate_candidate(only);
                match self.settle_instance(
                    id,
                    CallOfInstance {
                        call,
                        instance: &instance,
                    },
                    bound,
                ) {
                    Ok(()) => {
                        self.instance_bounds.extend(
                            bounded_effects
                                .into_iter()
                                .map(|var| InstanceBound { decision: id, var }),
                        );
                        self.begin_unbound_sources(id, &instance);
                        self.require_instances(id, required);
                        self.decisions[id.0 as usize].instance_flows = declared_flows(&only.ty);
                        Progress::Settled(Answer::Instance(only.instance))
                    }
                    Err(refused) => Progress::Failed(refused),
                }
            }
            ([first, rest @ ..], None)
                if tie == InstanceTie::Types && rest.iter().all(|c| c.ty == first.ty) =>
            {
                let instance = self.instantiate_open(&first.ty);
                match self.settle_instance(
                    id,
                    CallOfInstance {
                        call,
                        instance: &instance,
                    },
                    bound,
                ) {
                    Ok(()) => {
                        if let Decision::Instance { tie, .. } =
                            &mut self.decisions[id.0 as usize].decision
                        {
                            *tie = InstanceTie::TaskOnly;
                        }
                        Progress::Narrowed
                    }
                    Err(refused) => Progress::Failed(refused),
                }
            }
            _ => progress_without_answer,
        }
    }

    fn lent_referent(&self, call: &InferTy) -> Option<InferTy> {
        let TyTerm::Fn { params, .. } = self.terms.shallow_resolve_ty(call) else {
            return None;
        };
        let TyTerm::Ref(_, taken) = self.terms.shallow_resolve_ty(&params.first()?.ty) else {
            return None;
        };
        Some(self.terms.shallow_resolve_ty(&taken.ty()))
    }

    fn structural_head(&self, call: &InferTy) -> StructuralHead {
        match self.lent_referent(call) {
            Some(TyTerm::Var(_)) => StructuralHead::Open,
            Some(head) if components(&head).is_some() => StructuralHead::Takes,
            Some(_) | None => StructuralHead::Refuses,
        }
    }

    fn settle_structural(
        &mut self,
        id: DecisionId,
        call: &InferTy,
        bound: EffectBoundedBy,
        offer: &Rc<ComponentOffer>,
    ) -> Progress {
        let instance = offer.call_at(&self.fresh_ty_var());
        if let Err(refused) = self.settle_instance(
            id,
            CallOfInstance {
                call,
                instance: &instance,
            },
            bound,
        ) {
            return Progress::Failed(refused);
        }
        self.structural.insert(
            id,
            SettledStructural {
                call: call.clone(),
                offer: Rc::clone(offer),
                children: Vec::new(),
            },
        );
        self.open_undecided_components(id);
        Progress::Settled(Answer::Instance(InstanceKind::Structural))
    }

    fn open_undecided_components(&mut self, id: DecisionId) -> bool {
        let SettledStructural {
            call,
            offer,
            children,
        } = &self.structural[&id];
        let offer = Rc::clone(offer);
        let Some(referent) = self.lent_referent(call) else {
            return false;
        };
        let Some(parts) = components(&referent) else {
            return false;
        };
        let decided: Vec<Component> = children.iter().map(|(component, _)| *component).collect();
        let fresh: Vec<(Component, InferTy)> = parts
            .into_iter()
            .filter(|(component, _)| !decided.contains(component))
            .map(|(component, part)| (component, part.clone()))
            .collect();
        let opened = !fresh.is_empty();
        for (component, part) in fresh {
            let child = self.decide(Decision::Instance {
                call: offer.call_at(&part),
                candidates: offer.candidates.clone(),
                generic: None,
                required: Some(offer.signature),
                tie: InstanceTie::Types,
                structural: Some(Rc::clone(&offer)),
            });
            self.opened_children.push(OpenedChild {
                parent: id,
                child,
                component: Some(component),
            });
            if let Some(settled) = self.structural.get_mut(&id) {
                settled.children.push((component, child));
            }
        }
        opened
    }

    fn open_grown_components(&mut self) -> bool {
        let settled: Vec<DecisionId> = self.structural.keys().copied().collect();
        let mut opened = false;
        for id in settled {
            opened |= self.open_undecided_components(id);
        }
        opened
    }

    pub fn settled_structural(&self, decision: DecisionId) -> Option<&SettledStructural> {
        self.structural.get(&decision)
    }

    /// A component whose type nothing constrained holds no value, so its
    /// least element is `!` (RFC-0042 rule 3): `None == None` compares no
    /// payload.
    fn close_components_by_least_element(&mut self) {
        let open: Vec<TypeBoundId> = self
            .decisions
            .iter()
            .filter(|slot| matches!(slot.state, DecisionState::Open))
            .filter_map(|slot| match &slot.decision {
                Decision::Instance {
                    call,
                    required: Some(_),
                    structural: Some(_),
                    ..
                } => self.lent_referent(call),
                _ => None,
            })
            .filter_map(|referent| match referent {
                TyTerm::Var(var) => Some(self.terms.find_ty_root(var)),
                _ => None,
            })
            .collect();
        for root in open {
            if let TypeBound::Unresolved {
                bound: TyVarBound::Any,
            } = &self.terms.ty_bounds[root.0 as usize]
            {
                self.terms.ty_bounds[root.0 as usize] = TypeBound::Resolved {
                    ty: TyTerm::Never,
                    bound: TyVarBound::Any,
                    growth: Growth::Fixed,
                };
            }
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
        if let Some(Reborrowed {
            from: named_from,
            to: named_to,
        }) = reborrowed(&from_r, &to_r)
        {
            return match self.settle_join(&named_from.ty(), &named_to.ty()) {
                Ok(()) => Progress::Settled(Answer::Conversion(Conversion::Reborrow)),
                Err(_) => Progress::Failed(Unsettled::NoConversion {
                    decision: id,
                    from: self.terms.resolve_ty(from),
                    to: self.terms.resolve_ty(to),
                }),
            };
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
        let back = self.conversion_rules(&references.to.ty(), &references.from.ty());
        let [back] = back.as_slice() else {
            if back.is_empty() {
                return Progress::Failed(Unsettled::NoConversion {
                    decision: id,
                    from: self.terms.resolve_ty(&references.to.ty()),
                    to: self.terms.resolve_ty(&references.from.ty()),
                });
            }
            return Progress::Unchanged;
        };
        let answer = Conversion::ThroughRef {
            mutability: references.mutability,
            cast: rule.fn_ref,
            back: back.fn_ref,
        };
        let named_from = references.from.ty().into_owned();
        let named_to = references.to.ty().into_owned();
        self.settle_cast(id, &named_from, &named_to, &inst_from, &inst_to, answer)
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

    /// The type a report shows: as written, every type variable nothing
    /// resolved closed to `!`, whatever bound it carries (RFC-0043), and
    /// an open representation closed to its least element. A variable the
    /// solve bound outside its declaration's bound is shown as bound, not
    /// refused. An open identity or length still refuses: neither has a
    /// concrete form standing for "open".
    pub fn written_ty(&self, ty: &InferTy) -> Result<Ty, FreezeError> {
        self.freeze_ty_with(ty, Open::AsWritten)
    }

    fn freeze_ty_with(&self, ty: &InferTy, open: Open) -> Result<Ty, FreezeError> {
        self.terms.unstale(ty).try_map(
            &mut |id: TypeBoundId| self.freeze_ty_var(id, open),
            &mut |id: IdentityVarId| {
                self.freeze_identity(&IdentityTerm::Var(id))
                    .map(IdentityTerm::Known)
            },
            &mut |id: EffectVarId| Ok(EffectTerm::Known(self.freeze_effect(&EffectTerm::Var(id)))),
            &mut |id: LenVarId| self.freeze_len(id),
            &mut |id: ReprVarId| self.freeze_repr(id, open),
            &mut |id: FlowVarId| Ok(FlowTerm::Known(self.terms.freeze_flows(&FlowTerm::Var(id)))),
        )
    }

    fn freeze_ty_var(&self, id: TypeBoundId, open: Open) -> Result<Ty, FreezeError> {
        let root = self.terms.find_ty_root(id);
        match &self.terms.ty_bounds[root.0 as usize] {
            TypeBound::Resolved {
                ty: inner, bound, ..
            } => {
                let frozen = self.freeze_ty_with(inner, open)?;
                // A report shows the type the solve bound the
                // variable to even where the declaration's bound
                // refuses it: that type is what the position
                // carries, and the bound itself is refused on its
                // own by `MirErrorKind::TypeOutOfBound`.
                if bound.admits(&frozen) || matches!(open, Open::AsWritten) {
                    Ok(frozen)
                } else {
                    Err(FreezeError::OutOfBound {
                        var: root,
                        ty: frozen,
                        bound: bound.clone(),
                    })
                }
            }
            TypeBound::Unresolved { bound } => match (bound.integer_default(), bound, open) {
                (Some(k), _, _) => Ok(Ty::Int(k)),
                (None, TyVarBound::Any, Open::Never) | (None, _, Open::AsWritten) => Ok(Ty::Never),
                (None, _, _) => Err(FreezeError::UnresolvedType(root)),
            },
            TypeBound::Forward(_) => unreachable!("find_ty_root resolves forwards"),
        }
    }

    fn freeze_len(&self, id: LenVarId) -> Result<LenTerm<Concrete>, FreezeError> {
        match self.terms.resolve_len(&LenTerm::Var(id)) {
            LenTerm::Known(n) => Ok(LenTerm::Known(n)),
            LenTerm::Var(root) => Err(FreezeError::UnresolvedLen(root)),
        }
    }

    fn freeze_repr(&self, id: ReprVarId, open: Open) -> Result<Repr<Concrete>, FreezeError> {
        match self.terms.resolve_repr(id) {
            Repr::Uniform => Ok(Repr::Uniform),
            Repr::Specialized(held) => held
                .try_map(
                    &mut |id: TypeBoundId| self.freeze_ty_var(id, open),
                    &mut |id: IdentityVarId| {
                        self.freeze_identity(&IdentityTerm::Var(id))
                            .map(IdentityTerm::Known)
                    },
                    &mut |id: EffectVarId| {
                        Ok(EffectTerm::Known(self.freeze_effect(&EffectTerm::Var(id))))
                    },
                    &mut |id: LenVarId| self.freeze_len(id),
                    &mut |id: ReprVarId| self.freeze_repr(id, open),
                    &mut |id: FlowVarId| {
                        Ok(FlowTerm::Known(self.terms.freeze_flows(&FlowTerm::Var(id))))
                    },
                )
                .map(Repr::Specialized),
            // A report runs before `solve` has taken the least
            // elements, and `Uniform` is the one an open
            // representation takes there (RFC-0042 rule 3), so a report
            // shows the rest of the type rather than nothing.
            Repr::Var(_) if matches!(open, Open::AsWritten) => Ok(Repr::Uniform),
            Repr::Var(root) => Err(FreezeError::UnresolvedRepr(root)),
        }
    }

    pub fn freeze_effect(&self, term: &EffectTerm<Infer>) -> Effect {
        self.terms.freeze_effect(term)
    }

    /// An effect variable after `solve`, verified against the bound its
    /// root carries (RFC-0011 rule 5).
    pub fn close_effect_var(&self, var: EffectVarId) -> Result<Effect, EffectOutOfBound> {
        let root = self.terms.find_effect_root(var);
        let effect = self.terms.freeze_effect(&EffectTerm::Var(root));
        let bound = self.terms.effect_var_bound(root);
        match bound.admits(effect.task) {
            true => Ok(effect),
            false => Err(EffectOutOfBound {
                var: root,
                effect,
                bound,
            }),
        }
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
    pub(crate) fn instantiate_open(&mut self, ty: &PolyTy) -> InferTy {
        self.terms.instantiate_open(ty, self.registry)
    }

    /// An identity the settled instance returns and nothing bound, neither
    /// its parameters nor the call, is a source that begins at this call
    /// (RFC-0012), as it is for a declaration called by name.
    fn begin_unbound_sources(&mut self, decision: DecisionId, instance: &InferTy) {
        let TyTerm::Fn { params, ret, .. } = instance else {
            return;
        };
        let mut of_params: Vec<IdentityVarId> = Vec::new();
        for param in params {
            param.ty.for_each_identity(&mut |identity| {
                if let IdentityTerm::Var(var) = self.terms.resolve_identity(identity) {
                    of_params.push(var);
                }
            });
        }
        let mut open: Vec<IdentityVarId> = Vec::new();
        ret.for_each_identity(&mut |identity| {
            if let IdentityTerm::Var(var) = self.terms.resolve_identity(identity)
                && !of_params.contains(&var)
            {
                open.push(var);
            }
        });
        for var in open {
            let IdentityTerm::Var(root) = self.terms.resolve_identity(&IdentityTerm::Var(var))
            else {
                continue;
            };
            let source = self.sources.next();
            self.terms.identity_vars[root.0 as usize] = IdentityBound::Bound(source);
            self.begun_by_decisions
                .push(BegunSource { decision, source });
        }
    }

    pub fn take_begun_sources(&mut self) -> Vec<BegunSource> {
        std::mem::take(&mut self.begun_by_decisions)
    }

    // -- Requirements of a settled instance --------------------------

    /// The decisions the instance `decision` settled on requires, in the
    /// order its declaration states them (RFC-0070 rule 3).
    pub fn children_of(&self, decision: DecisionId) -> &[RequiredDecision] {
        self.children.get(&decision).map_or(&[], Vec::as_slice)
    }

    pub fn take_opened_children(&mut self) -> Vec<OpenedChild> {
        std::mem::take(&mut self.opened_children)
    }

    pub fn take_instance_bounds(&mut self) -> Vec<InstanceBound> {
        std::mem::take(&mut self.instance_bounds)
    }

    fn instantiate_candidate(&mut self, candidate: &Candidate) -> CandidateAt {
        let patterns: Vec<&PolyTy> = candidate.requires.iter().map(|r| &r.pattern).collect();
        let OpenInstance {
            ty,
            beside,
            bounded_effects,
        } = self.terms.instantiate_open_beside(
            &candidate.ty,
            &patterns,
            &candidate.effect_bounds,
            self.registry,
        );
        let required = candidate
            .requires
            .iter()
            .cloned()
            .zip(beside)
            .map(|(requirement, pattern)| RequiredCall {
                requirement,
                pattern,
            })
            .collect();
        CandidateAt {
            ty,
            required,
            bounded_effects,
        }
    }

    fn require_instances(&mut self, parent: DecisionId, required: Vec<RequiredCall>) {
        let signatures = self.signatures;
        let mut children: Vec<RequiredDecision> = Vec::with_capacity(required.len());
        for RequiredCall {
            requirement,
            pattern,
        } in required
        {
            let signature = requirement.signature;
            let instances = signatures.get(&signature).unwrap_or_else(|| {
                panic!(
                    "an instance requires {signature:?}, which is not a declared signature: \
                     `Externs::combine` refuses this"
                )
            });
            let id = self.decide(Decision::Instance {
                call: called_at(pattern, requirement.calls),
                candidates: instances
                    .concrete
                    .iter()
                    .enumerate()
                    .filter(|(_, sig)| sig.task <= requirement.calls)
                    .map(|(instance, sig)| Candidate {
                        instance: InstanceKind::Extern(instance),
                        ty: sig.ty.clone(),
                        admits: sig.admits,
                        requires: sig.requires.clone(),
                        effect_bounds: sig.effect_bounds.clone(),
                    })
                    .collect(),
                generic: None,
                required: Some(signature),
                tie: InstanceTie::Types,
                structural: None,
            });
            self.opened_children.push(OpenedChild {
                parent,
                child: id,
                component: None,
            });
            children.push(RequiredDecision { signature, id });
        }
        self.children.insert(parent, children);
    }

    pub fn fresh_shape(&mut self, pattern: &PolyTy) -> InferTy {
        self.instantiate_open(pattern)
    }

    pub fn instantiate_scheme(&mut self, scheme: &Scheme) -> Instantiated {
        self.instantiate_scheme_with(scheme, CompilerInstances::none())
    }

    /// A scheme with instances the compiler adds to the declared ones
    /// (RFC-0020): its own instructions for a shared signature.
    pub fn instantiate_scheme_with(
        &mut self,
        scheme: &Scheme,
        compiler: CompilerInstances,
    ) -> Instantiated {
        let CompilerInstances {
            candidates: compiler_instances,
            withholds,
            structural,
        } = compiler;
        let mut bounded: Vec<TypeBoundId> = Vec::new();
        let mut bounded_effects: Vec<EffectVarId> = Vec::new();
        let fixed_generic = scheme.instances.as_ref().is_some_and(|instances| {
            instances.concrete.is_empty()
                && instances.generic.is_none()
                && compiler_instances.is_empty()
                && structural.is_none()
        });
        let reprs = if fixed_generic {
            Reprs::Uniform
        } else {
            Reprs::Open
        };
        let patterns: Vec<&PolyTy> = scheme.requires.iter().map(|req| &req.pattern).collect();
        let (ty, required) = self.instantiate_with(
            &scheme.ty,
            &patterns,
            |var, fresh| {
                let bound = scheme.bound_of(var);
                if bound != TyVarBound::Any {
                    bounded.push(fresh);
                }
                bound
            },
            |var, fresh| {
                let bound = scheme.effect_bound_of(var);
                if bound != EffectVarBound::Any {
                    bounded_effects.push(fresh);
                }
                bound
            },
            reprs,
        );
        let requirements = scheme
            .requires
            .iter()
            .zip(required)
            .map(|(req, call)| {
                let call = called_at(call, req.calls);
                let id = self.decide(Decision::Instance {
                    call,
                    candidates: req
                        .instances
                        .concrete
                        .iter()
                        .enumerate()
                        .filter(|(_, sig)| sig.task <= req.calls)
                        .map(|(instance, sig)| Candidate {
                            instance: InstanceKind::Extern(instance),
                            ty: sig.ty.clone(),
                            admits: sig.admits,
                            requires: sig.requires.clone(),
                            effect_bounds: sig.effect_bounds.clone(),
                        })
                        .collect(),
                    generic: None,
                    required: Some(req.signature),
                    tie: InstanceTie::Types,
                    structural: None,
                });
                RequiredDecision {
                    signature: req.signature,
                    id,
                }
            })
            .collect();
        let instance = scheme.instances.as_ref().map(|instances| {
            if fixed_generic {
                return InstanceChoice::Fixed(instances.generic_index());
            }
            let shadowed_by_the_compiler: Vec<_> =
                compiler_instances.iter().map(|c| c.ty.clone()).collect();
            let withheld = |ty: &PolyTy| {
                shadowed_by_the_compiler.contains(ty)
                    || (withholds == Withholds::LanguageOwned && takes_a_language_owned_type(ty))
            };
            let id = self.decide(Decision::Instance {
                call: ty.clone(),
                candidates: instances
                    .concrete
                    .iter()
                    .cloned()
                    .enumerate()
                    .filter(|(_, sig)| !withheld(&sig.ty))
                    .map(|(instance, sig)| Candidate {
                        instance: InstanceKind::Extern(instance),
                        ty: sig.ty,
                        admits: sig.admits,
                        requires: sig.requires,
                        effect_bounds: sig.effect_bounds,
                    })
                    .chain(compiler_instances)
                    .collect(),
                generic: instances.generic.as_ref().map(|_| GenericInstance {
                    instance: instances.generic_index(),
                    ty: scheme.ty.clone(),
                }),
                required: None,
                tie: InstanceTie::Types,
                structural: structural.map(Rc::new),
            });
            InstanceChoice::Decided(id)
        });
        Instantiated {
            ty,
            bounded,
            bounded_effects,
            instance,
            requirements,
        }
    }

    /// `ty` and `beside` at one set of fresh variables: a variable `ty`
    /// and a pattern in `beside` both name is one solver variable.
    fn instantiate_with(
        &mut self,
        ty: &PolyTy,
        beside: &[&PolyTy],
        mut bound_for: impl FnMut(u32, TypeBoundId) -> TyVarBound,
        mut effect_bound_for: impl FnMut(u32, EffectVarId) -> EffectVarBound,
        reprs: Reprs,
    ) -> (InferTy, Vec<InferTy>) {
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
        let mut instantiate = |poly: &PolyTy| {
            poly.map(
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
                        if from_params.contains(&id) {
                            IdentityTerm::Var(alloc_identity_var(identity_vars))
                        } else {
                            IdentityTerm::Known(sources.next())
                        }
                    })
                },
                &mut |id: u32| {
                    EffectTerm::Var(*maps.effect.entry(id).or_insert_with(|| {
                        let fresh = alloc_effect_var(effect_vars, EffectVarBound::Any);
                        let bound = effect_bound_for(id, fresh);
                        effect_vars[fresh.0 as usize] = EffectBound::Range {
                            lower: Effect::PURE,
                            upper: Effect::TOP,
                            bound,
                        };
                        fresh
                    }))
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
                &mut crate::ty::no_flow_var,
            )
        };
        let mut instance = instantiate(ty);
        let mut beside: Vec<InferTy> = beside.iter().map(|poly| instantiate(poly)).collect();
        self.terms.open_flows(&mut instance);
        for poly in &mut beside {
            self.terms.open_flows(poly);
        }
        let registry = self.registry;
        (
            uniform_slots(instance, registry),
            beside
                .into_iter()
                .map(|poly| uniform_slots(poly, registry))
                .collect(),
        )
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
                    .or_insert_with(|| alloc_effect_var(effect_vars, EffectVarBound::Any)),
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
            &mut crate::ty::no_flow_var,
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
            &mut crate::ty::no_flow_var,
        );
        let (mut ia, mut ib) = (ia, ib);
        self.terms.open_flows(&mut ia);
        self.terms.open_flows(&mut ib);
        (
            uniform_slots(ia, self.registry),
            uniform_slots(ib, self.registry),
        )
    }

}

/// The placeholders one instantiation of a `PolyTy` has renamed so far.
struct OpenInstance {
    ty: InferTy,
    beside: Vec<InferTy>,
    bounded_effects: Vec<EffectVarId>,
}

#[derive(Default)]
struct PolyMaps {
    ty: FxHashMap<u32, TypeBoundId>,
    identity: FxHashMap<u32, IdentityTerm<Infer>>,
    effect: FxHashMap<u32, EffectVarId>,
    len: FxHashMap<u32, LenVarId>,
    repr: FxHashMap<u32, ReprVarId>,
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
                &mut crate::ty::no_flow_var::<Poly>,
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
        return conversion_rules(terms, registry, &references.from.ty(), &references.to.ty());
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

/// A parameter that takes a run of what its argument lends rather than the
/// storage itself: `&[T]` or `&mut [T]` (RFC-0047 rule 6), `&str` (RFC-0062
/// rule 3).
fn borrows_a_view<V>(shape: &TyTerm<V>) -> bool
where
    V: Phase,
{
    matches!(shape, TyTerm::Ref(_, pointee) if matches!(*pointee.ty(), TyTerm::Slice(_) | TyTerm::Str))
}

/// One declared rule (RFC-0023) takes `from` to `to`; through a reference,
/// the value is cast back when the call ends (RFC-0041).
/// A `&mut` reaching a `&` by a reborrow: what each names.
struct Reborrowed<'a> {
    from: &'a TypeArg<Infer>,
    to: &'a TypeArg<Infer>,
}

fn reborrowed<'a>(from: &'a InferTy, to: &'a InferTy) -> Option<Reborrowed<'a>> {
    match (from, to) {
        (TyTerm::Ref(Mutability::Mut, from), TyTerm::Ref(Mutability::Shared, to)) => {
            Some(Reborrowed { from, to })
        }
        _ => None,
    }
}

fn converts(terms: &Terms, registry: &TypeRegistry, from: &InferTy, to: &InferTy) -> bool {
    let from_r = terms.resolve_ty(from);
    let to_r = terms.resolve_ty(to);
    if let Some(Reborrowed {
        from: named_from,
        to: named_to,
    }) = reborrowed(&from_r, &to_r)
    {
        let mut trial = terms.clone();
        return trial
            .join(
                &named_from.ty(),
                &named_to.ty(),
                Position::Value,
                JoinKind::Decision,
                registry,
            )
            .is_ok();
    }
    let Some(references) = ReferencePair::of(&from_r, &to_r) else {
        return !conversion_rules(terms, registry, from, to).is_empty();
    };
    [
        (&references.from.ty(), &references.to.ty()),
        (&references.to.ty(), &references.from.ty()),
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
}

/// A signature enters the solver with the slots that do not specialize
/// (hash-types.md R1) set `Uniform`: their representation is not a fact
/// of the type.
fn uniform_slots(ty: InferTy, registry: &TypeRegistry) -> InferTy {
    fn arg(a: TypeArg<Infer>, specializing: bool, registry: &TypeRegistry) -> TypeArg<Infer> {
        let mut a = match a {
            TypeArg::Open(..) | TypeArg::Specialized(_) if !specializing => {
                TypeArg::Uniform(a.into_ty())
            }
            a => a,
        };
        a.rewrite_types(&mut |ty| {
            *ty = uniform_slots(std::mem::replace(ty, TyTerm::Unit), registry)
        });
        a
    }
    match ty {
        TyTerm::UserDefined {
            id,
            type_args,
            effect_args,
            identity_args,
            region_params,
        } => TyTerm::UserDefined {
            id,
            type_args: type_args
                .into_iter()
                .enumerate()
                .map(|(index, a)| arg(a, registry.specializes(id, index), registry))
                .collect(),
            effect_args,
            identity_args,
            region_params,
        },
        TyTerm::Ref(m, inner) => TyTerm::Ref(m, Box::new(arg(*inner, true, registry))),
        TyTerm::Array(inner, len) => TyTerm::Array(Box::new(uniform_slots(*inner, registry)), len),
        TyTerm::Slice(elem) => TyTerm::Slice(Box::new(uniform_slots(*elem, registry))),
        TyTerm::Str => TyTerm::Str,
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
        TyTerm::Object(object) => TyTerm::Object(
            object.with_fields(
                object
                    .iter()
                    .map(|(k, v)| (*k, uniform_slots(v.clone(), registry)))
                    .collect(),
            ),
        ),
        TyTerm::Fn {
            params,
            ret,
            captures,
            effect,
            flows,
        } => TyTerm::Fn {
            flows,
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
        TyTerm::Enum { name, variants, .. } => TyTerm::Enum {
            name,
            variants: variants
                .into_iter()
                .map(|(k, v)| (k, v.map(|t| Box::new(uniform_slots(*t, registry)))))
                .collect(),
            home: crate::ty::Home::NONE,
        },
        TyTerm::Handle(inner) => TyTerm::Handle(Box::new(uniform_slots(*inner, registry))),
        leaf @ (TyTerm::Int(_)
        | TyTerm::Float
        | TyTerm::Char
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

/// An effect variable froze at an effect whose task its declared bound
/// does not admit (RFC-0011 rule 5).
#[derive(Debug, Clone, PartialEq)]
pub struct EffectOutOfBound {
    pub var: EffectVarId,
    pub effect: Effect,
    pub bound: EffectVarBound,
}

/// A scheme instantiated into the solver; the caller verifies the bounded
/// variables where it chose to.
pub struct Instantiated {
    pub ty: InferTy,
    pub bounded: Vec<TypeBoundId>,
    /// The effect variables a declared bound floors (RFC-0011 rule 5).
    pub bounded_effects: Vec<EffectVarId>,
    /// `Some` for an Extern function.
    pub instance: Option<InstanceChoice>,
    /// One instance decision per requirement the scheme states, in the
    /// scheme's order (RFC-0068 rule 5).
    pub requirements: Vec<RequiredDecision>,
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

#[cfg(test)]
mod requirement_tests {
    use super::*;
    use crate::ty::{InstanceSig, Instances, Sources};
    use acvus_utils::{Astr, Interner};

    fn signature_of<P>(name: Astr, param: TyTerm<P>, ret: TyTerm<P>) -> TyTerm<P>
    where
        P: Phase,
    {
        TyTerm::Fn {
            params: vec![ParamTerm::new(name, param)],
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::PURE.into(),
            flows: crate::ty::Flows::Every.into(),
        }
    }

    fn at(tys: impl IntoIterator<Item = PolyTy>) -> Instances {
        Instances {
            concrete: tys
                .into_iter()
                .map(|ty| InstanceSig {
                    ty,
                    admits: Task::Heavy,
                    task: Task::Sync,
                    requires: Vec::new(),
                    effect_bounds: Vec::new(),
                    laws: crate::laws::Laws::None,
                    ensures: Vec::new(),
                    reaches: crate::laws::Reaches::Lent,
                    returns: crate::laws::Returns::Unstated,
                    copies: None,
                    cost: None,
                })
                .collect(),
            generic: None,
        }
    }

    /// A candidate that requires `signature` at its own variable.
    fn requiring(name: Astr, signature: QualifiedRef) -> Candidate {
        let own: PolyTy = signature_of(name, TyTerm::Var(0), TyTerm::Var(0));
        Candidate {
            instance: InstanceKind::Extern(0),
            ty: own.clone(),
            admits: Task::Heavy,
            requires: vec![RequirementSig {
                signature,
                pattern: own,
                calls: Task::Sync,
            }],
            effect_bounds: Vec::new(),
        }
    }

    fn call_of(name: Astr, candidate: Candidate) -> Decision {
        Decision::Instance {
            call: signature_of(name, TyTerm::I64, TyTerm::I64),
            candidates: vec![candidate],
            generic: None,
            required: None,
            tie: InstanceTie::Types,
            structural: None,
        }
    }

    #[test]
    fn a_settled_candidates_requirement_settles_at_the_type_it_was_called_with() {
        let interner = Interner::new();
        let name = interner.intern("a");
        let inner = QualifiedRef::root(interner.intern("inner"));
        let signatures = FxHashMap::from_iter([(
            inner,
            at([
                signature_of(name, TyTerm::String, TyTerm::String),
                signature_of(name, TyTerm::I64, TyTerm::I64),
            ]),
        )]);
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let mut solver = Solver::new(&mut sources, &registry, &signatures);

        let decision = solver.decide(call_of(name, requiring(name, inner)));
        let unsettled = solver.settle();

        assert!(unsettled.is_empty(), "{unsettled:?}");
        assert_eq!(
            solver.answer(decision),
            Some(Answer::Instance(InstanceKind::Extern(0)))
        );
        let [child] = *solver.children_of(decision) else {
            panic!("one requirement, one decision")
        };
        assert_eq!(child.signature, inner);
        assert_eq!(
            solver.answer(child.id),
            Some(Answer::Instance(InstanceKind::Extern(1)))
        );
        assert_eq!(
            solver
                .take_opened_children()
                .iter()
                .map(|o| (o.parent, o.child))
                .collect::<Vec<_>>(),
            vec![(decision, child.id)]
        );
    }

    #[test]
    fn a_requirement_no_instance_reaches_names_the_signature_it_asked_for() {
        let interner = Interner::new();
        let name = interner.intern("a");
        let inner = QualifiedRef::root(interner.intern("inner"));
        let signatures = FxHashMap::from_iter([(
            inner,
            at([signature_of(name, TyTerm::String, TyTerm::String)]),
        )]);
        let mut sources = Sources::new();
        let registry = TypeRegistry::new();
        let mut solver = Solver::new(&mut sources, &registry, &signatures);

        let decision = solver.decide(call_of(name, requiring(name, inner)));
        let unsettled = solver.settle();

        assert_eq!(
            solver.answer(decision),
            Some(Answer::Instance(InstanceKind::Extern(0)))
        );
        let [Unsettled::NoInstance { call, required, .. }] = unsettled.as_slice() else {
            panic!("the requirement reaches no instance: {unsettled:?}")
        };
        assert_eq!(*required, Some(inner));
        assert_eq!(
            solver.freeze_ty(call).expect("the call type is closed"),
            signature_of(name, TyTerm::I64, TyTerm::I64)
        );
    }
}
