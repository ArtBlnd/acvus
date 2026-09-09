use std::convert::Infallible;
use std::fmt;

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;
use crate::graph::types::QualifiedRef;

// ── UserDefined type system ──────────────────────────────────────────

/// Declaration of a user-defined type — the **single source of truth**
/// for parameter count and constraints. Registered once, referenced by QualifiedRef everywhere.
#[derive(Debug, Clone)]
pub struct UserDefinedDecl {
    pub qref: QualifiedRef,
    /// Type parameter constraints. `None` = unconstrained. `Some(vec)` = allowed types.
    pub type_params: Vec<Option<Vec<Ty>>>,
    pub effect_params: usize,
}

/// Immutable registry of all UserDefined type declarations and ExternCast rules.
/// Built once at setup, then frozen via `Freeze<TypeRegistry>` and shared everywhere.
///
/// Contains:
/// - `decls`: UserDefined type declarations (source of truth for params/constraints).
/// - `cast_rules`: ExternCast coercion rules (UserDefined → other type).
#[derive(Debug, Clone, Default)]
pub struct TypeRegistry {
    decls: FxHashMap<QualifiedRef, UserDefinedDecl>,
    /// ExternCast coercion rules, indexed by source UserDefined QualifiedRef.
    /// `from_rules`: keyed by the `from` type's QualifiedRef.
    /// `to_rules`: keyed by the `to` type's QualifiedRef (when target is UserDefined).
    // pub(crate) for test access.
    pub(crate) from_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
    pub(crate) to_rules: FxHashMap<QualifiedRef, Vec<CastRule>>,
}

/// A coercion rule: `from` can be implicitly converted to `to`.
/// Both `from` and `to` share positional Var placeholders (Poly phase),
/// so instantiating them together links corresponding parameters.
#[derive(Debug, Clone)]
pub struct CastRule {
    /// Source type pattern (must be UserDefined). May contain positional Var placeholders.
    pub from: PolyTy,
    /// Target type pattern (Var placeholders shared with `from`).
    pub to: PolyTy,
    /// The pure ExternFn that performs the conversion.
    pub fn_ref: QualifiedRef,
}

/// Head constructor of a type — used for duplicate cast rule detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum TyHead {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Byte,
    Array,
    Object,
    Tuple,
    Fn,
    Option,
    Enum,
    Handle,
    Identity,
    Ref,
    UserDefined(QualifiedRef),
    Error,
}

fn ty_head<V: Phase>(ty: &TyTerm<V>) -> TyHead {
    match ty {
        TyTerm::Int => TyHead::Int,
        TyTerm::Float => TyHead::Float,
        TyTerm::String => TyHead::String,
        TyTerm::Bool => TyHead::Bool,
        TyTerm::Unit => TyHead::Unit,
        TyTerm::Byte => TyHead::Byte,
        TyTerm::Array(..) => TyHead::Array,
        TyTerm::Object(_) => TyHead::Object,
        TyTerm::Tuple(_) => TyHead::Tuple,
        TyTerm::Fn { .. } => TyHead::Fn,
        TyTerm::Option(_) => TyHead::Option,
        TyTerm::Enum { .. } => TyHead::Enum,
        TyTerm::Handle(..) => TyHead::Handle,
        TyTerm::Identity(..) => TyHead::Identity,
        TyTerm::Ref(..) => TyHead::Ref,
        TyTerm::UserDefined { id, .. } => TyHead::UserDefined(*id),
        TyTerm::Error(_) => TyHead::Error,
        TyTerm::Var(_) => TyHead::Error,
    }
}

impl TypeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Type declarations ───────────────────────────────────────────

    /// Register a declaration. Panics on duplicate qref.
    pub fn register(&mut self, decl: UserDefinedDecl) {
        let qref = decl.qref;
        let prev = self.decls.insert(qref, decl);
        assert!(prev.is_none(), "duplicate UserDefined type: {qref:?}");
    }

    /// Look up a declaration by qref. Panics if not found — missing decl is a bug.
    pub fn get(&self, qref: QualifiedRef) -> &UserDefinedDecl {
        self.decls
            .get(&qref)
            .unwrap_or_else(|| panic!("unknown UserDefined type: {qref:?}"))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&QualifiedRef, &UserDefinedDecl)> {
        self.decls.iter()
    }

    // ── Cast rules ──────────────────────────────────────────────────

    /// Register a cast rule. Indexes by `from`'s QualifiedRef (if UserDefined)
    /// and by `to`'s QualifiedRef (if UserDefined). At least one side must be UserDefined.
    pub fn register_cast(&mut self, rule: CastRule) {
        let from_qref = match &rule.from {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        let to_qref = match &rule.to {
            TyTerm::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        assert!(
            from_qref.is_some() || to_qref.is_some(),
            "CastRule: at least one side must be UserDefined"
        );

        // Duplicate check (same from head + same to head).
        let from_head = ty_head(&rule.from);
        let to_head = ty_head(&rule.to);
        if let Some(fq) = from_qref {
            for existing in self.from_rules.get(&fq).into_iter().flatten() {
                assert!(
                    !(ty_head(&existing.from) == from_head && ty_head(&existing.to) == to_head),
                    "duplicate CastRule: same from and to head constructor"
                );
            }
        }

        // Index by from (if UserDefined).
        if let Some(fq) = from_qref {
            self.from_rules.entry(fq).or_default().push(rule.clone());
        }
        // Index by to (if UserDefined).
        if let Some(tq) = to_qref {
            self.to_rules.entry(tq).or_default().push(rule);
        }
    }

    /// Get all cast rules where `from` matches the given QualifiedRef.
    pub fn rules_from(&self, qref: QualifiedRef) -> &[CastRule] {
        self.from_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }

    /// Get all cast rules where `to` is a UserDefined matching the given QualifiedRef.
    pub fn rules_to(&self, qref: QualifiedRef) -> &[CastRule] {
        self.to_rules.get(&qref).map_or(&[], |v| v.as_slice())
    }
}

/// A named, typed function parameter.
pub type Param = ParamTerm<Concrete>;

/// Token for `Ty::Error` construction.
///
/// `Ty::Error` is a **poison type** — it suppresses cascading errors by unifying
/// with anything. Permitted uses:
///
/// - **Type checker / compiler**: After reporting a type error, return `Ty::error()`
///   so compilation continues and collects all errors (not just the first one).
/// - **Deserialization recovery**: When loading a persisted type that can't be parsed.
///
/// **Forbidden uses**:
///
/// - As a "don't know" placeholder (use the actual type instead).
/// - As a default/fallback when you're too lazy to propagate the real type.
/// - In runtime code paths — Error must never appear in a running program's types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorToken(());

impl ErrorToken {
    pub(crate) fn new() -> Self {
        Self(())
    }
}

/// Polarity for subtyping direction in unification.
///
/// - `Covariant`: `a ≤ b` — `a` may be a subtype of `b`.
/// - `Contravariant`: `b ≤ a` — reversed direction (e.g. function parameters).
/// - `Invariant`: `a = b` — no subtyping allowed, must be exactly equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Covariant,
    Contravariant,
    Invariant,
}

impl Polarity {
    /// Flip polarity: Covariant ↔ Contravariant, Invariant stays.
    pub fn flip(self) -> Self {
        match self {
            Polarity::Covariant => Polarity::Contravariant,
            Polarity::Contravariant => Polarity::Covariant,
            Polarity::Invariant => Polarity::Invariant,
        }
    }
}

/// 3-tier purity classification for types.
///
/// `Concrete` — scalars that can cross context boundaries as-is.
/// `Composite` — containers, closures, iterators — need deep inspection to determine pureability.
/// `Ephemeral` — opaque types that can never be purified.
///
/// `Ord` derive: `Concrete < Composite < Ephemeral`, so `max()` gives the least-pure tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Materiality {
    Concrete,
    Composite,
    Ephemeral,
}

/// Effect level of a call. The derived order is the chain `Pure < Idempotent < Opaque`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Effect {
    Pure,
    Idempotent,
    Opaque,
}

impl Effect {
    pub fn join(self, other: Effect) -> Effect {
        self.max(other)
    }
}

/// A required effect level that exceeds the level allowed at that point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectConflict {
    pub required: Effect,
    pub allowed: Effect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectTerm<V: Phase> {
    Known(Effect),
    Var(V::EffectVar),
}

impl<V: Phase> From<Effect> for EffectTerm<V> {
    fn from(effect: Effect) -> Self {
        EffectTerm::Known(effect)
    }
}

impl<V: Phase> EffectTerm<V> {
    pub fn map<W: Phase>(&self, on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>) -> EffectTerm<W> {
        match self {
            EffectTerm::Known(e) => EffectTerm::Known(*e),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
    ) -> Result<EffectTerm<W>, E> {
        match self {
            EffectTerm::Known(e) => Ok(EffectTerm::Known(*e)),
            EffectTerm::Var(v) => on_effect(*v),
        }
    }
}

impl EffectTerm<Concrete> {
    pub fn get(&self) -> Effect {
        match self {
            EffectTerm::Known(e) => *e,
            EffectTerm::Var(v) => match *v {},
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenTerm<V: Phase> {
    Known(usize),
    Var(V::LenVar),
}

impl<V: Phase> LenTerm<V> {
    pub fn map<W: Phase>(&self, on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>) -> LenTerm<W> {
        match self {
            LenTerm::Known(n) => LenTerm::Known(*n),
            LenTerm::Var(v) => on_len(*v),
        }
    }

    pub fn try_map<W: Phase, E>(
        &self,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
    ) -> Result<LenTerm<W>, E> {
        match self {
            LenTerm::Known(n) => Ok(LenTerm::Known(*n)),
            LenTerm::Var(v) => on_len(*v),
        }
    }
}

impl LenTerm<Concrete> {
    pub fn get(&self) -> usize {
        match self {
            LenTerm::Known(n) => *n,
            LenTerm::Var(v) => match *v {},
        }
    }
}

// ── Identity system ──────────────────────────────────────────────────

acvus_utils::declare_local_id!(pub IdentityId);

impl std::fmt::Display for IdentityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Identity({self:?})")
    }
}

/// Concrete type — always fully resolved. `Var(Infallible)` is uninhabitable.
pub type Ty = TyTerm<Concrete>;

impl TyTerm<Concrete> {
    /// Create an `Error` (poison) type. See [`ErrorToken`] for permitted uses.
    pub fn error() -> Self {
        Ty::Error(ErrorToken::new())
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error(_))
    }

    pub fn effect(&self) -> Option<Effect> {
        match self {
            Ty::Fn { effect, .. } => Some(effect.get()),
            _ => None,
        }
    }

    /// Extract the element type from a collection type.
    pub fn elem_of(&self) -> Option<&Ty> {
        match self {
            Ty::Array(elem, _) => Some(elem),
            _ => None,
        }
    }

    /// Returns the purity tier of this type (shallow — does not recurse into containers).
    pub fn materiality(&self) -> Materiality {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Byte => {
                Materiality::Concrete
            }
            Ty::Array(..)
            | Ty::Object(_)
            | Ty::Tuple(_)
            | Ty::Fn { .. }
            | Ty::Handle(..)
            | Ty::Option(_)
            | Ty::Enum { .. } => Materiality::Composite,
            Ty::UserDefined { .. } => Materiality::Ephemeral,
            Ty::Identity(_) => Materiality::Concrete,
            Ty::Ref(..) => Materiality::Ephemeral,
            Ty::Error(_) => Materiality::Ephemeral,
            Ty::Var(v) => match *v {},
        }
    }

    /// Returns true if this type can be deeply converted to a pure representation.
    pub fn is_pureable(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Byte => true,
            Ty::Array(inner, _) => inner.is_pureable(),
            Ty::Handle(inner) => inner.is_pureable(),
            Ty::Option(inner) => inner.is_pureable(),
            Ty::Tuple(elems) => elems.iter().all(|e| e.is_pureable()),
            Ty::Object(fields) => fields.values().all(|v| v.is_pureable()),
            Ty::Enum { variants, .. } => variants
                .values()
                .all(|p| p.as_ref().is_none_or(|ty| ty.is_pureable())),
            Ty::Fn { captures, ret, .. } => {
                captures.iter().all(|c| c.is_pureable()) && ret.is_pureable()
            }
            Ty::UserDefined { .. } | Ty::Ref(..) | Ty::Error(_) => false,
            Ty::Identity(_) => true,
            Ty::Var(v) => match *v {},
        }
    }

    /// Returns true if this type can be materialized.
    pub fn is_materializable(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Byte => true,
            Ty::Array(inner, _) => inner.is_materializable(),
            Ty::Option(inner) => inner.is_materializable(),
            Ty::Tuple(elems) => elems.iter().all(|e| e.is_materializable()),
            Ty::Object(fields) => fields.values().all(|v| v.is_materializable()),
            Ty::Enum { variants, .. } => variants
                .values()
                .all(|p| p.as_ref().is_none_or(|ty| ty.is_materializable())),
            Ty::Handle(..)
            | Ty::Fn { .. }
            | Ty::UserDefined { .. }
            | Ty::Identity(_)
            | Ty::Ref(..)
            | Ty::Error(_) => false,
            Ty::Var(v) => match *v {},
        }
    }

    pub fn display<'a>(&'a self, interner: &'a Interner) -> TyDisplay<'a> {
        TyDisplay { ty: self, interner }
    }
}

pub struct TyDisplay<'a> {
    ty: &'a Ty,
    interner: &'a Interner,
}

impl<'a> fmt::Display for TyDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ty {
            Ty::Int => write!(f, "Int"),
            Ty::Float => write!(f, "Float"),
            Ty::String => write!(f, "String"),
            Ty::Bool => write!(f, "Bool"),
            Ty::Unit => write!(f, "Unit"),
            Ty::Byte => write!(f, "Byte"),
            Ty::Object(fields) => {
                let mut sorted: Vec<_> = fields.iter().collect();
                sorted.sort_by_key(|(k, _)| self.interner.resolve(**k).to_string());
                write!(f, "{{")?;
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{}: {}",
                        self.interner.resolve(**k),
                        v.display(self.interner)
                    )?;
                }
                write!(f, "}}")
            }
            Ty::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", e.display(self.interner))?;
                }
                write!(f, ")")
            }
            Ty::Fn {
                params,
                ret,
                captures: _,
                effect,
            } => {
                write!(f, "Fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.ty.display(self.interner))?;
                }
                write!(f, ") -> {}", ret.display(self.interner))?;
                match effect.get() {
                    Effect::Pure => Ok(()),
                    other => write!(f, " with {other:?}"),
                }
            }
            Ty::Array(inner, len) => write!(f, "Array<{}, {}>", inner.display(self.interner), len.get()),
            Ty::Handle(inner) => {
                write!(f, "Handle<{}>", inner.display(self.interner))
            }
            Ty::Identity(id) => write!(f, "{id}"),
            Ty::Option(inner) => write!(f, "Option<{}>", inner.display(self.interner)),
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
            } => {
                let name = self.interner.resolve(id.name);
                write!(f, "{name}")?;
                if !type_args.is_empty() || !effect_args.is_empty() {
                    write!(f, "<")?;
                    let mut first = true;
                    for arg in type_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        write!(f, "{}", arg.display(self.interner))?;
                    }
                    for arg in effect_args {
                        if !first {
                            write!(f, ", ")?;
                        }
                        first = false;
                        write!(f, "{:?}", arg.get())?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            Ty::Enum { name, .. } => write!(f, "{}", self.interner.resolve(*name)),
            Ty::Ref(inner, volatile) => {
                if *volatile {
                    write!(f, "VolatileRef<{}>", inner.display(self.interner))
                } else {
                    write!(f, "Ref<{}>", inner.display(self.interner))
                }
            }
            Ty::Error(_) => write!(f, "<error>"),
            Ty::Var(v) => match *v {},
        }
    }
}

impl<'a> fmt::Debug for TyDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

// ── TypeEnv ──────────────────────────────────────────────────────────

/// Unified type environment for the type checker.
///
/// Replaces `ContextTypeRegistry` + internal `BuiltinRegistry`.
/// The type checker receives this as its sole external input —
/// it does not know whether a function is a builtin, extern, or user-defined.
/// All keys are QualifiedRef — the canonical identifier.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Context variable types — may contain inference variables (Solver-scoped).
    pub contexts: FxHashMap<QualifiedRef, InferTy>,
    /// Function type templates — polymorphic, instantiated per call site.
    pub functions: FxHashMap<QualifiedRef, PolyTy>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            contexts: FxHashMap::default(),
            functions: FxHashMap::default(),
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

// ── Phase-parameterized type system ─────────────────────────────────
//
// `TyTerm<V>` is a type term parameterized over inference variables.
// Two phases:
//   - `Concrete`: no inference variables (Var = Infallible). Post-inference.
//   - `Infer`:    may contain inference variables (Var = TypeBoundId). During inference.
//
// `type Ty = TyTerm<Concrete>` — always fully resolved. Compiler enforces this.
// `type InferTy = TyTerm<Infer>` — may have holes. Solver fills them in.

/// Phase marker trait — determines what can appear in inference variable slots.
pub trait Phase: 'static + Clone {
    /// Type inference variable. `Infallible` for concrete (uninhabitable).
    type TyVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Effect inference variable. `Infallible` for concrete (uninhabitable).
    type EffectVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
    /// Array length inference variable. `Infallible` for concrete (uninhabitable).
    type LenVar: fmt::Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy;
}

/// Post-inference phase — all types fully resolved.
/// `TyVar = Infallible` makes `TyTerm::Var` uninhabitable at type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concrete;

impl Phase for Concrete {
    type TyVar = Infallible;
    type EffectVar = Infallible;
    type LenVar = Infallible;
}

/// Polymorphic declaration phase — type templates stored in the graph.
/// `TyVar = u32` is a positional placeholder, not tied to any Solver instance.
/// Instantiated to `Infer` per call site during type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Poly;

impl Phase for Poly {
    type TyVar = u32;
    type EffectVar = u32;
    type LenVar = u32;
}

/// During-inference phase — types may contain unresolved variables.
/// `TyVar = TypeBoundId` is scoped to a specific `Solver` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infer;

impl Phase for Infer {
    type TyVar = TypeBoundId;
    type EffectVar = EffectVarId;
    type LenVar = LenVarId;
}

/// Polymorphic type — template with positional placeholders.
pub type PolyTy = TyTerm<Poly>;
/// Polymorphic function parameter.
pub type PolyParam = ParamTerm<Poly>;

// ── Solver types ────────────────────────────────────────────────────

/// Index into `Solver::ty_bounds`. Identifies a type inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeBoundId(pub u32);

/// Index into `Solver::effect_vars`. Identifies an effect inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectVarId(pub u32);

/// Index into `Solver::len_vars`. Identifies an array length inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LenVarId(pub u32);

// Re-export solver types — these were historically in ty.rs.
pub use crate::solver::{Capability, TypeBound, Solver, SolverSnapshot, FreezeError};

/// Type alias — always concrete, no inference variables.
pub type InferTy = TyTerm<Infer>;

/// A type term parameterized over inference phase.
///
/// When `V = Concrete`: `Var(Infallible)` is uninhabitable — type is always concrete.
/// When `V = Infer`: `Var(TypeBoundId)` references the solver's bound table.
#[derive(Debug, Clone, PartialEq)]
pub enum TyTerm<V: Phase> {
    // Primitives
    Int,
    Float,
    String,
    Bool,
    Unit,
    Byte,
    // Containers
    Array(Box<TyTerm<V>>, LenTerm<V>),
    Object(FxHashMap<Astr, TyTerm<V>>),
    Tuple(Vec<TyTerm<V>>),
    Option(Box<TyTerm<V>>),
    // Functions
    Fn {
        params: Vec<ParamTerm<V>>,
        ret: Box<TyTerm<V>>,
        captures: Vec<TyTerm<V>>,
        effect: EffectTerm<V>,
    },
    // Nominal
    UserDefined {
        id: QualifiedRef,
        type_args: Vec<TyTerm<V>>,
        effect_args: Vec<EffectTerm<V>>,
    },
    Enum {
        name: Astr,
        variants: FxHashMap<Astr, Option<Box<TyTerm<V>>>>,
    },
    // Resources
    Handle(Box<TyTerm<V>>),
    Identity(IdentityId),
    Ref(Box<TyTerm<V>>, bool),
    // Special
    Error(ErrorToken),
    /// Inference variable — only inhabitable when `V = Infer`.
    /// For `V = Concrete`, this is `Var(Infallible)` which cannot be constructed.
    Var(V::TyVar),
}

/// Named, typed function parameter — parameterized over phase.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamTerm<V: Phase> {
    pub name: Astr,
    pub ty: TyTerm<V>,
}

impl<V: Phase> ParamTerm<V> {
    pub fn new(name: Astr, ty: TyTerm<V>) -> Self {
        Self { name, ty }
    }
}

// ── Generic phase traversal ────────────────────────────────────────
//
// `map` and `try_map` provide the single recursive traversal over
// `TyTerm<V>`. All phase-to-phase transformations (lift, freeze,
// resolve, instantiate) are specializations of these two operations.

impl<V: Phase> TyTerm<V> {
    /// Map this type term from phase `V` to phase `W`.
    ///
    /// Structural recursion is automatic — only variable slots and
    /// identity slots need custom handling via the provided closures.
    pub fn map<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> TyTerm<W>,
        on_identity: &mut impl FnMut(IdentityId) -> TyTerm<W>,
        on_effect: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_len: &mut impl FnMut(V::LenVar) -> LenTerm<W>,
    ) -> TyTerm<W> {
        match self {
            TyTerm::Int => TyTerm::Int,
            TyTerm::Float => TyTerm::Float,
            TyTerm::String => TyTerm::String,
            TyTerm::Bool => TyTerm::Bool,
            TyTerm::Unit => TyTerm::Unit,
            TyTerm::Byte => TyTerm::Byte,
            TyTerm::Array(inner, len) => TyTerm::Array(Box::new(inner.map(on_var, on_identity, on_effect, on_len)), len.map(on_len)),
            TyTerm::Object(fields) => TyTerm::Object(
                fields.iter().map(|(k, v)| (*k, v.map(on_var, on_identity, on_effect, on_len))).collect(),
            ),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems.iter().map(|e| e.map(on_var, on_identity, on_effect, on_len)).collect(),
            ),
            TyTerm::Option(inner) => TyTerm::Option(Box::new(inner.map(on_var, on_identity, on_effect, on_len))),
            TyTerm::Fn { params, ret, captures, effect } => TyTerm::Fn {
                params: params.iter().map(|p| ParamTerm::new(p.name, p.ty.map(on_var, on_identity, on_effect, on_len))).collect(),
                ret: Box::new(ret.map(on_var, on_identity, on_effect, on_len)),
                captures: captures.iter().map(|c| c.map(on_var, on_identity, on_effect, on_len)).collect(),
                effect: effect.map(on_effect),
            },
            TyTerm::UserDefined { id, type_args, effect_args } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| t.map(on_var, on_identity, on_effect, on_len)).collect(),
                effect_args: effect_args.iter().map(|e| e.map(on_effect)).collect(),
            },
            TyTerm::Enum { name, variants } => TyTerm::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| {
                    (*tag, payload.as_ref().map(|ty| Box::new(ty.map(on_var, on_identity, on_effect, on_len))))
                }).collect(),
            },
            TyTerm::Handle(inner) => TyTerm::Handle(
                Box::new(inner.map(on_var, on_identity, on_effect, on_len)),
            ),
            TyTerm::Identity(id) => on_identity(*id),
            TyTerm::Ref(inner, volatile) => TyTerm::Ref(
                Box::new(inner.map(on_var, on_identity, on_effect, on_len)),
                *volatile,
            ),
            TyTerm::Error(token) => TyTerm::Error(*token),
            TyTerm::Var(v) => on_var(*v),
        }
    }

    /// Fallible version of `map` — short-circuits on first error.
    pub fn try_map<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::TyVar) -> Result<TyTerm<W>, E>,
        on_identity: &mut impl FnMut(IdentityId) -> Result<TyTerm<W>, E>,
        on_effect: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_len: &mut impl FnMut(V::LenVar) -> Result<LenTerm<W>, E>,
    ) -> Result<TyTerm<W>, E> {
        match self {
            TyTerm::Int => Ok(TyTerm::Int),
            TyTerm::Float => Ok(TyTerm::Float),
            TyTerm::String => Ok(TyTerm::String),
            TyTerm::Bool => Ok(TyTerm::Bool),
            TyTerm::Unit => Ok(TyTerm::Unit),
            TyTerm::Byte => Ok(TyTerm::Byte),
            TyTerm::Array(inner, len) => Ok(TyTerm::Array(Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?), len.try_map(on_len)?)),
            TyTerm::Object(fields) => {
                let mapped: Result<FxHashMap<_, _>, E> = fields.iter()
                    .map(|(k, v)| v.try_map(on_var, on_identity, on_effect, on_len).map(|mv| (*k, mv)))
                    .collect();
                Ok(TyTerm::Object(mapped?))
            }
            TyTerm::Tuple(elems) => Ok(TyTerm::Tuple(
                elems.iter().map(|e| e.try_map(on_var, on_identity, on_effect, on_len)).collect::<Result<_, _>>()?,
            )),
            TyTerm::Option(inner) => Ok(TyTerm::Option(Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?))),
            TyTerm::Fn { params, ret, captures, effect } => Ok(TyTerm::Fn {
                params: params.iter()
                    .map(|p| p.ty.try_map(on_var, on_identity, on_effect, on_len).map(|ty| ParamTerm::new(p.name, ty)))
                    .collect::<Result<_, _>>()?,
                ret: Box::new(ret.try_map(on_var, on_identity, on_effect, on_len)?),
                captures: captures.iter().map(|c| c.try_map(on_var, on_identity, on_effect, on_len)).collect::<Result<_, _>>()?,
                effect: effect.try_map(on_effect)?,
            }),
            TyTerm::UserDefined { id, type_args, effect_args } => Ok(TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| t.try_map(on_var, on_identity, on_effect, on_len)).collect::<Result<_, _>>()?,
                effect_args: effect_args.iter().map(|e| e.try_map(on_effect)).collect::<Result<_, _>>()?,
            }),
            TyTerm::Enum { name, variants } => {
                let mapped: Result<FxHashMap<_, _>, E> = variants.iter()
                    .map(|(tag, payload)| {
                        let mp = match payload {
                            Some(ty) => Some(Box::new(ty.try_map(on_var, on_identity, on_effect, on_len)?)),
                            None => None,
                        };
                        Ok((*tag, mp))
                    })
                    .collect();
                Ok(TyTerm::Enum { name: *name, variants: mapped? })
            }
            TyTerm::Handle(inner) => Ok(TyTerm::Handle(
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?),
            )),
            TyTerm::Identity(id) => on_identity(*id),
            TyTerm::Ref(inner, volatile) => Ok(TyTerm::Ref(
                Box::new(inner.try_map(on_var, on_identity, on_effect, on_len)?),
                *volatile,
            )),
            TyTerm::Error(token) => Ok(TyTerm::Error(*token)),
            TyTerm::Var(v) => on_var(*v),
        }
    }
}

// ── Lift: Concrete → any Phase ──────────────────────────────────────

/// Lift a concrete `Ty` into any phase (mechanical, zero information change).
/// Infallible because `Concrete` has `TyVar = Infallible` (uninhabitable).
pub fn lift_ty<W: Phase>(ty: &Ty) -> TyTerm<W> {
    ty.map(
        &mut |v: Infallible| match v {},
        &mut |id| TyTerm::Identity(id),
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
    )
}

/// Alias: lift `Ty` to `PolyTy`. Kept for call-site compatibility.
pub fn lift_to_poly(ty: &Ty) -> PolyTy { lift_ty(ty) }

/// Try to convert a `PolyTy` to a concrete `Ty`.
/// Returns `None` if the poly type contains any Var placeholders.
pub fn try_freeze_poly(ty: &PolyTy) -> Option<Ty> {
    ty.try_map(
        &mut |_: u32| Err(()),
        &mut |id| Ok(TyTerm::Identity(id)),
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
    ).ok()
}

// ── PolyBuilder ─────────────────────────────────────────────────────

/// Builder for polymorphic type templates. No Solver dependency.
/// Creates positional placeholders (Var(0), Var(1), ...) for type variables.
pub struct PolyBuilder {
    next_ty: u32,
    next_effect: u32,
    next_len: u32,
}

impl PolyBuilder {
    pub fn new() -> Self {
        Self { next_ty: 0, next_effect: 0, next_len: 0 }
    }

    pub fn fresh_len_var(&mut self) -> LenTerm<Poly> {
        let id = self.next_len;
        self.next_len += 1;
        LenTerm::Var(id)
    }

    pub fn fresh_effect_var(&mut self) -> EffectTerm<Poly> {
        let id = self.next_effect;
        self.next_effect += 1;
        EffectTerm::Var(id)
    }

    /// Create a fresh type placeholder.
    pub fn fresh_ty_var(&mut self) -> PolyTy {
        let id = self.next_ty;
        self.next_ty += 1;
        TyTerm::Var(id)
    }

    pub fn alloc_identity(&mut self, factory: &mut acvus_utils::LocalFactory<IdentityId>) -> PolyTy {
        TyTerm::Identity(factory.next())
    }
}

impl Default for PolyBuilder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::QualifiedRef;
    use acvus_utils::Interner;

    use Polarity::*;

    fn arr<V: Phase>(elem: TyTerm<V>, n: usize) -> TyTerm<V> {
        TyTerm::Array(Box::new(elem), LenTerm::Known(n))
    }

    /// Test helper: create a unique `QualifiedRef` for each call.
    /// Uses a thread-local counter to ensure uniqueness across tests.
    fn fresh_qref() -> QualifiedRef {
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        INTERNER.with(|i| QualifiedRef::root(i.intern(&format!("TestType{n}"))))
    }

    /// Test helper: create a `TyTerm::UserDefined` with a fresh id and no type args.
    fn test_user_defined() -> Ty {
        TyTerm::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
        }
    }

    #[test]
    fn unify_same_concrete() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        assert!(s.unify_ty(&TyTerm::Int, &TyTerm::Int, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&TyTerm::Float, &TyTerm::Float, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&TyTerm::String, &TyTerm::String, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&TyTerm::Bool, &TyTerm::Bool, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&TyTerm::Unit, &TyTerm::Unit, Invariant, &registry).is_ok());
    }

    #[test]
    fn unify_different_concrete_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        assert!(s.unify_ty(&TyTerm::Int, &TyTerm::Float, Invariant, &registry).is_err());
        assert!(s.unify_ty(&TyTerm::String, &TyTerm::Bool, Invariant, &registry).is_err());
    }

    #[test]
    fn unify_var_with_concrete() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let t = s.fresh_ty_var();
        assert!(s.unify_ty(&t, &TyTerm::Int, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::Int);
    }

    #[test]
    fn unify_object() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let interner = Interner::new();
        let t = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), t.clone()),
        ]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([
            (interner.intern("name"), TyTerm::String),
            (interner.intern("age"), TyTerm::Int),
        ]));
        assert!(s.unify_ty(&obj1, &obj2, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::Int);
    }

    #[test]
    fn unify_object_key_mismatch() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let interner = Interner::new();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(
            interner.intern("name"),
            TyTerm::String,
        )]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([(interner.intern("age"), TyTerm::Int)]));
        assert!(s.unify_ty(&obj1, &obj2, Invariant, &registry).is_err());
    }

    #[test]
    fn transitive_resolution() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let t1 = s.fresh_ty_var();
        let t2 = s.fresh_ty_var();
        assert!(s.unify_ty(&t1, &t2, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&t2, &TyTerm::String, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t1), TyTerm::String);
    }

    // -- Object merge tests --

    #[test]
    fn unify_object_disjoint_via_var() {
        // Var → {a} then Var → {b} should merge to {a, b}
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_a = TyTerm::Object(FxHashMap::from_iter([(i.intern("a"), TyTerm::Int)]));
        let obj_b = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::String)]));
        assert!(s.unify_ty(&v, &obj_a, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj_b, Invariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 2, "expected {{a, b}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::Int));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlapping_via_var() {
        // Var → {a, b} then Var → {b, c} should merge to {a, b, c}
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj_ab = TyTerm::Object(FxHashMap::from_iter([
            (i.intern("a"), TyTerm::Int),
            (i.intern("b"), TyTerm::String),
        ]));
        let obj_bc = TyTerm::Object(FxHashMap::from_iter([
            (i.intern("b"), TyTerm::String),
            (i.intern("c"), TyTerm::Bool),
        ]));
        assert!(s.unify_ty(&v, &obj_ab, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj_bc, Invariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 3, "expected {{a, b, c}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&TyTerm::Int));
                assert_eq!(fields.get(&i.intern("b")), Some(&TyTerm::String));
                assert_eq!(fields.get(&i.intern("c")), Some(&TyTerm::Bool));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlap_type_conflict_fails() {
        // {b: Int} and {b: String} via same Var should fail
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let v = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::Int)]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([(i.intern("b"), TyTerm::String)]));
        assert!(s.unify_ty(&v, &obj1, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj2, Invariant, &registry).is_err());
    }

    #[test]
    fn fresh_param_produces_unique_ids() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.fresh_ty_var();
        let o2 = s.fresh_ty_var();
        let o3 = s.fresh_ty_var();
        assert_ne!(o1, o2);
        assert_ne!(o2, o3);
        assert_ne!(o1, o3);
    }

    #[test]
    fn snapshot_rollback_preserves_identity_counter() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let _id1 = s.alloc_identity();
        let snap = s.snapshot();
        let id2 = s.alloc_identity();
        s.rollback(snap);
        let id_after = s.alloc_identity();
        // After rollback, next_identity should be restored, so id_after == id2.
        assert_eq!(id_after, id2, "rollback should restore identity counter");
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn polarity_flip() {
        assert_eq!(Covariant.flip(), Contravariant);
        assert_eq!(Contravariant.flip(), Covariant);
        assert_eq!(Invariant.flip(), Invariant);
    }

    // -- Variance unsoundness edge case tests --

    // ================================================================
    // Var chain + coercion 상호작용
    // ================================================================

    // ================================================================
    // Occurs check + polarity
    // ================================================================

    #[test]
    fn occurs_check_through_list_covariant() {
        // Var = List<Var> should fail (occurs) regardless of polarity.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        let cyclic = arr(v.clone(), 3);
        assert!(s.unify_ty(&v, &cyclic, Covariant, &registry).is_err());
    }

    // ================================================================
    // Deep nesting coercion
    // ================================================================

    // ================================================================
    // Object merge + coercion 동시 발생
    // ================================================================

    // ================================================================
    // Snapshot/rollback isolation
    // ================================================================

    // ================================================================
    // Polarity symmetry / duality 검증
    // ================================================================

    #[test]
    fn invariant_same_types_both_directions() {
        // Same concrete type: Invariant must succeed regardless of order.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let l1 = arr(TyTerm::Int, 3);
        let l2 = arr(TyTerm::Int, 3);
        assert!(s.unify_ty(&l1, &l2, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&l2, &l1, Invariant, &registry).is_ok());
    }

    // ================================================================
    // Unresolved Var containers + coercion
    // ================================================================

    // ================================================================
    // Bidirectional Var binding + coercion
    // ================================================================

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    // ================================================================
    // Mixed concrete/param identities
    // ================================================================

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    // ================================================================
    // Transitive coercion chains
    // ================================================================

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    // ================================================================
    // Coercion does NOT propagate across unrelated type constructors
    // ================================================================

    #[test]
    fn list_vs_tuple_fails_any_polarity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let l = arr(TyTerm::Int, 3);
        let t = TyTerm::Tuple(vec![TyTerm::Int]);
        assert!(s.unify_ty(&l, &t, Covariant, &registry).is_err());
        assert!(s.unify_ty(&l, &t, Invariant, &registry).is_err());
    }

    // ================================================================
    // Triple flip (Fn<Fn<Fn<...>>>)
    // ================================================================

    // ================================================================
    // Regression: same identity must not trigger demotion
    // ================================================================

    // ── Sequence identity tracking ─────────────────────────────────

    // ── UserDefined unification tests ───────────────────────────────

    fn ud(id: QualifiedRef, type_args: Vec<InferTy>) -> InferTy {
        TyTerm::UserDefined {
            id,
            type_args,
            effect_args: vec![],
        }
    }

    // -- Completeness: valid UserDefined unifications --

    #[test]
    fn user_defined_same_id_empty_args_unifies() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(&ud(id, vec![]), &ud(id, vec![]), Invariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn user_defined_same_id_concrete_type_args_unifies() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id, vec![TyTerm::Int]),
                &ud(id, vec![TyTerm::Int]),
                Invariant, &registry)
            .is_ok()
        );
    }

    #[test]
    fn user_defined_param_type_arg_resolved_via_unify() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &ud(id, vec![p.clone()]),
                &ud(id, vec![TyTerm::Int]),
                Invariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::Int);
    }

    #[test]
    fn user_defined_nested_type_arg_unifies() {
        // UserDefined<List<Param>> vs UserDefined<List<Int>> → resolves Param to Int
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &ud(id, vec![arr(p.clone(), 3)]),
                &ud(id, vec![arr(TyTerm::Int, 3)]),
                Invariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::Int);
    }

    // -- Soundness: invalid UserDefined unifications --

    #[test]
    fn user_defined_different_id_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id_a = fresh_qref();
        let id_b = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id_a, vec![]),
                &ud(id_b, vec![]),
                Invariant, &registry)
            .is_err()
        );
    }

    #[test]
    fn user_defined_type_arg_mismatch_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id, vec![TyTerm::Int]),
                &ud(id, vec![TyTerm::String]),
                Invariant, &registry)
            .is_err()
        );
    }

    #[test]
    fn user_defined_vs_other_ty_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(&ud(id, vec![]), &TyTerm::Int, Invariant, &registry)
                .is_err()
        );
        assert!(
            s.unify_ty(&TyTerm::String, &ud(id, vec![]), Invariant, &registry)
                .is_err()
        );
    }

    // -- Resolve --

    #[test]
    fn user_defined_inside_list_resolves() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        let ty = arr(ud(id, vec![p.clone()]), 3);
        assert!(s.unify_ty(&p, &TyTerm::Int, Invariant, &registry).is_ok());
        match s.resolve_ty(&ty) {
            TyTerm::Array(inner, _) => match *inner {
                TyTerm::UserDefined { type_args, .. } => assert_eq!(type_args, vec![TyTerm::Int]),
                other => panic!("expected UserDefined, got {other:?}"),
            },
            other => panic!("expected Array, got {other:?}"),
        }
    }

    // -- TypeRegistry --

    #[test]
    fn type_registry_register_and_get() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: 0,
        });
        let decl = reg.get(id);
        assert_eq!(decl.qref, id);
        assert_eq!(decl.type_params.len(), 1);
    }

    #[test]
    #[should_panic(expected = "duplicate")]
    fn type_registry_duplicate_panics() {
        let mut reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: 0,
        });
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: 0,
        });
    }

    #[test]
    #[should_panic(expected = "unknown")]
    fn type_registry_unknown_id_panics() {
        let reg = TypeRegistry::new();
        let id = fresh_qref();
        reg.get(id);
    }

    // ── ExternCast tests ────────────────────────────────────────────

    /// Helper: create a CastRule and a Solver with the rule registered.
    /// Returns (from_id, fn_id, solver, registry).
    /// CastRule uses PolyTy (positional Var placeholders from PolyBuilder).
    /// Solver is fresh for test unification.
    fn make_cast_solver(
        type_param_count: usize,
        build_to: impl FnOnce(&[PolyTy]) -> PolyTy,
    ) -> (QualifiedRef, QualifiedRef, Solver, TypeRegistry) {
        let id = fresh_qref();
        let i = acvus_utils::Interner::new();
        let fn_id = QualifiedRef::root(i.intern("cast_fn"));
        let mut builder = PolyBuilder::new();
        let params: Vec<PolyTy> = (0..type_param_count)
            .map(|_| builder.fresh_ty_var())
            .collect();
        let from = TyTerm::UserDefined {
            id,
            type_args: params.clone(),
            effect_args: vec![],
        };
        let to = build_to(&params);
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None; type_param_count],
            effect_params: 0,
        });
        reg.register_cast(CastRule {
            from,
            to,
            fn_ref: fn_id,
        });
        let solver = Solver::new();
        (id, fn_id, solver, reg)
    }

    // -- Completeness: valid ExternCast coercions --

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_basic_coercion() {
        // UserDefined(A, [T]) → List<T>
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| arr(p[0].clone(), 3));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],

        };
        let to = arr(TyTerm::Int, 3);
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
    }

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_with_param_resolution() {
        // UserDefined(A, [T]) → List<T>, where T is a fresh param on the consumer side
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| arr(p[0].clone(), 3));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],

        };
        let consumer_param = s.fresh_ty_var();
        let to = arr(consumer_param.clone(), 3);
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&consumer_param), TyTerm::Int);
    }

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_no_type_params() {
        // UserDefined(A, []) → Int
        let (id, _fn_id, mut s, registry) = make_cast_solver(0, |_| TyTerm::Int);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],

        };
        assert!(s.unify_ty(&from, &TyTerm::Int, Covariant, &registry).is_ok());
    }

    // -- Soundness: invalid ExternCast --

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_wrong_target_fails() {
        // Rule: A → List<T>, but expected String
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| arr(p[0].clone(), 3));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],

        };
        assert!(s.unify_ty(&from, &TyTerm::String, Covariant, &registry).is_err());
    }

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_no_rule_fails() {
        // No cast rules registered
        let id = fresh_qref();
        let mut s = Solver::new();
        let registry = TypeRegistry::new();

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],

        };
        assert!(s.unify_ty(&from, &TyTerm::Int, Covariant, &registry).is_err());
    }

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_invariant_not_attempted() {
        // ExternCast only works in covariant/contravariant, not invariant
        let (id, _fn_id, mut s, registry) = make_cast_solver(0, |_| TyTerm::Int);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],

        };
        assert!(s.unify_ty(&from, &TyTerm::Int, Invariant, &registry).is_err());
    }

    // -- Ambiguity --

    #[ignore = "pending identity integration"]
    #[test]
    fn extern_cast_ambiguity_rejected() {
        // Bypass TypeRegistry duplicate check — inject two rules with same to head
        // directly into the type_registry to test try_extern_cast ambiguity detection.
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        // Use a PolyBuilder for the CastRule variables, separate Solver for unification.
        let mut s = Solver::new();
        let mut builder = PolyBuilder::new();
        let t1 = builder.fresh_ty_var();
        let rule_a = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
    
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        };
        let t2 = builder.fresh_ty_var();
        let rule_b = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
    
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        };

        // Build registry manually (bypassing register_cast duplicate check)
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: 0,

        });
        reg.from_rules.entry(id).or_default().push(rule_a);
        reg.from_rules.entry(id).or_default().push(rule_b);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],

        };
        assert!(
            s.unify_ty(&from, &arr(TyTerm::Int, 3), Covariant, &reg)
                .is_err()
        );
    }

    // -- TypeRegistry cast rules --

    #[ignore = "pending identity integration"]
    #[test]
    #[should_panic(expected = "duplicate")]
    fn cast_registry_duplicate_panics() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));
        let mut builder = PolyBuilder::new();
        let t = builder.fresh_ty_var();

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: 0,

        });
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t.clone()],
                effect_args: vec![],
    
            },
            to: arr(t.clone(), 3),
            fn_ref: fn_id_a,
        });
        // Same from_id + same to head (List) → panic
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
    
            },
            to: arr(t2, 3),
            fn_ref: fn_id_b,
        });
    }

    #[ignore = "pending identity integration"]
    #[test]
    fn cast_registry_different_to_head_ok() {
        let i = acvus_utils::Interner::new();
        let id = fresh_qref();
        let fn_id_a = QualifiedRef::root(i.intern("cast_a"));
        let fn_id_b = QualifiedRef::root(i.intern("cast_b"));

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: 0,

        });
        let mut builder1 = PolyBuilder::new();
        let t1 = builder1.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
    
            },
            to: arr(t1, 3),
            fn_ref: fn_id_a,
        });
        // Different to head (Option vs List) → ok
        let mut builder2 = PolyBuilder::new();
        let t2 = builder2.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
    
            },
            to: TyTerm::Option(Box::new(t2)),
            fn_ref: fn_id_b,
        });
        assert_eq!(reg.rules_from(id).len(), 2);
    }

    // ── Purity tier tests ──────────────────────────────────────────────

    #[test]
    fn purity_object_is_lazy() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(i.intern("x"), Ty::Int)]));
        assert_eq!(obj.materiality(), Materiality::Composite);
    }

    #[test]
    fn purity_enum_is_lazy() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Color"),
            variants: FxHashMap::from_iter([(i.intern("Red"), None), (i.intern("Green"), None)]),
        };
        assert_eq!(enum_ty.materiality(), Materiality::Composite);
    }

    #[test]
    fn purity_user_defined_is_unpure() {
        assert_eq!(test_user_defined().materiality(), Materiality::Ephemeral);
    }

    #[test]
    fn purity_special_types() {
        // Unresolved types are conservatively Unpure.
        assert_eq!(Ty::error().materiality(), Materiality::Ephemeral);
    }

    #[test]
    fn purity_ord_pure_lt_lazy_lt_unpure() {
        assert!(Materiality::Concrete < Materiality::Composite);
        assert!(Materiality::Composite < Materiality::Ephemeral);
        assert!(Materiality::Concrete < Materiality::Ephemeral);
        // max() gives least-pure tier
        assert_eq!(
            std::cmp::max(Materiality::Concrete, Materiality::Composite),
            Materiality::Composite
        );
        assert_eq!(
            std::cmp::max(Materiality::Composite, Materiality::Ephemeral),
            Materiality::Ephemeral
        );
        assert_eq!(
            std::cmp::max(Materiality::Concrete, Materiality::Ephemeral),
            Materiality::Ephemeral
        );
    }

    // ── is_pureable() transitive tests ─────────────────────────────────

    #[test]
    fn pureable_list_of_scalars() {
        assert!(arr(Ty::Int, 3).is_pureable());
        assert!(arr(Ty::String, 3).is_pureable());
    }

    #[test]
    fn pureable_list_of_user_defined_is_not_pureable() {
        let list_ud = arr(test_user_defined(), 3);
        assert!(!list_ud.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_scalars() {
        // List<List<Int>> — pureable
        let nested = arr(arr(Ty::Int, 3), 3);
        assert!(nested.is_pureable());
    }

    #[test]
    fn pureable_option_of_scalar() {
        assert!(Ty::Option(Box::new(Ty::Int)).is_pureable());
    }

    #[test]
    fn pureable_option_of_user_defined() {
        assert!(!Ty::Option(Box::new(test_user_defined())).is_pureable());
    }

    #[test]
    fn pureable_tuple_all_scalars() {
        assert!(Ty::Tuple(vec![Ty::Int, Ty::String, Ty::Bool]).is_pureable());
    }

    #[test]
    fn pureable_tuple_with_user_defined() {
        assert!(!Ty::Tuple(vec![Ty::Int, test_user_defined()]).is_pureable());
    }

    #[test]
    fn pureable_object_all_scalars() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([
            (i.intern("x"), Ty::Int),
            (i.intern("y"), Ty::String),
        ]));
        assert!(obj.is_pureable());
    }

    #[test]
    fn pureable_object_with_user_defined_value() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(
            i.intern("handle"),
            test_user_defined(),
        )]));
        assert!(!obj.is_pureable());
    }

    #[test]
    fn pureable_enum_all_scalar_payloads() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Result"),
            variants: FxHashMap::from_iter([
                (i.intern("Ok"), Some(Box::new(Ty::Int))),
                (i.intern("Err"), Some(Box::new(Ty::String))),
            ]),
        };
        assert!(enum_ty.is_pureable());
    }

    #[test]
    fn pureable_enum_no_payload() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Color"),
            variants: FxHashMap::from_iter([(i.intern("Red"), None), (i.intern("Green"), None)]),
        };
        assert!(enum_ty.is_pureable());
    }

    #[test]
    fn pureable_enum_with_user_defined_payload() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Wrap"),
            variants: FxHashMap::from_iter([(
                i.intern("Some"),
                Some(Box::new(test_user_defined())),
            )]),
        };
        assert!(!enum_ty.is_pureable());
    }

    #[test]
    fn pureable_user_defined_never() {
        assert!(!test_user_defined().is_pureable());
        assert!(!test_user_defined().is_pureable());
    }

    #[test]
    fn pureable_mixed_tuple_list_option() {
        // (Int, List<String>, Option<Bool>) — all pureable
        let ty = Ty::Tuple(vec![
            Ty::Int,
            arr(Ty::String, 3),
            Ty::Option(Box::new(Ty::Bool)),
        ]);
        assert!(ty.is_pureable());
    }

    #[test]
    fn pureable_mixed_tuple_list_user_defined() {
        // (Int, List<UserDefined>) — not pureable
        let ty = Ty::Tuple(vec![Ty::Int, arr(test_user_defined(), 3)]);
        assert!(!ty.is_pureable());
    }

    #[test]
    fn pureable_deeply_nested_containers() {
        // List<Option<Tuple<(Int, List<String>)>>> — pureable
        let inner = Ty::Tuple(vec![Ty::Int, arr(Ty::String, 3)]);
        let ty = arr(Ty::Option(Box::new(inner)), 3);
        assert!(ty.is_pureable());
    }

    #[test]
    fn pureable_deeply_nested_with_user_defined_leaf() {
        // List<Option<Tuple<(Int, UserDefined)>>> — not pureable
        let inner = Ty::Tuple(vec![Ty::Int, test_user_defined()]);
        let ty = arr(Ty::Option(Box::new(inner)), 3);
        assert!(!ty.is_pureable());
    }

    // ================================================================
    // is_storable tests
    // ================================================================

    // -- Pure scalars: always storable --

    #[test]
    fn storable_int() {
        assert!(Ty::Int.is_materializable());
    }
    #[test]
    fn storable_float() {
        assert!(Ty::Float.is_materializable());
    }
    #[test]
    fn storable_string() {
        assert!(Ty::String.is_materializable());
    }
    #[test]
    fn storable_bool() {
        assert!(Ty::Bool.is_materializable());
    }
    #[test]
    fn storable_unit() {
        assert!(Ty::Unit.is_materializable());
    }
    #[test]
    fn storable_byte() {
        assert!(Ty::Byte.is_materializable());
    }
    // -- Lazy containers with pure contents: storable --

    #[test]
    fn storable_list_of_int() {
        assert!(arr(Ty::Int, 3).is_materializable());
    }
    #[test]
    fn storable_option_string() {
        assert!(Ty::Option(Box::new(Ty::String)).is_materializable());
    }
    #[test]
    fn storable_tuple() {
        assert!(Ty::Tuple(vec![Ty::Int, Ty::String]).is_materializable());
    }

    // -- Iterator/Sequence: always Ephemeral, never materializable --

    // -- Iterator/Sequence with Effectful: also NOT materializable --

    // -- UserDefined: never storable --

    #[test]
    fn not_storable_user_defined() {
        assert!(!test_user_defined().is_materializable());
    }

    // -- Recursive: container with non-storable inner --

    #[test]
    fn not_storable_list_of_user_defined() {
        assert!(!arr(test_user_defined(), 3).is_materializable());
    }
}
