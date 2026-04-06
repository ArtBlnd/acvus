use std::collections::BTreeSet;
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
    /// Effect parameter constraints. `None` = unconstrained.
    pub effect_params: Vec<Option<EffectConstraint>>,
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
    Range,
    Byte,
    List,
    Object,
    Tuple,
    Fn,
    Option,
    Enum,
    Handle,
    Deque,
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
        TyTerm::Range => TyHead::Range,
        TyTerm::Byte => TyHead::Byte,
        TyTerm::List(_) => TyHead::List,
        TyTerm::Object(_) => TyHead::Object,
        TyTerm::Tuple(_) => TyHead::Tuple,
        TyTerm::Fn { .. } => TyHead::Fn,
        TyTerm::Option(_) => TyHead::Option,
        TyTerm::Enum { .. } => TyHead::Enum,
        TyTerm::Handle(..) => TyHead::Handle,
        TyTerm::Deque(..) => TyHead::Deque,
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

// ── Effect target (Context vs Token) ────────────────────────────────

/// Distinguishes SSA-compatible context refs from external shared-state tokens.
///
/// - `Context(QualifiedRef)` — SSA-compatible. Compiler may convert to
///   `context_uses`/`context_defs` on Spawn/Eval/FunctionCall.
/// - `Token(QualifiedRef)` — NOT SSA-compatible. Functions sharing the same
///   Token must execute sequentially; the compiler must never lift a Token
///   into SSA context_uses/context_defs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EffectTarget {
    /// Context — SSA-compatible. Compiler can optimize (Spawn/Eval).
    Context(QualifiedRef),
    /// Token — NOT SSA-compatible. Named external shared state.
    /// Functions sharing the same Token QualifiedRef must execute sequentially.
    Token(QualifiedRef),
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
/// - `Covariant`: `a ≤ b` — `a` may be a subtype of `b` (e.g. Deque → List).
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

/// Scheduling hint for ExternFn execution.
/// Not an effect — does not affect SSA ordering or correctness.
/// Used by spawn-split and scheduler decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hint {
    /// Function performs I/O (network, disk, etc.)
    Io,
    /// Function is CPU-intensive.
    CpuHeavy,
}

/// Fine-grained effect information: which contexts are read/written.
///
/// Pure = all fields empty. No separate variant needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct EffectSet {
    pub reads: BTreeSet<EffectTarget>,
    pub writes: BTreeSet<EffectTarget>,
}

impl EffectSet {
    pub fn is_pure(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }

    /// Whether this effect set contains writes — i.e. the value is modifying.
    /// Modifying values are move-only (no Clone) and require sequential execution.
    pub fn is_modifying(&self) -> bool {
        !self.writes.is_empty()
    }

    /// Union of two effect sets. All effects propagate (contagious).
    pub fn union(&self, other: &Self) -> Self {
        Self {
            reads: self.reads.union(&other.reads).copied().collect(),
            writes: self.writes.union(&other.writes).copied().collect(),
        }
    }
}

impl fmt::Display for EffectSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_pure() {
            return write!(f, "Pure");
        }
        write!(f, "Effectful(")?;
        let mut parts = Vec::new();
        if !self.reads.is_empty() {
            parts.push(format!("r={}", self.reads.len()));
        }
        if !self.writes.is_empty() {
            parts.push(format!("w={}", self.writes.len()));
        }
        write!(f, "{})", parts.join(", "))
    }
}

// ── Effect constraint (upper bound) ─────────────────────────────────

/// An upper bound on a set of context references.
/// `Any` allows all contexts, `Only(set)` restricts to the given set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectCap {
    /// Any context/token access allowed.
    Any,
    /// Only the specified targets allowed.
    Only(BTreeSet<EffectTarget>),
}

/// Constraint on a function's effect — the upper bound of what it may do.
/// Separate from `EffectSet` which represents *actual* observed effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectConstraint {
    pub reads: EffectCap,
    pub writes: EffectCap,
}

impl EffectConstraint {
    /// Pure constraint — no effects allowed at all.
    pub fn pure() -> Self {
        Self {
            reads: EffectCap::Only(BTreeSet::new()),
            writes: EffectCap::Only(BTreeSet::new()),
        }
    }

    /// Read-only constraint — any reads allowed, no writes.
    pub fn read_only() -> Self {
        Self {
            reads: EffectCap::Any,
            writes: EffectCap::Only(BTreeSet::new()),
        }
    }
}

/// Effect classification — always resolved in concrete phase.
pub type Effect = EffectTerm<Concrete>;

impl EffectTerm<Concrete> {
    /// No side effects: reads nothing, writes nothing.
    pub fn pure() -> Self {
        EffectTerm::Resolved(EffectSet::default())
    }

    pub fn is_pure(&self) -> bool {
        matches!(self, EffectTerm::Resolved(s) if s.is_pure())
    }

    /// Returns true if this is a resolved, non-pure effect.
    pub fn is_effectful(&self) -> bool {
        matches!(self, EffectTerm::Resolved(s) if !s.is_pure())
    }

    /// Union two resolved effects.
    pub fn union(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (EffectTerm::Resolved(a), EffectTerm::Resolved(b)) => {
                Some(EffectTerm::Resolved(a.union(b)))
            }
            _ => None,
        }
    }
}

impl fmt::Display for EffectTerm<Concrete> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectTerm::Resolved(set) => write!(f, "{set}"),
            EffectTerm::Var(v) => match *v {},
        }
    }
}

// ── Identity system ──────────────────────────────────────────────────

acvus_utils::declare_local_id!(pub IdentityId);

/// Identity for Deque/Sequence types — prevents mixing collections from different sources.
///
/// - `Concrete(IdentityId)`: a fixed identity created by `[]` literals or `Fresh` instantiation.
/// - `Fresh(IdentityId)`: a signature-level marker meaning "allocate a new Concrete on instantiate".
///   Same `IdentityId` within one signature → same Concrete after instantiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Identity {
    Concrete(IdentityId),
    Fresh(IdentityId),
}

impl std::fmt::Display for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Identity::Concrete(id) => write!(f, "Identity({id:?})"),
            Identity::Fresh(id) => write!(f, "FreshIdentity({id:?})"),
        }
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

    /// Extract the effect carried by this type, if any.
    pub fn carried_effect(&self) -> Option<&Effect> {
        match self {
            Ty::Handle(_, effect) => Some(effect),
            Ty::Fn { effect, .. } => Some(effect),
            _ => None,
        }
    }

    /// Extract the scheduling hint, if this is a Fn type with a hint.
    pub fn hint(&self) -> Option<Hint> {
        match self {
            Ty::Fn { hint, .. } => *hint,
            _ => None,
        }
    }

    /// Extract the element type from a collection type.
    pub fn elem_of(&self) -> Option<&Ty> {
        match self {
            Ty::List(elem) | Ty::Deque(elem, _) => Some(elem),
            _ => None,
        }
    }

    /// Returns the purity tier of this type (shallow — does not recurse into containers).
    pub fn materiality(&self) -> Materiality {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => {
                Materiality::Concrete
            }
            Ty::List(_)
            | Ty::Deque(..)
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
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => true,
            Ty::List(inner) => inner.is_pureable(),
            Ty::Handle(inner, effect) => inner.is_pureable() && effect.is_pure(),
            Ty::Deque(inner, _) => inner.is_pureable(),
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
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => true,
            Ty::List(inner) | Ty::Deque(inner, _) => inner.is_materializable(),
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

    /// Convenience: `List<Byte>` (byte array type).
    pub fn bytes() -> Ty {
        Ty::List(Box::new(Ty::Byte))
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
            Ty::Range => write!(f, "Range"),
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
                hint: _,
            } => {
                let bang = if effect.is_effectful() { "!" } else { "" };
                write!(f, "Fn{bang}(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.ty.display(self.interner))?;
                }
                write!(f, ") -> {}", ret.display(self.interner))
            }
            Ty::List(inner) => write!(f, "List<{}>", inner.display(self.interner)),
            Ty::Handle(inner, effect) => {
                let bang = if effect.is_effectful() { "!" } else { "" };
                write!(f, "Handle{bang}<{}>", inner.display(self.interner))
            }
            Ty::Deque(inner, identity) => {
                write!(
                    f,
                    "Deque<{}, {}>",
                    inner.display(self.interner),
                    identity.display(self.interner)
                )
            }
            Ty::Identity(id) => write!(f, "{id}"),
            Ty::Option(inner) => write!(f, "Option<{}>", inner.display(self.interner)),
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                ..
            } => {
                let name = self.interner.resolve(id.name);
                write!(f, "{name}")?;
                if !type_args.is_empty() || !effect_args.is_empty() {
                    write!(f, "<")?;
                    for (i, arg) in type_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg.display(self.interner))?;
                    }
                    for (i, arg) in effect_args.iter().enumerate() {
                        if i > 0 || !type_args.is_empty() {
                            write!(f, ", ")?;
                        }
                        write!(f, "{arg:?}")?;
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
}

/// Post-inference phase — all types fully resolved.
/// `TyVar = Infallible` makes `TyTerm::Var` uninhabitable at type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Concrete;

impl Phase for Concrete {
    type TyVar = Infallible;
    type EffectVar = Infallible;
}

/// Polymorphic declaration phase — type templates stored in the graph.
/// `TyVar = u32` is a positional placeholder, not tied to any Solver instance.
/// Instantiated to `Infer` per call site during type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Poly;

impl Phase for Poly {
    type TyVar = u32;
    type EffectVar = u32;
}

/// During-inference phase — types may contain unresolved variables.
/// `TyVar = TypeBoundId` is scoped to a specific `Solver` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Infer;

impl Phase for Infer {
    type TyVar = TypeBoundId;
    type EffectVar = EffectBoundId;
}

/// Polymorphic type — template with positional placeholders.
pub type PolyTy = TyTerm<Poly>;
/// Polymorphic effect — template with positional placeholders.
pub type PolyEffect = EffectTerm<Poly>;
/// Polymorphic function parameter.
pub type PolyParam = ParamTerm<Poly>;

// ── Solver types ────────────────────────────────────────────────────

/// Index into `Solver::ty_bounds`. Identifies a type inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeBoundId(pub u32);

/// Index into `Solver::effect_bounds`. Identifies an effect inference variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectBoundId(pub u32);

// Re-export solver types — these were historically in ty.rs.
pub use crate::solver::{Capability, TypeBound, EffectBound, Solver, SolverSnapshot, FreezeError};

/// Type alias — always concrete, no inference variables.
pub type InferTy = TyTerm<Infer>;

/// Effect during inference — may contain unresolved variables.
pub type InferEffect = EffectTerm<Infer>;

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
    Range,
    Byte,
    // Containers
    List(Box<TyTerm<V>>),
    Object(FxHashMap<Astr, TyTerm<V>>),
    Tuple(Vec<TyTerm<V>>),
    Option(Box<TyTerm<V>>),
    Deque(Box<TyTerm<V>>, Box<TyTerm<V>>),
    // Functions
    Fn {
        params: Vec<ParamTerm<V>>,
        ret: Box<TyTerm<V>>,
        captures: Vec<TyTerm<V>>,
        effect: EffectTerm<V>,
        hint: Option<Hint>,
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
    Handle(Box<TyTerm<V>>, EffectTerm<V>),
    Identity(Identity),
    Ref(Box<TyTerm<V>>, bool),
    // Special
    Error(ErrorToken),
    /// Inference variable — only inhabitable when `V = Infer`.
    /// For `V = Concrete`, this is `Var(Infallible)` which cannot be constructed.
    Var(V::TyVar),
}

/// Effect term parameterized over inference phase.
///
/// When `V = Concrete`: always `Resolved(EffectSet)`.
/// When `V = Infer`: may be `Var(EffectBoundId)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectTerm<V: Phase> {
    Resolved(EffectSet),
    Var(V::EffectVar),
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
        on_effect_var: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
        on_identity: &mut impl FnMut(Identity) -> TyTerm<W>,
    ) -> TyTerm<W> {
        match self {
            TyTerm::Int => TyTerm::Int,
            TyTerm::Float => TyTerm::Float,
            TyTerm::String => TyTerm::String,
            TyTerm::Bool => TyTerm::Bool,
            TyTerm::Unit => TyTerm::Unit,
            TyTerm::Range => TyTerm::Range,
            TyTerm::Byte => TyTerm::Byte,
            TyTerm::List(inner) => TyTerm::List(Box::new(inner.map(on_var, on_effect_var, on_identity))),
            TyTerm::Object(fields) => TyTerm::Object(
                fields.iter().map(|(k, v)| (*k, v.map(on_var, on_effect_var, on_identity))).collect(),
            ),
            TyTerm::Tuple(elems) => TyTerm::Tuple(
                elems.iter().map(|e| e.map(on_var, on_effect_var, on_identity)).collect(),
            ),
            TyTerm::Option(inner) => TyTerm::Option(Box::new(inner.map(on_var, on_effect_var, on_identity))),
            TyTerm::Deque(inner, identity) => TyTerm::Deque(
                Box::new(inner.map(on_var, on_effect_var, on_identity)),
                Box::new(identity.map(on_var, on_effect_var, on_identity)),
            ),
            TyTerm::Fn { params, ret, captures, effect, hint } => TyTerm::Fn {
                params: params.iter().map(|p| ParamTerm::new(p.name, p.ty.map(on_var, on_effect_var, on_identity))).collect(),
                ret: Box::new(ret.map(on_var, on_effect_var, on_identity)),
                captures: captures.iter().map(|c| c.map(on_var, on_effect_var, on_identity)).collect(),
                effect: effect.map_effect(on_effect_var),
                hint: *hint,
            },
            TyTerm::UserDefined { id, type_args, effect_args } => TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| t.map(on_var, on_effect_var, on_identity)).collect(),
                effect_args: effect_args.iter().map(|e| e.map_effect(on_effect_var)).collect(),
            },
            TyTerm::Enum { name, variants } => TyTerm::Enum {
                name: *name,
                variants: variants.iter().map(|(tag, payload)| {
                    (*tag, payload.as_ref().map(|ty| Box::new(ty.map(on_var, on_effect_var, on_identity))))
                }).collect(),
            },
            TyTerm::Handle(inner, effect) => TyTerm::Handle(
                Box::new(inner.map(on_var, on_effect_var, on_identity)),
                effect.map_effect(on_effect_var),
            ),
            TyTerm::Identity(id) => on_identity(*id),
            TyTerm::Ref(inner, volatile) => TyTerm::Ref(
                Box::new(inner.map(on_var, on_effect_var, on_identity)),
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
        on_effect_var: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
        on_identity: &mut impl FnMut(Identity) -> Result<TyTerm<W>, E>,
    ) -> Result<TyTerm<W>, E> {
        match self {
            TyTerm::Int => Ok(TyTerm::Int),
            TyTerm::Float => Ok(TyTerm::Float),
            TyTerm::String => Ok(TyTerm::String),
            TyTerm::Bool => Ok(TyTerm::Bool),
            TyTerm::Unit => Ok(TyTerm::Unit),
            TyTerm::Range => Ok(TyTerm::Range),
            TyTerm::Byte => Ok(TyTerm::Byte),
            TyTerm::List(inner) => Ok(TyTerm::List(Box::new(inner.try_map(on_var, on_effect_var, on_identity)?))),
            TyTerm::Object(fields) => {
                let mapped: Result<FxHashMap<_, _>, E> = fields.iter()
                    .map(|(k, v)| v.try_map(on_var, on_effect_var, on_identity).map(|mv| (*k, mv)))
                    .collect();
                Ok(TyTerm::Object(mapped?))
            }
            TyTerm::Tuple(elems) => Ok(TyTerm::Tuple(
                elems.iter().map(|e| e.try_map(on_var, on_effect_var, on_identity)).collect::<Result<_, _>>()?,
            )),
            TyTerm::Option(inner) => Ok(TyTerm::Option(Box::new(inner.try_map(on_var, on_effect_var, on_identity)?))),
            TyTerm::Deque(inner, identity) => Ok(TyTerm::Deque(
                Box::new(inner.try_map(on_var, on_effect_var, on_identity)?),
                Box::new(identity.try_map(on_var, on_effect_var, on_identity)?),
            )),
            TyTerm::Fn { params, ret, captures, effect, hint } => Ok(TyTerm::Fn {
                params: params.iter()
                    .map(|p| p.ty.try_map(on_var, on_effect_var, on_identity).map(|ty| ParamTerm::new(p.name, ty)))
                    .collect::<Result<_, _>>()?,
                ret: Box::new(ret.try_map(on_var, on_effect_var, on_identity)?),
                captures: captures.iter().map(|c| c.try_map(on_var, on_effect_var, on_identity)).collect::<Result<_, _>>()?,
                effect: effect.try_map_effect(on_effect_var)?,
                hint: *hint,
            }),
            TyTerm::UserDefined { id, type_args, effect_args } => Ok(TyTerm::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| t.try_map(on_var, on_effect_var, on_identity)).collect::<Result<_, _>>()?,
                effect_args: effect_args.iter().map(|e| e.try_map_effect(on_effect_var)).collect::<Result<_, _>>()?,
            }),
            TyTerm::Enum { name, variants } => {
                let mapped: Result<FxHashMap<_, _>, E> = variants.iter()
                    .map(|(tag, payload)| {
                        let mp = match payload {
                            Some(ty) => Some(Box::new(ty.try_map(on_var, on_effect_var, on_identity)?)),
                            None => None,
                        };
                        Ok((*tag, mp))
                    })
                    .collect();
                Ok(TyTerm::Enum { name: *name, variants: mapped? })
            }
            TyTerm::Handle(inner, effect) => Ok(TyTerm::Handle(
                Box::new(inner.try_map(on_var, on_effect_var, on_identity)?),
                effect.try_map_effect(on_effect_var)?,
            )),
            TyTerm::Identity(id) => on_identity(*id),
            TyTerm::Ref(inner, volatile) => Ok(TyTerm::Ref(
                Box::new(inner.try_map(on_var, on_effect_var, on_identity)?),
                *volatile,
            )),
            TyTerm::Error(token) => Ok(TyTerm::Error(*token)),
            TyTerm::Var(v) => on_var(*v),
        }
    }
}

impl<V: Phase> EffectTerm<V> {
    /// Map this effect term from phase `V` to phase `W`.
    pub fn map_effect<W: Phase>(
        &self,
        on_var: &mut impl FnMut(V::EffectVar) -> EffectTerm<W>,
    ) -> EffectTerm<W> {
        match self {
            EffectTerm::Resolved(set) => EffectTerm::Resolved(set.clone()),
            EffectTerm::Var(v) => on_var(*v),
        }
    }

    /// Fallible version of `map_effect`.
    pub fn try_map_effect<W: Phase, E>(
        &self,
        on_var: &mut impl FnMut(V::EffectVar) -> Result<EffectTerm<W>, E>,
    ) -> Result<EffectTerm<W>, E> {
        match self {
            EffectTerm::Resolved(set) => Ok(EffectTerm::Resolved(set.clone())),
            EffectTerm::Var(v) => on_var(*v),
        }
    }
}

// ── Lift: Concrete → any Phase ──────────────────────────────────────

/// Lift a concrete `Ty` into any phase (mechanical, zero information change).
/// Infallible because `Concrete` has `TyVar = Infallible` (uninhabitable).
pub fn lift_ty<W: Phase>(ty: &Ty) -> TyTerm<W> {
    ty.map(
        &mut |v: Infallible| match v {},
        &mut |v: Infallible| match v {},
        &mut |id| TyTerm::Identity(id),
    )
}

/// Lift a concrete `Effect` into any phase.
pub fn lift_effect<W: Phase>(effect: &Effect) -> EffectTerm<W> {
    effect.map_effect(&mut |v: Infallible| match v {})
}

/// Alias: lift `Ty` to `PolyTy`. Kept for call-site compatibility.
pub fn lift_to_poly(ty: &Ty) -> PolyTy { lift_ty(ty) }

/// Alias: lift `Effect` to `PolyEffect`. Kept for call-site compatibility.
pub fn lift_effect_to_poly(effect: &Effect) -> PolyEffect { lift_effect(effect) }

/// Try to convert a `PolyTy` to a concrete `Ty`.
/// Returns `None` if the poly type contains any Var placeholders.
pub fn try_freeze_poly(ty: &PolyTy) -> Option<Ty> {
    ty.try_map(
        &mut |_: u32| Err(()),
        &mut |_: u32| Err(()),
        &mut |id| Ok(TyTerm::Identity(id)),
    ).ok()
}

/// Try to convert a `PolyEffect` to a concrete `Effect`.
/// Returns `None` if the poly effect contains any Var placeholders.
pub fn try_freeze_poly_effect(effect: &PolyEffect) -> Option<Effect> {
    effect.try_map_effect(&mut |_: u32| Err(())).ok()
}

// ── PolyBuilder ─────────────────────────────────────────────────────

/// Builder for polymorphic type templates. No Solver dependency.
/// Creates positional placeholders (Var(0), Var(1), ...) for type and effect variables.
pub struct PolyBuilder {
    next_ty: u32,
    next_effect: u32,
}

impl PolyBuilder {
    pub fn new() -> Self {
        Self { next_ty: 0, next_effect: 0 }
    }

    /// Create a fresh type placeholder.
    pub fn fresh_ty_var(&mut self) -> PolyTy {
        let id = self.next_ty;
        self.next_ty += 1;
        TyTerm::Var(id)
    }

    /// Create a fresh effect placeholder.
    pub fn fresh_effect_var(&mut self) -> PolyEffect {
        let id = self.next_effect;
        self.next_effect += 1;
        EffectTerm::Var(id)
    }

    /// Allocate an identity (same as Solver — Identity is phase-independent for now).
    pub fn alloc_identity(&mut self, factory: &mut acvus_utils::LocalFactory<IdentityId>, fresh: bool) -> PolyTy {
        let id = factory.next();
        if fresh {
            TyTerm::Identity(Identity::Fresh(id))
        } else {
            TyTerm::Identity(Identity::Concrete(id))
        }
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

    /// Test helper: create a non-pure resolved effect with a dummy write target.
    /// Used in place of the removed Effect::self_modifying().
    fn effectful() -> InferEffect {
        let i = Interner::new();
        EffectTerm::Resolved(EffectSet {
            reads: std::collections::BTreeSet::new(),
            writes: std::collections::BTreeSet::from([EffectTarget::Token(
                QualifiedRef::root(i.intern("__test")),
            )]),
        })
    }

    /// Test helper: create a concrete (non-infer) effectful effect.
    fn effectful_concrete() -> Effect {
        let i = Interner::new();
        Effect::Resolved(EffectSet {
            reads: std::collections::BTreeSet::new(),
            writes: std::collections::BTreeSet::from([EffectTarget::Token(
                QualifiedRef::root(i.intern("__test")),
            )]),
        })
    }

    /// Test helper: pure InferEffect.
    fn infer_pure() -> InferEffect {
        EffectTerm::Resolved(EffectSet::default())
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

    /// Test helper: create a `TyTerm::UserDefined` with a fresh id and no type/effect args.
    fn test_user_defined() -> Ty {
        TyTerm::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
        }
    }

    /// Test helper: create an `InferTy::UserDefined` with a fresh id and no type/effect args.
    fn test_user_defined_infer() -> InferTy {
        TyTerm::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
        }
    }

    /// Test helper: wrap an `InferTy` into a `ParamTerm<Infer>` with a dummy name.
    /// Uses a thread-local interner so all dummy names share the same interner id.
    fn tp(ty: InferTy) -> ParamTerm<Infer> {
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        INTERNER.with(|i| ParamTerm {
            name: i.intern(""),
            ty,
        })
    }

    /// Test helper: wrap a concrete `Ty` into a `Param` with a dummy name.
    fn tp_concrete(ty: Ty) -> Param {
        thread_local! {
            static INTERNER: Interner = Interner::new();
        }
        INTERNER.with(|i| Param {
            name: i.intern(""),
            ty,
        })
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
        assert!(s.unify_ty(&TyTerm::Range, &TyTerm::Range, Invariant, &registry).is_ok());
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
    fn unify_deque_of_var() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let t = s.fresh_ty_var();
        let deque_t = TyTerm::Deque(Box::new(t.clone()), Box::new(o.clone()));
        let deque_int = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o.clone()));
        assert!(s.unify_ty(&deque_t, &deque_int, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::Int);
        assert_eq!(
            s.resolve_ty(&deque_t),
            TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))
        );
    }

    #[test]
    fn unify_fn_types() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let t = s.fresh_ty_var();
        let u = s.fresh_ty_var();
        let fn_tu = TyTerm::Fn {
            params: vec![tp(t.clone())],
            ret: Box::new(u.clone()),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_int_bool = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::Bool),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_tu, &fn_int_bool, Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::Int);
        assert_eq!(s.resolve_ty(&u), TyTerm::Bool);
    }

    #[test]
    fn unify_fn_arity_mismatch() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let fn1 = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::Int),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn2 = TyTerm::Fn {
            params: vec![tp(TyTerm::Int), tp(TyTerm::Int)],
            ret: Box::new(TyTerm::Int),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn1, &fn2, Invariant, &registry).is_err());
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
    fn occurs_check() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let t = s.fresh_ty_var();
        let deque_t = TyTerm::Deque(Box::new(t.clone()), Box::new(o));
        // T = Deque<T, O> should fail
        assert!(s.unify_ty(&t, &deque_t, Invariant, &registry).is_err());
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

    // -- Deque type tests --

    #[test]
    fn unify_deque_same_identity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let t = s.fresh_ty_var();
        let d1 = TyTerm::Deque(Box::new(t.clone()), Box::new(o.clone()));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o.clone()));
        assert!(s.unify_ty(&d1, &d2, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&t), TyTerm::Int);
        assert_eq!(s.resolve_ty(&d1), TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)));
    }

    #[test]
    fn unify_deque_different_concrete_identity_fails() {
        // Invariant: different concrete identities → error
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        assert_ne!(o1, o2);
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2));
        assert!(
            s.unify_ty(&d1, &d2, Invariant, &registry).is_err(),
            "different concrete identities must not unify in Invariant"
        );
    }

    #[test]
    fn unify_deque_identity_param_binds_to_concrete() {
        // Param should bind to Identity::Concrete during unification
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let concrete = s.alloc_identity(false);
        let var = s.fresh_ty_var(); // identity variable
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(concrete.clone()));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(var.clone()));
        assert!(
            s.unify_ty(&d1, &d2, Invariant, &registry).is_ok(),
            "identity Param should bind to Concrete"
        );
        assert_eq!(s.resolve_ty(&var), concrete);
    }

    #[test]
    fn unify_deque_identity_param_preserves_identity() {
        // Two Deques through same identity Param should resolve to same concrete identity
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let concrete = s.alloc_identity(false);
        let var = s.fresh_ty_var();
        let d_concrete = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(concrete));
        let d_var = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(var.clone()));
        assert!(s.unify_ty(&d_concrete, &d_var, Invariant, &registry).is_ok());
        // Now a second concrete identity should NOT match the same var
        let concrete2 = s.alloc_identity(false);
        let d_concrete2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(concrete2));
        let d_var2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(var));
        assert!(
            s.unify_ty(&d_concrete2, &d_var2, Invariant, &registry).is_err(),
            "var already bound to different concrete"
        );
    }

    #[test]
    fn unify_deque_inner_type_mismatch_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o.clone()));
        let d2 = TyTerm::Deque(Box::new(TyTerm::String), Box::new(o));
        assert!(
            s.unify_ty(&d1, &d2, Invariant, &registry).is_err(),
            "inner type mismatch with same identity must fail"
        );
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
    fn resolve_deque_propagates_inner() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let t = s.fresh_ty_var();
        assert!(s.unify_ty(&t, &TyTerm::String, Invariant, &registry).is_ok());
        let deque = TyTerm::Deque(Box::new(t.clone()), Box::new(o.clone()));
        assert_eq!(
            s.resolve_ty(&deque),
            TyTerm::Deque(Box::new(TyTerm::String), Box::new(o))
        );
    }

    #[test]
    fn occurs_in_deque() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let t = s.fresh_ty_var();
        let deque_t = TyTerm::Deque(Box::new(t.clone()), Box::new(o));
        // T = Deque<T, O> should fail (occurs check)
        assert!(s.unify_ty(&t, &deque_t, Invariant, &registry).is_err());
    }

    #[test]
    fn snapshot_rollback_preserves_identity_counter() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let _id1 = s.alloc_identity(false);
        let snap = s.snapshot();
        let id2 = s.alloc_identity(false);
        s.rollback(snap);
        let id_after = s.alloc_identity(false);
        // After rollback, next_identity should be restored, so id_after == id2.
        assert_eq!(id_after, id2, "rollback should restore identity counter");
    }

    #[test]
    fn unify_deque_coerces_to_list() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(
            s.unify_ty(&d, &l, Covariant, &registry).is_ok(),
            "Deque should coerce to List"
        );
    }

    #[test]
    fn unify_list_does_not_coerce_to_deque() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let l = TyTerm::List(Box::new(TyTerm::Int));
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        assert!(
            s.unify_ty(&l, &d, Covariant, &registry).is_err(),
            "List must not coerce to Deque"
        );
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn deque_identity_mismatch_covariant_demotes_to_list() {
        // Covariant: Deque+Deque identity mismatch → List demotion
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2));
        // Bind v to d1, then unify v with d2 in Covariant → should demote to List
        assert!(s.unify_ty(&v, &d1, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &d2, Covariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        assert_eq!(
            resolved,
            TyTerm::List(Box::new(TyTerm::Int)),
            "should demote to List<Int>"
        );
    }

    #[test]
    fn deque_identity_mismatch_invariant_fails() {
        // Invariant: Deque+Deque identity mismatch → error
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2));
        assert!(s.unify_ty(&d1, &d2, Invariant, &registry).is_err());
    }

    #[test]
    fn deque_coerces_to_list_covariant() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&d, &l, Covariant, &registry).is_ok());
    }

    #[test]
    fn list_does_not_coerce_to_deque_covariant() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let l = TyTerm::List(Box::new(TyTerm::Int));
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        assert!(s.unify_ty(&l, &d, Covariant, &registry).is_err());
    }

    #[test]
    fn contravariant_list_deque_ok() {
        // Contravariant: (List, Deque) → reversed: Deque ≤ List → OK
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let l = TyTerm::List(Box::new(TyTerm::Int));
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        assert!(s.unify_ty(&l, &d, Contravariant, &registry).is_ok());
    }

    #[test]
    fn contravariant_deque_list_fails() {
        // Contravariant: (Deque, List) → reversed: List ≤ Deque → invalid
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&d, &l, Contravariant, &registry).is_err());
    }

    #[test]
    fn invariant_deque_list_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.fresh_ty_var();
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&d, &l, Invariant, &registry).is_err());
        assert!(s.unify_ty(&l, &d, Invariant, &registry).is_err());
    }

    #[test]
    fn fn_param_contravariant_ret_covariant() {
        // Fn(List<Int>) -> Deque<Int> ≤ Fn(Deque<Int>) -> List<Int> in Covariant
        // params flip: Deque ≤ List OK (contravariant)
        // ret keeps: Deque ≤ List OK (covariant)
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.fresh_ty_var();
        let o2 = s.fresh_ty_var();
        let fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)))],
            ret: Box::new(TyTerm::List(Box::new(TyTerm::Int))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_ok());
    }

    #[test]
    fn list_literal_mixed_deque_identities() {
        // Simulates: multiple Deque elements with different identities → List demotion
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let elem_var = s.fresh_ty_var();
        let d1 = TyTerm::Deque(Box::new(TyTerm::String), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::String), Box::new(o2));
        // First element sets the type
        assert!(s.unify_ty(&elem_var, &d1, Covariant, &registry).is_ok());
        // Second element with different identity → demotion
        assert!(s.unify_ty(&elem_var, &d2, Covariant, &registry).is_ok());
        let resolved = s.resolve_ty(&elem_var);
        assert_eq!(resolved, TyTerm::List(Box::new(TyTerm::String)));
    }

    #[test]
    fn polarity_flip() {
        assert_eq!(Covariant.flip(), Contravariant);
        assert_eq!(Contravariant.flip(), Covariant);
        assert_eq!(Invariant.flip(), Invariant);
    }

    // -- Variance unsoundness edge case tests --

    #[test]
    fn demotion_then_third_deque_still_works() {
        // [Deque(o1), Deque(o2), Deque(o3)] — after o1+o2 demotes to List,
        // the third Deque(o3) should still unify via Deque≤List coercion.
        // arg order: (new_elem, join_accum) → new ≤ existing.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let o3 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2));
        let d3 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o3));
        assert!(s.unify_ty(&d1, &v, Covariant, &registry).is_ok());
        assert!(
            s.unify_ty(&d2, &v, Covariant, &registry).is_ok(),
            "second deque should trigger demotion"
        );
        // v is now List<Int>. Third deque: Deque≤List in Covariant should succeed.
        assert!(
            s.unify_ty(&d3, &v, Covariant, &registry).is_ok(),
            "third deque should coerce to List via Deque≤List"
        );
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
    }

    #[test]
    fn demotion_then_list_unifies() {
        // After demotion to List, unifying with another List should succeed.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)), Covariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), Covariant, &registry)
                .is_ok()
        );
        assert!(s.unify_ty(&v, &TyTerm::List(Box::new(TyTerm::Int)), Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
    }

    #[test]
    fn demotion_then_deque_same_inner_type_via_var() {
        // After demotion, the Var-resolved List should accept further Deque coercion
        // even when inner type is a Var that later resolves.
        // arg order: (new_elem, join_accum) → new ≤ existing.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let o3 = s.alloc_identity(false);
        let inner_var = s.fresh_ty_var();
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &TyTerm::Deque(Box::new(inner_var.clone()), Box::new(o1)),
                &v,
                Covariant, &registry)
            .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), &v, Covariant, &registry)
                .is_ok()
        );
        // inner_var should now be Int, v should be List<Int>
        assert_eq!(s.resolve_ty(&inner_var), TyTerm::Int);
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
        // Third deque with same inner type
        assert!(
            s.unify_ty(&TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o3)), &v, Covariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn concrete_deque_deque_covariant_no_var_no_rebind() {
        // Two concrete Deques (no Var backing) with mismatched identities.
        // No Param to rebind → LUB cannot be applied → Err.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2));
        assert!(
            s.unify_ty(&d1, &d2, Covariant, &registry).is_err(),
            "concrete Deque identity mismatch with no Param should fail"
        );
    }

    #[test]
    fn concrete_deque_deque_inner_mismatch_plus_identity_mismatch() {
        // Both inner type AND identity mismatch — inner unify should fail first.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let d2 = TyTerm::Deque(Box::new(TyTerm::String), Box::new(o2));
        assert!(
            s.unify_ty(&d1, &d2, Covariant, &registry).is_err(),
            "inner type mismatch must fail regardless of demotion"
        );
    }

    #[test]
    fn demotion_inner_type_still_var() {
        // Demotion when inner type is an unresolved Var — should resolve to List<Var>.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let inner = s.fresh_ty_var();
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(inner.clone()), Box::new(o1)),
                Covariant, &registry)
            .is_ok()
        );
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(inner.clone()), Box::new(o2)),
                Covariant, &registry)
            .is_ok()
        );
        // v should be List<inner_var>, inner still unresolved
        let resolved = s.resolve_ty(&v);
        assert!(
            matches!(resolved, TyTerm::List(_)),
            "should be List, got {resolved:?}"
        );
        // Now bind inner to String
        assert!(s.unify_ty(&inner, &TyTerm::String, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::String)));
    }

    #[test]
    fn contravariant_demotion() {
        // Contravariant: Deque+Deque identity mismatch also demotes (pol != Invariant).
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)),
                Contravariant, &registry)
            .is_ok()
        );
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)),
                Contravariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
    }

    #[test]
    fn object_field_deque_coercion_covariant() {
        // {tags: Deque<String, o1>} vs {tags: List<String>} in Covariant.
        // Object field polarity is passed through → Deque≤List OK.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let obj_deque = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::Deque(Box::new(TyTerm::String), Box::new(o)),
        )]));
        let obj_list = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::List(Box::new(TyTerm::String)),
        )]));
        assert!(s.unify_ty(&obj_deque, &obj_list, Covariant, &registry).is_ok());
    }

    #[test]
    fn object_field_deque_coercion_invariant_fails() {
        // Same as above but Invariant — must fail.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let obj_deque = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::Deque(Box::new(TyTerm::String), Box::new(o)),
        )]));
        let obj_list = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::List(Box::new(TyTerm::String)),
        )]));
        assert!(s.unify_ty(&obj_deque, &obj_list, Invariant, &registry).is_err());
    }

    #[test]
    fn object_field_deque_identity_mismatch_demotion() {
        // {tags: Deque<S, o1>} vs {tags: Deque<S, o2>} in Covariant.
        // Inner Deque identity mismatch → demoted to List within the field.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::Deque(Box::new(TyTerm::String), Box::new(o1)),
        )]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            TyTerm::Deque(Box::new(TyTerm::String), Box::new(o2)),
        )]));
        assert!(s.unify_ty(&v, &obj1, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj2, Covariant, &registry).is_ok());
    }

    #[test]
    fn option_deque_to_list_covariant_fails() {
        // Option<Deque<Int>> vs Option<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside Option is a type error.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let opt_deque = TyTerm::Option(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))));
        let opt_list = TyTerm::Option(Box::new(TyTerm::List(Box::new(TyTerm::Int))));
        assert!(s.unify_ty(&opt_deque, &opt_list, Covariant, &registry).is_err());
    }

    #[test]
    fn option_deque_to_list_invariant_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let opt_deque = TyTerm::Option(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))));
        let opt_list = TyTerm::Option(Box::new(TyTerm::List(Box::new(TyTerm::Int))));
        assert!(s.unify_ty(&opt_deque, &opt_list, Invariant, &registry).is_err());
    }

    #[test]
    fn tuple_deque_coercion_covariant() {
        // (Deque<Int>, String) vs (List<Int>, String) in Covariant.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let t1 = TyTerm::Tuple(vec![TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), TyTerm::String]);
        let t2 = TyTerm::Tuple(vec![TyTerm::List(Box::new(TyTerm::Int)), TyTerm::String]);
        assert!(s.unify_ty(&t1, &t2, Covariant, &registry).is_ok());
    }

    #[test]
    fn tuple_deque_coercion_invariant_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let t1 = TyTerm::Tuple(vec![TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), TyTerm::String]);
        let t2 = TyTerm::Tuple(vec![TyTerm::List(Box::new(TyTerm::Int)), TyTerm::String]);
        assert!(s.unify_ty(&t1, &t2, Invariant, &registry).is_err());
    }

    #[test]
    fn double_flip_restores_covariant() {
        // Fn(Fn(Deque) -> Unit) -> Unit  vs  Fn(Fn(List) -> Unit) -> Unit
        // Outer Covariant → param flips to Contravariant → inner param flips back to Covariant.
        // So inner param: Deque vs List in Covariant → Deque≤List OK.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let inner_fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let inner_fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let outer_a = TyTerm::Fn {
            params: vec![tp(inner_fn_a)],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let outer_b = TyTerm::Fn {
            params: vec![tp(inner_fn_b)],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&outer_a, &outer_b, Covariant, &registry).is_ok());
    }

    #[test]
    fn double_flip_wrong_direction_fails() {
        // Fn(Fn(List) -> Unit) -> Unit  vs  Fn(Fn(Deque) -> Unit) -> Unit
        // Double flip = Covariant → inner param: List vs Deque in Covariant → List≤Deque → fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let inner_fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let inner_fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let outer_a = TyTerm::Fn {
            params: vec![tp(inner_fn_a)],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let outer_b = TyTerm::Fn {
            params: vec![tp(inner_fn_b)],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&outer_a, &outer_b, Covariant, &registry).is_err());
    }

    #[test]
    fn fn_ret_list_to_deque_covariant_fails() {
        // Fn() -> List<Int>  vs  Fn() -> Deque<Int, O>  in Covariant.
        // ret keeps polarity → List≤Deque invalid.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![],
            ret: Box::new(TyTerm::List(Box::new(TyTerm::Int))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![],
            ret: Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_err());
    }

    #[test]
    fn fn_param_deque_to_list_covariant_fails() {
        // Fn(Deque<Int>) -> Unit  vs  Fn(List<Int>) -> Unit  in Covariant.
        // param flips → Contravariant: Deque vs List → (Deque, List) in Contra → reversed: List≤Deque → fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_err());
    }

    #[test]
    fn fn_param_list_to_deque_covariant_ok() {
        // Fn(List<Int>) -> Unit  vs  Fn(Deque<Int>) -> Unit  in Covariant.
        // param flips → Contra: List vs Deque → (List, Deque) in Contra → reversed: Deque≤List → OK.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_ok());
    }

    #[test]
    fn snapshot_rollback_undoes_demotion() {
        // Demotion should be fully undone by rollback.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1.clone())),
                Covariant, &registry)
            .is_ok()
        );
        let snap = s.snapshot();
        let o2 = s.alloc_identity(false);
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), Covariant, &registry)
                .is_ok()
        );
        assert_eq!(
            s.resolve_ty(&v),
            TyTerm::List(Box::new(TyTerm::Int)),
            "demoted after second deque"
        );
        s.rollback(snap);
        // After rollback, v should be back to Deque<Int, o1>
        assert_eq!(
            s.resolve_ty(&v),
            TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)),
            "rollback should undo demotion"
        );
    }

    #[test]
    fn nested_list_of_deque_coercion_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside List is a type error.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let a = TyTerm::List(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))));
        let b = TyTerm::List(Box::new(TyTerm::List(Box::new(TyTerm::Int))));
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_err());
    }

    #[test]
    fn nested_list_of_deque_invariant_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Invariant.
        // Inner: Deque vs List in Invariant → fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let a = TyTerm::List(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o))));
        let b = TyTerm::List(Box::new(TyTerm::List(Box::new(TyTerm::Int))));
        assert!(s.unify_ty(&a, &b, Invariant, &registry).is_err());
    }

    #[test]
    fn enum_variant_deque_coercion_covariant() {
        // Enum with Deque payload vs same enum with List payload, Covariant.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = TyTerm::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))),
            )]),
        };
        let e2 = TyTerm::Enum {
            name,
            variants: FxHashMap::from_iter([(tag, Some(Box::new(TyTerm::List(Box::new(TyTerm::Int)))))]),
        };
        assert!(s.unify_ty(&e1, &e2, Covariant, &registry).is_ok());
    }

    #[test]
    fn enum_variant_deque_coercion_invariant_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = TyTerm::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)))),
            )]),
        };
        let e2 = TyTerm::Enum {
            name,
            variants: FxHashMap::from_iter([(tag, Some(Box::new(TyTerm::List(Box::new(TyTerm::Int)))))]),
        };
        assert!(s.unify_ty(&e1, &e2, Invariant, &registry).is_err());
    }

    // ================================================================
    // Var chain + coercion 상호작용
    // ================================================================

    #[test]
    fn var_chain_coercion_propagates() {
        // Var1 → Var2 → Deque(o1), then unify Var1 with List → Deque ≤ List via chain.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        assert!(s.unify_ty(&v1, &v2, Invariant, &registry).is_ok());
        assert!(
            s.unify_ty(&v2, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), Invariant, &registry)
                .is_ok()
        );
        // v1 → v2 → Deque(Int, o). Now v1 as Deque ≤ List.
        assert!(
            s.unify_ty(&v1, &TyTerm::List(Box::new(TyTerm::Int)), Covariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn var_chain_demotion_rebinds_leaf() {
        // Var1 → Var2 → Deque(o1). Unify Var1 with Deque(o2) covariant → demotion.
        // find_leaf_param should follow chain and rebind Var2 (the leaf).
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        assert!(s.unify_ty(&v1, &v2, Invariant, &registry).is_ok());
        assert!(
            s.unify_ty(&v2, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)), Invariant, &registry)
                .is_ok()
        );
        // Demotion via v1
        assert!(
            s.unify_ty(&TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), &v1, Covariant, &registry)
                .is_ok()
        );
        assert_eq!(s.resolve_ty(&v1), TyTerm::List(Box::new(TyTerm::Int)));
        assert_eq!(s.resolve_ty(&v2), TyTerm::List(Box::new(TyTerm::Int)));
    }

    #[test]
    fn two_vars_sharing_deque_demotion_affects_both() {
        // Chain v2 → v1 while both unbound, THEN bind v1 → Deque(o1).
        // Demote via v2 → find_leaf_param follows v2 → v1 → rebinds v1 to List.
        // Both Var1 and Var2 should resolve to List.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        // Must chain BEFORE binding to concrete — otherwise shallow_resolve
        // flattens the chain and v2 binds directly to Deque, not to v1.
        assert!(s.unify_ty(&v2, &v1, Invariant, &registry).is_ok());
        assert!(
            s.unify_ty(&v1, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)), Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), &v2, Covariant, &registry)
                .is_ok()
        );
        assert_eq!(s.resolve_ty(&v1), TyTerm::List(Box::new(TyTerm::Int)));
        assert_eq!(s.resolve_ty(&v2), TyTerm::List(Box::new(TyTerm::Int)));
    }

    // ================================================================
    // Occurs check + polarity
    // ================================================================

    #[test]
    fn occurs_check_through_list_covariant() {
        // Var = List<Var> should fail (occurs) regardless of polarity.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        let cyclic = TyTerm::List(Box::new(v.clone()));
        assert!(s.unify_ty(&v, &cyclic, Covariant, &registry).is_err());
    }

    #[test]
    fn occurs_check_through_deque_covariant() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        let cyclic = TyTerm::Deque(Box::new(v.clone()), Box::new(o));
        assert!(s.unify_ty(&v, &cyclic, Covariant, &registry).is_err());
    }

    #[test]
    fn occurs_check_through_fn_ret_covariant() {
        // Var = Fn() -> Var should fail (occurs) in any polarity.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        let cyclic = TyTerm::Fn {
            params: vec![],
            ret: Box::new(v.clone()),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&v, &cyclic, Covariant, &registry).is_err());
    }

    // ================================================================
    // Deep nesting coercion
    // ================================================================

    #[test]
    fn nested_deque_in_deque_coercion() {
        // Deque<Deque<Int, o1>, o2> vs Deque<List<Int>, o2> in Covariant.
        // Inner item type is invariant — Deque vs List inside Deque is a type error.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let a = TyTerm::Deque(
            Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),
            Box::new(o2.clone()),
        );
        let b = TyTerm::Deque(Box::new(TyTerm::List(Box::new(TyTerm::Int))), Box::new(o2));
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_err());
    }

    #[test]
    fn nested_deque_in_deque_invariant_inner_coercion_fails() {
        // Same structure but Invariant → inner Deque vs List fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let a = TyTerm::Deque(
            Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),
            Box::new(o2.clone()),
        );
        let b = TyTerm::Deque(Box::new(TyTerm::List(Box::new(TyTerm::Int))), Box::new(o2));
        assert!(s.unify_ty(&a, &b, Invariant, &registry).is_err());
    }

    #[test]
    fn deeply_nested_option_option_deque_covariant_fails() {
        // Option<Option<Deque<Int>>> vs Option<Option<List<Int>>> in Covariant.
        // Inner item type is invariant — nested coercion is a type error.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let a = TyTerm::Option(Box::new(TyTerm::Option(Box::new(TyTerm::Deque(
            Box::new(TyTerm::Int),
            Box::new(o),
        )))));
        let b = TyTerm::Option(Box::new(TyTerm::Option(Box::new(TyTerm::List(Box::new(TyTerm::Int))))));
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_err());
    }

    #[test]
    fn list_of_fn_with_coercion_in_param_and_ret_fails() {
        // List<Fn(List<Int>) -> Deque<Int>>  vs  List<Fn(Deque<Int>) -> List<Int>>
        // in Covariant.
        // Inner item type is invariant — Fn types with different param/ret don't match inside List.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![tp(TyTerm::List(Box::new(TyTerm::Int)))],
            ret: Box::new(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)))],
            ret: Box::new(TyTerm::List(Box::new(TyTerm::Int))),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let a = TyTerm::List(Box::new(fn_a));
        let b = TyTerm::List(Box::new(fn_b));
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_err());
    }

    // ================================================================
    // Object merge + coercion 동시 발생
    // ================================================================

    #[test]
    fn object_merge_plus_inner_demotion() {
        // Var → {a: Deque(o1)} then Var → {a: Deque(o2), b: Int}.
        // Merge adds field b, inner field a triggers demotion.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let i = Interner::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        let obj1 = TyTerm::Object(FxHashMap::from_iter([(
            i.intern("a"),
            TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)),
        )]));
        let obj2 = TyTerm::Object(FxHashMap::from_iter([
            (i.intern("a"), TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2))),
            (i.intern("b"), TyTerm::Int),
        ]));
        assert!(s.unify_ty(&v, &obj1, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &obj2, Covariant, &registry).is_ok());
        let resolved = s.resolve_ty(&v);
        match &resolved {
            TyTerm::Object(fields) => {
                assert_eq!(fields.len(), 2);
                assert!(fields.contains_key(&i.intern("b")));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    // ================================================================
    // Snapshot/rollback isolation
    // ================================================================

    #[test]
    fn snapshot_rollback_demotion_no_residue() {
        // Snapshot → demotion → rollback → same Var with different Deque (same identity).
        // Rollback must fully undo the demotion so the new unify works cleanly.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1.clone())),
                Invariant, &registry)
            .is_ok()
        );

        let snap = s.snapshot();
        assert!(
            s.unify_ty(&TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2)), &v, Covariant, &registry)
                .is_ok()
        );
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
        s.rollback(snap);

        // After rollback, v is still Deque(o1). Same-identity unify should work.
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1.clone())),
                Invariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&v), TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)));
    }

    // ================================================================
    // Polarity symmetry / duality 검증
    // ================================================================

    #[test]
    fn covariant_ab_equals_contravariant_ba() {
        // If unify(a, b, Cov) succeeds then unify(b, a, Contra) must also succeed.
        let mut s1 = Solver::new();
        let mut s2 = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false); // keep counter in sync
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s1.unify_ty(&d, &l, Covariant, &registry).is_ok());
        assert!(s2.unify_ty(&l, &d, Contravariant, &registry).is_ok());
    }

    #[test]
    fn covariant_ab_fail_equals_contravariant_ba_fail() {
        // If unify(a, b, Cov) fails then unify(b, a, Contra) must also fail.
        let mut s1 = Solver::new();
        let mut s2 = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false);
        let l = TyTerm::List(Box::new(TyTerm::Int));
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1));
        assert!(s1.unify_ty(&l, &d, Covariant, &registry).is_err()); // List ≤ Deque: no
        assert!(s2.unify_ty(&d, &l, Contravariant, &registry).is_err()); // reversed: List ≤ Deque: no
    }

    #[test]
    fn invariant_symmetric() {
        // Invariant: unify(a, b) and unify(b, a) must both fail/succeed equally.
        let mut s1 = Solver::new();
        let registry = TypeRegistry::new();
        let mut s2 = Solver::new();
        let registry = TypeRegistry::new();
        let o = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false);
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let l = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s1.unify_ty(&d, &l, Invariant, &registry).is_err());
        assert!(s2.unify_ty(&l, &d, Invariant, &registry).is_err());
    }

    #[test]
    fn invariant_same_types_both_directions() {
        // Same concrete type: Invariant must succeed regardless of order.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let l1 = TyTerm::List(Box::new(TyTerm::Int));
        let l2 = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&l1, &l2, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&l2, &l1, Invariant, &registry).is_ok());
    }

    // ================================================================
    // Multi-param Fn: per-param polarity
    // ================================================================

    #[test]
    fn fn_multi_param_one_fails_all_fails() {
        // Fn(List, Deque) -> Unit  vs  Fn(Deque, List) -> Unit  in Covariant.
        // param flip → Contra:
        //   param0: (List, Deque) in Contra → Deque≤List OK
        //   param1: (Deque, List) in Contra → List≤Deque FAIL
        // Whole Fn unify must fail.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![
                tp(TyTerm::List(Box::new(TyTerm::Int))),
                tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),
            ],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![
                tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2))),
                tp(TyTerm::List(Box::new(TyTerm::Int))),
            ],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_err());
    }

    #[test]
    fn fn_multi_param_all_ok() {
        // Fn(List, List) -> Unit  vs  Fn(Deque, Deque) -> Unit  in Covariant.
        // param flip → Contra: both (List, Deque) → Deque≤List OK.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = TyTerm::Fn {
            params: vec![
                tp(TyTerm::List(Box::new(TyTerm::Int))),
                tp(TyTerm::List(Box::new(TyTerm::Int))),
            ],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        let fn_b = TyTerm::Fn {
            params: vec![
                tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1))),
                tp(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o2))),
            ],
            ret: Box::new(TyTerm::Unit),

            effect: infer_pure(),
            captures: vec![],
            hint: None,
        };
        assert!(s.unify_ty(&fn_a, &fn_b, Covariant, &registry).is_ok());
    }

    // ================================================================
    // Unresolved Var containers + coercion
    // ================================================================

    #[test]
    fn deque_var_inner_coerces_to_list_var_inner() {
        // Deque<Var1, O> vs List<Var2> in Covariant → Deque≤List OK, Var1 binds to Var2.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        let d = TyTerm::Deque(Box::new(v1.clone()), Box::new(o));
        let l = TyTerm::List(Box::new(v2.clone()));
        assert!(s.unify_ty(&d, &l, Covariant, &registry).is_ok());
        // Bind v2 to String → v1 should follow.
        assert!(s.unify_ty(&v2, &TyTerm::String, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&v1), TyTerm::String);
    }

    // ================================================================
    // Bidirectional Var binding + coercion
    // ================================================================

    #[test]
    fn two_vars_coerce_deque_to_list() {
        // Var1 = Deque(Int, o1), Var2 = List(Int).
        // unify(Var1, Var2, Cov) → Deque≤List → OK.
        // After: Var1 still resolves to Deque (binding unchanged), Var2 still List.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        assert!(
            s.unify_ty(&v1, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&v2, &TyTerm::List(Box::new(TyTerm::Int)), Invariant, &registry)
                .is_ok()
        );
        assert!(s.unify_ty(&v1, &v2, Covariant, &registry).is_ok());
    }

    #[test]
    fn two_vars_coerce_list_to_deque_covariant_fails() {
        // Var1 = List(Int), Var2 = Deque(Int, o).
        // unify(Var1, Var2, Cov) → List≤Deque → fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_ty_var();
        let v2 = s.fresh_ty_var();
        assert!(
            s.unify_ty(&v1, &TyTerm::List(Box::new(TyTerm::Int)), Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(&v2, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), Invariant, &registry)
                .is_ok()
        );
        assert!(s.unify_ty(&v1, &v2, Covariant, &registry).is_err());
    }

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    #[test]
    fn five_deque_identities_join_to_list() {
        // [d1, d2, d3, d4, d5] each with distinct identity → all join to List.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        for _ in 0..5 {
            let o = s.alloc_identity(false);
            let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
            assert!(s.unify_ty(&d, &v, Covariant, &registry).is_ok());
        }
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
    }

    // ================================================================
    // Mixed concrete/param identities
    // ================================================================

    #[test]
    fn identity_param_binds_then_mismatch_demotes() {
        // Param identity Deque binds to concrete identity, then another concrete → mismatch → demotion.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let c1 = s.alloc_identity(false);
        let c2 = s.alloc_identity(false);
        let ov = s.fresh_ty_var(); // identity variable
        let v = s.fresh_ty_var();
        let d_var_identity = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(ov.clone()));
        let d_c1 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(c1.clone()));
        let d_c2 = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(c2));
        // Bind Param identity via d_var_identity = d_c1
        assert!(s.unify_ty(&v, &d_var_identity, Invariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &d_c1, Invariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&ov), c1);
        // Now d_c2 has different identity → demotion in Covariant
        assert!(s.unify_ty(&d_c2, &v, Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&v), TyTerm::List(Box::new(TyTerm::Int)));
    }

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    #[test]
    fn error_absorbs_any_polarity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        assert!(s.unify_ty(&lift_ty(&Ty::error()), &d, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&d, &lift_ty(&Ty::error()), Contravariant, &registry).is_ok());
        assert!(
            s.unify_ty(&lift_ty(&Ty::error()), &TyTerm::List(Box::new(TyTerm::Int)), Invariant, &registry)
                .is_ok()
        );
    }

    #[test]
    fn param_absorbs_any_polarity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let p1 = s.fresh_ty_var();
        let p2 = s.fresh_ty_var();
        let p3 = s.fresh_ty_var();
        assert!(s.unify_ty(&p1, &d, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&d, &p2, Contravariant, &registry).is_ok());
        assert!(
            s.unify_ty(&p3, &TyTerm::List(Box::new(TyTerm::Int)), Invariant, &registry)
                .is_ok()
        );
    }

    // ================================================================
    // Transitive coercion chains
    // ================================================================

    #[test]
    fn list_cannot_narrow_back_to_deque_covariant() {
        // Var = List(Int). List ≤ Deque is invalid.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(s.unify_ty(&v, &TyTerm::List(Box::new(TyTerm::Int)), Invariant, &registry).is_ok());
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)), Covariant, &registry)
                .is_err()
        );
    }

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    #[test]
    fn deque_to_list_inner_type_mismatch_fails() {
        // Deque<Int> ≤ List<String> → inner Int vs String fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        assert!(
            s.unify_ty(
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)),
                &TyTerm::List(Box::new(TyTerm::String)),
                Covariant, &registry)
            .is_err()
        );
    }

    #[test]
    fn demotion_inner_type_mismatch_fails() {
        // Deque<Int, o1> vs Deque<String, o2> in Covariant.
        // Identity mismatch triggers demotion path, but inner unify Int vs String fails first.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        assert!(
            s.unify_ty(
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o1)),
                &TyTerm::Deque(Box::new(TyTerm::String), Box::new(o2)),
                Covariant, &registry)
            .is_err()
        );
    }

    // ================================================================
    // Coercion does NOT propagate across unrelated type constructors
    // ================================================================

    #[test]
    fn deque_vs_option_fails_any_polarity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let d = TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o));
        let opt = TyTerm::Option(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&d, &opt, Covariant, &registry).is_err());
        assert!(s.unify_ty(&d, &opt, Contravariant, &registry).is_err());
        assert!(s.unify_ty(&d, &opt, Invariant, &registry).is_err());
    }

    #[test]
    fn list_vs_tuple_fails_any_polarity() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let l = TyTerm::List(Box::new(TyTerm::Int));
        let t = TyTerm::Tuple(vec![TyTerm::Int]);
        assert!(s.unify_ty(&l, &t, Covariant, &registry).is_err());
        assert!(s.unify_ty(&l, &t, Invariant, &registry).is_err());
    }

    // ================================================================
    // Triple flip (Fn<Fn<Fn<...>>>)
    // ================================================================

    #[test]
    fn triple_flip_reverses_back_to_contravariant() {
        // Fn(Fn(Fn(X) -> U) -> U) -> U in Covariant
        // outer param: flip → Contra
        // mid param: flip → Cov
        // inner param: flip → Contra
        // So innermost param is Contravariant.
        // (Deque, List) in Contra → reversed: List≤Deque → fails.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let mk = |inner_param: InferTy| -> InferTy {
            TyTerm::Fn {
                params: vec![tp(TyTerm::Fn {
                    params: vec![tp(TyTerm::Fn {
                        params: vec![tp(inner_param)],
                        ret: Box::new(TyTerm::Unit),

                        effect: infer_pure(),
                        captures: vec![],
                        hint: None,
                    })],
                    ret: Box::new(TyTerm::Unit),

                    effect: infer_pure(),
                    captures: vec![],
                    hint: None,
                })],
                ret: Box::new(TyTerm::Unit),

                effect: infer_pure(),
                captures: vec![],
                hint: None,
            }
        };
        let a = mk(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)));
        let b = mk(TyTerm::List(Box::new(TyTerm::Int)));
        // 3 flips from Covariant → Contra. (Deque, List) in Contra → fail.
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_err());
    }

    #[test]
    fn triple_flip_reversed_succeeds() {
        // Same structure but (List, Deque) at innermost.
        // 3 flips → Contra. (List, Deque) in Contra → reversed: Deque≤List → OK.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let mk = |inner_param: InferTy| -> InferTy {
            TyTerm::Fn {
                params: vec![tp(TyTerm::Fn {
                    params: vec![tp(TyTerm::Fn {
                        params: vec![tp(inner_param)],
                        ret: Box::new(TyTerm::Unit),

                        effect: infer_pure(),
                        captures: vec![],
                        hint: None,
                    })],
                    ret: Box::new(TyTerm::Unit),

                    effect: infer_pure(),
                    captures: vec![],
                    hint: None,
                })],
                ret: Box::new(TyTerm::Unit),

                effect: infer_pure(),
                captures: vec![],
                hint: None,
            }
        };
        let a = mk(TyTerm::List(Box::new(TyTerm::Int)));
        let b = mk(TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)));
        assert!(s.unify_ty(&a, &b, Covariant, &registry).is_ok());
    }

    // ================================================================
    // Regression: Deque with same identity must not trigger demotion
    // ================================================================

    #[test]
    fn same_identity_no_demotion_even_covariant() {
        // Same identity → identities unify → no demotion path, stays Deque.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o.clone())),
                Covariant, &registry)
            .is_ok()
        );
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o.clone())),
                Covariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&v), TyTerm::Deque(Box::new(TyTerm::Int), Box::new(o)));
    }

    #[test]
    fn same_identity_param_no_demotion() {
        // Identity Param binds to concrete. Second use with same Param → same concrete → no demotion.
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let c = s.alloc_identity(false);
        let ov = s.fresh_ty_var();
        let v = s.fresh_ty_var();
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(ov)), Invariant, &registry)
                .is_ok()
        );
        assert!(
            s.unify_ty(
                &v,
                &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(c.clone())),
                Covariant, &registry)
            .is_ok()
        );
        // ov now bound to c. Second concrete same as c → no mismatch.
        assert!(
            s.unify_ty(&v, &TyTerm::Deque(Box::new(TyTerm::Int), Box::new(c)), Covariant, &registry)
                .is_ok()
        );
        // Still Deque, not List.
        let resolved = s.resolve_ty(&v);
        assert!(
            matches!(resolved, TyTerm::Deque(_, _)),
            "should stay Deque, got {resolved:?}"
        );
    }

    // ── Sequence identity tracking ─────────────────────────────────

    // ── UserDefined unification tests ───────────────────────────────

    fn ud(id: QualifiedRef, type_args: Vec<InferTy>, effect_args: Vec<InferEffect>) -> InferTy {
        TyTerm::UserDefined {
            id,
            type_args,
            effect_args,
        }
    }

    // -- Completeness: valid UserDefined unifications --

    #[test]
    fn user_defined_same_id_empty_args_unifies() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(&ud(id, vec![], vec![]), &ud(id, vec![], vec![]), Invariant, &registry)
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
                &ud(id, vec![TyTerm::Int], vec![]),
                &ud(id, vec![TyTerm::Int], vec![]),
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
                &ud(id, vec![p.clone()], vec![]),
                &ud(id, vec![TyTerm::Int], vec![]),
                Invariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_ty(&p), TyTerm::Int);
    }

    #[test]
    fn user_defined_effect_var_resolved_via_unify() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let e = s.fresh_effect_var();
        assert!(
            s.unify_ty(
                &ud(id, vec![], vec![e.clone()]),
                &ud(id, vec![], vec![infer_pure()]),
                Covariant, &registry)
            .is_ok()
        );
        assert_eq!(s.resolve_infer_effect(&e), infer_pure());
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
                &ud(id, vec![TyTerm::List(Box::new(p.clone()))], vec![]),
                &ud(id, vec![TyTerm::List(Box::new(TyTerm::Int))], vec![]),
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
                &ud(id_a, vec![], vec![]),
                &ud(id_b, vec![], vec![]),
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
                &ud(id, vec![TyTerm::Int], vec![]),
                &ud(id, vec![TyTerm::String], vec![]),
                Invariant, &registry)
            .is_err()
        );
    }

    #[test]
    fn user_defined_effect_arg_mismatch_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        assert!(
            s.unify_ty(
                &ud(id, vec![], vec![infer_pure()]),
                &ud(id, vec![], vec![effectful()]),
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
            s.unify_ty(&ud(id, vec![], vec![]), &TyTerm::Int, Invariant, &registry)
                .is_err()
        );
        assert!(
            s.unify_ty(&TyTerm::String, &ud(id, vec![], vec![]), Invariant, &registry)
                .is_err()
        );
    }

    // -- Resolve --

    #[test]
    fn user_defined_resolve_substitutes_args() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        let e = s.fresh_effect_var();
        let ty = ud(id, vec![p.clone()], vec![e.clone()]);

        // Bind via unification
        assert!(s.unify_ty(&p, &TyTerm::Int, Invariant, &registry).is_ok());
        let eff = effectful();
        assert!(s.unify_infer_effects(&e, &eff, Covariant).is_ok());

        let resolved = s.resolve_ty(&ty);
        match resolved {
            TyTerm::UserDefined {
                id: rid,
                type_args,
                effect_args,
                ..
            } => {
                assert_eq!(rid, id);
                assert_eq!(type_args, vec![TyTerm::Int]);
                assert_eq!(effect_args, vec![eff]);
            }
            _ => panic!("expected UserDefined, got {resolved:?}"),
        }
    }

    #[test]
    fn user_defined_inside_list_resolves() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let id = fresh_qref();
        let p = s.fresh_ty_var();
        let ty = TyTerm::List(Box::new(ud(id, vec![p.clone()], vec![])));
        assert!(s.unify_ty(&p, &TyTerm::Int, Invariant, &registry).is_ok());
        match s.resolve_ty(&ty) {
            TyTerm::List(inner) => match *inner {
                TyTerm::UserDefined { type_args, .. } => assert_eq!(type_args, vec![TyTerm::Int]),
                other => panic!("expected UserDefined, got {other:?}"),
            },
            other => panic!("expected List, got {other:?}"),
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
            effect_params: vec![],
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
            effect_params: vec![],
        });
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: vec![],
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
            effect_params: vec![],
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

    #[test]
    fn extern_cast_basic_coercion() {
        // UserDefined(A, [T]) → List<T>
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| TyTerm::List(Box::new(p[0].clone())));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],
        };
        let to = TyTerm::List(Box::new(TyTerm::Int));
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
    }

    #[test]
    fn extern_cast_with_param_resolution() {
        // UserDefined(A, [T]) → List<T>, where T is a fresh param on the consumer side
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| TyTerm::List(Box::new(p[0].clone())));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],
        };
        let consumer_param = s.fresh_ty_var();
        let to = TyTerm::List(Box::new(consumer_param.clone()));
        assert!(s.unify_ty(&from, &to, Covariant, &registry).is_ok());
        assert_eq!(s.resolve_ty(&consumer_param), TyTerm::Int);
    }

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

    #[test]
    fn extern_cast_wrong_target_fails() {
        // Rule: A → List<T>, but expected String
        let (id, _fn_id, mut s, registry) = make_cast_solver(1, |p| TyTerm::List(Box::new(p[0].clone())));

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],
        };
        assert!(s.unify_ty(&from, &TyTerm::String, Covariant, &registry).is_err());
    }

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
            to: TyTerm::List(Box::new(t1)),
            fn_ref: fn_id_a,
        };
        let t2 = builder.fresh_ty_var();
        let rule_b = CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
            },
            to: TyTerm::List(Box::new(t2)),
            fn_ref: fn_id_b,
        };

        // Build registry manually (bypassing register_cast duplicate check)
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: vec![],
        });
        reg.from_rules.entry(id).or_default().push(rule_a);
        reg.from_rules.entry(id).or_default().push(rule_b);

        let from = TyTerm::UserDefined {
            id,
            type_args: vec![TyTerm::Int],
            effect_args: vec![],
        };
        assert!(
            s.unify_ty(&from, &TyTerm::List(Box::new(TyTerm::Int)), Covariant, &reg)
                .is_err()
        );
    }

    // -- TypeRegistry cast rules --

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
            effect_params: vec![],
        });
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t.clone()],
                effect_args: vec![],
            },
            to: TyTerm::List(Box::new(t.clone())),
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
            to: TyTerm::List(Box::new(t2)),
            fn_ref: fn_id_b,
        });
    }

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
            effect_params: vec![],
        });
        let mut builder1 = PolyBuilder::new();
        let t1 = builder1.fresh_ty_var();
        reg.register_cast(CastRule {
            from: TyTerm::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
            },
            to: TyTerm::List(Box::new(t1)),
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
    fn purity_scalars_are_pure() {
        assert_eq!(Ty::Int.materiality(), Materiality::Concrete);
        assert_eq!(Ty::Float.materiality(), Materiality::Concrete);
        assert_eq!(Ty::String.materiality(), Materiality::Concrete);
        assert_eq!(Ty::Bool.materiality(), Materiality::Concrete);
        assert_eq!(Ty::Unit.materiality(), Materiality::Concrete);
        assert_eq!(Ty::Range.materiality(), Materiality::Concrete);
        assert_eq!(Ty::Byte.materiality(), Materiality::Concrete);
    }

    #[test]
    fn purity_containers_are_lazy() {
        let mut s = Solver::new();
        let infer_o = s.alloc_identity(false);
        let o = s.freeze_ty(&infer_o).unwrap();
        assert_eq!(
            Ty::List(Box::new(Ty::Int)).materiality(),
            Materiality::Composite
        );
        assert_eq!(
            Ty::Deque(Box::new(Ty::Int), Box::new(o)).materiality(),
            Materiality::Composite
        );
        assert_eq!(
            Ty::Option(Box::new(Ty::Int)).materiality(),
            Materiality::Composite
        );
        assert_eq!(
            Ty::Tuple(vec![Ty::Int]).materiality(),
            Materiality::Composite
        );
    }

    #[test]
    fn purity_object_is_lazy() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(i.intern("x"), Ty::Int)]));
        assert_eq!(obj.materiality(), Materiality::Composite);
    }

    #[test]
    fn purity_fn_is_lazy() {
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::String),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert_eq!(fn_ty.materiality(), Materiality::Composite);
    }

    #[test]
    fn purity_extern_fn_is_lazy() {
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::String)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert_eq!(fn_ty.materiality(), Materiality::Composite);
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
    fn pureable_scalars() {
        assert!(Ty::Int.is_pureable());
        assert!(Ty::Float.is_pureable());
        assert!(Ty::String.is_pureable());
        assert!(Ty::Bool.is_pureable());
        assert!(Ty::Unit.is_pureable());
        assert!(Ty::Range.is_pureable());
        assert!(Ty::Byte.is_pureable());
    }

    #[test]
    fn pureable_list_of_scalars() {
        assert!(Ty::List(Box::new(Ty::Int)).is_pureable());
        assert!(Ty::List(Box::new(Ty::String)).is_pureable());
    }

    #[test]
    fn pureable_list_of_fn_with_pure_captures() {
        // Fn with empty captures and pure ret → pureable, so List<Fn> is also pureable.
        let list_fn = Ty::List(Box::new(Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        }));
        assert!(list_fn.is_pureable());
    }

    #[test]
    fn pureable_list_of_fn_with_user_defined_capture() {
        // Fn with UserDefined capture → not pureable, so List<Fn> is also not pureable.
        let list_fn = Ty::List(Box::new(Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        }));
        assert!(!list_fn.is_pureable());
    }

    #[test]
    fn pureable_list_of_user_defined_is_not_pureable() {
        let list_ud = Ty::List(Box::new(test_user_defined()));
        assert!(!list_ud.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_scalars() {
        // List<List<Int>> — pureable
        let nested = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(nested.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_fn_pure_captures() {
        // List<List<Fn(Int) -> Int>> with empty captures — pureable
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(nested.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_fn_user_defined_capture() {
        // List<List<Fn(Int) -> Int>> with UserDefined capture — not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(!nested.is_pureable());
    }

    #[test]
    fn pureable_deque_of_scalars() {
        let mut s = Solver::new();
        let infer_o = s.alloc_identity(false);
        let o = s.freeze_ty(&infer_o).unwrap();
        assert!(Ty::Deque(Box::new(Ty::Int), Box::new(o)).is_pureable());
    }

    #[test]
    fn pureable_deque_of_user_defined() {
        let mut s = Solver::new();
        let infer_o = s.alloc_identity(false);
        let o = s.freeze_ty(&infer_o).unwrap();
        assert!(!Ty::Deque(Box::new(test_user_defined()), Box::new(o)).is_pureable());
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
    fn pureable_tuple_with_fn_pure() {
        // Fn with no captures and pure ret → pureable
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert!(Ty::Tuple(vec![Ty::Int, fn_ty]).is_pureable());
    }

    #[test]
    fn pureable_tuple_with_fn_user_defined_capture() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        };
        assert!(!Ty::Tuple(vec![Ty::Int, fn_ty]).is_pureable());
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
    fn pureable_object_with_fn_pure() {
        // Fn with no captures, pure ret → object is pureable
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        let obj = Ty::Object(FxHashMap::from_iter([
            (i.intern("x"), Ty::Int),
            (i.intern("callback"), fn_ty),
        ]));
        assert!(obj.is_pureable());
    }

    #[test]
    fn pureable_object_with_fn_user_defined_capture() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        };
        let obj = Ty::Object(FxHashMap::from_iter([
            (i.intern("x"), Ty::Int),
            (i.intern("callback"), fn_ty),
        ]));
        assert!(!obj.is_pureable());
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
    fn pureable_enum_with_fn_pure_payload() {
        // Fn with no captures, pure ret → enum is pureable
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        let enum_ty = Ty::Enum {
            name: i.intern("Wrap"),
            variants: FxHashMap::from_iter([
                (i.intern("Some"), Some(Box::new(fn_ty))),
                (i.intern("None"), None),
            ]),
        };
        assert!(enum_ty.is_pureable());
    }

    #[test]
    fn pureable_enum_with_fn_user_defined_capture_payload() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        };
        let enum_ty = Ty::Enum {
            name: i.intern("Wrap"),
            variants: FxHashMap::from_iter([
                (i.intern("Some"), Some(Box::new(fn_ty))),
                (i.intern("None"), None),
            ]),
        };
        assert!(!enum_ty.is_pureable());
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
    fn pureable_fn_with_pure_captures_and_ret() {
        // Fn with captures=[Int, String] and ret=Bool → pureable
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![Ty::Int, Ty::String],
            hint: None,
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_user_defined_capture() {
        // Fn with captures=[UserDefined] → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
            hint: None,
        };
        assert!(!fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_fn_capture() {
        // Fn with captures=[Fn(Int)->Int (no captures)] → pureable (Fn with empty captures + pure ret)
        let inner_fn = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![inner_fn],
            hint: None,
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_user_defined_ret() {
        // Fn returning UserDefined → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(test_user_defined()),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert!(!fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_list_int_ret() {
        // Fn returning List<Int> → pureable
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_list_fn_ret() {
        // Fn returning List<Fn(Int)->Int> → not pureable (transitive)
        let inner_fn = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()], // inner Fn captures UserDefined
            hint: None,
        };
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(inner_fn))),

            effect: Effect::pure(),
            captures: vec![],
            hint: None,
        };
        assert!(!fn_ty.is_pureable());
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
            Ty::List(Box::new(Ty::String)),
            Ty::Option(Box::new(Ty::Bool)),
        ]);
        assert!(ty.is_pureable());
    }

    #[test]
    fn pureable_mixed_tuple_list_user_defined() {
        // (Int, List<UserDefined>) — not pureable
        let ty = Ty::Tuple(vec![Ty::Int, Ty::List(Box::new(test_user_defined()))]);
        assert!(!ty.is_pureable());
    }

    #[test]
    fn pureable_deeply_nested_containers() {
        // List<Option<Tuple<(Int, List<String>)>>> — pureable
        let inner = Ty::Tuple(vec![Ty::Int, Ty::List(Box::new(Ty::String))]);
        let ty = Ty::List(Box::new(Ty::Option(Box::new(inner))));
        assert!(ty.is_pureable());
    }

    #[test]
    fn pureable_deeply_nested_with_user_defined_leaf() {
        // List<Option<Tuple<(Int, UserDefined)>>> — not pureable
        let inner = Ty::Tuple(vec![Ty::Int, test_user_defined()]);
        let ty = Ty::List(Box::new(Ty::Option(Box::new(inner))));
        assert!(!ty.is_pureable());
    }

    // ================================================================
    // Effect coercion tests — Pure ≤ Effectful
    // ================================================================

    // -- Iterator effect coercion --

    // -- Sequence effect coercion --

    // -- Fn effect coercion --

    #[test]
    fn fn_same_effect_ok() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let a = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };
        let b = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };
        assert!(s.unify_ty(&a, &b, Invariant, &registry).is_ok());
    }

    #[test]
    fn fn_effect_mismatch_invariant_fails() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let a = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };
        let b = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: effectful(),
            hint: None,
        };
        assert!(s.unify_ty(&a, &b, Invariant, &registry).is_err());
    }

    #[test]
    fn fn_pure_to_effectful_covariant() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let v = s.fresh_ty_var();
        let pure_fn = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };
        let effectful_fn = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),

            captures: vec![],
            effect: effectful(),
            hint: None,
        };
        assert!(s.unify_ty(&v, &pure_fn, Covariant, &registry).is_ok());
        assert!(s.unify_ty(&v, &effectful_fn, Covariant, &registry).is_ok());
        match s.resolve_ty(&v) {
            TyTerm::Fn { effect, .. } => assert_eq!(effect, infer_pure()),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    // -- Coercion arm + effect interaction --

    // -- Effect display --

    #[test]
    fn display_effectful_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful_concrete(),
            hint: None,
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn!(Int) -> String");
    }

    #[test]
    fn display_pure_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
            hint: None,
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn(Int) -> String");
    }

    #[test]
    fn display_effectful_extern_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful_concrete(),
            hint: None,
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn!(Int) -> String");
    }

    // -- Three-way effect unification --

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
    #[test]
    fn storable_range() {
        assert!(Ty::Range.is_materializable());
    }

    // -- Lazy containers with pure contents: storable --

    #[test]
    fn storable_list_of_int() {
        assert!(Ty::List(Box::new(Ty::Int)).is_materializable());
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

    // -- Fn: never storable --

    #[test]
    fn not_storable_pure_fn() {
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: Effect::pure(),
            hint: None,
        };
        assert!(!fn_ty.is_materializable());
    }

    #[test]
    fn not_storable_effectful_fn() {
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: effectful_concrete(),
            hint: None,
        };
        assert!(!fn_ty.is_materializable());
    }

    // -- UserDefined: never storable --

    #[test]
    fn not_storable_user_defined() {
        assert!(!test_user_defined().is_materializable());
    }

    // -- Recursive: container with non-storable inner --

    #[test]
    fn not_storable_list_of_fn() {
        let fn_ty = Ty::Fn {
            params: vec![tp_concrete(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: Effect::pure(),
            hint: None,
        };
        assert!(!Ty::List(Box::new(fn_ty)).is_materializable());
    }

    #[test]
    fn not_storable_list_of_user_defined() {
        assert!(!Ty::List(Box::new(test_user_defined())).is_materializable());
    }

    // ================================================================
    // Effect subtyping in HOF signatures
    // ================================================================
    //
    // When Iterator<T, Effectful> is passed to filter/map, the shared effect var E
    // binds to Effectful. The Pure callback then unifies with E(=Effectful) covariant.
    // Pure ≤ Effectful → OK.

    // -- Sequence effect variable tests --

    // ================================================================
    // instantiate
    // ================================================================

    #[test]
    fn instantiate_remaps_params() {
        // Build a "template" signature with a separate TySubst.
        let mut sig_subst = Solver::new();
        let registry = TypeRegistry::new();
        let t = sig_subst.fresh_ty_var(); // Param(0)
        let sig = TyTerm::Fn {
            params: vec![tp(t.clone())],
            ret: Box::new(t),
            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };

        // Instantiate into a different TySubst.
        let mut inference = Solver::new();
        let registry = TypeRegistry::new();
        let _ = inference.fresh_ty_var(); // Param(0) — burn one to show remapping
        let inst = inference.instantiate_infer(&sig);

        // The instantiated type should have Var(1), not Var(0).
        match &inst {
            TyTerm::Fn { params, ret, .. } => {
                assert!(matches!(&params[0].ty, TyTerm::Var(id) if id.0 == 1));
                assert!(matches!(ret.as_ref(), TyTerm::Var(id) if id.0 == 1));
            }
            _ => panic!("expected Fn"),
        }
    }

    #[test]
    fn instantiate_concrete_untouched() {
        let mut s = Solver::new();
        let registry = TypeRegistry::new();
        let concrete = TyTerm::Fn {
            params: vec![tp(TyTerm::Int)],
            ret: Box::new(TyTerm::String),
            captures: vec![],
            effect: infer_pure(),
            hint: None,
        };
        let inst = s.instantiate_infer(&concrete);
        assert_eq!(inst, concrete);
    }

    // ── EffectSet tests ─────────────────────────────────────────────

    mod effect_set_tests {
        use super::*;
        use crate::graph::types::QualifiedRef;
        use acvus_utils::Interner;
        use Polarity::*;

        fn ctx(interner: &Interner, n: usize) -> EffectTarget {
            EffectTarget::Context(QualifiedRef::root(interner.intern(&format!("ctx_{n}"))))
        }

        /// Helper: a non-pure resolved concrete effect with a dummy write target.
        fn effectful_concrete() -> Effect {
            let i = Interner::new();
            Effect::Resolved(EffectSet {
                reads: std::collections::BTreeSet::new(),
                writes: std::collections::BTreeSet::from([EffectTarget::Token(
                    QualifiedRef::root(i.intern("__test")),
                )]),
            })
        }

        /// Helper: a non-pure resolved InferEffect with a dummy write target.
        fn effectful_infer() -> InferEffect {
            let i = Interner::new();
            EffectTerm::Resolved(EffectSet {
                reads: std::collections::BTreeSet::new(),
                writes: std::collections::BTreeSet::from([EffectTarget::Token(
                    QualifiedRef::root(i.intern("__test")),
                )]),
            })
        }

        /// Helper: pure InferEffect.
        fn infer_pure() -> InferEffect {
            EffectTerm::Resolved(EffectSet::default())
        }

        #[test]
        fn default_is_pure() {
            let s = EffectSet::default();
            assert!(s.is_pure());
        }

        #[test]
        fn writes_only_is_not_pure_and_modifying() {
            let i = Interner::new();
            let c = ctx(&i, 0);
            let s = EffectSet {
                writes: [c].into_iter().collect(),
                ..Default::default()
            };
            assert!(!s.is_pure());
            assert!(s.is_modifying());
        }

        #[test]
        fn reads_only_is_not_pure() {
            let i = Interner::new();
            let c = ctx(&i, 0);
            let s = EffectSet {
                reads: [c].into_iter().collect(),
                ..Default::default()
            };
            assert!(!s.is_pure());
        }

        #[test]
        fn writes_only_is_not_pure() {
            let i = Interner::new();
            let c = ctx(&i, 0);
            let s = EffectSet {
                writes: [c].into_iter().collect(),
                ..Default::default()
            };
            assert!(!s.is_pure());
        }

        #[test]
        fn union_pure_pure_is_pure() {
            let a = EffectSet::default();
            let b = EffectSet::default();
            assert!(a.union(&b).is_pure());
        }

        #[test]
        fn union_reads_disjoint() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let a = EffectSet {
                reads: [c1].into_iter().collect(),
                ..Default::default()
            };
            let b = EffectSet {
                reads: [c2].into_iter().collect(),
                ..Default::default()
            };
            let u = a.union(&b);
            assert!(u.reads.contains(&c1));
            assert!(u.reads.contains(&c2));
            assert!(u.writes.is_empty());
        }

        #[test]
        fn union_reads_overlap() {
            let i = Interner::new();
            let c = ctx(&i, 0);
            let a = EffectSet {
                reads: [c].into_iter().collect(),
                ..Default::default()
            };
            let b = EffectSet {
                reads: [c].into_iter().collect(),
                ..Default::default()
            };
            let u = a.union(&b);
            assert_eq!(u.reads.len(), 1);
            assert!(u.reads.contains(&c));
        }

        #[test]
        fn union_reads_writes_independent() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let a = EffectSet {
                reads: [c1].into_iter().collect(),
                ..Default::default()
            };
            let b = EffectSet {
                writes: [c2].into_iter().collect(),
                ..Default::default()
            };
            let u = a.union(&b);
            assert!(u.reads.contains(&c1));
            assert!(u.writes.contains(&c2));
            // reads and writes are independent — c1 is NOT in writes
            assert!(!u.writes.contains(&c1));
            assert!(!u.reads.contains(&c2));
        }

        #[test]
        fn union_modifying_propagates() {
            let i = Interner::new();
            let c = ctx(&i, 0);
            let a = EffectSet {
                writes: [c].into_iter().collect(),
                ..Default::default()
            };
            let b = EffectSet::default();
            assert!(a.union(&b).is_modifying());
            assert!(b.union(&a).is_modifying());
        }

        #[test]
        fn union_modifying_both_true() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let a = EffectSet {
                writes: [c1].into_iter().collect(),
                ..Default::default()
            };
            let b = EffectSet {
                writes: [c2].into_iter().collect(),
                ..Default::default()
            };
            assert!(a.union(&b).is_modifying());
        }

        // ── Effect enum tests ───────────────────────────────────────

        #[test]
        fn effect_pure_is_pure() {
            assert!(Effect::pure().is_pure());
            assert!(!Effect::pure().is_effectful());
        }

        #[test]
        fn effect_with_writes_is_effectful() {
            assert!(!effectful_concrete().is_pure());
            assert!(effectful_concrete().is_effectful());
        }

        #[test]
        fn effect_var_is_var() {
            // InferEffect can have Var; concrete Effect cannot (Infallible).
            let mut s = Solver::new();
            let v = s.fresh_effect_var();
            assert!(matches!(v, EffectTerm::Var(_)));
        }

        #[test]
        fn effect_union_resolved_resolved() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let a = Effect::Resolved(EffectSet {
                reads: [c1].into_iter().collect(),
                ..Default::default()
            });
            let b = Effect::Resolved(EffectSet {
                writes: [c2].into_iter().collect(),
                ..Default::default()
            });
            let u = a.union(&b).unwrap();
            match &u {
                Effect::Resolved(s) => {
                    assert!(s.reads.contains(&c1));
                    assert!(s.writes.contains(&c2));
                }
                _ => panic!("expected Resolved"),
            }
        }

        // effect_union_with_var_returns_none: removed.
        // Concrete Effect can no longer contain Var (Infallible phase).

        #[test]
        fn effect_union_pure_pure_is_pure() {
            let u = Effect::pure().union(&Effect::pure()).unwrap();
            assert!(u.is_pure());
        }

        // ── Solver effect unification tests ────────────────────────

        #[test]
        fn unify_var_binds_to_resolved() {
            let mut s = Solver::new();
            let var = s.fresh_effect_var();
            let concrete = effectful_infer();
            assert!(s.unify_infer_effects(&var, &concrete, Invariant).is_ok());
            let resolved = s.resolve_infer_effect(&var);
            assert!(matches!(&resolved, EffectTerm::Resolved(set) if !set.is_pure()));
        }

        #[test]
        fn unify_var_binds_to_pure() {
            let mut s = Solver::new();
            let var = s.fresh_effect_var();
            let pure = infer_pure();
            assert!(s.unify_infer_effects(&var, &pure, Invariant).is_ok());
            let resolved = s.resolve_infer_effect(&var);
            assert!(matches!(&resolved, EffectTerm::Resolved(set) if set.is_pure()));
        }

        #[test]
        fn unify_pure_pure_ok() {
            let mut s = Solver::new();
            assert!(s.unify_infer_effects(&infer_pure(), &infer_pure(), Invariant).is_ok());
        }

        #[test]
        fn unify_effectful_effectful_ok() {
            let mut s = Solver::new();
            assert!(s.unify_infer_effects(&effectful_infer(), &effectful_infer(), Invariant).is_ok());
        }

        #[test]
        fn unify_pure_effectful_invariant_fails() {
            let mut s = Solver::new();
            assert!(s.unify_infer_effects(&infer_pure(), &effectful_infer(), Invariant).is_err());
        }

        #[test]
        fn unify_pure_effectful_covariant_ok() {
            let mut s = Solver::new();
            // Pure ≤ Effectful in covariant position
            assert!(
                s.unify_infer_effects(&infer_pure(), &effectful_infer(), Covariant)
                    .is_ok()
            );
        }

        #[test]
        fn unify_effectful_pure_covariant_fails() {
            let mut s = Solver::new();
            // Effectful ≤ Pure in covariant position — should fail
            assert!(
                s.unify_infer_effects(&effectful_infer(), &infer_pure(), Covariant)
                    .is_err()
            );
        }

        #[test]
        fn resolve_unbound_var_returns_var() {
            let mut s = Solver::new();
            let var = s.fresh_effect_var();
            let resolved = s.resolve_infer_effect(&var);
            assert!(matches!(resolved, EffectTerm::Var(_)));
        }

        #[test]
        fn resolve_chain_follows_bindings() {
            let mut s = Solver::new();
            let v0 = s.fresh_effect_var();
            let v1 = s.fresh_effect_var();
            let concrete = effectful_infer();
            // v0 → v1 → concrete
            assert!(s.unify_infer_effects(&v0, &v1, Invariant).is_ok());
            assert!(s.unify_infer_effects(&v1, &concrete, Invariant).is_ok());
            let resolved = s.resolve_infer_effect(&v0);
            assert!(matches!(&resolved, EffectTerm::Resolved(set) if !set.is_pure()));
        }

        #[test]
        fn unify_two_vars_share_binding() {
            let mut s = Solver::new();
            let v0 = s.fresh_effect_var();
            let v1 = s.fresh_effect_var();
            assert!(s.unify_infer_effects(&v0, &v1, Invariant).is_ok());
            // Bind one → both resolve to same
            let concrete = effectful_infer();
            assert!(s.unify_infer_effects(&v1, &concrete, Invariant).is_ok());
            assert!(matches!(&s.resolve_infer_effect(&v0), EffectTerm::Resolved(set) if !set.is_pure()));
            assert!(matches!(&s.resolve_infer_effect(&v1), EffectTerm::Resolved(set) if !set.is_pure()));
        }

        #[test]
        fn effect_with_specific_contexts() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let e = Effect::Resolved(EffectSet {
                reads: [c1].into_iter().collect(),
                writes: [c2].into_iter().collect(),
            });
            assert!(!e.is_pure());
            assert!(e.is_effectful());
            match &e {
                Effect::Resolved(s) => {
                    assert_eq!(s.reads.len(), 1);
                    assert_eq!(s.writes.len(), 1);
                }
                _ => panic!("expected Resolved"),
            }
        }

        #[test]
        fn unify_var_with_context_effect() {
            let i = Interner::new();
            let mut s = Solver::new();
            let c = ctx(&i, 0);
            let var = s.fresh_effect_var();
            let effect = EffectTerm::Resolved(EffectSet {
                reads: [c].into_iter().collect(),
                ..Default::default()
            });
            assert!(s.unify_infer_effects(&var, &effect, Invariant).is_ok());
            let resolved = s.resolve_infer_effect(&var);
            match &resolved {
                EffectTerm::Resolved(set) => {
                    assert!(set.reads.contains(&c));
                    assert!(set.writes.is_empty());
                }
                _ => panic!("expected Resolved"),
            }
        }

        #[test]
        fn display_pure() {
            assert_eq!(format!("{}", Effect::pure()), "Pure");
        }

        #[test]
        fn display_effectful_writes_only() {
            // An effect with only writes displays as "Effectful(w=N)".
            assert_eq!(format!("{}", effectful_concrete()), "Effectful(w=1)");
        }

        #[test]
        fn display_reads_writes() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let e = Effect::Resolved(EffectSet {
                reads: [c1].into_iter().collect(),
                writes: [c2].into_iter().collect(),
            });
            let s = format!("{e}");
            assert!(s.starts_with("Effectful("));
            assert!(s.contains("r=1"));
            assert!(s.contains("w=1"));
        }

        #[test]
        fn display_var() {
            let var: InferEffect = EffectTerm::Var(EffectBoundId(42));
            assert_eq!(format!("{var:?}"), "Var(EffectBoundId(42))");
        }
    }
}
