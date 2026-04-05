use std::collections::BTreeSet;
use std::fmt;

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::graph::types::QualifiedRef;

// ── UserDefined type system ──────────────────────────────────────────

/// Ownership classification: how a value of this type may be used.
///
/// - `Copy`: trivially duplicated (e.g. integers, booleans).
/// - `Clone`: explicitly duplicated (e.g. strings, lists).
/// - `MoveOnly`: consumed on use, cannot be duplicated (e.g. handles, iterators).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ownership {
    Copy,
    Clone,
    MoveOnly,
}

/// Declaration of a user-defined type — the **single source of truth**
/// for parameter count and constraints. Registered once, referenced by QualifiedRef everywhere.
#[derive(Debug, Clone)]
pub struct UserDefinedDecl {
    pub qref: QualifiedRef,
    /// Type parameter constraints. `None` = unconstrained.
    pub type_params: Vec<Option<ParamConstraint>>,
    /// Effect parameter constraints. `None` = unconstrained.
    pub effect_params: Vec<Option<EffectConstraint>>,
    /// How values of this type may be used (Copy, Clone, or MoveOnly).
    pub ownership: Ownership,
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
/// Both `from` and `to` share Param/Var placeholders from the same TySubst,
/// so instantiating them together links corresponding parameters.
///
/// Example: `Iterator<T, E> → List<T>`
/// ```ignore
/// let mut s = TySubst::new();
/// let t = s.fresh_param();
/// let e = s.fresh_effect_var();
/// CastRule {
///     from: Ty::UserDefined { id: ITER_ID, type_args: vec![t.clone()], effect_args: vec![e] },
///     to: Ty::List(Box::new(t)),
///     fn_ref: collect_fn_ref,
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CastRule {
    /// Source type pattern (must be UserDefined).
    pub from: Ty,
    /// Target type pattern (Param/Var placeholders shared with `from`).
    pub to: Ty,
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
    Param,
    Error,
}

fn ty_head(ty: &Ty) -> TyHead {
    match ty {
        Ty::Int => TyHead::Int,
        Ty::Float => TyHead::Float,
        Ty::String => TyHead::String,
        Ty::Bool => TyHead::Bool,
        Ty::Unit => TyHead::Unit,
        Ty::Range => TyHead::Range,
        Ty::Byte => TyHead::Byte,
        Ty::List(_) => TyHead::List,
        Ty::Object(_) => TyHead::Object,
        Ty::Tuple(_) => TyHead::Tuple,
        Ty::Fn { .. } => TyHead::Fn,
        Ty::Option(_) => TyHead::Option,
        Ty::Enum { .. } => TyHead::Enum,
        Ty::Handle(..) => TyHead::Handle,
        Ty::Deque(..) => TyHead::Deque,
        Ty::Identity(..) => TyHead::Identity,
        Ty::Ref(..) => TyHead::Ref,
        Ty::UserDefined { id, .. } => TyHead::UserDefined(*id),
        Ty::Param { .. } => TyHead::Param,
        Ty::Error(_) => TyHead::Error,
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
            Ty::UserDefined { id, .. } => Some(*id),
            _ => None,
        };
        let to_qref = match &rule.to {
            Ty::UserDefined { id, .. } => Some(*id),
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
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Astr,
    pub ty: Ty,
}

impl Param {
    pub fn new(name: Astr, ty: Ty) -> Self {
        Self { name, ty }
    }
}

/// Allowed-type set for a constrained type parameter.
///
/// Represents the exact set of concrete types a Param may unify with.
/// When two constrained Params are unified, their sets are intersected.
/// Empty intersection = immediate type error (no valid instantiation exists).
#[derive(Debug, Clone, PartialEq)]
pub struct ParamConstraint(pub Vec<Ty>);

impl ParamConstraint {
    /// Does this concrete type satisfy the constraint?
    pub fn allows(&self, ty: &Ty) -> bool {
        self.0.contains(ty)
    }

    /// Intersect two constraint sets. Returns `None` if the result is empty
    /// (no concrete type can satisfy both constraints simultaneously).
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let result: Vec<Ty> = self
            .0
            .iter()
            .filter(|t| other.0.contains(t))
            .cloned()
            .collect();
        if result.is_empty() {
            None
        } else {
            Some(Self(result))
        }
    }

    /// Convenience constructors for builtin signatures.
    pub fn scalar() -> Self {
        Self(vec![Ty::Int, Ty::Float, Ty::String, Ty::Bool, Ty::Byte])
    }

    pub fn float_or_byte() -> Self {
        Self(vec![Ty::Float, Ty::Byte])
    }
}

/// Opaque token for `Ty::Param`. Only constructible via `TySubst::fresh_param()`.
///
/// This ensures type parameters (inference holes / generic slots) can only be
/// created through the substitution table, preventing accidental fabrication
/// of param ids that bypass the unification system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParamToken(u32);

impl ParamToken {
    /// Numeric id for serialization / display.
    pub fn id(self) -> u32 {
        self.0
    }
}

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
    fn new() -> Self {
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
pub enum EffectBound {
    /// Any context/token access allowed.
    Any,
    /// Only the specified targets allowed.
    Only(BTreeSet<EffectTarget>),
}

/// Constraint on a function's effect — the upper bound of what it may do.
/// Separate from `EffectSet` which represents *actual* observed effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectConstraint {
    pub reads: EffectBound,
    pub writes: EffectBound,
}

impl EffectConstraint {
    /// Pure constraint — no effects allowed at all.
    pub fn pure() -> Self {
        Self {
            reads: EffectBound::Only(BTreeSet::new()),
            writes: EffectBound::Only(BTreeSet::new()),
        }
    }

    /// Read-only constraint — any reads allowed, no writes.
    pub fn read_only() -> Self {
        Self {
            reads: EffectBound::Any,
            writes: EffectBound::Only(BTreeSet::new()),
        }
    }
}

/// Effect classification for functions and lazy computations.
///
/// `Resolved(EffectSet)` contains concrete effect information.
/// `Var(u32)` is an unresolved effect variable for polymorphism in HOF signatures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Effect {
    Resolved(EffectSet),
    Var(u32),
}

impl Effect {
    /// No side effects: reads nothing, writes nothing.
    pub fn pure() -> Self {
        Effect::Resolved(EffectSet::default())
    }

    pub fn is_pure(&self) -> bool {
        matches!(self, Effect::Resolved(s) if s.is_pure())
    }

    pub fn is_var(&self) -> bool {
        matches!(self, Effect::Var(_))
    }

    /// Returns true if this is a resolved, non-pure effect.
    pub fn is_effectful(&self) -> bool {
        matches!(self, Effect::Resolved(s) if !s.is_pure())
    }

    /// Union two resolved effects. If either is Var, returns None.
    pub fn union(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Effect::Resolved(a), Effect::Resolved(b)) => Some(Effect::Resolved(a.union(b))),
            _ => None,
        }
    }
}

impl fmt::Display for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Effect::Resolved(set) => write!(f, "{set}"),
            Effect::Var(id) => write!(f, "EffectVar({id})"),
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

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Range,
    /// Immutable list of elements. Coerces from Deque (losing identity), coerces to Iterator.
    List(Box<Ty>),
    Object(FxHashMap<Astr, Ty>),
    Tuple(Vec<Ty>),
    Fn {
        params: Vec<Param>,
        ret: Box<Ty>,
        captures: Vec<Ty>,
        effect: Effect,
    },
    Byte,
    /// User-defined type, identified by `QualifiedRef`.
    /// `type_args` and `effect_args` lengths must match the corresponding
    /// `UserDefinedDecl` (source of truth). Initially filled with inference
    /// variables (`Param`/`Var`), resolved to concrete types via unification.
    UserDefined {
        id: QualifiedRef,
        type_args: Vec<Ty>,
        effect_args: Vec<Effect>,
        ownership: Ownership,
    },
    Option(Box<Ty>),
    /// User-defined structural enum type.
    /// `name`: enum name (e.g. `Color`).
    /// `variants`: known variants → optional payload type (`None` = no payload).
    /// Open: unification merges variant sets. Same variant with conflicting payload = error.
    Enum {
        name: Astr,
        variants: FxHashMap<Astr, Option<Box<Ty>>>,
    },
    /// Deferred computation handle. Created by `spawn`, consumed by `eval`.
    /// Carries the spawned computation's return type and effect.
    /// Always move-only (Effectful) — eval consumes the handle exactly once.
    Handle(Box<Ty>, Effect),
    /// Deque type: tracked deque with identity.
    /// Second `Ty` is the identity slot: `Ty::Identity(...)` or `Ty::Param` (bound to Identity).
    Deque(Box<Ty>, Box<Ty>),
    /// Collection identity — prevents mixing deques/sequences from different sources.
    /// Only valid in the identity slot of Deque/Sequence, or bound to a Param.
    Identity(Identity),
    /// Type parameter: unification variable / generic slot.
    ///
    /// Used for both inference holes (resolved during type checking) and
    /// polymorphic signatures (preserved in graph-level function types).
    /// Only constructible via `TySubst::fresh_param()` — the private
    /// `ParamToken` constructor enforces this.
    ///
    /// `constraint`: allowed-type set. `None` = unconstrained.
    /// When two constrained Params unify, their sets are intersected.
    Param {
        token: ParamToken,
        constraint: Option<ParamConstraint>,
    },
    /// Projection type: a reference to a storage location of type `T`.
    /// `Load` converts `Ref<T>` → `T`. `Store` writes `T` through `Ref<T>`.
    /// Second field is the volatile qualifier — if true, loads/stores through
    /// this Ref must not be elided or forwarded by SSA. Transitive: any Ref
    /// derived from a volatile Ref inherits volatile.
    /// Ephemeral — cannot be materialized or purified.
    Ref(Box<Ty>, bool),
    /// Poison type: produced after a type error. Unifies with anything to suppress cascading errors.
    ///
    /// Contains a private token — only constructible via `Ty::error()` within `acvus_mir`.
    /// Downstream crates must never fabricate error types.
    Error(ErrorToken),
}

impl Ty {
    /// Create an `Error` (poison) type. See [`ErrorToken`] for permitted uses.
    pub fn error() -> Self {
        Ty::Error(ErrorToken::new())
    }

    pub fn is_param(&self) -> bool {
        matches!(self, Ty::Param { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error(_))
    }

    /// Extract the effect carried by this type, if any.
    /// Handle, Iterator, Sequence, and Fn types carry effects; other types are pure.
    pub fn carried_effect(&self) -> Option<&Effect> {
        match self {
            Ty::Handle(_, effect) => Some(effect),
            Ty::Fn { effect, .. } => Some(effect),
            _ => None,
        }
    }

    /// Extract the element type from a collection type.
    /// Returns `None` if this is not a collection.
    /// Single source of truth for "what is a collection's element type"
    /// (principle 6-A: one fact, one place).
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
            Ty::Param { .. } | Ty::Error(_) => Materiality::Ephemeral,
        }
    }

    /// Returns true if this type can be deeply converted to a pure representation.
    /// Transitively checks container contents — `List<Fn>` returns false.
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
            Ty::UserDefined { .. } => false,
            Ty::Identity(_) => true,
            Ty::Ref(..) => false,
            Ty::Param { .. } | Ty::Error(_) => false,
        }
    }

    /// Returns true if this type can be materialized — serialized to storage
    /// and restored without losing meaning.
    ///
    /// - **Concrete** (scalars): always materializable.
    /// - **Composite** (containers): materializable iff all contents are recursively
    ///   materializable. e.g. `List<Int>` yes, `List<Fn>` no.
    /// - **Ephemeral** (Iterator, Sequence, Fn, Handle, Opaque): never materializable.
    ///   These are runtime-only values that cannot cross storage boundaries.
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
            | Ty::Ref(..) => false,
            Ty::Param { .. } | Ty::Error(_) => false,
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
            Ty::Param { token: t, .. } => write!(f, "?{}", t.0),
            Ty::Error(_) => write!(f, "<error>"),
        }
    }
}

impl<'a> fmt::Debug for TyDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Snapshot of `TySubst` state for rollback during overload resolution.
pub struct TySubstSnapshot {
    bindings: FxHashMap<ParamToken, Ty>,
    effect_bindings: FxHashMap<u32, Effect>,
    next_param: u32,
    next_effect: u32,
    identity_factory: acvus_utils::LocalFactory<IdentityId>,
    last_extern_cast: Option<QualifiedRef>,
}

/// Substitution table for type unification.
pub struct TySubst {
    bindings: FxHashMap<ParamToken, Ty>,
    effect_bindings: FxHashMap<u32, Effect>,
    next_param: u32,
    next_effect: u32,
    identity_factory: acvus_utils::LocalFactory<IdentityId>,
    /// Frozen type registry — provides ExternCast rules for `try_coerce`.
    type_registry: Freeze<TypeRegistry>,
    /// Set by try_coerce when an ExternCast matches. Read by typeck.
    pub last_extern_cast: Option<QualifiedRef>,
}

impl Default for TySubst {
    fn default() -> Self {
        Self::new()
    }
}

impl TySubst {
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
            effect_bindings: FxHashMap::default(),
            next_param: 0,
            next_effect: 0,
            identity_factory: acvus_utils::LocalFactory::new(),
            type_registry: Freeze::default(),
            last_extern_cast: None,
        }
    }

    /// Create a TySubst with a frozen TypeRegistry for ExternCast support.
    pub fn with_registry(type_registry: Freeze<TypeRegistry>) -> Self {
        Self {
            type_registry,
            ..Self::new()
        }
    }

    /// Take the last ExternCast QualifiedRef recorded by `try_coerce`.
    pub fn take_last_extern_cast(&mut self) -> Option<QualifiedRef> {
        self.last_extern_cast.take()
    }

    /// Take a snapshot of the current state for later rollback.
    pub fn snapshot(&self) -> TySubstSnapshot {
        TySubstSnapshot {
            bindings: self.bindings.clone(),
            effect_bindings: self.effect_bindings.clone(),
            next_param: self.next_param,
            next_effect: self.next_effect,
            identity_factory: self.identity_factory.clone(),
            last_extern_cast: self.last_extern_cast,
        }
    }

    /// Restore state from a snapshot, discarding any bindings made since.
    pub fn rollback(&mut self, snap: TySubstSnapshot) {
        self.bindings = snap.bindings;
        self.effect_bindings = snap.effect_bindings;
        self.next_param = snap.next_param;
        self.next_effect = snap.next_effect;
        self.identity_factory = snap.identity_factory;
        self.last_extern_cast = snap.last_extern_cast;
    }

    /// Allocate an identity.
    /// `fresh = false`: Concrete identity for `[]` literals — fixed provenance.
    /// `fresh = true`: Fresh identity for signatures — instantiate replaces with Concrete.
    pub fn alloc_identity(&mut self, fresh: bool) -> Ty {
        let id = self.identity_factory.next();
        if fresh {
            Ty::Identity(Identity::Fresh(id))
        } else {
            Ty::Identity(Identity::Concrete(id))
        }
    }

    /// Allocate a fresh type parameter (inference hole).
    pub fn fresh_param(&mut self) -> Ty {
        let token = ParamToken(self.next_param);
        self.next_param += 1;
        Ty::Param {
            token,
            constraint: None,
        }
    }

    /// Allocate a fresh type parameter with a constraint (restricted inference hole).
    pub fn fresh_param_constrained(&mut self, constraint: ParamConstraint) -> Ty {
        let token = ParamToken(self.next_param);
        self.next_param += 1;
        Ty::Param {
            token,
            constraint: Some(constraint),
        }
    }

    /// Allocate a fresh effect variable for builtin signatures.
    pub fn fresh_effect_var(&mut self) -> Effect {
        let e = Effect::Var(self.next_effect);
        self.next_effect += 1;
        e
    }

    /// Unify two effects with invariant polarity.
    pub fn unify_effect(&mut self, a: &Effect, b: &Effect) -> Result<(), (Effect, Effect)> {
        self.unify_effects(a, b, Polarity::Invariant)
    }

    /// Resolve an effect by following binding chains for Effect::Var.
    pub fn resolve_effect(&self, e: &Effect) -> Effect {
        match e {
            Effect::Var(id) => match self.effect_bindings.get(id) {
                Some(bound) => self.resolve_effect(bound),
                None => e.clone(),
            },
            concrete => concrete.clone(),
        }
    }

    /// Effect unification with subtyping: `Pure ≤ Effectful`.
    ///
    /// For resolved effects, unification computes the union of their effect sets.
    /// Pure ≤ Effectful: a pure effect is a subeffect of any non-pure effect.
    ///
    /// - **Covariant** (`a ≤ b`): `Pure` flows into `Effectful` → OK.
    /// - **Contravariant** (`b ≤ a`): reversed direction.
    /// - **Invariant**: both sides contribute to the union.
    /// - **Var**: binds to concrete regardless of polarity.
    fn unify_effects(
        &mut self,
        a: &Effect,
        b: &Effect,
        pol: Polarity,
    ) -> Result<(), (Effect, Effect)> {
        let a = self.resolve_effect(a);
        let b = self.resolve_effect(b);
        match (&a, &b) {
            (Effect::Resolved(sa), Effect::Resolved(sb)) => {
                match (sa.is_pure(), sb.is_pure()) {
                    (true, true) => Ok(()),
                    // Pure ≤ Effectful (subeffect direction)
                    (true, false) => match pol {
                        Polarity::Covariant => Ok(()),
                        Polarity::Contravariant | Polarity::Invariant => {
                            Err((a.clone(), b.clone()))
                        }
                    },
                    (false, true) => match pol {
                        Polarity::Contravariant => Ok(()),
                        Polarity::Covariant | Polarity::Invariant => Err((a.clone(), b.clone())),
                    },
                    // Both effectful — invariant requires structural match,
                    // but for now we accept (union is implicitly applied).
                    (false, false) => Ok(()),
                }
            }

            (Effect::Var(v), other) | (other, Effect::Var(v)) => {
                if let Effect::Var(v2) = other
                    && v == v2
                {
                    return Ok(());
                }
                self.effect_bindings.insert(*v, other.clone());
                Ok(())
            }
        }
    }

    /// Find the leaf effect Var in a binding chain.
    /// Returns None if the effect is concrete (not a Var).
    fn find_leaf_effect_var(&self, e: &Effect) -> Option<u32> {
        match e {
            Effect::Var(id) => match self.effect_bindings.get(id) {
                Some(bound) => match bound {
                    Effect::Var(_) => self.find_leaf_effect_var(bound),
                    _ => Some(*id),
                },
                None => Some(*id),
            },
            _ => None,
        }
    }

    /// Effect LUB: compute the union of two effects, binding any Vars.
    /// Returns the merged effect.
    fn merge_effects(&mut self, ea: &Effect, eb: &Effect) -> Effect {
        let resolved_a = self.resolve_effect(ea);
        let resolved_b = self.resolve_effect(eb);
        match (&resolved_a, &resolved_b) {
            (Effect::Resolved(sa), Effect::Resolved(sb)) => {
                let merged = Effect::Resolved(sa.union(sb));
                if let Some(v) = self.find_leaf_effect_var(ea) {
                    self.effect_bindings.insert(v, merged.clone());
                }
                if let Some(v) = self.find_leaf_effect_var(eb) {
                    self.effect_bindings.insert(v, merged.clone());
                }
                merged
            }
            (Effect::Resolved(_), Effect::Var(_)) => {
                if let Some(v) = self.find_leaf_effect_var(eb) {
                    self.effect_bindings.insert(v, resolved_a.clone());
                }
                resolved_a
            }
            (Effect::Var(_), Effect::Resolved(_)) => {
                if let Some(v) = self.find_leaf_effect_var(ea) {
                    self.effect_bindings.insert(v, resolved_b.clone());
                }
                resolved_b
            }
            (Effect::Var(_), Effect::Var(_)) => {
                // Unify: bind a's leaf var to b's leaf var.
                let va = self.find_leaf_effect_var(ea);
                let vb = self.find_leaf_effect_var(eb);
                match (va, vb) {
                    (Some(va), Some(vb)) if va != vb => {
                        self.effect_bindings.insert(va, Effect::Var(vb));
                        Effect::Var(vb)
                    }
                    (Some(va), _) => Effect::Var(va),
                    (_, Some(vb)) => Effect::Var(vb),
                    _ => resolved_a,
                }
            }
        }
    }

    /// Compute LUB (least upper bound) of two same-constructor types whose
    /// generic parameters (Identity / Effect) don't match.
    ///
    /// This is the **single place** that decides what happens on parameter mismatch
    /// within the same type constructor. All such cases share the same contract:
    ///
    ///  - **Invariant polarity → always error.** A return-type annotation that
    ///    requires `Deque<X, O1>` must reject `Deque<X, O2>` — no silent coercion.
    ///  - **Non-invariant polarity → coerce both sides to the lattice join.**
    ///    This mirrors subtype coercion: `Deque<T,O1> ≤ List<T>` and
    ///    `Deque<T,O2> ≤ List<T>`, so the common supertype is `List<T>`.
    ///
    /// Lattice joins handled here:
    ///
    /// | mismatch | LUB |
    /// |----------|-----|
    /// | `Deque<T, O1>` vs `Deque<T, O2>` | `List<T>` (identity erased) |
    /// | `Sequence<T, O1, E>` vs `Sequence<T, O2, E>` | `Iterator<T, max(E1,E2)>` (identity erased) |
    /// | `Iterator<T, Pure>` vs `Iterator<T, Effectful>` | `Iterator<T, Effectful>` |
    /// | `Sequence<T, O, Pure>` vs `Sequence<T, O, Effectful>` | `Sequence<T, O, Effectful>` |
    /// | `Fn{Pure}` vs `Fn{Effectful}` (same params/ret) | `Fn{Effectful}` |
    fn try_lub(&mut self, a: &Ty, b: &Ty) -> Option<Ty> {
        match (a, b) {
            // Identity mismatch: Deque → List (identity erased)
            (Ty::Deque(ia, _), Ty::Deque(ib, _)) => {
                self.unify(ia, ib, Polarity::Invariant).ok()?;
                Some(Ty::List(Box::new(self.resolve(ia))))
            }
            // Effect mismatch: Fn (same params/ret) → Fn (effect = union)
            (
                Ty::Fn {
                    params: pa,
                    ret: ra,
                    effect: ea,
                    ..
                },
                Ty::Fn {
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
                    self.unify(&a.ty, &b.ty, Polarity::Invariant).ok()?;
                }
                self.unify(ra, rb, Polarity::Invariant).ok()?;
                let merged = self.merge_effects(ea, eb);
                Some(Ty::Fn {
                    params: pa
                        .iter()
                        .map(|p| Param {
                            name: p.name,
                            ty: self.resolve(&p.ty),
                        })
                        .collect(),
                    ret: Box::new(self.resolve(ra)),
                    captures: vec![],
                    effect: merged,
                })
            }
            // UserDefined: same id, effect mismatch → effect union.
            // type_args mismatch → CastRule-based LUB (common target).
            (
                Ty::UserDefined {
                    id: id_a,
                    type_args: ta_a,
                    effect_args: ea_a,
                    ownership: ownership_a,
                },
                Ty::UserDefined {
                    id: id_b,
                    type_args: ta_b,
                    effect_args: ea_b,
                    ..
                },
            ) if id_a == id_b => {
                assert_eq!(ta_a.len(), ta_b.len());
                assert_eq!(ea_a.len(), ea_b.len());

                // Try type_args invariant unify first.
                let snap = self.snapshot();
                let type_args_ok = ta_a
                    .iter()
                    .zip(ta_b.iter())
                    .all(|(a, b)| self.unify(a, b, Polarity::Invariant).is_ok());

                if type_args_ok {
                    // type_args match — union effect_args.
                    let merged_effects: Vec<Effect> = ea_a
                        .iter()
                        .zip(ea_b.iter())
                        .map(|(ea, eb)| self.merge_effects(ea, eb))
                        .collect();
                    Some(Ty::UserDefined {
                        id: *id_a,
                        type_args: ta_a.iter().map(|t| self.resolve(t)).collect(),
                        effect_args: merged_effects,
                        ownership: *ownership_a,
                    })
                } else {
                    // type_args mismatch — rollback and try CastRule-based LUB.
                    self.rollback(snap);
                    self.try_lub_via_cast_rules(*id_a, a, b)
                }
            }
            _ => None,
        }
    }

    /// Find LUB via CastRule: both sides cast to a common target type.
    fn try_lub_via_cast_rules(&mut self, from_id: QualifiedRef, a: &Ty, b: &Ty) -> Option<Ty> {
        let rules = self.type_registry.rules_from(from_id).to_vec();
        if rules.is_empty() {
            return None;
        }

        for rule in &rules {
            let snap = self.snapshot();

            // Cast A through this rule.
            let (inst_from_a, inst_to_a) = self.instantiate_pair(&rule.from, &rule.to);
            let a_ok = self.unify(a, &inst_from_a, Polarity::Invariant).is_ok();
            let target_a = if a_ok {
                Some(self.resolve(&inst_to_a))
            } else {
                None
            };
            self.rollback(snap);

            let target_a = target_a?;

            let snap = self.snapshot();
            // Cast B through the same rule.
            let (inst_from_b, inst_to_b) = self.instantiate_pair(&rule.from, &rule.to);
            let b_ok = self.unify(b, &inst_from_b, Polarity::Invariant).is_ok();
            let target_b = if b_ok {
                Some(self.resolve(&inst_to_b))
            } else {
                None
            };
            self.rollback(snap);

            let target_b = target_b?;

            // Try to unify the two cast targets (recursively — may invoke LUB again).
            let snap = self.snapshot();
            if self
                .unify(&target_a, &target_b, Polarity::Covariant)
                .is_ok()
            {
                let result = self.resolve(&target_a);
                self.rollback(snap);
                return Some(result);
            }
            self.rollback(snap);
        }

        None
    }

    /// Rebind leaf type-vars to the LUB, or return Err if polarity is Invariant.
    ///
    /// This is the shared fallback for all same-constructor parameter mismatches
    /// (Identity, Effect). Must only be called after the exact invariant unify
    /// has already failed.
    fn lub_or_err(
        &mut self,
        pol: Polarity,
        orig_a: &Ty,
        orig_b: &Ty,
        a: &Ty,
        b: &Ty,
    ) -> Result<(), (Ty, Ty)> {
        if pol == Polarity::Invariant {
            return Err((a.clone(), b.clone()));
        }
        let leaf_a = self.find_leaf_param(orig_a);
        let leaf_b = self.find_leaf_param(orig_b);
        // LUB only applies when at least one side traces back to a Param
        // that can be rebound. concrete vs concrete → no rebind target → fail.
        if leaf_a.is_none() && leaf_b.is_none() {
            return Err((a.clone(), b.clone()));
        }
        let lub = self.try_lub(a, b).ok_or_else(|| (a.clone(), b.clone()))?;
        if let Some(leaf) = leaf_a {
            self.bindings.insert(leaf, lub.clone());
        }
        if let Some(leaf) = leaf_b {
            self.bindings.insert(leaf, lub);
        }
        Ok(())
    }

    /// Resolve a type by following substitution chains.
    pub fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Param { token, constraint } => {
                if let Some(bound) = self.bindings.get(token) {
                    self.resolve(bound)
                } else {
                    Ty::Param {
                        token: *token,
                        constraint: constraint.clone(),
                    }
                }
            }
            Ty::List(inner) => Ty::List(Box::new(self.resolve(inner))),
            Ty::Deque(inner, identity) => Ty::Deque(
                Box::new(self.resolve(inner)),
                Box::new(self.resolve(identity)),
            ),
            Ty::Identity(_) => ty.clone(),
            Ty::Option(inner) => Ty::Option(Box::new(self.resolve(inner))),
            Ty::Object(fields) => {
                let resolved: FxHashMap<_, _> =
                    fields.iter().map(|(k, v)| (*k, self.resolve(v))).collect();
                Ty::Object(resolved)
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Fn {
                params,
                ret,
                captures,
                effect,
            } => Ty::Fn {
                params: params
                    .iter()
                    .map(|p| Param {
                        name: p.name,
                        ty: self.resolve(&p.ty),
                    })
                    .collect(),
                ret: Box::new(self.resolve(ret)),
                captures: captures.iter().map(|c| self.resolve(c)).collect(),
                effect: self.resolve_effect(effect),
            },
            Ty::Enum { name, variants } => {
                let resolved: FxHashMap<_, _> = variants
                    .iter()
                    .map(|(tag, payload)| {
                        (*tag, payload.as_ref().map(|ty| Box::new(self.resolve(ty))))
                    })
                    .collect();
                Ty::Enum {
                    name: *name,
                    variants: resolved,
                }
            }
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                ownership,
            } => Ty::UserDefined {
                id: *id,
                type_args: type_args.iter().map(|t| self.resolve(t)).collect(),
                effect_args: effect_args.iter().map(|e| self.resolve_effect(e)).collect(),
                ownership: *ownership,
            },
            other => other.clone(),
        }
    }

    /// Find the leaf Param in a chain that is bound to a concrete type.
    /// Returns None if `ty` is not a Param.
    pub fn find_leaf_param(&self, ty: &Ty) -> Option<ParamToken> {
        match ty {
            Ty::Param { token, .. } => {
                if let Some(bound) = self.bindings.get(token) {
                    match bound {
                        Ty::Param { .. } => self.find_leaf_param(bound),
                        _ => Some(*token),
                    }
                } else {
                    Some(*token)
                }
            }
            _ => None,
        }
    }

    /// Rebind a type parameter to a new type, replacing any existing binding.
    pub fn rebind(&mut self, param: ParamToken, ty: Ty) {
        self.bindings.insert(param, ty);
    }

    /// Shallow-resolve: follow Param chains but don't recurse into structure.
    pub fn shallow_resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Param { token, constraint } => {
                if let Some(bound) = self.bindings.get(token) {
                    self.shallow_resolve(bound)
                } else {
                    Ty::Param {
                        token: *token,
                        constraint: constraint.clone(),
                    }
                }
            }
            other => other.clone(),
        }
    }

    /// Unify two types with polarity-based subtyping.
    ///
    /// - `Covariant`: `a ≤ b` — `a` may be a subtype of `b` (Deque→List→Iterator).
    /// - `Contravariant`: `b ≤ a` — reversed direction.
    /// - `Invariant`: `a = b` — no subtyping, must be exactly equal.
    pub fn unify(&mut self, a: &Ty, b: &Ty, pol: Polarity) -> Result<(), (Ty, Ty)> {
        let orig_a = a;
        let orig_b = b;
        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);

        match (&a, &b) {
            // Error (poison) unifies with anything — suppresses cascading errors.
            (Ty::Error(_), _) | (_, Ty::Error(_)) => Ok(()),

            (Ty::Int, Ty::Int)
            | (Ty::Float, Ty::Float)
            | (Ty::String, Ty::String)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Unit, Ty::Unit)
            | (Ty::Range, Ty::Range)
            | (Ty::Byte, Ty::Byte) => Ok(()),

            (
                Ty::UserDefined {
                    id: id_a,
                    type_args: ta_args,
                    effect_args: ea_args,
                    ..
                },
                Ty::UserDefined {
                    id: id_b,
                    type_args: tb_args,
                    effect_args: eb_args,
                    ..
                },
            ) if id_a == id_b => {
                // Same UserDefined type — unify args pairwise.
                // type_args: invariant. effect_args: caller's polarity.
                // On any mismatch → lub_or_err for LUB (effect union, CastRule LUB).
                assert_eq!(ta_args.len(), tb_args.len(), "type_args length mismatch");
                assert_eq!(ea_args.len(), eb_args.len(), "effect_args length mismatch");
                let snap = self.snapshot();
                let type_ok = ta_args
                    .iter()
                    .zip(tb_args.iter())
                    .all(|(a, b)| self.unify(a, b, Polarity::Invariant).is_ok());
                let effect_ok = type_ok
                    && ea_args
                        .iter()
                        .zip(eb_args.iter())
                        .all(|(ea, eb)| self.unify_effects(ea, eb, pol).is_ok());
                if effect_ok {
                    Ok(())
                } else {
                    self.rollback(snap);
                    self.lub_or_err(pol, orig_a, orig_b, &a, &b)
                }
            }

            // Structural enum unification: merge variant sets (open matching).
            (
                Ty::Enum {
                    name: na,
                    variants: va,
                },
                Ty::Enum {
                    name: nb,
                    variants: vb,
                },
            ) => {
                if na != nb {
                    return Err((a, b));
                }
                // Unify overlapping variant payloads.
                for (tag, payload_a) in va {
                    if let Some(payload_b) = vb.get(tag) {
                        match (payload_a, payload_b) {
                            (None, None) => {}
                            (Some(ty_a), Some(ty_b)) => self.unify(ty_a, ty_b, pol)?,
                            _ => return Err((a.clone(), b.clone())),
                        }
                    }
                }
                // Merge if variant sets differ.
                let needs_merge = va.len() != vb.len() || va.keys().any(|k| !vb.contains_key(k));
                if needs_merge {
                    let mut merged: FxHashMap<Astr, Option<Box<Ty>>> = va.clone();
                    for (tag, payload) in vb {
                        merged.entry(*tag).or_insert_with(|| payload.clone());
                    }
                    let merged_ty = Ty::Enum {
                        name: *na,
                        variants: merged,
                    };
                    if let Some(leaf) = self.find_leaf_param(orig_a) {
                        self.bindings.insert(leaf, merged_ty.clone());
                    }
                    if let Some(leaf) = self.find_leaf_param(orig_b) {
                        self.bindings.insert(leaf, merged_ty);
                    }
                }
                Ok(())
            }

            (
                Ty::Param {
                    token: p,
                    constraint,
                },
                other,
            )
            | (
                other,
                Ty::Param {
                    token: p,
                    constraint,
                },
            ) => {
                if let Ty::Param { token: p2, .. } = other
                    && p == p2
                {
                    return Ok(());
                }
                if self.occurs_in(*p, other) {
                    return Err((a.clone(), b.clone()));
                }
                match other {
                    // Param + Param: intersect constraints onto the target.
                    Ty::Param {
                        token: p2,
                        constraint: c2,
                    } => {
                        let merged = match (constraint, c2) {
                            (None, None) => None,
                            (Some(c), None) | (None, Some(c)) => Some(c.clone()),
                            (Some(ca), Some(cb)) => {
                                let Some(inter) = ca.intersect(cb) else {
                                    return Err((a.clone(), b.clone()));
                                };
                                Some(inter)
                            }
                        };
                        let target = Ty::Param {
                            token: *p2,
                            constraint: merged,
                        };
                        self.bindings.insert(*p, target);
                    }
                    // Param + Error: always OK (poison absorption).
                    Ty::Error(_) => {
                        self.bindings.insert(*p, other.clone());
                    }
                    // Param + concrete: check constraint.
                    _ => {
                        if let Some(c) = constraint
                            && !c.allows(other)
                        {
                            return Err((a.clone(), b.clone()));
                        }
                        self.bindings.insert(*p, other.clone());
                    }
                }
                Ok(())
            }

            (Ty::Tuple(ea), Ty::Tuple(eb)) => {
                if ea.len() != eb.len() {
                    return Err((a.clone(), b.clone()));
                }
                for (ta, tb) in ea.iter().zip(eb.iter()) {
                    self.unify(ta, tb, pol)?;
                }
                Ok(())
            }

            // --- Same-constructor arms ---
            // All generic parameters (T, O, E) are invariant.
            // On parameter mismatch, delegate to lub_or_err which:
            //   - Invariant polarity → error (no silent coercion)
            //   - Non-invariant → coerce both sides to lattice join (try_lub)
            (Ty::List(a), Ty::List(b)) => self.unify(a, b, Polarity::Invariant),

            (Ty::Deque(ia, oa), Ty::Deque(ib, ob)) => {
                match self.unify(oa, ob, Polarity::Invariant) {
                    Ok(()) => self.unify(ia, ib, Polarity::Invariant),
                    Err(_) => self.lub_or_err(pol, orig_a, orig_b, &a, &b),
                }
            }

            // Identity: concrete identities must match exactly.
            (Ty::Identity(a), Ty::Identity(b)) => {
                if a == b {
                    Ok(())
                } else {
                    Err((Ty::Identity(*a), Ty::Identity(*b)))
                }
            }

            (Ty::Option(a), Ty::Option(b)) => self.unify(a, b, Polarity::Invariant),

            (Ty::Object(fa), Ty::Object(fb)) => {
                // Unify overlapping fields. On field mismatch, try field-level LUB.
                let snap = self.snapshot();
                let mut field_mismatch = false;
                for (key, ty_a) in fa {
                    if let Some(ty_b) = fb.get(key)
                        && self.unify(ty_a, ty_b, pol).is_err()
                    {
                        field_mismatch = true;
                        break;
                    }
                }
                if field_mismatch && pol != Polarity::Invariant {
                    // Rollback and try field-level LUB for mismatched fields.
                    self.rollback(snap);
                    let mut merged = FxHashMap::default();
                    for (key, ty_a) in fa {
                        if let Some(ty_b) = fb.get(key) {
                            // Try exact unify first (snapshot+rollback).
                            let fsnap = self.snapshot();
                            if self.unify(ty_a, ty_b, pol).is_ok() {
                                merged.insert(*key, self.resolve(ty_a));
                            } else {
                                self.rollback(fsnap);
                                // Field LUB.
                                let lub = self
                                    .try_lub(ty_a, ty_b)
                                    .ok_or_else(|| (a.clone(), b.clone()))?;
                                merged.insert(*key, lub);
                            }
                        } else {
                            merged.insert(*key, self.resolve(ty_a));
                        }
                    }
                    // Add fields only in fb.
                    for (key, ty_b) in fb {
                        if !fa.contains_key(key) {
                            merged.insert(*key, self.resolve(ty_b));
                        }
                    }
                    let merged_ty = Ty::Object(merged);
                    if let Some(leaf) = self.find_leaf_param(orig_a) {
                        self.bindings.insert(leaf, merged_ty.clone());
                    }
                    if let Some(leaf) = self.find_leaf_param(orig_b) {
                        self.bindings.insert(leaf, merged_ty);
                    }
                    return Ok(());
                } else if field_mismatch {
                    return Err((a.clone(), b.clone()));
                }

                // Check if fields differ (one side has keys the other doesn't).
                let a_only = fa.keys().any(|k| !fb.contains_key(k));
                let b_only = fb.keys().any(|k| !fa.contains_key(k));

                if !a_only && !b_only {
                    // Exact same key set — overlapping unify above is sufficient.
                    return Ok(());
                }

                // Fields differ: merge is only valid if at least one side
                // traces back to a Var (partial constraint that can grow).
                let leaf_a = self.find_leaf_param(orig_a);
                let leaf_b = self.find_leaf_param(orig_b);

                if leaf_a.is_none() && leaf_b.is_none() {
                    // Both concrete — differing fields is a type error.
                    return Err((a.clone(), b.clone()));
                }

                // Merge all fields.
                let mut merged = FxHashMap::default();
                for (k, v) in fa {
                    merged.insert(*k, self.resolve(v));
                }
                for (k, v) in fb {
                    merged.entry(*k).or_insert_with(|| self.resolve(v));
                }
                let merged_ty = Ty::Object(merged);

                if let Some(var) = leaf_a {
                    self.bindings.insert(var, merged_ty.clone());
                }
                if let Some(var) = leaf_b {
                    self.bindings.insert(var, merged_ty);
                }
                Ok(())
            }

            (
                Ty::Fn {
                    params: pa,
                    ret: ra,
                    captures: _,
                    effect: ea,
                },
                Ty::Fn {
                    params: pb,
                    ret: rb,
                    captures: _,
                    effect: eb,
                },
            ) => {
                if pa.len() != pb.len() {
                    return Err((a.clone(), b.clone()));
                }
                // Function params are contravariant: flip polarity.
                let param_pol = pol.flip();
                for (ta, tb) in pa.iter().zip(pb.iter()) {
                    self.unify(&ta.ty, &tb.ty, param_pol)?;
                }
                // Return type keeps polarity.
                self.unify(ra, rb, pol)?;
                self.unify_effects(ea, eb, pol)
                    .or_else(|_| self.lub_or_err(pol, orig_a, orig_b, &a, &b))
            }

            // Cross-type coercion: delegate to try_coerce based on polarity.
            // Covariant: a ≤ b. Contravariant: b ≤ a (flip and retry).
            _ => {
                if pol != Polarity::Invariant {
                    let (sub, sup) = match pol {
                        Polarity::Covariant => (&a, &b),
                        Polarity::Contravariant => (&b, &a),
                        Polarity::Invariant => unreachable!(),
                    };
                    if self.try_coerce(sub, sup).is_ok() {
                        return Ok(());
                    }
                }
                Err((a, b))
            }
        }
    }

    /// Try subtype coercion: `sub ≤ sup`.
    /// All coercion rules in one place. Inner types are always unified invariant.
    ///
    /// Lattice: Deque ≤ Sequence ≤ Iterator, Deque ≤ List ≤ Iterator.
    /// Effect: Pure ≤ Effectful (on types that carry Effect).
    fn try_coerce(&mut self, sub: &Ty, sup: &Ty) -> Result<(), ()> {
        match (sub, sup) {
            // Deque<T, O> ≤ List<T>
            (Ty::Deque(inner_d, _), Ty::List(inner_l)) => self
                .unify(inner_d, inner_l, Polarity::Invariant)
                .map_err(|_| ()),
            // ExternCast: UserDefined → anything, via registered CastRule (from_rules).
            (Ty::UserDefined { id, .. }, _) => {
                let rules = self.type_registry.rules_from(*id).to_vec();
                self.try_extern_cast_rules(&rules, sub, sup)
            }
            // ExternCast: anything → UserDefined, via registered CastRule (to_rules).
            (_, Ty::UserDefined { id, .. }) => {
                let rules = self.type_registry.rules_to(*id).to_vec();
                self.try_extern_cast_rules(&rules, sub, sup)
            }
            _ => Err(()),
        }
    }

    /// Try to find a unique ExternCast rule among `rules` that converts `sub` to `sup`.
    /// Sets `last_extern_cast` on success for typeck to read.
    fn try_extern_cast_rules(&mut self, rules: &[CastRule], sub: &Ty, sup: &Ty) -> Result<(), ()> {
        if rules.is_empty() {
            return Err(());
        }

        // Phase 1: probe — find how many rules match (snapshot+rollback each).
        let mut matched_idx = None;
        for (i, rule) in rules.iter().enumerate() {
            let snap = self.snapshot();
            let (inst_from, inst_to) = self.instantiate_pair(&rule.from, &rule.to);
            let ok = self.unify(sub, &inst_from, Polarity::Invariant).is_ok()
                && self.unify(&inst_to, sup, Polarity::Invariant).is_ok();
            self.rollback(snap);
            if ok {
                if matched_idx.is_some() {
                    return Err(()); // ambiguity
                }
                matched_idx = Some(i);
            }
        }

        // Phase 2: apply the unique match (no snapshot — bindings persist).
        let idx = matched_idx.ok_or(())?;
        let rule = &rules[idx];
        let fn_ref = rule.fn_ref;
        let (inst_from, inst_to) = self.instantiate_pair(&rule.from, &rule.to);
        self.unify(sub, &inst_from, Polarity::Invariant)
            .map_err(|_| ())?;
        self.unify(&inst_to, sup, Polarity::Invariant)
            .map_err(|_| ())?;
        self.last_extern_cast = Some(fn_ref);
        Ok(())
    }

    /// Instantiate a polymorphic type: replace all `Param`, `Effect::Var`,
    /// and `Identity::Fresh` in `ty` with fresh values from this substitution.
    ///
    /// Used when calling a generic function: the graph stores a signature
    /// with "template" Params (e.g. `Param(0)`), and each call site gets
    /// its own fresh inference variables via this method.
    /// `Identity::Fresh(id)` is replaced with `Identity::Concrete(new_id)`,
    /// where same `id` within one instantiation → same `new_id`.
    pub fn instantiate(&mut self, ty: &Ty) -> Ty {
        let mut param_map: FxHashMap<ParamToken, ParamToken> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, u32> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        self.instantiate_inner(ty, &mut param_map, &mut effect_map, &mut fresh_map)
    }

    /// Instantiate two types sharing the same Param/Effect/Identity mappings.
    /// Used for CastRule where `from` and `to` share placeholder variables.
    pub fn instantiate_pair(&mut self, a: &Ty, b: &Ty) -> (Ty, Ty) {
        let mut param_map: FxHashMap<ParamToken, ParamToken> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, u32> = FxHashMap::default();
        let mut fresh_map: FxHashMap<IdentityId, IdentityId> = FxHashMap::default();
        let ia = self.instantiate_inner(a, &mut param_map, &mut effect_map, &mut fresh_map);
        let ib = self.instantiate_inner(b, &mut param_map, &mut effect_map, &mut fresh_map);
        (ia, ib)
    }

    fn instantiate_inner(
        &mut self,
        ty: &Ty,
        param_map: &mut FxHashMap<ParamToken, ParamToken>,
        effect_map: &mut FxHashMap<u32, u32>,
        fresh_map: &mut FxHashMap<IdentityId, IdentityId>,
    ) -> Ty {
        match ty {
            Ty::Param { token, constraint } => {
                let new_token = *param_map.entry(*token).or_insert_with(|| {
                    let fresh = ParamToken(self.next_param);
                    self.next_param += 1;
                    fresh
                });
                Ty::Param {
                    token: new_token,
                    constraint: constraint.clone(),
                }
            }
            Ty::List(inner) => Ty::List(Box::new(
                self.instantiate_inner(inner, param_map, effect_map, fresh_map),
            )),
            Ty::Deque(inner, identity) => Ty::Deque(
                Box::new(self.instantiate_inner(inner, param_map, effect_map, fresh_map)),
                Box::new(self.instantiate_inner(identity, param_map, effect_map, fresh_map)),
            ),
            Ty::Identity(identity) => match identity {
                Identity::Fresh(id) => {
                    let new_id = *fresh_map
                        .entry(*id)
                        .or_insert_with(|| self.identity_factory.next());
                    Ty::Identity(Identity::Concrete(new_id))
                }
                Identity::Concrete(_) => ty.clone(),
            },
            Ty::Option(inner) => Ty::Option(Box::new(
                self.instantiate_inner(inner, param_map, effect_map, fresh_map),
            )),
            Ty::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|e| self.instantiate_inner(e, param_map, effect_map, fresh_map))
                    .collect(),
            ),
            Ty::Object(fields) => Ty::Object(
                fields
                    .iter()
                    .map(|(k, v)| {
                        (
                            *k,
                            self.instantiate_inner(v, param_map, effect_map, fresh_map),
                        )
                    })
                    .collect(),
            ),
            Ty::Fn {
                params,
                ret,
                captures,
                effect,
            } => {
                let new_e = self.instantiate_effect(effect, effect_map);
                Ty::Fn {
                    params: params
                        .iter()
                        .map(|p| Param {
                            name: p.name,
                            ty: self.instantiate_inner(&p.ty, param_map, effect_map, fresh_map),
                        })
                        .collect(),
                    ret: Box::new(self.instantiate_inner(ret, param_map, effect_map, fresh_map)),
                    captures: captures
                        .iter()
                        .map(|c| self.instantiate_inner(c, param_map, effect_map, fresh_map))
                        .collect(),
                    effect: new_e,
                }
            }
            Ty::Enum { name, variants } => Ty::Enum {
                name: *name,
                variants: variants
                    .iter()
                    .map(|(tag, payload)| {
                        (
                            *tag,
                            payload.as_ref().map(|ty| {
                                Box::new(
                                    self.instantiate_inner(ty, param_map, effect_map, fresh_map),
                                )
                            }),
                        )
                    })
                    .collect(),
            },
            Ty::UserDefined {
                id,
                type_args,
                effect_args,
                ownership,
            } => Ty::UserDefined {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|t| self.instantiate_inner(t, param_map, effect_map, fresh_map))
                    .collect(),
                effect_args: effect_args
                    .iter()
                    .map(|e| self.instantiate_effect(e, effect_map))
                    .collect(),
                ownership: *ownership,
            },
            Ty::Handle(inner, effect) => {
                let new_e = self.instantiate_effect(effect, effect_map);
                Ty::Handle(
                    Box::new(self.instantiate_inner(inner, param_map, effect_map, fresh_map)),
                    new_e,
                )
            }
            // Concrete types pass through unchanged.
            other => other.clone(),
        }
    }

    fn instantiate_effect(&mut self, e: &Effect, effect_map: &mut FxHashMap<u32, u32>) -> Effect {
        match e {
            Effect::Var(id) => {
                let new_id = *effect_map.entry(*id).or_insert_with(|| {
                    let fresh = self.next_effect;
                    self.next_effect += 1;
                    fresh
                });
                Effect::Var(new_id)
            }
            concrete => concrete.clone(),
        }
    }

    /// Occurs check: returns true if `param` appears in `ty`.
    fn occurs_in(&self, param: ParamToken, ty: &Ty) -> bool {
        match ty {
            Ty::Param { token: p, .. } => {
                if *p == param {
                    return true;
                }
                if let Some(bound) = self.bindings.get(p) {
                    self.occurs_in(param, bound)
                } else {
                    false
                }
            }
            Ty::List(inner) => self.occurs_in(param, inner),
            Ty::Deque(inner, identity) => {
                self.occurs_in(param, inner) || self.occurs_in(param, identity)
            }
            Ty::Identity(_) => false,
            Ty::Option(inner) => self.occurs_in(param, inner),
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(param, e)),
            Ty::Object(fields) => fields.values().any(|v| self.occurs_in(param, v)),
            Ty::Fn {
                params,
                ret,
                captures,
                ..
            } => {
                params.iter().any(|p| self.occurs_in(param, &p.ty))
                    || self.occurs_in(param, ret)
                    || captures.iter().any(|c| self.occurs_in(param, c))
            }
            Ty::Enum { variants, .. } => variants
                .values()
                .any(|p| p.as_ref().is_some_and(|ty| self.occurs_in(param, ty))),
            Ty::UserDefined { type_args, .. } => type_args.iter().any(|t| self.occurs_in(param, t)),
            _ => false,
        }
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
    /// Context variable types, keyed by QualifiedRef.
    pub contexts: FxHashMap<QualifiedRef, Ty>,
    /// Function types, keyed by QualifiedRef.
    pub functions: FxHashMap<QualifiedRef, Ty>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::QualifiedRef;
    use acvus_utils::Interner;

    use Polarity::*;

    /// Test helper: create a non-pure resolved effect with a dummy write target.
    /// Used in place of the removed Effect::self_modifying().
    fn effectful() -> Effect {
        let i = Interner::new();
        Effect::Resolved(EffectSet {
            reads: std::collections::BTreeSet::new(),
            writes: std::collections::BTreeSet::from([EffectTarget::Token(
                QualifiedRef::root(i.intern("__test")),
            )]),
        })
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

    /// Test helper: create a `Ty::UserDefined` with a fresh id and no type/effect args.
    fn test_user_defined() -> Ty {
        Ty::UserDefined {
            id: fresh_qref(),
            type_args: vec![],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        }
    }

    /// Test helper: wrap a `Ty` into a `Param` with a dummy name.
    /// Uses a thread-local interner so all dummy names share the same interner id.
    fn tp(ty: Ty) -> Param {
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
        let mut s = TySubst::new();
        assert!(s.unify(&Ty::Int, &Ty::Int, Invariant).is_ok());
        assert!(s.unify(&Ty::Float, &Ty::Float, Invariant).is_ok());
        assert!(s.unify(&Ty::String, &Ty::String, Invariant).is_ok());
        assert!(s.unify(&Ty::Bool, &Ty::Bool, Invariant).is_ok());
        assert!(s.unify(&Ty::Unit, &Ty::Unit, Invariant).is_ok());
        assert!(s.unify(&Ty::Range, &Ty::Range, Invariant).is_ok());
    }

    #[test]
    fn unify_different_concrete_fails() {
        let mut s = TySubst::new();
        assert!(s.unify(&Ty::Int, &Ty::Float, Invariant).is_err());
        assert!(s.unify(&Ty::String, &Ty::Bool, Invariant).is_err());
    }

    #[test]
    fn unify_var_with_concrete() {
        let mut s = TySubst::new();
        let t = s.fresh_param();
        assert!(s.unify(&t, &Ty::Int, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn unify_deque_of_var() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), Box::new(o.clone()));
        let deque_int = Ty::Deque(Box::new(Ty::Int), Box::new(o.clone()));
        assert!(s.unify(&deque_t, &deque_int, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(
            s.resolve(&deque_t),
            Ty::Deque(Box::new(Ty::Int), Box::new(o))
        );
    }

    #[test]
    fn unify_fn_types() {
        let mut s = TySubst::new();
        let t = s.fresh_param();
        let u = s.fresh_param();
        let fn_tu = Ty::Fn {
            params: vec![tp(t.clone())],
            ret: Box::new(u.clone()),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_int_bool = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_tu, &fn_int_bool, Covariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&u), Ty::Bool);
    }

    #[test]
    fn unify_fn_arity_mismatch() {
        let mut s = TySubst::new();
        let fn1 = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn2 = Ty::Fn {
            params: vec![tp(Ty::Int), tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn1, &fn2, Invariant).is_err());
    }

    #[test]
    fn unify_object() {
        let mut s = TySubst::new();
        let interner = Interner::new();
        let t = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), t.clone()),
        ]));
        let obj2 = Ty::Object(FxHashMap::from_iter([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), Ty::Int),
        ]));
        assert!(s.unify(&obj1, &obj2, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn unify_object_key_mismatch() {
        let mut s = TySubst::new();
        let interner = Interner::new();
        let obj1 = Ty::Object(FxHashMap::from_iter([(
            interner.intern("name"),
            Ty::String,
        )]));
        let obj2 = Ty::Object(FxHashMap::from_iter([(interner.intern("age"), Ty::Int)]));
        assert!(s.unify(&obj1, &obj2, Invariant).is_err());
    }

    #[test]
    fn occurs_check() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), Box::new(o));
        // T = Deque<T, O> should fail
        assert!(s.unify(&t, &deque_t, Invariant).is_err());
    }

    #[test]
    fn transitive_resolution() {
        let mut s = TySubst::new();
        let t1 = s.fresh_param();
        let t2 = s.fresh_param();
        assert!(s.unify(&t1, &t2, Invariant).is_ok());
        assert!(s.unify(&t2, &Ty::String, Invariant).is_ok());
        assert_eq!(s.resolve(&t1), Ty::String);
    }

    // -- Object merge tests --

    #[test]
    fn unify_object_disjoint_via_var() {
        // Var → {a} then Var → {b} should merge to {a, b}
        let mut s = TySubst::new();
        let i = Interner::new();
        let v = s.fresh_param();
        let obj_a = Ty::Object(FxHashMap::from_iter([(i.intern("a"), Ty::Int)]));
        let obj_b = Ty::Object(FxHashMap::from_iter([(i.intern("b"), Ty::String)]));
        assert!(s.unify(&v, &obj_a, Invariant).is_ok());
        assert!(s.unify(&v, &obj_b, Invariant).is_ok());
        let resolved = s.resolve(&v);
        match &resolved {
            Ty::Object(fields) => {
                assert_eq!(fields.len(), 2, "expected {{a, b}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&Ty::Int));
                assert_eq!(fields.get(&i.intern("b")), Some(&Ty::String));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlapping_via_var() {
        // Var → {a, b} then Var → {b, c} should merge to {a, b, c}
        let mut s = TySubst::new();
        let i = Interner::new();
        let v = s.fresh_param();
        let obj_ab = Ty::Object(FxHashMap::from_iter([
            (i.intern("a"), Ty::Int),
            (i.intern("b"), Ty::String),
        ]));
        let obj_bc = Ty::Object(FxHashMap::from_iter([
            (i.intern("b"), Ty::String),
            (i.intern("c"), Ty::Bool),
        ]));
        assert!(s.unify(&v, &obj_ab, Invariant).is_ok());
        assert!(s.unify(&v, &obj_bc, Invariant).is_ok());
        let resolved = s.resolve(&v);
        match &resolved {
            Ty::Object(fields) => {
                assert_eq!(fields.len(), 3, "expected {{a, b, c}}, got {fields:?}");
                assert_eq!(fields.get(&i.intern("a")), Some(&Ty::Int));
                assert_eq!(fields.get(&i.intern("b")), Some(&Ty::String));
                assert_eq!(fields.get(&i.intern("c")), Some(&Ty::Bool));
            }
            other => panic!("expected Object, got {other:?}"),
        }
    }

    #[test]
    fn unify_object_overlap_type_conflict_fails() {
        // {b: Int} and {b: String} via same Var should fail
        let mut s = TySubst::new();
        let i = Interner::new();
        let v = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([(i.intern("b"), Ty::Int)]));
        let obj2 = Ty::Object(FxHashMap::from_iter([(i.intern("b"), Ty::String)]));
        assert!(s.unify(&v, &obj1, Invariant).is_ok());
        assert!(s.unify(&v, &obj2, Invariant).is_err());
    }

    // -- Deque type tests --

    #[test]
    fn unify_deque_same_identity() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let t = s.fresh_param();
        let d1 = Ty::Deque(Box::new(t.clone()), Box::new(o.clone()));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o.clone()));
        assert!(s.unify(&d1, &d2, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&d1), Ty::Deque(Box::new(Ty::Int), Box::new(o)));
    }

    #[test]
    fn unify_deque_different_concrete_identity_fails() {
        // Invariant: different concrete identities → error
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        assert_ne!(o1, o2);
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o2));
        assert!(
            s.unify(&d1, &d2, Invariant).is_err(),
            "different concrete identities must not unify in Invariant"
        );
    }

    #[test]
    fn unify_deque_identity_param_binds_to_concrete() {
        // Param should bind to Identity::Concrete during unification
        let mut s = TySubst::new();
        let concrete = s.alloc_identity(false);
        let var = s.fresh_param(); // identity variable
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(concrete.clone()));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(var.clone()));
        assert!(
            s.unify(&d1, &d2, Invariant).is_ok(),
            "identity Param should bind to Concrete"
        );
        assert_eq!(s.resolve(&var), concrete);
    }

    #[test]
    fn unify_deque_identity_param_preserves_identity() {
        // Two Deques through same identity Param should resolve to same concrete identity
        let mut s = TySubst::new();
        let concrete = s.alloc_identity(false);
        let var = s.fresh_param();
        let d_concrete = Ty::Deque(Box::new(Ty::Int), Box::new(concrete));
        let d_var = Ty::Deque(Box::new(Ty::Int), Box::new(var.clone()));
        assert!(s.unify(&d_concrete, &d_var, Invariant).is_ok());
        // Now a second concrete identity should NOT match the same var
        let concrete2 = s.alloc_identity(false);
        let d_concrete2 = Ty::Deque(Box::new(Ty::Int), Box::new(concrete2));
        let d_var2 = Ty::Deque(Box::new(Ty::Int), Box::new(var));
        assert!(
            s.unify(&d_concrete2, &d_var2, Invariant).is_err(),
            "var already bound to different concrete"
        );
    }

    #[test]
    fn unify_deque_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o.clone()));
        let d2 = Ty::Deque(Box::new(Ty::String), Box::new(o));
        assert!(
            s.unify(&d1, &d2, Invariant).is_err(),
            "inner type mismatch with same identity must fail"
        );
    }

    #[test]
    fn fresh_param_produces_unique_ids() {
        let mut s = TySubst::new();
        let o1 = s.fresh_param();
        let o2 = s.fresh_param();
        let o3 = s.fresh_param();
        assert_ne!(o1, o2);
        assert_ne!(o2, o3);
        assert_ne!(o1, o3);
    }

    #[test]
    fn resolve_deque_propagates_inner() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let t = s.fresh_param();
        assert!(s.unify(&t, &Ty::String, Invariant).is_ok());
        let deque = Ty::Deque(Box::new(t.clone()), Box::new(o.clone()));
        assert_eq!(
            s.resolve(&deque),
            Ty::Deque(Box::new(Ty::String), Box::new(o))
        );
    }

    #[test]
    fn occurs_in_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), Box::new(o));
        // T = Deque<T, O> should fail (occurs check)
        assert!(s.unify(&t, &deque_t, Invariant).is_err());
    }

    #[test]
    fn snapshot_rollback_preserves_identity_counter() {
        let mut s = TySubst::new();
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
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(
            s.unify(&d, &l, Covariant).is_ok(),
            "Deque should coerce to List"
        );
    }

    #[test]
    fn unify_list_does_not_coerce_to_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        assert!(
            s.unify(&l, &d, Covariant).is_err(),
            "List must not coerce to Deque"
        );
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn deque_identity_mismatch_covariant_demotes_to_list() {
        // Covariant: Deque+Deque identity mismatch → List demotion
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o2));
        // Bind v to d1, then unify v with d2 in Covariant → should demote to List
        assert!(s.unify(&v, &d1, Covariant).is_ok());
        assert!(s.unify(&v, &d2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert_eq!(
            resolved,
            Ty::List(Box::new(Ty::Int)),
            "should demote to List<Int>"
        );
    }

    #[test]
    fn deque_identity_mismatch_invariant_fails() {
        // Invariant: Deque+Deque identity mismatch → error
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o2));
        assert!(s.unify(&d1, &d2, Invariant).is_err());
    }

    #[test]
    fn deque_coerces_to_list_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l, Covariant).is_ok());
    }

    #[test]
    fn list_does_not_coerce_to_deque_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        assert!(s.unify(&l, &d, Covariant).is_err());
    }

    #[test]
    fn contravariant_list_deque_ok() {
        // Contravariant: (List, Deque) → reversed: Deque ≤ List → OK
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        assert!(s.unify(&l, &d, Contravariant).is_ok());
    }

    #[test]
    fn contravariant_deque_list_fails() {
        // Contravariant: (Deque, List) → reversed: List ≤ Deque → invalid
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l, Contravariant).is_err());
    }

    #[test]
    fn invariant_deque_list_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_param();
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l, Invariant).is_err());
        assert!(s.unify(&l, &d, Invariant).is_err());
    }

    #[test]
    fn fn_param_contravariant_ret_covariant() {
        // Fn(List<Int>) -> Deque<Int> ≤ Fn(Deque<Int>) -> List<Int> in Covariant
        // params flip: Deque ≤ List OK (contravariant)
        // ret keeps: Deque ≤ List OK (covariant)
        let mut s = TySubst::new();
        let o1 = s.fresh_param();
        let o2 = s.fresh_param();
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o2)))],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_ok());
    }

    #[test]
    fn list_literal_mixed_deque_identities() {
        // Simulates: multiple Deque elements with different identities → List demotion
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let elem_var = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::String), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::String), Box::new(o2));
        // First element sets the type
        assert!(s.unify(&elem_var, &d1, Covariant).is_ok());
        // Second element with different identity → demotion
        assert!(s.unify(&elem_var, &d2, Covariant).is_ok());
        let resolved = s.resolve(&elem_var);
        assert_eq!(resolved, Ty::List(Box::new(Ty::String)));
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
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let o3 = s.alloc_identity(false);
        let v = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o2));
        let d3 = Ty::Deque(Box::new(Ty::Int), Box::new(o3));
        assert!(s.unify(&d1, &v, Covariant).is_ok());
        assert!(
            s.unify(&d2, &v, Covariant).is_ok(),
            "second deque should trigger demotion"
        );
        // v is now List<Int>. Third deque: Deque≤List in Covariant should succeed.
        assert!(
            s.unify(&d3, &v, Covariant).is_ok(),
            "third deque should coerce to List via Deque≤List"
        );
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn demotion_then_list_unifies() {
        // After demotion to List, unifying with another List should succeed.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(o1)), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(o2)), Covariant)
                .is_ok()
        );
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn demotion_then_deque_same_inner_type_via_var() {
        // After demotion, the Var-resolved List should accept further Deque coercion
        // even when inner type is a Var that later resolves.
        // arg order: (new_elem, join_accum) → new ≤ existing.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let o3 = s.alloc_identity(false);
        let inner_var = s.fresh_param();
        let v = s.fresh_param();
        assert!(
            s.unify(
                &Ty::Deque(Box::new(inner_var.clone()), Box::new(o1)),
                &v,
                Covariant
            )
            .is_ok()
        );
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), Box::new(o2)), &v, Covariant)
                .is_ok()
        );
        // inner_var should now be Int, v should be List<Int>
        assert_eq!(s.resolve(&inner_var), Ty::Int);
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        // Third deque with same inner type
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), Box::new(o3)), &v, Covariant)
                .is_ok()
        );
    }

    #[test]
    fn concrete_deque_deque_covariant_no_var_no_rebind() {
        // Two concrete Deques (no Var backing) with mismatched identities.
        // No Param to rebind → LUB cannot be applied → Err.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::Int), Box::new(o2));
        assert!(
            s.unify(&d1, &d2, Covariant).is_err(),
            "concrete Deque identity mismatch with no Param should fail"
        );
    }

    #[test]
    fn concrete_deque_deque_inner_mismatch_plus_identity_mismatch() {
        // Both inner type AND identity mismatch — inner unify should fail first.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let d1 = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let d2 = Ty::Deque(Box::new(Ty::String), Box::new(o2));
        assert!(
            s.unify(&d1, &d2, Covariant).is_err(),
            "inner type mismatch must fail regardless of demotion"
        );
    }

    #[test]
    fn demotion_inner_type_still_var() {
        // Demotion when inner type is an unresolved Var — should resolve to List<Var>.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let inner = s.fresh_param();
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(inner.clone()), Box::new(o1)),
                Covariant
            )
            .is_ok()
        );
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(inner.clone()), Box::new(o2)),
                Covariant
            )
            .is_ok()
        );
        // v should be List<inner_var>, inner still unresolved
        let resolved = s.resolve(&v);
        assert!(
            matches!(resolved, Ty::List(_)),
            "should be List, got {resolved:?}"
        );
        // Now bind inner to String
        assert!(s.unify(&inner, &Ty::String, Invariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::String)));
    }

    #[test]
    fn contravariant_demotion() {
        // Contravariant: Deque+Deque identity mismatch also demotes (pol != Invariant).
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o1)),
                Contravariant
            )
            .is_ok()
        );
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o2)),
                Contravariant
            )
            .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn object_field_deque_coercion_covariant() {
        // {tags: Deque<String, o1>} vs {tags: List<String>} in Covariant.
        // Object field polarity is passed through → Deque≤List OK.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let obj_deque = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), Box::new(o)),
        )]));
        let obj_list = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::List(Box::new(Ty::String)),
        )]));
        assert!(s.unify(&obj_deque, &obj_list, Covariant).is_ok());
    }

    #[test]
    fn object_field_deque_coercion_invariant_fails() {
        // Same as above but Invariant — must fail.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let obj_deque = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), Box::new(o)),
        )]));
        let obj_list = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::List(Box::new(Ty::String)),
        )]));
        assert!(s.unify(&obj_deque, &obj_list, Invariant).is_err());
    }

    #[test]
    fn object_field_deque_identity_mismatch_demotion() {
        // {tags: Deque<S, o1>} vs {tags: Deque<S, o2>} in Covariant.
        // Inner Deque identity mismatch → demoted to List within the field.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), Box::new(o1)),
        )]));
        let obj2 = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), Box::new(o2)),
        )]));
        assert!(s.unify(&v, &obj1, Covariant).is_ok());
        assert!(s.unify(&v, &obj2, Covariant).is_ok());
    }

    #[test]
    fn option_deque_to_list_covariant_fails() {
        // Option<Deque<Int>> vs Option<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside Option is a type error.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let opt_deque = Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o))));
        let opt_list = Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&opt_deque, &opt_list, Covariant).is_err());
    }

    #[test]
    fn option_deque_to_list_invariant_fails() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let opt_deque = Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o))));
        let opt_list = Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&opt_deque, &opt_list, Invariant).is_err());
    }

    #[test]
    fn tuple_deque_coercion_covariant() {
        // (Deque<Int>, String) vs (List<Int>, String) in Covariant.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let t1 = Ty::Tuple(vec![Ty::Deque(Box::new(Ty::Int), Box::new(o)), Ty::String]);
        let t2 = Ty::Tuple(vec![Ty::List(Box::new(Ty::Int)), Ty::String]);
        assert!(s.unify(&t1, &t2, Covariant).is_ok());
    }

    #[test]
    fn tuple_deque_coercion_invariant_fails() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let t1 = Ty::Tuple(vec![Ty::Deque(Box::new(Ty::Int), Box::new(o)), Ty::String]);
        let t2 = Ty::Tuple(vec![Ty::List(Box::new(Ty::Int)), Ty::String]);
        assert!(s.unify(&t1, &t2, Invariant).is_err());
    }

    #[test]
    fn double_flip_restores_covariant() {
        // Fn(Fn(Deque) -> Unit) -> Unit  vs  Fn(Fn(List) -> Unit) -> Unit
        // Outer Covariant → param flips to Contravariant → inner param flips back to Covariant.
        // So inner param: Deque vs List in Covariant → Deque≤List OK.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let inner_fn_a = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let inner_fn_b = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let outer_a = Ty::Fn {
            params: vec![tp(inner_fn_a)],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let outer_b = Ty::Fn {
            params: vec![tp(inner_fn_b)],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&outer_a, &outer_b, Covariant).is_ok());
    }

    #[test]
    fn double_flip_wrong_direction_fails() {
        // Fn(Fn(List) -> Unit) -> Unit  vs  Fn(Fn(Deque) -> Unit) -> Unit
        // Double flip = Covariant → inner param: List vs Deque in Covariant → List≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let inner_fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let inner_fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let outer_a = Ty::Fn {
            params: vec![tp(inner_fn_a)],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let outer_b = Ty::Fn {
            params: vec![tp(inner_fn_b)],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&outer_a, &outer_b, Covariant).is_err());
    }

    #[test]
    fn fn_ret_list_to_deque_covariant_fails() {
        // Fn() -> List<Int>  vs  Fn() -> Deque<Int, O>  in Covariant.
        // ret keeps polarity → List≤Deque invalid.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o))),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_err());
    }

    #[test]
    fn fn_param_deque_to_list_covariant_fails() {
        // Fn(Deque<Int>) -> Unit  vs  Fn(List<Int>) -> Unit  in Covariant.
        // param flips → Contravariant: Deque vs List → (Deque, List) in Contra → reversed: List≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_err());
    }

    #[test]
    fn fn_param_list_to_deque_covariant_ok() {
        // Fn(List<Int>) -> Unit  vs  Fn(Deque<Int>) -> Unit  in Covariant.
        // param flips → Contra: List vs Deque → (List, Deque) in Contra → reversed: Deque≤List → OK.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_ok());
    }

    #[test]
    fn snapshot_rollback_undoes_demotion() {
        // Demotion should be fully undone by rollback.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o1.clone())),
                Covariant
            )
            .is_ok()
        );
        let snap = s.snapshot();
        let o2 = s.alloc_identity(false);
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(o2)), Covariant)
                .is_ok()
        );
        assert_eq!(
            s.resolve(&v),
            Ty::List(Box::new(Ty::Int)),
            "demoted after second deque"
        );
        s.rollback(snap);
        // After rollback, v should be back to Deque<Int, o1>
        assert_eq!(
            s.resolve(&v),
            Ty::Deque(Box::new(Ty::Int), Box::new(o1)),
            "rollback should undo demotion"
        );
    }

    #[test]
    fn nested_list_of_deque_coercion_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside List is a type error.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let a = Ty::List(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o))));
        let b = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn nested_list_of_deque_invariant_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Invariant.
        // Inner: Deque vs List in Invariant → fails.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let a = Ty::List(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o))));
        let b = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn enum_variant_deque_coercion_covariant() {
        // Enum with Deque payload vs same enum with List payload, Covariant.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o)))),
            )]),
        };
        let e2 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(tag, Some(Box::new(Ty::List(Box::new(Ty::Int)))))]),
        };
        assert!(s.unify(&e1, &e2, Covariant).is_ok());
    }

    #[test]
    fn enum_variant_deque_coercion_invariant_fails() {
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.alloc_identity(false);
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o)))),
            )]),
        };
        let e2 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(tag, Some(Box::new(Ty::List(Box::new(Ty::Int)))))]),
        };
        assert!(s.unify(&e1, &e2, Invariant).is_err());
    }

    // ================================================================
    // Var chain + coercion 상호작용
    // ================================================================

    #[test]
    fn var_chain_coercion_propagates() {
        // Var1 → Var2 → Deque(o1), then unify Var1 with List → Deque ≤ List via chain.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), Box::new(o)), Invariant)
                .is_ok()
        );
        // v1 → v2 → Deque(Int, o). Now v1 as Deque ≤ List.
        assert!(
            s.unify(&v1, &Ty::List(Box::new(Ty::Int)), Covariant)
                .is_ok()
        );
    }

    #[test]
    fn var_chain_demotion_rebinds_leaf() {
        // Var1 → Var2 → Deque(o1). Unify Var1 with Deque(o2) covariant → demotion.
        // find_leaf_param should follow chain and rebind Var2 (the leaf).
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), Box::new(o1)), Invariant)
                .is_ok()
        );
        // Demotion via v1
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), Box::new(o2)), &v1, Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v1), Ty::List(Box::new(Ty::Int)));
        assert_eq!(s.resolve(&v2), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn two_vars_sharing_deque_demotion_affects_both() {
        // Chain v2 → v1 while both unbound, THEN bind v1 → Deque(o1).
        // Demote via v2 → find_leaf_param follows v2 → v1 → rebinds v1 to List.
        // Both Var1 and Var2 should resolve to List.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        // Must chain BEFORE binding to concrete — otherwise shallow_resolve
        // flattens the chain and v2 binds directly to Deque, not to v1.
        assert!(s.unify(&v2, &v1, Invariant).is_ok());
        assert!(
            s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), Box::new(o1)), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), Box::new(o2)), &v2, Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v1), Ty::List(Box::new(Ty::Int)));
        assert_eq!(s.resolve(&v2), Ty::List(Box::new(Ty::Int)));
    }

    // ================================================================
    // Occurs check + polarity
    // ================================================================

    #[test]
    fn occurs_check_through_list_covariant() {
        // Var = List<Var> should fail (occurs) regardless of polarity.
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let cyclic = Ty::List(Box::new(v.clone()));
        assert!(s.unify(&v, &cyclic, Covariant).is_err());
    }

    #[test]
    fn occurs_check_through_deque_covariant() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_param();
        let cyclic = Ty::Deque(Box::new(v.clone()), Box::new(o));
        assert!(s.unify(&v, &cyclic, Covariant).is_err());
    }

    #[test]
    fn occurs_check_through_fn_ret_covariant() {
        // Var = Fn() -> Var should fail (occurs) in any polarity.
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let cyclic = Ty::Fn {
            params: vec![],
            ret: Box::new(v.clone()),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&v, &cyclic, Covariant).is_err());
    }

    // ================================================================
    // Deep nesting coercion
    // ================================================================

    #[test]
    fn nested_deque_in_deque_coercion() {
        // Deque<Deque<Int, o1>, o2> vs Deque<List<Int>, o2> in Covariant.
        // Inner item type is invariant — Deque vs List inside Deque is a type error.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let a = Ty::Deque(
            Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),
            Box::new(o2.clone()),
        );
        let b = Ty::Deque(Box::new(Ty::List(Box::new(Ty::Int))), Box::new(o2));
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn nested_deque_in_deque_invariant_inner_coercion_fails() {
        // Same structure but Invariant → inner Deque vs List fails.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let a = Ty::Deque(
            Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),
            Box::new(o2.clone()),
        );
        let b = Ty::Deque(Box::new(Ty::List(Box::new(Ty::Int))), Box::new(o2));
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn deeply_nested_option_option_deque_covariant_fails() {
        // Option<Option<Deque<Int>>> vs Option<Option<List<Int>>> in Covariant.
        // Inner item type is invariant — nested coercion is a type error.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let a = Ty::Option(Box::new(Ty::Option(Box::new(Ty::Deque(
            Box::new(Ty::Int),
            Box::new(o),
        )))));
        let b = Ty::Option(Box::new(Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))))));
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn list_of_fn_with_coercion_in_param_and_ret_fails() {
        // List<Fn(List<Int>) -> Deque<Int>>  vs  List<Fn(Deque<Int>) -> List<Int>>
        // in Covariant.
        // Inner item type is invariant — Fn types with different param/ret don't match inside List.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), Box::new(o2)))],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
        };
        let a = Ty::List(Box::new(fn_a));
        let b = Ty::List(Box::new(fn_b));
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    // ================================================================
    // Object merge + coercion 동시 발생
    // ================================================================

    #[test]
    fn object_merge_plus_inner_demotion() {
        // Var → {a: Deque(o1)} then Var → {a: Deque(o2), b: Int}.
        // Merge adds field b, inner field a triggers demotion.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([(
            i.intern("a"),
            Ty::Deque(Box::new(Ty::Int), Box::new(o1)),
        )]));
        let obj2 = Ty::Object(FxHashMap::from_iter([
            (i.intern("a"), Ty::Deque(Box::new(Ty::Int), Box::new(o2))),
            (i.intern("b"), Ty::Int),
        ]));
        assert!(s.unify(&v, &obj1, Covariant).is_ok());
        assert!(s.unify(&v, &obj2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match &resolved {
            Ty::Object(fields) => {
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
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o1.clone())),
                Invariant
            )
            .is_ok()
        );

        let snap = s.snapshot();
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), Box::new(o2)), &v, Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        s.rollback(snap);

        // After rollback, v is still Deque(o1). Same-identity unify should work.
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o1.clone())),
                Invariant
            )
            .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), Box::new(o1)));
    }

    // ================================================================
    // Polarity symmetry / duality 검증
    // ================================================================

    #[test]
    fn covariant_ab_equals_contravariant_ba() {
        // If unify(a, b, Cov) succeeds then unify(b, a, Contra) must also succeed.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o1 = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false); // keep counter in sync
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s1.unify(&d, &l, Covariant).is_ok());
        assert!(s2.unify(&l, &d, Contravariant).is_ok());
    }

    #[test]
    fn covariant_ab_fail_equals_contravariant_ba_fail() {
        // If unify(a, b, Cov) fails then unify(b, a, Contra) must also fail.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o1 = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false);
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o1));
        assert!(s1.unify(&l, &d, Covariant).is_err()); // List ≤ Deque: no
        assert!(s2.unify(&d, &l, Contravariant).is_err()); // reversed: List ≤ Deque: no
    }

    #[test]
    fn invariant_symmetric() {
        // Invariant: unify(a, b) and unify(b, a) must both fail/succeed equally.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o = s1.alloc_identity(false);
        let _ = s2.alloc_identity(false);
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s1.unify(&d, &l, Invariant).is_err());
        assert!(s2.unify(&l, &d, Invariant).is_err());
    }

    #[test]
    fn invariant_same_types_both_directions() {
        // Same concrete type: Invariant must succeed regardless of order.
        let mut s = TySubst::new();
        let l1 = Ty::List(Box::new(Ty::Int));
        let l2 = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&l1, &l2, Invariant).is_ok());
        assert!(s.unify(&l2, &l1, Invariant).is_ok());
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
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![
                tp(Ty::List(Box::new(Ty::Int))),
                tp(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),
            ],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![
                tp(Ty::Deque(Box::new(Ty::Int), Box::new(o2))),
                tp(Ty::List(Box::new(Ty::Int))),
            ],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_err());
    }

    #[test]
    fn fn_multi_param_all_ok() {
        // Fn(List, List) -> Unit  vs  Fn(Deque, Deque) -> Unit  in Covariant.
        // param flip → Contra: both (List, Deque) → Deque≤List OK.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        let fn_a = Ty::Fn {
            params: vec![
                tp(Ty::List(Box::new(Ty::Int))),
                tp(Ty::List(Box::new(Ty::Int))),
            ],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![
                tp(Ty::Deque(Box::new(Ty::Int), Box::new(o1))),
                tp(Ty::Deque(Box::new(Ty::Int), Box::new(o2))),
            ],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_ok());
    }

    // ================================================================
    // Unresolved Var containers + coercion
    // ================================================================

    #[test]
    fn deque_var_inner_coerces_to_list_var_inner() {
        // Deque<Var1, O> vs List<Var2> in Covariant → Deque≤List OK, Var1 binds to Var2.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        let d = Ty::Deque(Box::new(v1.clone()), Box::new(o));
        let l = Ty::List(Box::new(v2.clone()));
        assert!(s.unify(&d, &l, Covariant).is_ok());
        // Bind v2 to String → v1 should follow.
        assert!(s.unify(&v2, &Ty::String, Invariant).is_ok());
        assert_eq!(s.resolve(&v1), Ty::String);
    }

    // ================================================================
    // Bidirectional Var binding + coercion
    // ================================================================

    #[test]
    fn two_vars_coerce_deque_to_list() {
        // Var1 = Deque(Int, o1), Var2 = List(Int).
        // unify(Var1, Var2, Cov) → Deque≤List → OK.
        // After: Var1 still resolves to Deque (binding unchanged), Var2 still List.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(
            s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), Box::new(o)), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&v2, &Ty::List(Box::new(Ty::Int)), Invariant)
                .is_ok()
        );
        assert!(s.unify(&v1, &v2, Covariant).is_ok());
    }

    #[test]
    fn two_vars_coerce_list_to_deque_covariant_fails() {
        // Var1 = List(Int), Var2 = Deque(Int, o).
        // unify(Var1, Var2, Cov) → List≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(
            s.unify(&v1, &Ty::List(Box::new(Ty::Int)), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), Box::new(o)), Invariant)
                .is_ok()
        );
        assert!(s.unify(&v1, &v2, Covariant).is_err());
    }

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    #[test]
    fn five_deque_identities_join_to_list() {
        // [d1, d2, d3, d4, d5] each with distinct identity → all join to List.
        let mut s = TySubst::new();
        let v = s.fresh_param();
        for _ in 0..5 {
            let o = s.alloc_identity(false);
            let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
            assert!(s.unify(&d, &v, Covariant).is_ok());
        }
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    // ================================================================
    // Mixed concrete/param identities
    // ================================================================

    #[test]
    fn identity_param_binds_then_mismatch_demotes() {
        // Param identity Deque binds to concrete identity, then another concrete → mismatch → demotion.
        let mut s = TySubst::new();
        let c1 = s.alloc_identity(false);
        let c2 = s.alloc_identity(false);
        let ov = s.fresh_param(); // identity variable
        let v = s.fresh_param();
        let d_var_identity = Ty::Deque(Box::new(Ty::Int), Box::new(ov.clone()));
        let d_c1 = Ty::Deque(Box::new(Ty::Int), Box::new(c1.clone()));
        let d_c2 = Ty::Deque(Box::new(Ty::Int), Box::new(c2));
        // Bind Param identity via d_var_identity = d_c1
        assert!(s.unify(&v, &d_var_identity, Invariant).is_ok());
        assert!(s.unify(&v, &d_c1, Invariant).is_ok());
        assert_eq!(s.resolve(&ov), c1);
        // Now d_c2 has different identity → demotion in Covariant
        assert!(s.unify(&d_c2, &v, Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    #[test]
    fn error_absorbs_any_polarity() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        assert!(s.unify(&Ty::error(), &d, Covariant).is_ok());
        assert!(s.unify(&d, &Ty::error(), Contravariant).is_ok());
        assert!(
            s.unify(&Ty::error(), &Ty::List(Box::new(Ty::Int)), Invariant)
                .is_ok()
        );
    }

    #[test]
    fn param_absorbs_any_polarity() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let p1 = s.fresh_param();
        let p2 = s.fresh_param();
        let p3 = s.fresh_param();
        assert!(s.unify(&p1, &d, Covariant).is_ok());
        assert!(s.unify(&d, &p2, Contravariant).is_ok());
        assert!(
            s.unify(&p3, &Ty::List(Box::new(Ty::Int)), Invariant)
                .is_ok()
        );
    }

    // ================================================================
    // Transitive coercion chains
    // ================================================================

    #[test]
    fn list_cannot_narrow_back_to_deque_covariant() {
        // Var = List(Int). List ≤ Deque is invalid.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(o)), Covariant)
                .is_err()
        );
    }

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    #[test]
    fn deque_to_list_inner_type_mismatch_fails() {
        // Deque<Int> ≤ List<String> → inner Int vs String fails.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        assert!(
            s.unify(
                &Ty::Deque(Box::new(Ty::Int), Box::new(o)),
                &Ty::List(Box::new(Ty::String)),
                Covariant,
            )
            .is_err()
        );
    }

    #[test]
    fn demotion_inner_type_mismatch_fails() {
        // Deque<Int, o1> vs Deque<String, o2> in Covariant.
        // Identity mismatch triggers demotion path, but inner unify Int vs String fails first.
        let mut s = TySubst::new();
        let o1 = s.alloc_identity(false);
        let o2 = s.alloc_identity(false);
        assert!(
            s.unify(
                &Ty::Deque(Box::new(Ty::Int), Box::new(o1)),
                &Ty::Deque(Box::new(Ty::String), Box::new(o2)),
                Covariant,
            )
            .is_err()
        );
    }

    // ================================================================
    // Coercion does NOT propagate across unrelated type constructors
    // ================================================================

    #[test]
    fn deque_vs_option_fails_any_polarity() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let d = Ty::Deque(Box::new(Ty::Int), Box::new(o));
        let opt = Ty::Option(Box::new(Ty::Int));
        assert!(s.unify(&d, &opt, Covariant).is_err());
        assert!(s.unify(&d, &opt, Contravariant).is_err());
        assert!(s.unify(&d, &opt, Invariant).is_err());
    }

    #[test]
    fn list_vs_tuple_fails_any_polarity() {
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let t = Ty::Tuple(vec![Ty::Int]);
        assert!(s.unify(&l, &t, Covariant).is_err());
        assert!(s.unify(&l, &t, Invariant).is_err());
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
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let mk = |inner_param: Ty| -> Ty {
            Ty::Fn {
                params: vec![tp(Ty::Fn {
                    params: vec![tp(Ty::Fn {
                        params: vec![tp(inner_param)],
                        ret: Box::new(Ty::Unit),

                        effect: Effect::pure(),
                        captures: vec![],
                    })],
                    ret: Box::new(Ty::Unit),

                    effect: Effect::pure(),
                    captures: vec![],
                })],
                ret: Box::new(Ty::Unit),

                effect: Effect::pure(),
                captures: vec![],
            }
        };
        let a = mk(Ty::Deque(Box::new(Ty::Int), Box::new(o)));
        let b = mk(Ty::List(Box::new(Ty::Int)));
        // 3 flips from Covariant → Contra. (Deque, List) in Contra → fail.
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn triple_flip_reversed_succeeds() {
        // Same structure but (List, Deque) at innermost.
        // 3 flips → Contra. (List, Deque) in Contra → reversed: Deque≤List → OK.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let mk = |inner_param: Ty| -> Ty {
            Ty::Fn {
                params: vec![tp(Ty::Fn {
                    params: vec![tp(Ty::Fn {
                        params: vec![tp(inner_param)],
                        ret: Box::new(Ty::Unit),

                        effect: Effect::pure(),
                        captures: vec![],
                    })],
                    ret: Box::new(Ty::Unit),

                    effect: Effect::pure(),
                    captures: vec![],
                })],
                ret: Box::new(Ty::Unit),

                effect: Effect::pure(),
                captures: vec![],
            }
        };
        let a = mk(Ty::List(Box::new(Ty::Int)));
        let b = mk(Ty::Deque(Box::new(Ty::Int), Box::new(o)));
        assert!(s.unify(&a, &b, Covariant).is_ok());
    }

    // ================================================================
    // Regression: Deque with same identity must not trigger demotion
    // ================================================================

    #[test]
    fn same_identity_no_demotion_even_covariant() {
        // Same identity → identities unify → no demotion path, stays Deque.
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o.clone())),
                Covariant
            )
            .is_ok()
        );
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(o.clone())),
                Covariant
            )
            .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), Box::new(o)));
    }

    #[test]
    fn same_identity_param_no_demotion() {
        // Identity Param binds to concrete. Second use with same Param → same concrete → no demotion.
        let mut s = TySubst::new();
        let c = s.alloc_identity(false);
        let ov = s.fresh_param();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(ov)), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(
                &v,
                &Ty::Deque(Box::new(Ty::Int), Box::new(c.clone())),
                Covariant
            )
            .is_ok()
        );
        // ov now bound to c. Second concrete same as c → no mismatch.
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), Box::new(c)), Covariant)
                .is_ok()
        );
        // Still Deque, not List.
        let resolved = s.resolve(&v);
        assert!(
            matches!(resolved, Ty::Deque(_, _)),
            "should stay Deque, got {resolved:?}"
        );
    }

    // ── Sequence identity tracking ─────────────────────────────────

    // ── UserDefined unification tests ───────────────────────────────

    fn ud(id: QualifiedRef, type_args: Vec<Ty>, effect_args: Vec<Effect>) -> Ty {
        Ty::UserDefined {
            id,
            type_args,
            effect_args,
            ownership: Ownership::MoveOnly,
        }
    }

    // -- Completeness: valid UserDefined unifications --

    #[test]
    fn user_defined_same_id_empty_args_unifies() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        assert!(
            s.unify(&ud(id, vec![], vec![]), &ud(id, vec![], vec![]), Invariant)
                .is_ok()
        );
    }

    #[test]
    fn user_defined_same_id_concrete_type_args_unifies() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        assert!(
            s.unify(
                &ud(id, vec![Ty::Int], vec![]),
                &ud(id, vec![Ty::Int], vec![]),
                Invariant,
            )
            .is_ok()
        );
    }

    #[test]
    fn user_defined_param_type_arg_resolved_via_unify() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        let p = s.fresh_param();
        assert!(
            s.unify(
                &ud(id, vec![p.clone()], vec![]),
                &ud(id, vec![Ty::Int], vec![]),
                Invariant,
            )
            .is_ok()
        );
        assert_eq!(s.resolve(&p), Ty::Int);
    }

    #[test]
    fn user_defined_effect_var_resolved_via_unify() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        let e = s.fresh_effect_var();
        assert!(
            s.unify(
                &ud(id, vec![], vec![e.clone()]),
                &ud(id, vec![], vec![Effect::pure()]),
                Covariant,
            )
            .is_ok()
        );
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    #[test]
    fn user_defined_nested_type_arg_unifies() {
        // UserDefined<List<Param>> vs UserDefined<List<Int>> → resolves Param to Int
        let mut s = TySubst::new();
        let id = fresh_qref();
        let p = s.fresh_param();
        assert!(
            s.unify(
                &ud(id, vec![Ty::List(Box::new(p.clone()))], vec![]),
                &ud(id, vec![Ty::List(Box::new(Ty::Int))], vec![]),
                Invariant,
            )
            .is_ok()
        );
        assert_eq!(s.resolve(&p), Ty::Int);
    }

    // -- Soundness: invalid UserDefined unifications --

    #[test]
    fn user_defined_different_id_fails() {
        let mut s = TySubst::new();
        let id_a = fresh_qref();
        let id_b = fresh_qref();
        assert!(
            s.unify(
                &ud(id_a, vec![], vec![]),
                &ud(id_b, vec![], vec![]),
                Invariant
            )
            .is_err()
        );
    }

    #[test]
    fn user_defined_type_arg_mismatch_fails() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        assert!(
            s.unify(
                &ud(id, vec![Ty::Int], vec![]),
                &ud(id, vec![Ty::String], vec![]),
                Invariant,
            )
            .is_err()
        );
    }

    #[test]
    fn user_defined_effect_arg_mismatch_fails() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        assert!(
            s.unify(
                &ud(id, vec![], vec![Effect::pure()]),
                &ud(id, vec![], vec![effectful()]),
                Invariant,
            )
            .is_err()
        );
    }

    #[test]
    fn user_defined_vs_other_ty_fails() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        assert!(
            s.unify(&ud(id, vec![], vec![]), &Ty::Int, Invariant)
                .is_err()
        );
        assert!(
            s.unify(&Ty::String, &ud(id, vec![], vec![]), Invariant)
                .is_err()
        );
    }

    // -- Resolve --

    #[test]
    fn user_defined_resolve_substitutes_args() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        let p = s.fresh_param();
        let e = s.fresh_effect_var();
        let ty = ud(id, vec![p.clone()], vec![e.clone()]);

        // Bind via unification
        assert!(s.unify(&p, &Ty::Int, Invariant).is_ok());
        let eff = effectful();
        assert!(s.unify_effects(&e, &eff, Covariant).is_ok());

        let resolved = s.resolve(&ty);
        match resolved {
            Ty::UserDefined {
                id: rid,
                type_args,
                effect_args,
                ..
            } => {
                assert_eq!(rid, id);
                assert_eq!(type_args, vec![Ty::Int]);
                assert_eq!(effect_args, vec![eff]);
            }
            _ => panic!("expected UserDefined, got {resolved:?}"),
        }
    }

    #[test]
    fn user_defined_inside_list_resolves() {
        let mut s = TySubst::new();
        let id = fresh_qref();
        let p = s.fresh_param();
        let ty = Ty::List(Box::new(ud(id, vec![p.clone()], vec![])));
        assert!(s.unify(&p, &Ty::Int, Invariant).is_ok());
        match s.resolve(&ty) {
            Ty::List(inner) => match *inner {
                Ty::UserDefined { type_args, .. } => assert_eq!(type_args, vec![Ty::Int]),
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
            ownership: Ownership::MoveOnly,
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
            ownership: Ownership::MoveOnly,
        });
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![],
            effect_params: vec![],
            ownership: Ownership::MoveOnly,
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

    /// Helper: create a CastRule and a TySubst with the rule registered.
    /// Returns (from_id, fn_id, subst).
    fn make_cast_subst(
        type_param_count: usize,
        build_to: impl FnOnce(&[Ty]) -> Ty,
    ) -> (QualifiedRef, QualifiedRef, TySubst) {
        let id = fresh_qref();
        let i = acvus_utils::Interner::new();
        let fn_id = QualifiedRef::root(i.intern("cast_fn"));
        let mut rule_subst = TySubst::new();
        let params: Vec<Ty> = (0..type_param_count)
            .map(|_| rule_subst.fresh_param())
            .collect();
        let from = Ty::UserDefined {
            id,
            type_args: params.clone(),
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        let to = build_to(&params);
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None; type_param_count],
            effect_params: vec![],
            ownership: Ownership::MoveOnly,
        });
        reg.register_cast(CastRule {
            from,
            to,
            fn_ref: fn_id,
        });
        let subst = TySubst::with_registry(Freeze::new(reg));
        (id, fn_id, subst)
    }

    // -- Completeness: valid ExternCast coercions --

    #[test]
    fn extern_cast_basic_coercion() {
        // UserDefined(A, [T]) → List<T>
        let (id, fn_id, mut s) = make_cast_subst(1, |p| Ty::List(Box::new(p[0].clone())));

        let from = Ty::UserDefined {
            id,
            type_args: vec![Ty::Int],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        let to = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&from, &to, Covariant).is_ok());
        assert_eq!(s.last_extern_cast, Some(fn_id));
    }

    #[test]
    fn extern_cast_with_param_resolution() {
        // UserDefined(A, [T]) → List<T>, where T is a fresh param on the consumer side
        let (id, _fn_id, mut s) = make_cast_subst(1, |p| Ty::List(Box::new(p[0].clone())));

        let from = Ty::UserDefined {
            id,
            type_args: vec![Ty::Int],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        let consumer_param = s.fresh_param();
        let to = Ty::List(Box::new(consumer_param.clone()));
        assert!(s.unify(&from, &to, Covariant).is_ok());
        assert_eq!(s.resolve(&consumer_param), Ty::Int);
    }

    #[test]
    fn extern_cast_no_type_params() {
        // UserDefined(A, []) → Int
        let (id, _fn_id, mut s) = make_cast_subst(0, |_| Ty::Int);

        let from = Ty::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        assert!(s.unify(&from, &Ty::Int, Covariant).is_ok());
    }

    // -- Soundness: invalid ExternCast --

    #[test]
    fn extern_cast_wrong_target_fails() {
        // Rule: A → List<T>, but expected String
        let (id, _fn_id, mut s) = make_cast_subst(1, |p| Ty::List(Box::new(p[0].clone())));

        let from = Ty::UserDefined {
            id,
            type_args: vec![Ty::Int],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        assert!(s.unify(&from, &Ty::String, Covariant).is_err());
    }

    #[test]
    fn extern_cast_no_rule_fails() {
        // No cast rules registered
        let id = fresh_qref();
        let mut s = TySubst::new();

        let from = Ty::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        assert!(s.unify(&from, &Ty::Int, Covariant).is_err());
    }

    #[test]
    fn extern_cast_invariant_not_attempted() {
        // ExternCast only works in covariant/contravariant, not invariant
        let (id, _fn_id, mut s) = make_cast_subst(0, |_| Ty::Int);

        let from = Ty::UserDefined {
            id,
            type_args: vec![],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        assert!(s.unify(&from, &Ty::Int, Invariant).is_err());
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

        let mut s1 = TySubst::new();
        let t1 = s1.fresh_param();
        let rule_a = CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::List(Box::new(t1)),
            fn_ref: fn_id_a,
        };
        let mut s2 = TySubst::new();
        let t2 = s2.fresh_param();
        let rule_b = CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::List(Box::new(t2)),
            fn_ref: fn_id_b,
        };

        // Build registry manually (bypassing register_cast duplicate check)
        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: vec![],
            ownership: Ownership::MoveOnly,
        });
        reg.from_rules.entry(id).or_default().push(rule_a);
        reg.from_rules.entry(id).or_default().push(rule_b);
        let mut s = TySubst::with_registry(Freeze::new(reg));

        let from = Ty::UserDefined {
            id,
            type_args: vec![Ty::Int],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        };
        assert!(
            s.unify(&from, &Ty::List(Box::new(Ty::Int)), Covariant)
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
        let mut s = TySubst::new();
        let t = s.fresh_param();

        let mut reg = TypeRegistry::new();
        reg.register(UserDefinedDecl {
            qref: id,
            type_params: vec![None],
            effect_params: vec![],
            ownership: Ownership::MoveOnly,
        });
        reg.register_cast(CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::List(Box::new(t.clone())),
            fn_ref: fn_id_a,
        });
        // Same from_id + same to head (List) → panic
        let mut s2 = TySubst::new();
        let t2 = s2.fresh_param();
        reg.register_cast(CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::List(Box::new(t2)),
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
            ownership: Ownership::MoveOnly,
        });
        let mut s1 = TySubst::new();
        let t1 = s1.fresh_param();
        reg.register_cast(CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t1.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::List(Box::new(t1)),
            fn_ref: fn_id_a,
        });
        // Different to head (Option vs List) → ok
        let mut s2 = TySubst::new();
        let t2 = s2.fresh_param();
        reg.register_cast(CastRule {
            from: Ty::UserDefined {
                id,
                type_args: vec![t2.clone()],
                effect_args: vec![],
                ownership: Ownership::MoveOnly,
            },
            to: Ty::Option(Box::new(t2)),
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
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert_eq!(fn_ty.materiality(), Materiality::Composite);
    }

    #[test]
    fn purity_extern_fn_is_lazy() {
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::String)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
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
        let mut s = TySubst::new();
        assert_eq!(s.fresh_param().materiality(), Materiality::Ephemeral);
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
        }));
        assert!(list_fn.is_pureable());
    }

    #[test]
    fn pureable_list_of_fn_with_user_defined_capture() {
        // Fn with UserDefined capture → not pureable, so List<Fn> is also not pureable.
        let list_fn = Ty::List(Box::new(Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(nested.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_fn_user_defined_capture() {
        // List<List<Fn(Int) -> Int>> with UserDefined capture — not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(!nested.is_pureable());
    }

    #[test]
    fn pureable_deque_of_scalars() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
        assert!(Ty::Deque(Box::new(Ty::Int), Box::new(o)).is_pureable());
    }

    #[test]
    fn pureable_deque_of_user_defined() {
        let mut s = TySubst::new();
        let o = s.alloc_identity(false);
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![Ty::Int, Ty::String],
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_user_defined_capture() {
        // Fn with captures=[UserDefined] → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![test_user_defined()],
        };
        assert!(!fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_fn_capture() {
        // Fn with captures=[Fn(Int)->Int (no captures)] → pureable (Fn with empty captures + pure ret)
        let inner_fn = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![inner_fn],
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_user_defined_ret() {
        // Fn returning UserDefined → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(test_user_defined()),

            effect: Effect::pure(),
            captures: vec![],
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
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_list_fn_ret() {
        // Fn returning List<Fn(Int)->Int> → not pureable (transitive)
        let inner_fn = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![test_user_defined()], // inner Fn captures UserDefined
        };
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(inner_fn))),

            effect: Effect::pure(),
            captures: vec![],
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
        let mut s = TySubst::new();
        let a = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
        };
        let b = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
        };
        assert!(s.unify(&a, &b, Invariant).is_ok());
    }

    #[test]
    fn fn_effect_mismatch_invariant_fails() {
        let mut s = TySubst::new();
        let a = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
        };
        let b = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful(),
        };
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn fn_pure_to_effectful_covariant() {
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let pure_fn = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
        };
        let effectful_fn = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful(),
        };
        assert!(s.unify(&v, &pure_fn, Covariant).is_ok());
        assert!(s.unify(&v, &effectful_fn, Covariant).is_ok());
        match s.resolve(&v) {
            Ty::Fn { effect, .. } => assert_eq!(effect, Effect::pure()),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    // -- Coercion arm + effect interaction --

    // -- Effect display --

    #[test]
    fn display_effectful_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful(),
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn!(Int) -> String");
    }

    #[test]
    fn display_pure_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::pure(),
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn(Int) -> String");
    }

    #[test]
    fn display_effectful_extern_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: effectful(),
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: Effect::pure(),
        };
        assert!(!fn_ty.is_materializable());
    }

    #[test]
    fn not_storable_effectful_fn() {
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: effectful(),
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
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: Effect::pure(),
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
        let mut sig_subst = TySubst::new();
        let t = sig_subst.fresh_param(); // Param(0)
        let sig = Ty::Fn {
            params: vec![tp(t.clone())],
            ret: Box::new(t),
            captures: vec![],
            effect: Effect::pure(),
        };

        // Instantiate into a different TySubst.
        let mut inference = TySubst::new();
        let _ = inference.fresh_param(); // Param(0) — burn one to show remapping
        let inst = inference.instantiate(&sig);

        // The instantiated type should have Param(1), not Param(0).
        match &inst {
            Ty::Fn { params, ret, .. } => {
                assert!(matches!(&params[0].ty, Ty::Param { token: p, .. } if p.id() == 1));
                assert!(matches!(ret.as_ref(), Ty::Param { token: p, .. } if p.id() == 1));
            }
            _ => panic!("expected Fn"),
        }
    }

    #[test]
    fn instantiate_concrete_untouched() {
        let mut s = TySubst::new();
        let concrete = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),
            captures: vec![],
            effect: Effect::pure(),
        };
        let inst = s.instantiate(&concrete);
        assert_eq!(inst, concrete);
    }

    // ── EffectSet tests ─────────────────────────────────────────────

    mod effect_set_tests {
        use super::*;
        use crate::graph::types::QualifiedRef;
        use acvus_utils::Interner;

        fn ctx(interner: &Interner, n: usize) -> EffectTarget {
            EffectTarget::Context(QualifiedRef::root(interner.intern(&format!("ctx_{n}"))))
        }

        /// Helper: a non-pure resolved effect with a dummy write target.
        /// Used in place of the removed Effect::self_modifying().
        fn effectful() -> Effect {
            let i = Interner::new();
            Effect::Resolved(EffectSet {
                reads: std::collections::BTreeSet::new(),
                writes: std::collections::BTreeSet::from([EffectTarget::Token(
                    QualifiedRef::root(i.intern("__test")),
                )]),
            })
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
            assert!(!Effect::pure().is_var());
        }

        #[test]
        fn effect_with_writes_is_effectful() {
            assert!(!effectful().is_pure());
            assert!(effectful().is_effectful());
            assert!(!effectful().is_var());
        }

        #[test]
        fn effect_var_is_var() {
            let v = Effect::Var(0);
            assert!(!v.is_pure());
            assert!(!v.is_effectful());
            assert!(v.is_var());
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

        #[test]
        fn effect_union_with_var_returns_none() {
            let a = Effect::pure();
            let b = Effect::Var(0);
            assert!(a.union(&b).is_none());
            assert!(b.union(&a).is_none());
        }

        #[test]
        fn effect_union_pure_pure_is_pure() {
            let u = Effect::pure().union(&Effect::pure()).unwrap();
            assert!(u.is_pure());
        }

        // ── TySubst effect unification tests ────────────────────────

        #[test]
        fn unify_var_binds_to_resolved() {
            let mut s = TySubst::new();
            let var = s.fresh_effect_var();
            let concrete = effectful();
            assert!(s.unify_effect(&var, &concrete).is_ok());
            let resolved = s.resolve_effect(&var);
            assert!(resolved.is_effectful());
        }

        #[test]
        fn unify_var_binds_to_pure() {
            let mut s = TySubst::new();
            let var = s.fresh_effect_var();
            let pure = Effect::pure();
            assert!(s.unify_effect(&var, &pure).is_ok());
            assert!(s.resolve_effect(&var).is_pure());
        }

        #[test]
        fn unify_pure_pure_ok() {
            let mut s = TySubst::new();
            assert!(s.unify_effect(&Effect::pure(), &Effect::pure()).is_ok());
        }

        #[test]
        fn unify_effectful_effectful_ok() {
            let mut s = TySubst::new();
            assert!(s.unify_effect(&effectful(), &effectful()).is_ok());
        }

        #[test]
        fn unify_pure_effectful_invariant_fails() {
            let mut s = TySubst::new();
            assert!(s.unify_effect(&Effect::pure(), &effectful()).is_err());
        }

        #[test]
        fn unify_pure_effectful_covariant_ok() {
            let mut s = TySubst::new();
            // Pure ≤ Effectful in covariant position
            assert!(
                s.unify_effects(&Effect::pure(), &effectful(), Covariant)
                    .is_ok()
            );
        }

        #[test]
        fn unify_effectful_pure_covariant_fails() {
            let mut s = TySubst::new();
            // Effectful ≤ Pure in covariant position — should fail
            assert!(
                s.unify_effects(&effectful(), &Effect::pure(), Covariant)
                    .is_err()
            );
        }

        #[test]
        fn resolve_unbound_var_returns_var() {
            let mut s = TySubst::new();
            let var = s.fresh_effect_var();
            let resolved = s.resolve_effect(&var);
            assert!(resolved.is_var());
        }

        #[test]
        fn resolve_chain_follows_bindings() {
            let mut s = TySubst::new();
            let v0 = s.fresh_effect_var();
            let v1 = s.fresh_effect_var();
            let concrete = effectful();
            // v0 → v1 → concrete
            assert!(s.unify_effect(&v0, &v1).is_ok());
            assert!(s.unify_effect(&v1, &concrete).is_ok());
            let resolved = s.resolve_effect(&v0);
            assert!(resolved.is_effectful());
        }

        #[test]
        fn unify_two_vars_share_binding() {
            let mut s = TySubst::new();
            let v0 = s.fresh_effect_var();
            let v1 = s.fresh_effect_var();
            assert!(s.unify_effect(&v0, &v1).is_ok());
            // Bind one → both resolve to same
            let concrete = effectful();
            assert!(s.unify_effect(&v1, &concrete).is_ok());
            assert!(s.resolve_effect(&v0).is_effectful());
            assert!(s.resolve_effect(&v1).is_effectful());
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
            let mut s = TySubst::new();
            let c = ctx(&i, 0);
            let var = s.fresh_effect_var();
            let effect = Effect::Resolved(EffectSet {
                reads: [c].into_iter().collect(),
                ..Default::default()
            });
            assert!(s.unify_effect(&var, &effect).is_ok());
            let resolved = s.resolve_effect(&var);
            match &resolved {
                Effect::Resolved(set) => {
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
            assert_eq!(format!("{}", effectful()), "Effectful(w=1)");
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
            assert_eq!(format!("{}", Effect::Var(42)), "EffectVar(42)");
        }
    }
}
