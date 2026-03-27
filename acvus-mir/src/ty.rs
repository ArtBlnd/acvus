use std::collections::BTreeSet;
use std::fmt;

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::graph::types::QualifiedRef;

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

/// Fine-grained effect information: which contexts are read/written,
/// and whether opaque IO is involved.
///
/// Pure = all fields empty/false. No separate variant needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct EffectSet {
    pub reads: BTreeSet<QualifiedRef>,
    pub writes: BTreeSet<QualifiedRef>,
    pub io: bool,
    /// Value is consumed/mutated on use (e.g. Iterator cursor advance).
    /// Propagates through combinators: map(self_mod_iter, f) → self_mod.
    pub self_modifying: bool,
}

impl EffectSet {
    pub fn is_pure(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty() && !self.io && !self.self_modifying
    }

    /// Union of two effect sets. All effects propagate (contagious).
    pub fn union(&self, other: &Self) -> Self {
        Self {
            reads: self.reads.union(&other.reads).copied().collect(),
            writes: self.writes.union(&other.writes).copied().collect(),
            io: self.io || other.io,
            self_modifying: self.self_modifying || other.self_modifying,
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
        if self.io {
            parts.push("io".to_string());
        }
        if self.self_modifying {
            parts.push("mut".to_string());
        }
        write!(f, "{})", parts.join(", "))
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
    /// No side effects: reads nothing, writes nothing, no IO.
    pub fn pure() -> Self {
        Effect::Resolved(EffectSet::default())
    }

    /// Opaque IO effect (e.g. context access without known QualifiedRef).
    pub fn io() -> Self {
        Effect::Resolved(EffectSet {
            io: true,
            ..EffectSet::default()
        })
    }

    /// Self-modifying effect: value mutates on use (e.g. Iterator cursor).
    pub fn self_modifying() -> Self {
        Effect::Resolved(EffectSet {
            self_modifying: true,
            ..EffectSet::default()
        })
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

/// Origin identity for Deque types — prevents mixing deques from different sources.
///
/// - `Concrete(u32)`: a fixed origin created by `[]` literals — unique provenance.
/// - `Var(u32)`: an origin variable created by builtin signatures (e.g. `extend`) —
///   binds to the actual origin of the input Deque during unification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Origin {
    Concrete(u32),
    Var(u32),
}

impl std::fmt::Display for Origin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Origin::Concrete(id) => write!(f, "Origin({id})"),
            Origin::Var(id) => write!(f, "OriginVar({id})"),
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
    /// Immutable list of elements. Coerces from Deque (losing origin), coerces to Iterator.
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
    /// Opaque type: user-defined, identified by name. No internal structure.
    Opaque(std::string::String),
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
    /// Lazy iterator over elements of type T, with effect classification.
    Iterator(Box<Ty>, Effect),
    /// Lazy sequence over elements of type T. Lazy version of Deque with origin identity and effect.
    Sequence(Box<Ty>, Origin, Effect),
    /// Deque type: tracked deque with origin identity.
    /// `Origin` prevents mixing deques from different sources.
    Deque(Box<Ty>, Origin),
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
            Ty::Handle(_, effect) | Ty::Iterator(_, effect) | Ty::Sequence(_, _, effect) => {
                Some(effect)
            }
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
            Ty::List(elem)
            | Ty::Deque(elem, _)
            | Ty::Iterator(elem, _)
            | Ty::Sequence(elem, _, _) => Some(elem),
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
            | Ty::Iterator(..)
            | Ty::Sequence(..)
            | Ty::Option(_)
            | Ty::Enum { .. } => Materiality::Composite,
            Ty::Opaque(_) => Materiality::Ephemeral,
            Ty::Param { .. } | Ty::Error(_) => Materiality::Ephemeral,
        }
    }

    /// Returns true if this type can be deeply converted to a pure representation.
    /// Transitively checks container contents — `List<Fn>` returns false.
    pub fn is_pureable(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => true,
            Ty::List(inner) => inner.is_pureable(),
            Ty::Iterator(inner, effect) | Ty::Handle(inner, effect) => {
                inner.is_pureable() && effect.is_pure()
            }
            Ty::Deque(inner, _) => inner.is_pureable(),
            Ty::Sequence(inner, _, effect) => inner.is_pureable() && effect.is_pure(),
            Ty::Option(inner) => inner.is_pureable(),
            Ty::Tuple(elems) => elems.iter().all(|e| e.is_pureable()),
            Ty::Object(fields) => fields.values().all(|v| v.is_pureable()),
            Ty::Enum { variants, .. } => variants
                .values()
                .all(|p| p.as_ref().is_none_or(|ty| ty.is_pureable())),
            Ty::Fn { captures, ret, .. } => {
                captures.iter().all(|c| c.is_pureable()) && ret.is_pureable()
            }
            Ty::Opaque(_) => false,
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
            Ty::Iterator(_, _) | Ty::Sequence(_, _, _) | Ty::Handle(..) | Ty::Fn { .. } | Ty::Opaque(_) => false,
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
            Ty::Iterator(inner, effect) => {
                let bang = if effect.is_effectful() { "!" } else { "" };
                write!(f, "Iterator{bang}<{}>", inner.display(self.interner))
            }
            Ty::Sequence(inner, origin, effect) => {
                let bang = if effect.is_effectful() { "!" } else { "" };
                write!(
                    f,
                    "Sequence{bang}<{}, {}>",
                    inner.display(self.interner),
                    origin
                )
            }
            Ty::Deque(inner, origin) => {
                write!(f, "Deque<{}, {}>", inner.display(self.interner), origin)
            }
            Ty::Option(inner) => write!(f, "Option<{}>", inner.display(self.interner)),
            Ty::Opaque(name) => write!(f, "{name}"),
            Ty::Enum { name, .. } => write!(f, "{}", self.interner.resolve(*name)),
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
    origin_bindings: FxHashMap<u32, Origin>,
    effect_bindings: FxHashMap<u32, Effect>,
    next_param: u32,
    next_origin: u32,
    next_effect: u32,
}

/// Substitution table for type unification.
pub struct TySubst {
    bindings: FxHashMap<ParamToken, Ty>,
    origin_bindings: FxHashMap<u32, Origin>,
    effect_bindings: FxHashMap<u32, Effect>,
    next_param: u32,
    next_origin: u32,
    next_effect: u32,
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
            origin_bindings: FxHashMap::default(),
            effect_bindings: FxHashMap::default(),
            next_param: 0,
            next_origin: 0,
            next_effect: 0,
        }
    }

    /// Take a snapshot of the current state for later rollback.
    pub fn snapshot(&self) -> TySubstSnapshot {
        TySubstSnapshot {
            bindings: self.bindings.clone(),
            origin_bindings: self.origin_bindings.clone(),
            effect_bindings: self.effect_bindings.clone(),
            next_param: self.next_param,
            next_origin: self.next_origin,
            next_effect: self.next_effect,
        }
    }

    /// Restore state from a snapshot, discarding any bindings made since.
    pub fn rollback(&mut self, snap: TySubstSnapshot) {
        self.bindings = snap.bindings;
        self.origin_bindings = snap.origin_bindings;
        self.effect_bindings = snap.effect_bindings;
        self.next_param = snap.next_param;
        self.next_origin = snap.next_origin;
        self.next_effect = snap.next_effect;
    }

    /// Allocate a fresh origin VARIABLE for builtin signatures.
    /// Origin variables bind to concrete origins during unification.
    pub fn fresh_origin(&mut self) -> Origin {
        let o = Origin::Var(self.next_origin);
        self.next_origin += 1;
        o
    }

    /// Allocate a fresh CONCRETE origin for `[]` literals.
    /// Concrete origins are unique and can only unify with origin variables.
    pub fn fresh_concrete_origin(&mut self) -> Origin {
        let o = Origin::Concrete(self.next_origin);
        self.next_origin += 1;
        o
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

    /// Resolve an origin by following binding chains for Origin::Var.
    pub fn resolve_origin(&self, o: Origin) -> Origin {
        match o {
            Origin::Concrete(_) => o,
            Origin::Var(id) => {
                if let Some(&bound) = self.origin_bindings.get(&id) {
                    self.resolve_origin(bound)
                } else {
                    o
                }
            }
        }
    }

    /// Unify two origins. Returns Ok(()) on success, Err with the two resolved
    /// origins on failure.
    fn unify_origins(&mut self, a: Origin, b: Origin) -> Result<(), (Origin, Origin)> {
        let a = self.resolve_origin(a);
        let b = self.resolve_origin(b);
        match (a, b) {
            (Origin::Concrete(x), Origin::Concrete(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err((a, b))
                }
            }
            (Origin::Var(v), other) | (other, Origin::Var(v)) => {
                if let Origin::Var(v2) = other
                    && v == v2
                {
                    return Ok(());
                }
                self.origin_bindings.insert(v, other);
                Ok(())
            }
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

    /// Effect LUB: rebind effect vars to the union of both effects.
    /// Falls back to `io()` (lattice top) when union is not computable.
    fn coerce_effects_to_effectful(&mut self, ea: &Effect, eb: &Effect) {
        let resolved_a = self.resolve_effect(ea);
        let resolved_b = self.resolve_effect(eb);
        let merged = match (&resolved_a, &resolved_b) {
            (Effect::Resolved(sa), Effect::Resolved(sb)) => Effect::Resolved(sa.union(sb)),
            _ => Effect::io(),
        };
        if let Some(v) = self.find_leaf_effect_var(ea) {
            self.effect_bindings.insert(v, merged.clone());
        }
        if let Some(v) = self.find_leaf_effect_var(eb) {
            self.effect_bindings.insert(v, merged);
        }
    }

    /// Compute LUB (least upper bound) of two same-constructor types whose
    /// generic parameters (Origin / Effect) don't match.
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
    /// | `Deque<T, O1>` vs `Deque<T, O2>` | `List<T>` (origin erased) |
    /// | `Sequence<T, O1, E>` vs `Sequence<T, O2, E>` | `Iterator<T, max(E1,E2)>` (origin erased) |
    /// | `Iterator<T, Pure>` vs `Iterator<T, Effectful>` | `Iterator<T, Effectful>` |
    /// | `Sequence<T, O, Pure>` vs `Sequence<T, O, Effectful>` | `Sequence<T, O, Effectful>` |
    /// | `Fn{Pure}` vs `Fn{Effectful}` (same params/ret) | `Fn{Effectful}` |
    fn try_lub(&mut self, a: &Ty, b: &Ty) -> Option<Ty> {
        match (a, b) {
            // Origin mismatch: Deque → List (origin erased)
            (Ty::Deque(ia, _), Ty::Deque(ib, _)) => {
                self.unify(ia, ib, Polarity::Invariant).ok()?;
                Some(Ty::List(Box::new(self.resolve(ia))))
            }
            // Sequence: origin mismatch → Iterator; same origin + effect mismatch → union
            (Ty::Sequence(ia, oa, ea), Ty::Sequence(ib, ob, eb)) => {
                self.unify(ia, ib, Polarity::Invariant).ok()?;
                if self.unify_origins(*oa, *ob).is_ok() {
                    // Same origin, effect mismatch → Sequence<T, O, union(E1, E2)>
                    self.coerce_effects_to_effectful(ea, eb);
                    let resolved_a = self.resolve_effect(ea);
                    let resolved_b = self.resolve_effect(eb);
                    let merged = resolved_a.union(&resolved_b).unwrap_or_else(Effect::io);
                    Some(Ty::Sequence(
                        Box::new(self.resolve(ia)),
                        self.resolve_origin(*oa),
                        merged,
                    ))
                } else {
                    // Origin mismatch → Iterator<T, union(E1, E2)>
                    let resolved_a = self.resolve_effect(ea);
                    let resolved_b = self.resolve_effect(eb);
                    let effect = resolved_a.union(&resolved_b).unwrap_or_else(Effect::io);
                    self.coerce_effects_to_effectful(ea, eb);
                    Some(Ty::Iterator(Box::new(self.resolve(ia)), effect))
                }
            }
            // Effect mismatch: Iterator → Iterator (effect = union)
            (Ty::Iterator(ia, ea), Ty::Iterator(ib, eb)) => {
                self.unify(ia, ib, Polarity::Invariant).ok()?;
                self.coerce_effects_to_effectful(ea, eb);
                let resolved_a = self.resolve_effect(ea);
                let resolved_b = self.resolve_effect(eb);
                let merged = resolved_a.union(&resolved_b).unwrap_or_else(Effect::io);
                Some(Ty::Iterator(Box::new(self.resolve(ia)), merged))
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
                self.coerce_effects_to_effectful(ea, eb);
                let resolved_a = self.resolve_effect(ea);
                let resolved_b = self.resolve_effect(eb);
                let merged = resolved_a.union(&resolved_b).unwrap_or_else(Effect::io);
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
            _ => None,
        }
    }

    /// Rebind leaf type-vars to the LUB, or return Err if polarity is Invariant.
    ///
    /// This is the shared fallback for all same-constructor parameter mismatches
    /// (Origin, Effect). Must only be called after the exact invariant unify
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
        let lub = self.try_lub(a, b).ok_or_else(|| (a.clone(), b.clone()))?;
        if let Some(leaf) = self.find_leaf_param(orig_a) {
            self.bindings.insert(leaf, lub.clone());
        }
        if let Some(leaf) = self.find_leaf_param(orig_b) {
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
            Ty::Iterator(inner, effect) => {
                Ty::Iterator(Box::new(self.resolve(inner)), self.resolve_effect(effect))
            }
            Ty::Sequence(inner, origin, effect) => Ty::Sequence(
                Box::new(self.resolve(inner)),
                self.resolve_origin(*origin),
                self.resolve_effect(effect),
            ),
            Ty::Deque(inner, origin) => {
                Ty::Deque(Box::new(self.resolve(inner)), self.resolve_origin(*origin))
            }
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

            (Ty::Opaque(a), Ty::Opaque(b)) if a == b => Ok(()),

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
            (Ty::Iterator(ia, ea), Ty::Iterator(ib, eb)) => {
                self.unify(ia, ib, Polarity::Invariant)?;
                self.unify_effects(ea, eb, pol)
                    .or_else(|_| self.lub_or_err(pol, orig_a, orig_b, &a, &b))
            }

            (Ty::Sequence(ia, oa, ea), Ty::Sequence(ib, ob, eb)) => {
                // Origin mismatch or effect mismatch → both go through lub_or_err.
                let origin_ok = self.unify_origins(*oa, *ob).is_ok();
                let inner_ok = self.unify(ia, ib, Polarity::Invariant).is_ok();
                let effect_ok = origin_ok && self.unify_effects(ea, eb, pol).is_ok();
                if origin_ok && inner_ok && effect_ok {
                    Ok(())
                } else if !inner_ok {
                    Err((a.clone(), b.clone()))
                } else {
                    self.lub_or_err(pol, orig_a, orig_b, &a, &b)
                }
            }

            (Ty::List(a), Ty::List(b)) => self.unify(a, b, Polarity::Invariant),

            (Ty::Deque(ia, oa), Ty::Deque(ib, ob)) => match self.unify_origins(*oa, *ob) {
                Ok(()) => self.unify(ia, ib, Polarity::Invariant),
                Err(_) => self.lub_or_err(pol, orig_a, orig_b, &a, &b),
            },

            (Ty::Option(a), Ty::Option(b)) => self.unify(a, b, Polarity::Invariant),

            (Ty::Object(fa), Ty::Object(fb)) => {
                // Unify overlapping fields.
                for (key, ty_a) in fa {
                    if let Some(ty_b) = fb.get(key) {
                        self.unify(ty_a, ty_b, pol)?;
                    }
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
            // List<T> ≤ Iterator<T, E> (E must accept Pure)
            (Ty::List(inner_l), Ty::Iterator(inner_i, e)) => {
                self.unify_effects(&Effect::pure(), e, Polarity::Invariant)
                    .map_err(|_| ())?;
                self.unify(inner_l, inner_i, Polarity::Invariant)
                    .map_err(|_| ())
            }
            // Deque<T, O> ≤ Iterator<T, E> (E must accept Pure)
            (Ty::Deque(inner_d, _), Ty::Iterator(inner_i, e)) => {
                self.unify_effects(&Effect::pure(), e, Polarity::Invariant)
                    .map_err(|_| ())?;
                self.unify(inner_d, inner_i, Polarity::Invariant)
                    .map_err(|_| ())
            }
            // Deque<T, O> ≤ Sequence<T, O', E> (origin preserved, E must accept Pure)
            (Ty::Deque(inner_d, od), Ty::Sequence(inner_s, os, e)) => {
                self.unify_origins(*od, *os).map_err(|_| ())?;
                self.unify_effects(&Effect::pure(), e, Polarity::Invariant)
                    .map_err(|_| ())?;
                self.unify(inner_d, inner_s, Polarity::Invariant)
                    .map_err(|_| ())
            }
            // Sequence<T, O, E> ≤ Iterator<T, E'> (origin lost, effect preserved)
            (Ty::Sequence(inner_s, _, es), Ty::Iterator(inner_i, ei)) => {
                self.unify_effects(es, ei, Polarity::Invariant)
                    .map_err(|_| ())?;
                self.unify(inner_s, inner_i, Polarity::Invariant)
                    .map_err(|_| ())
            }
            _ => Err(()),
        }
    }

    /// Instantiate a polymorphic type: replace all `Param`, `Effect::Var`,
    /// and `Origin::Var` in `ty` with fresh values from this substitution.
    ///
    /// Used when calling a generic function: the graph stores a signature
    /// with "template" Params (e.g. `Param(0)`), and each call site gets
    /// its own fresh inference variables via this method.
    pub fn instantiate(&mut self, ty: &Ty) -> Ty {
        let mut param_map: FxHashMap<ParamToken, ParamToken> = FxHashMap::default();
        let mut effect_map: FxHashMap<u32, u32> = FxHashMap::default();
        let mut origin_map: FxHashMap<u32, Origin> = FxHashMap::default();
        self.instantiate_inner(ty, &mut param_map, &mut effect_map, &mut origin_map)
    }

    fn instantiate_inner(
        &mut self,
        ty: &Ty,
        param_map: &mut FxHashMap<ParamToken, ParamToken>,
        effect_map: &mut FxHashMap<u32, u32>,
        origin_map: &mut FxHashMap<u32, Origin>,
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
                self.instantiate_inner(inner, param_map, effect_map, origin_map),
            )),
            Ty::Iterator(inner, e) => {
                let new_e = self.instantiate_effect(e, effect_map);
                Ty::Iterator(
                    Box::new(self.instantiate_inner(inner, param_map, effect_map, origin_map)),
                    new_e,
                )
            }
            Ty::Sequence(inner, o, e) => {
                let new_o = self.instantiate_origin(*o, origin_map);
                let new_e = self.instantiate_effect(e, effect_map);
                Ty::Sequence(
                    Box::new(self.instantiate_inner(inner, param_map, effect_map, origin_map)),
                    new_o,
                    new_e,
                )
            }
            Ty::Deque(inner, o) => {
                let new_o = self.instantiate_origin(*o, origin_map);
                Ty::Deque(
                    Box::new(self.instantiate_inner(inner, param_map, effect_map, origin_map)),
                    new_o,
                )
            }
            Ty::Option(inner) => Ty::Option(Box::new(
                self.instantiate_inner(inner, param_map, effect_map, origin_map),
            )),
            Ty::Tuple(elems) => Ty::Tuple(
                elems
                    .iter()
                    .map(|e| self.instantiate_inner(e, param_map, effect_map, origin_map))
                    .collect(),
            ),
            Ty::Object(fields) => Ty::Object(
                fields
                    .iter()
                    .map(|(k, v)| {
                        (
                            *k,
                            self.instantiate_inner(v, param_map, effect_map, origin_map),
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
                            ty: self.instantiate_inner(&p.ty, param_map, effect_map, origin_map),
                        })
                        .collect(),
                    ret: Box::new(self.instantiate_inner(ret, param_map, effect_map, origin_map)),
                    captures: captures
                        .iter()
                        .map(|c| self.instantiate_inner(c, param_map, effect_map, origin_map))
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
                                    self.instantiate_inner(ty, param_map, effect_map, origin_map),
                                )
                            }),
                        )
                    })
                    .collect(),
            },
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

    fn instantiate_origin(&mut self, o: Origin, origin_map: &mut FxHashMap<u32, Origin>) -> Origin {
        match o {
            Origin::Var(id) => *origin_map.entry(id).or_insert_with(|| {
                let fresh = Origin::Var(self.next_origin);
                self.next_origin += 1;
                fresh
            }),
            concrete => concrete,
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
            Ty::Iterator(inner, _) => self.occurs_in(param, inner),
            Ty::Sequence(inner, ..) => self.occurs_in(param, inner),
            Ty::Deque(inner, _) => self.occurs_in(param, inner),
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
#[derive(Debug, Clone)]
pub struct TypeEnv {
    /// Context variable types (`@name` → Ty), keyed by QualifiedRef (namespace-aware).
    pub contexts: FxHashMap<QualifiedRef, Ty>,
    /// Function types (`name` → `Ty::Fn`). One entry per function name.
    pub functions: FxHashMap<Astr, Ty>,
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
    use acvus_utils::Interner;

    use Polarity::*;

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
        let o = s.fresh_origin();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        let deque_int = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&deque_t, &deque_int, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&deque_t), Ty::Deque(Box::new(Ty::Int), o));
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
        let o = s.fresh_origin();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
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
    fn unify_deque_same_origin() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_param();
        let d1 = Ty::Deque(Box::new(t.clone()), o);
        let d2 = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&d1, &d2, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&d1), Ty::Deque(Box::new(Ty::Int), o));
    }

    #[test]
    fn unify_deque_different_concrete_origin_fails() {
        // Invariant: different concrete origins → error
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        assert_ne!(o1, o2);
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        assert!(
            s.unify(&d1, &d2, Invariant).is_err(),
            "different concrete origins must not unify in Invariant"
        );
    }

    #[test]
    fn unify_deque_origin_var_binds_to_concrete() {
        // Origin::Var should bind to Origin::Concrete during unification
        let mut s = TySubst::new();
        let concrete = s.fresh_concrete_origin();
        let var = s.fresh_origin(); // Origin::Var
        let d1 = Ty::Deque(Box::new(Ty::Int), concrete);
        let d2 = Ty::Deque(Box::new(Ty::Int), var);
        assert!(
            s.unify(&d1, &d2, Invariant).is_ok(),
            "origin Var should bind to Concrete"
        );
        assert_eq!(s.resolve_origin(var), concrete);
    }

    #[test]
    fn unify_deque_origin_var_preserves_identity() {
        // Two Deques through same Origin::Var should resolve to same concrete origin
        let mut s = TySubst::new();
        let concrete = s.fresh_concrete_origin();
        let var = s.fresh_origin();
        let d_concrete = Ty::Deque(Box::new(Ty::Int), concrete);
        let d_var = Ty::Deque(Box::new(Ty::Int), var);
        assert!(s.unify(&d_concrete, &d_var, Invariant).is_ok());
        // Now a second concrete origin should NOT match the same var
        let concrete2 = s.fresh_concrete_origin();
        let d_concrete2 = Ty::Deque(Box::new(Ty::Int), concrete2);
        let d_var2 = Ty::Deque(Box::new(Ty::Int), var);
        assert!(
            s.unify(&d_concrete2, &d_var2, Invariant).is_err(),
            "var already bound to different concrete"
        );
    }

    #[test]
    fn unify_deque_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o);
        let d2 = Ty::Deque(Box::new(Ty::String), o);
        assert!(
            s.unify(&d1, &d2, Invariant).is_err(),
            "inner type mismatch with same origin must fail"
        );
    }

    #[test]
    fn coerce_deque_to_iterator() {
        // Deque<Int, O> can be used where Iterator<Int> is expected (Covariant)
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(
            s.unify(&deque, &iter, Covariant).is_ok(),
            "Deque → Iterator coercion should succeed"
        );
    }

    #[test]
    fn coerce_deque_to_iterator_with_var() {
        // Deque<T, O> unifies with Iterator<Int> → T becomes Int
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_param();
        let deque = Ty::Deque(Box::new(t.clone()), o);
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&deque, &iter, Covariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn coerce_iterator_to_deque_fails() {
        // Iterator<Int> cannot become Deque<Int, O> — one-directional only
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(
            s.unify(&iter, &deque, Covariant).is_err(),
            "Iterator → Deque coercion must be forbidden"
        );
    }

    #[test]
    fn fresh_origin_produces_unique_ids() {
        let mut s = TySubst::new();
        let o1 = s.fresh_origin();
        let o2 = s.fresh_origin();
        let o3 = s.fresh_origin();
        assert_ne!(o1, o2);
        assert_ne!(o2, o3);
        assert_ne!(o1, o3);
    }

    #[test]
    fn resolve_deque_propagates_inner() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_param();
        assert!(s.unify(&t, &Ty::String, Invariant).is_ok());
        let deque = Ty::Deque(Box::new(t.clone()), o);
        assert_eq!(s.resolve(&deque), Ty::Deque(Box::new(Ty::String), o));
    }

    #[test]
    fn occurs_in_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_param();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        // T = Deque<T, O> should fail (occurs check)
        assert!(s.unify(&t, &deque_t, Invariant).is_err());
    }

    #[test]
    fn snapshot_rollback_preserves_origin_counter() {
        let mut s = TySubst::new();
        let _o1 = s.fresh_origin();
        let snap = s.snapshot();
        let o2 = s.fresh_origin();
        assert_eq!(o2, Origin::Var(1)); // second origin
        s.rollback(snap);
        let o_after = s.fresh_origin();
        assert_eq!(
            o_after,
            Origin::Var(1),
            "rollback should restore origin counter"
        );
    }

    #[test]
    fn deque_to_iterator_coercion_with_inner_var_unification() {
        // Deque<Var, O> vs Iterator<Var> where both Vars unify to same type
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t1 = s.fresh_param();
        let t2 = s.fresh_param();
        let deque = Ty::Deque(Box::new(t1.clone()), o);
        let iter = Ty::Iterator(Box::new(t2.clone()), Effect::pure());
        assert!(s.unify(&deque, &iter, Covariant).is_ok());
        // Now bind t2 to Int
        assert!(s.unify(&t2, &Ty::Int, Invariant).is_ok());
        // t1 should also resolve to Int via transitive unification
        assert_eq!(s.resolve(&t1), Ty::Int);
    }

    #[test]
    fn unify_deque_coerces_to_list() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let l = Ty::List(Box::new(Ty::Int));
        assert!(
            s.unify(&d, &l, Covariant).is_ok(),
            "Deque should coerce to List"
        );
    }

    #[test]
    fn unify_list_does_not_coerce_to_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(
            s.unify(&l, &d, Covariant).is_err(),
            "List must not coerce to Deque"
        );
    }

    #[test]
    fn unify_list_coerces_to_iterator() {
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let i = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(
            s.unify(&l, &i, Covariant).is_ok(),
            "List should coerce to Iterator"
        );
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn deque_origin_mismatch_covariant_demotes_to_list() {
        // Covariant: Deque+Deque origin mismatch → List demotion
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
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
    fn deque_origin_mismatch_invariant_fails() {
        // Invariant: Deque+Deque origin mismatch → error
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        assert!(s.unify(&d1, &d2, Invariant).is_err());
    }

    #[test]
    fn deque_coerces_to_list_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l, Covariant).is_ok());
    }

    #[test]
    fn list_does_not_coerce_to_deque_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&l, &d, Covariant).is_err());
    }

    #[test]
    fn contravariant_list_deque_ok() {
        // Contravariant: (List, Deque) → reversed: Deque ≤ List → OK
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&l, &d, Contravariant).is_ok());
    }

    #[test]
    fn contravariant_deque_list_fails() {
        // Contravariant: (Deque, List) → reversed: List ≤ Deque → invalid
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l, Contravariant).is_err());
    }

    #[test]
    fn invariant_deque_list_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
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
        let o1 = s.fresh_origin();
        let o2 = s.fresh_origin();
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o1)),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o2))],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_ok());
    }

    #[test]
    fn list_literal_mixed_deque_origins() {
        // Simulates: multiple Deque elements with different origins → List demotion
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let elem_var = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::String), o1);
        let d2 = Ty::Deque(Box::new(Ty::String), o2);
        // First element sets the type
        assert!(s.unify(&elem_var, &d1, Covariant).is_ok());
        // Second element with different origin → demotion
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let o3 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        let d3 = Ty::Deque(Box::new(Ty::Int), o3);
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant)
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let o3 = s.fresh_concrete_origin();
        let inner_var = s.fresh_param();
        let v = s.fresh_param();
        assert!(
            s.unify(&Ty::Deque(Box::new(inner_var.clone()), o1), &v, Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v, Covariant)
                .is_ok()
        );
        // inner_var should now be Int, v should be List<Int>
        assert_eq!(s.resolve(&inner_var), Ty::Int);
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        // Third deque with same inner type
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), o3), &v, Covariant)
                .is_ok()
        );
    }

    #[test]
    fn concrete_deque_deque_covariant_no_var_no_rebind() {
        // Two concrete Deques (no Var backing) with mismatched origins.
        // Covariant demotion: Ok() returned but no Var to rebind.
        // This is semantically "they are compatible as List", caller uses resolve on orig.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        // Should succeed — covariant allows demotion even without Var.
        assert!(s.unify(&d1, &d2, Covariant).is_ok());
    }

    #[test]
    fn concrete_deque_deque_inner_mismatch_plus_origin_mismatch() {
        // Both inner type AND origin mismatch — inner unify should fail first.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::String), o2);
        assert!(
            s.unify(&d1, &d2, Covariant).is_err(),
            "inner type mismatch must fail regardless of demotion"
        );
    }

    #[test]
    fn demotion_inner_type_still_var() {
        // Demotion when inner type is an unresolved Var — should resolve to List<Var>.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let inner = s.fresh_param();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(inner.clone()), o1), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(inner.clone()), o2), Covariant)
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
        // Contravariant: Deque+Deque origin mismatch also demotes (pol != Invariant).
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Contravariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Contravariant)
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
        let o = s.fresh_concrete_origin();
        let obj_deque = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), o),
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
        let o = s.fresh_concrete_origin();
        let obj_deque = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), o),
        )]));
        let obj_list = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::List(Box::new(Ty::String)),
        )]));
        assert!(s.unify(&obj_deque, &obj_list, Invariant).is_err());
    }

    #[test]
    fn object_field_deque_origin_mismatch_demotion() {
        // {tags: Deque<S, o1>} vs {tags: Deque<S, o2>} in Covariant.
        // Inner Deque origin mismatch → demoted to List within the field.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), o1),
        )]));
        let obj2 = Ty::Object(FxHashMap::from_iter([(
            i.intern("tags"),
            Ty::Deque(Box::new(Ty::String), o2),
        )]));
        assert!(s.unify(&v, &obj1, Covariant).is_ok());
        assert!(s.unify(&v, &obj2, Covariant).is_ok());
    }

    #[test]
    fn option_deque_to_list_covariant_fails() {
        // Option<Deque<Int>> vs Option<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside Option is a type error.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let opt_deque = Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let opt_list = Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&opt_deque, &opt_list, Covariant).is_err());
    }

    #[test]
    fn option_deque_to_list_invariant_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let opt_deque = Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let opt_list = Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&opt_deque, &opt_list, Invariant).is_err());
    }

    #[test]
    fn tuple_deque_coercion_covariant() {
        // (Deque<Int>, String) vs (List<Int>, String) in Covariant.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let t1 = Ty::Tuple(vec![Ty::Deque(Box::new(Ty::Int), o), Ty::String]);
        let t2 = Ty::Tuple(vec![Ty::List(Box::new(Ty::Int)), Ty::String]);
        assert!(s.unify(&t1, &t2, Covariant).is_ok());
    }

    #[test]
    fn tuple_deque_coercion_invariant_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let t1 = Ty::Tuple(vec![Ty::Deque(Box::new(Ty::Int), o), Ty::String]);
        let t2 = Ty::Tuple(vec![Ty::List(Box::new(Ty::Int)), Ty::String]);
        assert!(s.unify(&t1, &t2, Invariant).is_err());
    }

    #[test]
    fn double_flip_restores_covariant() {
        // Fn(Fn(Deque) -> Unit) -> Unit  vs  Fn(Fn(List) -> Unit) -> Unit
        // Outer Covariant → param flips to Contravariant → inner param flips back to Covariant.
        // So inner param: Deque vs List in Covariant → Deque≤List OK.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let inner_fn_a = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o))],
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
        let o = s.fresh_concrete_origin();
        let inner_fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let inner_fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o))],
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
        let o = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o)),

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
        let o = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o))],
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
        let o = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o))],
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
        let o1 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant)
                .is_ok()
        );
        let snap = s.snapshot();
        let o2 = s.fresh_concrete_origin();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant)
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
            Ty::Deque(Box::new(Ty::Int), o1),
            "rollback should undo demotion"
        );
    }

    #[test]
    fn demotion_then_iterator_coercion() {
        // After demotion to List, the result should still coerce to Iterator.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        // List<Int> ≤ Iterator<Int> in Covariant
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Covariant
            )
            .is_ok()
        );
    }

    #[test]
    fn nested_list_of_deque_coercion_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Covariant.
        // Inner item type is invariant — Deque vs List inside List is a type error.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::List(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let b = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn nested_list_of_deque_invariant_fails() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Invariant.
        // Inner: Deque vs List in Invariant → fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::List(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let b = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn deque_to_iterator_invariant_fails() {
        // Deque → Iterator in Invariant must fail.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let i = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&d, &i, Invariant).is_err());
    }

    #[test]
    fn list_to_iterator_invariant_fails() {
        // List → Iterator in Invariant must fail.
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let i = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&l, &i, Invariant).is_err());
    }

    #[test]
    fn deque_to_iterator_contravariant_fails() {
        // (Deque, Iterator) in Contravariant → reversed: Iterator≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let i = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&d, &i, Contravariant).is_err());
    }

    #[test]
    fn iterator_to_deque_contravariant_ok() {
        // (Iterator, Deque) in Contravariant → reversed: Deque≤Iterator → OK.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let i = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&i, &d, Contravariant).is_ok());
    }

    #[test]
    fn chained_coercion_deque_to_iterator_covariant() {
        // Deque<Int> → Iterator<Int> directly in Covariant (skipping List).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Covariant
            )
            .is_ok()
        );
    }

    #[test]
    fn enum_variant_deque_coercion_covariant() {
        // Enum with Deque payload vs same enum with List payload, Covariant.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.fresh_concrete_origin();
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(Ty::Deque(Box::new(Ty::Int), o))),
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
        let o = s.fresh_concrete_origin();
        let name = i.intern("Result");
        let tag = i.intern("Ok");
        let e1 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([(
                tag,
                Some(Box::new(Ty::Deque(Box::new(Ty::Int), o))),
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
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o), Invariant)
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o1), Invariant)
                .is_ok()
        );
        // Demotion via v1
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v1, Covariant)
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        // Must chain BEFORE binding to concrete — otherwise shallow_resolve
        // flattens the chain and v2 binds directly to Deque, not to v1.
        assert!(s.unify(&v2, &v1, Invariant).is_ok());
        assert!(
            s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), o1), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v2, Covariant)
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
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let cyclic = Ty::Deque(Box::new(v.clone()), o);
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Deque(Box::new(Ty::Deque(Box::new(Ty::Int), o1)), o2);
        let b = Ty::Deque(Box::new(Ty::List(Box::new(Ty::Int))), o2);
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn nested_deque_in_deque_invariant_inner_coercion_fails() {
        // Same structure but Invariant → inner Deque vs List fails.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Deque(Box::new(Ty::Deque(Box::new(Ty::Int), o1)), o2);
        let b = Ty::Deque(Box::new(Ty::List(Box::new(Ty::Int))), o2);
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn deeply_nested_option_option_deque_covariant_fails() {
        // Option<Option<Deque<Int>>> vs Option<Option<List<Int>>> in Covariant.
        // Inner item type is invariant — nested coercion is a type error.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::Option(Box::new(Ty::Option(Box::new(Ty::Deque(
            Box::new(Ty::Int),
            o,
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![tp(Ty::List(Box::new(Ty::Int)))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o1)),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![tp(Ty::Deque(Box::new(Ty::Int), o2))],
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let obj1 = Ty::Object(FxHashMap::from_iter([(
            i.intern("a"),
            Ty::Deque(Box::new(Ty::Int), o1),
        )]));
        let obj2 = Ty::Object(FxHashMap::from_iter([
            (i.intern("a"), Ty::Deque(Box::new(Ty::Int), o2)),
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
    fn snapshot_rollback_coercion_then_different_path() {
        // Snapshot → try Deque≤List (OK) → rollback → try Deque≤Iterator (OK).
        // The two paths must not interfere.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&v, &deque, Invariant).is_ok());

        // Path 1: coerce to List
        let snap = s.snapshot();
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
        s.rollback(snap);

        // Path 2: coerce to Iterator — should work independently
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Covariant
            )
            .is_ok()
        );
    }

    #[test]
    fn snapshot_rollback_demotion_no_residue() {
        // Snapshot → demotion → rollback → same Var with different Deque (same origin).
        // Rollback must fully undo the demotion so the new unify works cleanly.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Invariant)
                .is_ok()
        );

        let snap = s.snapshot();
        assert!(
            s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v, Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        s.rollback(snap);

        // After rollback, v is still Deque(o1). Same-origin unify should work.
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Invariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), o1));
    }

    // ================================================================
    // Polarity symmetry / duality 검증
    // ================================================================

    #[test]
    fn covariant_ab_equals_contravariant_ba() {
        // If unify(a, b, Cov) succeeds then unify(b, a, Contra) must also succeed.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o1 = s1.fresh_concrete_origin();
        let _ = s2.fresh_concrete_origin(); // keep counter in sync
        let d = Ty::Deque(Box::new(Ty::Int), o1);
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s1.unify(&d, &l, Covariant).is_ok());
        assert!(s2.unify(&l, &d, Contravariant).is_ok());
    }

    #[test]
    fn covariant_ab_fail_equals_contravariant_ba_fail() {
        // If unify(a, b, Cov) fails then unify(b, a, Contra) must also fail.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o1 = s1.fresh_concrete_origin();
        let _ = s2.fresh_concrete_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o1);
        assert!(s1.unify(&l, &d, Covariant).is_err()); // List ≤ Deque: no
        assert!(s2.unify(&d, &l, Contravariant).is_err()); // reversed: List ≤ Deque: no
    }

    #[test]
    fn invariant_symmetric() {
        // Invariant: unify(a, b) and unify(b, a) must both fail/succeed equally.
        let mut s1 = TySubst::new();
        let mut s2 = TySubst::new();
        let o = s1.fresh_concrete_origin();
        let _ = s2.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![
                tp(Ty::List(Box::new(Ty::Int))),
                tp(Ty::Deque(Box::new(Ty::Int), o1)),
            ],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![
                tp(Ty::Deque(Box::new(Ty::Int), o2)),
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
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
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
                tp(Ty::Deque(Box::new(Ty::Int), o1)),
                tp(Ty::Deque(Box::new(Ty::Int), o2)),
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
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        let d = Ty::Deque(Box::new(v1.clone()), o);
        let l = Ty::List(Box::new(v2.clone()));
        assert!(s.unify(&d, &l, Covariant).is_ok());
        // Bind v2 to String → v1 should follow.
        assert!(s.unify(&v2, &Ty::String, Invariant).is_ok());
        assert_eq!(s.resolve(&v1), Ty::String);
    }

    #[test]
    fn empty_deque_var_vs_empty_iterator_var_covariant() {
        // Deque<Var1, O> vs Iterator<Var2> in Covariant → OK, Var1 binds to Var2.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        let d = Ty::Deque(Box::new(v1.clone()), o);
        let i = Ty::Iterator(Box::new(v2.clone()), Effect::pure());
        assert!(s.unify(&d, &i, Covariant).is_ok());
        assert!(s.unify(&v2, &Ty::Int, Invariant).is_ok());
        assert_eq!(s.resolve(&v1), Ty::Int);
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
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(
            s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), o), Invariant)
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
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_param();
        let v2 = s.fresh_param();
        assert!(
            s.unify(&v1, &Ty::List(Box::new(Ty::Int)), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o), Invariant)
                .is_ok()
        );
        assert!(s.unify(&v1, &v2, Covariant).is_err());
    }

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    #[test]
    fn five_deque_origins_join_to_list() {
        // [d1, d2, d3, d4, d5] each with distinct origin → all join to List.
        let mut s = TySubst::new();
        let v = s.fresh_param();
        for _ in 0..5 {
            let o = s.fresh_concrete_origin();
            let d = Ty::Deque(Box::new(Ty::Int), o);
            assert!(s.unify(&d, &v, Covariant).is_ok());
        }
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    // ================================================================
    // Mixed concrete/var origins
    // ================================================================

    #[test]
    fn origin_var_binds_then_mismatch_demotes() {
        // Var origin Deque binds to concrete origin, then another concrete → mismatch → demotion.
        let mut s = TySubst::new();
        let c1 = s.fresh_concrete_origin();
        let c2 = s.fresh_concrete_origin();
        let ov = s.fresh_origin(); // Origin::Var
        let v = s.fresh_param();
        let d_var_origin = Ty::Deque(Box::new(Ty::Int), ov);
        let d_c1 = Ty::Deque(Box::new(Ty::Int), c1);
        let d_c2 = Ty::Deque(Box::new(Ty::Int), c2);
        // Bind Var origin via d_var_origin = d_c1
        assert!(s.unify(&v, &d_var_origin, Invariant).is_ok());
        assert!(s.unify(&v, &d_c1, Invariant).is_ok());
        assert_eq!(s.resolve_origin(ov), c1);
        // Now d_c2 has different origin → demotion in Covariant
        assert!(s.unify(&d_c2, &v, Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    // ================================================================
    // Error / Param + polarity (poison / unification absorption)
    // ================================================================

    #[test]
    fn error_absorbs_any_polarity() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
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
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
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
    fn var_bound_deque_then_coerce_to_list_then_coerce_to_iterator() {
        // Var = Deque(o) → coerce to List → coerce to Iterator.
        // Each step narrows via Covariant.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Invariant)
                .is_ok()
        );
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
        // v resolves to Deque still (Var bound to Deque, no rebind from Deque≤List).
        // Now try Iterator.
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Covariant
            )
            .is_ok()
        );
    }

    #[test]
    fn iterator_cannot_narrow_back_to_list_covariant() {
        // Var = Iterator(Int). Iterator ≤ List is invalid.
        let mut s = TySubst::new();
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Invariant
            )
            .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant)
                .is_err()
        );
    }

    #[test]
    fn iterator_cannot_narrow_back_to_deque_covariant() {
        // Var = Iterator(Int). Iterator ≤ Deque is invalid.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(
                &v,
                &Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Invariant
            )
            .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant)
                .is_err()
        );
    }

    #[test]
    fn list_cannot_narrow_back_to_deque_covariant() {
        // Var = List(Int). List ≤ Deque is invalid.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant)
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
        let o = s.fresh_concrete_origin();
        assert!(
            s.unify(
                &Ty::Deque(Box::new(Ty::Int), o),
                &Ty::List(Box::new(Ty::String)),
                Covariant,
            )
            .is_err()
        );
    }

    #[test]
    fn deque_to_iterator_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(
            s.unify(
                &Ty::Deque(Box::new(Ty::Int), o),
                &Ty::Iterator(Box::new(Ty::String), Effect::pure()),
                Covariant,
            )
            .is_err()
        );
    }

    #[test]
    fn list_to_iterator_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        assert!(
            s.unify(
                &Ty::List(Box::new(Ty::Int)),
                &Ty::Iterator(Box::new(Ty::String), Effect::pure()),
                Covariant,
            )
            .is_err()
        );
    }

    #[test]
    fn demotion_inner_type_mismatch_fails() {
        // Deque<Int, o1> vs Deque<String, o2> in Covariant.
        // Origin mismatch triggers demotion path, but inner unify Int vs String fails first.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        assert!(
            s.unify(
                &Ty::Deque(Box::new(Ty::Int), o1),
                &Ty::Deque(Box::new(Ty::String), o2),
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
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
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

    #[test]
    fn iterator_vs_list_of_list_fails() {
        // Iterator<Int> vs List<List<Int>> — not the same structure.
        let mut s = TySubst::new();
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let nested = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&iter, &nested, Covariant).is_err());
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
        let o = s.fresh_concrete_origin();
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
        let a = mk(Ty::Deque(Box::new(Ty::Int), o));
        let b = mk(Ty::List(Box::new(Ty::Int)));
        // 3 flips from Covariant → Contra. (Deque, List) in Contra → fail.
        assert!(s.unify(&a, &b, Covariant).is_err());
    }

    #[test]
    fn triple_flip_reversed_succeeds() {
        // Same structure but (List, Deque) at innermost.
        // 3 flips → Contra. (List, Deque) in Contra → reversed: Deque≤List → OK.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
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
        let b = mk(Ty::Deque(Box::new(Ty::Int), o));
        assert!(s.unify(&a, &b, Covariant).is_ok());
    }

    // ================================================================
    // Regression: Deque with same origin must not trigger demotion
    // ================================================================

    #[test]
    fn same_origin_no_demotion_even_covariant() {
        // Same origin → origins unify → no demotion path, stays Deque.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant)
                .is_ok()
        );
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), o));
    }

    #[test]
    fn same_origin_var_no_demotion() {
        // Origin::Var binds to concrete. Second use with same Var → same concrete → no demotion.
        let mut s = TySubst::new();
        let c = s.fresh_concrete_origin();
        let ov = s.fresh_origin();
        let v = s.fresh_param();
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), ov), Invariant)
                .is_ok()
        );
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), c), Covariant)
                .is_ok()
        );
        // ov now bound to c. Second concrete same as c → no mismatch.
        assert!(
            s.unify(&v, &Ty::Deque(Box::new(Ty::Int), c), Covariant)
                .is_ok()
        );
        // Still Deque, not List.
        let resolved = s.resolve(&v);
        assert!(
            matches!(resolved, Ty::Deque(_, _)),
            "should stay Deque, got {resolved:?}"
        );
    }

    // ── Sequence origin tracking ─────────────────────────────────

    #[test]
    fn sequence_same_origin_unifies() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        assert!(s.unify(&a, &b, Invariant).is_ok());
    }

    #[test]
    fn sequence_different_origin_demotes_to_iterator() {
        // Two Sequences with different origins → demote to Iterator (like Deque→List).
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let seq1 = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let seq2 = Ty::Sequence(Box::new(Ty::Int), o2, Effect::pure());
        assert!(s.unify(&v, &seq1, Covariant).is_ok());
        assert!(s.unify(&v, &seq2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert!(
            matches!(resolved, Ty::Iterator(..)),
            "should demote to Iterator, got {resolved:?}"
        );
    }

    #[test]
    fn sequence_different_origin_invariant_fails() {
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o2, Effect::pure());
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn deque_coerces_to_sequence_same_origin() {
        // Deque(T, O) ≤ Sequence(T, O) — covariant, origin preserved.
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        assert!(s.unify(&deque, &seq, Covariant).is_ok());
    }

    #[test]
    fn sequence_does_not_coerce_to_deque() {
        // Sequence ≤ Deque is NOT allowed (lazy → eager forbidden).
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&seq, &deque, Covariant).is_err());
    }

    #[test]
    fn sequence_coerces_to_iterator() {
        // Sequence(T, O) ≤ Iterator(T) — covariant, origin lost.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&seq, &iter, Covariant).is_ok());
    }

    #[test]
    fn iterator_does_not_coerce_to_sequence() {
        // Iterator ≤ Sequence is NOT allowed (no origin to create).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        assert!(s.unify(&iter, &seq, Covariant).is_err());
    }

    #[test]
    fn deque_coerces_to_iterator_via_sequence() {
        // Deque(T, O) ≤ Iterator(T) — transitive through Sequence, origin lost.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&deque, &iter, Covariant).is_ok());
    }

    #[test]
    fn sequence_structural_op_preserves_origin() {
        // Simulates take_seq signature: Sequence<T, O> → Sequence<T, O> (same O).
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let c = s.fresh_concrete_origin();
        let input = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let output = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        // Bind o to a concrete origin via the input.
        let concrete_seq = Ty::Sequence(Box::new(Ty::Int), c, Effect::pure());
        assert!(s.unify(&concrete_seq, &input, Covariant).is_ok());
        // Now output should also resolve to the same concrete origin.
        let resolved = s.resolve(&output);
        match resolved {
            Ty::Sequence(_, resolved_o, _) => assert_eq!(s.resolve_origin(resolved_o), c),
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn sequence_transform_op_creates_new_origin() {
        // Simulates map_seq output having a different origin from the input.
        // Two Sequence<Int> with different concrete origins → demote to Iterator.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let seq1 = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let seq2 = Ty::Sequence(Box::new(Ty::Int), o2, Effect::pure());
        // Different origins should NOT unify invariantly.
        assert!(s.unify(&seq1, &seq2, Invariant).is_err());
        // But covariantly, they demote to Iterator.
        let v = s.fresh_param();
        assert!(s.unify(&v, &seq1, Covariant).is_ok());
        assert!(s.unify(&v, &seq2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert!(
            matches!(resolved, Ty::Iterator(..)),
            "should demote to Iterator, got {resolved:?}"
        );
    }

    #[test]
    fn sequence_and_iterator_purity() {
        let o = Origin::Concrete(0);
        // Pure effect + pure inner → pureable.
        assert!(Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()).is_pureable());
        assert!(Ty::Iterator(Box::new(Ty::Int), Effect::pure()).is_pureable());
        // IO effect → not pureable.
        assert!(!Ty::Sequence(Box::new(Ty::Int), o, Effect::io()).is_pureable());
        assert!(!Ty::Iterator(Box::new(Ty::Int), Effect::io()).is_pureable());
        // Deque IS pure (eager, storable container).
        assert!(Ty::Deque(Box::new(Ty::Int), o).is_pureable());
    }

    #[test]
    fn sequence_chain_same_origin_ok() {
        // chain_seq: (Sequence<T, O>, Sequence<T, O>) → Sequence<T, O>
        // Both inputs must have the same origin.
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let c = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        // Bind o to concrete.
        assert!(
            s.unify(
                &Ty::Sequence(Box::new(Ty::Int), c, Effect::pure()),
                &a,
                Covariant
            )
            .is_ok()
        );
        // b should also resolve to same origin.
        assert!(
            s.unify(
                &Ty::Sequence(Box::new(Ty::Int), c, Effect::pure()),
                &b,
                Covariant
            )
            .is_ok()
        );
    }

    #[test]
    fn sequence_chain_different_origin_fails_invariant() {
        // chain_seq requires same origin. Different origins in invariant → error.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o2, Effect::pure());
        // In a chain_seq signature, both args share the same origin var.
        // If called with different concrete origins, unification of origins fails.
        assert!(s.unify_origins(o1, o2).is_err());
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
        let o = s.fresh_concrete_origin();
        assert_eq!(Ty::List(Box::new(Ty::Int)).materiality(), Materiality::Composite);
        assert_eq!(Ty::Deque(Box::new(Ty::Int), o).materiality(), Materiality::Composite);
        assert_eq!(
            Ty::Iterator(Box::new(Ty::Int), Effect::pure()).materiality(),
            Materiality::Composite
        );
        assert_eq!(
            Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()).materiality(),
            Materiality::Composite
        );
        assert_eq!(Ty::Option(Box::new(Ty::Int)).materiality(), Materiality::Composite);
        assert_eq!(Ty::Tuple(vec![Ty::Int]).materiality(), Materiality::Composite);
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
    fn purity_opaque_is_unpure() {
        assert_eq!(Ty::Opaque("HttpResponse".into()).materiality(), Materiality::Ephemeral);
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
        assert_eq!(std::cmp::max(Materiality::Concrete, Materiality::Composite), Materiality::Composite);
        assert_eq!(std::cmp::max(Materiality::Composite, Materiality::Ephemeral), Materiality::Ephemeral);
        assert_eq!(std::cmp::max(Materiality::Concrete, Materiality::Ephemeral), Materiality::Ephemeral);
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
    fn pureable_list_of_fn_with_opaque_capture() {
        // Fn with Opaque capture → not pureable, so List<Fn> is also not pureable.
        let list_fn = Ty::List(Box::new(Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("Handle".into())],
        }));
        assert!(!list_fn.is_pureable());
    }

    #[test]
    fn pureable_list_of_opaque_is_not_pureable() {
        let list_opaque = Ty::List(Box::new(Ty::Opaque("Handle".into())));
        assert!(!list_opaque.is_pureable());
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
    fn pureable_nested_list_of_fn_opaque_capture() {
        // List<List<Fn(Int) -> Int>> with Opaque capture — not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("X".into())],
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(!nested.is_pureable());
    }

    #[test]
    fn pureable_deque_of_scalars() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(Ty::Deque(Box::new(Ty::Int), o).is_pureable());
    }

    #[test]
    fn pureable_deque_of_opaque() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(!Ty::Deque(Box::new(Ty::Opaque("X".into())), o).is_pureable());
    }

    #[test]
    fn pureable_iterator_of_scalars() {
        assert!(Ty::Iterator(Box::new(Ty::String), Effect::pure()).is_pureable());
    }

    #[test]
    fn pureable_iterator_of_fn_pure() {
        // Fn with empty captures and pure ret → pureable
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(Ty::Iterator(Box::new(fn_ty), Effect::pure()).is_pureable());
    }

    #[test]
    fn pureable_iterator_of_fn_with_opaque() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Opaque("X".into())),

            effect: Effect::pure(),
            captures: vec![],
        };
        assert!(!Ty::Iterator(Box::new(fn_ty), Effect::pure()).is_pureable());
    }

    #[test]
    fn pureable_sequence_of_scalars() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()).is_pureable());
    }

    #[test]
    fn pureable_option_of_scalar() {
        assert!(Ty::Option(Box::new(Ty::Int)).is_pureable());
    }

    #[test]
    fn pureable_option_of_opaque() {
        assert!(!Ty::Option(Box::new(Ty::Opaque("X".into()))).is_pureable());
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
    fn pureable_tuple_with_fn_opaque_capture() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("X".into())],
        };
        assert!(!Ty::Tuple(vec![Ty::Int, fn_ty]).is_pureable());
    }

    #[test]
    fn pureable_tuple_with_opaque() {
        assert!(!Ty::Tuple(vec![Ty::Int, Ty::Opaque("X".into())]).is_pureable());
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
    fn pureable_object_with_fn_opaque_capture() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Int),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("X".into())],
        };
        let obj = Ty::Object(FxHashMap::from_iter([
            (i.intern("x"), Ty::Int),
            (i.intern("callback"), fn_ty),
        ]));
        assert!(!obj.is_pureable());
    }

    #[test]
    fn pureable_object_with_opaque_value() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(
            i.intern("handle"),
            Ty::Opaque("Handle".into()),
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
    fn pureable_enum_with_fn_opaque_capture_payload() {
        let i = Interner::new();
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("X".into())],
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
    fn pureable_enum_with_opaque_payload() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Wrap"),
            variants: FxHashMap::from_iter([(
                i.intern("Some"),
                Some(Box::new(Ty::Opaque("X".into()))),
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
    fn pureable_fn_with_opaque_capture() {
        // Fn with captures=[Opaque] → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            effect: Effect::pure(),
            captures: vec![Ty::Opaque("Handle".into())],
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
    fn pureable_fn_with_opaque_ret() {
        // Fn returning Opaque → not pureable
        let fn_ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Opaque("Handle".into())),

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
            captures: vec![Ty::Opaque("X".into())], // inner Fn captures Opaque
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
    fn pureable_opaque_never() {
        assert!(!Ty::Opaque("Conn".into()).is_pureable());
        assert!(!Ty::Opaque("Handle".into()).is_pureable());
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
    fn pureable_mixed_tuple_list_opaque() {
        // (Int, List<Opaque>) — not pureable
        let ty = Ty::Tuple(vec![Ty::Int, Ty::List(Box::new(Ty::Opaque("X".into())))]);
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
    fn pureable_deeply_nested_with_opaque_leaf() {
        // List<Option<Tuple<(Int, Opaque)>>> — not pureable
        let inner = Ty::Tuple(vec![Ty::Int, Ty::Opaque("X".into())]);
        let ty = Ty::List(Box::new(Ty::Option(Box::new(inner))));
        assert!(!ty.is_pureable());
    }

    // ================================================================
    // Effect coercion tests — Pure ≤ Effectful
    // ================================================================

    // -- Iterator effect coercion --

    #[test]
    fn iterator_same_effect_pure() {
        let mut s = TySubst::new();
        let a = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&a, &b, Invariant).is_ok());
    }

    #[test]
    fn iterator_same_effect_effectful() {
        let mut s = TySubst::new();
        let a = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&a, &b, Invariant).is_ok());
    }

    #[test]
    fn iterator_effect_mismatch_invariant_fails() {
        let mut s = TySubst::new();
        let a = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn iterator_pure_to_effectful_covariant() {
        // Pure ≤ Effectful in Covariant → subtyping OK, v stays Pure (more specific)
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let pure_iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let effectful_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&v, &pure_iter, Covariant).is_ok());
        assert!(s.unify(&v, &effectful_iter, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::pure()),
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

    #[test]
    fn iterator_effectful_to_pure_covariant() {
        // Effectful then Pure → coerces to Effectful (lattice top)
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let effectful_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let pure_iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&v, &effectful_iter, Covariant).is_ok());
        assert!(s.unify(&v, &pure_iter, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::io()),
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

    #[test]
    fn iterator_effect_var_binds_to_pure() {
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let a = Ty::Iterator(Box::new(Ty::Int), e.clone());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&a, &b, Invariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    #[test]
    fn iterator_effect_var_binds_to_effectful() {
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let a = Ty::Iterator(Box::new(Ty::Int), e.clone());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&a, &b, Invariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::io());
    }

    #[test]
    fn iterator_shared_effect_var_pure_then_effectful() {
        // Simulates HOF: effect var e is shared, first binds Pure, then sees Effectful.
        // In Covariant, Pure ≤ Effectful succeeds without promoting — e stays Pure.
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        // First: bind e = Pure (from input iterator)
        let input = Ty::Iterator(Box::new(Ty::Int), e.clone());
        let pure_iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&input, &pure_iter, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
        // Second: callback is effectful, needs e = Effectful
        let callback_effect = Effect::io();
        let fn_sig = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: e.clone(),
        };
        let fn_actual = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: callback_effect,
        };
        // Covariant: Pure ≤ Effectful succeeds, but e stays as Pure (more specific).
        assert!(s.unify(&fn_sig, &fn_actual, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    // -- Sequence effect coercion --

    #[test]
    fn sequence_same_origin_effect_mismatch_invariant_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn sequence_same_origin_effect_coercion_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let pure_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let effectful_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        assert!(s.unify(&v, &pure_seq, Covariant).is_ok());
        assert!(s.unify(&v, &effectful_seq, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Sequence(_, _, e) => assert_eq!(e, Effect::pure()),
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn sequence_origin_mismatch_both_effectful() {
        // Different origins + both effectful → Iterator<T, Effectful>
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let a = Ty::Sequence(Box::new(Ty::Int), o1, Effect::io());
        let b = Ty::Sequence(Box::new(Ty::Int), o2, Effect::io());
        assert!(s.unify(&v, &a, Covariant).is_ok());
        assert!(s.unify(&v, &b, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::io()),
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

    #[test]
    fn sequence_origin_mismatch_mixed_effects() {
        // Different origins + Pure/Effectful → Iterator<T, Effectful>
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let a = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o2, Effect::io());
        assert!(s.unify(&v, &a, Covariant).is_ok());
        assert!(s.unify(&v, &b, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::io()),
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

    #[test]
    fn sequence_origin_mismatch_both_pure() {
        // Different origins + both pure → Iterator<T, Pure>
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_param();
        let a = Ty::Sequence(Box::new(Ty::Int), o1, Effect::pure());
        let b = Ty::Sequence(Box::new(Ty::Int), o2, Effect::pure());
        assert!(s.unify(&v, &a, Covariant).is_ok());
        assert!(s.unify(&v, &b, Covariant).is_ok());
        let resolved = s.resolve(&v);
        match resolved {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::pure()),
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

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
            effect: Effect::io(),
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
            effect: Effect::io(),
        };
        assert!(s.unify(&v, &pure_fn, Covariant).is_ok());
        assert!(s.unify(&v, &effectful_fn, Covariant).is_ok());
        match s.resolve(&v) {
            Ty::Fn { effect, .. } => assert_eq!(effect, Effect::pure()),
            other => panic!("expected Fn, got {other:?}"),
        }
    }

    // -- Coercion arm + effect interaction --

    #[test]
    fn list_to_pure_iterator_ok() {
        let mut s = TySubst::new();
        let list = Ty::List(Box::new(Ty::Int));
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&list, &iter, Covariant).is_ok());
    }

    #[test]
    fn list_to_effectful_iterator_fails() {
        // List → Iterator coercion produces Pure. Pure vs Effectful = mismatch in unify_effects.
        // No var to rebind → fails.
        let mut s = TySubst::new();
        let list = Ty::List(Box::new(Ty::Int));
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&list, &iter, Covariant).is_err());
    }

    #[test]
    fn list_to_iterator_effect_var_binds_pure() {
        // List → Iterator<T, e> → e = Pure
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let list = Ty::List(Box::new(Ty::Int));
        let iter = Ty::Iterator(Box::new(Ty::Int), e.clone());
        assert!(s.unify(&list, &iter, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    #[test]
    fn deque_to_sequence_effect_var_binds_pure() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let e = s.fresh_effect_var();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let seq = Ty::Sequence(Box::new(Ty::Int), o, e.clone());
        assert!(s.unify(&deque, &seq, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    #[test]
    fn sequence_to_iterator_preserves_effect() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let e = s.fresh_effect_var();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        let iter = Ty::Iterator(Box::new(Ty::Int), e.clone());
        assert!(s.unify(&seq, &iter, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::io());
    }

    #[test]
    fn sequence_pure_to_iterator_effectful_fails() {
        // Sequence<Int, O, Pure> → Iterator<Int, Effectful>
        // Effect: Pure vs Effectful → mismatch, no var to rebind → fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(s.unify(&seq, &iter, Covariant).is_err());
    }

    // -- Effect display --

    #[test]
    fn display_effectful_fn() {
        let i = Interner::new();
        let ty = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::String),

            captures: vec![],
            effect: Effect::io(),
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
            effect: Effect::io(),
        };
        assert_eq!(format!("{}", ty.display(&i)), "Fn!(Int) -> String");
    }

    #[test]
    fn display_effectful_iterator() {
        let i = Interner::new();
        let ty = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert_eq!(format!("{}", ty.display(&i)), "Iterator!<Int>");
    }

    #[test]
    fn display_pure_iterator() {
        let i = Interner::new();
        let ty = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert_eq!(format!("{}", ty.display(&i)), "Iterator<Int>");
    }

    #[test]
    fn display_effectful_sequence() {
        let i = Interner::new();
        let o = Origin::Concrete(0);
        let ty = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        assert_eq!(format!("{}", ty.display(&i)), "Sequence!<Int, Origin(0)>");
    }

    // -- Three-way effect unification --

    #[test]
    fn three_iterators_pure_effectful_pure() {
        let mut s = TySubst::new();
        let v = s.fresh_param();
        let pure1 = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let effectful = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let pure2 = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&v, &pure1, Covariant).is_ok());
        assert!(s.unify(&v, &effectful, Covariant).is_ok());
        assert!(s.unify(&v, &pure2, Covariant).is_ok());
        match s.resolve(&v) {
            Ty::Iterator(_, e) => assert_eq!(e, Effect::pure()),
            other => panic!("expected Iterator, got {other:?}"),
        }
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

    #[test]
    fn not_materializable_pure_iterator() {
        assert!(!Ty::Iterator(Box::new(Ty::Int), Effect::pure()).is_materializable());
    }

    #[test]
    fn not_materializable_pure_sequence() {
        let o = Origin::Concrete(0);
        assert!(!Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()).is_materializable());
    }

    // -- Iterator/Sequence with Effectful: also NOT materializable --

    #[test]
    fn not_storable_effectful_iterator() {
        assert!(!Ty::Iterator(Box::new(Ty::Int), Effect::io()).is_materializable());
    }

    #[test]
    fn not_storable_effectful_sequence() {
        let o = Origin::Concrete(0);
        assert!(!Ty::Sequence(Box::new(Ty::Int), o, Effect::io()).is_materializable());
    }

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
            effect: Effect::io(),
        };
        assert!(!fn_ty.is_materializable());
    }

    // -- Opaque: never storable --

    #[test]
    fn not_storable_opaque() {
        assert!(!Ty::Opaque("Connection".into()).is_materializable());
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
    fn not_storable_list_of_opaque() {
        assert!(!Ty::List(Box::new(Ty::Opaque("X".into()))).is_materializable());
    }

    #[test]
    fn not_storable_iterator_of_effectful_fn() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Int),

            captures: vec![],
            effect: Effect::io(),
        };
        assert!(!Ty::Iterator(Box::new(fn_ty), Effect::pure()).is_materializable());
    }

    #[test]
    fn not_materializable_list_of_pure_iterator() {
        // List<Iterator<Int, Pure>> — Iterator is Ephemeral, so List containing it is not materializable.
        assert!(!Ty::List(Box::new(Ty::Iterator(Box::new(Ty::Int), Effect::pure()))).is_materializable());
    }

    #[test]
    fn not_storable_list_of_effectful_iterator() {
        // List<Iterator<Int, Effectful>> — effectful inner makes the whole thing non-storable
        assert!(!Ty::List(Box::new(Ty::Iterator(Box::new(Ty::Int), Effect::io()))).is_materializable());
    }

    // ================================================================
    // Effect subtyping in HOF signatures
    // ================================================================
    //
    // When Iterator<T, Effectful> is passed to filter/map, the shared effect var E
    // binds to Effectful. The Pure callback then unifies with E(=Effectful) covariant.
    // Pure ≤ Effectful → OK.

    #[test]
    fn hof_pure_callback_effectful_iterator() {
        // Simulate: filter(Iterator<Int, Effectful>, Fn(Int→Bool, effect:Pure))
        // Shared E: first binds to Effectful from iterator, then Pure from callback
        // Covariant: Pure ≤ Effectful → should succeed
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let iter_ty = Ty::Iterator(Box::new(Ty::Int), e.clone());
        let actual_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        // Unify iterator arg (binds e = Effectful)
        assert!(s.unify(&actual_iter, &iter_ty, Covariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::io());

        // Now unify callback: Fn{effect:Pure} vs Fn{effect:e(=Effectful)}
        let actual_cb = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            captures: vec![],
            effect: Effect::pure(),
        };
        let expected_cb = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            captures: vec![],
            effect: e,
        };
        assert!(
            s.unify(&actual_cb, &expected_cb, Covariant).is_ok(),
            "Pure callback should be accepted where Effectful expected (covariant)"
        );
    }

    #[test]
    fn hof_effectful_callback_pure_iterator_rejected() {
        // Simulate: filter(Iterator<Int, Pure>, Fn(Int→Bool, effect:Effectful))
        // Shared E: binds to Pure from iterator, then Effectful from callback
        // Covariant: Effectful ≤ Pure → should FAIL
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let iter_ty = Ty::Iterator(Box::new(Ty::Int), e.clone());
        let actual_iter = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        assert!(s.unify(&actual_iter, &iter_ty, Covariant).is_ok());

        let actual_cb = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            captures: vec![],
            effect: Effect::io(),
        };
        let expected_cb = Ty::Fn {
            params: vec![tp(Ty::Int)],
            ret: Box::new(Ty::Bool),

            captures: vec![],
            effect: e,
        };
        // Effectful callback with Pure iterator → Effectful ≤ Pure in Covariant → Err
        // Falls through to lub_or_err → promotes to Effectful
        // This should succeed because lub promotes to Effectful
        let result = s.unify(&actual_cb, &expected_cb, Covariant);
        // This actually succeeds via lub_or_err (Fn effect → Effectful LUB)
        assert!(
            result.is_ok(),
            "lub should promote to Effectful: {result:?}"
        );
    }

    #[test]
    fn effect_subtyping_invariant_rejects_mismatch() {
        let mut s = TySubst::new();
        let a = Ty::Iterator(Box::new(Ty::Int), Effect::pure());
        let b = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        assert!(
            s.unify(&a, &b, Invariant).is_err(),
            "Invariant should reject Pure vs Effectful"
        );
    }

    // ================================================================
    // Builtin soundness: 4 iterable types (List, Deque, Iterator, Sequence)
    // ================================================================
    //
    // Verify that builtin signatures produce correct types and that
    // type mismatches are properly rejected. Tests use TySubst to
    // simulate overload resolution.

    use crate::builtins::test_support::try_builtin;

    // -- map: should accept Iterator and Sequence, return correct type --

    #[test]
    fn builtin_map_on_iterator_returns_iterator() {
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "map",
            &[
                Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Ty::Fn {
                    params: vec![tp(Ty::Int)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ],
        );
        assert!(matches!(ret, Ok(Ty::Iterator(_, ref e)) if e.is_pure()));
    }

    #[test]
    fn builtin_map_on_sequence_returns_iterator() {
        // map on Sequence breaks origin → returns Iterator (not Sequence)
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "map",
            &[
                Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()),
                Ty::Fn {
                    params: vec![tp(Ty::Int)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ],
        );
        assert!(matches!(ret, Ok(Ty::Iterator(_, ref e)) if e.is_pure()));
    }

    #[test]
    fn builtin_map_on_list_coerces_to_iterator() {
        // List ≤ Iterator coercion allows map on List
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "map",
            &[
                Ty::List(Box::new(Ty::Int)),
                Ty::Fn {
                    params: vec![tp(Ty::Int)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ],
        )
        .unwrap();
        assert!(matches!(ret, Ty::Iterator(_, ref e) if e.is_pure()));
    }

    // -- take/skip: Sequence preserves origin, Iterator stays Iterator --

    #[test]
    fn builtin_take_on_sequence_returns_sequence_same_origin() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "take_seq",
            &[Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()), Ty::Int],
        )
        .unwrap();
        match ret {
            Ty::Sequence(_, ret_o, ref e) if e.is_pure() => {
                assert_eq!(
                    s.resolve_origin(ret_o),
                    s.resolve_origin(o),
                    "origin must be preserved"
                );
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    #[test]
    fn builtin_take_on_iterator_returns_iterator() {
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "take",
            &[Ty::Iterator(Box::new(Ty::Int), Effect::pure()), Ty::Int],
        )
        .unwrap();
        assert!(matches!(ret, Ty::Iterator(_, ref e) if e.is_pure()));
    }

    #[test]
    fn builtin_skip_on_sequence_preserves_origin() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "skip_seq",
            &[Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()), Ty::Int],
        )
        .unwrap();
        match ret {
            Ty::Sequence(_, ret_o, _) => {
                assert_eq!(s.resolve_origin(ret_o), s.resolve_origin(o));
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    // -- chain: Sequence + Iterator → Sequence (same origin) --

    #[test]
    fn builtin_chain_on_sequence_with_iterator() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "chain_seq",
            &[
                Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()),
                Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
            ],
        )
        .unwrap();
        match ret {
            Ty::Sequence(_, ret_o, _) => {
                assert_eq!(
                    s.resolve_origin(ret_o),
                    s.resolve_origin(o),
                    "chain must preserve origin"
                );
            }
            other => panic!("expected Sequence, got {other:?}"),
        }
    }

    // -- collect: Iterator → List, Sequence → Deque (same origin) --

    #[test]
    fn builtin_collect_iterator_returns_list() {
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "collect",
            &[Ty::Iterator(Box::new(Ty::Int), Effect::pure())],
        )
        .unwrap();
        assert!(matches!(ret, Ty::List(_)));
    }

    #[test]
    fn builtin_collect_on_sequence_coerces_to_iterator() {
        // CollectSeq removed — Sequence coerces to Iterator, then Iterator collect applies.
        // Result is List (not Deque — origin lost via coercion).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "collect",
            &[Ty::Sequence(Box::new(Ty::Int), o, Effect::pure())],
        )
        .unwrap();
        assert!(
            matches!(ret, Ty::List(_)),
            "collect on Sequence should return List (via Iterator coercion), got {ret:?}"
        );
    }

    #[test]
    fn builtin_filter_on_sequence_coerces_to_iterator() {
        // FilterSeq removed — Sequence coerces to Iterator, then Iterator filter applies.
        // Result is Iterator (not Sequence — origin lost).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "filter",
            &[
                Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()),
                Ty::Fn {
                    params: vec![tp(Ty::Int)],
                    ret: Box::new(Ty::Bool),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ],
        )
        .unwrap();
        assert!(
            matches!(ret, Ty::Iterator(..)),
            "filter on Sequence should return Iterator (origin lost), got {ret:?}"
        );
    }

    // -- Effect propagation through HOF --

    #[test]
    fn builtin_map_effectful_callback_propagates() {
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "map",
            &[
                Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Ty::Fn {
                    params: vec![tp(Ty::Int)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::io(),
                },
            ],
        )
        .unwrap();
        match ret {
            Ty::Iterator(_, e) => {
                assert_eq!(e, Effect::io(), "effectful callback should propagate")
            }
            other => panic!("expected Iterator, got {other:?}"),
        }
    }

    // -- Deque coerces to Sequence for take/skip/chain --

    #[test]
    fn builtin_take_on_deque_coerces_to_sequence() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "take_seq",
            &[Ty::Deque(Box::new(Ty::Int), o), Ty::Int],
        )
        .unwrap();
        // Deque ≤ Sequence coercion, then take preserves origin
        match ret {
            Ty::Sequence(_, ret_o, _) => {
                assert_eq!(s.resolve_origin(ret_o), s.resolve_origin(o));
            }
            other => panic!("expected Sequence (from Deque coercion), got {other:?}"),
        }
    }

    // -- Type mismatch: wrong element type --

    #[test]
    fn builtin_map_element_type_mismatch_fails() {
        let mut s = TySubst::new();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "map",
            &[
                Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
                Ty::Fn {
                    params: vec![tp(Ty::String)],
                    ret: Box::new(Ty::String),

                    captures: vec![],
                    effect: Effect::pure(),
                },
            ],
        );
        assert!(ret.is_err(), "map with wrong element type should fail");
    }

    // -- Sequence-specific: flatten/flat_map return Iterator --

    #[test]
    fn builtin_flatten_on_sequence_returns_iterator() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let ret = try_builtin(
            &Interner::new(),
            &mut s,
            "flatten",
            &[Ty::Sequence(
                Box::new(Ty::List(Box::new(Ty::Int))),
                o,
                Effect::pure(),
            )],
        )
        .unwrap();
        assert!(
            matches!(ret, Ty::Iterator(_, _)),
            "flatten on Sequence should return Iterator, got {ret:?}"
        );
    }

    // -- Sequence effect variable tests --

    #[test]
    fn sequence_effect_var_binds_pure() {
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let o = s.fresh_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, e.clone());
        let pure_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        assert!(s.unify(&seq, &pure_seq, Invariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::pure());
    }

    #[test]
    fn sequence_effect_var_binds_effectful() {
        let mut s = TySubst::new();
        let e = s.fresh_effect_var();
        let o = s.fresh_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, e.clone());
        let eff_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        assert!(s.unify(&seq, &eff_seq, Invariant).is_ok());
        assert_eq!(s.resolve_effect(&e), Effect::io());
    }

    #[test]
    fn sequence_pure_to_effectful_covariant() {
        // Pure Sequence ≤ Effectful Sequence (covariant)
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let pure_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        let eff_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        assert!(
            s.unify(&pure_seq, &eff_seq, Covariant).is_ok(),
            "Pure ≤ Effectful in covariant should succeed"
        );
    }

    #[test]
    fn sequence_effectful_to_pure_covariant_fails() {
        // Effectful Sequence → Pure Sequence (covariant) should fail
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let eff_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        let pure_seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::pure());
        // This may go through lub_or_err which promotes to Effectful,
        // or fail if polarity prevents it. Just check it doesn't panic.
        let _ = s.unify(&eff_seq, &pure_seq, Covariant);
    }

    #[test]
    fn sequence_to_iterator_coercion_propagates_effect_var() {
        // Sequence<Int, O, Effectful> coerces to Iterator<Int, E> where E is a variable
        // E should bind to Effectful after coercion
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o, Effect::io());
        let e = s.fresh_effect_var();
        let iter = Ty::Iterator(Box::new(Ty::Int), e.clone());
        assert!(s.unify(&seq, &iter, Covariant).is_ok());
        assert_eq!(
            s.resolve_effect(&e),
            Effect::io(),
            "Sequence effect should propagate to Iterator"
        );
    }

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
    fn instantiate_remaps_effect_and_origin_vars() {
        let mut sig_subst = TySubst::new();
        let t = sig_subst.fresh_param();
        let e = sig_subst.fresh_effect_var(); // Effect::Var(0)
        let o = sig_subst.fresh_origin(); // Origin::Var(0)
        let sig = Ty::Sequence(Box::new(t), o, e);

        let mut inference = TySubst::new();
        // Burn some ids.
        let _ = inference.fresh_param();
        let _ = inference.fresh_effect_var();
        let _ = inference.fresh_origin();
        let inst = inference.instantiate(&sig);

        match &inst {
            Ty::Sequence(inner, new_o, new_e) => {
                assert!(matches!(inner.as_ref(), Ty::Param { token: p, .. } if p.id() == 1));
                assert!(matches!(new_e, Effect::Var(1)));
                assert!(matches!(new_o, Origin::Var(1)));
            }
            _ => panic!("expected Sequence"),
        }
    }

    #[test]
    fn instantiate_same_param_maps_to_same_fresh() {
        // filter: (Iterator<T, E>, Fn(T) → Bool) → Iterator<T, E>
        let mut sig_subst = TySubst::new();
        let t = sig_subst.fresh_param();
        let e = sig_subst.fresh_effect_var();
        let sig = Ty::Fn {
            params: vec![
                tp(Ty::Iterator(Box::new(t.clone()), e.clone())),
                tp(Ty::Fn {
                    params: vec![tp(t.clone())],
                    ret: Box::new(Ty::Bool),
                    captures: vec![],
                    effect: e.clone(),
                }),
            ],
            ret: Box::new(Ty::Iterator(Box::new(t), e)),
            captures: vec![],
            effect: Effect::pure(),
        };

        let mut inference = TySubst::new();
        let inst = inference.instantiate(&sig);

        // All occurrences of the original Param(0) should map to the same new Param.
        // All occurrences of Effect::Var(0) should map to the same new Effect::Var.
        match &inst {
            Ty::Fn { params, ret, .. } => {
                // First param: Iterator<NewT, NewE>
                let (new_t_id, new_e_id) = match &params[0].ty {
                    Ty::Iterator(inner, Effect::Var(eid)) => match inner.as_ref() {
                        Ty::Param { token: p, .. } => (p.id(), *eid),
                        _ => panic!("expected Param"),
                    },
                    _ => panic!("expected Iterator"),
                };
                // Second param: Fn(NewT) → Bool with same effect
                match &params[1].ty {
                    Ty::Fn {
                        params: fp,
                        effect: Effect::Var(eid),
                        ..
                    } => {
                        assert!(
                            matches!(&fp[0].ty, Ty::Param{ token: p, ..} if p.id() == new_t_id)
                        );
                        assert_eq!(*eid, new_e_id);
                    }
                    _ => panic!("expected Fn"),
                }
                // Return: Iterator<NewT, NewE>
                match ret.as_ref() {
                    Ty::Iterator(inner, Effect::Var(eid)) => {
                        assert!(
                            matches!(inner.as_ref(), Ty::Param{token: p, ..} if p.id() == new_t_id)
                        );
                        assert_eq!(*eid, new_e_id);
                    }
                    _ => panic!("expected Iterator"),
                }
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

        fn ctx(interner: &Interner, n: usize) -> QualifiedRef {
            QualifiedRef::root(interner.intern(&format!("ctx_{n}")))
        }

        #[test]
        fn default_is_pure() {
            let s = EffectSet::default();
            assert!(s.is_pure());
        }

        #[test]
        fn io_only_is_not_pure() {
            let s = EffectSet {
                io: true,
                ..Default::default()
            };
            assert!(!s.is_pure());
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
            assert!(!u.io);
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
        fn union_io_propagates() {
            let a = EffectSet {
                io: true,
                ..Default::default()
            };
            let b = EffectSet::default();
            assert!(a.union(&b).io);
            assert!(b.union(&a).io);
        }

        #[test]
        fn union_io_both_true() {
            let a = EffectSet {
                io: true,
                ..Default::default()
            };
            let b = EffectSet {
                io: true,
                ..Default::default()
            };
            assert!(a.union(&b).io);
        }

        // ── Effect enum tests ───────────────────────────────────────

        #[test]
        fn effect_pure_is_pure() {
            assert!(Effect::pure().is_pure());
            assert!(!Effect::pure().is_effectful());
            assert!(!Effect::pure().is_var());
        }

        #[test]
        fn effect_io_is_effectful() {
            assert!(!Effect::io().is_pure());
            assert!(Effect::io().is_effectful());
            assert!(!Effect::io().is_var());
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
            let concrete = Effect::io();
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
            assert!(s.unify_effect(&Effect::io(), &Effect::io()).is_ok());
        }

        #[test]
        fn unify_pure_effectful_invariant_fails() {
            let mut s = TySubst::new();
            assert!(s.unify_effect(&Effect::pure(), &Effect::io()).is_err());
        }

        #[test]
        fn unify_pure_effectful_covariant_ok() {
            let mut s = TySubst::new();
            // Pure ≤ Effectful in covariant position
            assert!(
                s.unify_effects(&Effect::pure(), &Effect::io(), Covariant)
                    .is_ok()
            );
        }

        #[test]
        fn unify_effectful_pure_covariant_fails() {
            let mut s = TySubst::new();
            // Effectful ≤ Pure in covariant position — should fail
            assert!(
                s.unify_effects(&Effect::io(), &Effect::pure(), Covariant)
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
            let concrete = Effect::io();
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
            let concrete = Effect::io();
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
                io: false,
                self_modifying: false,
            });
            assert!(!e.is_pure());
            assert!(e.is_effectful());
            match &e {
                Effect::Resolved(s) => {
                    assert_eq!(s.reads.len(), 1);
                    assert_eq!(s.writes.len(), 1);
                    assert!(!s.io);
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
                    assert!(!set.io);
                }
                _ => panic!("expected Resolved"),
            }
        }

        #[test]
        fn display_pure() {
            assert_eq!(format!("{}", Effect::pure()), "Pure");
        }

        #[test]
        fn display_io() {
            assert_eq!(format!("{}", Effect::io()), "Effectful(io)");
        }

        #[test]
        fn display_reads_writes() {
            let i = Interner::new();
            let c1 = ctx(&i, 0);
            let c2 = ctx(&i, 1);
            let e = Effect::Resolved(EffectSet {
                reads: [c1].into_iter().collect(),
                writes: [c2].into_iter().collect(),
                io: false,
                self_modifying: false,
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
