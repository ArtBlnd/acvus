use std::fmt;

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

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
/// `Pure` — scalars that can cross context boundaries as-is.
/// `Lazy` — containers, closures, iterators — need deep inspection to determine pureability.
/// `Unpure` — opaque types that can never be purified.
///
/// `Ord` derive: `Pure < Lazy < Unpure`, so `max()` gives the least-pure tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Purity {
    Pure,
    Lazy,
    Unpure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(pub u32);

/// Origin identity for Deque types — prevents mixing deques from different sources.
///
/// - `Concrete(u32)`: a fixed origin created by `[]` literals — unique provenance.
/// - `Var(u32)`: an origin variable created by builtin signatures (e.g. `extend`) —
///   binds to the actual origin of the input Deque during unification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        params: Vec<Ty>,
        ret: Box<Ty>,
        is_extern: bool,
        captures: Vec<Ty>,
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
    /// Lazy iterator over elements of type T.
    Iterator(Box<Ty>),
    /// Lazy sequence over elements of type T. Lazy version of Deque with origin identity.
    Sequence(Box<Ty>, Origin),
    /// Deque type: tracked deque with origin identity.
    /// `Origin` prevents mixing deques from different sources.
    Deque(Box<Ty>, Origin),
    /// Unification variable. Must not appear in final resolved types.
    Var(TyVar),
    /// Inferred type: signals the type checker to create a fresh Var internally.
    /// Input-only -- must not appear in output types.
    Infer,
    /// Poison type: produced after a type error. Unifies with anything to suppress cascading errors.
    Error,
}

impl Ty {
    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error)
    }

    /// Returns true if this type can be represented as a `PureValue` at runtime.
    /// Non-pure types (Fn, Opaque) can only be used in restricted contexts (e.g. call-only).
    #[deprecated(note = "use purity() or is_pureable()")]
    pub fn is_pure(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit
            | Ty::Range | Ty::Byte => true,
            Ty::List(inner) => inner.is_pure(),
            Ty::Deque(inner, _) => inner.is_pure(),
            Ty::Option(inner) => inner.is_pure(),
            Ty::Tuple(elems) => elems.iter().all(|e| e.is_pure()),
            Ty::Object(fields) => fields.values().all(|v| v.is_pure()),
            Ty::Enum { variants, .. } => variants.values().all(|p| {
                p.as_ref().map_or(true, |ty| ty.is_pure())
            }),
            Ty::Fn { .. } | Ty::Opaque(_) | Ty::Iterator(_) | Ty::Sequence(..) => false,
            Ty::Var(_) | Ty::Infer | Ty::Error => true,
        }
    }

    /// Returns the purity tier of this type (shallow — does not recurse into containers).
    pub fn purity(&self) -> Purity {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit
            | Ty::Range | Ty::Byte => Purity::Pure,
            Ty::List(_) | Ty::Deque(..) | Ty::Object(_) | Ty::Tuple(_)
            | Ty::Fn { .. } | Ty::Iterator(_) | Ty::Sequence(..)
            | Ty::Option(_) | Ty::Enum { .. } => Purity::Lazy,
            Ty::Opaque(_) => Purity::Unpure,
            Ty::Var(_) | Ty::Infer | Ty::Error => Purity::Pure,
        }
    }

    /// Returns true if this type can be deeply converted to a pure representation.
    /// Transitively checks container contents — `List<Fn>` returns false.
    pub fn is_pureable(&self) -> bool {
        match self {
            Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit
            | Ty::Range | Ty::Byte => true,
            Ty::List(inner) | Ty::Iterator(inner) => inner.is_pureable(),
            Ty::Deque(inner, _) | Ty::Sequence(inner, _) => inner.is_pureable(),
            Ty::Option(inner) => inner.is_pureable(),
            Ty::Tuple(elems) => elems.iter().all(|e| e.is_pureable()),
            Ty::Object(fields) => fields.values().all(|v| v.is_pureable()),
            Ty::Enum { variants, .. } => variants.values().all(|p| {
                p.as_ref().map_or(true, |ty| ty.is_pureable())
            }),
            Ty::Fn { captures, ret, .. } => {
                captures.iter().all(|c| c.is_pureable()) && ret.is_pureable()
            }
            Ty::Opaque(_) => false,
            Ty::Var(_) | Ty::Infer | Ty::Error => true,
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
            Ty::Fn { params, ret, is_extern, captures: _ } => {
                let prefix = if *is_extern { "ExternFn(" } else { "Fn(" };
                write!(f, "{prefix}")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.display(self.interner))?;
                }
                write!(f, ") -> {}", ret.display(self.interner))
            }
            Ty::List(inner) => write!(f, "List<{}>", inner.display(self.interner)),
            Ty::Iterator(inner) => write!(f, "Iterator<{}>", inner.display(self.interner)),
            Ty::Sequence(inner, origin) => write!(f, "Sequence<{}, {}>", inner.display(self.interner), origin),
            Ty::Deque(inner, origin) => write!(f, "Deque<{}, {}>", inner.display(self.interner), origin),
            Ty::Option(inner) => write!(f, "Option<{}>", inner.display(self.interner)),
            Ty::Opaque(name) => write!(f, "{name}"),
            Ty::Enum { name, .. } => write!(f, "{}", self.interner.resolve(*name)),
            Ty::Var(v) => write!(f, "?{}", v.0),
            Ty::Infer => write!(f, "<infer>"),
            Ty::Error => write!(f, "<error>"),
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
    bindings: FxHashMap<TyVar, Ty>,
    origin_bindings: FxHashMap<u32, Origin>,
    next_var: u32,
    next_origin: u32,
}

/// Substitution table for type unification.
pub struct TySubst {
    bindings: FxHashMap<TyVar, Ty>,
    origin_bindings: FxHashMap<u32, Origin>,
    next_var: u32,
    next_origin: u32,
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
            next_var: 0,
            next_origin: 0,
        }
    }

    /// Take a snapshot of the current state for later rollback.
    pub fn snapshot(&self) -> TySubstSnapshot {
        TySubstSnapshot {
            bindings: self.bindings.clone(),
            origin_bindings: self.origin_bindings.clone(),
            next_var: self.next_var,
            next_origin: self.next_origin,
        }
    }

    /// Restore state from a snapshot, discarding any bindings made since.
    pub fn rollback(&mut self, snap: TySubstSnapshot) {
        self.bindings = snap.bindings;
        self.origin_bindings = snap.origin_bindings;
        self.next_var = snap.next_var;
        self.next_origin = snap.next_origin;
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

    /// Allocate a fresh type variable.
    pub fn fresh_var(&mut self) -> Ty {
        let v = TyVar(self.next_var);
        self.next_var += 1;
        Ty::Var(v)
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
                if x == y { Ok(()) } else { Err((a, b)) }
            }
            (Origin::Var(v), other) | (other, Origin::Var(v)) => {
                if let Origin::Var(v2) = other {
                    if v == v2 { return Ok(()); }
                }
                self.origin_bindings.insert(v, other);
                Ok(())
            }
        }
    }

    /// Resolve a type by following substitution chains.
    pub fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(v) => {
                if let Some(bound) = self.bindings.get(v) {
                    self.resolve(bound)
                } else {
                    Ty::Var(*v)
                }
            }
            Ty::List(inner) => Ty::List(Box::new(self.resolve(inner))),
            Ty::Iterator(inner) => Ty::Iterator(Box::new(self.resolve(inner))),
            Ty::Sequence(inner, origin) => Ty::Sequence(Box::new(self.resolve(inner)), self.resolve_origin(*origin)),
            Ty::Deque(inner, origin) => Ty::Deque(Box::new(self.resolve(inner)), self.resolve_origin(*origin)),
            Ty::Option(inner) => Ty::Option(Box::new(self.resolve(inner))),
            Ty::Object(fields) => {
                let resolved: FxHashMap<_, _> =
                    fields.iter().map(|(k, v)| (*k, self.resolve(v))).collect();
                Ty::Object(resolved)
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Fn { params, ret, is_extern, captures } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
                is_extern: *is_extern,
                captures: captures.iter().map(|c| self.resolve(c)).collect(),
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

    /// Find the leaf Var in a chain that is bound to a concrete type.
    /// Returns None if `ty` is not a Var.
    pub fn find_leaf_var(&self, ty: &Ty) -> Option<TyVar> {
        match ty {
            Ty::Var(v) => {
                if let Some(bound) = self.bindings.get(v) {
                    match bound {
                        Ty::Var(_) => self.find_leaf_var(bound),
                        _ => Some(*v),
                    }
                } else {
                    Some(*v)
                }
            }
            _ => None,
        }
    }

    /// Rebind a type variable to a new type, replacing any existing binding.
    pub fn rebind(&mut self, var: TyVar, ty: Ty) {
        self.bindings.insert(var, ty);
    }

    /// Shallow-resolve: follow Var chains but don't recurse into structure.
    pub fn shallow_resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Var(v) => {
                if let Some(bound) = self.bindings.get(v) {
                    self.shallow_resolve(bound)
                } else {
                    Ty::Var(*v)
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
            // Error (poison) and Infer (unknown) unify with anything.
            (Ty::Error, _) | (_, Ty::Error) | (Ty::Infer, _) | (_, Ty::Infer) => Ok(()),

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
                let needs_merge = va.len() != vb.len()
                    || va.keys().any(|k| !vb.contains_key(k));
                if needs_merge {
                    let mut merged: FxHashMap<Astr, Option<Box<Ty>>> = va.clone();
                    for (tag, payload) in vb {
                        merged.entry(*tag).or_insert_with(|| payload.clone());
                    }
                    let merged_ty = Ty::Enum {
                        name: *na,
                        variants: merged,
                    };
                    if let Some(leaf) = self.find_leaf_var(orig_a) {
                        self.bindings.insert(leaf, merged_ty.clone());
                    }
                    if let Some(leaf) = self.find_leaf_var(orig_b) {
                        self.bindings.insert(leaf, merged_ty);
                    }
                }
                Ok(())
            }

            (Ty::Var(v), other) | (other, Ty::Var(v)) => {
                if let Ty::Var(v2) = other
                    && v == v2
                {
                    return Ok(());
                }
                if self.occurs_in(*v, other) {
                    return Err((a.clone(), b.clone()));
                }
                self.bindings.insert(*v, other.clone());
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

            (Ty::Iterator(a), Ty::Iterator(b)) => self.unify(a, b, pol),
            // Sequence vs Sequence: inner types unify, origins unify.
            // On origin mismatch with non-Invariant polarity → demote both to Iterator.
            (Ty::Sequence(ia, oa), Ty::Sequence(ib, ob)) => {
                match self.unify_origins(*oa, *ob) {
                    Ok(()) => self.unify(ia, ib, pol),
                    Err(_) => {
                        if pol == Polarity::Invariant {
                            return Err((a.clone(), b.clone()));
                        }
                        // Demote to Iterator<T>: unify inner types, then rebind Var leaves to Iterator.
                        self.unify(ia, ib, pol)?;
                        let inner_resolved = self.resolve(ia);
                        let iter_ty = Ty::Iterator(Box::new(inner_resolved));
                        if let Some(leaf) = self.find_leaf_var(orig_a) {
                            self.bindings.insert(leaf, iter_ty.clone());
                        }
                        if let Some(leaf) = self.find_leaf_var(orig_b) {
                            self.bindings.insert(leaf, iter_ty);
                        }
                        Ok(())
                    }
                }
            }

            (Ty::List(a), Ty::List(b)) => self.unify(a, b, pol),

            // Deque vs Deque: inner types unify, origins unify.
            // On origin mismatch with non-Invariant polarity → demote both to List.
            (Ty::Deque(ia, oa), Ty::Deque(ib, ob)) => {
                match self.unify_origins(*oa, *ob) {
                    Ok(()) => self.unify(ia, ib, pol),
                    Err(_) => {
                        if pol == Polarity::Invariant {
                            return Err((a.clone(), b.clone()));
                        }
                        // Demote to List<T>: unify inner types, then rebind Var leaves to List.
                        self.unify(ia, ib, pol)?;
                        let inner_resolved = self.resolve(ia);
                        let list_ty = Ty::List(Box::new(inner_resolved));
                        if let Some(leaf) = self.find_leaf_var(orig_a) {
                            self.bindings.insert(leaf, list_ty.clone());
                        }
                        if let Some(leaf) = self.find_leaf_var(orig_b) {
                            self.bindings.insert(leaf, list_ty);
                        }
                        Ok(())
                    }
                }
            }

            // Deque → List coercion: Covariant (a≤b) means Deque≤List OK.
            (Ty::Deque(inner_d, _), Ty::List(inner_l)) => {
                match pol {
                    Polarity::Covariant => self.unify(inner_d, inner_l, pol),
                    Polarity::Contravariant => Err((a, b)), // List ≤ Deque is invalid
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // List → Deque: Contravariant (b≤a) means Deque≤List OK.
            (Ty::List(inner_l), Ty::Deque(inner_d, _)) => {
                match pol {
                    Polarity::Contravariant => self.unify(inner_l, inner_d, pol),
                    Polarity::Covariant => Err((a, b)), // List ≤ Deque is invalid
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // List → Iterator coercion: Covariant (a≤b) means List≤Iterator OK.
            (Ty::List(inner_l), Ty::Iterator(inner_i)) => {
                match pol {
                    Polarity::Covariant => self.unify(inner_l, inner_i, pol),
                    Polarity::Contravariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Iterator → List: Contravariant (b≤a) means List≤Iterator OK.
            (Ty::Iterator(inner_i), Ty::List(inner_l)) => {
                match pol {
                    Polarity::Contravariant => self.unify(inner_i, inner_l, pol),
                    Polarity::Covariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Deque → Iterator coercion: Covariant (a≤b) means Deque≤Iterator OK.
            (Ty::Deque(inner_d, _), Ty::Iterator(inner_i)) => {
                match pol {
                    Polarity::Covariant => self.unify(inner_d, inner_i, pol),
                    Polarity::Contravariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Iterator → Deque: Contravariant (b≤a) means Deque≤Iterator OK.
            (Ty::Iterator(inner_i), Ty::Deque(inner_d, _)) => {
                match pol {
                    Polarity::Contravariant => self.unify(inner_i, inner_d, pol),
                    Polarity::Covariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Deque → Sequence coercion: Covariant (a≤b) means Deque≤Sequence OK (origin preserved).
            (Ty::Deque(inner_d, od), Ty::Sequence(inner_s, os)) => {
                match pol {
                    Polarity::Covariant => {
                        self.unify_origins(*od, *os).map_err(|_| (a.clone(), b.clone()))?;
                        self.unify(inner_d, inner_s, pol)
                    }
                    Polarity::Contravariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Sequence → Deque: Contravariant (b≤a) means Deque≤Sequence OK.
            (Ty::Sequence(inner_s, os), Ty::Deque(inner_d, od)) => {
                match pol {
                    Polarity::Contravariant => {
                        self.unify_origins(*os, *od).map_err(|_| (a.clone(), b.clone()))?;
                        self.unify(inner_s, inner_d, pol)
                    }
                    Polarity::Covariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Sequence → Iterator coercion: Covariant (a≤b) means Sequence≤Iterator OK (origin lost).
            (Ty::Sequence(inner_s, _), Ty::Iterator(inner_i)) => {
                match pol {
                    Polarity::Covariant => self.unify(inner_s, inner_i, pol),
                    Polarity::Contravariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            // Iterator → Sequence: Contravariant (b≤a) means Sequence≤Iterator OK.
            (Ty::Iterator(inner_i), Ty::Sequence(inner_s, _)) => {
                match pol {
                    Polarity::Contravariant => self.unify(inner_i, inner_s, pol),
                    Polarity::Covariant => Err((a, b)),
                    Polarity::Invariant => Err((a, b)),
                }
            }

            (Ty::Option(a), Ty::Option(b)) => self.unify(a, b, pol),

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
                let leaf_a = self.find_leaf_var(orig_a);
                let leaf_b = self.find_leaf_var(orig_b);

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
                    is_extern: ea,
                    captures: _,
                },
                Ty::Fn {
                    params: pb,
                    ret: rb,
                    is_extern: eb,
                    captures: _,
                },
            ) => {
                if ea != eb || pa.len() != pb.len() {
                    return Err((a.clone(), b.clone()));
                }
                // Function params are contravariant: flip polarity.
                let param_pol = pol.flip();
                for (ta, tb) in pa.iter().zip(pb.iter()) {
                    self.unify(ta, tb, param_pol)?;
                }
                // Return type keeps polarity.
                self.unify(ra, rb, pol)
            }

            _ => Err((a, b)),
        }
    }

    /// Occurs check: returns true if `var` appears in `ty`.
    fn occurs_in(&self, var: TyVar, ty: &Ty) -> bool {
        match ty {
            Ty::Var(v) => {
                if *v == var {
                    return true;
                }
                if let Some(bound) = self.bindings.get(v) {
                    self.occurs_in(var, bound)
                } else {
                    false
                }
            }
            Ty::List(inner) => self.occurs_in(var, inner),
            Ty::Iterator(inner) => self.occurs_in(var, inner),
            Ty::Sequence(inner, _) => self.occurs_in(var, inner),
            Ty::Deque(inner, _) => self.occurs_in(var, inner),
            Ty::Option(inner) => self.occurs_in(var, inner),
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            Ty::Object(fields) => fields.values().any(|v| self.occurs_in(var, v)),
            Ty::Fn { params, ret, captures, .. } => {
                params.iter().any(|p| self.occurs_in(var, p))
                    || self.occurs_in(var, ret)
                    || captures.iter().any(|c| self.occurs_in(var, c))
            }
            Ty::Enum { variants, .. } => variants
                .values()
                .any(|p| p.as_ref().map_or(false, |ty| self.occurs_in(var, ty))),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    use Polarity::*;

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
        let t = s.fresh_var();
        assert!(s.unify(&t, &Ty::Int, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn unify_deque_of_var() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        let deque_int = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&deque_t, &deque_int, Invariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&deque_t), Ty::Deque(Box::new(Ty::Int), o));
    }

    #[test]
    fn unify_fn_types() {
        let mut s = TySubst::new();
        let t = s.fresh_var();
        let u = s.fresh_var();
        let fn_tu = Ty::Fn {
            params: vec![t.clone()],
            ret: Box::new(u.clone()),
            is_extern: false,
                captures: vec![],
        };
        let fn_int_bool = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
            is_extern: false,
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
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
                captures: vec![],
        };
        let fn2 = Ty::Fn {
            params: vec![Ty::Int, Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
                captures: vec![],
        };
        assert!(s.unify(&fn1, &fn2, Invariant).is_err());
    }

    #[test]
    fn unify_object() {
        let mut s = TySubst::new();
        let interner = Interner::new();
        let t = s.fresh_var();
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
        let t = s.fresh_var();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        // T = Deque<T, O> should fail
        assert!(s.unify(&t, &deque_t, Invariant).is_err());
    }

    #[test]
    fn transitive_resolution() {
        let mut s = TySubst::new();
        let t1 = s.fresh_var();
        let t2 = s.fresh_var();
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
        let v = s.fresh_var();
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
        let v = s.fresh_var();
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
        let v = s.fresh_var();
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
        let t = s.fresh_var();
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
        assert!(s.unify(&d1, &d2, Invariant).is_err(), "different concrete origins must not unify in Invariant");
    }

    #[test]
    fn unify_deque_origin_var_binds_to_concrete() {
        // Origin::Var should bind to Origin::Concrete during unification
        let mut s = TySubst::new();
        let concrete = s.fresh_concrete_origin();
        let var = s.fresh_origin(); // Origin::Var
        let d1 = Ty::Deque(Box::new(Ty::Int), concrete);
        let d2 = Ty::Deque(Box::new(Ty::Int), var);
        assert!(s.unify(&d1, &d2, Invariant).is_ok(), "origin Var should bind to Concrete");
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
        assert!(s.unify(&d_concrete2, &d_var2, Invariant).is_err(), "var already bound to different concrete");
    }

    #[test]
    fn unify_deque_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o);
        let d2 = Ty::Deque(Box::new(Ty::String), o);
        assert!(s.unify(&d1, &d2, Invariant).is_err(), "inner type mismatch with same origin must fail");
    }

    #[test]
    fn coerce_deque_to_iterator() {
        // Deque<Int, O> can be used where Iterator<Int> is expected (Covariant)
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&deque, &iter, Covariant).is_ok(), "Deque → Iterator coercion should succeed");
    }

    #[test]
    fn coerce_deque_to_iterator_with_var() {
        // Deque<T, O> unifies with Iterator<Int> → T becomes Int
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let deque = Ty::Deque(Box::new(t.clone()), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&deque, &iter, Covariant).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn coerce_iterator_to_deque_fails() {
        // Iterator<Int> cannot become Deque<Int, O> — one-directional only
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let iter = Ty::Iterator(Box::new(Ty::Int));
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&iter, &deque, Covariant).is_err(), "Iterator → Deque coercion must be forbidden");
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
        let t = s.fresh_var();
        assert!(s.unify(&t, &Ty::String, Invariant).is_ok());
        let deque = Ty::Deque(Box::new(t.clone()), o);
        assert_eq!(s.resolve(&deque), Ty::Deque(Box::new(Ty::String), o));
    }

    #[test]
    fn occurs_in_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
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
        assert_eq!(o_after, Origin::Var(1), "rollback should restore origin counter");
    }

    #[test]
    fn deque_to_iterator_coercion_with_inner_var_unification() {
        // Deque<Var, O> vs Iterator<Var> where both Vars unify to same type
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t1 = s.fresh_var();
        let t2 = s.fresh_var();
        let deque = Ty::Deque(Box::new(t1.clone()), o);
        let iter = Ty::Iterator(Box::new(t2.clone()));
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
        assert!(s.unify(&d, &l, Covariant).is_ok(), "Deque should coerce to List");
    }

    #[test]
    fn unify_list_does_not_coerce_to_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&l, &d, Covariant).is_err(), "List must not coerce to Deque");
    }

    #[test]
    fn unify_list_coerces_to_iterator() {
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let i = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&l, &i, Covariant).is_ok(), "List should coerce to Iterator");
    }

    // -- Polarity-based subtyping tests --

    #[test]
    fn deque_origin_mismatch_covariant_demotes_to_list() {
        // Covariant: Deque+Deque origin mismatch → List demotion
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        // Bind v to d1, then unify v with d2 in Covariant → should demote to List
        assert!(s.unify(&v, &d1, Covariant).is_ok());
        assert!(s.unify(&v, &d2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert_eq!(resolved, Ty::List(Box::new(Ty::Int)), "should demote to List<Int>");
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
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o1)),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![Ty::Deque(Box::new(Ty::Int), o2)],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),
            is_extern: false,
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
        let elem_var = s.fresh_var();
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
        let v = s.fresh_var();
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        let d3 = Ty::Deque(Box::new(Ty::Int), o3);
        assert!(s.unify(&d1, &v, Covariant).is_ok());
        assert!(s.unify(&d2, &v, Covariant).is_ok(), "second deque should trigger demotion");
        // v is now List<Int>. Third deque: Deque≤List in Covariant should succeed.
        assert!(s.unify(&d3, &v, Covariant).is_ok(), "third deque should coerce to List via Deque≤List");
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn demotion_then_list_unifies() {
        // After demotion to List, unifying with another List should succeed.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant).is_ok());
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
        let inner_var = s.fresh_var();
        let v = s.fresh_var();
        assert!(s.unify(&Ty::Deque(Box::new(inner_var.clone()), o1), &v, Covariant).is_ok());
        assert!(s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v, Covariant).is_ok());
        // inner_var should now be Int, v should be List<Int>
        assert_eq!(s.resolve(&inner_var), Ty::Int);
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        // Third deque with same inner type
        assert!(s.unify(&Ty::Deque(Box::new(Ty::Int), o3), &v, Covariant).is_ok());
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
        assert!(s.unify(&d1, &d2, Covariant).is_err(), "inner type mismatch must fail regardless of demotion");
    }

    #[test]
    fn demotion_inner_type_still_var() {
        // Demotion when inner type is an unresolved Var — should resolve to List<Var>.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let inner = s.fresh_var();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(inner.clone()), o1), Covariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(inner.clone()), o2), Covariant).is_ok());
        // v should be List<inner_var>, inner still unresolved
        let resolved = s.resolve(&v);
        assert!(matches!(resolved, Ty::List(_)), "should be List, got {resolved:?}");
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
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Contravariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Contravariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn object_field_deque_coercion_covariant() {
        // {tags: Deque<String, o1>} vs {tags: List<String>} in Covariant.
        // Object field polarity is passed through → Deque≤List OK.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.fresh_concrete_origin();
        let obj_deque = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::Deque(Box::new(Ty::String), o)),
        ]));
        let obj_list = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::List(Box::new(Ty::String))),
        ]));
        assert!(s.unify(&obj_deque, &obj_list, Covariant).is_ok());
    }

    #[test]
    fn object_field_deque_coercion_invariant_fails() {
        // Same as above but Invariant — must fail.
        let mut s = TySubst::new();
        let i = Interner::new();
        let o = s.fresh_concrete_origin();
        let obj_deque = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::Deque(Box::new(Ty::String), o)),
        ]));
        let obj_list = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::List(Box::new(Ty::String))),
        ]));
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
        let v = s.fresh_var();
        let obj1 = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::Deque(Box::new(Ty::String), o1)),
        ]));
        let obj2 = Ty::Object(FxHashMap::from_iter([
            (i.intern("tags"), Ty::Deque(Box::new(Ty::String), o2)),
        ]));
        assert!(s.unify(&v, &obj1, Covariant).is_ok());
        assert!(s.unify(&v, &obj2, Covariant).is_ok());
    }

    #[test]
    fn option_deque_to_list_covariant() {
        // Option<Deque<Int>> vs Option<List<Int>> in Covariant.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let opt_deque = Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let opt_list = Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&opt_deque, &opt_list, Covariant).is_ok());
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
            params: vec![Ty::Deque(Box::new(Ty::Int), o)],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let inner_fn_b = Ty::Fn {
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let outer_a = Ty::Fn {
            params: vec![inner_fn_a],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let outer_b = Ty::Fn {
            params: vec![inner_fn_b],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let inner_fn_b = Ty::Fn {
            params: vec![Ty::Deque(Box::new(Ty::Int), o)],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let outer_a = Ty::Fn {
            params: vec![inner_fn_a],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let outer_b = Ty::Fn {
            params: vec![inner_fn_b],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o)),
            is_extern: false,
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
            params: vec![Ty::Deque(Box::new(Ty::Int), o)],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![Ty::Deque(Box::new(Ty::Int), o)],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        assert!(s.unify(&fn_a, &fn_b, Covariant).is_ok());
    }

    #[test]
    fn snapshot_rollback_undoes_demotion() {
        // Demotion should be fully undone by rollback.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant).is_ok());
        let snap = s.snapshot();
        let o2 = s.fresh_concrete_origin();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)), "demoted after second deque");
        s.rollback(snap);
        // After rollback, v should be back to Deque<Int, o1>
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), o1), "rollback should undo demotion");
    }

    #[test]
    fn demotion_then_iterator_coercion() {
        // After demotion to List, the result should still coerce to Iterator.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Covariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o2), Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        // List<Int> ≤ Iterator<Int> in Covariant
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Covariant).is_ok());
    }

    #[test]
    fn nested_list_of_deque_coercion() {
        // List<Deque<Int, o1>> vs List<List<Int>> in Covariant.
        // Inner: Deque≤List in Covariant → OK.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::List(Box::new(Ty::Deque(Box::new(Ty::Int), o)));
        let b = Ty::List(Box::new(Ty::List(Box::new(Ty::Int))));
        assert!(s.unify(&a, &b, Covariant).is_ok());
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
        let i = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&d, &i, Invariant).is_err());
    }

    #[test]
    fn list_to_iterator_invariant_fails() {
        // List → Iterator in Invariant must fail.
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let i = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&l, &i, Invariant).is_err());
    }

    #[test]
    fn deque_to_iterator_contravariant_fails() {
        // (Deque, Iterator) in Contravariant → reversed: Iterator≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let i = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&d, &i, Contravariant).is_err());
    }

    #[test]
    fn iterator_to_deque_contravariant_ok() {
        // (Iterator, Deque) in Contravariant → reversed: Deque≤Iterator → OK.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let i = Ty::Iterator(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&i, &d, Contravariant).is_ok());
    }

    #[test]
    fn chained_coercion_deque_to_iterator_covariant() {
        // Deque<Int> → Iterator<Int> directly in Covariant (skipping List).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant).is_ok());
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Covariant).is_ok());
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
            variants: FxHashMap::from_iter([
                (tag, Some(Box::new(Ty::Deque(Box::new(Ty::Int), o)))),
            ]),
        };
        let e2 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([
                (tag, Some(Box::new(Ty::List(Box::new(Ty::Int))))),
            ]),
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
            variants: FxHashMap::from_iter([
                (tag, Some(Box::new(Ty::Deque(Box::new(Ty::Int), o)))),
            ]),
        };
        let e2 = Ty::Enum {
            name,
            variants: FxHashMap::from_iter([
                (tag, Some(Box::new(Ty::List(Box::new(Ty::Int))))),
            ]),
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
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o), Invariant).is_ok());
        // v1 → v2 → Deque(Int, o). Now v1 as Deque ≤ List.
        assert!(s.unify(&v1, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
    }

    #[test]
    fn var_chain_demotion_rebinds_leaf() {
        // Var1 → Var2 → Deque(o1). Unify Var1 with Deque(o2) covariant → demotion.
        // find_leaf_var should follow chain and rebind Var2 (the leaf).
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        assert!(s.unify(&v1, &v2, Invariant).is_ok());
        assert!(s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o1), Invariant).is_ok());
        // Demotion via v1
        assert!(s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v1, Covariant).is_ok());
        assert_eq!(s.resolve(&v1), Ty::List(Box::new(Ty::Int)));
        assert_eq!(s.resolve(&v2), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn two_vars_sharing_deque_demotion_affects_both() {
        // Chain v2 → v1 while both unbound, THEN bind v1 → Deque(o1).
        // Demote via v2 → find_leaf_var follows v2 → v1 → rebinds v1 to List.
        // Both Var1 and Var2 should resolve to List.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        // Must chain BEFORE binding to concrete — otherwise shallow_resolve
        // flattens the chain and v2 binds directly to Deque, not to v1.
        assert!(s.unify(&v2, &v1, Invariant).is_ok());
        assert!(s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), o1), Invariant).is_ok());
        assert!(s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v2, Covariant).is_ok());
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
        let v = s.fresh_var();
        let cyclic = Ty::List(Box::new(v.clone()));
        assert!(s.unify(&v, &cyclic, Covariant).is_err());
    }

    #[test]
    fn occurs_check_through_deque_covariant() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_var();
        let cyclic = Ty::Deque(Box::new(v.clone()), o);
        assert!(s.unify(&v, &cyclic, Covariant).is_err());
    }

    #[test]
    fn occurs_check_through_fn_ret_covariant() {
        // Var = Fn() -> Var should fail (occurs) in any polarity.
        let mut s = TySubst::new();
        let v = s.fresh_var();
        let cyclic = Ty::Fn {
            params: vec![],
            ret: Box::new(v.clone()),
            is_extern: false,
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
        // Same outer origin, inner Deque≤List.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Deque(Box::new(Ty::Deque(Box::new(Ty::Int), o1)), o2);
        let b = Ty::Deque(Box::new(Ty::List(Box::new(Ty::Int))), o2);
        assert!(s.unify(&a, &b, Covariant).is_ok());
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
    fn deeply_nested_option_option_deque_covariant() {
        // Option<Option<Deque<Int>>> vs Option<Option<List<Int>>> in Covariant.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::Option(Box::new(Ty::Option(Box::new(Ty::Deque(Box::new(Ty::Int), o)))));
        let b = Ty::Option(Box::new(Ty::Option(Box::new(Ty::List(Box::new(Ty::Int))))));
        assert!(s.unify(&a, &b, Covariant).is_ok());
    }

    #[test]
    fn list_of_fn_with_coercion_in_param_and_ret() {
        // List<Fn(List<Int>) -> Deque<Int>>  vs  List<Fn(Deque<Int>) -> List<Int>>
        // in Covariant.
        // Inner Fn: params flip → Contra: List vs Deque → Deque≤List OK.
        //           ret keeps → Cov: Deque vs List → Deque≤List OK.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let fn_a = Ty::Fn {
            params: vec![Ty::List(Box::new(Ty::Int))],
            ret: Box::new(Ty::Deque(Box::new(Ty::Int), o1)),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![Ty::Deque(Box::new(Ty::Int), o2)],
            ret: Box::new(Ty::List(Box::new(Ty::Int))),
            is_extern: false,
                captures: vec![],
        };
        let a = Ty::List(Box::new(fn_a));
        let b = Ty::List(Box::new(fn_b));
        assert!(s.unify(&a, &b, Covariant).is_ok());
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
        let v = s.fresh_var();
        let obj1 = Ty::Object(FxHashMap::from_iter([
            (i.intern("a"), Ty::Deque(Box::new(Ty::Int), o1)),
        ]));
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
        let v = s.fresh_var();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&v, &deque, Invariant).is_ok());

        // Path 1: coerce to List
        let snap = s.snapshot();
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
        s.rollback(snap);

        // Path 2: coerce to Iterator — should work independently
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Covariant).is_ok());
    }

    #[test]
    fn snapshot_rollback_demotion_no_residue() {
        // Snapshot → demotion → rollback → same Var with different Deque (same origin).
        // Rollback must fully undo the demotion so the new unify works cleanly.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Invariant).is_ok());

        let snap = s.snapshot();
        assert!(s.unify(&Ty::Deque(Box::new(Ty::Int), o2), &v, Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::List(Box::new(Ty::Int)));
        s.rollback(snap);

        // After rollback, v is still Deque(o1). Same-origin unify should work.
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o1), Invariant).is_ok());
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
        assert!(s1.unify(&l, &d, Covariant).is_err());   // List ≤ Deque: no
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
                Ty::List(Box::new(Ty::Int)),
                Ty::Deque(Box::new(Ty::Int), o1),
            ],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![
                Ty::Deque(Box::new(Ty::Int), o2),
                Ty::List(Box::new(Ty::Int)),
            ],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
                Ty::List(Box::new(Ty::Int)),
                Ty::List(Box::new(Ty::Int)),
            ],
            ret: Box::new(Ty::Unit),
            is_extern: false,
                captures: vec![],
        };
        let fn_b = Ty::Fn {
            params: vec![
                Ty::Deque(Box::new(Ty::Int), o1),
                Ty::Deque(Box::new(Ty::Int), o2),
            ],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
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
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        let d = Ty::Deque(Box::new(v1.clone()), o);
        let i = Ty::Iterator(Box::new(v2.clone()));
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
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        assert!(s.unify(&v1, &Ty::Deque(Box::new(Ty::Int), o), Invariant).is_ok());
        assert!(s.unify(&v2, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(s.unify(&v1, &v2, Covariant).is_ok());
    }

    #[test]
    fn two_vars_coerce_list_to_deque_covariant_fails() {
        // Var1 = List(Int), Var2 = Deque(Int, o).
        // unify(Var1, Var2, Cov) → List≤Deque → fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v1 = s.fresh_var();
        let v2 = s.fresh_var();
        assert!(s.unify(&v1, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(s.unify(&v2, &Ty::Deque(Box::new(Ty::Int), o), Invariant).is_ok());
        assert!(s.unify(&v1, &v2, Covariant).is_err());
    }

    // ================================================================
    // N-way demotion (large fan-out)
    // ================================================================

    #[test]
    fn five_deque_origins_join_to_list() {
        // [d1, d2, d3, d4, d5] each with distinct origin → all join to List.
        let mut s = TySubst::new();
        let v = s.fresh_var();
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
        let v = s.fresh_var();
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
    // Error / Infer + polarity (poison absorption)
    // ================================================================

    #[test]
    fn error_absorbs_any_polarity() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&Ty::Error, &d, Covariant).is_ok());
        assert!(s.unify(&d, &Ty::Error, Contravariant).is_ok());
        assert!(s.unify(&Ty::Error, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
    }

    #[test]
    fn infer_absorbs_any_polarity() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&Ty::Infer, &d, Covariant).is_ok());
        assert!(s.unify(&d, &Ty::Infer, Contravariant).is_ok());
        assert!(s.unify(&Ty::Infer, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
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
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Invariant).is_ok());
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_ok());
        // v resolves to Deque still (Var bound to Deque, no rebind from Deque≤List).
        // Now try Iterator.
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Covariant).is_ok());
    }

    #[test]
    fn iterator_cannot_narrow_back_to_list_covariant() {
        // Var = Iterator(Int). Iterator ≤ List is invalid.
        let mut s = TySubst::new();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Covariant).is_err());
    }

    #[test]
    fn iterator_cannot_narrow_back_to_deque_covariant() {
        // Var = Iterator(Int). Iterator ≤ Deque is invalid.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Iterator(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant).is_err());
    }

    #[test]
    fn list_cannot_narrow_back_to_deque_covariant() {
        // Var = List(Int). List ≤ Deque is invalid.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::List(Box::new(Ty::Int)), Invariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant).is_err());
    }

    // ================================================================
    // Inner type mismatch under coercion (must not be masked)
    // ================================================================

    #[test]
    fn deque_to_list_inner_type_mismatch_fails() {
        // Deque<Int> ≤ List<String> → inner Int vs String fails.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(s.unify(
            &Ty::Deque(Box::new(Ty::Int), o),
            &Ty::List(Box::new(Ty::String)),
            Covariant,
        ).is_err());
    }

    #[test]
    fn deque_to_iterator_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(s.unify(
            &Ty::Deque(Box::new(Ty::Int), o),
            &Ty::Iterator(Box::new(Ty::String)),
            Covariant,
        ).is_err());
    }

    #[test]
    fn list_to_iterator_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        assert!(s.unify(
            &Ty::List(Box::new(Ty::Int)),
            &Ty::Iterator(Box::new(Ty::String)),
            Covariant,
        ).is_err());
    }

    #[test]
    fn demotion_inner_type_mismatch_fails() {
        // Deque<Int, o1> vs Deque<String, o2> in Covariant.
        // Origin mismatch triggers demotion path, but inner unify Int vs String fails first.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        assert!(s.unify(
            &Ty::Deque(Box::new(Ty::Int), o1),
            &Ty::Deque(Box::new(Ty::String), o2),
            Covariant,
        ).is_err());
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
        let iter = Ty::Iterator(Box::new(Ty::Int));
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
                params: vec![Ty::Fn {
                    params: vec![Ty::Fn {
                        params: vec![inner_param],
                        ret: Box::new(Ty::Unit),
                        is_extern: false,
                                        captures: vec![],
                    }],
                    ret: Box::new(Ty::Unit),
                    is_extern: false,
                                captures: vec![],
                }],
                ret: Box::new(Ty::Unit),
                is_extern: false,
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
                params: vec![Ty::Fn {
                    params: vec![Ty::Fn {
                        params: vec![inner_param],
                        ret: Box::new(Ty::Unit),
                        is_extern: false,
                                        captures: vec![],
                    }],
                    ret: Box::new(Ty::Unit),
                    is_extern: false,
                                captures: vec![],
                }],
                ret: Box::new(Ty::Unit),
                is_extern: false,
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
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), o), Covariant).is_ok());
        assert_eq!(s.resolve(&v), Ty::Deque(Box::new(Ty::Int), o));
    }

    #[test]
    fn same_origin_var_no_demotion() {
        // Origin::Var binds to concrete. Second use with same Var → same concrete → no demotion.
        let mut s = TySubst::new();
        let c = s.fresh_concrete_origin();
        let ov = s.fresh_origin();
        let v = s.fresh_var();
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), ov), Invariant).is_ok());
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), c), Covariant).is_ok());
        // ov now bound to c. Second concrete same as c → no mismatch.
        assert!(s.unify(&v, &Ty::Deque(Box::new(Ty::Int), c), Covariant).is_ok());
        // Still Deque, not List.
        let resolved = s.resolve(&v);
        assert!(matches!(resolved, Ty::Deque(_, _)), "should stay Deque, got {resolved:?}");
    }

    // ── Sequence origin tracking ─────────────────────────────────

    #[test]
    fn sequence_same_origin_unifies() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o);
        let b = Ty::Sequence(Box::new(Ty::Int), o);
        assert!(s.unify(&a, &b, Invariant).is_ok());
    }

    #[test]
    fn sequence_different_origin_demotes_to_iterator() {
        // Two Sequences with different origins → demote to Iterator (like Deque→List).
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let v = s.fresh_var();
        let seq1 = Ty::Sequence(Box::new(Ty::Int), o1);
        let seq2 = Ty::Sequence(Box::new(Ty::Int), o2);
        assert!(s.unify(&v, &seq1, Covariant).is_ok());
        assert!(s.unify(&v, &seq2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert!(matches!(resolved, Ty::Iterator(_)), "should demote to Iterator, got {resolved:?}");
    }

    #[test]
    fn sequence_different_origin_invariant_fails() {
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o1);
        let b = Ty::Sequence(Box::new(Ty::Int), o2);
        assert!(s.unify(&a, &b, Invariant).is_err());
    }

    #[test]
    fn deque_coerces_to_sequence_same_origin() {
        // Deque(T, O) ≤ Sequence(T, O) — covariant, origin preserved.
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let seq = Ty::Sequence(Box::new(Ty::Int), o);
        assert!(s.unify(&deque, &seq, Covariant).is_ok());
    }

    #[test]
    fn sequence_does_not_coerce_to_deque() {
        // Sequence ≤ Deque is NOT allowed (lazy → eager forbidden).
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o);
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&seq, &deque, Covariant).is_err());
    }

    #[test]
    fn sequence_coerces_to_iterator() {
        // Sequence(T, O) ≤ Iterator(T) — covariant, origin lost.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let seq = Ty::Sequence(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&seq, &iter, Covariant).is_ok());
    }

    #[test]
    fn iterator_does_not_coerce_to_sequence() {
        // Iterator ≤ Sequence is NOT allowed (no origin to create).
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let iter = Ty::Iterator(Box::new(Ty::Int));
        let seq = Ty::Sequence(Box::new(Ty::Int), o);
        assert!(s.unify(&iter, &seq, Covariant).is_err());
    }

    #[test]
    fn deque_coerces_to_iterator_via_sequence() {
        // Deque(T, O) ≤ Iterator(T) — transitive through Sequence, origin lost.
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&deque, &iter, Covariant).is_ok());
    }

    #[test]
    fn sequence_structural_op_preserves_origin() {
        // Simulates take_seq signature: Sequence<T, O> → Sequence<T, O> (same O).
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let c = s.fresh_concrete_origin();
        let input = Ty::Sequence(Box::new(Ty::Int), o);
        let output = Ty::Sequence(Box::new(Ty::Int), o);
        // Bind o to a concrete origin via the input.
        let concrete_seq = Ty::Sequence(Box::new(Ty::Int), c);
        assert!(s.unify(&concrete_seq, &input, Covariant).is_ok());
        // Now output should also resolve to the same concrete origin.
        let resolved = s.resolve(&output);
        match resolved {
            Ty::Sequence(_, resolved_o) => assert_eq!(s.resolve_origin(resolved_o), c),
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
        let seq1 = Ty::Sequence(Box::new(Ty::Int), o1);
        let seq2 = Ty::Sequence(Box::new(Ty::Int), o2);
        // Different origins should NOT unify invariantly.
        assert!(s.unify(&seq1, &seq2, Invariant).is_err());
        // But covariantly, they demote to Iterator.
        let v = s.fresh_var();
        assert!(s.unify(&v, &seq1, Covariant).is_ok());
        assert!(s.unify(&v, &seq2, Covariant).is_ok());
        let resolved = s.resolve(&v);
        assert!(matches!(resolved, Ty::Iterator(_)), "should demote to Iterator, got {resolved:?}");
    }

    #[test]
    fn sequence_is_not_pure() {
        let o = Origin::Concrete(0);
        assert!(!Ty::Sequence(Box::new(Ty::Int), o).is_pure());
        assert!(!Ty::Iterator(Box::new(Ty::Int)).is_pure());
        // Deque IS pure (it's an eager, storable container).
        assert!(Ty::Deque(Box::new(Ty::Int), o).is_pure());
    }

    #[test]
    fn sequence_chain_same_origin_ok() {
        // chain_seq: (Sequence<T, O>, Sequence<T, O>) → Sequence<T, O>
        // Both inputs must have the same origin.
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let c = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o);
        let b = Ty::Sequence(Box::new(Ty::Int), o);
        // Bind o to concrete.
        assert!(s.unify(&Ty::Sequence(Box::new(Ty::Int), c), &a, Covariant).is_ok());
        // b should also resolve to same origin.
        assert!(s.unify(&Ty::Sequence(Box::new(Ty::Int), c), &b, Covariant).is_ok());
    }

    #[test]
    fn sequence_chain_different_origin_fails_invariant() {
        // chain_seq requires same origin. Different origins in invariant → error.
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        let a = Ty::Sequence(Box::new(Ty::Int), o1);
        let b = Ty::Sequence(Box::new(Ty::Int), o2);
        // In a chain_seq signature, both args share the same origin var.
        // If called with different concrete origins, unification of origins fails.
        assert!(s.unify_origins(o1, o2).is_err());
    }

    // ── Purity tier tests ──────────────────────────────────────────────

    #[test]
    fn purity_scalars_are_pure() {
        assert_eq!(Ty::Int.purity(), Purity::Pure);
        assert_eq!(Ty::Float.purity(), Purity::Pure);
        assert_eq!(Ty::String.purity(), Purity::Pure);
        assert_eq!(Ty::Bool.purity(), Purity::Pure);
        assert_eq!(Ty::Unit.purity(), Purity::Pure);
        assert_eq!(Ty::Range.purity(), Purity::Pure);
        assert_eq!(Ty::Byte.purity(), Purity::Pure);
    }

    #[test]
    fn purity_containers_are_lazy() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert_eq!(Ty::List(Box::new(Ty::Int)).purity(), Purity::Lazy);
        assert_eq!(Ty::Deque(Box::new(Ty::Int), o).purity(), Purity::Lazy);
        assert_eq!(Ty::Iterator(Box::new(Ty::Int)).purity(), Purity::Lazy);
        assert_eq!(Ty::Sequence(Box::new(Ty::Int), o).purity(), Purity::Lazy);
        assert_eq!(Ty::Option(Box::new(Ty::Int)).purity(), Purity::Lazy);
        assert_eq!(Ty::Tuple(vec![Ty::Int]).purity(), Purity::Lazy);
    }

    #[test]
    fn purity_object_is_lazy() {
        let i = Interner::new();
        let obj = Ty::Object(FxHashMap::from_iter([(i.intern("x"), Ty::Int)]));
        assert_eq!(obj.purity(), Purity::Lazy);
    }

    #[test]
    fn purity_fn_is_lazy() {
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::String),
            is_extern: false,
            captures: vec![],
        };
        assert_eq!(fn_ty.purity(), Purity::Lazy);
    }

    #[test]
    fn purity_extern_fn_is_lazy() {
        let fn_ty = Ty::Fn {
            params: vec![Ty::String],
            ret: Box::new(Ty::Int),
            is_extern: true,
            captures: vec![],
        };
        assert_eq!(fn_ty.purity(), Purity::Lazy);
    }

    #[test]
    fn purity_enum_is_lazy() {
        let i = Interner::new();
        let enum_ty = Ty::Enum {
            name: i.intern("Color"),
            variants: FxHashMap::from_iter([
                (i.intern("Red"), None),
                (i.intern("Green"), None),
            ]),
        };
        assert_eq!(enum_ty.purity(), Purity::Lazy);
    }

    #[test]
    fn purity_opaque_is_unpure() {
        assert_eq!(Ty::Opaque("HttpResponse".into()).purity(), Purity::Unpure);
    }

    #[test]
    fn purity_special_types() {
        assert_eq!(Ty::Error.purity(), Purity::Pure);
        assert_eq!(Ty::Infer.purity(), Purity::Pure);
        assert_eq!(Ty::Var(TyVar(0)).purity(), Purity::Pure);
    }

    #[test]
    fn purity_ord_pure_lt_lazy_lt_unpure() {
        assert!(Purity::Pure < Purity::Lazy);
        assert!(Purity::Lazy < Purity::Unpure);
        assert!(Purity::Pure < Purity::Unpure);
        // max() gives least-pure tier
        assert_eq!(std::cmp::max(Purity::Pure, Purity::Lazy), Purity::Lazy);
        assert_eq!(std::cmp::max(Purity::Lazy, Purity::Unpure), Purity::Unpure);
        assert_eq!(std::cmp::max(Purity::Pure, Purity::Unpure), Purity::Unpure);
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
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
            captures: vec![],
        }));
        assert!(list_fn.is_pureable());
    }

    #[test]
    fn pureable_list_of_fn_with_opaque_capture() {
        // Fn with Opaque capture → not pureable, so List<Fn> is also not pureable.
        let list_fn = Ty::List(Box::new(Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
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
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
            captures: vec![],
        };
        let nested = Ty::List(Box::new(Ty::List(Box::new(fn_ty))));
        assert!(nested.is_pureable());
    }

    #[test]
    fn pureable_nested_list_of_fn_opaque_capture() {
        // List<List<Fn(Int) -> Int>> with Opaque capture — not pureable
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
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
        assert!(Ty::Iterator(Box::new(Ty::String)).is_pureable());
    }

    #[test]
    fn pureable_iterator_of_fn_pure() {
        // Fn with empty captures and pure ret → pureable
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),
            is_extern: false,
            captures: vec![],
        };
        assert!(Ty::Iterator(Box::new(fn_ty)).is_pureable());
    }

    #[test]
    fn pureable_iterator_of_fn_with_opaque() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Opaque("X".into())),
            is_extern: false,
            captures: vec![],
        };
        assert!(!Ty::Iterator(Box::new(fn_ty)).is_pureable());
    }

    #[test]
    fn pureable_sequence_of_scalars() {
        let mut s = TySubst::new();
        let o = s.fresh_concrete_origin();
        assert!(Ty::Sequence(Box::new(Ty::Int), o).is_pureable());
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
            is_extern: false,
            captures: vec![],
        };
        assert!(Ty::Tuple(vec![Ty::Int, fn_ty]).is_pureable());
    }

    #[test]
    fn pureable_tuple_with_fn_opaque_capture() {
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::Unit),
            is_extern: false,
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
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
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
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
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
        let obj = Ty::Object(FxHashMap::from_iter([
            (i.intern("handle"), Ty::Opaque("Handle".into())),
        ]));
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
            variants: FxHashMap::from_iter([
                (i.intern("Red"), None),
                (i.intern("Green"), None),
            ]),
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
            is_extern: false,
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
            is_extern: false,
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
            variants: FxHashMap::from_iter([
                (i.intern("Some"), Some(Box::new(Ty::Opaque("X".into())))),
            ]),
        };
        assert!(!enum_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_pure_captures_and_ret() {
        // Fn with captures=[Int, String] and ret=Bool → pureable
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
            is_extern: false,
            captures: vec![Ty::Int, Ty::String],
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_opaque_capture() {
        // Fn with captures=[Opaque] → not pureable
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
            is_extern: false,
            captures: vec![Ty::Opaque("Handle".into())],
        };
        assert!(!fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_fn_capture() {
        // Fn with captures=[Fn(Int)->Int (no captures)] → pureable (Fn with empty captures + pure ret)
        let inner_fn = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
            captures: vec![],
        };
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
            is_extern: false,
            captures: vec![inner_fn],
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_opaque_ret() {
        // Fn returning Opaque → not pureable
        let fn_ty = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Opaque("Handle".into())),
            is_extern: false,
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
            is_extern: false,
            captures: vec![],
        };
        assert!(fn_ty.is_pureable());
    }

    #[test]
    fn pureable_fn_with_list_fn_ret() {
        // Fn returning List<Fn(Int)->Int> → not pureable (transitive)
        let inner_fn = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
            captures: vec![Ty::Opaque("X".into())], // inner Fn captures Opaque
        };
        let fn_ty = Ty::Fn {
            params: vec![],
            ret: Box::new(Ty::List(Box::new(inner_fn))),
            is_extern: false,
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
        let ty = Ty::Tuple(vec![
            Ty::Int,
            Ty::List(Box::new(Ty::Opaque("X".into()))),
        ]);
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
}
