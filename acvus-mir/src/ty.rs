use std::fmt;

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

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
            Ty::Fn { .. } | Ty::Opaque(_) | Ty::Iterator(_) => false,
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
            Ty::Fn { params, ret, is_extern } => {
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
            Ty::Deque(inner, origin) => Ty::Deque(Box::new(self.resolve(inner)), self.resolve_origin(*origin)),
            Ty::Option(inner) => Ty::Option(Box::new(self.resolve(inner))),
            Ty::Object(fields) => {
                let resolved: FxHashMap<_, _> =
                    fields.iter().map(|(k, v)| (*k, self.resolve(v))).collect();
                Ty::Object(resolved)
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Fn { params, ret, is_extern } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
                is_extern: *is_extern,
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

    /// Unify two types. Returns Ok(()) on success, Err with the two conflicting
    /// types (after resolution) on failure.
    pub fn unify(&mut self, a: &Ty, b: &Ty) -> Result<(), (Ty, Ty)> {
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
                            (Some(ty_a), Some(ty_b)) => self.unify(ty_a, ty_b)?,
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
                    self.unify(ta, tb)?;
                }
                Ok(())
            }

            (Ty::Iterator(a), Ty::Iterator(b)) => self.unify(a, b),

            (Ty::List(a), Ty::List(b)) => self.unify(a, b),

            // Deque vs Deque: inner types unify, origins unify
            (Ty::Deque(ia, oa), Ty::Deque(ib, ob)) => {
                self.unify_origins(*oa, *ob).map_err(|_| (a.clone(), b.clone()))?;
                self.unify(ia, ib)
            }

            // Deque → List coercion (one-directional: Deque can become List, losing origin)
            (Ty::Deque(inner_d, _), Ty::List(inner_l)) => self.unify(inner_d, inner_l),

            // List → Iterator coercion (one-directional: List can be iterated)
            (Ty::List(inner_l), Ty::Iterator(inner_i)) => self.unify(inner_l, inner_i),

            // Deque → Iterator coercion (one-directional: Deque can be iterated)
            (Ty::Deque(inner_d, _), Ty::Iterator(inner_i)) => self.unify(inner_d, inner_i),

            (Ty::Option(a), Ty::Option(b)) => self.unify(a, b),

            (Ty::Object(fa), Ty::Object(fb)) => {
                // Unify overlapping fields.
                for (key, ty_a) in fa {
                    if let Some(ty_b) = fb.get(key) {
                        self.unify(ty_a, ty_b)?;
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
                },
                Ty::Fn {
                    params: pb,
                    ret: rb,
                    is_extern: eb,
                },
            ) => {
                if ea != eb || pa.len() != pb.len() {
                    return Err((a.clone(), b.clone()));
                }
                for (ta, tb) in pa.iter().zip(pb.iter()) {
                    self.unify(ta, tb)?;
                }
                self.unify(ra, rb)
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
            Ty::Deque(inner, _) => self.occurs_in(var, inner),
            Ty::Option(inner) => self.occurs_in(var, inner),
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            Ty::Object(fields) => fields.values().any(|v| self.occurs_in(var, v)),
            Ty::Fn { params, ret, .. } => {
                params.iter().any(|p| self.occurs_in(var, p)) || self.occurs_in(var, ret)
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

    #[test]
    fn unify_same_concrete() {
        let mut s = TySubst::new();
        assert!(s.unify(&Ty::Int, &Ty::Int).is_ok());
        assert!(s.unify(&Ty::Float, &Ty::Float).is_ok());
        assert!(s.unify(&Ty::String, &Ty::String).is_ok());
        assert!(s.unify(&Ty::Bool, &Ty::Bool).is_ok());
        assert!(s.unify(&Ty::Unit, &Ty::Unit).is_ok());
        assert!(s.unify(&Ty::Range, &Ty::Range).is_ok());
    }

    #[test]
    fn unify_different_concrete_fails() {
        let mut s = TySubst::new();
        assert!(s.unify(&Ty::Int, &Ty::Float).is_err());
        assert!(s.unify(&Ty::String, &Ty::Bool).is_err());
    }

    #[test]
    fn unify_var_with_concrete() {
        let mut s = TySubst::new();
        let t = s.fresh_var();
        assert!(s.unify(&t, &Ty::Int).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn unify_deque_of_var() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        let deque_int = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&deque_t, &deque_int).is_ok());
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
        };
        let fn_int_bool = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
            is_extern: false,
        };
        assert!(s.unify(&fn_tu, &fn_int_bool).is_ok());
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
        };
        let fn2 = Ty::Fn {
            params: vec![Ty::Int, Ty::Int],
            ret: Box::new(Ty::Int),
            is_extern: false,
        };
        assert!(s.unify(&fn1, &fn2).is_err());
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
        assert!(s.unify(&obj1, &obj2).is_ok());
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
        assert!(s.unify(&obj1, &obj2).is_err());
    }

    #[test]
    fn occurs_check() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let deque_t = Ty::Deque(Box::new(t.clone()), o);
        // T = Deque<T, O> should fail
        assert!(s.unify(&t, &deque_t).is_err());
    }

    #[test]
    fn transitive_resolution() {
        let mut s = TySubst::new();
        let t1 = s.fresh_var();
        let t2 = s.fresh_var();
        assert!(s.unify(&t1, &t2).is_ok());
        assert!(s.unify(&t2, &Ty::String).is_ok());
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
        assert!(s.unify(&v, &obj_a).is_ok());
        assert!(s.unify(&v, &obj_b).is_ok());
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
        assert!(s.unify(&v, &obj_ab).is_ok());
        assert!(s.unify(&v, &obj_bc).is_ok());
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
        assert!(s.unify(&v, &obj1).is_ok());
        assert!(s.unify(&v, &obj2).is_err());
    }

    // -- Deque type tests --

    #[test]
    fn unify_deque_same_origin() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let d1 = Ty::Deque(Box::new(t.clone()), o);
        let d2 = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&d1, &d2).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&d1), Ty::Deque(Box::new(Ty::Int), o));
    }

    #[test]
    fn unify_deque_different_concrete_origin_fails() {
        let mut s = TySubst::new();
        let o1 = s.fresh_concrete_origin();
        let o2 = s.fresh_concrete_origin();
        assert_ne!(o1, o2);
        let d1 = Ty::Deque(Box::new(Ty::Int), o1);
        let d2 = Ty::Deque(Box::new(Ty::Int), o2);
        assert!(s.unify(&d1, &d2).is_err(), "different concrete origins must not unify");
    }

    #[test]
    fn unify_deque_origin_var_binds_to_concrete() {
        // Origin::Var should bind to Origin::Concrete during unification
        let mut s = TySubst::new();
        let concrete = s.fresh_concrete_origin();
        let var = s.fresh_origin(); // Origin::Var
        let d1 = Ty::Deque(Box::new(Ty::Int), concrete);
        let d2 = Ty::Deque(Box::new(Ty::Int), var);
        assert!(s.unify(&d1, &d2).is_ok(), "origin Var should bind to Concrete");
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
        assert!(s.unify(&d_concrete, &d_var).is_ok());
        // Now a second concrete origin should NOT match the same var
        let concrete2 = s.fresh_concrete_origin();
        let d_concrete2 = Ty::Deque(Box::new(Ty::Int), concrete2);
        let d_var2 = Ty::Deque(Box::new(Ty::Int), var);
        assert!(s.unify(&d_concrete2, &d_var2).is_err(), "var already bound to different concrete");
    }

    #[test]
    fn unify_deque_inner_type_mismatch_fails() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d1 = Ty::Deque(Box::new(Ty::Int), o);
        let d2 = Ty::Deque(Box::new(Ty::String), o);
        assert!(s.unify(&d1, &d2).is_err(), "inner type mismatch with same origin must fail");
    }

    #[test]
    fn coerce_deque_to_iterator() {
        // Deque<Int, O> can be used where Iterator<Int> is expected
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&deque, &iter).is_ok(), "Deque → Iterator coercion should succeed");
    }

    #[test]
    fn coerce_deque_to_iterator_with_var() {
        // Deque<T, O> unifies with Iterator<Int> → T becomes Int
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let t = s.fresh_var();
        let deque = Ty::Deque(Box::new(t.clone()), o);
        let iter = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&deque, &iter).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
    }

    #[test]
    fn coerce_iterator_to_deque_fails() {
        // Iterator<Int> cannot become Deque<Int, O> — one-directional only
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let iter = Ty::Iterator(Box::new(Ty::Int));
        let deque = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&iter, &deque).is_err(), "Iterator → Deque coercion must be forbidden");
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
        assert!(s.unify(&t, &Ty::String).is_ok());
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
        assert!(s.unify(&t, &deque_t).is_err());
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
        assert!(s.unify(&deque, &iter).is_ok());
        // Now bind t2 to Int
        assert!(s.unify(&t2, &Ty::Int).is_ok());
        // t1 should also resolve to Int via transitive unification
        assert_eq!(s.resolve(&t1), Ty::Int);
    }

    #[test]
    fn unify_deque_coerces_to_list() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let d = Ty::Deque(Box::new(Ty::Int), o);
        let l = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&d, &l).is_ok(), "Deque should coerce to List");
    }

    #[test]
    fn unify_list_does_not_coerce_to_deque() {
        let mut s = TySubst::new();
        let o = s.fresh_origin();
        let l = Ty::List(Box::new(Ty::Int));
        let d = Ty::Deque(Box::new(Ty::Int), o);
        assert!(s.unify(&l, &d).is_err(), "List must not coerce to Deque");
    }

    #[test]
    fn unify_list_coerces_to_iterator() {
        let mut s = TySubst::new();
        let l = Ty::List(Box::new(Ty::Int));
        let i = Ty::Iterator(Box::new(Ty::Int));
        assert!(s.unify(&l, &i).is_ok(), "List should coerce to Iterator");
    }
}
