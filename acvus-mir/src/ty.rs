use std::fmt;

use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TyVar(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Int,
    Float,
    String,
    Bool,
    Unit,
    Range,
    List(Box<Ty>),
    Object(FxHashMap<Astr, Ty>),
    Tuple(Vec<Ty>),
    Fn {
        params: Vec<Ty>,
        ret: Box<Ty>,
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

    /// Convenience: `List<Byte>` (replaces the old `Ty::Bytes`).
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
            Ty::List(inner) => write!(f, "List<{}>", inner.display(self.interner)),
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
            Ty::Fn { params, ret } => {
                write!(f, "Fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.display(self.interner))?;
                }
                write!(f, ") -> {}", ret.display(self.interner))
            }
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

/// Substitution table for type unification.
pub struct TySubst {
    bindings: FxHashMap<TyVar, Ty>,
    next_var: u32,
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
            next_var: 0,
        }
    }

    /// Allocate a fresh type variable.
    pub fn fresh_var(&mut self) -> Ty {
        let v = TyVar(self.next_var);
        self.next_var += 1;
        Ty::Var(v)
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
            Ty::Option(inner) => Ty::Option(Box::new(self.resolve(inner))),
            Ty::Object(fields) => {
                let resolved: FxHashMap<_, _> =
                    fields.iter().map(|(k, v)| (*k, self.resolve(v))).collect();
                Ty::Object(resolved)
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
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
    fn shallow_resolve(&self, ty: &Ty) -> Ty {
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

            (Ty::List(la), Ty::List(lb)) => self.unify(la, lb),
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
                },
                Ty::Fn {
                    params: pb,
                    ret: rb,
                },
            ) => {
                if pa.len() != pb.len() {
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
            Ty::Option(inner) => self.occurs_in(var, inner),
            Ty::Tuple(elems) => elems.iter().any(|e| self.occurs_in(var, e)),
            Ty::Object(fields) => fields.values().any(|v| self.occurs_in(var, v)),
            Ty::Fn { params, ret } => {
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
    fn unify_list_of_var() {
        let mut s = TySubst::new();
        let t = s.fresh_var();
        let list_t = Ty::List(Box::new(t.clone()));
        let list_int = Ty::List(Box::new(Ty::Int));
        assert!(s.unify(&list_t, &list_int).is_ok());
        assert_eq!(s.resolve(&t), Ty::Int);
        assert_eq!(s.resolve(&list_t), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn unify_fn_types() {
        let mut s = TySubst::new();
        let t = s.fresh_var();
        let u = s.fresh_var();
        let fn_tu = Ty::Fn {
            params: vec![t.clone()],
            ret: Box::new(u.clone()),
        };
        let fn_int_bool = Ty::Fn {
            params: vec![Ty::Int],
            ret: Box::new(Ty::Bool),
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
        };
        let fn2 = Ty::Fn {
            params: vec![Ty::Int, Ty::Int],
            ret: Box::new(Ty::Int),
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
        let t = s.fresh_var();
        let list_t = Ty::List(Box::new(t.clone()));
        // T = List<T> should fail
        assert!(s.unify(&t, &list_t).is_err());
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
}
