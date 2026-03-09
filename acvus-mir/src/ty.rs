use std::collections::HashMap;
use std::fmt;

use acvus_utils::Astr;

use crate::user_type::UserTypeId;

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
    Object(HashMap<Astr, Ty>),
    Tuple(Vec<Ty>),
    Fn {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },
    Byte,
    /// Opaque type: user-defined, identified by name. No internal structure.
    Opaque(std::string::String),
    Option(Box<Ty>),
    /// User-defined enum type.
    UserType(UserTypeId),
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
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Int => write!(f, "Int"),
            Ty::Float => write!(f, "Float"),
            Ty::String => write!(f, "String"),
            Ty::Bool => write!(f, "Bool"),
            Ty::Unit => write!(f, "Unit"),
            Ty::Range => write!(f, "Range"),
            Ty::Byte => write!(f, "Byte"),
            Ty::List(inner) => write!(f, "List<{inner}>"),
            Ty::Object(fields) => {
                let mut sorted: Vec<_> = fields.iter().collect();
                sorted.sort_by_key(|(k, _)| format!("{k}"));
                write!(f, "{{")?;
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            Ty::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            Ty::Fn { params, ret } => {
                write!(f, "Fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            Ty::Option(inner) => write!(f, "Option<{inner}>"),
            Ty::Opaque(name) => write!(f, "{name}"),
            Ty::UserType(id) => write!(f, "UserType({})", id.0),
            Ty::Var(v) => write!(f, "?{}", v.0),
            Ty::Infer => write!(f, "<infer>"),
            Ty::Error => write!(f, "<error>"),
        }
    }
}

/// Substitution table for type unification.
pub struct TySubst {
    bindings: HashMap<TyVar, Ty>,
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
            bindings: HashMap::new(),
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
                let resolved: HashMap<_, _> = fields
                    .iter()
                    .map(|(k, v)| (*k, self.resolve(v)))
                    .collect();
                Ty::Object(resolved)
            }
            Ty::Tuple(elems) => Ty::Tuple(elems.iter().map(|e| self.resolve(e)).collect()),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(|p| self.resolve(p)).collect(),
                ret: Box::new(self.resolve(ret)),
            },
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
            // Error (poison) unifies with anything -- suppress cascading errors.
            (Ty::Error, _) | (_, Ty::Error) => Ok(()),

            (Ty::Int, Ty::Int)
            | (Ty::Float, Ty::Float)
            | (Ty::String, Ty::String)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Unit, Ty::Unit)
            | (Ty::Range, Ty::Range)
            | (Ty::Byte, Ty::Byte) => Ok(()),

            (Ty::Opaque(a), Ty::Opaque(b)) if a == b => Ok(()),
            (Ty::UserType(a), Ty::UserType(b)) if a == b => Ok(()),

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
                if fa.len() == fb.len() {
                    // Same-size: exact key match.
                    for (key, ty_a) in fa {
                        if let Some(ty_b) = fb.get(key) {
                            self.unify(ty_a, ty_b)?;
                        } else {
                            return Err((a.clone(), b.clone()));
                        }
                    }
                    Ok(())
                } else {
                    // Different sizes: subset matching.
                    // The smaller must be a subset of the larger, and must trace
                    // back to a Var (partial constraint from field projection).
                    let (smaller, larger, smaller_orig) = if fa.len() < fb.len() {
                        (fa, fb, orig_a)
                    } else {
                        (fb, fa, orig_b)
                    };

                    if let Some(leaf_var) = self.find_leaf_var(smaller_orig) {
                        for (key, ty_s) in smaller {
                            if let Some(ty_l) = larger.get(key) {
                                self.unify(ty_s, ty_l)?;
                            } else {
                                return Err((a.clone(), b.clone()));
                            }
                        }
                        // Rebind the leaf var to the fully-resolved larger Object.
                        let larger_resolved = Ty::Object(
                            larger
                                .iter()
                                .map(|(k, v)| (*k, self.resolve(v)))
                                .collect(),
                        );
                        self.bindings.insert(leaf_var, larger_resolved);
                        Ok(())
                    } else {
                        Err((a.clone(), b.clone()))
                    }
                }
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
        let obj1 = Ty::Object(HashMap::from([
            (interner.intern("name"), Ty::String),
            (interner.intern("age"), t.clone()),
        ]));
        let obj2 = Ty::Object(HashMap::from([
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
        let obj1 = Ty::Object(HashMap::from([(interner.intern("name"), Ty::String)]));
        let obj2 = Ty::Object(HashMap::from([(interner.intern("age"), Ty::Int)]));
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
}
