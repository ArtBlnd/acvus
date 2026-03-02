use crate::ty::{Ty, TySubst};

/// Post-unification constraint on resolved arg types.
/// Returns `Some(error_message)` if the constraint is violated.
pub type BuiltinConstraint = fn(&[Ty]) -> Option<String>;

pub struct BuiltinFn {
    pub name: &'static str,
    /// Returns (param_types, return_type) with fresh type variables for generics.
    pub signature: fn(&mut TySubst) -> (Vec<Ty>, Ty),
    pub is_effectful: bool,
    pub constraint: Option<BuiltinConstraint>,
}

fn is_scalar(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Byte)
}

fn require_scalar(args: &[Ty]) -> Option<String> {
    match &args[0] {
        ty if is_scalar(ty) => None,
        Ty::Var(_) | Ty::Error => None, // not yet resolved or error — skip
        ty => Some(format!(
            "`to_string` requires a scalar type (Int, Float, Bool, String, Byte), got {ty}",
        )),
    }
}

fn require_to_int(args: &[Ty]) -> Option<String> {
    match &args[0] {
        Ty::Float | Ty::Byte => None,
        Ty::Var(_) | Ty::Error => None,
        ty => Some(format!(
            "`to_int` requires Float or Byte, got {ty}",
        )),
    }
}

pub fn builtins() -> Vec<BuiltinFn> {
    vec![
        BuiltinFn {
            name: "filter",
            signature: |subst| {
                // filter: (List<T>, Fn(T) -> Bool) -> List<T>
                let t = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t.clone()],
                            ret: Box::new(Ty::Bool),
                        },
                    ],
                    Ty::List(Box::new(t)),
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "map",
            signature: |subst| {
                // map: (List<T>, Fn(T) -> U) -> List<U>
                let t = subst.fresh_var();
                let u = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t],
                            ret: Box::new(u.clone()),
                        },
                    ],
                    Ty::List(Box::new(u)),
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "pmap",
            signature: |subst| {
                // pmap: (List<T>, Fn(T) -> U) -> List<U>
                let t = subst.fresh_var();
                let u = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t],
                            ret: Box::new(u.clone()),
                        },
                    ],
                    Ty::List(Box::new(u)),
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "to_string",
            signature: |subst| {
                // to_string: (T) -> String
                let t = subst.fresh_var();
                (vec![t], Ty::String)
            },
            is_effectful: false,
            constraint: Some(require_scalar),
        },
        BuiltinFn {
            name: "to_float",
            signature: |_| {
                // to_float: (Int) -> Float
                (vec![Ty::Int], Ty::Float)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "to_int",
            signature: |subst| {
                // to_int: (Float | Byte) -> Int
                let t = subst.fresh_var();
                (vec![t], Ty::Int)
            },
            is_effectful: false,
            constraint: Some(require_to_int),
        },
        BuiltinFn {
            name: "find",
            signature: |subst| {
                // find: (List<T>, Fn(T) -> Bool) -> T
                let t = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t.clone()],
                            ret: Box::new(Ty::Bool),
                        },
                    ],
                    t,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "reduce",
            signature: |subst| {
                // reduce: (List<T>, Fn(T, T) -> T) -> T
                let t = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t.clone(), t.clone()],
                            ret: Box::new(t.clone()),
                        },
                    ],
                    t,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "fold",
            signature: |subst| {
                // fold: (List<T>, U, Fn(U, T) -> U) -> U
                let t = subst.fresh_var();
                let u = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        u.clone(),
                        Ty::Fn {
                            params: vec![u.clone(), t],
                            ret: Box::new(u.clone()),
                        },
                    ],
                    u,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "any",
            signature: |subst| {
                // any: (List<T>, Fn(T) -> Bool) -> Bool
                let t = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t],
                            ret: Box::new(Ty::Bool),
                        },
                    ],
                    Ty::Bool,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "all",
            signature: |subst| {
                // all: (List<T>, Fn(T) -> Bool) -> Bool
                let t = subst.fresh_var();
                (
                    vec![
                        Ty::List(Box::new(t.clone())),
                        Ty::Fn {
                            params: vec![t],
                            ret: Box::new(Ty::Bool),
                        },
                    ],
                    Ty::Bool,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "len",
            signature: |subst| {
                // len: (List<T>) -> Int
                let t = subst.fresh_var();
                (vec![Ty::List(Box::new(t))], Ty::Int)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "reverse",
            signature: |subst| {
                // reverse: (List<T>) -> List<T>
                let t = subst.fresh_var();
                (
                    vec![Ty::List(Box::new(t.clone()))],
                    Ty::List(Box::new(t)),
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "join",
            signature: |_| {
                // join: (List<String>, String) -> String
                (
                    vec![Ty::List(Box::new(Ty::String)), Ty::String],
                    Ty::String,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "char_to_int",
            signature: |_| {
                // char_to_int: (String) -> Int
                (vec![Ty::String], Ty::Int)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "int_to_char",
            signature: |_| {
                // int_to_char: (Int) -> String
                (vec![Ty::Int], Ty::String)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "contains",
            signature: |subst| {
                // contains: (List<T>, T) -> Bool
                let t = subst.fresh_var();
                (
                    vec![Ty::List(Box::new(t.clone())), t],
                    Ty::Bool,
                )
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "substring",
            signature: |_| {
                // substring: (String, Int, Int) -> String
                (vec![Ty::String, Ty::Int, Ty::Int], Ty::String)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "len_str",
            signature: |_| {
                // len_str: (String) -> Int
                (vec![Ty::String], Ty::Int)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "to_bytes",
            signature: |_| {
                // to_bytes: (String) -> List<Byte>
                (vec![Ty::String], Ty::bytes())
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "to_utf8",
            signature: |_| {
                // to_utf8: (List<Byte>) -> String
                (vec![Ty::bytes()], Ty::String)
            },
            is_effectful: false,
            constraint: None,
        },
        BuiltinFn {
            name: "to_utf8_lossy",
            signature: |_| {
                // to_utf8_lossy: (List<Byte>) -> String
                (vec![Ty::bytes()], Ty::String)
            },
            is_effectful: false,
            constraint: None,
        },
    ]
}
