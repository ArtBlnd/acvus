use crate::ty::{Ty, TySubst};

pub struct BuiltinFn {
    pub name: &'static str,
    /// Returns (param_types, return_type) with fresh type variables for generics.
    pub signature: fn(&mut TySubst) -> (Vec<Ty>, Ty),
    pub is_effectful: bool,
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
        },
        BuiltinFn {
            name: "to_string",
            signature: |subst| {
                // to_string: (T) -> String
                let t = subst.fresh_var();
                (vec![t], Ty::String)
            },
            is_effectful: false,
        },
        BuiltinFn {
            name: "to_float",
            signature: |_| {
                // to_float: (Int) -> Float
                (vec![Ty::Int], Ty::Float)
            },
            is_effectful: false,
        },
        BuiltinFn {
            name: "to_int",
            signature: |_| {
                // to_int: (Float) -> Int
                (vec![Ty::Float], Ty::Int)
            },
            is_effectful: false,
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
        },
        BuiltinFn {
            name: "len",
            signature: |subst| {
                // len: (List<T>) -> Int
                let t = subst.fresh_var();
                (vec![Ty::List(Box::new(t))], Ty::Int)
            },
            is_effectful: false,
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
        },
    ]
}
