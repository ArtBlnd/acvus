use crate::ty::{Ty, TySubst};

pub struct BuiltinFn {
    pub name: &'static str,
    /// Returns (param_types, return_type) with fresh type variables for generics.
    pub signature: fn(&mut TySubst) -> (Vec<Ty>, Ty),
    pub is_sync: bool,
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
            is_sync: true,
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
            is_sync: true,
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
            is_sync: true,
        },
        BuiltinFn {
            name: "to_string",
            signature: |subst| {
                // to_string: (T) -> String
                let t = subst.fresh_var();
                (vec![t], Ty::String)
            },
            is_sync: true,
        },
        BuiltinFn {
            name: "to_float",
            signature: |_| {
                // to_float: (Int) -> Float
                (vec![Ty::Int], Ty::Float)
            },
            is_sync: true,
        },
        BuiltinFn {
            name: "to_int",
            signature: |_| {
                // to_int: (Float) -> Int
                (vec![Ty::Float], Ty::Int)
            },
            is_sync: true,
        },
    ]
}
