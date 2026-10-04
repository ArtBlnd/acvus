//! A written type's form, computed through traits rather than read from
//! syntax. An extern-defined type with type arguments is written in one of
//! two forms: uniform, every argument a `Uniform` at its root, or whole and
//! native, no `Uniform` anywhere in it, which is marked `#`. Any other form
//! specializes at a type argument and fails the build, as does a whole native
//! extern-defined type standing as an argument of a uniform one, since a mark
//! stands at a root alone (rfc_compiler types §8).
//!
//! The types below stand for the extern crate's and an author's. A runtime
//! parameter is left out, since no form depends on it, and a type variable is
//! written as its stand-in, `Uniform<Var<N>>`, as a declaration writes it.

use std::marker::PhantomData;

/// A value held uniformly.
pub struct Uniform<X: ?Sized>(PhantomData<fn(&X) -> &X>);

/// A declaration's `N`-th type variable, standing inside a `Uniform`.
pub struct Var<const N: u32>;

pub trait Form {
    /// A `Uniform` stands somewhere in the type.
    const HOLDS_UNIFORM: bool;
    /// The type is a `Uniform` at its root.
    const UNIFORM_AT_ROOT: bool;
    /// A whole native extern-defined type with type arguments stands
    /// somewhere in the type, itself included.
    const HOLDS_MARKED: bool;
    /// The type is a whole native extern-defined type with type arguments.
    const MARKED: bool;
    /// Evaluating it fails the build where an extern-defined type in the
    /// type is in neither form.
    const CHECKED: ();
}

/// The check of one extern-defined type, from its arguments' facts, each in
/// order.
pub const fn check_extern(uniform_at_root: &[bool], holds_uniform: &[bool], holds_marked: &[bool]) {
    if uniform_at_root.is_empty() {
        return;
    }
    if all(uniform_at_root) {
        if any(holds_marked) {
            panic!("a whole native extern-defined type stands as an argument of a uniform one");
        }
        return;
    }
    if any(holds_uniform) {
        panic!("an extern-defined type's arguments are neither all uniform at their root nor all native");
    }
}

/// Whether an extern-defined type with these arguments is whole and native.
pub const fn marked(holds_uniform: &[bool]) -> bool {
    !holds_uniform.is_empty() && !any(holds_uniform)
}

pub const fn any(facts: &[bool]) -> bool {
    let mut at = 0;
    while at < facts.len() {
        if facts[at] {
            return true;
        }
        at += 1;
    }
    false
}

pub const fn all(facts: &[bool]) -> bool {
    let mut at = 0;
    while at < facts.len() {
        if !facts[at] {
            return false;
        }
        at += 1;
    }
    true
}

impl<X: Form + ?Sized> Form for Uniform<X> {
    const HOLDS_UNIFORM: bool = true;
    const UNIFORM_AT_ROOT: bool = true;
    const HOLDS_MARKED: bool = X::HOLDS_MARKED;
    const MARKED: bool = false;
    const CHECKED: () = X::CHECKED;
}

impl<const N: u32> Form for Var<N> {
    const HOLDS_UNIFORM: bool = false;
    const UNIFORM_AT_ROOT: bool = false;
    const HOLDS_MARKED: bool = false;
    const MARKED: bool = false;
    const CHECKED: () = ();
}

macro_rules! leaves {
    ($($leaf:ty),* $(,)?) => {$(
        impl Form for $leaf {
            const HOLDS_UNIFORM: bool = false;
            const UNIFORM_AT_ROOT: bool = false;
            const HOLDS_MARKED: bool = false;
            const MARKED: bool = false;
            const CHECKED: () = ();
        }
    )*};
}

leaves!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, (), String, str);

/// A structure the language defines: what its parts hold, and checked
/// part by part.
macro_rules! structures {
    ($([$($param:ident),+] $ty:ty;)*) => {$(
        impl<$($param: Form + ?Sized),+> Form for $ty {
            const HOLDS_UNIFORM: bool = any(&[$($param::HOLDS_UNIFORM),+]);
            const UNIFORM_AT_ROOT: bool = false;
            const HOLDS_MARKED: bool = any(&[$($param::HOLDS_MARKED),+]);
            const MARKED: bool = false;
            const CHECKED: () = { $(let () = $param::CHECKED;)+ };
        }
    )*};
}

structures! {
    [X] &X;
    [X] &mut X;
}

macro_rules! sized_structures {
    ($([$($param:ident),+] $ty:ty;)*) => {$(
        impl<$($param: Form),+> Form for $ty {
            const HOLDS_UNIFORM: bool = any(&[$($param::HOLDS_UNIFORM),+]);
            const UNIFORM_AT_ROOT: bool = false;
            const HOLDS_MARKED: bool = any(&[$($param::HOLDS_MARKED),+]);
            const MARKED: bool = false;
            const CHECKED: () = { $(let () = $param::CHECKED;)+ };
        }
    )*};
}

sized_structures! {
    [X] [X];
    [X] Option<X>;
    [X, E] Result<X, E>;
    [A] (A,);
    [A, B] (A, B);
    [A, B, C] (A, B, C);
}

/// What `#[derive(Opaque)]` would emit for an extern-defined type, one rule
/// over all of them: its arguments' facts decide its form.
macro_rules! extern_defined {
    ($([$($param:ident),*] $ty:ty;)*) => {$(
        impl<$($param: Form),*> Form for $ty {
            const HOLDS_UNIFORM: bool = any(&[$($param::HOLDS_UNIFORM),*]);
            const UNIFORM_AT_ROOT: bool = false;
            const HOLDS_MARKED: bool = Self::MARKED || any(&[$($param::HOLDS_MARKED),*]);
            const MARKED: bool = marked(&[$($param::HOLDS_UNIFORM),*]);
            const CHECKED: () = {
                $(let () = $param::CHECKED;)*
                check_extern(
                    &[$($param::UNIFORM_AT_ROOT),*],
                    &[$($param::HOLDS_UNIFORM),*],
                    &[$($param::HOLDS_MARKED),*],
                )
            };
        }
    )*};
}

pub struct Stack<T>(PhantomData<T>);
pub struct Map<K, V>(PhantomData<(K, V)>);
pub struct Counter;

extern_defined! {
    [T] Stack<T>;
    [K, V] Map<K, V>;
    [] Counter;
}
