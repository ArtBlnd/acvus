//! Whether a `#` member is its uniform member at the marked type, checked by
//! Rust. The uniform member's macro states its signature as a family over its
//! lifetimes and type variables: its parameters as one tuple, and its
//! result. The `#` member's macro writes its own signature the same way, its
//! lifetimes named in the order they first appear, puts the choice it writes
//! into the family, and asks rustc whether the family's native form is its
//! own signature. The question is put through trait matching, which equates
//! its two sides, so lifetimes must match exactly too.
//!
//! The types below stand for the extern crate's and an author's. A runtime
//! parameter is left out, since no form depends on it.

use std::marker::PhantomData;

/// A value held uniformly.
pub struct Uniform<X: ?Sized>(PhantomData<fn(&X) -> &X>);

/// A shared reference to one value, in the runtime's form.
pub struct Ref<'a, X: ?Sized>(PhantomData<(fn(&X) -> &X, fn() -> &'a ())>);

/// An author's extern-defined type.
pub struct Stack<T>(PhantomData<T>);

/// A type's native form: what a `#` member writes where its uniform member
/// writes the type. A `Uniform` is the value it holds, written natively.
/// A `Ref` has none, since a `#` value holds no `Value` to lend.
pub trait Native {
    type Out: ?Sized;
}

impl<X: Native + ?Sized> Native for Uniform<X> {
    type Out = X::Out;
}

macro_rules! leaves {
    ($($leaf:ty),* $(,)?) => {$(
        impl Native for $leaf {
            type Out = $leaf;
        }
    )*};
}

leaves!(i8, i16, i32, i64, u8, u16, u32, u64, f64, bool, char, (), String, str);

impl<'a, X: Native + ?Sized> Native for &'a X {
    type Out = &'a X::Out;
}

impl<'a, X: Native + ?Sized> Native for &'a mut X {
    type Out = &'a mut X::Out;
}

impl<X: Native> Native for Option<X>
where
    X::Out: Sized,
{
    type Out = Option<X::Out>;
}

impl<A: Native> Native for (A,)
where
    A::Out: Sized,
{
    type Out = (A::Out,);
}

impl<A: Native, B: Native> Native for (A, B)
where
    A::Out: Sized,
    B::Out: Sized,
{
    type Out = (A::Out, B::Out);
}

impl<A: Native, B: Native, C: Native> Native for (A, B, C)
where
    A::Out: Sized,
    B::Out: Sized,
    C::Out: Sized,
{
    type Out = (A::Out, B::Out, C::Out);
}

/// What `#[derive(Opaque)]` would emit for `Stack`, one rule over every
/// extern-defined type: its arguments' native forms.
impl<T: Native> Native for Stack<T>
where
    T::Out: Sized,
{
    type Out = Stack<T::Out>;
}

pub trait Same<T: ?Sized> {}

impl<T: ?Sized> Same<T> for T {}

pub fn same<A, B>()
where
    A: Same<B> + ?Sized,
    B: ?Sized,
{
}
