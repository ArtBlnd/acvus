//! A compile-time query on lifetimes. Each type states, as a type, the
//! lifetimes it carries in the order of its own parameters: its own lifetime
//! arguments first, then each type argument's, in order. The macro reads the
//! same sequence from the signature's syntax and asks rustc whether the two
//! are one type. The question is put through trait matching, `Same<A> for A`,
//! which equates its two sides rather than relating them by subtyping: their
//! shapes must match, and borrowck then requires each pair of lifetimes to be
//! one.
//!
//! The types below stand for the extern crate's and an author's, enough to
//! pose the queries the note's candidate needs.

use std::marker::PhantomData;

/// One lifetime. It is invariant, though the query does not rest on that:
/// see `CovLt`.
pub struct Lt<'a>(PhantomData<fn(&'a ()) -> &'a ()>);

/// The control: a covariant marker. A query put through trait matching stays
/// exact under it too, since trait matching equates rather than subtypes
/// (q08a, q08b).
pub struct CovLt<'a>(PhantomData<&'a ()>);

pub struct Nil;
pub struct Cons<H, T>(PhantomData<(H, T)>);

pub trait Append<B> {
    type Out;
}

impl<B> Append<B> for Nil {
    type Out = B;
}

impl<H, T, B> Append<B> for Cons<H, T>
where
    T: Append<B>,
{
    type Out = Cons<H, T::Out>;
}

pub trait Lifetimed {
    type Seq;
}

pub trait Same<B> {}

impl<A> Same<A> for A {}

pub fn same<A, B>()
where
    A: Same<B>,
{
}

/// What the macro puts for a type variable when it poses a query.
pub struct Var;

impl Lifetimed for Var {
    type Seq = Nil;
}

pub enum Never {}

impl Lifetimed for Never {
    type Seq = Nil;
}

impl Lifetimed for String {
    type Seq = Nil;
}

impl Lifetimed for i64 {
    type Seq = Nil;
}

impl Lifetimed for str {
    type Seq = Nil;
}

impl<T> Lifetimed for [T]
where
    T: Lifetimed,
{
    type Seq = T::Seq;
}

impl<'a, T> Lifetimed for &'a T
where
    T: Lifetimed + ?Sized,
{
    type Seq = Cons<Lt<'a>, T::Seq>;
}

impl<'a, T> Lifetimed for &'a mut T
where
    T: Lifetimed + ?Sized,
{
    type Seq = Cons<Lt<'a>, T::Seq>;
}

impl<T> Lifetimed for Option<T>
where
    T: Lifetimed,
{
    type Seq = T::Seq;
}

impl<A, B> Lifetimed for (A, B)
where
    A: Lifetimed,
    B: Lifetimed,
    A::Seq: Append<B::Seq>,
{
    type Seq = <A::Seq as Append<B::Seq>>::Out;
}

pub struct Uniform<X>(PhantomData<fn(X) -> X>);

impl<X> Lifetimed for Uniform<X>
where
    X: Lifetimed,
{
    type Seq = X::Seq;
}

pub struct Shared<'a, X>(PhantomData<(fn(X) -> X, fn() -> &'a ())>);

impl<'a, X> Lifetimed for Shared<'a, X>
where
    X: Lifetimed,
{
    type Seq = Cons<Lt<'a>, X::Seq>;
}

pub struct Ref<'a, X>(PhantomData<(fn(X) -> X, fn() -> &'a ())>);

impl<'a, X> Lifetimed for Ref<'a, X>
where
    X: Lifetimed,
{
    type Seq = Cons<Lt<'a>, X::Seq>;
}

pub struct Mut<'a, X>(PhantomData<(fn(X) -> X, fn() -> &'a ())>);

impl<'a, X> Lifetimed for Mut<'a, X>
where
    X: Lifetimed,
{
    type Seq = Cons<Lt<'a>, X::Seq>;
}

/// A `#[fn_type]` marker. Its effect variable carries no lifetime.
pub struct Pred<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);

impl<'f, T, E> Lifetimed for Pred<'f, T, E>
where
    T: Lifetimed,
{
    type Seq = Cons<Lt<'f>, T::Seq>;
}

/// An opaque type with one lifetime and one type argument, as a derive would
/// state it.
pub struct Cursor<'a, T>(PhantomData<(&'a (), T)>);

impl<'a, T> Lifetimed for Cursor<'a, T>
where
    T: Lifetimed,
{
    type Seq = Cons<Lt<'a>, T::Seq>;
}

/// Two lifetimes and two type arguments.
pub struct Zip<'a, 'b, T, U>(PhantomData<(&'a (), &'b (), T, U)>);

impl<'a, 'b, T, U> Lifetimed for Zip<'a, 'b, T, U>
where
    T: Lifetimed,
    U: Lifetimed,
    T::Seq: Append<U::Seq>,
{
    type Seq = Cons<Lt<'a>, Cons<Lt<'b>, <T::Seq as Append<U::Seq>>::Out>>;
}

/// A structural type whose fields are laid out in their names' order, `a`
/// before `z`: its sequence is still in its parameters' order.
pub struct S<'a, 'b> {
    pub z: Ref<'a, Var>,
    pub a: Ref<'b, Var>,
}

impl<'a, 'b> Lifetimed for S<'a, 'b> {
    type Seq = Cons<Lt<'a>, Cons<Lt<'b>, Nil>>;
}

pub struct Pair<'a, 'b>(PhantomData<(&'a (), &'b ())>);

impl<'a, 'b> Lifetimed for Pair<'a, 'b> {
    type Seq = Cons<Lt<'a>, Cons<Lt<'b>, Nil>>;
}
