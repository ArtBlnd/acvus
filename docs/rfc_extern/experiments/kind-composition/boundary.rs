//! The extern side of the model: `Uniform`, `Ref` and `Mut`, the sealed
//! `TypeVar`, each type's crossing by value through its own kind (values §3),
//! and its opening one layer at a time (`Open`). This file has no branch: a
//! nested type crosses as its layers' kinds composed, the innermost first,
//! and telling an option's or a result's variants apart is the runtime's.
//!
//! What is not yet opened is held in one of two forms: owned, a `Uniform`,
//! which releases its value on drop, or borrowed, a `Ref` or a `Mut`, which
//! has no drop, so that a `Ref` copies (references §1).

use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use crate::contract::{Options, Results, Runtime, Scalars, Strings, Tuples, Views};

/// A value the runtime holds, its type kept by the marker `X` (values §2).
#[repr(transparent)]
pub struct Uniform<R, X>
where
    R: Runtime,
{
    value: R::Value,
    marker: PhantomData<fn(X) -> X>,
}

impl<R, X> Uniform<R, X>
where
    R: Runtime,
{
    /// values §2: `X: 'static`, so it holds a value made as `X`.
    pub fn new(x: X) -> Self
    where
        X: Cross<R> + 'static,
    {
        Uniform { value: x.give(), marker: PhantomData }
    }

    /// Gives up the held `Value` without releasing it.
    pub fn into_value(self) -> R::Value {
        ManuallyDrop::new(self).value
    }

    /// # Safety
    /// `value` is live, owned by the caller, and was made as `X` or as an
    /// instantiation that differs from `X` only in a zero-sized marker
    /// (variables §3).
    pub unsafe fn from_value(value: R::Value) -> Self {
        Uniform { value, marker: PhantomData }
    }

    pub fn as_ref(&self) -> Ref<'_, R, X> {
        // SAFETY: a shared reference to the value this owner holds, for its
        // borrow.
        unsafe { Ref::from_value(R::value_as_ref(&self.value)) }
    }

    pub fn as_mut(&mut self) -> Mut<'_, R, X> {
        // SAFETY: an exclusive reference to the value this owner holds, for
        // its exclusive borrow.
        unsafe { Mut::from_value(R::value_as_mut(&mut self.value)) }
    }

    /// Opens one layer, shared.
    pub fn project<'s, O>(&'s self, lent: impl FnOnce(&<X as Open<'s, R>>::Layer) -> O) -> O
    where
        X: Open<'s, R>,
    {
        self.as_ref().project(lent)
    }

    /// Opens one layer, exclusive.
    pub fn project_mut<'s, O>(&'s mut self, lent: impl FnOnce(&mut <X as Open<'s, R>>::LayerMut) -> O) -> O
    where
        X: Open<'s, R>,
    {
        // SAFETY: as `as_mut`; the reference lives for this borrow alone.
        unsafe { X::project_mut(R::value_as_mut(&mut self.value), lent) }
    }
}

impl<R, X> Drop for Uniform<R, X>
where
    R: Runtime,
{
    fn drop(&mut self) {
        // SAFETY: a `Uniform` owns its value and releases it once, here.
        unsafe { R::release(self.value) }
    }
}

/// A shared reference to one value, held as the reference value the runtime
/// made (references §1). Invariant in `X`, covariant in `'a`.
pub struct Ref<'a, R, X>
where
    R: Runtime,
{
    value: R::Value,
    marker: PhantomData<(fn(X) -> X, &'a ())>,
}

impl<R, X> Clone for Ref<'_, R, X>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<R, X> Copy for Ref<'_, R, X> where R: Runtime {}

impl<'a, R, X> Ref<'a, R, X>
where
    R: Runtime,
{
    /// # Safety
    /// `value` is a live shared reference value to a value made as `X`, or
    /// as such an instantiation, for `'a`.
    pub unsafe fn from_value(value: R::Value) -> Self {
        Ref { value, marker: PhantomData }
    }

    pub fn as_ref(self) -> Ref<'a, R, X> {
        self
    }

    /// Opens one layer of the value it names.
    pub fn project<O>(self, lent: impl FnOnce(&<X as Open<'a, R>>::Layer) -> O) -> O
    where
        X: Open<'a, R>,
    {
        // SAFETY: a `Ref` is a live shared reference to an `X` for `'a`.
        unsafe { X::project(self.value, lent) }
    }
}

/// An exclusive reference to one value (references §1). Invariant in `X`,
/// covariant in `'a`.
pub struct Mut<'a, R, X>
where
    R: Runtime,
{
    value: R::Value,
    marker: PhantomData<(fn(X) -> X, &'a mut ())>,
}

impl<'a, R, X> Mut<'a, R, X>
where
    R: Runtime,
{
    /// # Safety
    /// `value` is a live exclusive reference value to a value made as `X`,
    /// or as such an instantiation, for `'a`, reached through nothing else.
    pub unsafe fn from_value(value: R::Value) -> Self {
        Mut { value, marker: PhantomData }
    }

    /// A shared reborrow: a reference to a reference is that reference.
    pub fn as_ref(&self) -> Ref<'_, R, X> {
        // SAFETY: the place holds this live exclusive reference, which the
        // shared borrow of `self` keeps from use while the reborrow lives.
        unsafe { Ref::from_value(R::value_as_ref(&self.value)) }
    }

    /// An exclusive reborrow.
    pub fn as_mut(&mut self) -> Mut<'_, R, X> {
        // SAFETY: as `as_ref`, exclusive.
        unsafe { Mut::from_value(R::value_as_mut(&mut self.value)) }
    }

    /// Opens one layer of the value it names, exclusive.
    pub fn project_mut<'s, O>(&'s mut self, lent: impl FnOnce(&mut <X as Open<'s, R>>::LayerMut) -> O) -> O
    where
        X: Open<'s, R>,
    {
        // SAFETY: as `as_mut`; the reborrow lives for this borrow alone.
        unsafe { X::project_mut(R::value_as_mut(&mut self.value), lent) }
    }
}

/// What a scope lends of the arguments it holds, for a loan `'s` the scope
/// makes (functions §3): a shared or an exclusive reference to each. The
/// loan is a parameter of the trait, not of an associated type, so a scope
/// quantified over every loan asks nothing of the arguments' own lifetimes.
pub trait Lend<'s, R>
where
    R: Runtime,
{
    type Lent;

    fn lend(&'s mut self) -> Self::Lent;
}

/// An argument a scope lends shared.
pub struct ByRef<T>(pub T);

/// An argument a scope lends exclusively.
pub struct ByMut<T>(pub T);

impl<'s, R, T> Lend<'s, R> for ByRef<T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    type Lent = Ref<'s, R, T>;

    fn lend(&'s mut self) -> Ref<'s, R, T> {
        TypeVar::as_ref(&self.0)
    }
}

impl<'s, R, T> Lend<'s, R> for ByMut<T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    type Lent = Mut<'s, R, T>;

    fn lend(&'s mut self) -> Mut<'s, R, T> {
        TypeVar::as_mut(&mut self.0)
    }
}

impl<'s, R, A, B> Lend<'s, R> for (A, B)
where
    R: Runtime,
    A: Lend<'s, R>,
    B: Lend<'s, R>,
{
    type Lent = (A::Lent, B::Lent);

    fn lend(&'s mut self) -> Self::Lent {
        (self.0.lend(), self.1.lend())
    }
}

/// The callable a scope lends to its call: a shared reference to the
/// function value, which fits that scope's loans alone.
pub struct Calling<'s, R, F>
where
    R: Runtime,
{
    function: R::Value,
    marker: PhantomData<(fn(F) -> F, fn(&'s ()) -> &'s ())>,
}

impl<R, F> Calling<'_, R, F>
where
    R: Runtime,
{
    /// The reference value to the function, which the call gives the
    /// runtime.
    pub fn into_value(self) -> R::Value {
        self.function
    }
}

/// Lends `function` and `arguments` to the call `start` makes, and gives
/// both back with its result (functions §3). `start` is quantified over
/// every loan the scope may make, so neither a loan of its nor the call
/// leaves it, and a loan from outside fits none of its calls.
pub async fn scope<R, F, A, O>(
    function: Uniform<R, F>,
    mut arguments: A,
    start: impl for<'s> AsyncFnOnce(Calling<'s, R, F>, <A as Lend<'s, R>>::Lent) -> O,
) -> (Uniform<R, F>, A, O)
where
    R: Runtime,
    A: for<'s> Lend<'s, R>,
{
    let reference = <Ref<'_, R, F> as Cross<R>>::give(function.as_ref());
    let out = start(Calling { function: reference, marker: PhantomData }, arguments.lend()).await;
    (function, arguments, out)
}

/// The callable an exclusive scope lends to its call: an exclusive reference
/// to the function value, which fits that scope's loans alone.
pub struct CallingMut<'s, R, F>
where
    R: Runtime,
{
    function: R::Value,
    marker: PhantomData<(fn(F) -> F, fn(&'s ()) -> &'s ())>,
}

impl<R, F> CallingMut<'_, R, F>
where
    R: Runtime,
{
    /// The exclusive reference value to the function, which the call gives
    /// the runtime.
    pub fn into_value(self) -> R::Value {
        self.function
    }
}

/// The callable an owning scope gives its call: the function value itself,
/// which the call consumes, and which is released where no call takes it.
pub struct CallingOnce<'s, R, F>
where
    R: Runtime,
{
    function: Uniform<R, F>,
    marker: PhantomData<fn(&'s ()) -> &'s ()>,
}

impl<R, F> CallingOnce<'_, R, F>
where
    R: Runtime,
{
    /// The function value, given up to the call.
    pub fn into_value(self) -> R::Value {
        self.function.into_value()
    }
}

/// As `scope`, the function lent exclusively: the call may change the
/// function value, which the scope gives back with the arguments and the
/// result.
pub async fn scope_mut<R, F, A, O>(
    mut function: Uniform<R, F>,
    mut arguments: A,
    start: impl for<'s> AsyncFnOnce(CallingMut<'s, R, F>, <A as Lend<'s, R>>::Lent) -> O,
) -> (Uniform<R, F>, A, O)
where
    R: Runtime,
    A: for<'s> Lend<'s, R>,
{
    let reference = <Mut<'_, R, F> as Cross<R>>::give(function.as_mut());
    let out = start(CallingMut { function: reference, marker: PhantomData }, arguments.lend()).await;
    (function, arguments, out)
}

/// As `scope`, the function given by value: the call consumes it, so the
/// scope gives back the arguments and the result alone.
pub async fn scope_once<R, F, A, O>(
    function: Uniform<R, F>,
    mut arguments: A,
    start: impl for<'s> AsyncFnOnce(CallingOnce<'s, R, F>, <A as Lend<'s, R>>::Lent) -> O,
) -> (A, O)
where
    R: Runtime,
    A: for<'s> Lend<'s, R>,
{
    let out = start(CallingOnce { function, marker: PhantomData }, arguments.lend()).await;
    (arguments, out)
}

/// The one fixed uninhabited marker a type variable is monomorphized at
/// (variables §3).
pub enum Generic {}

/// The one bound of an effect variable (variables §1).
pub trait EffectVar: seal::EffectVar {}

/// The marker an effect variable is instantiated at.
pub enum Effect0 {}

impl seal::EffectVar for Effect0 {}

impl EffectVar for Effect0 {}

pub(crate) mod seal {
    /// `TypeVar`'s seal, held by every `Uniform` and by nothing else.
    pub trait TypeVar {}

    /// `EffectVar`'s seal.
    pub trait EffectVar {}
}

/// The one bound of a type variable. It holds for every `Uniform` and for
/// nothing else (variables §3).
pub trait TypeVar<R>: seal::TypeVar
where
    R: Runtime,
{
    /// The one way generic code makes a shared reference to a variable's
    /// value that it can give out (variables §1).
    fn as_ref(&self) -> Ref<'_, R, Self>
    where
        Self: Sized;

    /// As `as_ref`, exclusive.
    fn as_mut(&mut self) -> Mut<'_, R, Self>
    where
        Self: Sized;

    fn give(self) -> R::Value
    where
        Self: Sized;

    /// # Safety
    /// As `Uniform::from_value`.
    unsafe fn take(value: R::Value) -> Self
    where
        Self: Sized;
}

impl<R, X> seal::TypeVar for Uniform<R, X> where R: Runtime {}

impl<R, X> TypeVar<R> for Uniform<R, X>
where
    R: Runtime,
{
    fn as_ref(&self) -> Ref<'_, R, Self> {
        // SAFETY: a variable's value is its `Uniform`'s, so a reference to the
        // one is a reference to the other.
        unsafe { Ref::from_value(Uniform::as_ref(self).value) }
    }

    fn as_mut(&mut self) -> Mut<'_, R, Self> {
        // SAFETY: as `as_ref`, exclusive.
        unsafe { Mut::from_value(Uniform::as_mut(self).value) }
    }

    fn give(self) -> R::Value {
        self.into_value()
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { Uniform::from_value(value) }
    }
}

/// A type's crossing by value through its own kind (values §3).
///
/// # Safety
/// `take` takes back whole what `give` gave, as the same type.
pub unsafe trait Cross<R>: Sized
where
    R: Runtime,
{
    fn give(self) -> R::Value;

    /// # Safety
    /// `value` was given as `Self`, or as an instantiation that differs from it
    /// only in a zero-sized marker (variables §3), is live, and is owned by the caller.
    unsafe fn take(value: R::Value) -> Self;
}

/// The type a value of this type is held at in a uniform position: a
/// `Uniform` of it, or, for a variable, the variable itself, which is never
/// wrapped twice (variables §2).
///
/// # Safety
/// `As` is `Uniform<R, Self>`, or `Self` where `Self` is a variable.
pub unsafe trait Uniformly<R>: Cross<R>
where
    R: Runtime,
{
    type As: TypeVar<R>;
}

/// A type whose value opens one layer at a time, by its own kind: a shared
/// reference to one opens to `Layer`, an exclusive one to `LayerMut`, each
/// with its parts not yet opened. The lifetime is the reference's.
///
/// # Safety
/// `project` and `project_mut` open a value of this type by its own kind.
pub unsafe trait Open<'a, R>: Cross<R>
where
    R: Runtime,
{
    type Layer;
    type LayerMut;

    /// # Safety
    /// `reference` is a live shared reference value to a `Self` for `'a`.
    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Self::Layer) -> O) -> O;

    /// # Safety
    /// `reference` is a live exclusive reference value to a `Self` for
    /// `'a`, used nowhere else while `lent` runs.
    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Self::LayerMut) -> O) -> O;
}

// SAFETY: a variable crosses as the `Value` its `Uniform` holds.
unsafe impl<R, T> Cross<R> for T
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn give(self) -> R::Value {
        <T as TypeVar<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <T as TypeVar<R>>::take(value) }
    }
}

// SAFETY: a variable is held as itself.
unsafe impl<R, T> Uniformly<R> for T
where
    R: Runtime,
    T: TypeVar<R>,
{
    type As = T;
}

// SAFETY: a shared reference crosses as its reference value (references §2).
unsafe impl<R, X> Cross<R> for Ref<'_, R, X>
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        self.value
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { Ref::from_value(value) }
    }
}

// SAFETY: as `Ref`'s.
unsafe impl<R, X> Cross<R> for Mut<'_, R, X>
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        self.value
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { Mut::from_value(value) }
    }
}

macro_rules! scalars {
    ($($scalar:ty),* $(,)?) => {$(
        // SAFETY: the scalar kind takes back what it gave.
        unsafe impl<R> Cross<R> for $scalar
        where
            R: Runtime,
        {
            fn give(self) -> R::Value {
                <R::Scalars as Scalars<R>>::give(self)
            }

            unsafe fn take(value: R::Value) -> Self {
                // SAFETY: the caller's premise.
                unsafe { <R::Scalars as Scalars<R>>::take(value) }
            }
        }

        // SAFETY: held as a `Uniform` of itself.
        unsafe impl<R> Uniformly<R> for $scalar
        where
            R: Runtime,
        {
            type As = Uniform<R, $scalar>;
        }

        // SAFETY: the scalar kind lends the scalar itself.
        unsafe impl<'a, R> Open<'a, R> for $scalar
        where
            R: Runtime,
        {
            type Layer = $scalar;
            type LayerMut = $scalar;

            unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&$scalar) -> O) -> O {
                // SAFETY: the caller's premise.
                unsafe { <R::Scalars as Scalars<R>>::project(reference, lent) }
            }

            unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut $scalar) -> O) -> O {
                // SAFETY: the caller's premise.
                unsafe { <R::Scalars as Scalars<R>>::project_mut(reference, lent) }
            }
        }
    )*};
}

scalars!(i64, bool, ());

// SAFETY: the string kind takes back what it gave.
unsafe impl<R> Cross<R> for String
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        <R::Strings as Strings<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Strings as Strings<R>>::take(value) }
    }
}

// SAFETY: held as a `Uniform` of itself.
unsafe impl<R> Uniformly<R> for String
where
    R: Runtime,
{
    type As = Uniform<R, String>;
}

// SAFETY: the string kind lends the string itself.
unsafe impl<'a, R> Open<'a, R> for String
where
    R: Runtime,
{
    type Layer = String;
    type LayerMut = String;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&String) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Strings as Strings<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut String) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Strings as Strings<R>>::project_mut(reference, lent) }
    }
}

// SAFETY: the view kind takes back the text it was given; the taker names
// the lifetime, an entry its own (declarations §2).
unsafe impl<'a, R> Cross<R> for &'a str
where
    R: Runtime,
{
    fn give(self) -> R::Value {
        <R::Views as Views<R>>::give_text(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Views as Views<R>>::take_text(value) }
    }
}

// SAFETY: as `&str`'s, for a run of values.
unsafe impl<'a, R, T> Cross<R> for &'a [T]
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn give(self) -> R::Value {
        <R::Views as Views<R>>::give_run(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Views as Views<R>>::take_run(value) }
    }
}

// SAFETY: as `&[T]`'s, exclusive.
unsafe impl<'a, R, T> Cross<R> for &'a mut [T]
where
    R: Runtime,
    T: TypeVar<R>,
{
    fn give(self) -> R::Value {
        <R::Views as Views<R>>::give_run_mut(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Views as Views<R>>::take_run_mut(value) }
    }
}

macro_rules! tuples {
    ($(($arity:literal; $($part:ident $value:ident),+))+) => {$(
        // SAFETY: the tuple kind takes back the parts it was given, each
        // taken back by its own kind.
        unsafe impl<R, $($part),+> Cross<R> for ($($part,)+)
        where
            R: Runtime,
            $($part: Cross<R>,)+
        {
            fn give(self) -> R::Value {
                let ($($value,)+) = self;
                // SAFETY: each part's value was just made by its own kind and
                // is given up to the tuple.
                unsafe { <R::Tuples as Tuples<R>>::give([$(<$part as Cross<R>>::give($value)),+]) }
            }

            unsafe fn take(value: R::Value) -> Self {
                // SAFETY: the caller's premise; each part was given as its
                // own type and is the caller's along with the tuple.
                let [$($value),+] = unsafe { <R::Tuples as Tuples<R>>::take::<$arity>(value) };
                ($(unsafe { <$part as Cross<R>>::take($value) },)+)
            }
        }

        // SAFETY: held as a `Uniform` of itself.
        unsafe impl<R, $($part),+> Uniformly<R> for ($($part,)+)
        where
            R: Runtime,
            $($part: Cross<R>,)+
        {
            type As = Uniform<R, ($($part,)+)>;
        }

        // SAFETY: a reference to a tuple is projected to its parts' references
        // (runtime §2), each projected later by its own kind.
        unsafe impl<'a, R, $($part),+> Open<'a, R> for ($($part,)+)
        where
            R: Runtime,
            $($part: Cross<R>,)+
        {
            type Layer = ($(Ref<'a, R, $part>,)+);
            type LayerMut = ($(Mut<'a, R, $part>,)+);

            unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Self::Layer) -> O) -> O {
                // SAFETY: the caller's premise; each part's reference lives as
                // long as the tuple's.
                let [$($value),+] = unsafe { <R::Tuples as Tuples<R>>::parts::<$arity>(reference) };
                lent(&($(unsafe { Ref::from_value($value) },)+))
            }

            unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Self::LayerMut) -> O) -> O {
                // SAFETY: as `project`, each part reached through its own
                // reference alone.
                let [$($value),+] = unsafe { <R::Tuples as Tuples<R>>::parts_mut::<$arity>(reference) };
                lent(&mut ($(unsafe { Mut::from_value($value) },)+))
            }
        }
    )+};
}

tuples! {
    (2; A a, B b)
    (3; A a, B b, C c)
}

// SAFETY: the option kind takes back what it gave, its payload by the
// payload's own kind.
unsafe impl<R, P> Cross<R> for Option<P>
where
    R: Runtime,
    P: Cross<R>,
{
    fn give(self) -> R::Value {
        <R::Options as Options<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Options as Options<R>>::take(value) }
    }
}

// SAFETY: held as a `Uniform` of itself.
unsafe impl<R, P> Uniformly<R> for Option<P>
where
    R: Runtime,
    P: Cross<R>,
{
    type As = Uniform<R, Option<P>>;
}

// SAFETY: the option kind lends the option in Rust's shape, its payload not
// yet opened (runtime §2).
unsafe impl<'a, R, P> Open<'a, R> for Option<P>
where
    R: Runtime,
    P: Uniformly<R>,
{
    type Layer = Option<P::As>;
    type LayerMut = Option<P::As>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Option<P::As>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Options as Options<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Option<P::As>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Options as Options<R>>::project_mut(reference, lent) }
    }
}

// SAFETY: as `Option`'s.
unsafe impl<R, T, E> Cross<R> for Result<T, E>
where
    R: Runtime,
    T: Cross<R>,
    E: Cross<R>,
{
    fn give(self) -> R::Value {
        <R::Results as Results<R>>::give(self)
    }

    unsafe fn take(value: R::Value) -> Self {
        // SAFETY: the caller's premise.
        unsafe { <R::Results as Results<R>>::take(value) }
    }
}

// SAFETY: held as a `Uniform` of itself.
unsafe impl<R, T, E> Uniformly<R> for Result<T, E>
where
    R: Runtime,
    T: Cross<R>,
    E: Cross<R>,
{
    type As = Uniform<R, Result<T, E>>;
}

// SAFETY: as `Option`'s.
unsafe impl<'a, R, T, E> Open<'a, R> for Result<T, E>
where
    R: Runtime,
    T: Uniformly<R>,
    E: Uniformly<R>,
{
    type Layer = Result<T::As, E::As>;
    type LayerMut = Result<T::As, E::As>;

    unsafe fn project<O>(reference: R::Value, lent: impl FnOnce(&Result<T::As, E::As>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Results as Results<R>>::project(reference, lent) }
    }

    unsafe fn project_mut<O>(reference: R::Value, lent: impl FnOnce(&mut Result<T::As, E::As>) -> O) -> O {
        // SAFETY: the caller's premise.
        unsafe { <R::Results as Results<R>>::project_mut(reference, lent) }
    }
}
