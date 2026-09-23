//! A declaration's marker names every type at `'static`, because `TyArg`,
//! `Var` and a box key are `'static`; the glue hands a handler the same type
//! at the call's lifetime and reads its result back at the marker
//! (RFC-0079 rule 6). `#[extern_fn]`'s `at_static` writes the marker.

use crate::canonical::same_layout;

/// `Self` with every lifetime a value it holds names at `'a`. A type a
/// carrier only names, `Ref`'s referent or `Closure`'s parameters, stays at
/// its marker: the carrier reads it at its own brand.
///
/// # Safety
/// `At<'a>` is `Self` with lifetimes changed and nothing else, so the two
/// are one layout: the functions below read a value of either as the other.
/// The glue hands a handler `At<'a>` and relies on it keeping nothing a call
/// lent past `'a` (RFC-0079 rule 6): each lifetime parameter and each part a
/// type parameter fills is at `'a` there, and a part named at one fixed type
/// holds only what Rust code built at that type.
pub unsafe trait Branded {
    type At<'a>: 'a;
}

/// A type that names no lifetime, and so is its own `At<'a>` at every
/// `'a`: what a read in place at the declared type (`Stored`,
/// `TransparentOver`) and a closure result returned as itself
/// (`PassedByValue`) ask for.
pub trait Unbranded: for<'a> Branded<At<'a> = Self> {}

impl<T> Unbranded for T where T: for<'a> Branded<At<'a> = T> {}

#[doc(hidden)]
#[macro_export]
macro_rules! unbranded {
    ($t:ty $(, const $n:ident: $c:ty)*) => {
        // SAFETY: the type names no lifetime, so it is its own `At<'a>`.
        unsafe impl<$(const $n: $c),*> $crate::Branded for $t {
            type At<'__a> = Self;
        }
    };
}

/// # Safety
/// Every storage and entry `value` names is live for `'a` (RFC-0018).
#[doc(hidden)]
#[inline(always)]
pub unsafe fn brand<'a, T>(value: T) -> T::At<'a>
where
    T: Branded,
{
    same_layout!(T, T::At<'a>);
    let held = std::mem::ManuallyDrop::new(value);
    // SAFETY: `Branded`'s contract, with the layout checked above; `held` is
    // never read again, so the value moves out exactly once.
    unsafe { std::ptr::read((&raw const *held).cast::<T::At<'a>>()) }
}

/// # Safety
/// The result is erased into the runtime's value before safe code sees it.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn unbrand<'a, T>(value: T::At<'a>) -> T
where
    T: Branded,
{
    same_layout!(T, T::At<'a>);
    let held = std::mem::ManuallyDrop::new(value);
    // SAFETY: as `brand`'s, read the other way.
    unsafe { std::ptr::read((&raw const *held).cast::<T>()) }
}

/// # Safety
/// As `brand`'s, for what `value` holds.
#[doc(hidden)]
#[inline(always)]
pub unsafe fn brand_ref<'a, T>(value: &'a T) -> &'a T::At<'a>
where
    T: Branded,
{
    same_layout!(T, T::At<'a>);
    // SAFETY: `Branded`'s contract, with the layout checked above.
    unsafe { &*(value as *const T).cast::<T::At<'a>>() }
}

/// # Safety
/// As `brand`'s, for what `value` holds. What the handler writes into it at
/// `'a` stays there at the marker; the loans it holds are the checker's to
/// keep, through the region parameters the type declares (RFC-0079 rule 2).
#[doc(hidden)]
#[inline(always)]
pub unsafe fn brand_mut<'a, T>(value: &'a mut T) -> &'a mut T::At<'a>
where
    T: Branded,
{
    same_layout!(T, T::At<'a>);
    // SAFETY: as `brand_ref`'s, exclusively.
    unsafe { &mut *(value as *mut T).cast::<T::At<'a>>() }
}

// SAFETY: a `PhantomData` holds no value.
unsafe impl<T> Branded for std::marker::PhantomData<T>
where
    T: ?Sized + 'static,
{
    type At<'a> = Self;
}

// SAFETY: the element is its own `At<'a>`.
unsafe impl<T> Branded for std::collections::VecDeque<T>
where
    T: Branded,
{
    type At<'a> = std::collections::VecDeque<T::At<'a>>;
}

// SAFETY: the element is its own `At<'a>`.
unsafe impl<T> Branded for std::vec::IntoIter<T>
where
    T: Branded,
{
    type At<'a> = std::vec::IntoIter<T::At<'a>>;
}

macro_rules! branded_tuple {
    ($($t:ident),+) => {
        // SAFETY: each part is its own `At<'a>`.
        unsafe impl<$($t),+> Branded for ($($t,)+)
        where
            $($t: Branded,)+
        {
            type At<'a> = ($($t::At<'a>,)+);
        }
    };
}

branded_tuple!(A);
branded_tuple!(A, B);
branded_tuple!(A, B, C);
branded_tuple!(A, B, C, D);

// SAFETY: the element is its own `At<'a>`.
unsafe impl<T, const N: usize> Branded for [T; N]
where
    T: Branded,
{
    type At<'a> = [T::At<'a>; N];
}
