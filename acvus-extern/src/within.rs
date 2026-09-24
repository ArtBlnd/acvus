//! A handler is handed each carrier at its call's lifetime, and cannot keep
//! it past the call because Rust refuses the keep (RFC-0079 rule 6). The glue
//! materializes a parameter at the type the handler wrote, with the carrier
//! lifetimes that type names left for inference, and requires
//! `Within<'a>` of it, where `'a` is a lifetime of the glue's own call.

/// Every carrier a value of `Self` holds is at `'s`: a carrier implements
/// this at its own lifetime only, a compound type where each of its parts
/// does, and a type that holds no carrier at every `'s`.
///
/// # Safety
/// An impl at a lifetime other than a carrier's own, or at a compound type
/// one of whose parts does not implement it at the same `'s`, hands a handler
/// a carrier it can keep past its call. No part a lifetime or a type
/// parameter reaches sits behind an `UnsafeCell` (`Cell`, `RefCell`,
/// `Mutex`, an atomic, …): a shared borrow of such a part could store a
/// carrier where `'s` no longer names it (RFC-0079 rule 8). An impl that
/// holds one asserts, in its own `unsafe`, that no call's carrier is ever
/// stored there.
pub unsafe trait Within<'s> {}

#[doc(hidden)]
#[macro_export]
macro_rules! within_every {
    ($t:ty $(, const $n:ident: $c:ty)*) => {
        // SAFETY: the type names no lifetime and holds no carrier.
        unsafe impl<'__s $(, const $n: $c)*> $crate::Within<'__s> for $t {}
    };
}

// SAFETY: a `PhantomData` holds no value.
unsafe impl<'s, T> Within<'s> for std::marker::PhantomData<T> where T: ?Sized {}

// SAFETY: the element holds its carriers at `'s`.
unsafe impl<'s, T> Within<'s> for std::collections::VecDeque<T> where T: Within<'s> {}

// SAFETY: the element holds its carriers at `'s`.
unsafe impl<'s, T> Within<'s> for std::vec::IntoIter<T> where T: Within<'s> {}

// SAFETY: the element holds its carriers at `'s`.
unsafe impl<'s, T> Within<'s> for Vec<T> where T: Within<'s> {}

// SAFETY: the payload holds its carriers at `'s`.
unsafe impl<'s, T> Within<'s> for Option<T> where T: Within<'s> {}

// SAFETY: each side holds its carriers at `'s`.
unsafe impl<'s, T, E> Within<'s> for Result<T, E>
where
    T: Within<'s>,
    E: Within<'s>,
{
}

// SAFETY: the element holds its carriers at `'s`.
unsafe impl<'s, T, const N: usize> Within<'s> for [T; N] where T: Within<'s> {}

macro_rules! within_tuple {
    ($($t:ident),+) => {
        // SAFETY: each part holds its carriers at `'s`.
        unsafe impl<'s, $($t),+> Within<'s> for ($($t,)+)
        where
            $($t: Within<'s>,)+
        {
        }
    };
}

within_tuple!(A);
within_tuple!(A, B);
within_tuple!(A, B, C);
within_tuple!(A, B, C, D);
within_tuple!(A, B, C, D, E);
within_tuple!(A, B, C, D, E, F);
within_tuple!(A, B, C, D, E, F, G);
within_tuple!(A, B, C, D, E, F, G, H);
