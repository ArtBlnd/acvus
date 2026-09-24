//! `UniformPayload`: a type whose layout holds its type parameters only as
//! fields (RFC-0076).

use std::collections::VecDeque;
use std::marker::PhantomData;

/// A type whose layout reaches its type parameters only by holding them. A
/// box is keyed by its payload with each uniform parameter at its canonical
/// form and read at the parameter's own, and a payload that reached a
/// parameter through a trait would have whatever layout that parameter's
/// impl chose. `#[derive(Payload)]` implements it for a struct or enum
/// by bounding each field that names a type parameter, and
/// `#[derive(ExternType)]` proves it of an extension type's payload.
///
/// `M` is the marker a proof is made at. The derive proves a payload at a
/// marker that only its own check names, with each type variable assumed
/// `UniformPayload` there, so no bound a user writes and no associated type
/// a user declares supplies that marker: the proof goes through the impls
/// that hold at every `M`, down to the variables it holds.
///
/// # Safety
/// `Self`'s layout reaches a type parameter through a trait only through a
/// bound this impl's `where` clause requires, of that parameter or of a type
/// that names it. Every other way it reaches a parameter is by holding it,
/// directly or in a type that is itself `UniformPayload<M>` at that argument.
///
/// No part a type parameter reaches sits behind an `UnsafeCell` (`Cell`,
/// `RefCell`, `Mutex`, an atomic, …): through a shared borrow of the
/// payload a handler could store there a value lent to one call and read it
/// in another (RFC-0079 rule 8). An impl that holds one asserts, in its own
/// `unsafe`, that no call's loan is ever stored there.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not known to hold its type parameters only as fields",
    label = "`{Self}` is not `UniformPayload`",
    note = "an extension type's payload that names a type parameter is proved `UniformPayload`, field type by field type, so that its layout reaches each parameter only by holding it (RFC-0076)",
    note = "derive `Payload` for `{Self}` if it is a struct or enum of this crate; for another crate's type, `#[extern_type(unsafe(uniform_payload))]` on the extension type asserts it of the payload by hand"
)]
pub unsafe trait UniformPayload<M> {}

macro_rules! closed {
    ($($T:ty),+ $(,)?) => {
        $(
            // SAFETY: a type with no type parameter reaches none.
            unsafe impl<M> UniformPayload<M> for $T {}
        )+
    };
}

closed!(
    bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64, (), String,
);

// SAFETY: `PhantomData<T>` has size 0 and alignment 1 at every `T`.
unsafe impl<M, T> UniformPayload<M> for PhantomData<T> where T: ?Sized {}

// SAFETY: a `Box` of a sized `T` is a thin pointer to one `T`.
unsafe impl<M, T> UniformPayload<M> for Box<T> where T: UniformPayload<M> {}

// SAFETY: a `Vec` is a pointer to its elements, a capacity and a length at
// every `T`.
unsafe impl<M, T> UniformPayload<M> for Vec<T> where T: UniformPayload<M> {}

// SAFETY: as `Vec`'s: a ring buffer over one pointer to its elements.
unsafe impl<M, T> UniformPayload<M> for VecDeque<T> where T: UniformPayload<M> {}

// SAFETY: as `Vec`'s: the buffer a `Vec` gave up and two cursors into it.
unsafe impl<M, T> UniformPayload<M> for std::vec::IntoIter<T> where T: UniformPayload<M> {}

// SAFETY: an `Option` holds its `T` as its one payload, and its niche is
// `T`'s.
unsafe impl<M, T> UniformPayload<M> for Option<T> where T: UniformPayload<M> {}

// SAFETY: a `Result` holds each arm as its payload.
unsafe impl<M, T, E> UniformPayload<M> for Result<T, E>
where
    T: UniformPayload<M>,
    E: UniformPayload<M>,
{
}

// SAFETY: an array holds `N` of its element.
unsafe impl<M, T, const N: usize> UniformPayload<M> for [T; N] where T: UniformPayload<M> {}

macro_rules! tuple {
    ($($T:ident),+) => {
        // SAFETY: a tuple holds each part as a field.
        unsafe impl<M, $($T),+> UniformPayload<M> for ($($T,)+)
        where
            $($T: UniformPayload<M>,)+
        {
        }
    };
}

tuple!(A);
tuple!(A, B);
tuple!(A, B, C);
tuple!(A, B, C, D);
tuple!(A, B, C, D, E);
tuple!(A, B, C, D, E, F);
tuple!(A, B, C, D, E, F, G);
tuple!(A, B, C, D, E, F, G, H);
