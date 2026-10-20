// expect: E0277
// A projection of an option is a Rust reference to its shape, its payload uniform
// (runtime §2): a payload written as a tuple is refused.
use model::{Options, Runtime};

pub fn is_some<R>(reference: R::Value) -> bool
where
    R: Runtime,
{
    // SAFETY: the caller's premise.
    unsafe { <R::Options as Options<R>>::project::<(i64, i64), _>(reference, Option::is_some) }
}
