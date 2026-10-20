// expect: E0133
// A tuple is made from its parts' values, which it takes over: giving them is
// an `unsafe fn`, since a `Value` carries neither its type nor its owner
// (safety §3).
use model::{Runtime, Tuples};

pub fn tie<R>(first: R::Value, second: R::Value) -> R::Value
where
    R: Runtime,
{
    <R::Tuples as Tuples<R>>::give([first, second])
}
