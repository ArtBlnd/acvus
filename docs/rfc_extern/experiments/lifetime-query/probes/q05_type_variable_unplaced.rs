// expect: E0277
// A type variable left as itself: its one bound says nothing of its lifetimes.
use lq::*;
pub trait TypeVar {}
pub fn check<'x, T: TypeVar>() { same::<Cons<Lt<'x>, Nil>, <Cursor<'x, T> as Lifetimed>::Seq>(); }
