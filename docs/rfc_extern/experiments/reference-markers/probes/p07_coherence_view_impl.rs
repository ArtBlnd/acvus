// expect: E0119
// An author's impl reached through the type variable bound covers a reference view too.
use model::{Shared, TypeVar, Uniform};
pub trait H { type A; }
impl<T: TypeVar> H for T { type A = u8; }
impl<'a> H for Uniform<Shared<'a, i64>> { type A = u64; }
