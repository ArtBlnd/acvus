// expect: E0310
use model::{Lend, TypeVar, Uniform};
pub fn wrap<T: TypeVar + Lend>(t: T) -> Uniform<T> { Uniform::new(t) }
