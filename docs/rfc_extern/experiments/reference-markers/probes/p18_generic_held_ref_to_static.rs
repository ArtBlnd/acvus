// expect: lifetime may not live long enough
use model::{Shared, TypeVar, Uniform};
pub fn leak<'a, T: TypeVar>(t: &'a T) -> Uniform<Shared<'static, T>> { Uniform::from_ref(t.as_ref_var()) }
