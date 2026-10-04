// expect: pass
// Generic code holds a reference to a type variable's value uniformly.
use model::{Shared, TypeVar, Uniform};
pub fn hold<'a, T: TypeVar>(t: &'a T) -> Uniform<Shared<'a, T>> { Uniform::from_ref(t.as_ref_var()) }
