// expect: E0277
// A choice, `push<i64>`, that is not what the `#` member writes.
use pair::*;
pub type PushSig<T> = for<'l0> fn(<&'l0 mut Stack<T> as Native>::Out, <T as Native>::Out) -> <() as Native>::Out;
pub fn check() { same::<PushSig<i64>, for<'l0> fn(&'l0 mut Stack<f64>, f64)>(); }
