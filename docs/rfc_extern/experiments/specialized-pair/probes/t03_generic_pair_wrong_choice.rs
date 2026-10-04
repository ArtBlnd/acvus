// expect: E0277
// A choice, `push<i64>`, that is not what the `#` member writes.
use pair::*;
pub type PushSig<'l0, T> = <((&'l0 mut Stack<T>, T), ()) as Native>::Out;
pub fn check<'l0>() { same::<PushSig<'l0, i64>, ((&'l0 mut Stack<f64>, f64), ())>(); }
