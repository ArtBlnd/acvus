// expect: E0277
// A `#` member whose parameter is not the uniform member's at the choice.
use pair::*;
pub type PushSig<'l0, T> = <((&'l0 mut Stack<T>, T), ()) as Native>::Out;
pub fn check<'l0>() { same::<PushSig<'l0, f64>, ((&'l0 mut Stack<f64>, i64), ())>(); }
