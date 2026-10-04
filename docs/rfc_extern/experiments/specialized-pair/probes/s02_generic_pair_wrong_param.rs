// expect: E0277
// A `#` member whose parameter is not the uniform member's at the choice.
use pair::*;
pub type PushSig<T> = for<'l0> fn(<&'l0 mut Stack<T> as Native>::Out, <T as Native>::Out) -> <() as Native>::Out;
pub fn check() { same::<PushSig<f64>, for<'l0> fn(&'l0 mut Stack<f64>, i64)>(); }
