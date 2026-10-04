// expect: E0277
// The uniform member shares one lifetime; the `#` member splits it in two.
use pair::*;
pub type KeepSig<T> = for<'l0> fn(<&'l0 Stack<T> as Native>::Out, <&'l0 Stack<T> as Native>::Out) -> <u64 as Native>::Out;
pub fn check() { same::<KeepSig<f64>, for<'a, 'b> fn(&'a Stack<f64>, &'b Stack<f64>) -> u64>(); }
