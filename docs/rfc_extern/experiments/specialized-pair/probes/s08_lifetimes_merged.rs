// expect: E0277
// The uniform member has two lifetimes; the `#` member merges them into one.
use pair::*;
pub type PickSig<T> = for<'l0, 'l1> fn(<&'l0 Stack<T> as Native>::Out, <&'l1 Stack<T> as Native>::Out) -> <u64 as Native>::Out;
pub fn check() { same::<PickSig<f64>, for<'a> fn(&'a Stack<f64>, &'a Stack<f64>) -> u64>(); }
