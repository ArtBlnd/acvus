// expect: E0107
// The uniform member has two lifetimes; the `#` member merges them into one.
use pair::*;
pub type PickSig<'l0, 'l1, T> = <((&'l0 Stack<T>, &'l1 Stack<T>), u64) as Native>::Out;
pub fn check<'l0>() { same::<PickSig<'l0, f64>, ((&'l0 Stack<f64>, &'l0 Stack<f64>), u64)>(); }
