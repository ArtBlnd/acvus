// expect: E0107
// The uniform member shares one lifetime; the `#` member splits it in two, so
// it passes the family two lifetimes where the family takes one.
use pair::*;
pub type KeepSig<'l0, T> = <((&'l0 Stack<T>, &'l0 Stack<T>), u64) as Native>::Out;
pub fn check<'l0, 'l1>() { same::<KeepSig<'l0, 'l1, f64>, ((&'l0 Stack<f64>, &'l1 Stack<f64>), u64)>(); }
