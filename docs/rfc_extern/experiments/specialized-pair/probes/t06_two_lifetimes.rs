// expect: pass
// `pick<'a, 'b, T>(a: &'a Stack<T>, b: &'b mut Stack<T>) -> u64` at `T = f64`.
use pair::*;
pub type PickSig<'l0, 'l1, T> = <((&'l0 Stack<T>, &'l1 mut Stack<T>), u64) as Native>::Out;
pub fn check<'l0, 'l1>() { same::<PickSig<'l0, 'l1, f64>, ((&'l0 Stack<f64>, &'l1 mut Stack<f64>), u64)>(); }
