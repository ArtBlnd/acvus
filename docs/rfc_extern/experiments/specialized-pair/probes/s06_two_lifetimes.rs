// expect: pass
// `pick<'a, 'b, T>(a: &'a Stack<T>, b: &'b mut Stack<T>) -> u64` at `T = f64`.
use pair::*;
pub type PickSig<T> = for<'l0, 'l1> fn(<&'l0 Stack<T> as Native>::Out, <&'l1 mut Stack<T> as Native>::Out) -> <u64 as Native>::Out;
pub fn check() { same::<PickSig<f64>, for<'a, 'b> fn(&'a Stack<f64>, &'b mut Stack<f64>) -> u64>(); }
