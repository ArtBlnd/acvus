// expect: lifetime may not live long enough
// As many lifetimes on each side, in another pattern: (a, b, a) against (a, b, b).
use pair::*;
pub type Sig<'l0, 'l1, T> = <((&'l0 Stack<T>, &'l1 Stack<T>, &'l0 Stack<T>), u64) as Native>::Out;
pub fn check<'l0, 'l1>() { same::<Sig<'l0, 'l1, f64>, ((&'l0 Stack<f64>, &'l1 Stack<f64>, &'l1 Stack<f64>), u64)>(); }
