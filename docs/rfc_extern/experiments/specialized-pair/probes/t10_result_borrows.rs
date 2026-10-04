// expect: pass
// A result that borrows the parameter: `view<'a, T>(v: &'a Stack<T>) -> &'a Stack<T>`.
use pair::*;
pub type ViewSig<'l0, T> = <((&'l0 Stack<T>,), &'l0 Stack<T>) as Native>::Out;
pub fn check<'l0>() { same::<ViewSig<'l0, f64>, ((&'l0 Stack<f64>,), &'l0 Stack<f64>)>(); }
