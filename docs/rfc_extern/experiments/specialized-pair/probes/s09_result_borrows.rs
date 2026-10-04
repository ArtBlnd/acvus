// expect: pass
// A result that borrows the parameter: `view<'a, T>(v: &'a Stack<T>) -> &'a Stack<T>`.
use pair::*;
pub type ViewSig<T> = for<'l0> fn(<&'l0 Stack<T> as Native>::Out) -> <&'l0 Stack<T> as Native>::Out;
pub fn check() { same::<ViewSig<f64>, for<'a> fn(&'a Stack<f64>) -> &'a Stack<f64>>(); }
