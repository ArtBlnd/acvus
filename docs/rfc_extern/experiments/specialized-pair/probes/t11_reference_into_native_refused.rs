// expect: E0277
// `first<'a, T>(v: &'a Stack<T>) -> Option<Ref<'a, T>>` lends an element; a `#`
// value holds no `Value` to lend, so the pair has no native form.
use pair::*;
pub type FirstSig<'l0, T> = <((&'l0 Stack<T>,), Option<Ref<'l0, T>>) as Native>::Out;
pub fn check<'l0>() { same::<FirstSig<'l0, f64>, ((&'l0 Stack<f64>,), Option<&'l0 f64>)>(); }
