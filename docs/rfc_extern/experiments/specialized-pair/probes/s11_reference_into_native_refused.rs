// expect: E0277
// `first<'a, T>(v: &'a Stack<T>) -> Option<Ref<'a, T>>` lends an element; a `#`
// value holds no `Value` to lend, so the pair has no native form.
use pair::*;
pub type FirstSig<T> = for<'l0> fn(<&'l0 Stack<T> as Native>::Out) -> <Option<Ref<'l0, T>> as Native>::Out;
pub fn check() { same::<FirstSig<f64>, for<'a> fn(&'a Stack<f64>) -> Option<&'a f64>>(); }
