// expect: pass
// `sum(v: &Stack<Uniform<f64>>) -> f64` and its `#` member `sum_native`.
use pair::*;
pub type SumSig = for<'l0> fn(<&'l0 Stack<Uniform<f64>> as Native>::Out) -> <f64 as Native>::Out;
pub fn check() { same::<SumSig, for<'l0> fn(&'l0 Stack<f64>) -> f64>(); }
