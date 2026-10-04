// expect: pass
// `sum(v: &Stack<Uniform<f64>>) -> f64` and its `#` member `sum_native`.
use pair::*;
pub type SumSig<'l0> = <((&'l0 Stack<Uniform<f64>>,), f64) as Native>::Out;
pub fn check<'l0>() { same::<SumSig<'l0>, ((&'l0 Stack<f64>,), f64)>(); }
