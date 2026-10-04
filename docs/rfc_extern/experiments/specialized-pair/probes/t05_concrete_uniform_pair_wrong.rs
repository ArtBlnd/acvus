// expect: E0277
// A `#` member whose stack holds another element type.
use pair::*;
pub type SumSig<'l0> = <((&'l0 Stack<Uniform<f64>>,), f64) as Native>::Out;
pub fn check<'l0>() { same::<SumSig<'l0>, ((&'l0 Stack<i64>,), f64)>(); }
