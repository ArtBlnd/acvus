// expect: E0277
// A `#` member whose stack holds another element type.
use pair::*;
pub type SumSig = for<'l0> fn(<&'l0 Stack<Uniform<f64>> as Native>::Out) -> <f64 as Native>::Out;
pub fn check() { same::<SumSig, for<'l0> fn(&'l0 Stack<i64>) -> f64>(); }
