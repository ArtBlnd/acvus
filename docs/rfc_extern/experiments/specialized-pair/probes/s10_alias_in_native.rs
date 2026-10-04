// expect: pass
// The `#` member writes its stack through an alias.
use pair::*;
type Floats = Stack<f64>;
pub type SumSig = for<'l0> fn(<&'l0 Stack<Uniform<f64>> as Native>::Out) -> <f64 as Native>::Out;
pub fn check() { same::<SumSig, for<'l0> fn(&'l0 Floats) -> f64>(); }
