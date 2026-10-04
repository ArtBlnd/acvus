// expect: pass
// The `#` member writes its stack through an alias.
use pair::*;
type Floats = Stack<f64>;
pub type SumSig<'l0> = <((&'l0 Stack<Uniform<f64>>,), f64) as Native>::Out;
pub fn check<'l0>() { same::<SumSig<'l0>, ((&'l0 Floats,), f64)>(); }
