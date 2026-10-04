// expect: E0080
// A mixed type behind a reference in a signature is still refused.
use forms::*;
const _: () = <(&Map<f64, Uniform<Var<0>>>, u64) as Form>::CHECKED;
