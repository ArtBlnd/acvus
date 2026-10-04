// expect: pass
// A signature's positions, each checked: a marked type behind a reference, a
// uniform one inside an `Option`, and a scalar.
use forms::*;
const _: () = <(&Stack<f64>, Option<Stack<Uniform<Var<0>>>>, u64) as Form>::CHECKED;
