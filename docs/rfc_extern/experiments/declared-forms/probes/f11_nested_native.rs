// expect: pass
// A native type whose argument is itself native: one whole, marked at its root.
use forms::*;
const _: () = <Stack<Stack<f64>> as Form>::CHECKED;
const _: () = assert!(<Stack<Stack<f64>> as Form>::MARKED);
