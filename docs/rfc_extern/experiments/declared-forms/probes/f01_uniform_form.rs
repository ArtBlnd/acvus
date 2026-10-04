// expect: pass
// Every argument a `Uniform` at its root: the uniform form, unmarked.
use forms::*;
const _: () = <Stack<Uniform<f64>> as Form>::CHECKED;
const _: () = assert!(!<Stack<Uniform<f64>> as Form>::MARKED);
