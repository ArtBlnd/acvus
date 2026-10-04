// expect: E0080
// A whole native type held uniformly as an argument: a mark would stand off a root.
use forms::*;
const _: () = <Stack<Uniform<Stack<f64>>> as Form>::CHECKED;
