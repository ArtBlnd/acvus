// expect: E0080
// An argument rooted at an extern-defined type that holds a `Uniform`: neither form.
use forms::*;
const _: () = <Stack<Stack<Uniform<f64>>> as Form>::CHECKED;
