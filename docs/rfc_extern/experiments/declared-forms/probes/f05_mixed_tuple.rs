// expect: E0080
// An argument rooted at a tuple that holds a `Uniform`: neither form.
use forms::*;
const _: () = <Stack<(f64, Uniform<String>)> as Form>::CHECKED;
