// expect: E0080
// An argument rooted at an `Option` that holds a `Uniform`: neither form.
use forms::*;
const _: () = <Stack<Option<Uniform<f64>>> as Form>::CHECKED;
