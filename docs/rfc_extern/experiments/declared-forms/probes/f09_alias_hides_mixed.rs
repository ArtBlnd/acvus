// expect: E0080
// An alias that hides a mixed argument from the syntax.
use forms::*;
type P = (f64, Uniform<String>);
const _: () = <Stack<P> as Form>::CHECKED;
