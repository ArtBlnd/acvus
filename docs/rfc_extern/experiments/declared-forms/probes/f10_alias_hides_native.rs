// expect: pass
// An alias that hides a whole native type from the syntax is still marked.
use forms::*;
type Floats = Stack<f64>;
const _: () = <Option<Floats> as Form>::CHECKED;
const _: () = assert!(<Floats as Form>::MARKED);
