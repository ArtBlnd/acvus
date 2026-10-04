// expect: pass
// An alias that resolves to a `Uniform` is uniform at its root.
use forms::*;
type U = Uniform<f64>;
const _: () = <Stack<U> as Form>::CHECKED;
const _: () = assert!(!<Stack<U> as Form>::MARKED);
