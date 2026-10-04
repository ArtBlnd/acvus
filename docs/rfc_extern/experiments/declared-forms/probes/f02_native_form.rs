// expect: pass
// No `Uniform` anywhere: whole and native, marked.
use forms::*;
const _: () = <Stack<f64> as Form>::CHECKED;
const _: () = assert!(<Stack<f64> as Form>::MARKED);
