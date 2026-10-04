// expect: E0080
// One argument uniform, the other native: neither form.
use forms::*;
const _: () = <Map<Uniform<String>, i64> as Form>::CHECKED;
