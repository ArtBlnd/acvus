// expect: pass
// A type with no argument has no other form to be told apart from: unmarked.
use forms::*;
const _: () = <Counter as Form>::CHECKED;
const _: () = assert!(!<Counter as Form>::MARKED);
