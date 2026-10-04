// expect: pass
// A type variable stands as `Uniform<Var<N>>`, so a container of it is uniform.
use forms::*;
const _: () = <Stack<Uniform<Var<0>>> as Form>::CHECKED;
const _: () = assert!(!<Stack<Uniform<Var<0>>> as Form>::MARKED);
