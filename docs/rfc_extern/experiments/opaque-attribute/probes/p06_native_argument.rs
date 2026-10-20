// expect: E0277
// A type argument bounded by `TypeVar` takes only a `Uniform`: the type
// written with a native argument, as a whole native `#` form would write
// it (types §3), is not a type. The runtime is made used by a marker, so
// that the refusal is the bound's alone.
use model::*;

pub struct Stack<R, T>
where
    R: Runtime,
    T: TypeVar<R>,
{
    pub items: Vec<T>,
    pub runtime: core::marker::PhantomData<fn() -> R>,
}

pub fn native(_: Stack<Probe, i64>) {}
