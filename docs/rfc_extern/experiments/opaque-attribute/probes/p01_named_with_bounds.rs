// expect: pass
// What `#[opaque(lifetimes('f), runtime = R, args(It, T), effect = E)]`
// writes: each named parameter bounded by its role, so the fields that
// hold an `Instance` and a kept function are well formed. Both are callable,
// so each carries the type's effect variable `E`: the `Instance` through its
// choice of `next`'s variables, the kept function through its alias's.
use model::*;

pub struct Filter<'f, R, It, T, E>
where
    R: Runtime,
    It: TypeVar<R>,
    T: TypeVar<R>,
    E: EffectVar,
{
    pub source: Instance<R, Next, It, (It, T, E)>,
    pub pred: Uniform<R, Pred<'f, T, E>>,
}
