// expect: E0621
// The view's result borrows only the argument its signature names.
use std::marker::PhantomData;
use model::{Ref, TypeVar, Uniform};
pub struct Pick<'f, T>(PhantomData<(fn(T) -> T, fn() -> &'f ())>);
pub trait PickFn<T> { fn as_fn<'s>(&'s self) -> impl for<'p, 'q> Fn(&'p T, &'q T) -> Ref<'p, T> + 's; }
pub fn pick_wrong<'x, T: TypeVar>(f: &impl PickFn<T>, a: &T, b: &'x T) -> Ref<'x, T> { (f.as_fn())(a, b) }
