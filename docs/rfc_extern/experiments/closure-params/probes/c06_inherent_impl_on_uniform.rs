// expect: E0116
// The author's crate cannot add an inherent method to the extern crate's
// `Uniform`, so a kept function's projection is a trait of the author's crate.
use std::marker::PhantomData;
use model::{EffectVar, TypeVar, Uniform};
pub struct Pred<'f, T, E>(PhantomData<(fn(T) -> T, fn() -> &'f (), fn(E) -> E)>);
impl<'f, T: TypeVar, E: EffectVar> Uniform<Pred<'f, T, E>> {
    pub fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's { move |_x: &T| { let _ = self.bits(); true } }
}
