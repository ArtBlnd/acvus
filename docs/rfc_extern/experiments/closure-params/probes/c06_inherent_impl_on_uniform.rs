// expect: E0116
// The author's crate cannot add an inherent method to the extern crate's
// `Uniform`, so a kept function's view is a trait of the author's crate.
use std::marker::PhantomData;
use model::{TypeVar, Uniform};
pub struct Pred<'f, T>(PhantomData<(fn(T) -> T, fn() -> &'f ())>);
impl<'f, T: TypeVar> Uniform<Pred<'f, T>> {
    pub fn as_fn<'s>(&'s self) -> impl Fn(&T) -> bool + 's { move |_x: &T| { let _ = self.bits(); true } }
}
