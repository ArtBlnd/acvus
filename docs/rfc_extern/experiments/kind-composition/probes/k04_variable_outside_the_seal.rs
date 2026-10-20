// expect: E0277
// No crate but the extern crate makes a type a `TypeVar`: the seal is not
// held by an author's type (variables §3).
use model::{Mut, Ref, Runtime, TypeVar};

pub struct Local;

impl<R> TypeVar<R> for Local
where
    R: Runtime,
{
    fn as_ref(&self) -> Ref<'_, R, Self> {
        unimplemented!()
    }

    fn as_mut(&mut self) -> Mut<'_, R, Self> {
        unimplemented!()
    }

    fn give(self) -> R::Value {
        unimplemented!()
    }

    unsafe fn take(_: R::Value) -> Self {
        unimplemented!()
    }
}
