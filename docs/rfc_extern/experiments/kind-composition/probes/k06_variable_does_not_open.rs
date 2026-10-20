// expect: E0277
// Generic code does not open a variable's value: its kind is not known there,
// so a reference to it stays a reference (variables §1).
use model::{Ref, Runtime, TypeVar};

pub fn peek<R, T>(item: Ref<'_, R, T>)
where
    R: Runtime,
    T: TypeVar<R>,
{
    item.project(|_| ());
}
