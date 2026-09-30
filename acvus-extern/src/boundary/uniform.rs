//! A type variable's value: the runtime's uniform value indexed by the
//! variable, owned, and released once by its drop (G7).

use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use crate::runtime::Runtime;

/// The permission to release one value, made only by an owning holder's drop.
pub struct Releasing {
    _private: (),
}

/// A runtime value's release. The value is `Copy`; only its owning holder
/// releases it, through a `Releasing` it alone can make.
pub trait Release: Copy {
    fn release(self, releasing: Releasing);
}

/// The uniform value at a type variable `X`. No trait implemented on it has an
/// item that depends on `X` (runtime contract §1).
#[repr(transparent)]
pub struct Uniform<R: Runtime, X> {
    value: ManuallyDrop<R::Value>,
    _index: PhantomData<fn() -> X>,
}

impl<R: Runtime, X> Drop for Uniform<R, X> {
    fn drop(&mut self) {
        todo!("release the value once through a Releasing")
    }
}
