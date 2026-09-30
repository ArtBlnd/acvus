//! A type and its canonical form are one layout (G5), so a value stored as its
//! canonical form is read at the type the checker settled.

use std::marker::PhantomData;

use crate::boundary::signed::Structural;

/// The proof a layout-only cast needs. Made only from a type's `Structural`
/// signature; every later typed read of what it cast goes back to the runtime.
pub struct LayoutWitness<A, B> {
    _same: PhantomData<(fn(A) -> A, fn(B) -> B)>,
}

impl<T: Structural> LayoutWitness<T, T::Canon> {
    /// The witness a type's structural signature gives, with its size and
    /// alignment asserted per instantiation.
    pub(crate) const fn canonical() -> Self {
        panic!("fill: const-assert size and alignment of T and T::Canon")
    }
}

impl<A, B> LayoutWitness<A, B> {
    pub(crate) fn cast(self, value: A) -> B {
        todo!("cast by value")
    }

    pub(crate) fn cast_ref(self, value: &A) -> &B {
        todo!("cast a shared reference")
    }

    pub(crate) fn cast_back_ref(self, value: &B) -> &A {
        todo!("cast a shared reference back")
    }

    pub(crate) fn cast_back_mut(self, value: &mut B) -> &mut A {
        todo!("cast an exclusive reference back")
    }
}
