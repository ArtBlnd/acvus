//! Array-length positions in ExternFn signatures.
//!
//! A length variable is named by `Nth<kind::Length, K>` while the
//! declaration's type is built. `Arr<T, N>` is an array whose length the
//! script decides.

use std::marker::PhantomData;

use acvus_mir::ty::{HeldTy, Poly, PolyTy, TypeArg};
use acvus_utils::Interner;

use crate::canonical::Canonical;
use crate::ty_arg::{PolyVars, SlotRepr, Term, TyArg, Var, kind};

/// The runtime carries no length: a length variable is settled before the
/// handler runs, so the runtime fills it with nothing.
impl Var<kind::Length> for () {}

// SAFETY: a length holds no `Erased`.
unsafe impl Canonical<kind::Length> for () {
    type Canon = Self;
}

/// `Array<T, N>` with N a length variable. Holds the elements at runtime.
///
/// `Cross::deref` in `obj.rs` reads the runtime's `Arr<Value, ()>` as an
/// `Arr<Value, N>` through this `repr(transparent)`; drop the attribute and
/// that file still compiles while the read stands on nothing the language
/// promises.
#[repr(transparent)]
pub struct Arr<T, N>(pub Vec<T>, PhantomData<N>)
where
    N: Var<kind::Length>;

impl<T, N> Arr<T, N>
where
    T: Send + Sync + 'static,
    N: Var<kind::Length>,
{
    pub fn new(items: Vec<T>) -> Self {
        Self(items, PhantomData)
    }
}

impl<T, N> IntoIterator for Arr<T, N>
where
    T: Send + Sync + 'static,
    N: Var<kind::Length>,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T, N> Var<kind::Type> for Arr<T, N>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
}

// SAFETY: the element is its own canonical form's, and a length holds no
// `Erased`.
unsafe impl<T, N> Canonical<kind::Type> for Arr<T, N>
where
    T: Var<kind::Type>,
    N: Var<kind::Length>,
{
    type Canon = Arr<T::Canon, N>;
}

// SAFETY: an `Arr` is a `Vec<T>`, and a length has no layout.
unsafe impl<M, T, N> crate::UniformPayload<M> for Arr<T, N>
where
    T: Send + Sync + 'static + crate::UniformPayload<M>,
    N: Var<kind::Length>,
{
}

impl<T, N> TyArg for Arr<T, N>
where
    T: TyArg + Send + Sync + 'static,
    N: Term<kind::Length> + Var<kind::Length>,
{
    const SLOT: SlotRepr = T::SLOT;

    fn poly_ty(i: &Interner, vars: &PolyVars) -> PolyTy {
        PolyTy::Array(Box::new(T::poly_ty(i, vars)), N::poly(vars))
    }

    fn held(i: &Interner, vars: &PolyVars) -> TypeArg<Poly> {
        TypeArg::Specialized(HeldTy::Array(Box::new(T::held(i, vars)), N::poly(vars)))
    }
}
