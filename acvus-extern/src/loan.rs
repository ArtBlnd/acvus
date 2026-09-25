//! The strength of a borrow, as a type parameter.
//!
//! A bare marker pair like `Uniform`/`Specialized` was the other candidate and
//! folds nothing: the two impls that differ only in `&`/`&mut` would still be
//! written twice, once per tag. What lets one body serve both bits is that the
//! loan carries the Rust borrow, the runtime calls that open it, and the acvus
//! mutability.

use acvus_mir::ty::Mutability;

use crate::handler::{Borrowable, Lends};
use crate::projection::{Borrowed, Project};
use crate::runtime::Runtime;

/// The strength of a borrow: what separated every `X`/`XMut` pair. A
/// handler names its two markers, `Shared` and `Mut`, as type arguments of
/// `Ref` and `Slice`; the methods are the crossing's.
#[doc(hidden)]
pub trait Loan: Send + Sync + 'static {
    const MUTABILITY: Mutability;

    /// A Rust borrow of `T` at this strength.
    type Of<'a, T>
    where
        T: ?Sized + 'a;

    type Projection<'a, T>
    where
        T: Borrowed + 'a;

    /// # Safety
    /// As `Runtime::deref`, exclusively for `Mut`.
    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;

    /// # Safety
    /// `reference` names a live storage holding what `T`'s crossing wrote,
    /// exclusively for `Mut`, and `table` was built from its settled type.
    unsafe fn project<'a, T, Rt>(
        rt: &'a Rt,
        reference: &'a Rt::Value,
        table: &<T as Project<Rt>>::Table,
    ) -> Self::Projection<'a, T>
    where
        T: Project<Rt> + 'a,
        Rt: Runtime;

    /// # Safety
    /// As `Borrowable::deref`, exclusively for `Mut`.
    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> Self::Of<'a, T>
    where
        T: Send + Sync,
        Rep: Lends<T, Rt>,
        Rt: Runtime;

    /// # Safety
    /// As `Runtime::value_as_ref`.
    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: Self::Of<'a, Rt::Value>) -> Self::Of<'a, T>
    where
        T: Send + Sync + 'static,
        Rt: Runtime;

    /// A borrow `deref` or `borrow` made of the storage `reference` names
    /// has ended: at `Mut`, the storage goes to `Runtime::loan_ended`.
    ///
    /// # Safety
    /// `reference` names a live storage, and no borrow of it is live.
    unsafe fn loan_ended<Rt>(rt: &Rt, reference: &Rt::Value)
    where
        Rt: Runtime;
}

pub struct Shared;
pub struct Mut;

impl Loan for Shared {
    const MUTABILITY: Mutability = Mutability::Shared;

    type Of<'a, T>
        = &'a T
    where
        T: ?Sized + 'a;

    type Projection<'a, T>
        = <T as Borrowed>::Ref<'a>
    where
        T: Borrowed + 'a;

    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.deref::<T>(reference) }
    }

    unsafe fn project<'a, T, Rt>(
        rt: &'a Rt,
        reference: &'a Rt::Value,
        table: &<T as Project<Rt>>::Table,
    ) -> <T as Borrowed>::Ref<'a>
    where
        T: Project<Rt> + 'a,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract: a live storage, which holds one of
        // the runtime's values.
        let value = unsafe { <Rt::Value as Borrowable<Rt>>::deref(rt, reference) };
        // SAFETY: the caller's contract.
        unsafe { T::project(rt, value, table) }
    }

    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync,
        Rep: Lends<T, Rt>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { Rep::deref(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a Rt::Value) -> &'a T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract.
        unsafe { rt.value_as_ref::<T>(value) }
    }

    #[inline(always)]
    unsafe fn loan_ended<Rt>(_: &Rt, _: &Rt::Value)
    where
        Rt: Runtime,
    {
    }
}

impl Loan for Mut {
    const MUTABILITY: Mutability = Mutability::Mut;

    type Of<'a, T>
        = &'a mut T
    where
        T: ?Sized + 'a;

    type Projection<'a, T>
        = <T as Borrowed>::Mut<'a>
    where
        T: Borrowed + 'a;

    unsafe fn deref<'a, T, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.deref_mut::<T>(reference) }
    }

    unsafe fn project<'a, T, Rt>(
        rt: &'a Rt,
        reference: &'a Rt::Value,
        table: &<T as Project<Rt>>::Table,
    ) -> <T as Borrowed>::Mut<'a>
    where
        T: Project<Rt> + 'a,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        let value = unsafe { <Rt::Value as Borrowable<Rt>>::deref_mut(rt, reference) };
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { T::project_mut(rt, value, table) }
    }

    unsafe fn borrow<'a, T, Rep, Rt>(rt: &Rt, reference: &'a Rt::Value) -> &'a mut T
    where
        T: Send + Sync,
        Rep: Lends<T, Rt>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { Rep::deref_mut(rt, reference) }
    }

    unsafe fn value_as<'a, T, Rt>(rt: &'a Rt, value: &'a mut Rt::Value) -> &'a mut T
    where
        T: Send + Sync + 'static,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract, exclusive for this loan.
        unsafe { rt.value_as_mut::<T>(value) }
    }

    #[inline(always)]
    unsafe fn loan_ended<Rt>(rt: &Rt, reference: &Rt::Value)
    where
        Rt: Runtime,
    {
        // SAFETY: the caller's contract: the storage is live and no other
        // name of it is in use.
        let storage = unsafe { <Rt::Value as Borrowable<Rt>>::deref_mut(rt, reference) };
        Rt::loan_ended(storage);
    }
}

/// An exclusive loan of the storage `reference` names, ended when this is
/// dropped: after the borrow's user returns, and while a panic in it unwinds.
pub(crate) struct Ending<'r, Rt>
where
    Rt: Runtime,
{
    rt: &'r Rt,
    reference: &'r Rt::Value,
}

impl<'r, Rt> Ending<'r, Rt>
where
    Rt: Runtime,
{
    /// # Safety
    /// `reference` names a live storage that outlives this value, and every
    /// borrow of it made meanwhile ends before this is dropped.
    pub(crate) unsafe fn of(rt: &'r Rt, reference: &'r Rt::Value) -> Ending<'r, Rt> {
        Ending { rt, reference }
    }
}

impl<Rt> Drop for Ending<'_, Rt>
where
    Rt: Runtime,
{
    fn drop(&mut self) {
        // SAFETY: `of`'s contract.
        unsafe { <Mut as Loan>::loan_ended(self.rt, self.reference) }
    }
}
