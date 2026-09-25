//! The strength of a borrow, as a type parameter.
//!
//! A bare marker pair like `Uniform`/`Specialized` was the other candidate and
//! folds nothing: the two impls that differ only in `&`/`&mut` would still be
//! written twice, once per tag. What lets one body serve both bits is that the
//! loan carries the Rust borrow, the runtime calls that open it, and the acvus
//! mutability.

use std::marker::PhantomData;

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
    const EXCLUSIVE: bool = matches!(Self::MUTABILITY, Mutability::Mut);

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

    /// A projection `project` made of the storage `reference` names has
    /// ended: at `Mut`, each storage the projection lent in place goes to
    /// `Runtime::loan_ended`, through `Project::loan_ended` over the same
    /// storage and table.
    ///
    /// # Safety
    /// As `project`'s, and no borrow the projection handed out is live.
    unsafe fn projection_ended<T, Rt>(
        rt: &Rt,
        reference: &Rt::Value,
        table: &<T as Project<Rt>>::Table,
    ) where
        T: Project<Rt>,
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

    #[inline(always)]
    unsafe fn projection_ended<T, Rt>(_: &Rt, _: &Rt::Value, _: &<T as Project<Rt>>::Table)
    where
        T: Project<Rt>,
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

    #[inline(always)]
    unsafe fn projection_ended<T, Rt>(rt: &Rt, reference: &Rt::Value, table: &<T as Project<Rt>>::Table)
    where
        T: Project<Rt>,
        Rt: Runtime,
    {
        // SAFETY: the caller's contract: the storage `project` read is live,
        // and no borrow of it is.
        let value = unsafe { <Rt::Value as Borrowable<Rt>>::deref_mut(rt, reference) };
        // SAFETY: as above, over the value `project_mut` was handed.
        unsafe { T::loan_ended(rt, value, table) }
    }
}

/// What a `Lending` lent, as far as the end of its loan needs it.
#[doc(hidden)]
pub trait Ending<Rt>
where
    Rt: Runtime,
{
    /// As `Borrowable::LENDS_A_WORD`.
    const LENDS_A_WORD: bool;
}

/// A `D` lent through representation `C`'s `Lends`.
#[doc(hidden)]
pub struct Through<C, D>(PhantomData<fn() -> (C, D)>);

impl<C, D, Rt> Ending<Rt> for Through<C, D>
where
    C: Lends<D, Rt>,
    Rt: Runtime,
{
    const LENDS_A_WORD: bool = <C as Lends<D, Rt>>::LENDS_A_WORD;
}

/// A storage whose type the lender does not name, which may be a word.
#[doc(hidden)]
pub struct Unnamed;

impl<Rt> Ending<Rt> for Unnamed
where
    Rt: Runtime,
{
    const LENDS_A_WORD: bool = true;
}

/// A borrow lent in place through the reference word this holds, and the
/// end of that loan: dropping it hands the storage to `M::loan_ended`, after
/// the borrow's user returns and while a panic in it unwinds, where `W`
/// lends a word.
///
/// The borrow is read through `reference`, whose result lives no longer than
/// the borrow of this value, so no borrow made through it outlives the drop
/// that ends it. The glue a macro writes lends a receiver and each exclusive
/// position of a signature's rest through one, and `Ref::with` its storage.
#[doc(hidden)]
pub struct Lending<'r, M, Rt, W = Unnamed>
where
    M: Loan,
    Rt: Runtime,
    W: Ending<Rt>,
{
    rt: &'r Rt,
    reference: Rt::Value,
    loan: PhantomData<fn() -> (M, W)>,
}

impl<'r, M, Rt, W> Lending<'r, M, Rt, W>
where
    M: Loan,
    Rt: Runtime,
    W: Ending<Rt>,
{
    /// # Safety
    /// `reference` is a reference word naming a live storage that outlives
    /// this value, named by nothing else while a borrow made through it at
    /// `Mut` lives.
    #[inline(always)]
    pub unsafe fn of(rt: &'r Rt, reference: Rt::Value) -> Lending<'r, M, Rt, W> {
        Lending {
            rt,
            reference,
            loan: PhantomData,
        }
    }

    /// The reference word a borrow of the storage is read through.
    #[inline(always)]
    pub fn reference(&self) -> &Rt::Value {
        &self.reference
    }
}

impl<M, Rt, W> Drop for Lending<'_, M, Rt, W>
where
    M: Loan,
    Rt: Runtime,
    W: Ending<Rt>,
{
    #[inline(always)]
    fn drop(&mut self) {
        if W::LENDS_A_WORD {
            // SAFETY: `of`'s contract: the storage is live, and every borrow
            // read through `reference` borrowed `self`, so none is live here.
            unsafe { M::loan_ended(self.rt, &self.reference) }
        }
    }
}
