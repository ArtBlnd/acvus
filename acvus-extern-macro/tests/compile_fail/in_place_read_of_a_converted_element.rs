//! Every path that reads a container's storage in place asks `Borrowable`
//! (through `Lends` where it is written over the representation), so none
//! reaches the storage of a `Vec<i64>`, which is a `Vec<Owned<Rt>>`: the
//! in-place read itself, `Loan::borrow`, `Ref::with`, and each `Restore*`
//! position a shared signature's glue restores.
use acvus_extern::{
    Borrowable, ByRef, Crossing, Loan, Mut, Ref, RestoreExclusive, RestoreShared, Runtime, Shared,
    Uniform,
};

unsafe fn read_in_place<Rt>(rt: &Rt, reference: &Rt::Value) -> usize
where
    Rt: Runtime,
{
    unsafe { <Vec<i64> as Borrowable<Rt>>::deref(rt, reference) }.len()
}

unsafe fn borrow<Rt>(rt: &Rt, reference: &Rt::Value) -> usize
where
    Rt: Runtime,
{
    unsafe { <Shared as Loan>::borrow::<Vec<i64>, Uniform, Rt>(rt, reference) }.len()
}

fn with<Rt>(rt: &Rt, r: &Ref<Vec<i64>, Shared, Rt>) -> usize
where
    Rt: Runtime,
{
    r.with(rt, |xs| xs.len())
}

unsafe fn restore_shared<Rt>(rt: Crossing<'_, Rt>, at: &mut Rt::Value, crossed: &Rt::Value) -> usize
where
    Rt: Runtime,
{
    unsafe {
        <&Vec<i64> as RestoreShared<'_, ByRef<Vec<i64>, Shared, Uniform>, Rt>>::restore_shared(
            rt, at, crossed,
        )
    }
    .len()
}

unsafe fn restore_exclusive<Rt>(rt: Crossing<'_, Rt>, at: &mut Rt::Value, crossed: &mut Rt::Value)
where
    Rt: Runtime,
{
    unsafe {
        <&mut Vec<i64> as RestoreExclusive<'_, ByRef<Vec<i64>, Mut, Uniform>, Rt>>::restore_exclusive(
            rt, at, crossed,
        )
    }
    .clear()
}

fn main() {}
