//! A type and its canonical form are one layout (G5), so a value stored as its
//! canonical form is read at the type the checker settled.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;

use crate::boundary::signed::Structural;

/// The proof a layout-only cast needs. Made only from a type's `Structural`
/// signature; every later typed read of what it cast goes back to the runtime.
pub struct LayoutWitness<A, B> {
    _same: PhantomData<(fn(A) -> A, fn(B) -> B)>,
}

impl<T> LayoutWitness<T, T::Canon>
where
    T: Structural,
{
    /// The witness a type's structural signature gives, with its size and
    /// alignment asserted per instantiation.
    pub(crate) const fn canonical() -> Self {
        const {
            assert!(
                size_of::<T>() == size_of::<T::Canon>(),
                "a type and its canonical form differ in size"
            );
            assert!(
                align_of::<T>() == align_of::<T::Canon>(),
                "a type and its canonical form differ in alignment"
            );
        }
        LayoutWitness { _same: PhantomData }
    }
}

// Every cast below rests on one fact: a witness exists only where `canonical`
// made it, so `A` is a `Structural` type and `B` its `Canon`. Their size and
// alignment are equal by `canonical`'s assert at this instantiation, and their
// field offsets, the validity of each byte and their release are one by `A`'s
// `Structural` signature (signed.rs).
impl<A, B> LayoutWitness<A, B> {
    pub(crate) fn cast(self, value: A) -> B {
        let value = ManuallyDrop::new(value);
        // SAFETY: the witness's fact puts a valid `B` in `value`'s place, and
        // `ManuallyDrop` keeps `value` from being released, so what `B`'s drop
        // releases is released once.
        unsafe { ptr::read(ptr::from_ref(&value).cast::<B>()) }
    }

    pub(crate) fn cast_ref(self, value: &A) -> &B {
        // SAFETY: the witness's fact makes the place a valid `B` for as long as
        // it is a valid `A`, and the shared borrow is carried over unchanged.
        unsafe { &*ptr::from_ref(value).cast::<B>() }
    }

    pub(crate) fn cast_back_ref(self, value: &B) -> &A {
        // SAFETY: as `cast_ref`, from the other side.
        unsafe { &*ptr::from_ref(value).cast::<A>() }
    }

    pub(crate) fn cast_back_mut(self, value: &mut B) -> &mut A {
        // SAFETY: the witness's fact makes every valid `A` written through the
        // result a valid `B` in the same place, and the exclusive borrow is
        // carried over unchanged.
        unsafe { &mut *ptr::from_mut(value).cast::<A>() }
    }
}

#[cfg(test)]
mod tests {
    use super::LayoutWitness;

    type Parts = (i64, String, Option<char>, Result<bool, u8>);

    fn parts() -> Parts {
        (-7, String::from("canon"), Some('x'), Err(3))
    }

    #[test]
    fn a_value_cast_to_its_canonical_form_is_the_same_value() {
        let cast = LayoutWitness::<Parts, _>::canonical().cast(parts());

        assert_eq!(cast, parts());
    }

    #[test]
    fn a_shared_reference_reads_the_same_place_either_way() {
        let value = parts();

        let canon = LayoutWitness::<Parts, _>::canonical().cast_ref(&value);
        let back = LayoutWitness::<Parts, _>::canonical().cast_back_ref(canon);

        assert_eq!(canon, &parts());
        assert!(std::ptr::eq(back, &value));
    }

    #[test]
    fn a_write_through_the_cast_back_reference_lands_in_the_canonical_place() {
        let mut canon = parts();

        let back = LayoutWitness::<Parts, _>::canonical().cast_back_mut(&mut canon);
        back.1.push_str("ical");
        back.2 = None;

        assert_eq!(canon, (-7, String::from("canonical"), None, Err(3)));
    }
}
