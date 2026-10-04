//! Runtime behaviour of the decision, on the model: release, the borrowed
//! marker and the reference markers. Run as tests, natively and under Miri.

use model::{Exclusive, Never, Ref, Shared, TypeVar, Uniform, clear_arena, owners, view_as};

/// Generic code that owns a container of a type variable's values and drops
/// it, as an extern body may.
fn drop_all<T>(items: Vec<T>)
where
    T: TypeVar,
{
    drop(items);
}

/// A container of `T` standing for `&String`: the elements are references
/// held uniformly; generic code reads the container as `Vec<Uniform<Never>>`
/// and drops it. The owner keeps its one owner throughout.
#[test]
fn a_generic_drop_of_held_references_ends_no_owner() {
    let owner = Uniform::new(String::from("hello"));
    let held: Vec<Uniform<Shared<'_, String>>> = vec![Uniform::from_ref(owner.as_ref()), Uniform::from_ref(owner.as_ref())];
    drop_all(view_as::<_, Uniform<Never>>(held));
    assert_eq!(owners(owner.bits()), 1);
    let bits = owner.bits();
    drop(owner);
    assert_eq!(owners(bits), 0);
    clear_arena();
}

/// The negative control: a `Ref` that copies the owning `Value` bit for bit.
/// The same flow ends the owner twice before the owner itself drops.
#[test]
fn a_verbatim_ref_in_a_generic_container_ends_the_owner_early() {
    let owner = Uniform::new(String::from("hello"));
    let held: Vec<Uniform<Shared<'_, String>>> = vec![Uniform::from_ref(owner.as_ref_verbatim()), Uniform::from_ref(owner.as_ref_verbatim())];
    drop_all(view_as::<_, Uniform<Never>>(held));
    assert_eq!(owners(owner.bits()), -1);
    let bits = owner.bits();
    drop(owner);
    assert_eq!(owners(bits), -2);
    clear_arena();
}

/// A `Ref` is made once by `as_ref`; its copies, the `Ref` a held reference
/// gives back, and a type variable's `as_ref` of a held reference are all the
/// same `Value`.
#[test]
fn a_ref_copies_plainly_and_a_ref_of_a_ref_is_that_ref() {
    {
        let owner = Uniform::new(String::from("text"));
        let first = owner.as_ref();
        let copy = first;
        let held = Uniform::from_ref(first);
        let given_back = held.get();
        let of_a_variable: Ref<'_, Uniform<Shared<'_, String>>> = held.as_ref_var();
        assert_eq!(first.bits(), copy.bits());
        assert_eq!(first.bits(), given_back.bits());
        assert_eq!(first.bits(), of_a_variable.bits());
        assert_ne!(first.bits(), owner.bits());
        assert_eq!(given_back.with(String::clone), "text");
        drop(held);
        assert_eq!(owners(owner.bits()), 1);
    }
    clear_arena();
}

/// A scalar is held in place; its `Ref` is the same bits, and a release of
/// either ends nothing.
#[test]
fn a_scalar_ref_is_its_bits() {
    let owner = Uniform::new(7i64);
    let reference = owner.as_ref();
    assert_eq!(reference.bits(), owner.bits());
    assert_eq!(reference.bits().p, 0);
    let held = vec![Uniform::from_ref(reference)];
    drop_all(view_as::<_, Uniform<Never>>(held));
    assert_eq!(owner.with(|scalar| *scalar), 7);
}

/// An exclusive reference held uniformly writes through its place, and its
/// drop ends nothing.
#[test]
fn a_held_exclusive_reference_writes_through_its_place() {
    let mut owner = Uniform::new(1i64);
    {
        let mut held = Uniform::<Exclusive<'_, i64>>::from_mut(owner.as_mut());
        held.as_mut().set(9);
        let as_variable = vec![held];
        drop_all(view_as::<_, Uniform<Never>>(as_variable));
    }
    assert_eq!(owner.with(|scalar| *scalar), 9);
}
