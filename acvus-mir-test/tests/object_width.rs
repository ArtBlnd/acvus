//! An object's field positions cross the boundary as `acvus_extern::FieldAt`,
//! a `u16`, so an object type wider than `ObjectTy::MAX_FIELDS` has positions
//! the machine cannot name. The checker refuses one where the width is made:
//! an object literal, and the union two field sets join to. What that is at
//! the contract: the refusal is a compile error with the two counts in it,
//! and nothing downstream converts a position.

use std::ops::Range;

use acvus_mir::ty::{Concrete, ObjectMeet, ObjectTy, TyTerm};
use acvus_mir_test::compile_script_ir;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

/// `{ f0: 0, f1: 1, ... }` with `fields` of them, bound and returned so the
/// literal's type is the one the refusal reads.
fn object_literal(fields: usize) -> String {
    let mut source = String::from("let wide = {\n");
    for at in 0..fields {
        source.push_str(&format!("  f{at}: {at},\n"));
    }
    source.push_str("};\nwide\n");
    source
}

/// The field set `f{at}: bool` over `names`.
fn field_set(interner: &Interner, names: Range<usize>) -> FxHashMap<Astr, TyTerm<Concrete>> {
    names
        .map(|at| (interner.intern(&format!("f{at}")), TyTerm::Bool))
        .collect()
}

#[test]
fn an_object_literal_over_the_bound_is_refused_with_both_counts() {
    let most = ObjectTy::<Concrete>::MAX_FIELDS;
    let interner = Interner::new();
    let refusal = compile_script_ir(&interner, &object_literal(most + 1), &FxHashMap::default())
        .expect_err("an object literal of MAX_FIELDS + 1 fields is refused");
    assert!(
        refusal.contains(&format!("object has {} fields", most + 1))
            && refusal.contains(&format!("at most {most}")),
        "the refusal names the literal's width and the bound: {refusal}"
    );
}

#[test]
fn a_field_set_of_exactly_the_bound_is_within_it() {
    let most = ObjectTy::<Concrete>::MAX_FIELDS;
    let interner = Interner::new();
    let at_the_bound = field_set(&interner, 0..most);
    assert_eq!(
        at_the_bound.len(),
        most,
        "the set is exactly the bound wide"
    );
    assert_eq!(
        ObjectTy::<Concrete>::too_wide(&at_the_bound),
        None,
        "the bound is the widest object admitted, not the first refused"
    );
}

/// Two field sets whose union is one field over the bound: the literal check
/// cannot see this, and `ObjectTy::meet` is where that width is made.
#[test]
fn the_union_of_two_admitted_field_sets_is_refused_over_the_bound() {
    let most = ObjectTy::<Concrete>::MAX_FIELDS;
    let interner = Interner::new();
    let half = most / 2 + 1;
    let a = ObjectTy::written(field_set(&interner, 0..half));
    let b = ObjectTy::written(field_set(&interner, half..half * 2));
    let distinct: FxHashSet<_> = a.keys().chain(b.keys()).collect();
    assert_eq!(
        distinct.len(),
        most + 1,
        "the two sets are disjoint and one field over the bound together"
    );
    assert_eq!(
        ObjectTy::<Concrete>::too_wide(&a),
        None,
        "the first side is within the bound on its own"
    );
    assert_eq!(
        ObjectTy::<Concrete>::too_wide(&b),
        None,
        "the second side is within the bound on its own"
    );
    assert!(
        matches!(ObjectTy::meet(&a, &b), ObjectMeet::TooWide { fields } if fields == most + 1),
        "the union is refused by its width, not joined"
    );
}
