//! The law, as programs: each test is a ruling of `TERMS.md` run.
//!
//! A program is built behind each count of `let`s from none to three and settled
//! in 24 orders of asking, and comes to one outcome in all of them: a class is
//! decided once, and only when nothing can still be merged into it.

use std::collections::{BTreeMap, BTreeSet};

use super::language::Language;
use super::law::{Law, Looked, Offer, Pair, Place, Scribe};
use super::testing::{EXCLUSIVE, SHARED, U8, World, reorder};
use crate::primitive::adjustment::{Adjustment, Adjustments};
use crate::primitive::bound::{Bound, BoundOf};
use crate::primitive::class::{Live, Side, State};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::record::Coercion;
use crate::primitive::refusal::Refusal;
use crate::primitive::ty::{PartKey, Scalar, Shape};
use crate::primitive::{ClassId, ClosureId, ItemId, Name};
use crate::store::classes::Classes;

const U16: Shape<ClassId> = Shape::Scalar(Scalar::U16);
const CHAR: Shape<ClassId> = Shape::Scalar(Scalar::Char);
const INTEGERS: [Scalar; 9] = [
    Scalar::I8,
    Scalar::I16,
    Scalar::I32,
    Scalar::I64,
    Scalar::U8,
    Scalar::U16,
    Scalar::U32,
    Scalar::U64,
    Scalar::Usize,
];

fn field(name: &str) -> PartKey {
    PartKey::Field(Name::new(name))
}

fn object(world: &mut World, fields: &[(&str, ClassId)]) -> ClassId {
    world.text(Shape::Object(fields.iter().map(|(name, part)| (Name::new(*name), *part)).collect()))
}

fn some(world: &mut World, payload: ClassId) -> ClassId {
    world.text(Shape::Option(payload))
}

/// `if c { first } else { second }`.
fn either(world: &mut World, first: ClassId, second: ClassId) -> ClassId {
    let place = world.meets(first);
    world.enter(second, place);
    place
}

/// `value` as it is read after passing through `lets` bindings.
fn behind(world: &mut World, mut value: ClassId, lets: usize) -> ClassId {
    for _ in 0..lets {
        value = world.meets(value);
    }
    value
}

/// The one outcome `build` comes to, the value it marks being read behind each
/// count of `let`s and its classes asked in each order.
fn comes_to<F>(expected: &str, build: F)
where
    F: Fn(&mut World, usize) -> ClassId,
{
    for lets in 0..4 {
        for order in 0..24 {
            let mut world = World::new();
            let observed = build(&mut world, lets);
            reorder(&mut world.open, order);
            world.settle(&Language);
            assert_eq!(world.outcome_of(observed), expected, "behind {lets} lets, in order {order}");
        }
    }
}

// Join is not subtyping.

#[test]
fn an_exclusive_and_a_shared_reference_meet_as_shared_and_the_exclusive_one_is_reborrowed() {
    comes_to("&u8", |w, lets| {
        let (x, y) = (w.text(U8), w.text(U8));
        let (exclusive, _) = w.lend(x, EXCLUSIVE);
        let exclusive = behind(w, exclusive, lets);
        let (shared, _) = w.lend(y, SHARED);
        either(w, exclusive, shared)
    });

    let mut world = World::new();
    let (x, y) = (world.text(U8), world.text(U8));
    let (exclusive, _) = world.lend(x, EXCLUSIVE);
    let (shared, _) = world.lend(y, SHARED);
    let place = world.meets(exclusive);
    let exclusive_enters = world.items.iter().last().expect("an item was recorded").0;
    world.enter(shared, place);
    world.settle(&Language);
    let reborrow = vec![Adjustment::Coerce(Coercion::Reborrow)];
    assert_eq!(world.outcome.adjustments.get(&exclusive_enters), Some(&reborrow));
}

#[test]
fn objects_that_meet_have_the_fields_of_both() {
    comes_to("{x: u8, y: u8}", |w, lets| {
        let (a, b) = (w.text(U8), w.text(U8));
        let with_x = object(w, &[("x", a)]);
        let with_x = behind(w, with_x, lets);
        let with_y = object(w, &[("y", b)]);
        either(w, with_x, with_y)
    });
}

#[test]
fn never_meets_any_head_as_that_head() {
    comes_to("u8", |w, lets| {
        let returns = w.text(Shape::Never);
        let returns = behind(w, returns, lets);
        let one = w.text(U8);
        either(w, returns, one)
    });
}

#[test]
fn heads_of_two_constructors_do_not_meet_and_the_class_has_no_type() {
    comes_to("refused: Mismatch, UndecidedClass", |w, lets| {
        let one = w.text(U8);
        let one = behind(w, one, lets);
        let c = w.text(CHAR);
        either(w, one, c)
    });
}

// A hole is where types meet: a part is one class with what it meets, and its
// head is the join of everything that enters that class.

#[test]
fn the_parts_of_what_meets_are_one_class_whose_head_is_the_join_of_what_enters_it() {
    comes_to("Option<u8>", |w, lets| {
        let returns = w.text(Shape::Never);
        let returns = behind(w, returns, lets);
        let two = w.text(U8);
        let (first, second) = (some(w, returns), some(w, two));
        either(w, first, second)
    });
    comes_to("Option<Option<u8>>", |w, lets| {
        let returns = w.text(Shape::Never);
        let two = w.text(U8);
        let inner = some(w, returns);
        let inner = behind(w, inner, lets);
        let other = some(w, two);
        let (first, second) = (some(w, inner), some(w, other));
        either(w, first, second)
    });
    comes_to("Option<&u8>", |w, lets| {
        let (a, b) = (w.text(U8), w.text(U8));
        let (exclusive, _) = w.lend(a, EXCLUSIVE);
        let exclusive = behind(w, exclusive, lets);
        let (shared, _) = w.lend(b, SHARED);
        let (first, second) = (some(w, exclusive), some(w, shared));
        either(w, first, second)
    });
}

// References.

#[test]
fn a_lend_writes_a_reference_to_its_place_as_the_place_is() {
    comes_to("&&u8", |w, lets| {
        let x = w.text(U8);
        let (lent_x, _) = w.lend(x, SHARED);
        let a = behind(w, lent_x, lets);
        w.lend(a, SHARED).0
    });
    comes_to("refused: Mismatch, UndecidedClass", |w, lets| {
        let (x, y) = (w.text(U8), w.text(U8));
        let (lent_x, _) = w.lend(x, SHARED);
        let r = behind(w, lent_x, lets + 1);
        let (lent_r, _) = w.lend(r, SHARED);
        let (lent_y, _) = w.lend(y, SHARED);
        either(w, lent_r, lent_y)
    });
}

// A head with parts is made from its parts.

#[test]
fn a_part_named_on_a_class_is_the_part_in_its_place_once_the_class_is_decided() {
    comes_to("u8", |w, lets| {
        let x = w.text(U8);
        let o = object(w, &[("x", x)]);
        let o = behind(w, o, lets + 1);
        w.project(o, field("x"), Side::Upper).0
    });
}

#[test]
fn a_part_named_on_a_reference_is_read_through_it_and_the_projection_derefs() {
    comes_to("u8", |w, lets| {
        let x = w.text(U8);
        let o = object(w, &[("x", x)]);
        let o = behind(w, o, lets);
        let (lent, _) = w.lend(o, SHARED);
        let r = w.meets(lent);
        w.project(r, field("x"), Side::Upper).0
    });

    let mut world = World::new();
    let x = world.text(U8);
    let o = object(&mut world, &[("x", x)]);
    let (lent, _) = world.lend(o, SHARED);
    let r = world.meets(lent);
    let (_, projection) = world.project(r, field("x"), Side::Upper);
    world.settle(&Language);
    assert_eq!(world.outcome.adjustments, Adjustments::from([(projection, vec![Adjustment::Deref])]));
}

#[test]
fn a_part_named_on_a_class_with_no_place_for_it_is_refused() {
    comes_to("refused: NotHeld", |w, lets| {
        let x = w.text(U8);
        let x = behind(w, x, lets + 1);
        w.project(x, field("x"), Side::Upper).0
    });
}

#[test]
fn a_part_written_through_a_reference_is_one_class_with_the_part_it_lies_at() {
    comes_to("u8", |w, lets| {
        let returns = w.text(Shape::Never);
        let p = behind(w, returns, lets);
        let o = object(w, &[("x", p)]);
        let o = w.meets(o);
        let (lent, _) = w.lend(o, EXCLUSIVE);
        let r = w.meets(lent);
        let (part, _) = w.project(r, field("x"), Side::Lower);
        let two = w.text(U8);
        w.enter(two, part);
        part
    });
}

// Deciding: the rungs, and what is refused at the end.

#[test]
fn a_literal_nothing_else_decides_takes_the_fallback_of_its_domain() {
    comes_to("i32", |w, lets| {
        let zero = w.literal(&INTEGERS);
        behind(w, zero, lets)
    });
    comes_to("u8", |w, lets| {
        let zero = w.literal(&INTEGERS);
        let one = w.text(U8);
        let one = behind(w, one, lets);
        w.enter(one, zero);
        zero
    });
}

#[test]
fn a_class_nothing_is_said_of_takes_never() {
    comes_to("Option<!>", |w, lets| {
        let payload = w.undecided();
        let none = some(w, payload);
        behind(w, none, lets)
    });
}

#[test]
fn a_class_that_is_its_own_part_is_refused() {
    comes_to("refused: Occurs", |w, lets| {
        let x = w.undecided();
        let inside = behind(w, x, lets);
        let wrapped = some(w, inside);
        w.enter(wrapped, x);
        x
    });
}

#[test]
fn classes_that_enter_one_another_are_one_class() {
    comes_to("u8", |w, lets| {
        let (a, b) = (w.undecided(), w.undecided());
        w.enter(a, b);
        w.enter(b, a);
        let one = w.text(U8);
        let one = behind(w, one, lets);
        w.enter(one, a);
        b
    });
}

#[test]
fn a_part_meeting_a_declared_part_is_compared_with_it_and_not_merged() {
    comes_to("refused: Mismatch", |w, lets| {
        let declared_u8 = w.external(U8);
        let declared = w.external(Shape::Option(declared_u8));
        let sixteen = w.text(U16);
        let sixteen = behind(w, sixteen, lets);
        let written = some(w, sixteen);
        either(w, declared, written)
    });
}

// Subtyping the language gives, where an argument is passed.

/// `h(Option<U>, Option<U>)`: its two parameters, and the hole.
fn two_options(world: &mut World) -> (ClassId, ClassId, ClassId) {
    let hole = world.undecided();
    (world.external(Shape::Option(hole)), world.external(Shape::Option(hole)), hole)
}

/// `f(&T)`: its parameter, and the hole.
fn reference_to_hole(world: &mut World) -> (ClassId, ClassId) {
    let hole = world.undecided();
    (world.external(Shape::Ref(SHARED, hole)), hole)
}

/// `f(&u8)`.
fn reference_to_u8(world: &mut World) -> ClassId {
    let declared = world.external(U8);
    world.external(Shape::Ref(SHARED, declared))
}

/// What is recorded at the boundary `build` gives, with nothing else recorded
/// anywhere, or the causes the program is refused by.
fn taken_at<F>(build: F) -> Result<Vec<Adjustment>, String>
where
    F: Fn(&mut World) -> ItemId,
{
    let mut world = World::new();
    let at = build(&mut world);
    world.settle(&Language);
    if !world.outcome.refusals.is_empty() {
        let causes: BTreeSet<String> = world.outcome.refusals.iter().map(|refusal| format!("{:?}", refusal.cause)).collect();
        return Err(causes.into_iter().collect::<Vec<_>>().join(", "));
    }
    let taken = world.outcome.adjustments.get(&at).cloned().unwrap_or_default();
    let elsewhere = world.outcome.adjustments.keys().filter(|item| **item != at).count();
    assert_eq!(elsewhere, 0, "an adjustment is recorded at the boundary alone");
    Ok(taken)
}

fn coerced(coercions: &[Coercion]) -> Result<Vec<Adjustment>, String> {
    Ok(coercions.iter().copied().map(Adjustment::Coerce).collect())
}

#[test]
fn a_hole_passed_to_from_two_arguments_is_one_class_with_the_parts_of_both() {
    comes_to("u8", |w, lets| {
        let (first, second, hole) = two_options(w);
        let returns = w.text(Shape::Never);
        let p = behind(w, returns, lets);
        let two = w.text(U8);
        let (a, b) = (some(w, p), some(w, two));
        w.pass(a, first);
        w.pass(b, second);
        hole
    });
    comes_to("u8", |w, lets| {
        let (first, second, hole) = two_options(w);
        let returns = w.text(Shape::Never);
        let s = some(w, returns);
        let s = behind(w, s, lets);
        let two = w.text(U8);
        let b = some(w, two);
        w.pass(s, first);
        w.pass(b, second);
        hole
    });
    comes_to("&u8", |w, lets| {
        let (first, second, hole) = two_options(w);
        let (a, b) = (w.text(U8), w.text(U8));
        let (exclusive, _) = w.lend(a, EXCLUSIVE);
        let r = behind(w, exclusive, lets);
        let (shared, _) = w.lend(b, SHARED);
        let (x, y) = (some(w, r), some(w, shared));
        w.pass(x, first);
        w.pass(y, second);
        hole
    });
    comes_to("Option<u8>", |w, lets| {
        let hole = w.undecided();
        let second = w.external(Shape::Option(hole));
        let returns = w.text(Shape::Never);
        let p = behind(w, returns, lets);
        let a = some(w, p);
        let two = w.text(U8);
        let inner = some(w, two);
        let outer = some(w, inner);
        w.enter(a, hole);
        w.pass(outer, second);
        hole
    });
}

#[test]
fn a_hole_at_a_target_is_one_class_with_the_target_that_comes() {
    comes_to("u8", |w, lets| {
        let (first, hole) = reference_to_hole(w);
        let returns = w.text(Shape::Never);
        let p = behind(w, returns, lets);
        let (lent, _) = w.lend(p, SHARED);
        let two = w.text(U8);
        w.pass(lent, first);
        w.enter(two, hole);
        hole
    });
    comes_to("&u8", |w, lets| {
        let (first, hole) = reference_to_hole(w);
        let x = w.text(U8);
        let (lent_x, _) = w.lend(x, SHARED);
        let r = behind(w, lent_x, lets + 1);
        let (lent_r, _) = w.lend(r, SHARED);
        w.pass(lent_r, first);
        hole
    });
}

#[test]
fn a_part_passed_to_a_declared_part_is_compared_with_it() {
    comes_to("refused: Mismatch", |w, lets| {
        let declared = w.external(U8);
        let parameter = w.external(Shape::Option(declared));
        let returns = w.text(Shape::Never);
        let p = behind(w, returns, lets);
        let a = some(w, p);
        w.pass(a, parameter);
        p
    });
}

#[test]
fn never_is_passed_to_any_type() {
    assert_eq!(
        taken_at(|w| {
            let declared = w.external(U8);
            let returns = w.text(Shape::Never);
            w.pass(returns, declared)
        }),
        coerced(&[Coercion::Never])
    );
}

#[test]
fn an_exclusive_reference_is_passed_as_a_shared_one_and_not_the_other_way() {
    let passed = |of_argument, of_given| {
        taken_at(move |w| {
            let declared = w.external(U8);
            let parameter = w.external(Shape::Ref(of_given, declared));
            let x = w.text(U8);
            let (lent, _) = w.lend(x, of_argument);
            w.pass(lent, parameter)
        })
    };
    assert_eq!(passed(EXCLUSIVE, SHARED), coerced(&[Coercion::Reborrow]));
    assert_eq!(passed(SHARED, SHARED), coerced(&[]));
    assert_eq!(passed(EXCLUSIVE, EXCLUSIVE), coerced(&[]));
    assert_eq!(passed(SHARED, EXCLUSIVE), Err("NoChange".to_owned()));
}

#[test]
fn an_object_is_passed_as_one_with_fewer_fields_where_no_reference_is_above_it() {
    let fields = |w: &mut World, names: &[&str], declared: bool| -> ClassId {
        let parts: BTreeMap<Name, ClassId> = names
            .iter()
            .map(|name| (Name::new(*name), if declared { w.external(U8) } else { w.text(U8) }))
            .collect();
        if declared { w.external(Shape::Object(parts)) } else { w.text(Shape::Object(parts)) }
    };
    let passed = |argument: &'static [&'static str], given: &'static [&'static str], behind_reference: bool| {
        taken_at(move |w| {
            let (argument, given) = (fields(w, argument, false), fields(w, given, true));
            if behind_reference {
                let parameter = w.external(Shape::Ref(SHARED, given));
                let (lent, _) = w.lend(argument, SHARED);
                w.pass(lent, parameter)
            } else {
                w.pass(argument, given)
            }
        })
    };
    assert_eq!(passed(&["x", "y"], &["x"], false), coerced(&[Coercion::DropFields]));
    assert_eq!(passed(&["x"], &["x"], false), coerced(&[]));
    assert_eq!(passed(&["x"], &["x", "y"], false), Err("NoChange".to_owned()));
    assert_eq!(passed(&["x", "y"], &["x"], true), Err("NoChange".to_owned()));
    assert_eq!(passed(&["x"], &["x"], true), coerced(&[]));
}

#[test]
fn an_array_behind_a_reference_is_passed_as_a_slice() {
    let passed = |of_argument, of_given| {
        taken_at(move |w| {
            let declared = w.external(U8);
            let slice = w.external(Shape::Slice(declared));
            let parameter = w.external(Shape::Ref(of_given, slice));
            let element = w.text(U8);
            let array = w.text(Shape::Array(2, element));
            let (lent, _) = w.lend(array, of_argument);
            w.pass(lent, parameter)
        })
    };
    assert_eq!(passed(SHARED, SHARED), coerced(&[Coercion::Unsize]));
    assert_eq!(passed(EXCLUSIVE, SHARED), coerced(&[Coercion::Unsize, Coercion::Reborrow]));
    assert_eq!(
        taken_at(|w| {
            let declared = w.external(U8);
            let slice = w.external(Shape::Slice(declared));
            let element = w.text(U8);
            let array = w.text(Shape::Array(2, element));
            w.pass(array, slice)
        }),
        Err("NoChange".to_owned())
    );
}

#[test]
fn references_above_a_shared_reference_are_read_through_where_the_types_do_not_match_as_written() {
    // `let r = &ₘ x; f(&ₙ r);` with `f(&u8)`.
    let passed = |of_inner, of_outer| {
        taken_at(move |w| {
            let parameter = reference_to_u8(w);
            let x = w.text(U8);
            let (lent_x, _) = w.lend(x, of_inner);
            let r = w.meets(lent_x);
            let (lent_r, _) = w.lend(r, of_outer);
            w.pass(lent_r, parameter)
        })
    };
    assert_eq!(passed(SHARED, SHARED), Ok(vec![Adjustment::Deref]));
    assert_eq!(
        passed(SHARED, EXCLUSIVE),
        Ok(vec![Adjustment::Deref, Adjustment::Coerce(Coercion::Reborrow)])
    );
    assert_eq!(passed(EXCLUSIVE, SHARED), Err("NoChange".to_owned()));
    assert_eq!(passed(EXCLUSIVE, EXCLUSIVE), Err("NoChange".to_owned()));

    // `let r = &x; let q = &r; f(&q);` with `f(&u8)`.
    assert_eq!(
        taken_at(|w| {
            let parameter = reference_to_u8(w);
            let x = w.text(U8);
            let (lent_x, _) = w.lend(x, SHARED);
            let r = w.meets(lent_x);
            let (lent_r, _) = w.lend(r, SHARED);
            let q = w.meets(lent_r);
            let (lent_q, _) = w.lend(q, SHARED);
            w.pass(lent_q, parameter)
        }),
        Ok(vec![Adjustment::Deref, Adjustment::Deref])
    );

    // `let r = &[1u8, 2]; f(&r);` with `f(&[u8])`.
    assert_eq!(
        taken_at(|w| {
            let declared = w.external(U8);
            let slice = w.external(Shape::Slice(declared));
            let parameter = w.external(Shape::Ref(SHARED, slice));
            let element = w.text(U8);
            let array = w.text(Shape::Array(2, element));
            let (lent_array, _) = w.lend(array, SHARED);
            let r = w.meets(lent_array);
            let (lent_r, _) = w.lend(r, SHARED);
            w.pass(lent_r, parameter)
        }),
        Ok(vec![Adjustment::Deref, Adjustment::Coerce(Coercion::Unsize)])
    );
}

#[test]
fn a_function_value_is_passed_only_to_one_of_the_same_closures() {
    let mut closures = Arena::<ClosureId, ()>::default();
    let (first, second) = (closures.push(()), closures.push(()));
    let passed = |argument: &'static [usize], given: &'static [usize]| {
        let ids = [first, second];
        taken_at(move |w| {
            let value = |of: &[usize]| Shape::<ClassId>::FnValue(of.iter().map(|index| ids[*index]).collect());
            let (argument, given) = (w.text(value(argument)), w.external(value(given)));
            w.pass(argument, given)
        })
    };
    assert_eq!(passed(&[0, 1], &[0, 1]), coerced(&[]));
    assert_eq!(passed(&[0], &[0, 1]), Err("NoChange".to_owned()));
    assert_eq!(passed(&[0, 1], &[0]), Err("NoChange".to_owned()));
}

// Deciding from above: a class nothing enters takes the one head, of those it
// will be held against, that every bound waiting on it would have it take.

#[test]
fn a_literal_takes_the_head_it_is_passed_to() {
    comes_to("u8", |w, lets| {
        let declared = w.external(U8);
        let zero = w.literal(&INTEGERS);
        let x = behind(w, zero, lets);
        w.pass(x, declared);
        zero
    });
}

#[test]
fn a_literal_takes_the_head_of_what_it_meets() {
    comes_to("u8", |w, lets| {
        let hole = w.undecided();
        let zero = w.literal(&INTEGERS);
        let zero_read = behind(w, zero, lets);
        let one = w.text(U8);
        w.enter(zero_read, hole);
        w.enter(one, hole);
        zero
    });
}

#[test]
fn a_class_held_against_two_heads_takes_neither_and_does_not_fall_back() {
    comes_to("refused: UndecidedClass", |w, lets| {
        let (first, second) = (w.external(Shape::Scalar(Scalar::I32)), w.external(U8));
        let zero = w.literal(&INTEGERS);
        let x = behind(w, zero, lets);
        w.pass(x, first);
        w.pass(x, second);
        zero
    });
}

#[test]
fn a_variable_takes_the_head_it_is_passed_to_and_a_part_named_on_it_is_placed_there() {
    comes_to("u8", |w, lets| {
        let declared = w.external(U8);
        let parameter = w.external(Shape::Option(declared));
        let a = w.undecided();
        let (payload, _) = w.project(a, PartKey::Payload(Name::new("Some")), Side::Upper);
        let a_read = behind(w, a, lets);
        w.pass(a_read, parameter);
        payload
    });
}

#[test]
fn a_target_takes_the_head_the_type_given_writes_at_the_target() {
    comes_to("&[u8]", |w, lets| {
        let declared = w.external(U8);
        let slice = w.external(Shape::Slice(declared));
        let parameter = w.external(Shape::Ref(SHARED, slice));
        let a = w.undecided();
        let (lent, _) = w.lend(a, SHARED);
        let lent = behind(w, lent, lets);
        w.pass(lent, parameter);
        lent
    });
}

#[test]
fn a_class_a_declared_result_enters_has_parts_of_its_own_compared_with_the_declared_ones() {
    comes_to("Option<u8>", |w, lets| {
        let declared_u8 = w.external(U8);
        let result = w.external(Shape::Option(declared_u8));
        behind(w, result, lets + 1)
    });

    let mut world = World::new();
    let declared_u8 = world.external(U8);
    let result = world.external(Shape::Option(declared_u8));
    let y = world.meets(result);
    world.settle(&Language);
    let Some(Shape::Option(payload)) = world.head(y) else {
        panic!("`y` is an option");
    };
    assert!(!world.same(payload, declared_u8));
}

// The contracts admit any law: a law that is not the language's runs through the
// same three operations.

/// A law of no language. What enters a class is the payload of that class, with a
/// deref recorded. A class with a payload named on it is an option, and any other
/// class is `u8`.
struct Odd;

impl Law<Live> for Odd {
    fn look(&self, _: &mut Scribe<'_, Live>, bound: &Bound<Live>, _: ItemId) -> Looked<Live> {
        Looked::Holds {
            pairs: vec![Pair::Part {
                part: bound.lower(),
                of: bound.upper(),
                key: PartKey::Payload(Name::new("Some")),
            }],
            adjustments: vec![Adjustment::Deref],
        }
    }

    fn same(&self, part: ClassId, declared: ClassId) -> Bound<Live> {
        Bound::Equal {
            lower: part,
            upper: declared,
        }
    }

    fn one(&self, _: &Classes<Live>) -> Vec<(ClassId, ClassId, ItemId)> {
        Vec::new()
    }

    fn place(&self, _: &Shape<ClassId>, _: &PartKey) -> Place {
        Place::Nowhere
    }

    fn admits(&self, _: &Shape<ClassId>, _: &Domain) -> bool {
        true
    }

    fn rungs(&self) -> usize {
        1
    }

    fn offer(&self, classes: &Classes<Live>, _: usize, class: ClassId) -> Offer<Live> {
        let State::Undecided(constraints) = &classes.get(classes.find(class)).state else {
            return Offer::Nothing;
        };
        let shape = if constraints.parts.is_empty() { U8 } else { Shape::Option(class) };
        Offer::Head {
            shape,
            by: BTreeSet::new(),
        }
    }

    fn end(&self, _: &Classes<Live>, _: ClassId) -> Vec<Refusal> {
        Vec::new()
    }
}

#[test]
fn a_law_of_no_language_runs_through_the_same_operations() {
    for order in 0..24 {
        let mut world = World::new();
        let (a, b, c) = (world.undecided(), world.undecided(), world.undecided());
        let a_into_b = world.enter(a, b);
        let b_into_c = world.enter(b, c);
        reorder(&mut world.open, order);
        world.settle(&Odd);
        assert_eq!(world.outcome.refusals, Vec::new());
        assert_eq!(world.ty(c), "Option<Option<u8>>", "in order {order}");
        let deref = vec![Adjustment::Deref];
        assert_eq!(
            world.outcome.adjustments,
            BTreeMap::from([(a_into_b, deref.clone()), (b_into_c, deref)])
        );
        assert!(world.same(a, a));
    }
}
