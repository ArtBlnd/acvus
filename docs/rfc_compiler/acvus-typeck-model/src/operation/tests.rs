//! The law, as programs: each test is a ruling of `TERMS.md` run.
//!
//! A program is built behind each count of `let`s from none to three and settled
//! in 24 orders of asking, and comes to one outcome in all of them: a class is
//! decided once, and only when nothing can still be merged into it.

use std::collections::{BTreeMap, BTreeSet};

use super::language::Language;
use super::law::{Heads, Law, Looked, Offer, Pair, Place, Scribe};
use super::testing::{EXCLUSIVE, SHARED, U8, World, reorder};
use crate::primitive::adjustment::{Adjustment, Adjustments};
use crate::primitive::constraint::{Bound, Constraint, ConstraintOf};
use crate::primitive::class::{Live, Side, State};
use crate::primitive::domain::Domain;
use crate::primitive::id::Arena;
use crate::primitive::adjustment::Subtyping;
use crate::primitive::refusal::{Cause, Refused};
use crate::primitive::ty::{PartKey, Scalar, Shape, Ty, VariantOf};
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

/// The refusals of a world in one order, to compare what two orders of asking
/// come to: a refusal names the same items whichever class is asked first.
fn refusals_of(world: &World) -> Vec<(Cause, BTreeSet<ItemId>, BTreeSet<ItemId>)> {
    let mut refusals: Vec<_> = world
        .outcome
        .refusals
        .iter()
        .map(|refusal| (refusal.cause, refusal.items.clone(), refusal.connections.clone()))
        .collect();
    refusals.sort();
    refusals
}

/// The one outcome `build` comes to, the value it marks being read behind each
/// count of `let`s and its classes asked in each order.
fn comes_to<F>(expected: &str, build: F)
where
    F: Fn(&mut World, usize) -> ClassId,
{
    for lets in 0..4 {
        let mut first = None;
        for order in 0..24 {
            let mut world = World::new();
            let observed = build(&mut world, lets);
            reorder(&mut world.open, order);
            world.settle(&Language::default());
            assert_eq!(world.outcome_of(observed), expected, "behind {lets} lets, in order {order}");
            let refusals = refusals_of(&world);
            let first = first.get_or_insert_with(|| refusals.clone());
            assert_eq!(&refusals, first, "what the refusals name behind {lets} lets, in order {order}");
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
    world.settle(&Language::default());
    let reborrow = vec![Adjustment::Subtype(Subtyping::Reborrow)];
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
    world.settle(&Language::default());
    assert_eq!(world.outcome.adjustments, Adjustments::from([(projection, vec![Adjustment::Deref])]));
}

#[test]
fn a_part_named_on_references_to_a_reference_is_read_through_each_and_derefs_once_for_each() {
    // `r.x` with `r: &&{x: u8}`, and with `r: &&&{x: u8}`.
    let build = |references: usize| {
        move |w: &mut World, lets: usize| {
            let x = w.text(U8);
            let o = object(w, &[("x", x)]);
            let mut r = behind(w, o, lets);
            for _ in 0..references {
                let (lent, _) = w.lend(r, SHARED);
                r = behind(w, lent, lets);
            }
            let r = w.meets(r);
            w.project(r, field("x"), Side::Upper)
        }
    };
    for references in 2..4 {
        comes_to("u8", |w, lets| build(references)(w, lets).0);
        for lets in 0..4 {
            for order in 0..24 {
                let mut world = World::new();
                let (_, projection) = build(references)(&mut world, lets);
                reorder(&mut world.open, order);
                world.settle(&Language::default());
                assert_eq!(
                    world.outcome.adjustments.get(&projection),
                    Some(&vec![Adjustment::Deref; references]),
                    "behind {lets} lets, in order {order}"
                );
            }
        }
    }
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

// A class nothing gives a head takes the head made from the parts named of it.

/// `let a; a.x = 1u8;`, the write made behind `lets` bindings of `a`: the class
/// of `a`, and the class written through.
fn written_field(world: &mut World, lets: usize) -> (ClassId, ClassId) {
    let a = world.undecided();
    let last = behind(world, a, lets);
    let (part, _) = world.project(last, field("x"), Side::Lower);
    let one = world.text(U8);
    world.enter(one, part);
    (a, last)
}

fn variant(of: VariantOf, name: &str) -> PartKey {
    PartKey::Payload(of, Name::new(name))
}

#[test]
fn a_class_nothing_gives_a_head_takes_the_head_the_parts_named_on_it_make() {
    // `let a; a.x`, read behind each count of `let`s: the class read is an
    // object, and the variable nothing was said of is `!`.
    let read = |observe_read: bool| {
        move |w: &mut World, lets: usize| {
            let a = w.undecided();
            let last = behind(w, a, lets + 1);
            w.project(last, field("x"), Side::Upper);
            if observe_read { last } else { a }
        }
    };
    comes_to("{x: !}", read(true));
    comes_to("!", read(false));
}

#[test]
fn a_field_written_into_a_class_is_a_field_of_it() {
    // `let a; a.x = 1u8;`
    comes_to("{x: u8}", |w, lets| written_field(w, lets).1);
    // `let a = {y: 1u8}; a.x = 2u8;`: an object that entered has the field
    // written besides its own.
    comes_to("{x: u8, y: u8}", |w, lets| {
        let y = w.text(U8);
        let a = object(w, &[("y", y)]);
        let a = behind(w, a, lets + 1);
        let (x, _) = w.project(a, field("x"), Side::Lower);
        let two = w.text(U8);
        w.enter(two, x);
        a
    });
    // `let a = @cfg; a.extra = 1;`: so has one a declared object entered.
    comes_to("{extra: i32, name: String}", |w, lets| {
        let name = w.external(Shape::String);
        let cfg = w.external(Shape::Object([(Name::new("name"), name)].into_iter().collect()));
        let a = w.meets(cfg);
        let a = behind(w, a, lets);
        let (extra, _) = w.project(a, field("extra"), Side::Lower);
        let one = w.literal(&INTEGERS);
        w.enter(one, extra);
        a
    });
}

#[test]
fn a_field_is_written_into_the_class_it_is_written_on_and_no_other() {
    // `let a; a.x = 1u8; let b = a; b.y = 0;`: `b` has both, and `a` its own.
    let build = |observe_b: bool| {
        move |w: &mut World, lets: usize| {
            let (a, _) = written_field(w, 0);
            let b = behind(w, a, lets + 1);
            let (y, _) = w.project(b, field("y"), Side::Lower);
            let zero = w.literal(&INTEGERS);
            w.enter(zero, y);
            if observe_b { b } else { a }
        }
    };
    comes_to("{x: u8, y: i32}", build(true));
    comes_to("{x: u8}", build(false));
}

#[test]
fn a_field_read_is_no_field_of_an_object_that_entered() {
    // `let a = {y: 1u8}; a.x`
    comes_to("refused: NotHeld", |w, lets| {
        let y = w.text(U8);
        let a = object(w, &[("y", y)]);
        let a = behind(w, a, lets + 1);
        w.project(a, field("x"), Side::Upper).0
    });
}

#[test]
fn a_field_written_through_a_reference_is_no_field_of_what_it_refers_to() {
    // `let o = {y: 1u8}; let r = &mut o; r.x = 2u8;`, and the same with `@cfg`.
    let build = |declared: bool| {
        move |w: &mut World, lets: usize| {
            let o = if declared {
                let y = w.external(U8);
                w.external(Shape::Object([(Name::new("y"), y)].into_iter().collect()))
            } else {
                let y = w.text(U8);
                let o = object(w, &[("y", y)]);
                w.meets(o)
            };
            let (lent, _) = w.lend(o, EXCLUSIVE);
            let r = behind(w, lent, lets);
            let (x, _) = w.project(r, field("x"), Side::Lower);
            let two = w.text(U8);
            w.enter(two, x);
            r
        }
    };
    comes_to("refused: NotHeld", build(false));
    comes_to("refused: NotHeld", build(true));
}

#[test]
fn a_part_written_that_says_no_whole_head_writes_none() {
    // `let t; t.0 = 1u8;`, `let x; x[i] = 1u8;`, `let r; *r = 1u8;`
    for key in [PartKey::Component(0), PartKey::Elem, PartKey::Target] {
        comes_to("refused: NotHeld", |w, lets| {
            let a = w.undecided();
            let a = behind(w, a, lets);
            let (part, _) = w.project(a, key.clone(), Side::Lower);
            let one = w.text(U8);
            w.enter(one, part);
            a
        });
    }
    // `let x = [return; 3]; x[0] = 1u8;`: what is written goes into the place.
    comes_to("[u8; 3]", |w, lets| {
        let element = w.text(Shape::Never);
        let x = w.text(Shape::Array(3, element));
        let x = behind(w, x, lets);
        let (part, _) = w.project(x, PartKey::Elem, Side::Lower);
        let one = w.text(U8);
        w.enter(one, part);
        x
    });
}

#[test]
fn never_says_nothing_so_a_class_it_enters_takes_the_head_its_parts_make() {
    // `let b = return; b.x = 1u8;`
    comes_to("{x: u8}", |w, lets| {
        let b = w.text(Shape::Never);
        let last = behind(w, b, lets);
        let (part, _) = w.project(last, field("x"), Side::Lower);
        let one = w.text(U8);
        w.enter(one, part);
        last
    });
}

#[test]
fn a_payload_named_makes_the_type_its_variant_is_of() {
    // `let A::V(x) = a`, and `let Some(x) = a`.
    let build = |key: PartKey| {
        move |w: &mut World, lets: usize| {
            let a = w.undecided();
            let last = behind(w, a, lets);
            w.project(last, key.clone(), Side::Upper);
            last
        }
    };
    comes_to("A{V(!)}", build(variant(VariantOf::Enum(Name::new("A")), "V")));
    comes_to("Option<!>", build(variant(VariantOf::Option, "Some")));
}

#[test]
fn a_payload_is_held_only_by_the_type_its_variant_is_of() {
    // `let Some(x) = a` with `a` an `A{Some(u8)}`.
    comes_to("refused: NotHeld", |w, lets| {
        let payload = w.text(U8);
        let a = w.text(Shape::Enum(Name::new("A"), [(Name::new("Some"), payload)].into_iter().collect()));
        let a = behind(w, a, lets + 1);
        w.project(a, variant(VariantOf::Option, "Some"), Side::Upper).0
    });
}

#[test]
fn parts_whose_names_do_not_say_one_head_make_none() {
    // `let a; a.0`, `let a; *a`, and `let a; a.x; let A::V(y) = a`.
    let build = |keys: Vec<PartKey>| {
        move |w: &mut World, lets: usize| {
            let a = w.undecided();
            let last = behind(w, a, lets);
            for key in &keys {
                w.project(last, key.clone(), Side::Upper);
            }
            a
        }
    };
    comes_to("refused: NotHeld", build(vec![PartKey::Component(0)]));
    comes_to("refused: NotHeld", build(vec![PartKey::Target]));
    comes_to("refused: NotHeld", build(vec![field("x"), variant(VariantOf::Enum(Name::new("A")), "V")]));
}

#[test]
fn a_head_from_above_comes_before_the_head_parts_read_make_and_after_what_is_written() {
    // `let a; a.x; f(a)` and `let a; a.x = 1u8; f(a)`, with `f(&mut {x: u8})`.
    let build = |side: Side| {
        move |w: &mut World, lets: usize| {
            let a = w.undecided();
            let (x, _) = w.project(a, field("x"), side);
            if side == Side::Lower {
                let one = w.text(U8);
                w.enter(one, x);
            }
            let last = behind(w, a, lets);
            let declared = w.external(U8);
            let object = w.external(Shape::Object([(Name::new("x"), declared)].into_iter().collect()));
            let parameter = w.external(Shape::Ref(EXCLUSIVE, object));
            w.pass(last, parameter);
            a
        }
    };
    comes_to("&mut {x: u8}", build(Side::Upper));
    // What is written gives the class its head as what enters does, and an
    // object is not passed as a reference to one.
    comes_to("refused: NoChange", build(Side::Lower));
}

#[test]
fn a_class_asked_two_heads_takes_none_from_its_parts() {
    // `let a; a.x; f(a); g(a)` with `f(u8)` and `g(u16)`.
    comes_to("refused: UndecidedClass", |w, lets| {
        let a = w.undecided();
        w.project(a, field("x"), Side::Upper);
        let last = behind(w, a, lets);
        let (eight, sixteen) = (w.external(U8), w.external(U16));
        w.pass(last, eight);
        w.pass(last, sixteen);
        a
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
    world.settle(&Language::default());
    if !world.outcome.refusals.is_empty() {
        let causes: BTreeSet<String> = world.outcome.refusals.iter().map(|refusal| format!("{:?}", refusal.cause)).collect();
        return Err(causes.into_iter().collect::<Vec<_>>().join(", "));
    }
    let taken = world.outcome.adjustments.get(&at).cloned().unwrap_or_default();
    let elsewhere = world.outcome.adjustments.keys().filter(|item| **item != at).count();
    assert_eq!(elsewhere, 0, "an adjustment is recorded at the boundary alone");
    Ok(taken)
}

fn subtyped(taken: &[Subtyping]) -> Result<Vec<Adjustment>, String> {
    Ok(taken.iter().copied().map(Adjustment::Subtype).collect())
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
        subtyped(&[Subtyping::Never])
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
    assert_eq!(passed(EXCLUSIVE, SHARED), subtyped(&[Subtyping::Reborrow]));
    assert_eq!(passed(SHARED, SHARED), subtyped(&[]));
    assert_eq!(passed(EXCLUSIVE, EXCLUSIVE), subtyped(&[]));
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
    assert_eq!(passed(&["x", "y"], &["x"], false), subtyped(&[Subtyping::DropFields]));
    assert_eq!(passed(&["x"], &["x"], false), subtyped(&[]));
    assert_eq!(passed(&["x"], &["x", "y"], false), Err("NoChange".to_owned()));
    assert_eq!(passed(&["x", "y"], &["x"], true), Err("NoChange".to_owned()));
    assert_eq!(passed(&["x"], &["x"], true), subtyped(&[]));
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
    assert_eq!(passed(SHARED, SHARED), subtyped(&[Subtyping::Unsize]));
    assert_eq!(passed(EXCLUSIVE, SHARED), subtyped(&[Subtyping::Unsize, Subtyping::Reborrow]));
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
        Ok(vec![Adjustment::Deref, Adjustment::Subtype(Subtyping::Reborrow)])
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
        Ok(vec![Adjustment::Deref, Adjustment::Subtype(Subtyping::Unsize)])
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
    assert_eq!(passed(&[0, 1], &[0, 1]), subtyped(&[]));
    assert_eq!(passed(&[0], &[0, 1]), Err("NoChange".to_owned()));
    assert_eq!(passed(&[0, 1], &[0]), Err("NoChange".to_owned()));
}

// Deciding from above: a class nothing enters takes the one head, of those it
// will be held against, that every constraint waiting on it would have it take.

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
        let (payload, _) = w.project(a, PartKey::Payload(VariantOf::Option, Name::new("Some")), Side::Upper);
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
    world.settle(&Language::default());
    let Some(Shape::Option(payload)) = world.head(y) else {
        panic!("`y` is an option");
    };
    assert!(!world.same(payload, declared_u8));
}

// Declared functions: a table, and one rule. An argument goes through a row
// `P -> R` by being passed to `P` and what the row gives being passed on.

fn hole() -> Ty {
    Ty::Hole(Name::new("T"))
}

fn node(shape: Shape<Ty>) -> Ty {
    Ty::Node(Box::new(shape))
}

fn vec_of(element: Ty) -> Ty {
    node(Shape::Struct(Name::new("Vec"), vec![element]))
}

fn named(name: &str) -> Ty {
    node(Shape::Struct(Name::new(name), Vec::new()))
}

/// `as_slice: &Vec<T> -> &[T]`, and the same for `&mut`.
fn as_slice(mutability: crate::primitive::ty::Mutability) -> (&'static str, Ty, Ty) {
    let name = if mutability == SHARED { "as_slice" } else { "as_slice_mut" };
    (
        name,
        node(Shape::Ref(mutability, vec_of(hole()))),
        node(Shape::Ref(mutability, node(Shape::Slice(hole())))),
    )
}

/// The language's law with `rows` declared.
fn law_with(world: &mut World, rows: &[(&'static str, Ty, Ty)]) -> Language {
    let mut law = Language::default();
    for (name, parameter, result) in rows {
        let by = world.items.push("declared");
        law.functions.declare(Name::new(*name), std::slice::from_ref(parameter), result, &world.store.scopes, by);
    }
    law
}

/// As [`comes_to`], by the law with `rows` declared.
fn comes_to_with<F>(rows: &[(&'static str, Ty, Ty)], expected: &str, build: F)
where
    F: Fn(&mut World, usize) -> ClassId,
{
    for lets in 0..4 {
        let mut first = None;
        for order in 0..24 {
            let mut world = World::new();
            let law = law_with(&mut world, rows);
            let observed = build(&mut world, lets);
            reorder(&mut world.open, order);
            world.settle(&law);
            assert_eq!(world.outcome_of(observed), expected, "behind {lets} lets, in order {order}");
            let refusals = refusals_of(&world);
            let first = first.get_or_insert_with(|| refusals.clone());
            assert_eq!(&refusals, first, "what the refusals name behind {lets} lets, in order {order}");
        }
    }
}

/// As [`taken_at`], by the law with `rows` declared.
fn taken_with<F>(rows: &[(&'static str, Ty, Ty)], build: F) -> Result<Vec<Adjustment>, String>
where
    F: Fn(&mut World) -> ItemId,
{
    let mut world = World::new();
    let law = law_with(&mut world, rows);
    let at = build(&mut world);
    world.settle(&law);
    if !world.outcome.refusals.is_empty() {
        let causes: BTreeSet<String> = world.outcome.refusals.iter().map(|refusal| format!("{:?}", refusal.cause)).collect();
        return Err(causes.into_iter().collect::<Vec<_>>().join(", "));
    }
    Ok(world.outcome.adjustments.get(&at).cloned().unwrap_or_default())
}

fn called(name: &str) -> Adjustment {
    Adjustment::Row(Name::new(name))
}

fn vec_class(world: &mut World, element: ClassId) -> ClassId {
    world.text(Shape::Struct(Name::new("Vec"), vec![element]))
}

/// `f(&[u8])`.
fn reference_to_slice(world: &mut World) -> ClassId {
    let declared = world.external(U8);
    let slice = world.external(Shape::Slice(declared));
    world.external(Shape::Ref(SHARED, slice))
}

#[test]
fn an_argument_goes_through_a_row_and_a_hole_of_the_row_connects_both_its_sides() {
    // `let v = new(); sum(&v);` with `new() -> Vec<T>`, `sum(&[u8])`: the element
    // of `v` takes `u8` through the row.
    comes_to_with(&[as_slice(SHARED)], "Vec<u8>", |w, lets| {
        let parameter = reference_to_slice(w);
        let element = w.undecided();
        let result = w.external(Shape::Struct(Name::new("Vec"), vec![element]));
        let v = behind(w, result, lets + 1);
        let (lent, _) = w.lend(v, SHARED);
        w.pass(lent, parameter);
        v
    });
}

#[test]
fn what_the_language_gives_is_taken_on_the_way_into_a_row() {
    let passed = |of_inner: Option<crate::primitive::ty::Mutability>, of_outer| {
        taken_with(&[as_slice(SHARED)], move |w| {
            let parameter = reference_to_slice(w);
            let element = w.text(U8);
            let v = vec_class(w, element);
            let (mut lent, _) = w.lend(v, of_inner.unwrap_or(of_outer));
            if of_inner.is_some() {
                let r = w.meets(lent);
                lent = w.lend(r, of_outer).0;
            }
            w.pass(lent, parameter)
        })
    };
    assert_eq!(passed(None, SHARED), Ok(vec![called("as_slice")]));
    assert_eq!(
        passed(None, EXCLUSIVE),
        Ok(vec![Adjustment::Subtype(Subtyping::Reborrow), called("as_slice")])
    );
    assert_eq!(passed(Some(SHARED), SHARED), Ok(vec![Adjustment::Deref, called("as_slice")]));
}

#[test]
fn what_a_row_gives_goes_through_another_row() {
    let rows = [("a_to_b", named("A"), named("B")), ("b_to_c", named("B"), named("C"))];
    let taken = taken_with(&rows, |w| {
        let parameter = w.external(Shape::Struct(Name::new("C"), Vec::new()));
        let a = w.text(Shape::Struct(Name::new("A"), Vec::new()));
        w.pass(a, parameter)
    });
    assert_eq!(taken, Ok(vec![called("a_to_b"), called("b_to_c")]));
}

#[test]
fn exactly_one_way_through_rows_must_go() {
    let two = [("first", named("A"), named("B")), ("second", named("A"), named("B"))];
    let through = |rows: &'static [(&'static str, Ty, Ty)]| {
        taken_with(rows, |w| {
            let parameter = w.external(Shape::Struct(Name::new("B"), Vec::new()));
            let a = w.text(Shape::Struct(Name::new("A"), Vec::new()));
            w.pass(a, parameter)
        })
    };
    assert_eq!(through(Box::leak(Box::new(two))), Err("NoChange".to_owned()));
    assert_eq!(through(&[]), Err("NoChange".to_owned()));
}

#[test]
fn a_row_is_looked_for_only_where_the_language_gives_nothing() {
    // `&mut [u8; 2]` passed to `&[u8]` goes by the language, though a row would go.
    let rows = [(
        "array_as_slice",
        node(Shape::Ref(SHARED, node(Shape::Array(2, hole())))),
        node(Shape::Ref(SHARED, node(Shape::Slice(hole())))),
    )];
    let taken = taken_with(&rows, |w| {
        let parameter = reference_to_slice(w);
        let element = w.text(U8);
        let array = w.text(Shape::Array(2, element));
        let (lent, _) = w.lend(array, SHARED);
        w.pass(lent, parameter)
    });
    assert_eq!(taken, subtyped(&[Subtyping::Unsize]));
}

#[test]
fn each_passing_makes_the_signature_of_a_row_anew() {
    let mut world = World::new();
    let law = law_with(&mut world, &[as_slice(SHARED)]);
    for head in [U8, U16] {
        let declared = world.external(head.clone());
        let slice = world.external(Shape::Slice(declared));
        let parameter = world.external(Shape::Ref(SHARED, slice));
        let element = world.text(head);
        let v = vec_class(&mut world, element);
        let (lent, _) = world.lend(v, SHARED);
        world.pass(lent, parameter);
    }
    world.settle(&law);
    assert_eq!(world.outcome.refusals, Vec::new());
}

#[test]
fn what_lies_below_a_row_is_compared_and_not_changed() {
    // `Vec<u16>` passed to `&[u8]`: the row goes, and its hole is held against
    // both `u16` and `u8`, which no one head is.
    comes_to_with(&[as_slice(SHARED)], "refused: UndecidedClass", |w, lets| {
        let parameter = reference_to_slice(w);
        let sixteen = w.external(U16);
        let result = w.external(Shape::Struct(Name::new("Vec"), vec![sixteen]));
        let v = behind(w, result, lets + 1);
        let (lent, _) = w.lend(v, SHARED);
        w.pass(lent, parameter);
        v
    });
}

#[test]
fn a_hole_a_passing_not_told_yet_may_reach_is_not_decided() {
    // `take((x, 1u16), return);` with `take(Vec<U>, U)` and the row
    // `(u8, T) -> Vec<T>`. The row writes `u8` at the first component, so it is
    // told only once `x` has a head, which comes later the more `let`s `x` is read
    // behind. Until then `U` is not decided from the `!` that enters it alone.
    let row = (
        "second",
        node(Shape::Tuple(vec![node(Shape::Scalar(Scalar::U8)), hole()])),
        vec_of(hole()),
    );
    comes_to_with(&[row], "u16", |w, lets| {
        let hole = w.undecided();
        let parameter = w.external(Shape::Struct(Name::new("Vec"), vec![hole]));
        let one = w.text(U8);
        let x = behind(w, one, lets + 2);
        let sixteen = w.text(U16);
        let pair = w.text(Shape::Tuple(vec![x, sixteen]));
        let returns = w.text(Shape::Never);
        w.pass(pair, parameter);
        w.enter(returns, hole);
        hole
    });
}

#[test]
fn no_row_is_taken_before_every_row_is_told_of() {
    // `take((x, 1u16));` with `take(Vec<U>)` and two rows: `(A, T) -> Vec<T>`,
    // which any pair goes to, and `(u8, T) -> Vec<T>`, which is told only once `x`
    // has a head. Where `x` is `u8` both go, and the program is refused, however
    // late `x` comes to its head.
    let rows = || {
        [
            (
                "any",
                node(Shape::Tuple(vec![Ty::Hole(Name::new("A")), hole()])),
                vec_of(hole()),
            ),
            (
                "of_u8",
                node(Shape::Tuple(vec![node(Shape::Scalar(Scalar::U8)), hole()])),
                vec_of(hole()),
            ),
        ]
    };
    let build = |first: Shape<ClassId>| {
        move |w: &mut World, lets: usize| {
            let hole = w.undecided();
            let parameter = w.external(Shape::Struct(Name::new("Vec"), vec![hole]));
            let x = w.text(first.clone());
            let x = behind(w, x, lets + 2);
            let sixteen = w.text(U16);
            let pair = w.text(Shape::Tuple(vec![x, sixteen]));
            w.pass(pair, parameter);
            hole
        }
    };
    comes_to_with(&rows(), "refused: NoChange", build(U8));
    comes_to_with(&rows(), "u16", build(CHAR));
}

// `e as _` decides alone: the type `e` comes to by itself is a declared type to
// what is outside.

#[test]
fn a_literal_as_underscore_falls_back_and_takes_no_head_from_what_it_is_passed_to() {
    // `let x = 0 as _; f(x);` with `f(u8)`.
    let build = |alone: bool| {
        move |w: &mut World, lets: usize| {
            let zero = w.literal(&INTEGERS);
            let value = if alone { w.alone(zero) } else { zero };
            let x = behind(w, value, lets);
            let parameter = w.external(U8);
            w.pass(x, parameter);
            x
        }
    };
    comes_to("u8", build(false));
    comes_to("refused: NoChange", build(true));
}

#[test]
fn what_as_underscore_writes_is_decided_at_every_depth() {
    // `Some(0) as _`, and `None as _`.
    comes_to("Option<i32>", |w, lets| {
        let zero = w.literal(&INTEGERS);
        let option = some(w, zero);
        let alone = w.alone(option);
        behind(w, alone, lets)
    });
    comes_to("Option<!>", |w, lets| {
        let payload = w.undecided();
        let none = some(w, payload);
        let alone = w.alone(none);
        behind(w, alone, lets)
    });
}

#[test]
fn a_part_as_underscore_writes_is_compared_with_what_it_meets_and_not_merged() {
    // `if c { None } else { Some(1u8) }`, and the same with `None as _`.
    let build = |alone: bool| {
        move |w: &mut World, lets: usize| {
            let payload = w.undecided();
            let none = some(w, payload);
            let none = if alone { w.alone(none) } else { none };
            let none = behind(w, none, lets);
            let one = w.text(U8);
            let one = some(w, one);
            either(w, none, one)
        }
    };
    comes_to("Option<u8>", build(false));
    comes_to("refused: Mismatch", build(true));
}

#[test]
fn as_underscore_waits_for_a_class_it_names_and_does_not_decide_it() {
    // `let x = 0; let y = x as _; f(x);` with `f(u8)`: `x` is `u8`, and so is `y`.
    let build = |observe_y: bool| {
        move |w: &mut World, lets: usize| {
            let zero = w.literal(&INTEGERS);
            let x = w.meets(zero);
            let y = w.alone(x);
            let y = behind(w, y, lets);
            let parameter = w.external(U8);
            w.pass(x, parameter);
            if observe_y { y } else { x }
        }
    };
    comes_to("u8", build(false));
    comes_to("u8", build(true));
}

#[test]
fn as_underscore_of_what_has_no_type_gives_none() {
    // `let x; x = Some(x); let y = x as _;`
    comes_to("refused: Occurs, UndecidedClass", |w, lets| {
        let x = w.undecided();
        let option = some(w, x);
        w.enter(option, x);
        let y = w.alone(x);
        behind(w, y, lets)
    });
}

// `e?`: its value is the payload of `Ok`, and what it exits with puts an error
// into the body's result.

/// A type of the name `name`.
fn of_name(world: &mut World, name: &str) -> ClassId {
    world.text(Shape::Struct(Name::new(name), Vec::new()))
}

/// An expression of the type `Result<ok, error>`.
fn result_of(world: &mut World, ok: ClassId, error: ClassId) -> ClassId {
    world.text(Shape::Result(ok, error))
}

/// The tail `Ok(value)` of the body whose result is `body`.
fn tail_ok(world: &mut World, value: ClassId, body: ClassId) {
    let error = world.undecided();
    let tail = result_of(world, value, error);
    world.enter(tail, body);
}

/// `let a = e?; Ok(a)`, with `e: Result<u8, error>` read behind `lets` bindings:
/// the result of the body.
fn exits_with<F>(world: &mut World, lets: usize, error: F) -> ClassId
where
    F: Fn(&mut World) -> ClassId,
{
    let body = world.undecided();
    let ok = world.text(U8);
    let error = error(world);
    let e = result_of(world, ok, error);
    let e = behind(world, e, lets);
    let a = world.question(e, body);
    tail_ok(world, a, body);
    body
}

#[test]
fn an_exit_with_a_named_type_puts_a_variant_of_that_name_into_the_error() {
    comes_to("Result<u8, Error{ParseError(ParseError)}>", |w, lets| {
        exits_with(w, lets, |w| of_name(w, "ParseError"))
    });
}

#[test]
fn exits_with_named_types_unite_their_variants() {
    // `let a = parse()?; read()?; Ok(a)`
    comes_to("Result<u8, Error{IoError(IoError), ParseError(ParseError)}>", |w, lets| {
        let body = exits_with(w, lets, |w| of_name(w, "ParseError"));
        let ok = w.text(U16);
        let error = of_name(w, "IoError");
        let read = result_of(w, ok, error);
        w.question(read, body);
        body
    });
}

#[test]
fn an_exit_with_an_error_puts_its_variants_among_the_others() {
    comes_to("Result<u8, Error{A(A), B(B)}>", |w, lets| {
        let body = exits_with(w, lets, |w| {
            let a = of_name(w, "A");
            w.text(Shape::Error([(Name::new("A"), a)].into_iter().collect()))
        });
        let ok = w.text(U16);
        let error = of_name(w, "B");
        let other = result_of(w, ok, error);
        w.question(other, body);
        body
    });
}

#[test]
fn an_exit_with_a_type_of_no_name_puts_an_error_of_it() {
    comes_to("Result<u8, Error<u16>>", |w, lets| exits_with(w, lets, |w| w.text(U16)));
}

#[test]
fn an_error_of_a_type_and_an_error_with_variants_do_not_join() {
    // `let a = e?; parse()?; Ok(a)` with `e: Result<u8, u16>`.
    comes_to("refused: Mismatch, UndecidedClass", |w, lets| {
        let body = exits_with(w, lets, |w| w.text(U16));
        let ok = w.text(U16);
        let error = of_name(w, "ParseError");
        let parse = result_of(w, ok, error);
        w.question(parse, body);
        body
    });
}

#[test]
fn an_exit_with_never_puts_nothing_into_the_error() {
    comes_to("Result<u8, !>", |w, lets| exits_with(w, lets, |w| w.text(Shape::Never)));
}

#[test]
fn an_option_has_no_exit() {
    // `let a = o?; Ok(a)` with `o: Option<u8>`.
    comes_to("refused: NotHeld", |w, lets| {
        let body = w.undecided();
        let payload = w.text(U8);
        let o = some(w, payload);
        let o = behind(w, o, lets);
        let a = w.question(o, body);
        tail_ok(w, a, body);
        body
    });
}

#[test]
fn a_class_only_an_exit_names_parts_of_is_a_result() {
    // `let e; let a = e?; Ok(a)`
    comes_to("Result<!, !>", |w, lets| {
        let body = w.undecided();
        let e = w.undecided();
        let e = behind(w, e, lets);
        let a = w.question(e, body);
        tail_ok(w, a, body);
        body
    });
}

// A refusal names what made the classes it is of meet.

/// The item recorded last.
fn last_item(world: &World) -> ItemId {
    world.items.iter().last().expect("an item was recorded").0
}

/// The connections the one refusal by `cause` names, the same in each order of
/// asking, with the items `build` gives.
fn connections_of<F>(cause: Cause, build: F) -> (BTreeSet<ItemId>, BTreeSet<ItemId>)
where
    F: Fn(&mut World) -> Vec<ItemId>,
{
    let mut named = None;
    let mut given = BTreeSet::new();
    for order in 0..24 {
        let mut world = World::new();
        given = build(&mut world).into_iter().collect();
        reorder(&mut world.open, order);
        world.settle(&Language::default());
        let mut refusals = world.outcome.refusals.iter().filter(|refusal| refusal.cause == cause);
        let refusal = refusals.next().expect("the program is refused by the cause");
        assert_eq!(refusals.next(), None, "one refusal by the cause, in order {order}");
        let named = named.get_or_insert_with(|| refusal.connections.clone());
        assert_eq!(&refusal.connections, named, "in order {order}");
    }
    (named.expect("an order was run"), given)
}

#[test]
fn parts_that_do_not_join_are_refused_naming_each_entering_that_made_them_meet() {
    // `let p = Some(1u8); let q = Some(2u16); let r = if c { p } else { q }; let s = r;`
    // Each of the five enterings made the two payloads one class with another.
    let (named, enterings) = connections_of(Cause::Mismatch, |w| {
        let one = w.text(U8);
        let p = some(w, one);
        let written_into_p = last_item(w);
        let two = w.text(U16);
        let q = some(w, two);
        let written_into_q = last_item(w);
        let r = w.undecided();
        let s = w.undecided();
        vec![written_into_p, written_into_q, w.enter(p, r), w.enter(q, r), w.enter(r, s)]
    });
    assert_eq!(named, enterings);
}

#[test]
fn arguments_that_do_not_join_in_a_hole_are_refused_naming_each_passing() {
    // `h(a, b)` with `h(T, T)`, `a: u8`, `b: u16`.
    let (named, passings) = connections_of(Cause::Mismatch, |w| {
        let hole = w.undecided();
        let a = w.text(U8);
        let b = w.text(U16);
        vec![w.pass(a, hole), w.pass(b, hole)]
    });
    assert_eq!(named, passings);
}

#[test]
fn parts_made_one_because_what_holds_them_is_one_are_refused_naming_what_made_that_one() {
    // `h(Some(1u8), Some(2u16))` with `h(T, T)`: the payloads meet because the
    // two options meet in the hole.
    let (named, passings) = connections_of(Cause::Mismatch, |w| {
        let hole = w.undecided();
        let one = w.text(U8);
        let p = some(w, one);
        let two = w.text(U16);
        let q = some(w, two);
        vec![w.pass(p, hole), w.pass(q, hole)]
    });
    assert!(named.is_superset(&passings), "{named:?} names {passings:?}");
}

// The contracts admit any law: a law that is not the language's runs through the
// same three operations.

/// A law of no language. What enters a class is the payload of that class, with a
/// deref recorded. A class with a payload named on it is an option, and any other
/// class is `u8`.
struct Odd;

impl Law<Live> for Odd {
    fn look(&self, _: &mut Scribe<'_, Live>, constraint: &Constraint<Live>, _: ItemId) -> Looked<Live> {
        Looked::Holds {
            pairs: vec![Pair::Part {
                part: constraint.lower(),
                of: constraint.upper(),
                key: PartKey::Payload(VariantOf::Option, Name::new("Some")),
            }],
            adjustments: vec![Adjustment::Deref],
            constraints: Vec::new(),
        }
    }

    fn same(&self, part: ClassId, declared: ClassId) -> Constraint<Live> {
        Bound::Equal {
            lower: part,
            upper: declared,
        }
        .into()
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

    fn heads(&self, classes: &Classes<Live>, asked: &[ClassId], held: &dyn Fn(ClassId) -> bool) -> Heads<Live> {
        let only = |class: ClassId| {
            let State::Undecided(undecided) = &classes.get(classes.find(class)).state else {
                return Offer::Nothing;
            };
            let shape = if undecided.parts.is_empty() { U8 } else { Shape::Option(class) };
            Offer::Head {
                shape,
                by: BTreeSet::new(),
            }
        };
        let ladder: [&dyn Fn(ClassId) -> Offer<Live>; 1] = [&only];
        Heads::of_ladder(ladder, asked, held)
    }

    fn end(&self, _: &Classes<Live>, _: ClassId) -> Vec<Refused<Live>> {
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
