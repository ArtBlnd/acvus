//! initck and moveck over programs built from source: parsed, checked by
//! typeck, lowered, and taken through the required steps up to
//! `clone_strings`, as the checks see them.

use super::{Fixture, covers, span_of};
use crate::check::{Refusal, RefusalKind};
use crate::ty::{Mutability, TyKind, TypeArg};

/// ownership §4's example: on the one path that reaches the whole use,
/// every field of the binding's type is written before it.
#[test]
fn a_binding_built_field_by_field_is_used_whole_where_every_field_is_written() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let int = fx.int();
    let x = fx.object(&[("x", int)]);
    fx.declare("f", &[x], int);
    let refusals = fx.refusals("let a; if $c { a.x = 0; f(a) } else { 1 }");
    assert_eq!(refusals, []);
}

/// ownership §4: a field is read only where every path to the read has
/// written it.
#[test]
fn a_field_one_path_did_not_write_is_not_read() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let source = "let a; if $c { a.x = 0; 1 } else { 2 }; a.x";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a.x", 1)), "at the read: {span:?}");
}

/// ownership §4: the value is used whole only where every path to the use
/// has written every field of its type, which grew by a later store.
#[test]
fn a_whole_use_before_a_field_of_its_type_is_written_is_refused() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let source = "let a; if $c { a.x = 1; 1 } else { 2 }; a.y = true; a";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a", 3)), "at the whole use: {span:?}");
}

/// ownership §4: a binding declared without a value is not read before a
/// store writes it.
#[test]
fn a_binding_read_before_any_store_is_refused() {
    let fx = Fixture::new();
    let source = "let a; let b = a; a = 1; b";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::NotWritten, span }] = refusals[..] else {
        panic!("one NotWritten: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "a", 1)), "at the read: {span:?}");
}

/// ownership §2: a use moves a value that does not copy, and a later use
/// is refused, naming the move.
#[test]
fn a_value_that_does_not_copy_is_not_used_after_its_move() {
    let mut fx = Fixture::new();
    fx.vecs();
    let source = "let v = make(); consume(v); consume(v)";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::UsedAfterMove { moved }, span }] = refusals[..] else {
        panic!("one UsedAfterMove: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "v", 2)), "at the second use: {span:?}");
    assert!(covers(moved, span_of(source, "v", 1)), "naming the first use: {moved:?}");
}

/// ownership §6: a value a part of which moved out is not used whole
/// again.
#[test]
fn a_value_whose_part_moved_out_is_not_used_whole() {
    let mut fx = Fixture::new();
    fx.vecs();
    let source = "let p = { a: make(), b: 1, }; consume(p.a); p";
    let refusals = fx.refusals(source);
    let [Refusal { kind: RefusalKind::UsedAfterMove { moved }, span }] = refusals[..] else {
        panic!("one UsedAfterMove: {refusals:?}");
    };
    assert!(covers(span, span_of(source, "p", 2)), "at the whole use: {span:?}");
    assert!(covers(moved, span_of(source, "p.a", 0)), "naming the part's move: {moved:?}");
}

/// ownership §1-2: a value that copies clones by its copy, so every use
/// reads it.
#[test]
fn a_value_that_copies_is_used_again() {
    let fx = Fixture::new();
    let refusals = fx.refusals("let x = 1; let y = x; (x, y, x)");
    assert_eq!(refusals, []);
}

/// ownership §3: a `String`'s use that is not its last is a clone, so only
/// its last use moves it.
#[test]
fn a_string_used_twice_is_cloned_before_its_last_use() {
    let mut fx = Fixture::new();
    let string = fx.ty(TyKind::String);
    fx.declare("name", &[], string);
    let refusals = fx.refusals("let s = name(); (s, s)");
    assert_eq!(refusals, []);
}

/// ownership §5: a `&mut` does not copy, and each use reborrows it, so its
/// holder keeps it and a second use is no use after a move.
#[test]
fn a_mut_reference_used_twice_is_reborrowed_not_moved() {
    let mut fx = Fixture::new();
    let (vec, int) = (fx.vecs(), fx.int());
    let lent = fx.ty(TyKind::Ref(Mutability::Mut, TypeArg::Uniform(vec)));
    fx.declare("touch", &[lent], int);
    let refusals = fx.refusals("let v = make(); let r = &mut v; touch(r); touch(r)");
    assert_eq!(refusals, []);
}

/// ownership §4: a read no path reaches has every path to it writing what
/// it reads, so it is admitted.
#[test]
fn a_read_no_path_reaches_is_admitted() {
    let mut fx = Fixture::new();
    let bool_ty = fx.ty(TyKind::Bool);
    fx.input("c", bool_ty);
    let refusals = fx.refusals("let x; if $c { return 2; x } else { 1 }");
    assert_eq!(refusals, []);
}
