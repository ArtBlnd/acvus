//! Intent tests for RFC-0051: a `match` is one dispatch, and it is
//! exhaustive.

use acvus_mir::ty::Ty;
use acvus_mir_test::*;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn flag(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([(i.intern("c"), Ty::Bool)])
}

/// `@e` holds an `E{A(i64), B(i64)}` a host put there: it comes from
/// outside, and its type names the variant set all the same.
fn flag_and_outside_enum(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([
        (i.intern("c"), Ty::Bool),
        (
            i.intern("e"),
            Ty::Enum {
                name: i.intern("E"),
                variants: FxHashMap::from_iter([
                    (i.intern("A"), Some(Box::new(Ty::I64))),
                    (i.intern("B"), Some(Box::new(Ty::I64))),
                ]),
            },
        ),
    ])
}

fn flag_and_option(i: &Interner) -> FxHashMap<acvus_utils::Astr, Ty> {
    FxHashMap::from_iter([
        (i.intern("c"), Ty::Bool),
        (i.intern("opt"), Ty::Option(Box::new(Ty::I64))),
    ])
}

/// `let e = if @c { E::A(1) } else { E::B(2) };` -- two constructions joined,
/// which the solver unifies into `E{A(i64), B(i64)}` on `e`.
const TWO_CONSTRUCTIONS: &str = "let e = if @c { E::A(1) } else { E::B(2) }; ";

#[test]
fn a_match_over_an_enum_built_in_this_body_needs_no_catch_all() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}match e {{ E::A(v) => v, E::B(v) => v }}");
    let raw = compile_script_raw(&i, &source, &flag(&i)).unwrap();
    assert!(
        raw.contains("switch ") && raw.contains("A -> ") && raw.contains("B -> "),
        "the lowering writes one dispatch: {raw}"
    );
    assert!(
        !raw.contains("_ -> "),
        "no catch-all was written, so the Switch has no default: {raw}"
    );
    let optimized = compile_script_optimized(&i, &source, &flag(&i)).unwrap();
    assert!(!optimized.contains("switch "), "{optimized}");
    assert!(
        !optimized.contains("variant "),
        "an enum this body never lets out is never built: {optimized}"
    );
    assert_eq!(
        optimized.matches("is A").count() + optimized.matches("is B").count(),
        0,
        "no instruction reads a tag out of a value that does not exist: {optimized}"
    );
    assert_eq!(
        optimized.matches("jump_if").count(),
        1,
        "the branch that chooses the constructor is the one that chooses the arm: {optimized}"
    );
}

#[test]
fn a_match_over_an_enum_built_in_this_body_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}match e {{ E::A(v) => v }}");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert!(
        err.contains("non-exhaustive match") && err.contains("E::B"),
        "{err}"
    );
}

#[test]
fn an_arm_naming_a_variant_the_scrutinee_cannot_hold_is_refused() {
    let i = Interner::new();
    let source =
        format!("{TWO_CONSTRUCTIONS}match e {{ E::A(v) => v, E::B(v) => v, E::C(v) => v }}");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert!(
        err.contains("unreachable pattern: `E::C(_)` is not a variant of `E{A(i64), B(i64)}`"),
        "{err}"
    );
}

#[test]
fn a_match_on_a_value_from_outside_is_closed_by_its_type() {
    let i = Interner::new();
    let source = "match @e { E::A(v) => v, E::B(v) => v }";
    compile_script_ir(&i, source, &flag_and_outside_enum(&i)).unwrap();
}

#[test]
fn a_match_on_a_value_from_outside_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = "match @e { E::A(v) => v }";
    let err = compile_script_ir(&i, source, &flag_and_outside_enum(&i)).unwrap_err();
    assert!(
        err.contains("non-exhaustive match") && err.contains("E::B"),
        "{err}"
    );
}

#[test]
fn a_match_on_a_value_from_outside_with_a_catch_all_is_accepted() {
    let i = Interner::new();
    let source = "match @e { E::A(v) => v, _ => 0 }";
    let raw = compile_script_raw(&i, source, &flag_and_outside_enum(&i)).unwrap();
    assert!(raw.contains("_ -> "), "the catch-all is the default: {raw}");
}

#[test]
fn a_match_on_an_option_needs_no_catch_all() {
    let i = Interner::new();
    let source = "match @opt { Some(v) => v, None => 0 }";
    compile_script_ir(&i, source, &flag_and_option(&i)).unwrap();
}

#[test]
fn a_match_on_an_option_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = "match @opt { Some(v) => v }";
    let err = compile_script_ir(&i, source, &flag_and_option(&i)).unwrap_err();
    assert!(
        err.contains("`Option` has 2 variants and the arms cover 1"),
        "{err}"
    );
}

#[test]
fn a_nested_refutable_pattern_is_not_one_dispatch_and_needs_a_catch_all() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}match Some(e) {{ Some(E::A(v)) => v, None => 0 }}");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert!(
        err.contains("not one dispatch over a tag") && err.contains("add a `_` arm"),
        "{err}"
    );

    let with_catch_all =
        format!("{TWO_CONSTRUCTIONS}match Some(e) {{ Some(E::A(v)) => v, _ => 0 }}");
    compile_script_ir(&i, &with_catch_all, &flag(&i)).unwrap();
}

#[test]
fn a_match_whose_scrutinee_is_lent_to_a_call_stays_closed() {
    // A callee holding `&E{A(i64), B(i64)}` can write only what that type
    // names, so a lend cannot widen the set the match must cover.
    let i = Interner::new();
    let source = format!(
        "{TWO_CONSTRUCTIONS}let f = |q| -> 1; let n = f(&e); match e {{ E::A(v) => v, E::B(v) => v }}"
    );
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_parameter_is_closed_by_its_type() {
    let i = Interner::new();
    let source =
        format!("{TWO_CONSTRUCTIONS}let f = |q| -> match q {{ E::A(v) => v, E::B(v) => v }}; f(e)");
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_parameter_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}let f = |q| -> match q {{ E::A(v) => v }}; f(e)");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert!(
        err.contains("non-exhaustive match") && err.contains("E::B"),
        "{err}"
    );
}

#[test]
fn a_match_on_a_parameter_with_a_catch_all_is_accepted() {
    let i = Interner::new();
    let source =
        format!("{TWO_CONSTRUCTIONS}let f = |q| -> match q {{ E::A(v) => v, _ => 0 }}; f(e)");
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}

/// The Brainfuck bench's `match prog[pc]` shape: the scrutinee is an element
/// of a container, and the queue of 2026-09-19 recorded it as forcing a `_`.
#[test]
fn a_match_on_a_container_element_needs_no_catch_all() {
    let i = Interner::new();
    let source = "let v = [E::A(1), E::B(2)]; match &v[0] { E::A(x) => x, E::B(x) => x }";
    compile_script_ir(&i, source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_field_needs_no_catch_all() {
    let i = Interner::new();
    let source = format!(
        "{TWO_CONSTRUCTIONS}let o = {{ f: e, }}; match &o.f {{ E::A(x) => x, E::B(x) => x }}"
    );
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_field_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}let o = {{ f: e, }}; match &o.f {{ E::A(x) => x }}");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert_eq!(
        err,
        "non-exhaustive match: `E::B` is not covered; add that arm or a `_` arm"
    );
}

#[test]
fn a_match_on_a_returned_value_needs_no_catch_all() {
    let i = Interner::new();
    let source =
        format!("{TWO_CONSTRUCTIONS}let f = |q| -> q; match f(e) {{ E::A(x) => x, E::B(x) => x }}");
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}
