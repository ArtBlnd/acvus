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
                home: acvus_mir::ty::Home::NONE,
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
    let branches = optimized.matches(" if ").count() + optimized.matches("jump_if ").count();
    assert_eq!(
        branches, 1,
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
    let source = "let v = [E::A(1), E::B(2)]; match &v[0] { E::A(x) => *x, E::B(x) => *x }";
    compile_script_ir(&i, source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_field_needs_no_catch_all() {
    let i = Interner::new();
    let source = format!(
        "{TWO_CONSTRUCTIONS}let o = {{ f: e, }}; match &o.f {{ E::A(x) => *x, E::B(x) => *x }}"
    );
    compile_script_ir(&i, &source, &flag(&i)).unwrap();
}

#[test]
fn a_match_on_a_field_that_misses_a_variant_is_refused() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}let o = {{ f: e, }}; match &o.f {{ E::A(x) => *x }}");
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

// -- Literal arms (RFC-0051, extended 2026-09-20) --------------------

#[test]
fn a_match_on_integer_literals_is_one_switch() {
    let i = Interner::new();
    let raw = compile_script_raw(
        &i,
        "let n = 2; match n { 1 => 10, 2 => 20, _ => 0 }",
        &flag(&i),
    )
    .unwrap();
    assert!(
        raw.contains("switch ") && raw.contains("1 -> ") && raw.contains("2 -> "),
        "the printer writes the literal keys: {raw}"
    );
    assert!(
        !raw.contains(" test "),
        "no chain of tests is left beside the dispatch: {raw}"
    );
}

#[test]
fn a_match_on_string_literals_is_one_switch() {
    let i = Interner::new();
    let raw = compile_script_raw(
        &i,
        "let s = \"put\"; match s { \"get\" => 1, \"put\" => 2, _ => 0 }",
        &flag(&i),
    )
    .unwrap();
    assert!(
        raw.contains("switch ") && raw.contains("\"get\" -> ") && raw.contains("\"put\" -> "),
        "the printer writes the string keys as written: {raw}"
    );
    assert!(!raw.contains(" test "), "{raw}");
}

#[test]
fn a_match_on_char_literals_is_one_switch() {
    let i = Interner::new();
    let raw = compile_script_raw(
        &i,
        "let c = 'b'; match c { 'a' => 1, 'b' => 2, _ => 0 }",
        &flag(&i),
    )
    .unwrap();
    assert!(
        raw.contains("switch ") && raw.contains("'a' -> ") && raw.contains("'b' -> "),
        "the printer writes the char keys quoted: {raw}"
    );
}

#[test]
fn a_match_on_both_bools_needs_no_catch_all() {
    let i = Interner::new();
    let raw = compile_script_raw(&i, "match @c { true => 1, false => 0 }", &flag(&i)).unwrap();
    assert!(
        raw.contains("switch ") && raw.contains("true -> ") && raw.contains("false -> "),
        "{raw}"
    );
    assert!(
        !raw.contains("_ -> "),
        "no catch-all was written, so the Switch has no default: {raw}"
    );
    compile_script_ir(&i, "match @c { true => 1, false => 0 }", &flag(&i)).unwrap();
}

#[test]
fn a_match_on_one_bool_and_no_catch_all_is_refused() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "match @c { true => 1 }", &flag(&i)).unwrap_err();
    assert_eq!(
        err,
        "non-exhaustive match: `false` is not covered; add that arm or a `_` arm"
    );
}

#[test]
fn a_match_on_integer_literals_without_a_catch_all_is_refused() {
    let i = Interner::new();
    let err =
        compile_script_ir(&i, "let n = 2; match n { 1 => 10, 2 => 20 }", &flag(&i)).unwrap_err();
    assert_eq!(
        err,
        "non-exhaustive match: the integers are an open value space; add a `_` arm"
    );
}

#[test]
fn a_match_on_string_literals_without_a_catch_all_is_refused() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "let s = \"put\"; match s { \"get\" => 1, \"put\" => 2 }",
        &flag(&i),
    )
    .unwrap_err();
    assert_eq!(
        err,
        "non-exhaustive match: the strings are an open value space; add a `_` arm"
    );
}

#[test]
fn a_match_on_char_literals_without_a_catch_all_is_refused() {
    let i = Interner::new();
    let err = compile_script_ir(&i, "let c = 'b'; match c { 'a' => 1, 'b' => 2 }", &flag(&i))
        .unwrap_err();
    assert_eq!(
        err,
        "non-exhaustive match: the chars are an open value space; add a `_` arm"
    );
}

#[test]
fn two_arms_naming_one_literal_are_refused() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "let n = 2; match n { 1 => 10, 1 => 20, _ => 0 }",
        &flag(&i),
    )
    .unwrap_err();
    assert!(
        err.contains("unreachable pattern: `1` is already covered by an earlier arm"),
        "{err}"
    );
}

#[test]
fn two_arms_naming_one_string_are_refused() {
    let i = Interner::new();
    let err = compile_script_ir(
        &i,
        "let s = \"a\"; match s { \"a\" => 1, \"a\" => 2, _ => 0 }",
        &flag(&i),
    )
    .unwrap_err();
    assert!(
        err.contains("unreachable pattern: `\"a\"` is already covered by an earlier arm"),
        "{err}"
    );
}

#[test]
fn two_arms_naming_one_tag_are_refused() {
    let i = Interner::new();
    let source = format!("{TWO_CONSTRUCTIONS}match e {{ E::A(v) => v, E::A(v) => v, _ => 0 }}");
    let err = compile_script_ir(&i, &source, &flag(&i)).unwrap_err();
    assert!(
        err.contains("unreachable pattern: `A` is already covered by an earlier arm"),
        "{err}"
    );
}

/// Equality on a float is not a jump, so a float arm is no dispatch key and
/// the `match` keeps the chain of tests -- which is why it still needs the
/// `_` that makes it exhaustive.
#[test]
fn a_match_on_float_literals_keeps_the_chain() {
    let i = Interner::new();
    let source = "let x = 1.5; match x { 1.5 => 1, 2.5 => 2, _ => 0 }";
    let raw = compile_script_raw(&i, source, &flag(&i)).unwrap();
    assert!(!raw.contains("switch "), "no dispatch is written: {raw}");
    assert_eq!(
        raw.matches(" test ").count(),
        2,
        "one test per tested arm: {raw}"
    );
    let err = compile_script_ir(&i, "let x = 1.5; match x { 1.5 => 1, 2.5 => 2 }", &flag(&i))
        .unwrap_err();
    assert!(err.contains("not one dispatch over a tag"), "{err}");
}

/// The keys of one `Switch` are of one kind, and `typeck` is what makes that
/// so: an arm whose pattern is not the scrutinee's type is refused before
/// the lowering asks whether the arms are one dispatch. `Dispatch::plan`'s
/// one-kind rule is the defence behind this, and no source reaches it.
#[test]
fn an_arm_of_another_kind_than_the_scrutinee_is_refused_by_typeck() {
    let i = Interner::new();
    let tag_and_int = format!("{TWO_CONSTRUCTIONS}match e {{ E::A(v) => v, 1 => 2, _ => 0 }}");
    let err = compile_script_ir(&i, &tag_and_int, &flag(&i)).unwrap_err();
    assert!(err.contains("pattern type i64 incompatible"), "{err}");

    let int_and_char = "let n = 1; match n { 1 => 1, 'a' => 2, _ => 0 }";
    let err = compile_script_ir(&i, int_and_char, &flag(&i)).unwrap_err();
    assert!(err.contains("expected char, got i64"), "{err}");
}
