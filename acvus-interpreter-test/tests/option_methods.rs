//! `Option`'s `std` methods at the script contract: each name reads as
//! Rust's, resolves against the `Iter` and `Result` methods that share it,
//! and answers the same at both optimization levels.

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Refusal, check_source, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn compile_and_run(source: &str, ret: Ty, opt: Opt) -> Result<Value, Refusal> {
    let i = Interner::new();
    let ast = ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("the script parses"));
    let cr = check_source(
        &i,
        ast,
        &FxHashMap::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        &i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()))
}

fn run_or_report(source: &str, ret: Ty, opt: Opt) -> Value {
    match compile_and_run(source, ret, opt) {
        Ok(value) => value,
        Err(Refusal { messages, .. }) => {
            panic!("`{source}` did not compile:\n  {}", messages.join("\n  "))
        }
    }
}

fn at_both_opts(source: &str, ret: Ty) -> Value {
    let none = run_or_report(source, ret.clone(), Opt::None);
    let full = run_or_report(source, ret, Opt::Full);
    assert_eq!(
        format!("{none:?}"),
        format!("{full:?}"),
        "the two optimization levels disagree on `{source}`"
    );
    full
}

fn int(source: &str) -> i64 {
    at_both_opts(source, Ty::I64).as_int()
}

fn boolean(source: &str) -> bool {
    at_both_opts(source, Ty::Bool).as_bool()
}

fn text(source: &str) -> String {
    let value = at_both_opts(source, Ty::String);
    // SAFETY: the script's declared result type is `String`.
    unsafe { value.as_str() }.to_string()
}

/// `Some(3)` or `None`, written so that the script's own inference gives the
/// payload its type.
fn of(present: bool, body: &str) -> String {
    format!("let o = if {present} {{ Some(3) }} else {{ None }}; {body}")
}

// -- unwrap / expect ----------------------------------------------------

#[test]
fn unwrap_gives_the_payload() {
    assert_eq!(int(&of(true, "o.unwrap()")), 3);
}

#[test]
#[should_panic(expected = "unwrap: called on None")]
fn unwrap_on_none_traps() {
    int(&of(false, "o.unwrap()"));
}

#[test]
fn expect_gives_the_payload() {
    assert_eq!(int(&of(true, r#"o.expect("a three".to_string())"#)), 3);
}

#[test]
#[should_panic(expected = "a three")]
fn expect_on_none_traps_with_its_message() {
    int(&of(false, r#"o.expect("a three".to_string())"#));
}

// -- unwrap_or / unwrap_or_else -----------------------------------------

#[test]
fn unwrap_or_takes_the_default_only_on_none() {
    assert_eq!(int(&of(true, "o.unwrap_or(9)")), 3);
    assert_eq!(int(&of(false, "o.unwrap_or(9)")), 9);
}

#[test]
fn unwrap_or_else_calls_its_closure_only_on_none() {
    assert_eq!(int(&of(true, "o.unwrap_or_else(| | -> 9)")), 3);
    assert_eq!(int(&of(false, "o.unwrap_or_else(| | -> 9)")), 9);
}

// -- predicates ---------------------------------------------------------

#[test]
fn is_some_and_is_none_are_complements() {
    assert!(boolean(&of(true, "o.is_some()")));
    assert!(!boolean(&of(true, "o.is_none()")));
    assert!(!boolean(&of(false, "o.is_some()")));
    assert!(boolean(&of(false, "o.is_none()")));
}

#[test]
fn is_some_and_tests_the_payload() {
    assert!(boolean(&of(true, "o.is_some_and(|x| -> x == 3)")));
    assert!(!boolean(&of(true, "o.is_some_and(|x| -> x == 4)")));
    assert!(!boolean(&of(false, "o.is_some_and(|x| -> x == 3)")));
}

#[test]
fn is_none_or_holds_on_none() {
    assert!(boolean(&of(true, "o.is_none_or(|x| -> x == 3)")));
    assert!(!boolean(&of(true, "o.is_none_or(|x| -> x == 4)")));
    assert!(boolean(&of(false, "o.is_none_or(|x| -> x == 4)")));
}

// -- map_or / map_or_else ----------------------------------------------

#[test]
fn map_or_takes_the_default_only_on_none() {
    assert_eq!(int(&of(true, "o.map_or(0, |x| -> x * 2)")), 6);
    assert_eq!(int(&of(false, "o.map_or(0, |x| -> x * 2)")), 0);
}

#[test]
fn map_or_else_calls_the_default_closure_only_on_none() {
    assert_eq!(int(&of(true, "o.map_or_else(| | -> -1, |x| -> x * 2)")), 6);
    assert_eq!(
        int(&of(false, "o.map_or_else(| | -> -1, |x| -> x * 2)")),
        -1
    );
}

// -- and / and_then / or / or_else / xor --------------------------------

#[test]
fn and_keeps_the_second_option_only_when_the_first_is_some() {
    assert_eq!(int(&of(true, "o.and(Some(7)).unwrap_or(0)")), 7);
    assert_eq!(int(&of(false, "o.and(Some(7)).unwrap_or(0)")), 0);
}

#[test]
fn and_then_chains_an_option_returning_closure() {
    assert_eq!(
        int(&of(
            true,
            "o.and_then(|x| -> if x > 2 { Some(x * 10) } else { None }).unwrap_or(0)"
        )),
        30
    );
    assert_eq!(
        int(&of(
            false,
            "o.and_then(|x| -> if x > 2 { Some(x * 10) } else { None }).unwrap_or(0)"
        )),
        0
    );
}

#[test]
fn or_takes_the_second_option_only_when_the_first_is_none() {
    assert_eq!(int(&of(true, "o.or(Some(7)).unwrap_or(0)")), 3);
    assert_eq!(int(&of(false, "o.or(Some(7)).unwrap_or(0)")), 7);
}

#[test]
fn or_else_calls_its_closure_only_when_the_first_is_none() {
    assert_eq!(int(&of(true, "o.or_else(| | -> Some(7)).unwrap_or(0)")), 3);
    assert_eq!(int(&of(false, "o.or_else(| | -> Some(7)).unwrap_or(0)")), 7);
}

#[test]
fn xor_is_some_when_exactly_one_side_is() {
    assert_eq!(int(&of(true, "o.xor(None).unwrap_or(0)")), 3);
    assert_eq!(int(&of(false, "o.xor(Some(7)).unwrap_or(0)")), 7);
    assert_eq!(int(&of(true, "o.xor(Some(7)).unwrap_or(0)")), 0);
    assert_eq!(int(&of(false, "o.xor(None).unwrap_or(0)")), 0);
}

// -- flatten ------------------------------------------------------------

#[test]
fn flatten_drops_one_layer() {
    assert_eq!(
        int("let o = if true { Some(Some(3)) } else { None }; o.flatten().unwrap_or(0)"),
        3
    );
    assert_eq!(
        int("let o = if false { Some(Some(3)) } else { None }; o.flatten().unwrap_or(0)"),
        0
    );
}

// -- ok_or / ok_or_else -------------------------------------------------

#[test]
fn ok_or_turns_none_into_the_given_error() {
    assert_eq!(
        text(&of(
            true,
            r#"if let Ok(v) = o.ok_or("gone".to_string()) { v.to_string() } else { "err".to_string() }"#
        )),
        "3"
    );
    assert_eq!(
        text(&of(
            false,
            r#"if let Err(e) = o.ok_or("gone".to_string()) { e } else { "ok".to_string() }"#
        )),
        "gone"
    );
}

#[test]
fn ok_or_else_calls_its_closure_only_on_none() {
    assert_eq!(
        text(&of(
            false,
            r#"if let Err(e) = o.ok_or_else(| | -> "gone".to_string()) { e } else { "ok".to_string() }"#
        )),
        "gone"
    );
}

// -- into_iter ----------------------------------------------------------

#[test]
fn an_option_iterates_over_its_payload() {
    assert_eq!(int(&of(true, "(o.into_iter() | sum())")), 3);
    assert_eq!(int(&of(false, "(o.into_iter() | sum())")), 0);
}

// -- the names `Iter` also carries ---------------------------------------

/// `flatten` names an `Option` method and an `Iter` method and takes no
/// closure, so a call picks by its receiver. `map` and `filter`, which do
/// take one, lost their rows to that same second entry, and this pins the
/// line between the two cases.
#[test]
fn flatten_still_resolves_over_an_iterator() {
    assert_eq!(
        int("let v = vec([vec([1, 2]), vec([3, 4])]);
             (v.into_iter() | flatten() | filter(|x| -> *x > 2) | sum())"),
        7
    );
}

// -- refusal ------------------------------------------------------------

#[test]
fn and_then_refuses_a_closure_that_does_not_read_the_payload() {
    let refusal = compile_and_run(
        &of(true, "o.and_then(|s| -> Some(s.len())).unwrap_or(0)"),
        Ty::I64,
        Opt::Full,
    );
    assert!(
        refusal.is_err(),
        "an `i64` payload reaching a `&str` method compiled"
    );
}
