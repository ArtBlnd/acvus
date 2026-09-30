//! `Result`'s `result::` methods at the script contract: each name reads as
//! Rust's, resolves against the `Option` and `Iter` methods that share it,
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
    Ok(runtime.block_on(interp.execute()).expect("the seeds hold every context the run fetches"))
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

/// `Ok(3)` or `Err("boom")`, written so that the script's own inference gives
/// both arms their types.
fn of(ok: bool, body: &str) -> String {
    format!(r#"let r = if {ok} {{ Ok(3) }} else {{ Err("boom".to_string()) }}; {body}"#)
}

// -- predicates ---------------------------------------------------------

#[test]
fn is_ok_and_is_err_are_complements() {
    assert!(boolean(&of(true, "r.is_ok()")));
    assert!(!boolean(&of(true, "r.is_err()")));
    assert!(!boolean(&of(false, "r.is_ok()")));
    assert!(boolean(&of(false, "r.is_err()")));
}

#[test]
fn is_ok_and_tests_the_ok_payload() {
    assert!(boolean(&of(true, "r.is_ok_and(|x| -> x == 3)")));
    assert!(!boolean(&of(true, "r.is_ok_and(|x| -> x == 4)")));
    assert!(!boolean(&of(false, "r.is_ok_and(|x| -> x == 3)")));
}

#[test]
fn is_err_and_tests_the_err_payload() {
    assert!(boolean(&of(false, r#"r.is_err_and(|e| -> e.len() == 4)"#)));
    assert!(!boolean(&of(false, r#"r.is_err_and(|e| -> e.len() == 9)"#)));
    assert!(!boolean(&of(true, r#"r.is_err_and(|e| -> e.len() == 4)"#)));
}

// -- ok / err -----------------------------------------------------------

#[test]
fn ok_and_err_each_keep_one_side() {
    assert_eq!(int(&of(true, "r.ok().unwrap_or(0)")), 3);
    assert_eq!(int(&of(false, "r.ok().unwrap_or(0)")), 0);
    assert_eq!(
        text(&of(false, r#"r.err().unwrap_or("none".to_string())"#)),
        "boom"
    );
    assert_eq!(
        text(&of(true, r#"r.err().unwrap_or("none".to_string())"#)),
        "none"
    );
}

// -- unwrap / unwrap_err / expect / expect_err --------------------------

#[test]
fn unwrap_gives_the_ok_payload() {
    assert_eq!(int(&of(true, "r.unwrap()")), 3);
}

#[test]
#[should_panic(expected = "unwrap: called on Err")]
fn unwrap_on_err_traps() {
    int(&of(false, "r.unwrap()"));
}

#[test]
fn unwrap_err_gives_the_err_payload() {
    assert_eq!(text(&of(false, "r.unwrap_err()")), "boom");
}

#[test]
#[should_panic(expected = "unwrap_err: called on Ok")]
fn unwrap_err_on_ok_traps() {
    text(&of(true, "r.unwrap_err()"));
}

#[test]
fn expect_and_expect_err_name_their_messages() {
    assert_eq!(int(&of(true, r#"r.expect("a three".to_string())"#)), 3);
    assert_eq!(
        text(&of(false, r#"r.expect_err("an error".to_string())"#)),
        "boom"
    );
}

#[test]
#[should_panic(expected = "a three")]
fn expect_on_err_traps_with_its_message() {
    int(&of(false, r#"r.expect("a three".to_string())"#));
}

#[test]
#[should_panic(expected = "an error")]
fn expect_err_on_ok_traps_with_its_message() {
    text(&of(true, r#"r.expect_err("an error".to_string())"#));
}

// -- unwrap_or / unwrap_or_else -----------------------------------------

#[test]
fn unwrap_or_takes_the_default_only_on_err() {
    assert_eq!(int(&of(true, "r.unwrap_or(9)")), 3);
    assert_eq!(int(&of(false, "r.unwrap_or(9)")), 9);
}

#[test]
fn unwrap_or_else_reads_the_error() {
    let body = "r.unwrap_or_else(|e| -> if e.len() == 4 { 44 } else { 0 })";
    assert_eq!(int(&of(true, body)), 3);
    assert_eq!(int(&of(false, body)), 44);
}

// -- map_err / map_or / map_or_else -------------------------------------

#[test]
fn map_err_rebuilds_the_result_at_the_new_err_payload() {
    let body =
        "r.map_err(|e| -> e.len() == 4).unwrap_or_else(|short| -> if short { 44 } else { 0 })";
    assert_eq!(int(&of(false, body)), 44);
    assert_eq!(int(&of(true, body)), 3);
}

#[test]
fn map_or_takes_the_default_only_on_err() {
    assert_eq!(int(&of(true, "r.map_or(0, |x| -> x * 2)")), 6);
    assert_eq!(int(&of(false, "r.map_or(0, |x| -> x * 2)")), 0);
}

#[test]
fn map_or_else_hands_the_error_to_its_default_closure() {
    let body = "r.map_or_else(|e| -> if e.len() == 4 { 44 } else { 0 }, |x| -> x * 2)";
    assert_eq!(int(&of(true, body)), 6);
    assert_eq!(int(&of(false, body)), 44);
}

// -- and / and_then / or / or_else --------------------------------------

#[test]
fn and_keeps_the_second_result_only_when_the_first_is_ok() {
    assert_eq!(
        int(&of(
            true,
            r#"r.and(if true { Ok(7) } else { Err("x".to_string()) }).unwrap_or(0)"#
        )),
        7
    );
    assert_eq!(
        int(&of(
            false,
            r#"r.and(if true { Ok(7) } else { Err("x".to_string()) }).unwrap_or(0)"#
        )),
        0
    );
}

#[test]
fn and_then_chains_a_result_returning_closure() {
    let body = r#"r.and_then(|x| -> if x > 2 { Ok(x * 10) } else { Err("small".to_string()) }).unwrap_or(0)"#;
    assert_eq!(int(&of(true, body)), 30);
    assert_eq!(int(&of(false, body)), 0);
}

#[test]
fn or_takes_the_second_result_only_when_the_first_is_err() {
    let body = r#"r.or(if true { Ok(7) } else { Err(1) }).unwrap_or(0)"#;
    assert_eq!(int(&of(true, body)), 3);
    assert_eq!(int(&of(false, body)), 7);
}

#[test]
fn or_else_reads_the_error_and_may_retype_it() {
    let body = r#"r.or_else(|e| -> if e.len() == 4 { Ok(7) } else { Err(1) }).unwrap_or(0)"#;
    assert_eq!(int(&of(true, body)), 3);
    assert_eq!(int(&of(false, body)), 7);
}

// -- flatten / transpose ------------------------------------------------

#[test]
fn flatten_drops_one_ok_layer() {
    let src = |ok: bool, inner: bool| {
        format!(
            r#"let r = if {ok} {{ Ok(if {inner} {{ Ok(3) }} else {{ Err("inner".to_string()) }}) }}
                      else {{ Err("outer".to_string()) }};
               r.flatten().unwrap_or(0)"#
        )
    };
    assert_eq!(int(&src(true, true)), 3);
    assert_eq!(int(&src(true, false)), 0);
    assert_eq!(int(&src(false, true)), 0);
}

#[test]
fn transpose_swaps_the_two_layers() {
    let src = |ok: bool, present: bool| {
        format!(
            r#"let r = if {ok} {{ Ok(if {present} {{ Some(3) }} else {{ None }}) }}
                      else {{ Err("boom".to_string()) }};
               r.transpose().map_or(0, |inner| -> inner.unwrap_or(-1))"#
        )
    };
    assert_eq!(int(&src(true, true)), 3);
    assert_eq!(int(&src(true, false)), 0);
    assert_eq!(int(&src(false, true)), -1);
}

// -- into_iter ----------------------------------------------------------

#[test]
fn a_result_iterates_over_its_ok_payload() {
    assert_eq!(int(&of(true, "(r.into_iter() | sum())")), 3);
    assert_eq!(int(&of(false, "(r.into_iter() | sum())")), 0);
}

// -- refusal ------------------------------------------------------------

#[test]
fn map_err_refuses_a_closure_that_reads_the_ok_side() {
    let refusal = compile_and_run(
        &of(false, "r.map_err(|e| -> e * 2).unwrap_or(0)"),
        Ty::I64,
        Opt::Full,
    );
    assert!(
        refusal.is_err(),
        "a `String` error reaching integer multiplication compiled"
    );
}
