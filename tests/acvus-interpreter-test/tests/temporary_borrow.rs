//! A temporary can be borrowed: the value a method call's receiver, or a
//! `&` argument, just produced is bound to a temporary storage, and the
//! parameter borrows that temporary. Every value written beside a program
//! is the value Rust gives for the same expression.

use acvus_interpreter::{AcvusRuntime, SequentialExecutor, Value};
use acvus_interpreter_test::{Refusal, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{IntTy, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;

fn compile_and_run(i: &Interner, main: &str, ret: Ty, opt: Opt) -> Result<Value, Refusal> {
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let cr = check_graph(
        i,
        parsed,
        &[],
        &FxHashMap::default(),
        acvus_ext::std_registries::<AcvusRuntime>(),
        ret,
        opt,
        |_| {},
    )?;
    let (_, mut interp) = execute_compiled(
        i,
        cr,
        std::collections::HashMap::new(),
        Arc::new(SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    Ok(runtime.block_on(interp.execute()).expect("the seeds hold every context the run fetches"))
}

fn at_both(main: &str, ret: Ty) -> Value {
    let i = Interner::new();
    let at = |opt| {
        compile_and_run(&i, main, ret.clone(), opt)
            .unwrap_or_else(|r| panic!("{opt:?} refused:\n  {}", r.messages.join("\n  ")))
    };
    let full = at(Opt::Full);
    let none = at(Opt::None);
    assert_eq!(
        format!("{full:?}"),
        format!("{none:?}"),
        "the two levels read one program: {main}"
    );
    full
}

fn text(main: &str) -> String {
    let v = at_both(main, Ty::String);
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    unsafe { v.as_str() }.to_owned()
}

fn count(main: &str) -> u64 {
    at_both(main, Ty::Int(IntTy::U64)).as_int() as u64
}

fn int(main: &str) -> i64 {
    at_both(main, Ty::I64).as_int()
}

fn boolean(main: &str) -> bool {
    at_both(main, Ty::Bool).as_bool()
}

fn refusal(main: &str, ret: Ty) -> String {
    let i = Interner::new();
    match compile_and_run(&i, main, ret, Opt::Full) {
        Ok(value) => panic!("expected a refusal, ran to {value:?}"),
        Err(r) => r.messages.join("\n"),
    }
}

#[test]
fn a_sum_of_floats_is_its_own_receiver() {
    assert_eq!(text("let a = 1.5; let b = 2.0; (a + b).to_string()"), "3.5");
}

#[test]
fn a_call_s_result_is_its_own_receiver() {
    assert_eq!(text("let v = vec([1, 2]); v.len().to_string()"), "2");
}

#[test]
fn a_sum_of_ints_is_its_own_receiver() {
    assert_eq!(text("let a = 3; (a + 1).to_string()"), "4");
}

#[test]
fn a_joined_string_is_its_own_receiver() {
    assert_eq!(count(r#"let s = "x".to_string(); (s + "y").len()"#), 2);
}

#[test]
fn the_binding_the_rule_removes_reads_the_same() {
    assert_eq!(text("let a = 3; let b = a + 1; b.to_string()"), "4");
}

#[test]
fn a_string_temporary_reaches_a_view_parameter() {
    assert_eq!(count(r#"("a".to_string() + "bc").len()"#), 3);
    assert_eq!(text(r#"("ab".to_string() + "c").upper()"#), "ABC");
    assert!(boolean(r#"("ab".to_string() + "c").starts_with("ab")"#));
}

#[test]
fn a_literal_s_owned_copy_is_its_own_receiver() {
    assert_eq!(count(r#""lit".to_string().len()"#), 3);
}

#[test]
fn a_vec_temporary_is_its_own_receiver() {
    assert_eq!(count("vec([1, 2]).len()"), 2);
    assert_eq!(count("[1, 2, 3].len()"), 3);
}

#[test]
fn a_temporary_inside_a_loop_body_is_released_each_iteration() {
    assert_eq!(
        int("let total = 0;\
             for i in 0..4 {\
                 total = total + ((\"x\".to_string() + \"y\").len() as i64);\
             }\
             total"),
        8
    );
}

#[test]
fn a_temporary_inside_a_closure_is_the_closure_s() {
    assert_eq!(
        int("let xs = [1, 2, 3]; xs.as_iter().map(|x| -> (*x + 1).to_string().len() as i64) | sum"),
        3
    );
}

#[test]
fn a_temporary_outlives_the_loan_a_binding_holds_on_it() {
    assert_eq!(int("let a = 1; let b = &(a + 1); *b"), 2);
    assert_eq!(
        count(r#"let s = &("a".to_string() + "b"); let n = len(s); n + len(s)"#),
        4
    );
}

#[test]
fn a_reference_into_a_temporary_is_not_stored() {
    let refused = refusal(r#"let a = 1; let xs = [&(a + 1)]; 0"#, Ty::I64);
    assert!(
        refused.contains("a reference cannot be stored"),
        "expected the storage refusal, got {refused}"
    );
}
