//! RFC-0048 rule 8 on the machine, at both optimization levels.

use acvus_interpreter::HostError;
use acvus_interpreter_test::{Helper, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, Ty, TyTerm};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

const DIVIDE_BY_ZERO: &str = "attempt to divide by zero";

#[derive(Debug, PartialEq, Eq)]
enum Ended {
    Value(i64),
    Trapped(String),
}

fn run(i: &Interner, helper: Option<&str>, main: &str, opt: Opt) -> Ended {
    run_once_compiled(i, helper, main, opt, || {})
}

fn run_once_compiled(
    i: &Interner,
    helper: Option<&str>,
    main: &str,
    opt: Opt,
    on_compiled: impl FnOnce(),
) -> Ended {
    let helpers: Vec<Helper<'_>> = helper
        .into_iter()
        .map(|source| Helper {
            name: "h",
            source,
            params: vec![ParamTerm::<Poly>::new(i.intern("n"), TyTerm::I64)],
        })
        .collect();
    let parsed = ParsedAst::Script(acvus_ast::parse_script(i, main).expect("main parses"));
    let compiled = check_graph(
        i,
        parsed,
        &helpers,
        &FxHashMap::default(),
        acvus_ext::std_registries::<acvus_interpreter::AcvusRuntime>(),
        Ty::I64,
        opt,
        |_| {},
    )
    .unwrap_or_else(|r| panic!("{opt:?} refused {main}:\n  {}", r.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        i,
        compiled,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    on_compiled();
    match runtime.block_on(interp.execute()) {
        Ok(value) => Ended::Value(value.as_int()),
        Err(HostError::Trapped { message }) => Ended::Trapped(message),
        Err(other) => panic!("at {opt:?}, {main} ended with {other}"),
    }
}

fn ends_at_both_levels(helper: Option<&str>, main: &str, ended: Ended) {
    let i = Interner::new();
    for opt in [Opt::None, Opt::Full] {
        assert_eq!(run(&i, helper, main, opt), ended, "at {opt:?}: {main}");
    }
}

fn trapped(text: &str) -> Ended {
    Ended::Trapped(text.to_owned())
}

#[test]
fn an_unused_call_of_a_local_function_that_divides_by_zero_traps() {
    ends_at_both_levels(Some("$n / 0"), "let d = h(1); 5", trapped(DIVIDE_BY_ZERO));
}

#[test]
fn an_unused_call_of_a_local_function_that_cannot_trap_runs_past() {
    ends_at_both_levels(Some("$n / 2"), "let d = h(1); 5", Ended::Value(5));
}

#[test]
fn an_unused_call_of_a_recursive_local_function_traps() {
    let helper = "if $n == 0 { 10 / $n } else { h($n - 1) }";
    ends_at_both_levels(Some(helper), "let d = h(3); 5", trapped(DIVIDE_BY_ZERO));
}

#[test]
fn an_unused_call_of_an_extern_not_declared_total_traps() {
    ends_at_both_levels(
        None,
        "let z = 0; let d = 7.wrapping_div(z); 5",
        trapped("wrapping_div: divisor is zero"),
    );
}

#[test]
fn an_unused_call_of_an_extern_declared_total_runs_past() {
    ends_at_both_levels(None, "let d = 9223372036854775807.wrapping_add(1); 5", Ended::Value(5));
}

#[test]
fn an_unused_index_past_the_length_traps() {
    ends_at_both_levels(
        None,
        "let a = [1, 2]; let k = 5; let e = a[k]; 3",
        trapped("index out of bounds: the len is 2 but the index is 5"),
    );
}

#[test]
fn an_unused_index_within_the_length_runs_past() {
    ends_at_both_levels(
        None,
        "let a = [1, 2]; let k = 1; let e = a[k]; 3",
        Ended::Value(3),
    );
}

const WHILE_ENDING_ONLY_WHERE_N_IS_AT_MOST_SEVEN: &str =
    "let i = 0; while i < $n { i = i % 7 + 1; } i";

/// The machine has no way to stop a run, so a run that does not end keeps
/// its thread busy until the test binary exits.
fn run_within(
    helper: &'static str,
    main: &'static str,
    opt: Opt,
    limit: Duration,
) -> Option<Ended> {
    let (compiled, is_compiled) = mpsc::channel();
    let (ended, has_ended) = mpsc::channel();
    std::thread::spawn(move || {
        let i = Interner::new();
        let end = run_once_compiled(&i, Some(helper), main, opt, || {
            compiled.send(()).expect("the test waits for the compile");
        });
        let _ = ended.send(end);
    });
    is_compiled
        .recv()
        .unwrap_or_else(|_| panic!("at {opt:?}, {main} did not compile"));
    match has_ended.recv_timeout(limit) {
        Ok(end) => Some(end),
        Err(RecvTimeoutError::Timeout) => None,
        Err(RecvTimeoutError::Disconnected) => panic!("at {opt:?}, the run of {main} panicked"),
    }
}

#[test]
fn an_unused_call_of_a_local_function_whose_while_does_not_end_does_not_end() {
    for opt in [Opt::None, Opt::Full] {
        let helper = WHILE_ENDING_ONLY_WHERE_N_IS_AT_MOST_SEVEN;
        let ended = run_within(helper, "let d = h(10); 5", opt, Duration::from_secs(2));
        assert_eq!(ended, None, "at {opt:?}");
    }
}

#[test]
fn an_unused_call_of_a_local_function_whose_while_ends_runs_past() {
    ends_at_both_levels(
        Some(WHILE_ENDING_ONLY_WHERE_N_IS_AT_MOST_SEVEN),
        "let d = h(5); 5",
        Ended::Value(5),
    );
}

#[test]
fn an_unused_call_of_a_local_function_whose_loop_is_a_for_runs_past() {
    ends_at_both_levels(
        Some("let i = 0; for k in 0..$n { i = i % 7 + 1; } i"),
        "let d = h(10); 5",
        Ended::Value(5),
    );
}

#[test]
fn an_unused_call_of_a_local_function_calling_an_extern_that_states_no_return_traps() {
    ends_at_both_levels(
        Some("$n.wrapping_div(0)"),
        "let d = h(7); 5",
        trapped("wrapping_div: divisor is zero"),
    );
}
