//! RFC-0100 rule 5: a recursion past the machine's depth bound traps
//! (RFC-0048) rather than overflowing the native stack, on every call path a
//! recursion can take.
//!
//! A recursion that reaches the bound runs in a child process: an overflow
//! aborts the whole process, so a regression here fails the one test that
//! spawned the child rather than ending the runner. The child's tests are
//! ignored in every other run, and each runs its script on a spawned thread
//! of the smallest stack the bound was derived for.

use std::process::Command;
use std::sync::Arc;

use acvus_extern::{Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, DEPTH_TRAP, Depth, HostError};
use acvus_interpreter_test::{Helper, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, Ty, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// The path libtest names the child's tests by, which the parent filters on.
const CHILD: &str = "fn_depth::child::";

/// The tests under `child`, each of which the parent's run must report
/// passed.
const CHILD_TESTS: usize = 9;

#[test]
fn every_recursion_path_traps_at_the_bound_in_a_child_process() {
    let exe = std::env::current_exe().expect("the test binary's own path");
    let out = Command::new(exe)
        .args([CHILD, "--ignored", "--test-threads=1"])
        // The panic hook then prints a backtrace at the deepest frame, so
        // the trap spends that stack too, as a host's default hook does.
        .env("RUST_BACKTRACE", "1")
        .output()
        .expect("the test binary runs as a child");
    let stdout = String::from_utf8(out.stdout).expect("libtest writes UTF-8");
    let stderr = String::from_utf8(out.stderr).expect("libtest writes UTF-8");
    assert!(out.status.success(), "the child ended with {}:\n{stdout}\n{stderr}", out.status);
    let passed = format!("test result: ok. {CHILD_TESTS} passed");
    assert!(stdout.contains(&passed), "the child ran other than {CHILD_TESTS} tests:\n{stdout}");
}

/// The exact text the trap reports for a chain that reaches one frame past
/// the bound.
fn past_the_bound() -> String {
    format!(
        "{DEPTH_TRAP}: a call nests {} frames deep, and a call chain nests at most {} \
         (RFC-0100 rule 5)",
        Depth::BOUND + 1,
        Depth::BOUND
    )
}

#[extern_fn(effect = pure)]
async fn later(n: i64) -> i64 {
    n
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "depth", fns: [later] }
}

/// A host function of one `i64` parameter, `$n`.
struct HostFn {
    name: &'static str,
    source: &'static str,
}

#[derive(Debug)]
enum Ended {
    Value(i64),
    Trapped(String),
}

/// `main` run to its end on a spawned thread of `Depth::SMALLEST_STACK`, at
/// both optimization levels, which must end alike.
fn ended(main: &str, host_fns: &'static [HostFn]) -> Ended {
    let main = main.to_owned();
    std::thread::Builder::new()
        .stack_size(Depth::SMALLEST_STACK)
        .spawn(move || {
            let [none, full] = [Opt::None, Opt::Full].map(|opt| run(&main, host_fns, opt));
            match (none, full) {
                (Ended::Value(a), Ended::Value(b)) if a == b => Ended::Value(a),
                (Ended::Trapped(a), Ended::Trapped(b)) if a == b => Ended::Trapped(a),
                (none, full) => panic!("the two optimization levels disagree: {none:?} and {full:?}"),
            }
        })
        .expect("a thread of the smallest stack starts")
        .join()
        .expect("the run ends in a value or a trap")
}

fn run(main: &str, host_fns: &[HostFn], opt: Opt) -> Ended {
    let i = Interner::new();
    let helpers: Vec<Helper<'_>> = host_fns
        .iter()
        .map(|f| Helper {
            name: f.name,
            source: f.source,
            params: vec![ParamTerm::<Poly>::new(i.intern("n"), lift_to_poly(&Ty::I64))],
        })
        .collect();
    let mut registries = acvus_ext::std_registries::<AcvusRuntime>();
    registries.push(registry());
    let parsed = ParsedAst::Script(acvus_ast::parse_script(&i, main).expect("main parses"));
    let compiled = check_graph(&i, parsed, &helpers, &FxHashMap::default(), registries, Ty::I64, opt, |_| {})
        .unwrap_or_else(|r| panic!("{opt:?} refused {main}:\n  {}", r.messages.join("\n  ")));
    let (_, mut interp) = execute_compiled(
        &i,
        compiled,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    match runtime.block_on(interp.execute()) {
        Ok(value) => Ended::Value(value.as_int()),
        Err(HostError::Trapped { message }) => Ended::Trapped(message),
        Err(other) => panic!("the run ended with {other:?}, neither a value nor a trap"),
    }
}

fn assert_traps_past_the_bound(main: &str, host_fns: &'static [HostFn]) {
    match ended(main, host_fns) {
        Ended::Trapped(message) => assert_eq!(message, past_the_bound()),
        Ended::Value(value) => panic!("the recursion ended with {value}"),
    }
}

mod child {
    use super::*;

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_without_end_traps() {
        assert_traps_past_the_bound("fn f(n) { f(n + 1) }\nf(0)", &[]);
    }

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_mutual_recursion_without_end_traps() {
        assert_traps_past_the_bound("fn e(n) { o(n + 1) }\nfn o(n) { e(n + 1) }\ne(0)", &[]);
    }

    /// `main` is the chain's first frame, and `f(k)` nests `k + 1` more, so
    /// `f(BOUND - 2)` fills the bound exactly and `f(BOUND - 1)` passes it.
    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_that_fills_the_bound_completes_and_one_more_frame_traps() {
        let deepest = i64::from(Depth::BOUND) - 2;
        let count = "fn f(n) { if n == 0 { 0 } else { 1 + f(n - 1) } }";
        match ended(&format!("{count}\nf({deepest})"), &[]) {
            Ended::Value(value) => assert_eq!(value, deepest),
            Ended::Trapped(message) => panic!("a recursion at the bound trapped: {message}"),
        }
        assert_traps_past_the_bound(&format!("{count}\nf({})", deepest + 1), &[]);
    }

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_through_a_closure_traps() {
        assert_traps_past_the_bound("fn f(n) { let g = |x| -> f(x); g(n + 1) }\nf(0)", &[]);
    }

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_through_an_extern_calling_back_traps() {
        assert_traps_past_the_bound(
            "fn f(n) { [n + 1].as_iter().map(|x| -> f(*x)).sum() }\nf(0)",
            &[],
        );
    }

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_through_awaited_calls_traps() {
        assert_traps_past_the_bound("fn f(n) { later(n) + f(n + 1) }\nf(0)", &[]);
    }

    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn host_functions_calling_each_other_trap() {
        const PING_PONG: &[HostFn] = &[
            HostFn { name: "ping", source: "if $n < 0 { 0 } else { pong($n + 1) }" },
            HostFn { name: "pong", source: "if $n < 0 { 0 } else { ping($n + 1) }" },
        ];
        assert_traps_past_the_bound("ping(0)", PING_PONG);
    }

    /// Each call from outside a component gets its own copy (RFC-0100 rule
    /// 3): two sites, two copies, and the chain through either is counted.
    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_through_a_second_copy_of_a_fn_traps() {
        assert_traps_past_the_bound(
            "fn f(n) { if n < 0 { 0 } else { f(n + 1) } }\nlet a = f(-1);\na + f(0)",
            &[],
        );
    }

    /// A host function whose recursion runs through an extern's callback:
    /// the chain is host function, extern, closure, host function.
    #[test]
    #[ignore = "reaches the depth bound; run by the parent test in a child process"]
    fn a_recursion_from_a_host_function_through_an_extern_traps() {
        const THROUGH: &[HostFn] = &[HostFn {
            name: "h",
            source: "if $n < 0 { 0 } else { [$n + 1].as_iter().map(|x| -> h(*x)).sum() }",
        }];
        assert_traps_past_the_bound("h(0)", THROUGH);
    }
}
