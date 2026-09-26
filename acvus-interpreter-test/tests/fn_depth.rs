//! RFC-0100 rule 5: a recursion that would run its thread's native stack
//! past what the machine keeps for a trap traps (RFC-0048) rather than
//! overflowing the stack, on every call path a recursion can take, on every
//! kind of thread that runs a machine.
//!
//! The recursions run in a child process: an overflow aborts the whole
//! process, so a regression here fails the one test that spawned the child
//! rather than ending the runner. The child's tests are ignored in every
//! other run.

use std::process::Command;
use std::sync::Arc;

use acvus_extern::{
    Closure, ClosureFn, Ctx, OneValue, Registry, Runtime, Var, extern_fn, extern_registry, kind,
};
use acvus_interpreter::{AcvusRuntime, DEPTH_TRAP, HostError, Interpreter};
use acvus_interpreter_test::{Helper, check_graph, execute_compiled};
use acvus_mir::graph::ParsedAst;
use acvus_mir::graph::optimize::Opt;
use acvus_mir::ty::{ParamTerm, Poly, Ty, lift_to_poly};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// The path libtest names the child's tests by, which the parent filters on.
const CHILD: &str = "fn_depth::child::";

const CHILD_TESTS: usize = 17;

/// A spawned `std` thread's and a tokio worker's default stack.
const DEFAULT_STACK: usize = 2 << 20;

/// wasmtime's default `max_wasm_stack`, and half of it.
const SMALL_STACKS: [usize; 2] = [512 << 10, 256 << 10];

/// The frame of `heavy`'s handler: most of what the extern contract lets a
/// handler spend before it calls a function value back (RFC-0100 rule 5).
const HANDLER_FRAME: usize = 24 << 10;

/// Nested regions and straight-line statements a level of a recursion runs
/// through: past the seven the previous probe nested, and past
/// `GUARD_EVERY`, so a level holds `StackGuard`s of its own.
const NESTING: usize = 100;

#[test]
fn every_recursion_path_traps_in_a_child_process() {
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

#[extern_fn(effect = pure)]
async fn later(n: i64) -> i64 {
    n
}

#[extern_fn(effect = pure)]
fn same(n: i64) -> i64 {
    std::hint::black_box(n)
}

fn heavy_now<E, Rt>(ctx: &mut Ctx<'_, Rt>, n: i64, f: Closure<'_, (i64,), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
    i64: OneValue<Rt>,
{
    let mut frame = [0u8; HANDLER_FRAME];
    std::hint::black_box(&mut frame);
    let called = f.call_now(ctx, (n,));
    called + i64::from(std::hint::black_box(&frame)[HANDLER_FRAME - 1])
}

#[extern_fn(effect = E, sync = heavy_now)]
async fn heavy<E, Rt>(ctx: &mut Ctx<'_, Rt>, n: i64, f: Closure<'_, (i64,), i64, E, Rt>) -> i64
where
    E: Var<kind::Effect>,
    Rt: Runtime,
    i64: OneValue<Rt>,
{
    let mut frame = [0u8; HANDLER_FRAME];
    std::hint::black_box(&mut frame);
    let called = f.call(ctx, (n,)).await;
    called + i64::from(std::hint::black_box(&frame)[HANDLER_FRAME - 1])
}

fn registry() -> Registry<AcvusRuntime> {
    extern_registry! { ns: "depth", fns: [later, same, heavy] }
}

struct HostFn {
    name: &'static str,
    source: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
enum Ended {
    Value(i64),
    Trapped { frames: usize },
}

fn under_nested_regions() -> String {
    let open = "if a != -1 { a = a + 1; ".repeat(NESTING);
    let close = "} ".repeat(NESTING);
    format!("fn f(n) {{ let acc = 0; let a = n; {open}acc = f(n + 1); {close}acc }}\nf(0)")
}

fn after_a_long_straight_line() -> String {
    let line = "a = same(a); ".repeat(NESTING);
    format!("fn f(n) {{ let a = n; {line}f(a + 1) }}\nf(0)")
}

/// Levels of a recursion no thread's stack holds, at a byte a level.
const NO_STACK_HOLDS: usize = 1 << 40;

const THROUGH_A_LARGE_HANDLER: &str = "fn f(n) { heavy(n + 1, |x| -> f(x)) }\nf(0)";

const DIRECT: &str = "fn f(n) { f(n + 1) }\nf(0)";

const AWAITED: &str = "fn f(n) { later(n) + f(n + 1) }\nf(0)";

fn trapped_frames(message: &str) -> usize {
    let prefix = format!("{DEPTH_TRAP}: a call nests ");
    let suffix = " frames deep, past the stack its thread has left (RFC-0100 rule 5)";
    message
        .strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(suffix))
        .and_then(|frames| frames.parse().ok())
        .unwrap_or_else(|| panic!("the run trapped other than by the depth trap: {message}"))
}

/// The type checker recurses into a nested block, and a debug build's
/// checker overflows a 2 MiB thread at fifty nested `if`s, so a script is
/// compiled on a thread of this stack and run on another.
const COMPILE_STACK: usize = 256 << 20;

fn prepared(main: &str, host_fns: &'static [HostFn], opt: Opt) -> Interpreter {
    let main = main.to_owned();
    std::thread::Builder::new()
        .stack_size(COMPILE_STACK)
        .spawn(move || compiled(&main, host_fns, opt))
        .expect("the compiling thread starts")
        .join()
        .expect("the script compiles")
}

fn compiled(main: &str, host_fns: &[HostFn], opt: Opt) -> Interpreter {
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
    let (_, interp) = execute_compiled(
        &i,
        compiled,
        std::collections::HashMap::new(),
        Arc::new(acvus_interpreter::SequentialExecutor),
    );
    interp
}

fn driven(mut interp: Interpreter) -> Ended {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread runtime");
    ended_as(runtime.block_on(interp.execute()))
}

fn ended_as(ran: Result<acvus_interpreter::Value, HostError>) -> Ended {
    match ran {
        Ok(value) => Ended::Value(value.as_int()),
        Err(HostError::Trapped { message }) => Ended::Trapped {
            frames: trapped_frames(&message),
        },
        Err(other) => panic!("the run ended with {other:?}, neither a value nor a trap"),
    }
}

fn ended_on(main: &str, host_fns: &'static [HostFn], opt: Opt, stack: usize) -> Ended {
    let interp = prepared(main, host_fns, opt);
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || driven(interp))
        .expect("the run's thread starts")
        .join()
        .expect("the run ends in a value or a trap")
}

/// The two optimization levels end in the same value or both trap. A trap's
/// depth follows the stack each level's frames take, so theirs may differ.
fn ended(main: &str, host_fns: &'static [HostFn], stack: usize) -> Ended {
    let none = ended_on(main, host_fns, Opt::None, stack);
    let full = ended_on(main, host_fns, Opt::Full, stack);
    match (none, full) {
        (Ended::Value(a), Ended::Value(b)) if a == b => Ended::Value(a),
        (Ended::Trapped { frames }, Ended::Trapped { .. }) => Ended::Trapped { frames },
        (none, full) => panic!("the two optimization levels disagree: {none:?} and {full:?}"),
    }
}

fn assert_traps_on(main: &str, host_fns: &'static [HostFn], stack: usize) {
    if let Ended::Value(value) = ended(main, host_fns, stack) {
        panic!("the recursion ended with {value}")
    }
}

fn assert_traps(main: &str, host_fns: &'static [HostFn]) {
    assert_traps_on(main, host_fns, DEFAULT_STACK);
}

mod child {
    use super::*;

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_without_end_traps() {
        assert_traps(DIRECT, &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_mutual_recursion_without_end_traps() {
        assert_traps("fn e(n) { o(n + 1) }\nfn o(n) { e(n + 1) }\ne(0)", &[]);
    }

    /// `main` is the chain's first frame, and `f(k)` nests `k + 1` more. A
    /// run that traps at frame `n` passed every check below it, and a thread
    /// of the same stack lays each frame where the trapping run laid it, so
    /// `f(n - 3)` fits and `f(n - 2)` traps at frame `n` again.
    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_that_fits_completes_and_one_more_frame_traps() {
        let count = "fn f(n) { if n == 0 { 0 } else { 1 + f(n - 1) } }";
        for opt in [Opt::None, Opt::Full] {
            let at = |k: usize| ended_on(&format!("{count}\nf({k})"), &[], opt, DEFAULT_STACK);
            let Ended::Trapped { frames } = at(NO_STACK_HOLDS) else {
                panic!("{opt:?}: a recursion of {NO_STACK_HOLDS} levels ended in a value")
            };
            let fits = frames.checked_sub(3).expect("a trap past main and two frames of f");
            let value = i64::try_from(fits).expect("a depth a stack holds is an i64");
            assert_eq!(at(fits), Ended::Value(value), "{opt:?}");
            assert_eq!(at(fits + 1), Ended::Trapped { frames }, "{opt:?}");
        }
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_through_a_closure_traps() {
        assert_traps("fn f(n) { let g = |x| -> f(x); g(n + 1) }\nf(0)", &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_through_an_extern_calling_back_traps() {
        assert_traps("fn f(n) { [n + 1].as_iter().map(|x| -> f(*x)).sum() }\nf(0)", &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_through_awaited_calls_traps() {
        assert_traps(AWAITED, &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn host_functions_calling_each_other_trap() {
        const PING_PONG: &[HostFn] = &[
            HostFn { name: "ping", source: "if $n < 0 { 0 } else { pong($n + 1) }" },
            HostFn { name: "pong", source: "if $n < 0 { 0 } else { ping($n + 1) }" },
        ];
        assert_traps("ping(0)", PING_PONG);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn host_functions_calling_each_other_with_no_base_trap() {
        const PING_PONG: &[HostFn] = &[
            HostFn { name: "ping", source: "pong($n + 1)" },
            HostFn { name: "pong", source: "ping($n + 1)" },
        ];
        assert_traps("ping(0)", PING_PONG);
        assert_traps("pong(0)", PING_PONG);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_host_function_calling_itself_with_no_base_traps_also_as_an_operand() {
        const F: &[HostFn] = &[HostFn { name: "f", source: "f($n + 1)" }];
        for main in ["f(0)", "f(0) + 1", "1 + f(0)", "-f(0)"] {
            assert_traps(main, F);
        }
    }

    /// Two call sites from outside `f`'s component make two copies of it
    /// (RFC-0100 rule 3).
    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_through_a_second_copy_of_a_fn_traps() {
        assert_traps("fn f(n) { if n < 0 { 0 } else { f(n + 1) } }\nlet a = f(-1);\na + f(0)", &[]);
    }

    /// The chain is host function, extern, closure, host function.
    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_from_a_host_function_through_an_extern_traps() {
        const THROUGH: &[HostFn] = &[HostFn {
            name: "h",
            source: "if $n < 0 { 0 } else { [$n + 1].as_iter().map(|x| -> h(*x)).sum() }",
        }];
        assert_traps("h(0)", THROUGH);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_under_nested_regions_traps() {
        assert_traps(&under_nested_regions(), &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_after_a_long_straight_line_traps() {
        assert_traps(&after_a_long_straight_line(), &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_through_an_extern_with_a_large_frame_traps() {
        assert_traps(THROUGH_A_LARGE_HANDLER, &[]);
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn each_recursion_traps_on_a_512_kib_thread() {
        for main in [DIRECT.to_owned(), AWAITED.to_owned(), under_nested_regions(), after_a_long_straight_line(), THROUGH_A_LARGE_HANDLER.to_owned()] {
            assert_traps_on(&main, &[], SMALL_STACKS[0]);
        }
    }

    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn each_recursion_traps_on_a_256_kib_thread() {
        for main in [DIRECT.to_owned(), AWAITED.to_owned(), under_nested_regions(), after_a_long_straight_line(), THROUGH_A_LARGE_HANDLER.to_owned()] {
            assert_traps_on(&main, &[], SMALL_STACKS[1]);
        }
    }

    /// The host's own thread is whichever thread libtest runs this on.
    #[test]
    #[ignore = "exhausts a thread's stack; run by the parent test in a child process"]
    fn a_recursion_traps_on_the_hosts_thread_a_worker_and_a_blocking_thread() {
        for opt in [Opt::None, Opt::Full] {
            assert!(matches!(driven(prepared(DIRECT, &[], opt)), Ended::Trapped { .. }), "{opt:?}");

            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .build()
                .expect("a multi-thread runtime");
            let mut on_worker = prepared(DIRECT, &[], opt);
            let worker = runtime
                .block_on(runtime.spawn(async move { ended_as(on_worker.execute().await) }))
                .expect("the worker's task ends in a value or a trap");
            assert!(matches!(worker, Ended::Trapped { .. }), "{opt:?}: {worker:?}");

            let on_pool = prepared(DIRECT, &[], opt);
            let pooled = runtime
                .block_on(runtime.spawn_blocking(move || driven(on_pool)))
                .expect("the blocking task ends in a value or a trap");
            assert!(matches!(pooled, Ended::Trapped { .. }), "{opt:?}: {pooled:?}");
        }
    }
}
