//! A run of extern calls that feed each other, and the deref that may close
//! it, is one operation (RFC-0044, stage 6).
//!
//! Every fusing case asserts a value as well as a shape. A run that hands
//! the wrong intermediate to the wrong call prepares the same number of
//! operations, so only the value can catch it.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, IdentityVar, Registry, extern_fn, extern_registry};
use acvus_interpreter::code::{Code, Payload};
use acvus_interpreter::{AcvusRuntime, PrepareCtx, Value, prepare_module};
use acvus_interpreter_test::{
    Context, compile_source_with_externs, run_parsed_with_externs, split_context, value_from_json,
};
use acvus_mir::graph::ParsedAst;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

static ELEMENTS_DROPPED: AtomicUsize = AtomicUsize::new(0);

/// `ELEMENTS_DROPPED` is one counter and the harness runs these tests on
/// parallel threads, so every script that builds a `Row` runs under this
/// lock and the counter has one writer at a time.
static ROW_SCRIPTS: Mutex<()> = Mutex::new(());

fn row_scripts() -> MutexGuard<'static, ()> {
    ROW_SCRIPTS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One element of a `Row`, counted as it is dropped: the row a fused run
/// carries from one call to the next owns `ROW_LEN` of these, so a leak or
/// a double drop is a count that is not `ROW_LEN`.
struct Counted(i64);

impl Drop for Counted {
    fn drop(&mut self) {
        ELEMENTS_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

const ROW_LEN: i64 = 3;

#[derive(ExternType)]
#[repr(transparent)]
struct Row<I>(Vec<Counted>, PhantomData<I>)
where
    I: IdentityVar;

#[extern_fn(effect = pure)]
fn pure_row<I>(n: i64) -> Row<I>
where
    I: IdentityVar,
{
    Row((0..n).map(Counted).collect(), PhantomData)
}

#[extern_fn(effect = opaque)]
fn opaque_row<I>(n: i64) -> Row<I>
where
    I: IdentityVar,
{
    Row((0..n).map(Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn row_at<I>(row: Row<I>, index: i64) -> i64
where
    I: IdentityVar,
{
    row.0[index as usize].0
}

#[extern_fn(effect = pure)]
fn double(n: i64) -> i64 {
    n * 2
}

#[extern_fn(effect = pure)]
fn row_boom<I>(_row: Row<I>, _index: i64) -> i64
where
    I: IdentityVar,
{
    panic!("row_boom")
}

fn registries() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(extern_registry! {
        ns: "fused",
        types: [Row<_>],
        fns: [pure_row, opaque_row, row_at, row_boom, double],
    });
    regs
}

const SIDE: i64 = 3;

#[derive(Clone, Copy)]
struct Cell {
    row: i64,
    column: i64,
}

fn element_at(cell: Cell) -> i64 {
    cell.row * 10 + cell.column
}

fn matrix(interner: &Interner) -> Context {
    let rows: Vec<serde_json::Value> = (0..SIDE)
        .map(|row| {
            (0..SIDE)
                .map(|column| serde_json::json!(element_at(Cell { row, column })))
                .collect()
        })
        .collect();
    let json = serde_json::json!({ "m": rows, "flag": true });
    json.as_object()
        .expect("an object of contexts")
        .iter()
        .map(|(name, value)| (interner.intern(name), value_from_json(interner, value)))
        .collect()
}

fn parsed(interner: &Interner, source: &str) -> ParsedAst {
    ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"))
}

#[derive(PartialEq, Eq, Debug)]
struct RunShape {
    calls: usize,
    tail: bool,
}

/// In payload order, over the entry body and then every closure it makes.
fn run_shapes(source: &str) -> Vec<RunShape> {
    let interner = Interner::new();
    let (context_types, _snapshot): (FxHashMap<Astr, Ty>, _) =
        split_context(&interner, matrix(&interner));
    let cr = compile_source_with_externs(
        &interner,
        parsed(&interner, source),
        &context_types,
        registries(),
    );
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
    };

    let mut found = Vec::new();
    for module in cr.modules.values() {
        let prepared = prepare_module(module, &ctx);
        let bodies = std::iter::once(&prepared.main).chain(prepared.closures.values());
        for code in bodies {
            let payloads: &[Payload] = match &**code {
                Code::Body(body) => &body.payloads,
                Code::Expr(_) => &[],
            };
            for payload in payloads {
                if let Payload::Fused(run) = payload {
                    found.push(RunShape {
                        calls: run.calls.len(),
                        tail: run.tail.is_some(),
                    });
                }
            }
        }
    }
    found
}

async fn value_of(source: &str) -> Value {
    let interner = Interner::new();
    let context = matrix(&interner);
    run_parsed_with_externs(
        &interner,
        parsed(&interner, source),
        context,
        registries(),
        |_| {},
    )
    .await
    .value
}

fn one_call_and_a_deref() -> RunShape {
    RunShape {
        calls: 1,
        tail: true,
    }
}

fn two_calls_and_a_deref() -> RunShape {
    RunShape {
        calls: 2,
        tail: true,
    }
}

fn dropped_since(before: usize) -> i64 {
    i64::try_from(ELEMENTS_DROPPED.load(Ordering::Relaxed) - before)
        .expect("a drop count below i64::MAX")
}

fn two_calls_and_no_deref() -> RunShape {
    RunShape {
        calls: 2,
        tail: false,
    }
}

#[tokio::test]
async fn a_call_whose_result_the_next_call_reads_is_one_operation_with_it() {
    assert_eq!(run_shapes("*@m.get(1).get(0)"), [two_calls_and_a_deref()]);
    assert_eq!(
        value_of("*@m.get(1).get(0)").await.as_int(),
        element_at(Cell { row: 1, column: 0 })
    );
}

#[tokio::test]
async fn every_index_of_a_run_reaches_the_call_the_source_gave_it() {
    for row in 0..SIDE {
        for column in 0..SIDE {
            let source = format!("*@m.get({row}).get({column})");
            assert_eq!(run_shapes(&source), [two_calls_and_a_deref()]);
            assert_eq!(
                value_of(&source).await.as_int(),
                element_at(Cell { row, column }),
                "{source}"
            );
        }
    }
}

/// One test, because `ELEMENTS_DROPPED` is one counter: two tests reading
/// it would race on the harness's threads.
///
/// The intermediate is moved into the second handler's argument before that
/// handler runs, so a panic there unwinds the callee's local, not the fused
/// operation's. Either way the row reaches exactly one owner.
#[test]
fn a_large_intermediate_moves_into_the_next_call_and_is_dropped_once() {
    let _guard = row_scripts();
    let moved = &format!("row_at(pure_row({ROW_LEN}), 2)");
    let panics = &format!("row_boom(pure_row({ROW_LEN}), 0)");
    assert_eq!(run_shapes(moved), [two_calls_and_no_deref()]);
    assert_eq!(run_shapes(panics), [two_calls_and_no_deref()]);

    let rt = || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a current-thread tokio runtime")
    };

    let before = ELEMENTS_DROPPED.load(Ordering::Relaxed);
    assert_eq!(rt().block_on(value_of(moved)).as_int(), 2);
    assert_eq!(
        dropped_since(before),
        ROW_LEN,
        "the row the first call made is dropped once, by the call it moved into"
    );

    let before = ELEMENTS_DROPPED.load(Ordering::Relaxed);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ran = std::panic::catch_unwind(|| rt().block_on(value_of(panics)));
    std::panic::set_hook(previous);

    assert!(ran.is_err(), "row_boom panics");
    assert_eq!(
        dropped_since(before),
        ROW_LEN,
        "the unwind drops the row exactly once"
    );
}

#[tokio::test]
async fn a_second_reader_of_an_intermediate_leaves_the_outer_call_unfused() {
    let source = "let r = @m.get(1); *r.get(0) + *r.get(2)";
    assert_eq!(
        run_shapes(source),
        [one_call_and_a_deref(), one_call_and_a_deref()],
        "the row `r` is read twice, so `@m.get(1)` joins neither run"
    );
    assert_eq!(
        value_of(source).await.as_int(),
        element_at(Cell { row: 1, column: 0 }) + element_at(Cell { row: 1, column: 2 })
    );
}

#[tokio::test]
async fn an_order_edge_leaves_the_call_unfused() {
    let source = &format!("row_at(opaque_row({ROW_LEN}), 2)");
    assert_eq!(run_shapes(source), []);
    let _guard = row_scripts();
    assert_eq!(value_of(source).await.as_int(), 2);
}

#[tokio::test]
async fn a_call_whose_result_arithmetic_reads_is_not_a_run() {
    assert_eq!(run_shapes("double(1) + 2"), []);
    assert_eq!(value_of("double(1) + 2").await.as_int(), 4);
}

#[tokio::test]
async fn a_run_inside_a_loop_body_is_one_operation() {
    let source =
        "let acc = 0; let i = 0; while i < 3 { acc = acc + *@m.get(i).get(i); i = i + 1; } acc";
    assert_eq!(run_shapes(source), [two_calls_and_a_deref()]);
    let diagonal: i64 = (0..SIDE)
        .map(|i| element_at(Cell { row: i, column: i }))
        .sum();
    assert_eq!(value_of(source).await.as_int(), diagonal);
}

#[tokio::test]
async fn a_run_inside_a_diamond_arm_is_one_operation() {
    let source = "let v = if @flag { *@m.get(0).get(1) } else { *@m.get(2).get(0) }; v";
    assert_eq!(
        run_shapes(source),
        [two_calls_and_a_deref(), two_calls_and_a_deref()]
    );
    assert_eq!(
        value_of(source).await.as_int(),
        element_at(Cell { row: 0, column: 1 })
    );
}
