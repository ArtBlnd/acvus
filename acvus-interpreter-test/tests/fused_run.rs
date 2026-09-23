//! A run of extern calls that feed each other, and the deref that may close
//! it, is one operation (RFC-0044 rule 7).
//!
//! Every fusing case asserts a value as well as a shape. A run that hands
//! the wrong intermediate to the wrong call prepares the same number of
//! operations, so only the value can catch it.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use acvus_extern::{ExternType, Registry, Var, extern_fn, extern_registry, kind};
use acvus_interpreter::listing::body_listing;
use acvus_interpreter::{AcvusRuntime, PrepareCtx, Value, prepare_module};
use acvus_interpreter_test::listing::{code_listing, ops_of_anywhere};
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
    I: Var<kind::Identity>;

#[extern_fn(effect = pure)]
fn pure_row<I>(n: i64) -> Row<I>
where
    I: Var<kind::Identity>,
{
    Row((0..n).map(Counted).collect(), PhantomData)
}

#[extern_fn(effect = opaque)]
fn opaque_row<I>(n: i64) -> Row<I>
where
    I: Var<kind::Identity>,
{
    Row((0..n).map(Counted).collect(), PhantomData)
}

#[extern_fn(effect = pure)]
fn row_at<I>(row: Row<I>, index: i64) -> i64
where
    I: Var<kind::Identity>,
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
    I: Var<kind::Identity>,
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
        Ty::I64,
    );
    let ctx = PrepareCtx {
        interner: &interner,
        externs: &cr.extern_executables,
        context_names: &cr.context_names,
        instances: &cr.instances,
    };

    let mut found = Vec::new();
    for module in cr.modules.values() {
        let prepared = prepare_module(module, &ctx);
        let listings = std::iter::once(body_listing(&prepared.main))
            .chain(prepared.closures.values().map(|code| code_listing(code)));
        for listing in listings {
            for op in ops_of_anywhere(&listing) {
                if let Some(shape) = fused_shape(&op) {
                    found.push(shape);
                }
            }
        }
    }
    found
}

/// `Fused<CALLS, TAIL, LARGE>` carries its own shape in its type: RFC-0044 rule 7
/// made the call count and the tail const parameters, so the instance name
/// is the shape and nothing has to be read out of a payload.
fn fused_shape(op: &str) -> Option<RunShape> {
    let args = op.strip_prefix("Fused<")?.strip_suffix('>')?;
    let mut args = args.split(',').map(str::trim);
    let calls = args.next()?.parse().expect("a Fused call count");
    let tail = args.next()? == "true";
    Some(RunShape { calls, tail })
}

async fn value_of(source: &str) -> Value {
    let interner = Interner::new();
    let context = matrix(&interner);
    run_parsed_with_externs(
        &interner,
        parsed(&interner, source),
        context,
        registries(),
        Ty::I64,
        |_| {},
    )
    .await
    .value
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
    assert_eq!(run_shapes("double(double(1))"), [two_calls_and_no_deref()]);
    assert_eq!(value_of("double(double(1))").await.as_int(), 4);
}

#[tokio::test]
async fn every_index_of_a_run_reaches_the_call_the_source_gave_it() {
    let _guard = row_scripts();
    for index in 0..SIDE {
        let source = format!("row_at(pure_row({SIDE}), {index})");
        assert_eq!(run_shapes(&source), [two_calls_and_no_deref()], "{source}");
        assert_eq!(value_of(&source).await.as_int(), index, "{source}");
    }
}

/// An element access is no longer a run: `a[i]` is one `AsSlice` per index
/// expression, and nothing reads an `AsSlice`'s result but the `Index`
/// beside it, which is an operation and not a call (RFC-0047 rule 3).
#[tokio::test]
async fn an_index_expression_is_no_run_at_all() {
    assert_eq!(run_shapes("@m[1][0]"), []);
    assert_eq!(
        value_of("@m[1][0]").await.as_int(),
        element_at(Cell { row: 1, column: 0 })
    );
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
    let _guard = row_scripts();
    let source = &format!("let r = pure_row({SIDE}); row_at(r, 0) + 0");
    assert_eq!(
        run_shapes(source),
        [two_calls_and_no_deref()],
        "one reader fuses the row into the call that reads it"
    );
    let shared = &format!("let n = double(1); double(n) + double(n)");
    assert_eq!(
        run_shapes(shared),
        [],
        "`n` is read twice, so `double(1)` joins neither run"
    );
    assert_eq!(value_of(shared).await.as_int(), 8);
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
    let _guard = row_scripts();
    let source = &format!(
        "let acc = 0; let i = 0; while i < 3 {{ acc = acc + row_at(pure_row({SIDE}), i); i = i + 1; }} acc"
    );
    assert_eq!(run_shapes(source), [two_calls_and_no_deref()]);
    assert_eq!(value_of(source).await.as_int(), (0..SIDE).sum::<i64>());
}

#[tokio::test]
async fn a_run_inside_a_diamond_arm_is_one_operation() {
    let _guard = row_scripts();
    let source = &format!(
        "let v = if @flag {{ row_at(pure_row({SIDE}), 1) }} else {{ row_at(pure_row({SIDE}), 2) }}; v"
    );
    assert_eq!(
        run_shapes(source),
        [two_calls_and_no_deref(), two_calls_and_no_deref()]
    );
    assert_eq!(value_of(source).await.as_int(), 1);
}
