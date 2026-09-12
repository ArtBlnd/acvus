//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{ExternFn, ExternItems, ExternRegistry, ExternType, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Executable, RuntimeError, TokioExecutor, Value};
use acvus_interpreter_test::*;
use acvus_mir::ir::InstKind;
use acvus_mir::ty::{Effect, Ty, TypeRegistry};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

/// A registry of stateless, concrete closures with one effect.
fn closures(
    effect: Effect,
    fns: impl Fn(&Interner) -> Vec<ExternFn<AcvusRuntime>> + 'static,
) -> ExternRegistry<AcvusRuntime> {
    ExternRegistry::new(move |i| ExternItems {
        types: vec![],
        fns: fns(i).into_iter().map(|f| f.with_effect(effect.clone())).collect(),
        persist: vec![],
    })
}

fn ctx(i: &Interner, entries: &[(&str, Value)]) -> FxHashMap<acvus_utils::Astr, Value> {
    entries
        .iter()
        .map(|(name, val)| (i.intern(name), val.clone()))
        .collect()
}

// =======================================================================
//  Pure ExternFn
// =======================================================================

#[tokio::test]
async fn extern_pure_add() {
    let i = Interner::new();

    let registry = closures(Effect::PURE, |i| {
        vec![ExternFn::sync(
            i,
            "ext_add",
            |_: &Interner, a: i64, b: i64| Ok(a + b),
        )]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "ext_add(10, 32)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

#[tokio::test]
async fn extern_pure_string_transform() {
    let i = Interner::new();

    let registry = closures(Effect::PURE, |i| {
        vec![ExternFn::sync(i, "shout", |_: &Interner, s: String| {
            Ok(s.to_uppercase())
        })]
    });

    let c = ctx(&i, &[("msg", Value::string("hello"))]);
    let result = run_script_with_externs(&i, "shout(@msg)", c, vec![registry]).await;
    assert_eq!(result.value, Value::string("HELLO"));
}

// =======================================================================
//  ExternFn capturing Rust environment
// =======================================================================

#[tokio::test]
async fn extern_captures_environment() {
    let i = Interner::new();
    let secret = 7i64;

    let registry = closures(Effect::PURE, move |i| {
        vec![ExternFn::sync(
            i,
            "multiply_secret",
            move |_: &Interner, x: i64| Ok(x * secret),
        )]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "multiply_secret(6)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

// =======================================================================
//  Regex ExternFn (legacy sync_handler, Builtin path)
// =======================================================================

#[tokio::test]
async fn regex_match_via_extern() {
    let i = Interner::new();

    let registry = acvus_ext::regex_registry();
    let c = ctx(&i, &[("text", Value::string("hello world 42"))]);
    let result = run_script_with_externs(
        &i,
        r#"re = regex("[0-9]+"); regex_match(re, @text)"#,
        c,
        vec![registry],
    )
    .await;
    assert_eq!(result.value, Value::Bool(true));
}

#[tokio::test]
async fn regex_find_via_extern() {
    let i = Interner::new();

    let registry = acvus_ext::regex_registry();
    let c = ctx(&i, &[("text", Value::string("price is 42 dollars"))]);
    let result = run_script_with_externs(
        &i,
        r#"re = regex("[0-9]+"); regex_find(re, @text)"#,
        c,
        vec![registry],
    )
    .await;
    // regex_find returns a Variant (Some/None). Check it contains "42".
    match &result.value {
        Value::Variant(v) => {
            assert_eq!(i.resolve(v.tag), "Some");
            let inner = v.payload.as_ref().expect("Some should have payload");
            assert_eq!(**inner, Value::string("42"));
        }
        Value::String(s) => assert_eq!(&**s, "42"),
        other => panic!("expected String or Variant(Some), got {other:?}"),
    }
}

// =======================================================================
//  IR verification: FunctionCall has correct context_uses/context_defs
// =======================================================================

/// Pure ExternFn should have empty context_uses/context_defs in IR.
#[test]
fn ir_pure_function_call_no_context_bindings() {
    let i = Interner::new();

    let registry = closures(Effect::PURE, |i| {
        vec![ExternFn::sync(
            i,
            "double",
            |_: &Interner, x: i64| Ok(x * 2),
        )]
    });

    let context_types: FxHashMap<acvus_utils::Astr, Ty> = FxHashMap::default();

    let source = "double(21)";
    let cr = compile_source_with_externs(
        &i,
        acvus_mir::graph::ParsedAst::Script(
            acvus_ast::parse_script(&i, source).expect("parse error"),
        ),
        &context_types,
        vec![registry],
        acvus_mir::ty::TypeRegistry::new(),
    );

    let entry_module = cr.modules.get(&cr.entry_qref).unwrap();
    let module = match entry_module {
        Executable::Module(m) => m,
        _ => panic!("entry should be a Module"),
    };

    let call_insts: Vec<_> = module
        .main
        .insts
        .iter()
        .filter(|inst| {
            matches!(
                &inst.kind,
                InstKind::FunctionCall {
                    callee: acvus_mir::ir::Callee::Direct(id),
                    ..
                } if cr.extern_executables.contains_key(&id)
            )
        })
        .collect();

    assert!(
        !call_insts.is_empty(),
        "should have a FunctionCall to double"
    );
}

// =======================================================================
//  IO ExternFn - Parallelization end-to-end
// =======================================================================
//
// Tests verify that the full optimizer pipeline (SpawnSplit -> CodeMotion ->
// Reorder -> SSA -> RegColor) produces correct MIR structure AND correct
// execution results for various IO parallelization patterns.
//
// Each test dumps the optimized MIR to stderr (--nocapture) for inspection.

#[extern_fn]
fn fetch_a(_: &Interner) -> i64 {
    100
}

#[extern_fn]
fn fetch_b(_: &Interner) -> i64 {
    200
}

#[extern_fn]
fn fetch_c(_: &Interner) -> i64 {
    300
}

#[extern_fn]
fn fetch_d(_: &Interner) -> i64 {
    400
}

#[extern_fn]
fn fetch_by(_: &Interner, x: i64) -> i64 {
    x * 10
}

/// Four independent Opaque fetches and one parameterized.
/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw_a(_: &Interner) -> i64 {
    5
}

#[extern_fn(effect = idempotent, commutative)]
fn draw_b(_: &Interner) -> i64 {
    7
}

/// Adds `by` to the lent place and returns the new value (RFC-0015).
#[extern_fn(effect = pure)]
fn bump(_: &Interner, n: &mut i64, by: i64) -> i64 {
    *n += by;
    *n
}

fn io_registry() -> ExternRegistry<AcvusRuntime> {
    extern_registry! {
        fns: [fetch_a, fetch_b, fetch_c, fetch_d, fetch_by, draw_a, draw_b, bump],
    }
}

/// Compile a script with io_registry, return (CompileResult, entry MirModule ref).
fn compile_io_script(source: &str) -> (Interner, CompileResult) {
    compile_io_script_with_ctx(source, &[])
}

fn compile_io_script_with_ctx(
    source: &str,
    context: &[(&str, Value)],
) -> (Interner, CompileResult) {
    let i = Interner::new();
    let ast =
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    let cr = compile_io_parsed(&i, ast, context);
    (i, cr)
}

/// Script mode (`anyorder`, `while`, `let`) with io_registry.
fn compile_io_script_mode(source: &str, context: &[(&str, Value)]) -> (Interner, CompileResult) {
    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script_mode(&i, source).expect("parse"),
    );
    let cr = compile_io_parsed(&i, ast, context);
    (i, cr)
}

fn compile_io_parsed(
    i: &Interner,
    ast: acvus_mir::graph::ParsedAst,
    context: &[(&str, Value)],
) -> CompileResult {
    let context_types: FxHashMap<acvus_utils::Astr, Ty> = context
        .iter()
        .map(|(name, val)| (i.intern(name), infer_ty(val)))
        .collect();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    compile_source_with_externs(i, ast, &context_types, regs, TypeRegistry::new())
}

async fn run_io_script_mode(source: &str, context: &[(&str, Value)]) -> Value {
    let i = Interner::new();
    run_io_script_mode_on(&i, source, context).await
}

async fn run_io_script_mode_on(i: &Interner, source: &str, context: &[(&str, Value)]) -> Value {
    let i = i.clone();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script_mode(&i, source).expect("parse"),
    );
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    run_parsed_with_externs(&i, ast, ctx(&i, context), regs, TypeRegistry::new())
        .await
        .value
}

fn infer_ty(v: &Value) -> Ty {
    match v {
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Bool(_) => Ty::Bool,
        Value::String(_) => Ty::String,
        Value::Array(items) => {
            let elem = items.first().map(infer_ty).unwrap_or(Ty::Int);
            Ty::Array(Box::new(elem), acvus_mir::ty::LenTerm::Known(items.len()))
        }
        _ => Ty::Unit,
    }
}

/// Dump MIR and return (spawn_positions, eval_positions) for assertion.
fn dump_and_positions(label: &str, i: &Interner, cr: &CompileResult) -> (Vec<usize>, Vec<usize>) {
    let module = match cr.modules.get(&cr.entry_qref).unwrap() {
        Executable::Module(m) => m,
        _ => panic!("expected Module"),
    };
    let dump = acvus_mir::printer::dump_with(i, module);
    eprintln!("=== {label} ===\n{dump}");

    let spawns: Vec<usize> = module
        .main
        .insts
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i.kind, InstKind::Spawn { .. }))
        .map(|(idx, _)| idx)
        .collect();
    let evals: Vec<usize> = module
        .main
        .insts
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i.kind, InstKind::Eval { .. }))
        .map(|(idx, _)| idx)
        .collect();
    (spawns, evals)
}

/// RFC-0007: with no `anyorder` block, effectful calls keep source order.
/// Each call is issued after the previous one has completed: the k-th
/// spawn follows the (k-1)-th eval, and each eval follows its own spawn.
fn assert_source_order(spawns: &[usize], evals: &[usize]) {
    assert_eq!(spawns.len(), evals.len(), "one eval per spawn");
    for (k, (&s, &e)) in spawns.iter().zip(evals).enumerate() {
        assert!(s < e, "spawn[{k}] at {s} must precede eval[{k}] at {e}");
        if k > 0 {
            let prev = evals[k - 1];
            assert!(
                prev < s,
                "eval[{}] at {prev} must precede spawn[{k}] at {s}: calls keep source order",
                k - 1
            );
        }
    }
}

// -- 1. Two independent IO calls ------------------------------------

/// fetch_a() + fetch_b() -> spawn both before eval either.
#[tokio::test]
async fn io_two_independent() {
    let i = Interner::new();
    let result = run_script_with_externs(
        &i,
        "fetch_a() + fetch_b()",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(300));
}

#[test]
fn io_two_independent_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b()");
    let (spawns, evals) = dump_and_positions("two_independent", &i, &cr);
    assert_eq!(spawns.len(), 2, "expected 2 spawns");
    assert_source_order(&spawns, &evals);
}

// -- 2. Four independent IO calls -----------------------------------

/// Four IO calls with no data dependency still run in source order.
#[tokio::test]
async fn io_four_way_parallel() {
    let i = Interner::new();
    let result = run_script_with_externs(
        &i,
        "fetch_a() + fetch_b() + fetch_c() + fetch_d()",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(1000));
}

#[test]
fn io_four_way_parallel_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b() + fetch_c() + fetch_d()");
    let (spawns, evals) = dump_and_positions("four_way_parallel", &i, &cr);
    assert_eq!(spawns.len(), 4, "expected 4 spawns");
    assert_source_order(&spawns, &evals);
}

// -- 3. Dependent chain + independent IO ----------------------------
//
// a = fetch_a()          // IO, independent
// b = fetch_by(a)        // IO, depends on a
// c = fetch_c()          // IO, independent of a and b
// b + c
//
// Source order: a, then by(a), then c; the data dependency adds nothing.

#[tokio::test]
async fn io_chain_with_independent() {
    let i = Interner::new();
    // fetch_a() = 100, fetch_by(100) = 1000, fetch_c() = 300
    let result = run_script_with_externs(
        &i,
        "a = fetch_a(); b = fetch_by(a); c = fetch_c(); b + c",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(1300));
}

#[test]
fn io_chain_with_independent_mir() {
    let (i, cr) = compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_c(); b + c");
    let (spawns, evals) = dump_and_positions("chain_with_independent", &i, &cr);

    assert_eq!(spawns.len(), 3, "expected 3 spawns");
    assert_source_order(&spawns, &evals);
}

// -- 4. Diamond dependency ------------------------------------------
//
// a = fetch_a()          // IO
// b = fetch_by(a)        // IO, depends on a
// c = fetch_by(a)        // IO, depends on a (same dep as b, but independent of b)
// b + c
//
// Both fetch_by(a) calls follow eval(a); the second follows the first.

#[tokio::test]
async fn io_diamond_dependency() {
    let i = Interner::new();
    // fetch_a() = 100, fetch_by(100) = 1000, fetch_by(100) = 1000
    let result = run_script_with_externs(
        &i,
        "a = fetch_a(); b = fetch_by(a); c = fetch_by(a); b + c",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(2000));
}

#[test]
fn io_diamond_dependency_mir() {
    let (i, cr) = compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_by(a); b + c");
    let (spawns, evals) = dump_and_positions("diamond_dependency", &i, &cr);

    assert_eq!(spawns.len(), 3, "expected 3 spawns (fetch_a + 2x fetch_by)");
    assert_source_order(&spawns, &evals);
}

// -- 5. Deep sequential chain ---------------------------------------
//
// a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d
//
// No parallelism possible: each depends on the previous.
// spawn->eval->spawn->eval->spawn->eval->spawn->eval

#[tokio::test]
async fn io_deep_chain() {
    let i = Interner::new();
    // 100 -> 1000 -> 10000 -> 100000
    let result = run_script_with_externs(
        &i,
        "a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(100000));
}

#[test]
fn io_deep_chain_mir() {
    let (i, cr) =
        compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d");
    let (spawns, evals) = dump_and_positions("deep_chain", &i, &cr);

    assert_eq!(spawns.len(), 4, "4 IO calls in chain");
    assert_source_order(&spawns, &evals);
}

// -- 6. Two independent chains --------------------------------------
//
// a = fetch_a(); b = fetch_by(a);    // chain 1: a -> b
// c = fetch_c(); d = fetch_by(c);    // chain 2: c -> d (independent of chain 1)
// b + d
//
// Source order: a, b, c, d. The chains are independent in data, not in order.

#[tokio::test]
async fn io_two_independent_chains() {
    let i = Interner::new();
    // chain 1: 100 -> 1000, chain 2: 300 -> 3000
    let result = run_script_with_externs(
        &i,
        "a = fetch_a(); b = fetch_by(a); c = fetch_c(); d = fetch_by(c); b + d",
        ctx(&i, &[]),
        vec![io_registry()],
    )
    .await;
    assert_eq!(result.value, Value::Int(4000));
}

#[test]
fn io_two_independent_chains_mir() {
    let (i, cr) =
        compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_c(); d = fetch_by(c); b + d");
    let (spawns, evals) = dump_and_positions("two_independent_chains", &i, &cr);

    assert_eq!(spawns.len(), 4, "4 IO calls");
    assert_source_order(&spawns, &evals);
}

// -- Lent places (RFC-0015) -----------------------------------------
//
// `f(&mut place)` loads the place, the callee changes what it received,
// and the value it left is stored back. A context, a local, and a field
// of a context are all places.

#[tokio::test]
async fn a_context_lent_mutably_comes_back_changed() {
    let v = run_io_script_mode("bump(&mut @n, 5); @n", &[("n", Value::Int(10))]).await;
    assert_eq!(v, Value::Int(15));
}

#[tokio::test]
async fn a_local_lent_mutably_comes_back_changed() {
    let v = run_io_script_mode("let x = 1; bump(&mut x, 2); bump(&mut x, 3); x", &[]).await;
    assert_eq!(v, Value::Int(6));
}

#[tokio::test]
async fn a_field_of_a_context_is_a_place() {
    let i = Interner::new();
    let a = Value::object(FxHashMap::from_iter([(i.intern("n"), Value::Int(1))]));
    let v = run_io_script_mode_on(&i, "bump(&mut @a.n, 1); @a.n", &[("a", a)]).await;
    assert_eq!(v, Value::Int(2));
}

#[tokio::test]
async fn the_return_value_of_a_lending_call_is_free() {
    let v = run_io_script_mode("let x = 40; let y = bump(&mut x, 2); x + y", &[]).await;
    assert_eq!(v, Value::Int(84));
}

// -- Commutative runs (RFC-0013) ------------------------------------
//
// Two draws are neighbours on the chain and both commute: they take the
// same entry order and are issued together, and a merge of what they
// yield follows. A call that does not commute between them keeps them
// apart.

#[tokio::test]
async fn commutative_draws_add_up() {
    let i = Interner::new();
    let result =
        run_script_with_externs(&i, "draw_a() + draw_b()", ctx(&i, &[]), vec![io_registry()]).await;
    assert_eq!(result.value, Value::Int(12));
}

#[test]
fn commutative_run_is_issued_together_mir() {
    let (i, cr) = compile_io_script("draw_a() + draw_b()");
    let (spawns, evals) = dump_and_positions("commutative_run", &i, &cr);
    assert_eq!(spawns.len(), 2, "expected 2 spawns");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "both draws are issued before either is awaited"
    );
    assert!(has_merge(&cr), "a merge joins what the run yielded");
}

#[test]
fn a_call_that_does_not_commute_keeps_source_order_mir() {
    let (i, cr) = compile_io_script("draw_a() + fetch_a() + draw_b()");
    let (spawns, evals) = dump_and_positions("broken_run", &i, &cr);
    assert_eq!(spawns.len(), 3, "expected 3 spawns");
    assert_source_order(&spawns, &evals);
    assert!(!has_merge(&cr), "no run, no merge");
}

#[test]
fn a_commutative_call_after_a_branch_is_issued_with_the_one_before_mir() {
    let (i, cr) = compile_io_script_mode(
        "@a = draw_a(); if @c { @x = 1; }; @b = draw_b(); @a + @b",
        &[
            ("a", Value::Int(0)),
            ("b", Value::Int(0)),
            ("c", Value::Bool(true)),
            ("x", Value::Int(0)),
        ],
    );
    let (spawns, evals) = dump_and_positions("run_across_branch", &i, &cr);
    assert_eq!(spawns.len(), 2, "expected 2 spawns");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "the draw after the branch is issued with the one before it"
    );
    assert!(has_merge(&cr));
}

fn has_merge(cr: &CompileResult) -> bool {
    let Executable::Module(m) = cr.modules.get(&cr.entry_qref).unwrap() else {
        panic!("expected Module");
    };
    m.main
        .insts
        .iter()
        .any(|i| matches!(i.kind, InstKind::Merge { .. }))
}

// -- anyorder (RFC-0007) --------------------------------------------
//
// Inside the block every effectful call takes the block's entry order and
// the block yields a merge of what they yielded; outside it, source order
// resumes. A loop inside accumulates through a loop phi.

#[tokio::test]
async fn anyorder_block_computes_the_same_value() {
    let v = run_io_script_mode(
        "anyorder { @a = fetch_a(); @b = fetch_b(); } @a + @b",
        &[("a", Value::Int(0)), ("b", Value::Int(0))],
    )
    .await;
    assert_eq!(v, Value::Int(300));
}

#[test]
fn anyorder_block_issues_its_calls_together_mir() {
    let (i, cr) = compile_io_script_mode(
        "anyorder { @a = fetch_a(); @b = fetch_b(); } @a + @b",
        &[("a", Value::Int(0)), ("b", Value::Int(0))],
    );
    let (spawns, evals) = dump_and_positions("anyorder", &i, &cr);
    assert_eq!(spawns.len(), 2, "expected 2 spawns");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "both calls are issued before either is awaited"
    );
    assert!(has_merge(&cr), "the block yields a merge");
}

#[test]
fn source_order_resumes_after_an_anyorder_block_mir() {
    let (i, cr) = compile_io_script_mode(
        "anyorder { @a = fetch_a(); @b = fetch_b(); } @c = fetch_c(); @a + @b + @c",
        &[
            ("a", Value::Int(0)),
            ("b", Value::Int(0)),
            ("c", Value::Int(0)),
        ],
    );
    let (spawns, evals) = dump_and_positions("anyorder_then", &i, &cr);
    assert_eq!(spawns.len(), 3, "expected 3 spawns");
    assert!(
        evals[0] < spawns[2] && evals[1] < spawns[2],
        "fetch_c waits for both calls of the block"
    );
}

#[tokio::test]
async fn a_loop_inside_anyorder_accumulates() {
    let v = run_io_script_mode(
        "anyorder { while @n > 0 { @s = @s + fetch_a(); @n = @n - 1; } } @s",
        &[("n", Value::Int(3)), ("s", Value::Int(0))],
    )
    .await;
    assert_eq!(v, Value::Int(300));
}

#[test]
fn a_loop_inside_anyorder_merges_per_iteration_mir() {
    let (i, cr) = compile_io_script_mode(
        "anyorder { while @n > 0 { @s = @s + fetch_a(); @n = @n - 1; } } @s",
        &[("n", Value::Int(3)), ("s", Value::Int(0))],
    );
    let (spawns, _) = dump_and_positions("anyorder_loop", &i, &cr);
    assert_eq!(spawns.len(), 1, "one call in the loop body");
    assert!(has_merge(&cr), "each iteration merges into the accumulator");
}

// -- Concurrency on TokioExecutor -----------------------------------
//
// Work starts at Spawn as a task. Two calls the compiler placed before
// their evals overlap; two calls on the chain do not. The probes count
// how many of them are in flight at once.

/// How long a probe stays in flight. Not a measurement: a window in which
/// another probe can start, so the counter can see two at once.
const PROBE_HOLD: std::time::Duration = std::time::Duration::from_millis(2);

/// How many probe calls were in flight at the same time.
struct Probe {
    in_flight: Arc<AtomicUsize>,
    max: Arc<AtomicUsize>,
}

impl Probe {
    fn new() -> Self {
        Probe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            max: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    /// `probe_a` and `probe_b`: each returns 100 after yielding once.
    fn registry(&self, effect: Effect) -> ExternRegistry<AcvusRuntime> {
        let in_flight = Arc::clone(&self.in_flight);
        let max = Arc::clone(&self.max);
        ExternRegistry::new(move |i| {
            let fns = ["probe_a", "probe_b"]
                .into_iter()
                .map(|name| {
                    let in_flight = Arc::clone(&in_flight);
                    let max = Arc::clone(&max);
                    ExternFn::r#async(i, name, move |_: Interner| {
                        let in_flight = Arc::clone(&in_flight);
                        let max = Arc::clone(&max);
                        async move {
                            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                            max.fetch_max(now, Ordering::SeqCst);
                            tokio::time::sleep(PROBE_HOLD).await;
                            in_flight.fetch_sub(1, Ordering::SeqCst);
                            Ok::<i64, RuntimeError>(100)
                        }
                    })
                    .with_effect(effect.clone())
                })
                .collect();
            ExternItems { types: vec![], fns, persist: vec![] }
        })
    }
}

async fn run_on_tokio(
    source: &str,
    script_mode: bool,
    context: &[(&str, Value)],
    registry: ExternRegistry<AcvusRuntime>,
) -> Value {
    let i = Interner::new();
    let script = if script_mode {
        acvus_ast::parse_script_mode(&i, source).expect("parse")
    } else {
        acvus_ast::parse_script(&i, source).expect("parse")
    };
    let ast = acvus_mir::graph::ParsedAst::Script(script);
    run_parsed_on(
        &i,
        ast,
        ctx(&i, context),
        vec![registry],
        TypeRegistry::new(),
        Arc::new(TokioExecutor),
    )
    .await
    .value
}

#[tokio::test]
async fn calls_on_the_chain_do_not_overlap() {
    let probe = Probe::new();
    let v = run_on_tokio(
        "@a = probe_a(); @b = probe_b(); @a + @b",
        true,
        &[("a", Value::Int(0)), ("b", Value::Int(0))],
        probe.registry(Effect::OPAQUE),
    )
    .await;
    assert_eq!(v, Value::Int(200));
    assert_eq!(
        probe.max(),
        1,
        "the second call starts after the first completes"
    );
}

#[tokio::test]
async fn an_anyorder_block_overlaps_its_calls() {
    let probe = Probe::new();
    let v = run_on_tokio(
        "anyorder { @a = probe_a(); @b = probe_b(); } @a + @b",
        true,
        &[("a", Value::Int(0)), ("b", Value::Int(0))],
        probe.registry(Effect::OPAQUE),
    )
    .await;
    assert_eq!(v, Value::Int(200));
    assert_eq!(probe.max(), 2, "both calls are in flight at once");
}

#[tokio::test]
async fn commutative_calls_overlap_across_a_branch() {
    let probe = Probe::new();
    let v = run_on_tokio(
        "@a = probe_a(); if @c { @x = 1; }; @b = probe_b(); @a + @b",
        true,
        &[
            ("a", Value::Int(0)),
            ("b", Value::Int(0)),
            ("c", Value::Bool(true)),
            ("x", Value::Int(0)),
        ],
        probe.registry(Effect::IDEMPOTENT.commutative()),
    )
    .await;
    assert_eq!(v, Value::Int(200));
    assert_eq!(
        probe.max(),
        2,
        "the branch between them does not keep them apart"
    );
}

#[tokio::test]
async fn commutative_calls_overlap_without_a_block() {
    let probe = Probe::new();
    let v = run_on_tokio(
        "probe_a() + probe_b()",
        false,
        &[],
        probe.registry(Effect::IDEMPOTENT.commutative()),
    )
    .await;
    assert_eq!(v, Value::Int(200));
    assert_eq!(probe.max(), 2, "a commutative run is in flight at once");
}

// -- 7. IO in iteration ---------------------------------------------
//
// Iterate over list, call IO per element, accumulate.
// Within each iteration: spawn should precede eval.

#[tokio::test]
async fn io_in_iteration() {
    let i = Interner::new();
    // fetch_by(1)=10, fetch_by(2)=20, fetch_by(3)=30 -> sum=60
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            ),
            ("sum", Value::Int(0)),
        ],
    );
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    let result = run_script_with_externs_and_types(
        &i,
        "@items | iter | map(|x| -> fetch_by(x)) | fold(@sum, |a, b| -> a + b)",
        c,
        regs,
        TypeRegistry::new(),
    )
    .await;
    assert_eq!(result.value, Value::Int(60));
}

// -- 8. Compiler pipeline pattern -----------------------------------
//
// Simulates: resolve_imports + parse_types (independent IO),
// then dependent passes that use both results.
//
// imports = fetch_a()           // "resolve imports" - IO
// types   = fetch_b()           // "parse types"     - IO, independent
// refs    = fetch_by(imports)   // "resolve refs"    - depends on imports
// checked = refs + types        // "type check"      - depends on refs + types
// extra   = fetch_c()           // "lint"            - independent of everything
// checked + extra
//
// Source order: imports, types, refs, extra. Pure work between them moves freely.

#[tokio::test]
async fn io_compiler_pipeline() {
    let i = Interner::new();
    // imports=100, types=200, refs=fetch_by(100)=1000, checked=1000+200=1200, extra=300
    // result = 1200 + 300 = 1500
    let result = run_script_with_externs(
        &i,
        "imports = fetch_a(); types = fetch_b(); refs = fetch_by(imports); checked = refs + types; extra = fetch_c(); checked + extra",
        ctx(&i, &[]),
        vec![io_registry()],
    ).await;
    assert_eq!(result.value, Value::Int(1500));
}

#[test]
fn io_compiler_pipeline_mir() {
    let (i, cr) = compile_io_script(
        "imports = fetch_a(); types = fetch_b(); refs = fetch_by(imports); checked = refs + types; extra = fetch_c(); checked + extra",
    );
    let (spawns, evals) = dump_and_positions("compiler_pipeline", &i, &cr);

    // 4 IO calls: fetch_a, fetch_b, fetch_by, fetch_c
    assert_eq!(spawns.len(), 4, "expected 4 spawns");
    assert_source_order(&spawns, &evals);
}

// =======================================================================
//  Move-only opaque value through an IO ExternFn (Spawn + Eval path)
// =======================================================================

#[derive(ExternType)]
struct Tok<I>(i64, std::marker::PhantomData<I>)
where
    I: acvus_extern::IdentityVar;

#[extern_fn(effect = pure)]
fn mk_tok<I>(_: &Interner) -> Tok<I>
where
    I: acvus_extern::IdentityVar,
{
    Tok(7, std::marker::PhantomData)
}

#[extern_fn]
fn consume_tok<I>(_: &Interner, tok: Tok<I>) -> i64
where
    I: acvus_extern::IdentityVar,
{
    tok.0
}

#[tokio::test]
async fn io_extern_consumes_move_only_opaque() {
    let i = Interner::new();
    let registry: ExternRegistry<AcvusRuntime> = extern_registry! {
        types: [Tok<_>],
        fns: [mk_tok, consume_tok],
    };

    let result = run_script_with_externs_and_types(
        &i,
        "t = mk_tok(); consume_tok(t)",
        ctx(&i, &[]),
        vec![registry],
        TypeRegistry::new(),
    )
    .await;
    assert_eq!(result.value, Value::Int(7));
}

#[tokio::test]
async fn io_inside_iterator_pipeline() {
    let i = Interner::new();
    let c = ctx(
        &i,
        &[(
            "items",
            Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        )],
    );
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    let result = run_script_with_externs_and_types(
        &i,
        "@items | iter | map(|x| -> fetch_by(x)) | collect",
        c,
        regs,
        TypeRegistry::new(),
    )
    .await;
    let Value::Extern(list) = result.value else {
        panic!("collect returns a List, got {:?}", result.value);
    };
    assert_eq!(
        list.downcast_ref::<Vec<Value>>().expect("List payload"),
        &vec![Value::Int(10), Value::Int(20), Value::Int(30)]
    );
}
