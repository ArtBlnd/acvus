//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{ExternType, Owned, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Executable, TokioExecutor, Value};
use acvus_interpreter_test::*;
use acvus_mir::ir::InstKind;
use acvus_mir::ty::{LenTerm, ObjectTy, Ty};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn ctx(i: &Interner, entries: Vec<(&str, TypedValue)>) -> Context {
    entries
        .into_iter()
        .map(|(name, val)| (i.intern(name), val))
        .collect()
}

fn int(n: i64) -> TypedValue {
    typed(Ty::I64, Value::int(n))
}

fn bool_(b: bool) -> TypedValue {
    typed(Ty::Bool, Value::bool_(b))
}

fn string(s: &str) -> TypedValue {
    typed(Ty::String, Value::string(s))
}

fn ints(xs: &[i64]) -> TypedValue {
    typed(
        Ty::Array(Box::new(Ty::I64), LenTerm::Known(xs.len())),
        Value::array(
            xs.iter()
                .map(|&x| Owned::from_value(Value::int(x)))
                .collect(),
        ),
    )
}

fn assert_str(v: &Value, expected: &str) {
    assert!(v.is_string(), "expected a String, got {v:?}");
    // SAFETY: the witness is String.
    assert_eq!(unsafe { v.as_str() }, expected);
}

// =======================================================================
//  Pure ExternFn
// =======================================================================

#[extern_fn(effect = pure)]
fn ext_add(a: i64, b: i64) -> i64 {
    a + b
}

#[tokio::test]
async fn extern_pure_add() {
    let i = Interner::new();
    let registry: Registry<AcvusRuntime> = extern_registry! {
        ns: "t",
        fns: [ext_add],
    };

    let c = ctx(&i, vec![]);
    let result = run_script_with_externs(&i, "ext_add(10, 32)", c, vec![registry], Ty::I64).await;
    assert_eq!(result.value.as_int(), 42);
}

#[extern_fn(effect = pure)]
fn shout(s: String) -> String {
    s.to_uppercase()
}

#[tokio::test]
async fn extern_pure_string_transform() {
    let i = Interner::new();
    let registry: Registry<AcvusRuntime> = extern_registry! {
        ns: "t",
        fns: [shout],
    };

    let c = ctx(&i, vec![("msg", string("hello"))]);
    let result = run_script_with_externs(&i, "shout(@msg)", c, vec![registry], Ty::String).await;
    assert_str(&result.value, "HELLO");
}

// =======================================================================
//  ExternFn holding Rust state (RFC-0021)
// =======================================================================

#[extern_fn(effect = pure)]
fn multiply_secret(#[state] secret: &i64, x: i64) -> i64 {
    x * secret
}

#[tokio::test]
async fn extern_captures_environment() {
    let i = Interner::new();
    let secret = 7i64;

    let registry: Registry<AcvusRuntime> = extern_registry! {
        ns: "t",
        fns: [multiply_secret(secret)],
    };

    let c = ctx(&i, vec![]);
    let result =
        run_script_with_externs(&i, "multiply_secret(6)", c, vec![registry], Ty::I64).await;
    assert_eq!(result.value.as_int(), 42);
}

// =======================================================================
//  Regex ExternFn
// =======================================================================

#[tokio::test]
async fn regex_match_via_extern() {
    let i = Interner::new();

    let registries = vec![
        acvus_ext::regex_registry(),
        acvus_ext::conversion_registry(),
    ];
    let c = ctx(&i, vec![("text", string("hello world 42"))]);
    let result = run_script_mode_with_externs(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") { is_match(&re, &@text) } else { false }"#,
        c,
        registries,
        Ty::Bool,
    )
    .await;
    assert!(result.value.as_bool());
}

#[tokio::test]
async fn regex_find_via_extern() {
    let i = Interner::new();

    let registries = vec![
        acvus_ext::regex_registry(),
        acvus_ext::conversion_registry(),
    ];
    let c = ctx(&i, vec![("text", string("price is 42 dollars"))]);
    let result = run_script_mode_with_externs(
        &i,
        r#"if let Ok(re) = regex("[0-9]+") {
             if let Some(m) = find(&re, &@text) { m.text } else { "no match".to_string() }
           } else { "?".to_string() }"#,
        c,
        registries,
        Ty::String,
    )
    .await;
    assert_str(&result.value, "42");
}

// =======================================================================
//  IR verification: FunctionCall has correct context_uses/context_defs
// =======================================================================

#[extern_fn(effect = pure)]
fn double(x: i64) -> i64 {
    x * 2
}

/// Pure ExternFn should have empty context_uses/context_defs in IR.
#[test]
fn ir_pure_function_call_no_context_bindings() {
    let i = Interner::new();

    let registry: Registry<AcvusRuntime> = extern_registry! {
        ns: "t",
        fns: [double],
    };

    let context_types: FxHashMap<acvus_utils::Astr, Ty> = FxHashMap::default();

    let source = "double(21)";
    let cr = compile_source_with_externs(
        &i,
        acvus_mir::graph::ParsedAst::Script(
            acvus_ast::parse_script(&i, source).expect("parse error"),
        ),
        &context_types,
        vec![registry],
        Ty::I64,
    );

    let module = cr.modules.get(&cr.entry_qref).unwrap();

    let call_insts: Vec<_> = module
        .main
        .insts
        .iter()
        .filter(|inst| {
            matches!(
                &inst.kind,
                InstKind::FunctionCall {
                    callee: acvus_mir::ir::Callee::Extern { id, .. },
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
fn fetch_a() -> i64 {
    100
}

#[extern_fn]
fn fetch_b() -> i64 {
    200
}

#[extern_fn]
fn fetch_c() -> i64 {
    300
}

#[extern_fn]
fn fetch_d() -> i64 {
    400
}

#[extern_fn]
fn fetch_by(x: i64) -> i64 {
    x * 10
}

/// Four independent Opaque fetches and one parameterized.
/// A fresh draw: two draws in either order are the same program.
#[extern_fn(effect = idempotent, commutative)]
fn draw_a() -> i64 {
    5
}

#[extern_fn(effect = idempotent, commutative)]
fn draw_b() -> i64 {
    7
}

/// Adds `by` to the lent place and returns the new value (RFC-0015).
#[extern_fn(effect = pure)]
fn bump(n: &mut i64, by: i64) -> i64 {
    *n += by;
    *n
}

fn io_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "io",
        fns: [fetch_a, fetch_b, fetch_c, fetch_d, fetch_by, draw_a, draw_b, bump],
    }
}

/// Compile a script with io_registry, return (CompileResult, entry MirModule ref).
fn compile_io_script(source: &str, ret: Ty) -> (Interner, CompileResult) {
    let i = Interner::new();
    let ast =
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    let cr = compile_io_parsed(&i, ast, Context::default(), ret);
    (i, cr)
}

/// Script mode (`anyorder`, `while`, `let`) with io_registry.
fn compile_io_script_mode(
    source: &str,
    context: Vec<(&str, TypedValue)>,
    ret: Ty,
) -> (Interner, CompileResult) {
    let i = Interner::new();
    let ast =
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    let context = ctx(&i, context);
    let cr = compile_io_parsed(&i, ast, context, ret);
    (i, cr)
}

fn compile_io_parsed(
    i: &Interner,
    ast: acvus_mir::graph::ParsedAst,
    context: Context,
    ret: Ty,
) -> CompileResult {
    let context_types: FxHashMap<acvus_utils::Astr, Ty> = context
        .into_iter()
        .map(|(name, val)| (name, val.ty))
        .collect();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    compile_source_with_externs(i, ast, &context_types, regs, ret)
}

async fn run_io_script_mode(source: &str, context: Vec<(&str, TypedValue)>, ret: Ty) -> Value {
    let i = Interner::new();
    run_io_script_mode_on(&i, source, context, ret).await
}

async fn run_io_script_mode_on(
    i: &Interner,
    source: &str,
    context: Vec<(&str, TypedValue)>,
    ret: Ty,
) -> Value {
    let i = i.clone();
    let ast =
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    run_parsed_with_externs(&i, ast, ctx(&i, context), regs, ret, |_| {})
        .await
        .value
}

/// Dump MIR and return (spawn_positions, eval_positions) for assertion.
fn dump_and_positions(label: &str, i: &Interner, cr: &CompileResult) -> (Vec<usize>, Vec<usize>) {
    let module = cr.modules.get(&cr.entry_qref).unwrap();
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
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 300);
}

#[test]
fn io_two_independent_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b()", Ty::I64);
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
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 1000);
}

#[test]
fn io_four_way_parallel_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b() + fetch_c() + fetch_d()", Ty::I64);
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
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_c(); b + c",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 1300);
}

#[test]
fn io_chain_with_independent_mir() {
    let (i, cr) = compile_io_script(
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_c(); b + c",
        Ty::I64,
    );
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
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_by(a); b + c",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 2000);
}

#[test]
fn io_diamond_dependency_mir() {
    let (i, cr) = compile_io_script(
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_by(a); b + c",
        Ty::I64,
    );
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
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_by(b); let d = fetch_by(c); d",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 100000);
}

#[test]
fn io_deep_chain_mir() {
    let (i, cr) = compile_io_script(
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_by(b); let d = fetch_by(c); d",
        Ty::I64,
    );
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
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_c(); let d = fetch_by(c); b + d",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 4000);
}

#[test]
fn io_two_independent_chains_mir() {
    let (i, cr) = compile_io_script(
        "let a = fetch_a(); let b = fetch_by(a); let c = fetch_c(); let d = fetch_by(c); b + d",
        Ty::I64,
    );
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
    let v = run_io_script_mode("bump(&mut @n, 5); @n", vec![("n", int(10))], Ty::I64).await;
    assert_eq!(v.as_int(), 15);
}

#[tokio::test]
async fn a_local_lent_mutably_comes_back_changed() {
    let v = run_io_script_mode(
        "let x = 1; bump(&mut x, 2); bump(&mut x, 3); x",
        vec![],
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 6);
}

#[tokio::test]
async fn a_field_of_a_context_is_a_place() {
    let i = Interner::new();
    let n = i.intern("n");
    let a = typed(
        Ty::Object(ObjectTy::written(FxHashMap::from_iter([(n, Ty::I64)]))),
        Value::object_by_name(&i, [(n, Owned::from_value(Value::int(1)))]),
    );
    let v = run_io_script_mode_on(&i, "bump(&mut @a.n, 1); @a.n", vec![("a", a)], Ty::I64).await;
    assert_eq!(v.as_int(), 2);
}

#[tokio::test]
async fn the_return_value_of_a_lending_call_is_free() {
    let v = run_io_script_mode(
        "let x = 40; let y = bump(&mut x, 2); x + y",
        vec![],
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 84);
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
    let result = run_script_with_externs(
        &i,
        "draw_a() + draw_b()",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 12);
}

#[test]
fn commutative_run_is_issued_together_mir() {
    let (i, cr) = compile_io_script("draw_a() + draw_b()", Ty::I64);
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
    let (i, cr) = compile_io_script("draw_a() + fetch_a() + draw_b()", Ty::I64);
    let (spawns, evals) = dump_and_positions("broken_run", &i, &cr);
    assert_eq!(spawns.len(), 3, "expected 3 spawns");
    assert_source_order(&spawns, &evals);
    assert!(!has_merge(&cr), "no run, no merge");
}

#[test]
fn a_commutative_call_after_a_branch_is_issued_with_the_one_before_mir() {
    let (i, cr) = compile_io_script_mode(
        "@a = draw_a(); if @c { @x = 1; }; @b = draw_b(); @a + @b",
        vec![
            ("a", int(0)),
            ("b", int(0)),
            ("c", bool_(true)),
            ("x", int(0)),
        ],
        Ty::I64,
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
    let m = cr.modules.get(&cr.entry_qref).unwrap();
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
        vec![("a", int(0)), ("b", int(0))],
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 300);
}

#[test]
fn anyorder_block_issues_its_calls_together_mir() {
    let (i, cr) = compile_io_script_mode(
        "anyorder { @a = fetch_a(); @b = fetch_b(); } @a + @b",
        vec![("a", int(0)), ("b", int(0))],
        Ty::I64,
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
        vec![("a", int(0)), ("b", int(0)), ("c", int(0))],
        Ty::I64,
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
        vec![("n", int(3)), ("s", int(0))],
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 300);
}

#[test]
fn a_loop_inside_anyorder_merges_per_iteration_mir() {
    let (i, cr) = compile_io_script_mode(
        "anyorder { while @n > 0 { @s = @s + fetch_a(); @n = @n - 1; } } @s",
        vec![("n", int(3)), ("s", int(0))],
        Ty::I64,
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
    in_flight: AtomicUsize,
    max: AtomicUsize,
}

impl Probe {
    fn new() -> Arc<Self> {
        Arc::new(Probe {
            in_flight: AtomicUsize::new(0),
            max: AtomicUsize::new(0),
        })
    }

    fn max(&self) -> usize {
        self.max.load(Ordering::SeqCst)
    }

    async fn hold(&self) -> i64 {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(PROBE_HOLD).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        100
    }
}

#[extern_fn(name = "probe_a", effect = opaque)]
async fn probe_a_opaque(#[state] p: &Arc<Probe>) -> i64 {
    p.hold().await
}

#[extern_fn(name = "probe_b", effect = opaque)]
async fn probe_b_opaque(#[state] p: &Arc<Probe>) -> i64 {
    p.hold().await
}

#[extern_fn(name = "probe_a", effect = idempotent, commutative)]
async fn probe_a_commutative(#[state] p: &Arc<Probe>) -> i64 {
    p.hold().await
}

#[extern_fn(name = "probe_b", effect = idempotent, commutative)]
async fn probe_b_commutative(#[state] p: &Arc<Probe>) -> i64 {
    p.hold().await
}

fn opaque_probes(p: &Arc<Probe>) -> Registry<AcvusRuntime> {
    let p = Arc::clone(p);
    extern_registry! {
        ns: "probe",
        fns: [probe_a_opaque(Arc::clone(&p)), probe_b_opaque(Arc::clone(&p))],
    }
}

fn commutative_probes(p: &Arc<Probe>) -> Registry<AcvusRuntime> {
    let p = Arc::clone(p);
    extern_registry! {
        ns: "probe",
        fns: [probe_a_commutative(Arc::clone(&p)), probe_b_commutative(Arc::clone(&p))],
    }
}

async fn run_on_tokio(
    source: &str,
    script_mode: bool,
    context: Vec<(&str, TypedValue)>,
    registry: Registry<AcvusRuntime>,
    ret: Ty,
) -> Value {
    let i = Interner::new();
    let script = if script_mode {
        acvus_ast::parse_script(&i, source).expect("parse")
    } else {
        acvus_ast::parse_script(&i, source).expect("parse")
    };
    let ast = acvus_mir::graph::ParsedAst::Script(script);
    run_parsed_on(
        &i,
        ast,
        ctx(&i, context),
        vec![registry],
        ret,
        |_| {},
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
        vec![("a", int(0)), ("b", int(0))],
        opaque_probes(&probe),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 200);
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
        vec![("a", int(0)), ("b", int(0))],
        opaque_probes(&probe),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 200);
    assert_eq!(probe.max(), 2, "both calls are in flight at once");
}

#[tokio::test]
async fn commutative_calls_overlap_across_a_branch() {
    let probe = Probe::new();
    let v = run_on_tokio(
        "@a = probe_a(); if @c { @x = 1; }; @b = probe_b(); @a + @b",
        true,
        vec![
            ("a", int(0)),
            ("b", int(0)),
            ("c", bool_(true)),
            ("x", int(0)),
        ],
        commutative_probes(&probe),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 200);
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
        vec![],
        commutative_probes(&probe),
        Ty::I64,
    )
    .await;
    assert_eq!(v.as_int(), 200);
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
    let c = ctx(&i, vec![("items", ints(&[1, 2, 3])), ("sum", int(0))]);
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    let result = run_script_with_externs_and_types(
        &i,
        "as_iter(&@items) | map(|x| -> fetch_by(*x)) | fold(@sum, |a, b| -> a + b)",
        c,
        regs,
        Ty::I64,
        |_| {},
    )
    .await;
    assert_eq!(result.value.as_int(), 60);
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
        "let imports = fetch_a(); let types = fetch_b(); let refs = fetch_by(imports); let checked = refs + types; let extra = fetch_c(); checked + extra",
        ctx(&i, vec![]),
        vec![io_registry()],
        Ty::I64,
    ).await;
    assert_eq!(result.value.as_int(), 1500);
}

#[test]
fn io_compiler_pipeline_mir() {
    let (i, cr) = compile_io_script(
        "let imports = fetch_a(); let types = fetch_b(); let refs = fetch_by(imports); let checked = refs + types; let extra = fetch_c(); checked + extra",
        Ty::I64,
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
#[repr(transparent)]
struct Tok<I>(i64, std::marker::PhantomData<I>)
where
    I: acvus_extern::Var<acvus_extern::kind::Identity>;

#[extern_fn(effect = pure)]
fn mk_tok<I>() -> Tok<I>
where
    I: acvus_extern::Var<acvus_extern::kind::Identity>,
{
    Tok(7, std::marker::PhantomData)
}

#[extern_fn]
fn consume_tok<I>(tok: Tok<I>) -> i64
where
    I: acvus_extern::Var<acvus_extern::kind::Identity>,
{
    tok.0
}

#[tokio::test]
async fn io_extern_consumes_move_only_opaque() {
    let i = Interner::new();
    let registry: Registry<AcvusRuntime> = extern_registry! {
        ns: "tok",
        types: [Tok<_>],
        fns: [mk_tok, consume_tok],
    };

    let result = run_script_with_externs_and_types(
        &i,
        "let t = mk_tok(); consume_tok(t)",
        ctx(&i, vec![]),
        vec![registry],
        Ty::I64,
        |_| {},
    )
    .await;
    assert_eq!(result.value.as_int(), 7);
}

#[tokio::test]
async fn io_inside_iterator_pipeline() {
    let i = Interner::new();
    let c = ctx(&i, vec![("items", ints(&[1, 2, 3]))]);
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    let result = run_script_with_externs_and_types(
        &i,
        "as_iter(&@items) | map(|x| -> fetch_by(*x)) | collect",
        c,
        regs,
        acvus_extern::vec_ty(&i, Ty::I64),
        |_| {},
    )
    .await;
    // SAFETY: `collect` returns a `Vec<T>`, and the store of an owned
    // runtime value is `Owned<AcvusRuntime>` (RFC-0048).
    let list: Vec<Owned<AcvusRuntime>> = unsafe { result.value.materialize() };
    let items: Vec<i64> = list.iter().map(|v| v.as_int()).collect();
    assert_eq!(items, vec![10, 20, 30]);
}
