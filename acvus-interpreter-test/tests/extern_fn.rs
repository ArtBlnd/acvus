//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use acvus_extern::{ExternType, Owned, Registry, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, TokioExecutor, Value};
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
        acvus_ext::iterator_registry(),
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
        acvus_ext::vec_registry(),
        acvus_ext::iterator_registry(),
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

/// Adds `by` to the lent place and returns the new value (RFC-0018).
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

// -- Lent places (RFC-0018) -----------------------------------------
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

// =======================================================================
//  A captured parameter's effect reaches the capturing lambda (RFC-0014)
// =======================================================================

static TICKS: AtomicUsize = AtomicUsize::new(0);

#[extern_fn(effect = opaque)]
fn tick(x: i64) -> i64 {
    TICKS.fetch_add(1, Ordering::SeqCst);
    x
}

/// A lambda calling a parameter of its enclosing lambda takes the effect of
/// the function that parameter settles to: the captured function flows into
/// the type the body reads it at, so its effect is at most the reading's.
#[tokio::test]
async fn a_lambda_calling_a_captured_parameter_takes_its_effect() {
    let registry = || -> Registry<AcvusRuntime> {
        extern_registry! {
            ns: "t",
            fns: [tick],
        }
    };
    let programs = [
        ("let h = |g| -> (|x| -> g(x))(1); h(|y| -> tick(y))", 1),
        (
            "let h = |g| -> { let k = |x| -> g(x); k(1) }; h(|y| -> tick(y))",
            1,
        ),
        (
            "let h = |g| -> { (|x| -> g(x))(1); 0 }; h(|y| -> tick(y))",
            0,
        ),
        (
            "let each_any = |f| -> f(1); let h = |g| -> each_any(|x| -> g(x)); h(|y| -> tick(y))",
            1,
        ),
    ];
    for (source, value) in programs {
        let i = Interner::new();
        let before = TICKS.load(Ordering::SeqCst);
        let result =
            run_script_with_externs(&i, source, ctx(&i, vec![]), vec![registry()], Ty::I64).await;
        assert_eq!(result.value.as_int(), value, "{source}");
        assert_eq!(TICKS.load(Ordering::SeqCst) - before, 1, "{source}");
    }
}

// =======================================================================
//  A container of an extension type, handed to an extern
// =======================================================================

/// An extension type stored as its payload: a `Vec<Tag>` is kept as a
/// `Vec<Owned<Rt>>` of erased `i64`s, which is no `Vec<Tag>`, so no extern
/// borrows one as `&Vec<Tag>`, and no `Slice<Tag, _, Rt>` reads one;
/// `acvus-extern-macro`'s `borrowed_vec_of_converted` and
/// `slice_of_a_payload_type` compile-fail cases pin both refusals.
#[derive(ExternType)]
#[repr(transparent)]
struct Tag(i64);

type TagSlice<M, Rt> = acvus_extern::Slice<acvus_extern::Erased<Rt, Tag>, M, Rt>;

#[extern_fn(effect = pure)]
fn make_tags(n: i64) -> Vec<Tag> {
    (1..=n).map(Tag).collect()
}

#[extern_fn(effect = pure)]
fn sum_tag_slice<Rt>(
    ctx: &mut acvus_extern::Ctx<'_, Rt>,
    xs: TagSlice<acvus_extern::Shared, Rt>,
) -> i64
where
    Rt: acvus_extern::Runtime,
{
    let rt = ctx.rt;
    xs.with(|tags| tags.iter().map(|t| t.as_ref(rt).0).sum())
}

#[extern_fn(effect = pure)]
fn bump_tags<Rt>(ctx: &mut acvus_extern::Ctx<'_, Rt>, xs: TagSlice<acvus_extern::Mut, Rt>)
where
    Rt: acvus_extern::Runtime,
{
    let rt = ctx.rt;
    let mut xs = xs;
    xs.with(|tags| {
        for t in tags {
            t.as_mut(rt).0 += 10;
        }
    });
}

#[extern_fn(effect = pure)]
fn grow_tags<Rt>(
    ctx: &mut acvus_extern::Ctx<'_, Rt>,
    xs: TagSlice<acvus_extern::Shared, Rt>,
) -> Vec<Tag>
where
    Rt: acvus_extern::Runtime,
{
    let rt = ctx.rt;
    xs.with(|tags| {
        tags.iter()
            .map(|t| Tag(t.as_ref(rt).0 + 1))
            .chain([Tag(1)])
            .collect()
    })
}

/// By value, each element is materialized from its own value, and an
/// `async` declaration may take it, which it may not a slice.
#[extern_fn(effect = pure)]
async fn sum_tags(xs: Vec<Tag>) -> i64 {
    xs.iter().map(|t| t.0).sum()
}

fn tag_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "tag",
        types: [Tag],
        fns: [make_tags, sum_tag_slice, bump_tags, grow_tags, sum_tags],
    }
}

async fn run_tags(source: &str) -> i64 {
    let i = Interner::new();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(tag_registry());
    let result = run_script_with_externs(&i, source, ctx(&i, vec![]), regs, Ty::I64).await;
    result.value.as_int()
}

#[tokio::test]
async fn a_vec_of_an_extension_type_is_lent_as_a_slice() {
    assert_eq!(run_tags("let v = make_tags(3); sum_tag_slice(&v)").await, 6);
}

#[tokio::test]
async fn a_vec_of_an_extension_type_is_edited_through_a_mut_slice() {
    assert_eq!(
        run_tags("let v = make_tags(3); bump_tags(&mut v); sum_tag_slice(&v)").await,
        36
    );
}

/// [1] -> [2, 1] -> [3, 2, 1] -> [4, 3, 2, 1].
#[tokio::test]
async fn a_vec_is_reassigned_from_a_call_that_borrows_it() {
    assert_eq!(
        run_tags("let v = make_tags(1); for i in 0..3 { v = grow_tags(&v); } sum_tag_slice(&v)")
            .await,
        10
    );
}

#[tokio::test]
async fn a_vec_of_an_extension_type_is_read_by_value() {
    assert_eq!(run_tags("sum_tags(make_tags(3))").await, 6);
}

// =======================================================================
//  An argument written concrete is held
// =======================================================================

/// A map, a set, a deque and a derived extension type are each one box of
/// the Rust type their arguments name. The constructors are generic, so a
/// script's map is the box at its variables' run-time instantiation; a
/// declaration naming `i64` or `Pure` there names another box, and its
/// argument is held (`#i64`, `#Pure`), so the two meet as two types.
#[extern_fn(effect = pure)]
fn keys_at_pure<K, V, I, Rt>(_k: &mut acvus_ext::Keys<K, V, acvus_extern::Pure, I, Rt>) -> i64
where
    K: acvus_extern::Var<acvus_extern::kind::Type>,
    V: acvus_extern::Var<acvus_extern::kind::Type>,
    I: acvus_extern::Var<acvus_extern::kind::Identity>,
    Rt: acvus_extern::Runtime,
{
    1
}

#[extern_fn(effect = pure)]
fn keys_at_pure_by_value<K, V, I, Rt>(_k: acvus_ext::Keys<K, V, acvus_extern::Pure, I, Rt>) -> i64
where
    K: acvus_extern::Var<acvus_extern::kind::Type>,
    V: acvus_extern::Var<acvus_extern::kind::Type>,
    I: acvus_extern::Var<acvus_extern::kind::Identity>,
    Rt: acvus_extern::Runtime,
{
    2
}

#[extern_fn(effect = pure)]
fn map_at_ints_by_value<Rt>(_m: acvus_ext::HashMap<i64, i64, acvus_extern::Pure, Rt>) -> i64
where
    Rt: acvus_extern::Runtime,
{
    3
}

#[extern_fn(effect = pure)]
fn deque_at_ints_by_value(d: acvus_ext::Deque<i64>) -> i64 {
    d.iter().sum()
}

#[extern_fn(effect = pure)]
fn deque_of_ints(n: i64) -> acvus_ext::Deque<i64> {
    let mut d = acvus_ext::Deque::default();
    for x in 1..=n {
        d.push_back(x);
    }
    d
}

#[extern_fn(effect = pure)]
fn deque_at_ints_width(d: &acvus_ext::Deque<i64>) -> i64 {
    d.len() as i64
}

/// `HashMap<K, V, Pure, Rt>` with its key and value variables: the effect
/// alone is written concrete.
#[extern_fn(effect = pure)]
fn map_at_pure<K, V, Rt>(_m: &acvus_ext::HashMap<K, V, acvus_extern::Pure, Rt>) -> i64
where
    K: acvus_extern::Var<acvus_extern::kind::Type>,
    V: acvus_extern::Var<acvus_extern::kind::Type>,
    Rt: acvus_extern::Runtime,
{
    4
}

#[extern_fn(effect = pure)]
fn map_at_pure_by_value<K, V, Rt>(_m: acvus_ext::HashMap<K, V, acvus_extern::Pure, Rt>) -> i64
where
    K: acvus_extern::Var<acvus_extern::kind::Type>,
    V: acvus_extern::Var<acvus_extern::kind::Type>,
    Rt: acvus_extern::Runtime,
{
    5
}

fn held_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "held",
        fns: [
            keys_at_pure,
            keys_at_pure_by_value,
            map_at_ints_by_value,
            deque_at_ints_by_value,
            deque_of_ints,
            deque_at_ints_width,
            map_at_pure,
            map_at_pure_by_value,
        ],
    }
}

fn held_refusal(source: &str) -> String {
    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(&i, source).expect("parse error"),
    );
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(held_registry());
    match check_source(
        &i,
        ast,
        &FxHashMap::default(),
        regs,
        Ty::I64,
        acvus_mir::graph::optimize::Opt::Full,
        |_| {},
    ) {
        Ok(_) => panic!("the program was admitted: {source}"),
        Err(refusal) => refusal.messages.join(" | "),
    }
}

const MAP_OF_INTS: &str =
    "let m = hash_map_by(|k| -> hash(k), |a, b| -> a == b); insert(&mut m, 1, 10);";

#[test]
fn a_derived_type_at_a_known_effect_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal(&format!(
        "{MAP_OF_INTS} let k = keys(&m); keys_at_pure(&mut k)"
    ));
    assert!(
        messages.contains("expected &mut Keys<i64, i64, #Pure>, got &mut Keys<i64, i64, Pure>"),
        "{messages}"
    );
}

#[test]
fn a_derived_type_at_a_known_effect_by_value_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal(&format!(
        "{MAP_OF_INTS} let k = keys(&m); keys_at_pure_by_value(k)"
    ));
    assert!(
        messages.contains("expected Keys<i64, i64, #Pure>, got Keys<i64, i64, Pure>"),
        "{messages}"
    );
}

#[test]
fn a_map_at_concrete_types_by_value_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal(&format!("{MAP_OF_INTS} map_at_ints_by_value(m)"));
    assert!(
        messages.contains("expected HashMap<#i64, #i64, #Pure>, got HashMap<i64, i64, Pure>"),
        "{messages}"
    );
}

#[test]
fn a_deque_at_a_concrete_type_by_value_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal("let d = deque(); push_back(&mut d, 1); deque_at_ints_by_value(d)");
    assert!(
        messages.contains("expected Deque<#i64>, got Deque<i64>"),
        "{messages}"
    );
}

/// The effect alone written concrete: the map `hash_map_by` made is the box
/// at `E = ()`, and a declaration at `Pure` names the box at `Pure`.
#[test]
fn a_map_at_a_known_effect_by_reference_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal(&format!("{MAP_OF_INTS} map_at_pure(&m)"));
    assert!(
        messages.contains("expected &HashMap<i64, i64, #Pure>, got &HashMap<i64, i64, Pure>"),
        "{messages}"
    );
}

#[test]
fn a_map_at_a_known_effect_by_value_is_not_the_one_a_generic_constructor_made() {
    let messages = held_refusal(&format!("{MAP_OF_INTS} map_at_pure_by_value(m)"));
    assert!(
        messages.contains("expected HashMap<i64, i64, #Pure>, got HashMap<i64, i64, Pure>"),
        "{messages}"
    );
}

#[tokio::test]
async fn a_deque_made_and_borrowed_at_one_concrete_type_is_read_in_place() {
    let i = Interner::new();
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(held_registry());
    let result = run_script_with_externs(
        &i,
        "let d = deque_of_ints(4); deque_at_ints_width(&d) * 100 + deque_at_ints_by_value(d)",
        ctx(&i, vec![]),
        regs,
        Ty::I64,
    )
    .await;
    assert_eq!(result.value.as_int(), 410);
}

// =======================================================================
//  A held argument's representation follows its type part by part
// =======================================================================

/// A derived type whose payload names its argument: `Bag<T>` is stored as
/// a `Vec<T>`, so a generic constructor's bag is a `Vec<Owned<Rt>>`, a bag
/// of `(i64, U)` a `Vec<(i64, Owned<Rt>)>`, and a bag of `Erased<Rt, i64>` is
/// keyed at its canonical form, a `Vec<Owned<Rt>>`.
#[derive(ExternType)]
#[repr(transparent)]
#[extern_type(name = "Bag")]
struct Bag<T>(Vec<T>)
where
    T: acvus_extern::Var<acvus_extern::kind::Type>;

#[extern_fn(effect = pure)]
fn bag_of<T>(x: T) -> Bag<T>
where
    T: acvus_extern::Var<acvus_extern::kind::Type>,
{
    Bag(vec![x])
}

#[extern_fn(effect = pure)]
fn bag_width<T>(b: Bag<T>) -> i64
where
    T: acvus_extern::Var<acvus_extern::kind::Type>,
{
    i64::try_from(b.0.len()).expect("a bag shorter than i64::MAX")
}

#[extern_fn(effect = pure)]
fn erased_bag_of<Rt>(
    ctx: &mut acvus_extern::Ctx<'_, Rt>,
    x: i64,
) -> Bag<acvus_extern::Erased<Rt, i64>>
where
    Rt: acvus_extern::Runtime,
{
    Bag(vec![acvus_extern::Erased::new(ctx.rt, x)])
}

#[extern_fn(effect = pure)]
fn erased_bag_head<Rt>(b: Bag<acvus_extern::Erased<Rt, i64>>) -> i64
where
    Rt: acvus_extern::Runtime,
{
    b.0.first().map_or(-1, |x| x.get())
}

#[extern_fn(effect = pure)]
fn erased_bag_head_ref<Rt>(b: &Bag<acvus_extern::Erased<Rt, i64>>) -> i64
where
    Rt: acvus_extern::Runtime,
{
    b.0.first().map_or(-1, |x| x.get())
}

/// A bag whose argument is a composite holding a variable, read by its
/// written part.
#[extern_fn(effect = pure)]
fn pair_bag_first<U>(b: Bag<(i64, U)>) -> i64
where
    U: acvus_extern::Var<acvus_extern::kind::Type>,
{
    b.0.first().map_or(-1, |x| x.0)
}

/// A bag at a concrete composite, made where the Rust type is written.
#[extern_fn(effect = pure)]
fn pair_bag_of_ints(a: i64, b: i64) -> Bag<(i64, i64)> {
    Bag(vec![(a, b)])
}

/// A bag at the composite `pair_bag_first` reads, made where that Rust type
/// is written.
#[extern_fn(effect = pure)]
fn pair_bag_of<U>(a: i64, u: U) -> Bag<(i64, U)>
where
    U: acvus_extern::Var<acvus_extern::kind::Type>,
{
    Bag(vec![(a, u)])
}

/// A bag whose composite holds its variable at the other part.
#[extern_fn(effect = pure)]
fn int_second_bag_of<V>(v: V, b: i64) -> Bag<(V, i64)>
where
    V: acvus_extern::Var<acvus_extern::kind::Type>,
{
    Bag(vec![(v, b)])
}

/// A bag of deques: `Vec<Deque<Owned<Rt>>>`.
#[extern_fn(effect = pure)]
fn deque_bag_width<T>(b: Bag<acvus_ext::Deque<T>>) -> i64
where
    T: acvus_extern::Var<acvus_extern::kind::Type>,
{
    b.0.first().map_or(-1, |d| {
        i64::try_from(d.len()).expect("a deque shorter than i64::MAX")
    })
}

/// A bag of Rust arrays: `Vec<[i64; 2]>`, each array's two elements in
/// place.
#[extern_fn(effect = pure)]
fn rust_array_bag_of(a: i64) -> Bag<[i64; 2]> {
    Bag(vec![[a, a + 1]])
}

#[extern_fn(effect = pure)]
fn rust_array_bag_sum(b: Bag<[i64; 2]>) -> i64 {
    b.0.first().map_or(-1, |x| x[0] + x[1])
}

/// A bag of the language's arrays: `Vec<Arr<i64, ()>>`, each array's
/// elements in a buffer of their own.
#[extern_fn(effect = pure)]
fn array_bag_of<N>(a: acvus_extern::Arr<i64, N>) -> Bag<acvus_extern::Arr<i64, N>>
where
    N: acvus_extern::Var<acvus_extern::kind::Length>,
{
    Bag(vec![a])
}

#[extern_fn(effect = pure)]
fn array_bag_sum<N>(b: Bag<acvus_extern::Arr<i64, N>>) -> i64
where
    N: acvus_extern::Var<acvus_extern::kind::Length>,
{
    b.0.first().map_or(-1, |x| x.0.iter().sum())
}

fn erased_held_registry() -> Registry<AcvusRuntime> {
    extern_registry! {
        ns: "erased_held",
        types: [Bag<_>],
        fns: [
            bag_of,
            bag_width,
            erased_bag_of,
            erased_bag_head,
            erased_bag_head_ref,
            pair_bag_first,
            pair_bag_of_ints,
            pair_bag_of,
            int_second_bag_of,
            deque_bag_width,
            rust_array_bag_of,
            rust_array_bag_sum,
            array_bag_of,
            array_bag_sum,
        ],
    }
}

fn erased_held_regs() -> Vec<Registry<AcvusRuntime>> {
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(erased_held_registry());
    regs
}

async fn run_erased_held(source: &str) -> i64 {
    let i = Interner::new();
    let result =
        run_script_with_externs(&i, source, ctx(&i, vec![]), erased_held_regs(), Ty::I64).await;
    result.value.as_int()
}

fn erased_held_refusal(source: &str) -> String {
    erased_held_refusal_at(source, acvus_mir::graph::optimize::Opt::Full)
}

fn erased_held_refusal_at(source: &str, opt: acvus_mir::graph::optimize::Opt) -> String {
    let i = Interner::new();
    let ast = acvus_mir::graph::ParsedAst::Script(
        acvus_ast::parse_script(&i, source).expect("parse error"),
    );
    match check_source(
        &i,
        ast,
        &FxHashMap::default(),
        erased_held_regs(),
        Ty::I64,
        opt,
        |_| {},
    ) {
        Ok(_) => panic!("the program was admitted at {opt:?}: {source}"),
        Err(refusal) => refusal.messages.join(" | "),
    }
}

#[tokio::test]
async fn probe_erased_made_and_read_at_erased() {
    assert_eq!(
        run_erased_held("erased_bag_head(erased_bag_of(7))").await,
        7
    );
}

/// A generic constructor's box is `Vec<Owned>`, the canonical form of the
/// `Vec<Erased<Rt, i64>>` the declaration reads, so the two key one box.
#[tokio::test]
async fn probe_erased_by_value_from_a_generic_constructor() {
    assert_eq!(run_erased_held("erased_bag_head(bag_of(7))").await, 7);
}

/// As the by-value read, through a borrow of the box.
#[tokio::test]
async fn probe_erased_by_reference_from_a_generic_constructor() {
    assert_eq!(
        run_erased_held("let b = bag_of(7); erased_bag_head_ref(&b)").await,
        7
    );
}

/// The reverse: a box made at `Vec<Erased<Rt, i64>>` is keyed at `Vec<Owned>`,
/// which a generic declaration reads.
#[tokio::test]
async fn probe_erased_constructor_read_by_a_generic_declaration() {
    assert_eq!(run_erased_held("bag_width(erased_bag_of(7))").await, 1);
}

/// A generic constructor's bag is a `Vec<Owned>`, and a bag of `(i64, U)`
/// a `Vec<(i64, Owned)>`: the argument is uniform against `#(#i64, U)`.
#[test]
fn a_composite_holding_a_variable_is_not_what_a_generic_constructor_made() {
    let messages = erased_held_refusal("pair_bag_first(bag_of((1, 2)))");
    assert!(
        messages.contains("expected Bag<#(#i64, i64)>, got Bag<(i64, i64)>"),
        "{messages}"
    );
}

/// `Vec<(i64, i64)>` against `Vec<(i64, Owned)>`: the two meet at `#i64`
/// and part at `U`, which is uniform against `#i64`.
#[test]
fn a_composite_holding_a_variable_is_not_the_concrete_composite() {
    let messages = erased_held_refusal("pair_bag_first(pair_bag_of_ints(1, 2))");
    assert!(
        messages.contains("expected Bag<#(#i64, i64)>, got Bag<#(#i64, #i64)>"),
        "{messages}"
    );
}

/// `Vec<(i64, Owned)>` against `Vec<(Owned, i64)>`: each has one uniform
/// part, at a different place.
#[test]
fn two_composites_holding_a_variable_at_different_parts_are_two_types() {
    let messages = erased_held_refusal("pair_bag_first(int_second_bag_of(1, 2))");
    assert!(
        messages.contains("expected Bag<#(#i64, i64)>, got Bag<#(i64, #i64)>"),
        "{messages}"
    );
}

/// `Vec<Owned>` against `Vec<Deque<Owned>>`.
#[test]
fn a_bag_of_deques_is_not_what_a_generic_constructor_made() {
    let messages = erased_held_refusal("deque_bag_width(bag_of(deque()))");
    assert!(
        messages.contains("expected Bag<#Deque<_>>, got Bag<Deque<_>>"),
        "{messages}"
    );
}

/// Made and read at one composite: the reader takes the written part out of
/// the `Vec<(i64, Owned)>` the maker boxed.
#[tokio::test]
async fn a_composite_holding_a_variable_is_read_by_its_parts_where_it_was_made() {
    assert_eq!(
        run_erased_held("pair_bag_first(pair_bag_of(1, 2))").await,
        1
    );
}

// =======================================================================
//  A Rust array and the language's array are two boxes
// =======================================================================

/// Refuses `source` at both levels, with `expected` among its messages.
fn array_refused_with(source: &str, expected: &str) {
    for opt in [
        acvus_mir::graph::optimize::Opt::None,
        acvus_mir::graph::optimize::Opt::Full,
    ] {
        let messages = erased_held_refusal_at(source, opt);
        assert!(messages.contains(expected), "at {opt:?}: {messages}");
    }
}

/// `Vec<Arr<i64, ()>>` against `Vec<[i64; 2]>`: one acvus type, two Rust
/// heads, so `#Array` and `#[_; 2]` are two trees.
#[test]
fn a_bag_of_arrays_is_not_a_bag_of_rust_arrays() {
    array_refused_with(
        "rust_array_bag_sum(array_bag_of([1, 2]))",
        "expected Bag<#[#i64; 2]>, got Bag<#Array<#i64, 2>>",
    );
}

/// The reverse: `Vec<[i64; 2]>` against `Vec<Arr<i64, ()>>`.
#[test]
fn a_bag_of_rust_arrays_is_not_a_bag_of_arrays() {
    array_refused_with(
        "array_bag_sum(rust_array_bag_of(1))",
        "expected Bag<#Array<#i64, 2>>, got Bag<#[#i64; 2]>",
    );
}

/// Run, the program is refused before any box is read at the other Rust
/// type.
#[tokio::test]
#[should_panic(expected = "expected Bag<#[#i64; 2]>, got Bag<#Array<#i64, 2>>")]
async fn a_bag_of_arrays_does_not_run_as_a_bag_of_rust_arrays() {
    run_erased_held("rust_array_bag_sum(array_bag_of([1, 2]))").await;
}

#[tokio::test]
#[should_panic(expected = "expected Bag<#Array<#i64, 2>>, got Bag<#[#i64; 2]>")]
async fn a_bag_of_rust_arrays_does_not_run_as_a_bag_of_arrays() {
    run_erased_held("array_bag_sum(rust_array_bag_of(1))").await;
}

/// Each array made and read at its own Rust head.
#[tokio::test]
async fn an_array_is_read_at_the_rust_head_it_was_made_at() {
    assert_eq!(
        run_erased_held("rust_array_bag_sum(rust_array_bag_of(1))").await,
        3
    );
    assert_eq!(
        run_erased_held("array_bag_sum(array_bag_of([1, 2]))").await,
        3
    );
}
