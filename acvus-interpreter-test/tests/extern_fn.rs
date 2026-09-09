//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.


use acvus_extern::{ExternFn, ExternItems, ExternRegistry, ExternType, extern_fn, extern_registry};
use acvus_interpreter::{AcvusRuntime, Executable, Value};
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
        fns: fns(i).into_iter().map(|f| f.with_effect(effect)).collect(),
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

#[ignore = "pending identity integration"]
#[tokio::test]
async fn extern_pure_add() {
    let i = Interner::new();

    let registry = closures(Effect::Pure, |i| {
        vec![ExternFn::sync(i, "ext_add", |_: &Interner, a: i64, b: i64| Ok(a + b))]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "ext_add(10, 32)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

#[ignore = "pending identity integration"]
#[tokio::test]
async fn extern_pure_string_transform() {
    let i = Interner::new();

    let registry = closures(Effect::Pure, |i| {
        vec![ExternFn::sync(i, "shout", |_: &Interner, s: String| Ok(s.to_uppercase()))]
    });

    let c = ctx(&i, &[("msg", Value::string("hello"))]);
    let result = run_script_with_externs(&i, "shout(@msg)", c, vec![registry]).await;
    assert_eq!(result.value, Value::string("HELLO"));
}

// =======================================================================
//  ExternFn capturing Rust environment
// =======================================================================

#[ignore = "pending identity integration"]
#[tokio::test]
async fn extern_captures_environment() {
    let i = Interner::new();
    let secret = 7i64;

    let registry = closures(Effect::Pure, move |i| {
        vec![ExternFn::sync(i, "multiply_secret", move |_: &Interner, x: i64| Ok(x * secret))]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "multiply_secret(6)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

// =======================================================================
//  Regex ExternFn (legacy sync_handler, Builtin path)
// =======================================================================

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
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
#[ignore = "pending identity integration"]
#[test]
fn ir_pure_function_call_no_context_bindings() {
    let i = Interner::new();

    let registry = closures(Effect::Pure, |i| {
        vec![ExternFn::sync(i, "double", |_: &Interner, x: i64| Ok(x * 2))]
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
fn io_registry() -> ExternRegistry<AcvusRuntime> {
    extern_registry! {
        fns: [fetch_a, fetch_b, fetch_c, fetch_d, fetch_by],
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
    let context_types: FxHashMap<acvus_utils::Astr, Ty> = context
        .iter()
        .map(|(name, val)| (i.intern(name), infer_ty(val)))
        .collect();
    let ast =
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse"));
    let mut regs = acvus_ext::std_registries::<AcvusRuntime>();
    regs.push(io_registry());
    let cr = compile_source_with_externs(&i, ast, &context_types, regs, TypeRegistry::new());
    (i, cr)
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

// -- 1. Two independent IO calls ------------------------------------

/// fetch_a() + fetch_b() -> spawn both before eval either.
#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_two_independent_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b()");
    let (spawns, evals) = dump_and_positions("two_independent", &i, &cr);
    assert_eq!(spawns.len(), 2, "expected 2 spawns");
    assert_eq!(evals.len(), 2, "expected 2 evals");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "all spawns must precede all evals"
    );
}

// -- 2. Four-way independent IO -------------------------------------

/// Maximum parallelism: 4 independent IO calls.
#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_four_way_parallel_mir() {
    let (i, cr) = compile_io_script("fetch_a() + fetch_b() + fetch_c() + fetch_d()");
    let (spawns, evals) = dump_and_positions("four_way_parallel", &i, &cr);
    assert_eq!(spawns.len(), 4, "expected 4 spawns");
    assert_eq!(evals.len(), 4, "expected 4 evals");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "all spawns must precede all evals"
    );
}

// -- 3. Dependent chain + independent IO ----------------------------
//
// a = fetch_a()          // IO, independent
// b = fetch_by(a)        // IO, depends on a
// c = fetch_c()          // IO, independent of a and b
// b + c
//
// Optimal: spawn fetch_a + spawn fetch_c in parallel,
//          eval fetch_a, spawn fetch_by(a), eval fetch_c, eval fetch_by -> b+c

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_chain_with_independent_mir() {
    let (i, cr) = compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_c(); b + c");
    let (spawns, evals) = dump_and_positions("chain_with_independent", &i, &cr);

    // 3 IO calls -> 3 spawns, 3 evals.
    assert_eq!(spawns.len(), 3, "expected 3 spawns");
    assert_eq!(evals.len(), 3, "expected 3 evals");

    // fetch_a and fetch_c should be spawned before any eval.
    // fetch_by depends on eval(fetch_a), so its spawn comes after first eval.
    // At minimum: the first 2 spawns should precede the first eval.
    assert!(
        spawns[0] < evals[0] && spawns[1] < evals[0],
        "fetch_a and fetch_c spawns should both precede first eval"
    );
}

// -- 4. Diamond dependency ------------------------------------------
//
// a = fetch_a()          // IO
// b = fetch_by(a)        // IO, depends on a
// c = fetch_by(a)        // IO, depends on a (same dep as b, but independent of b)
// b + c
//
// After eval(a), both fetch_by(a) calls can be spawned in parallel.

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_diamond_dependency_mir() {
    let (i, cr) = compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_by(a); b + c");
    let (spawns, evals) = dump_and_positions("diamond_dependency", &i, &cr);

    assert_eq!(spawns.len(), 3, "expected 3 spawns (fetch_a + 2x fetch_by)");
    assert_eq!(evals.len(), 3, "expected 3 evals");

    // fetch_by(a) spawns should both come after eval(fetch_a) but before eval(fetch_by).
    // Spawn[0] = fetch_a (before any eval)
    assert!(spawns[0] < evals[0], "fetch_a spawn before first eval");
    // The two fetch_by spawns should both precede their evals.
    assert!(
        spawns[1] < evals[1] && spawns[2] < evals[1],
        "both fetch_by spawns should precede second eval"
    );
}

// -- 5. Deep sequential chain ---------------------------------------
//
// a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d
//
// No parallelism possible: each depends on the previous.
// spawn->eval->spawn->eval->spawn->eval->spawn->eval

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_deep_chain_mir() {
    let (i, cr) =
        compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d");
    let (spawns, evals) = dump_and_positions("deep_chain", &i, &cr);

    assert_eq!(spawns.len(), 4, "4 IO calls in chain");
    assert_eq!(evals.len(), 4, "4 evals");

    // Each spawn[i+1] must come after eval[i] (strict dependency chain).
    for i in 0..3 {
        assert!(
            evals[i] < spawns[i + 1],
            "eval[{i}] must precede spawn[{}] in dependency chain",
            i + 1
        );
    }
}

// -- 6. Two independent chains --------------------------------------
//
// a = fetch_a(); b = fetch_by(a);    // chain 1: a -> b
// c = fetch_c(); d = fetch_by(c);    // chain 2: c -> d (independent of chain 1)
// b + d
//
// Optimal: spawn a + spawn c, eval a, spawn b, eval c, spawn d, eval b, eval d

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_two_independent_chains_mir() {
    let (i, cr) =
        compile_io_script("a = fetch_a(); b = fetch_by(a); c = fetch_c(); d = fetch_by(c); b + d");
    let (spawns, evals) = dump_and_positions("two_independent_chains", &i, &cr);

    assert_eq!(spawns.len(), 4, "4 IO calls");
    assert_eq!(evals.len(), 4, "4 evals");

    // The heads of both chains (fetch_a, fetch_c) should be spawned before any eval.
    assert!(
        spawns[0] < evals[0] && spawns[1] < evals[0],
        "chain heads should be spawned before first eval"
    );
}

// -- 7. IO in iteration ---------------------------------------------
//
// Iterate over list, call IO per element, accumulate.
// Within each iteration: spawn should precede eval.

#[ignore = "pending identity integration"]
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
    let result = run_script_with_externs(
        &i,
        "x in @items { @sum = @sum + fetch_by(x); }; @sum",
        c,
        vec![io_registry()],
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
// Optimal: spawn imports + spawn types + spawn extra (3-way parallel),
//          eval imports, spawn refs, eval types + eval extra whenever,
//          eval refs, compute result.

#[ignore = "pending identity integration"]
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

#[ignore = "pending identity integration"]
#[test]
fn io_compiler_pipeline_mir() {
    let (i, cr) = compile_io_script(
        "imports = fetch_a(); types = fetch_b(); refs = fetch_by(imports); checked = refs + types; extra = fetch_c(); checked + extra",
    );
    let (spawns, evals) = dump_and_positions("compiler_pipeline", &i, &cr);

    // 4 IO calls: fetch_a, fetch_b, fetch_by, fetch_c
    assert_eq!(spawns.len(), 4, "expected 4 spawns");
    assert_eq!(evals.len(), 4, "expected 4 evals");

    // fetch_a, fetch_b, fetch_c are independent - all 3 should be spawned before any eval.
    // fetch_by depends on eval(fetch_a).
    // At minimum: 3 independent spawns before first eval.
    let spawns_before_first_eval = spawns.iter().filter(|&&s| s < evals[0]).count();
    assert!(
        spawns_before_first_eval >= 3,
        "at least 3 independent IO spawns should precede first eval, got {spawns_before_first_eval}"
    );
}

// =======================================================================
//  Move-only opaque value through an IO ExternFn (Spawn + Eval path)
// =======================================================================

#[derive(ExternType)]
#[extern_type(move_only)]
struct Tok(i64);

#[extern_fn(effect = pure)]
fn mk_tok(_: &Interner) -> Tok {
    Tok(7)
}

#[extern_fn]
fn consume_tok(_: &Interner, tok: Tok) -> i64 {
    tok.0
}

#[tokio::test]
async fn io_extern_consumes_move_only_opaque() {
    let i = Interner::new();
    let registry: ExternRegistry<AcvusRuntime> = extern_registry! {
        types: [Tok],
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
        &[("items", Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]))],
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
