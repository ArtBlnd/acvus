//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.

use std::collections::BTreeSet;

use acvus_interpreter::{Defs, Executable, ExternFnBuilder, ExternRegistry, Uses, Value};
use acvus_interpreter_test::*;
use acvus_mir::graph::{Constraint, FnConstraint, QualifiedRef, Signature};
use acvus_mir::ir::InstKind;
use acvus_mir::ty::{Effect, EffectSet, EffectTarget, Hint, Param, Ty, TypeRegistry};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
        hint: None,
    }
}

fn sig_effect(interner: &Interner, params: Vec<Ty>, ret: Ty, effect: Effect) -> FnConstraint {
    sig_effect_hint(interner, params, ret, effect, None)
}

fn sig_effect_hint(
    interner: &Interner,
    params: Vec<Ty>,
    ret: Ty,
    effect: Effect,
    hint: Option<Hint>,
) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature {
            params: named.clone(),
        }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect,
        }),
        effect: None,
        hint,
    }
}

fn ctx(i: &Interner, entries: &[(&str, Value)]) -> FxHashMap<acvus_utils::Astr, Value> {
    entries
        .iter()
        .map(|(name, val)| (i.intern(name), val.clone()))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Pure ExternFn (no context)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_pure_add() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new("ext_add", sig(interner, vec![Ty::Int, Ty::Int], Ty::Int))
                .handler(
                    |_interner: &Interner, (a, b): (i64, i64), Uses(()): Uses<()>| {
                        Ok((a + b, Defs(())))
                    },
                ),
        ]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "ext_add(10, 32)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

#[tokio::test]
async fn extern_pure_string_transform() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new("shout", sig(interner, vec![Ty::String], Ty::String)).handler(
                |_interner: &Interner, (s,): (String,), Uses(()): Uses<()>| {
                    Ok((s.to_uppercase(), Defs(())))
                },
            ),
        ]
    });

    let c = ctx(&i, &[("msg", Value::string("hello"))]);
    let result = run_script_with_externs(&i, "shout(@msg)", c, vec![registry]).await;
    assert_eq!(result.value, Value::string("HELLO"));
}

// ═══════════════════════════════════════════════════════════════════════
//  ExternFn with context reads (Uses)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_reads_context() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        let qref = QualifiedRef::root(interner.intern("offset"));
        vec![
            ExternFnBuilder::new(
                "add_offset",
                sig_effect(
                    interner,
                    vec![Ty::Int],
                    Ty::Int,
                    Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::new(),
                        self_modifying: false,
                    }),
                ),
            )
            .handler(
                |_interner: &Interner, (x,): (i64,), Uses((offset,)): Uses<(i64,)>| {
                    Ok((x + offset, Defs(())))
                },
            ),
        ]
    });

    let c = ctx(&i, &[("offset", Value::Int(100))]);
    let result = run_script_with_externs(&i, "add_offset(5)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(105));
}

// ═══════════════════════════════════════════════════════════════════════
//  ExternFn with context writes (Defs)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_writes_context() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        let qref = QualifiedRef::root(interner.intern("counter"));
        vec![
            ExternFnBuilder::new(
                "increment",
                sig_effect(
                    interner,
                    vec![],
                    Ty::Unit,
                    Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                ),
            )
            .handler(
                |_interner: &Interner, (): (), Uses((count,)): Uses<(i64,)>| {
                    Ok(((), Defs((count + 1,))))
                },
            ),
        ]
    });

    let c = ctx(&i, &[("counter", Value::Int(0))]);
    let result = run_script_with_externs(&i, "increment(); @counter", c, vec![registry]).await;
    // After increment, @counter should be 1.
    assert_eq!(result.value, Value::Int(1));
}

// ═══════════════════════════════════════════════════════════════════════
//  ExternFn with reads + writes (append to history)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_reads_and_writes_context() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        let qref = QualifiedRef::root(interner.intern("history"));
        vec![
            ExternFnBuilder::new(
                "record",
                sig_effect(
                    interner,
                    vec![Ty::Int],
                    Ty::Int,
                    Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                ),
            )
            .handler(
                |_interner: &Interner, (item,): (Value,), Uses((history,)): Uses<(Vec<Value>,)>| {
                    let mut new_history = history;
                    new_history.push(item);
                    let len = new_history.len() as i64;
                    Ok((len, Defs((new_history,))))
                },
            ),
        ]
    });

    let c = ctx(
        &i,
        &[("history", Value::list(vec![Value::Int(1), Value::Int(2)]))],
    );
    let result = run_script_with_externs(&i, "record(3)", c, vec![registry]).await;
    // ret = 3 (new length)
    assert_eq!(result.value, Value::Int(3));
}

// ═══════════════════════════════════════════════════════════════════════
//  Multiple ExternFn calls in sequence
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_multiple_calls_sequential() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        let qref = QualifiedRef::root(interner.intern("acc"));
        vec![
            ExternFnBuilder::new(
                "add_to_acc",
                sig_effect(
                    interner,
                    vec![Ty::Int],
                    Ty::Unit,
                    Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                ),
            )
            .handler(
                |_interner: &Interner, (x,): (i64,), Uses((acc,)): Uses<(i64,)>| {
                    Ok(((), Defs((acc + x,))))
                },
            ),
        ]
    });

    let c = ctx(&i, &[("acc", Value::Int(0))]);
    let result = run_script_with_externs(
        &i,
        "add_to_acc(10); add_to_acc(20); add_to_acc(12); @acc",
        c,
        vec![registry],
    )
    .await;
    assert_eq!(result.value, Value::Int(42));
}

// ═══════════════════════════════════════════════════════════════════════
//  ExternFn capturing Rust environment
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn extern_captures_environment() {
    let i = Interner::new();
    let secret = 7i64;

    let registry = ExternRegistry::new(move |interner| {
        vec![
            ExternFnBuilder::new("multiply_secret", sig(interner, vec![Ty::Int], Ty::Int)).handler(
                move |_interner: &Interner, (x,): (i64,), Uses(()): Uses<()>| {
                    Ok((x * secret, Defs(())))
                },
            ),
        ]
    });

    let c = ctx(&i, &[]);
    let result = run_script_with_externs(&i, "multiply_secret(6)", c, vec![registry]).await;
    assert_eq!(result.value, Value::Int(42));
}

// ═══════════════════════════════════════════════════════════════════════
//  Regex ExternFn (legacy sync_handler, Builtin path)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn regex_match_via_extern() {
    let i = Interner::new();

    let mut tr = TypeRegistry::new();
    let registry = acvus_ext::regex_registry(&i, &mut tr);
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

    let mut tr = TypeRegistry::new();
    let registry = acvus_ext::regex_registry(&i, &mut tr);
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

// ═══════════════════════════════════════════════════════════════════════
//  IR verification: FunctionCall has correct context_uses/context_defs
// ═══════════════════════════════════════════════════════════════════════

/// Verify that after compilation, FunctionCall instructions for ExternFn
/// with reads/writes have non-empty context_uses/context_defs (filled by SSA pass).
#[test]
fn ir_function_call_has_context_bindings() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        let qref = QualifiedRef::root(interner.intern("counter"));
        vec![
            ExternFnBuilder::new(
                "bump",
                sig_effect(
                    interner,
                    vec![],
                    Ty::Unit,
                    Effect::Resolved(EffectSet {
                        reads: BTreeSet::from([EffectTarget::Context(qref)]),
                        writes: BTreeSet::from([EffectTarget::Context(qref)]),
                        self_modifying: false,
                    }),
                ),
            )
            .handler(|_interner: &Interner, (): (), Uses((n,)): Uses<(i64,)>| {
                Ok(((), Defs((n + 1,))))
            }),
        ]
    });

    let context_types: FxHashMap<acvus_utils::Astr, Ty> =
        FxHashMap::from_iter([(i.intern("counter"), Ty::Int)]);

    let source = "bump(); @counter";
    let cr = compile_source_with_externs(
        &i,
        acvus_mir::graph::ParsedAst::Script(
            acvus_ast::parse_script(&i, source).expect("parse error"),
        ),
        &context_types,
        vec![registry],
        acvus_mir::ty::TypeRegistry::new(),
    );

    // Find the FunctionCall in the entry module's IR.
    let entry_module = cr.modules.get(&cr.entry_qref).unwrap();
    let module = match entry_module {
        Executable::Module(m) => m,
        _ => panic!("entry should be a Module"),
    };

    let call_insts: Vec<_> = module
        .main
        .insts
        .iter()
        .filter(|inst| matches!(&inst.kind, InstKind::FunctionCall { .. }))
        .collect();

    // There should be at least one FunctionCall (to "bump").
    let bump_call = call_insts
        .iter()
        .find(|inst| {
            if let InstKind::FunctionCall {
                callee: acvus_mir::ir::Callee::Direct(id),
                ..
            } = &inst.kind
            {
                // Find the "bump" function by checking extern executables.
                cr.extern_executables.contains_key(&id)
            } else {
                false
            }
        })
        .expect("should have a FunctionCall to bump");

    if let InstKind::FunctionCall {
        context_uses,
        context_defs,
        ..
    } = &bump_call.kind
    {
        assert!(
            !context_uses.is_empty(),
            "FunctionCall to bump should have context_uses (reads @counter)"
        );
        assert!(
            !context_defs.is_empty(),
            "FunctionCall to bump should have context_defs (writes @counter)"
        );

        // Verify QualifiedRef is @counter.
        let counter_ref = QualifiedRef::root(i.intern("counter"));
        assert_eq!(context_uses[0].0, counter_ref);
        assert_eq!(context_defs[0].0, counter_ref);
    } else {
        panic!("expected FunctionCall");
    }
}

/// Pure ExternFn should have empty context_uses/context_defs in IR.
#[test]
fn ir_pure_function_call_no_context_bindings() {
    let i = Interner::new();

    let registry = ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new("double", sig(interner, vec![Ty::Int], Ty::Int)).handler(
                |_interner: &Interner, (x,): (i64,), Uses(()): Uses<()>| Ok((x * 2, Defs(()))),
            ),
        ]
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

    for inst in call_insts {
        if let InstKind::FunctionCall {
            context_uses,
            context_defs,
            ..
        } = &inst.kind
        {
            assert!(
                context_uses.is_empty(),
                "pure function should have empty context_uses"
            );
            assert!(
                context_defs.is_empty(),
                "pure function should have empty context_defs"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  IO ExternFn — Parallelization end-to-end
// ═══════════════════════════════════════════════════════════════════════
//
// Tests verify that the full optimizer pipeline (SpawnSplit → CodeMotion →
// Reorder → SSA → RegColor) produces correct MIR structure AND correct
// execution results for various IO parallelization patterns.
//
// Each test dumps the optimized MIR to stderr (--nocapture) for inspection.

fn io_effect() -> Effect {
    Effect::Resolved(EffectSet {
        self_modifying: true,
        ..Default::default()
    })
}

/// Registry with 4 independent IO functions (no args) + 1 parameterized.
fn io_registry() -> ExternRegistry {
    ExternRegistry::new(|interner| {
        vec![
            ExternFnBuilder::new(
                "fetch_a",
                sig_effect_hint(interner, vec![], Ty::Int, io_effect(), Some(Hint::Io)),
            )
            .handler(|_: &Interner, (): (), Uses(()): Uses<()>| Ok((100i64, Defs(())))),
            ExternFnBuilder::new(
                "fetch_b",
                sig_effect_hint(interner, vec![], Ty::Int, io_effect(), Some(Hint::Io)),
            )
            .handler(|_: &Interner, (): (), Uses(()): Uses<()>| Ok((200i64, Defs(())))),
            ExternFnBuilder::new(
                "fetch_c",
                sig_effect_hint(interner, vec![], Ty::Int, io_effect(), Some(Hint::Io)),
            )
            .handler(|_: &Interner, (): (), Uses(()): Uses<()>| Ok((300i64, Defs(())))),
            ExternFnBuilder::new(
                "fetch_d",
                sig_effect_hint(interner, vec![], Ty::Int, io_effect(), Some(Hint::Io)),
            )
            .handler(|_: &Interner, (): (), Uses(()): Uses<()>| Ok((400i64, Defs(())))),
            // Parameterized: fetch_by(x) = x * 10
            ExternFnBuilder::new(
                "fetch_by",
                sig_effect_hint(interner, vec![Ty::Int], Ty::Int, io_effect(), Some(Hint::Io)),
            )
            .handler(|_: &Interner, (x,): (i64,), Uses(()): Uses<()>| Ok((x * 10, Defs(())))),
        ]
    })
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
    let mut tr = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(&i, &mut tr);
    let mut regs = std_regs;
    regs.push(io_registry());
    let cr = compile_source_with_externs(&i, ast, &context_types, regs, tr);
    (i, cr)
}

fn infer_ty(v: &Value) -> Ty {
    match v {
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Bool(_) => Ty::Bool,
        Value::String(_) => Ty::String,
        Value::List(items) => {
            let elem = items.first().map(infer_ty).unwrap_or(Ty::Int);
            Ty::List(Box::new(elem))
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

// ── 1. Two independent IO calls ────────────────────────────────────

/// fetch_a() + fetch_b() → spawn both before eval either.
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
    assert_eq!(evals.len(), 2, "expected 2 evals");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "all spawns must precede all evals"
    );
}

// ── 2. Four-way independent IO ─────────────────────────────────────

/// Maximum parallelism: 4 independent IO calls.
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
    assert_eq!(evals.len(), 4, "expected 4 evals");
    assert!(
        spawns.iter().all(|&s| evals.iter().all(|&e| s < e)),
        "all spawns must precede all evals"
    );
}

// ── 3. Dependent chain + independent IO ────────────────────────────
//
// a = fetch_a()          // IO, independent
// b = fetch_by(a)        // IO, depends on a
// c = fetch_c()          // IO, independent of a and b
// b + c
//
// Optimal: spawn fetch_a + spawn fetch_c in parallel,
//          eval fetch_a, spawn fetch_by(a), eval fetch_c, eval fetch_by → b+c

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

    // 3 IO calls → 3 spawns, 3 evals.
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

// ── 4. Diamond dependency ──────────────────────────────────────────
//
// a = fetch_a()          // IO
// b = fetch_by(a)        // IO, depends on a
// c = fetch_by(a)        // IO, depends on a (same dep as b, but independent of b)
// b + c
//
// After eval(a), both fetch_by(a) calls can be spawned in parallel.

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

// ── 5. Deep sequential chain ───────────────────────────────────────
//
// a = fetch_a(); b = fetch_by(a); c = fetch_by(b); d = fetch_by(c); d
//
// No parallelism possible: each depends on the previous.
// spawn→eval→spawn→eval→spawn→eval→spawn→eval

#[tokio::test]
async fn io_deep_chain() {
    let i = Interner::new();
    // 100 → 1000 → 10000 → 100000
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

// ── 6. Two independent chains ──────────────────────────────────────
//
// a = fetch_a(); b = fetch_by(a);    // chain 1: a → b
// c = fetch_c(); d = fetch_by(c);    // chain 2: c → d (independent of chain 1)
// b + d
//
// Optimal: spawn a + spawn c, eval a, spawn b, eval c, spawn d, eval b, eval d

#[tokio::test]
async fn io_two_independent_chains() {
    let i = Interner::new();
    // chain 1: 100 → 1000, chain 2: 300 → 3000
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
    assert_eq!(evals.len(), 4, "4 evals");

    // The heads of both chains (fetch_a, fetch_c) should be spawned before any eval.
    assert!(
        spawns[0] < evals[0] && spawns[1] < evals[0],
        "chain heads should be spawned before first eval"
    );
}

// ── 7. IO in iteration ─────────────────────────────────────────────
//
// Iterate over list, call IO per element, accumulate.
// Within each iteration: spawn should precede eval.

#[tokio::test]
async fn io_in_iteration() {
    let i = Interner::new();
    // fetch_by(1)=10, fetch_by(2)=20, fetch_by(3)=30 → sum=60
    let c = ctx(
        &i,
        &[
            (
                "items",
                Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
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

// ── 8. Compiler pipeline pattern ───────────────────────────────────
//
// Simulates: resolve_imports + parse_types (independent IO),
// then dependent passes that use both results.
//
// imports = fetch_a()           // "resolve imports" — IO
// types   = fetch_b()           // "parse types"     — IO, independent
// refs    = fetch_by(imports)   // "resolve refs"    — depends on imports
// checked = refs + types        // "type check"      — depends on refs + types
// extra   = fetch_c()           // "lint"            — independent of everything
// checked + extra
//
// Optimal: spawn imports + spawn types + spawn extra (3-way parallel),
//          eval imports, spawn refs, eval types + eval extra whenever,
//          eval refs, compute result.

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
    assert_eq!(evals.len(), 4, "expected 4 evals");

    // fetch_a, fetch_b, fetch_c are independent — all 3 should be spawned before any eval.
    // fetch_by depends on eval(fetch_a).
    // At minimum: 3 independent spawns before first eval.
    let spawns_before_first_eval = spawns.iter().filter(|&&s| s < evals[0]).count();
    assert!(
        spawns_before_first_eval >= 3,
        "at least 3 independent IO spawns should precede first eval, got {spawns_before_first_eval}"
    );
}
