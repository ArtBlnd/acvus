//! Interpreter e2e tests for ExternFn: uses/defs, context reads/writes via handler.

use std::collections::BTreeSet;

use acvus_interpreter::{Defs, Executable, ExternFnBuilder, ExternRegistry, Uses, Value};
use acvus_interpreter_test::*;
use acvus_mir::graph::{Constraint, FnConstraint, QualifiedRef, Signature};
use acvus_mir::ir::InstKind;
use acvus_mir::ty::{Effect, EffectSet, EffectTarget, Param, Ty, TypeRegistry};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

fn sig(interner: &Interner, params: Vec<Ty>, ret: Ty) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect: Effect::pure(),
        }),
        effect: None,
    }
}

fn sig_effect(interner: &Interner, params: Vec<Ty>, ret: Ty, effect: Effect) -> FnConstraint {
    let named: Vec<Param> = params
        .into_iter()
        .enumerate()
        .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty))
        .collect();
    FnConstraint {
        signature: Some(Signature { params: named.clone() }),
        output: Constraint::Exact(Ty::Fn {
            params: named,
            ret: Box::new(ret),
            captures: vec![],
            effect,
        }),
        effect: None,
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
            ExternFnBuilder::new("shout", sig(interner, vec![Ty::String], Ty::String))
                .handler(
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
            ExternFnBuilder::new("add_offset", sig_effect(
                interner,
                vec![Ty::Int],
                Ty::Int,
                Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::new(),
                    io: false,
                    self_modifying: false,
                }),
            ))
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
            ExternFnBuilder::new("increment", sig_effect(
                interner,
                vec![],
                Ty::Unit,
                Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            ))
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
            ExternFnBuilder::new("record", sig_effect(
                interner,
                vec![Ty::Int],
                Ty::Int,
                Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            ))
                .handler(
                    |_interner: &Interner,
                     (item,): (Value,),
                     Uses((history,)): Uses<(Vec<Value>,)>| {
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
            ExternFnBuilder::new("add_to_acc", sig_effect(
                interner,
                vec![Ty::Int],
                Ty::Unit,
                Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            ))
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
            ExternFnBuilder::new("multiply_secret", sig(interner, vec![Ty::Int], Ty::Int))
                .handler(
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
    let registry = acvus_ext::regex_registry(&mut tr);
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
    let registry = acvus_ext::regex_registry(&mut tr);
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
        Value::Variant { tag, payload, .. } => {
            assert_eq!(i.resolve(*tag), "Some");
            let inner = payload.as_ref().expect("Some should have payload");
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
            ExternFnBuilder::new("bump", sig_effect(
                interner,
                vec![],
                Ty::Unit,
                Effect::Resolved(EffectSet {
                    reads: BTreeSet::from([EffectTarget::Context(qref)]),
                    writes: BTreeSet::from([EffectTarget::Context(qref)]),
                    io: false,
                    self_modifying: false,
                }),
            ))
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
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error")),
        &context_types,
        vec![registry],
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
                cr.extern_executables.contains_key(id)
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
            ExternFnBuilder::new("double", sig(interner, vec![Ty::Int], Ty::Int))
                .handler(|_interner: &Interner, (x,): (i64,), Uses(()): Uses<()>| {
                    Ok((x * 2, Defs(())))
                }),
        ]
    });

    let context_types: FxHashMap<acvus_utils::Astr, Ty> = FxHashMap::default();

    let source = "double(21)";
    let cr = compile_source_with_externs(
        &i,
        acvus_mir::graph::ParsedAst::Script(acvus_ast::parse_script(&i, source).expect("parse error")),
        &context_types,
        vec![registry],
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
                } if cr.extern_executables.contains_key(id)
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
