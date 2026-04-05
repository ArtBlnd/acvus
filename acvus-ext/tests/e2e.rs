//! E2E tests for ext functions: compile script with ext registry → execute → check result.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_interpreter::builtins::build_builtins;
use acvus_interpreter::*;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::{CastRule, Effect, Ownership, Param, Ty, TypeRegistry, UserDefinedDecl};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile + execute a script with ext registries.
async fn run_ext(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
    registries: Vec<ExternRegistry>,
) -> Value {
    run_ext_with_registry(interner, source, context, registries, TypeRegistry::new()).await
}

/// Compile + execute with a custom TypeRegistry (for ExternCast tests).
async fn run_ext_with_registry(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
    registries: Vec<ExternRegistry>,
    mut type_registry: TypeRegistry,
) -> Value {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_value_ty(v)))
        .collect();

    // Register all ext functions (stdlib + caller-provided).
    let mut all_registries = std_registries(interner, &mut type_registry);
    all_registries.extend(registries);
    let registered: Vec<Registered> = all_registries
        .into_iter()
        .map(|r| r.register(interner))
        .collect();

    // Build contexts.
    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();

    // Build function list: builtins + ext + entry.
    let entry_qref = QualifiedRef::root(interner.intern("test"));
    let mut functions = Vec::new();
    for reg in &registered {
        functions.extend(reg.functions.iter().cloned());
    }
    functions.push(Function {
        qref: entry_qref,
        kind: FnKind::Local(ParsedAst::Script(
            acvus_ast::parse_script(interner, source).expect("parse"),
        )),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
            hint: None,
        },
    });

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

    // Compile.
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );
    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());

    if result.has_errors() {
        let errs: Vec<String> = result
            .errors
            .iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("{}", e.display(interner)))
            .collect();
        panic!("compile failed: {}", errs.join("; "));
    }

    // Build runtime functions: modules + builtins + ext handlers.
    let mut exec_fns: FxHashMap<QualifiedRef, Executable> = result
        .modules
        .into_iter()
        .map(|(qref, module)| (qref, Executable::Module(module)))
        .collect();

    let builtin_ids: FxHashMap<Astr, QualifiedRef> = graph
        .functions
        .iter()
        .map(|f| (f.qref.name, f.qref))
        .collect();
    for (qref, handler) in build_builtins(&builtin_ids, interner) {
        exec_fns.insert(qref, Executable::Builtin(handler));
    }
    for reg in registered {
        exec_fns.extend(reg.executables);
    }

    // Execute.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();
    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared =
        InterpreterContext::new(interner, exec_fns, executor).with_context_names(context_names);
    let page = InMemoryContext::new(snapshot, interner.clone());
    let mut interp = Interpreter::new(shared, entry_qref, page);
    interp.execute().await.expect("execution failed").value
}

/// Shallow type inference from Value.
fn infer_value_ty(v: &Value) -> Ty {
    match v {
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Bool(_) => Ty::Bool,
        Value::String(_) => Ty::String,
        Value::Unit => Ty::Unit,
        Value::Byte(_) => Ty::Byte,
        Value::List(items) => {
            let elem = items.first().map(infer_value_ty).unwrap_or(Ty::Int);
            Ty::List(Box::new(elem))
        }
        Value::Object(fields) => Ty::Object(
            fields
                .iter()
                .map(|(k, v)| (*k, infer_value_ty(v)))
                .collect(),
        ),
        Value::Opaque(o) => Ty::UserDefined {
            id: o.type_id,
            type_args: vec![],
            effect_args: vec![],
            ownership: Ownership::MoveOnly,
        },
        _ => Ty::Unit,
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Regex
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn regex_match_true() {
    let i = Interner::new();
    let c = FxHashMap::default();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc123")"#,
        c,
        vec![regex_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::Bool(true));
}

#[tokio::test]
async fn regex_match_false() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc")"#,
        FxHashMap::default(),
        vec![regex_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::Bool(false));
}

#[tokio::test]
async fn regex_find_all_collect() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_find_all(re, "a1b22c333") | collect"#,
        FxHashMap::default(),
        vec![regex_registry(&i, &mut tr)],
    )
    .await;
    let Value::List(items) = result else {
        panic!("expected List")
    };
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], Value::string("1"));
    assert_eq!(items[1], Value::string("22"));
    assert_eq!(items[2], Value::string("333"));
}

#[tokio::test]
async fn regex_replace() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\s+"); regex_replace("hello   world", re, " ")"#,
        FxHashMap::default(),
        vec![regex_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::string("hello world"));
}

#[tokio::test]
async fn regex_split_collect() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"re = regex("[,;]\\s*"); regex_split(re, "a, b;c") | collect"#,
        FxHashMap::default(),
        vec![regex_registry(&i, &mut tr)],
    )
    .await;
    let Value::List(items) = result else {
        panic!("expected List")
    };
    assert_eq!(
        *items,
        vec![Value::string("a"), Value::string("b"), Value::string("c"),]
    );
}

// ═══════════════════════════════════════════════════════════════════════
//  Encoding
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn base64_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"base64_decode(base64_encode("hello world"))"#,
        FxHashMap::default(),
        vec![encoding_registry()],
    )
    .await;
    assert_eq!(result, Value::string("hello world"));
}

#[tokio::test]
async fn url_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"url_decode(url_encode("hello world&foo=bar"))"#,
        FxHashMap::default(),
        vec![encoding_registry()],
    )
    .await;
    assert_eq!(result, Value::string("hello world&foo=bar"));
}

// ═══════════════════════════════════════════════════════════════════════
//  DateTime
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn datetime_format_from_timestamp() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    // 2024-01-01 00:00:00 UTC = epoch 1704067200
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); format_date(dt, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::string("2024-01-01"));
}

#[tokio::test]
async fn datetime_timestamp_roundtrip() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); timestamp(dt)"#,
        FxHashMap::default(),
        vec![datetime_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::Int(1704067200));
}

#[tokio::test]
async fn datetime_add_days() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); dt2 = add_days(dt, 1); format_date(dt2, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry(&i, &mut tr)],
    )
    .await;
    assert_eq!(result, Value::string("2024-01-02"));
}

#[tokio::test]
async fn datetime_parse_and_format() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"dt = parse_date("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S"); format_date(dt, "%m/%d/%Y")"#,
        FxHashMap::default(),
        vec![datetime_registry(&i, &mut tr)],
    ).await;
    assert_eq!(result, Value::string("06/15/2024"));
}

// ═══════════════════════════════════════════════════════════════════════
//  Multiple registries
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn mixed_regex_and_encoding() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let result = run_ext(
        &i,
        r#"base64_encode("hello") + " " + to_string(regex_match(regex("\\d+"), "abc123"))"#,
        FxHashMap::default(),
        vec![regex_registry(&i, &mut tr), encoding_registry()],
    )
    .await;
    assert_eq!(result, Value::string("aGVsbG8= true"));
}

// ═══════════════════════════════════════════════════════════════════════
//  ExternCast — coercion via registered CastRule
// ═══════════════════════════════════════════════════════════════════════

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

/// Build a test setup for ExternCast:
/// - UserDefined "MyNum" (no type params)
/// - ExternFn "make_num" → returns MyNum wrapping 42
/// - ExternFn "to_int" → MyNum → Int (extracts inner value)
/// - CastRule: MyNum → Int (fn = to_int)
/// - ExternFn "double" → Int → Int (doubles the value)
fn extern_cast_setup(interner: &Interner, tr: &mut TypeRegistry) -> Vec<ExternRegistry> {
    let my_num_qref = QualifiedRef::root(interner.intern("MyNum"));
    tr.register(UserDefinedDecl {
        qref: my_num_qref,
        type_params: vec![],
        effect_params: vec![],
        ownership: Ownership::MoveOnly,
    });

    let my_num_ty = Ty::UserDefined {
        id: my_num_qref,
        type_args: vec![],
        effect_args: vec![],
        ownership: Ownership::MoveOnly,
    };

    // We need to register ExternFns first to get FunctionIds, then register CastRule.
    // Use ExternRegistry for the ExternFns, then find to_int's FunctionId.
    let ty_clone = my_num_ty.clone();
    let reg = ExternRegistry::new(move |interner| {
        vec![
            // make_num() → MyNum (wrapping 42)
            ExternFnBuilder::new("make_num", sig(interner, vec![], ty_clone.clone())).handler(
                move |_interner: &acvus_utils::Interner, (): (), Uses(()): Uses<()>| {
                    Ok((
                        Value::opaque(OpaqueValue::new(my_num_qref, 42i64)),
                        Defs(()),
                    ))
                },
            ),
            // to_int(MyNum) → Int
            ExternFnBuilder::new("to_int", sig(interner, vec![ty_clone.clone()], Ty::Int)).handler(
                |_interner: &acvus_utils::Interner, (v,): (Value,), Uses(()): Uses<()>| match v {
                    Value::Opaque(o) => {
                        let n = *o.downcast_ref::<i64>().unwrap();
                        Ok((n, Defs(())))
                    }
                    _ => panic!("expected Opaque"),
                },
            ),
            // double(Int) → Int
            ExternFnBuilder::new("double", sig(interner, vec![Ty::Int], Ty::Int)).handler(
                |_interner: &acvus_utils::Interner, (n,): (i64,), Uses(()): Uses<()>| {
                    Ok((n * 2, Defs(())))
                },
            ),
        ]
    });

    // Register, find to_int's FunctionId, then register CastRule.
    // Note: we return the registry, but CastRule needs to_int's fn_id.
    // Problem: fn_id is allocated inside register(). We need a different approach.
    //
    // Solution: pre-allocate fn_id and build Function manually, OR
    // register first, find id, then register cast rule.

    // Actually, we can register the ExternFns, get the Registered, find to_int's id,
    // and return the Registered. But ExternRegistry consumes itself in register()...
    //
    // Simpler: just return the vec of ExternRegistry and let the caller handle it.
    // The caller (test) can register, find the fn_id, then add the CastRule.
    vec![reg]
}

#[tokio::test]
async fn extern_cast_auto_coercion() {
    let i = Interner::new();
    let mut tr = TypeRegistry::new();
    let registries = extern_cast_setup(&i, &mut tr);

    // Register stdlib + test registries into the same TypeRegistry.
    let mut all_registries = std_registries(&i, &mut tr);
    all_registries.extend(registries);
    let registered: Vec<Registered> = all_registries.into_iter().map(|r| r.register(&i)).collect();

    // Find to_int's QualifiedRef for the CastRule.
    let to_int_qref = registered
        .iter()
        .flat_map(|r| r.functions.iter())
        .find(|f| i.resolve(f.qref.name) == "to_int")
        .unwrap()
        .qref;

    let my_num_qref = QualifiedRef::root(i.intern("MyNum"));
    let my_num_ty = Ty::UserDefined {
        id: my_num_qref,
        type_args: vec![],
        effect_args: vec![],
        ownership: Ownership::MoveOnly,
    };

    // Register CastRule: MyNum → Int
    tr.register_cast(CastRule {
        from: my_num_ty,
        to: Ty::Int,
        fn_ref: to_int_qref,
    });

    let type_registry = Freeze::new(tr);

    // Build graph manually (same as run_ext_with_registry but with pre-registered fns).
    let entry_qref = QualifiedRef::root(i.intern("test"));
    let mut functions = Vec::new();
    for reg in &registered {
        functions.extend(reg.functions.iter().cloned());
    }
    let ast = acvus_ast::parse_script(&i, "double(make_num())").expect("parse error");
    functions.push(Function {
        qref: entry_qref,
        kind: FnKind::Local(ParsedAst::Script(ast)),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
            hint: None,
        },
    });

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(vec![]),
    };

    // Compile.
    let ext = extract::extract(&i, &graph);
    let inf = infer::infer(
        &i,
        &graph,
        &ext,
        &FxHashMap::default(),
        type_registry,
        &FxHashMap::default(),
    );
    let compile_result = graph_lower::lower(&i, &graph, &ext, &inf, &FxHashMap::default());

    if compile_result.has_errors() {
        let errs: Vec<String> = compile_result
            .errors
            .iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("{}", e.display(&i)))
            .collect();
        panic!("compile failed: {}", errs.join("; "));
    }

    // Verify: the lowered MIR should contain a FunctionCall to to_int (the cast fn).
    let module = compile_result.module(entry_qref).unwrap();
    let has_cast_call = module.main.insts.iter().any(|inst| {
        matches!(
            &inst.kind,
            acvus_mir::ir::InstKind::FunctionCall {
                callee: acvus_mir::ir::Callee::Direct(qref),
                ..
            } if *qref == to_int_qref
        )
    });
    assert!(
        has_cast_call,
        "expected FunctionCall to to_int (ExternCast), got: {:#?}",
        module.main.insts
    );

    // Execute.
    let mut exec_fns: FxHashMap<QualifiedRef, Executable> = compile_result
        .modules
        .into_iter()
        .map(|(qref, module)| (qref, Executable::Module(module)))
        .collect();

    let builtin_ids: FxHashMap<Astr, QualifiedRef> = graph
        .functions
        .iter()
        .map(|f| (f.qref.name, f.qref))
        .collect();
    for (qref, handler) in acvus_interpreter::builtins::build_builtins(&builtin_ids, &i) {
        exec_fns.insert(qref, Executable::Builtin(handler));
    }
    for reg in registered {
        exec_fns.extend(reg.executables);
    }

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(&i, exec_fns, executor);
    let page = InMemoryContext::new(HashMap::new(), i.clone());
    let mut interp = Interpreter::new(shared, entry_qref, page);
    let result = interp.execute().await.expect("execution failed");

    // make_num() → MyNum(42), auto-cast to_int → 42, double → 84
    assert_eq!(result.value, Value::Int(84));
}
