//! E2E tests for ext functions: compile script with ext registry → execute → check result.

use std::collections::HashMap;
use std::sync::Arc;

use acvus_ext::*;
use acvus_interpreter::*;
use acvus_interpreter::builtins::build_builtins;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, infer, lower as graph_lower};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Compile + execute a script with ext registries.
async fn run_ext(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
    registries: Vec<ExternRegistry>,
) -> Value {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_value_ty(v)))
        .collect();

    // Register all ext functions — alloc FunctionIds.
    let registered: Vec<Registered> = registries
        .into_iter()
        .map(|r| r.register(interner))
        .collect();

    // Build contexts.
    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            name: *name,
            namespace: None,
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();

    // Build function list: builtins + ext + entry.
    let entry_id = FunctionId::alloc();
    let mut functions = acvus_mir::builtins::standard_builtins(interner);
    for reg in &registered {
        functions.extend(reg.functions.iter().cloned());
    }
    functions.push(Function {
        id: entry_id,
        name: interner.intern("test"),
        namespace: None,
        kind: FnKind::Local(SourceCode {
            name: interner.intern("test"),
            source: interner.intern(source),
            kind: SourceKind::Script,
        }),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    });

    let graph = CompilationGraph {
        namespaces: Freeze::new(vec![]),
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

    // Compile.
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext, &FxHashMap::default());
    let result = graph_lower::lower(interner, &graph, &ext, &inf);

    if result.has_errors() {
        let errs: Vec<String> = result.errors.iter()
            .flat_map(|e| e.errors.iter())
            .map(|e| format!("{}", e.display(interner)))
            .collect();
        panic!("compile failed: {}", errs.join("; "));
    }

    // Build runtime functions: modules + builtins + ext handlers.
    let mut exec_fns: FxHashMap<FunctionId, Executable> = result.modules
        .into_iter()
        .map(|(id, (module, _))| (id, Executable::Module(module)))
        .collect();

    let builtin_ids: FxHashMap<Astr, FunctionId> = graph.functions
        .iter()
        .map(|f| (f.name, f.id))
        .collect();
    for (id, handler) in build_builtins(&builtin_ids, interner) {
        exec_fns.insert(id, Executable::Builtin(handler));
    }
    for reg in registered {
        exec_fns.extend(reg.executables);
    }

    // Execute.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph.contexts
        .iter()
        .map(|ctx| (ctx.qualified_ref(), ctx.name))
        .collect();
    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, exec_fns, executor)
        .with_context_names(context_names);
    let overlay = ContextOverlay::new(Arc::new(snapshot), interner.clone());
    let mut interp = Interpreter::new(shared, entry_id, overlay);
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
        Value::Object(fields) => {
            Ty::Object(fields.iter().map(|(k, v)| (*k, infer_value_ty(v))).collect())
        }
        Value::Opaque(o) => Ty::Opaque(o.type_name.to_string()),
        _ => Ty::Unit,
    }
}

fn ctx(i: &Interner, entries: &[(&str, Value)]) -> FxHashMap<Astr, Value> {
    entries.iter().map(|(name, val)| (i.intern(name), val.clone())).collect()
}

// ═══════════════════════════════════════════════════════════════════════
//  Regex
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn regex_match_true() {
    let i = Interner::new();
    let c = FxHashMap::default();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc123")"#,
        c,
        vec![regex_registry()],
    ).await;
    assert_eq!(result, Value::Bool(true));
}

#[tokio::test]
async fn regex_match_false() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_match(re, "abc")"#,
        FxHashMap::default(),
        vec![regex_registry()],
    ).await;
    assert_eq!(result, Value::Bool(false));
}

#[tokio::test]
async fn regex_find_all_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\d+"); regex_find_all(re, "a1b22c333") | collect"#,
        FxHashMap::default(),
        vec![regex_registry()],
    ).await;
    let Value::List(items) = result else { panic!("expected List") };
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], Value::string("1"));
    assert_eq!(items[1], Value::string("22"));
    assert_eq!(items[2], Value::string("333"));
}

#[tokio::test]
async fn regex_replace() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("\\s+"); regex_replace("hello   world", re, " ")"#,
        FxHashMap::default(),
        vec![regex_registry()],
    ).await;
    assert_eq!(result, Value::string("hello world"));
}

#[tokio::test]
async fn regex_split_collect() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"re = regex("[,;]\\s*"); regex_split(re, "a, b;c") | collect"#,
        FxHashMap::default(),
        vec![regex_registry()],
    ).await;
    let Value::List(items) = result else { panic!("expected List") };
    assert_eq!(*items, vec![
        Value::string("a"),
        Value::string("b"),
        Value::string("c"),
    ]);
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
    ).await;
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
    ).await;
    assert_eq!(result, Value::string("hello world&foo=bar"));
}

// ═══════════════════════════════════════════════════════════════════════
//  DateTime
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn datetime_format_from_timestamp() {
    let i = Interner::new();
    // 2024-01-01 00:00:00 UTC = epoch 1704067200
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); format_date(dt, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry()],
    ).await;
    assert_eq!(result, Value::string("2024-01-01"));
}

#[tokio::test]
async fn datetime_timestamp_roundtrip() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); timestamp(dt)"#,
        FxHashMap::default(),
        vec![datetime_registry()],
    ).await;
    assert_eq!(result, Value::Int(1704067200));
}

#[tokio::test]
async fn datetime_add_days() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = from_timestamp(1704067200); dt2 = add_days(dt, 1); format_date(dt2, "%Y-%m-%d")"#,
        FxHashMap::default(),
        vec![datetime_registry()],
    ).await;
    assert_eq!(result, Value::string("2024-01-02"));
}

#[tokio::test]
async fn datetime_parse_and_format() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"dt = parse_date("2024-06-15 12:30:00", "%Y-%m-%d %H:%M:%S"); format_date(dt, "%m/%d/%Y")"#,
        FxHashMap::default(),
        vec![datetime_registry()],
    ).await;
    assert_eq!(result, Value::string("06/15/2024"));
}

// ═══════════════════════════════════════════════════════════════════════
//  Multiple registries
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn mixed_regex_and_encoding() {
    let i = Interner::new();
    let result = run_ext(
        &i,
        r#"base64_encode("hello") + " " + to_string(regex_match(regex("\\d+"), "abc123"))"#,
        FxHashMap::default(),
        vec![regex_registry(), encoding_registry()],
    ).await;
    assert_eq!(result, Value::string("aGVsbG8= true"));
}
