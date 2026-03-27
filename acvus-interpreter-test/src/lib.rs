use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{
    ContextOverlay, ExecResult, Executable, ExternFn, ExternRegistry, Interpreter,
    InterpreterContext, Registered, SequentialExecutor, Value,
};
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

// ── Core pipeline ───────────────────────────────────────────────

/// Compile a template source → MirModule + context id mapping.
pub struct CompileResult {
    pub entry_id: FunctionId,
    pub modules: FxHashMap<FunctionId, Executable>,
    pub context_names: FxHashMap<QualifiedRef, Astr>,
    pub builtin_ids: FxHashMap<Astr, FunctionId>,
    pub fn_types: FxHashMap<FunctionId, Ty>,
    pub extern_executables: FxHashMap<FunctionId, Executable>,
}

fn compile(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    compile_source(interner, source, context_types, SourceKind::Template)
}

fn compile_source(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    kind: SourceKind,
) -> CompileResult {
    compile_source_with_externs(interner, source, context_types, kind, vec![])
}

pub fn compile_source_with_externs(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    kind: SourceKind,
    extern_registries: Vec<ExternRegistry>,
) -> CompileResult {
    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            name: *name,
            namespace: None,
            constraint: Constraint::Exact(ty.clone()),
        })
        .collect();

    let entry_id = FunctionId::alloc();
    let mut functions = acvus_mir::builtins::standard_builtins(interner);
    functions.push(Function {
        id: entry_id,
        name: interner.intern("test"),
        namespace: None,
        kind: FnKind::Local(SourceCode {
            name: interner.intern("test"),
            source: interner.intern(source),
            kind,
        }),
        constraint: FnConstraint {
            signature: None,
            output: Constraint::Inferred,
            effect: None,
        },
    });

    // Register ExternFns.
    let mut extern_executables: FxHashMap<FunctionId, Executable> = FxHashMap::default();
    let mut fn_types: FxHashMap<FunctionId, Ty> = FxHashMap::default();
    for registry in extern_registries {
        let registered = registry.register(interner);
        for func in &registered.functions {
            // Extract Ty from constraint for fn_types map.
            if let Constraint::Exact(ty) = &func.constraint.output {
                fn_types.insert(func.id, ty.clone());
            }
        }
        functions.extend(registered.functions);
        extern_executables.extend(registered.executables);
    }

    let graph = CompilationGraph {
        namespaces: Freeze::new(vec![]),
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

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

    // Collect all modules as Executable::Module.
    let modules: FxHashMap<FunctionId, Executable> = result.modules
        .into_iter()
        .map(|(id, (module, _hints))| (id, Executable::Module(module)))
        .collect();

    // Build context id → name mapping.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph.contexts
        .iter()
        .map(|ctx| (ctx.qualified_ref(), ctx.name))
        .collect();

    // Build builtin name → id mapping from the same graph functions.
    let builtin_ids: FxHashMap<Astr, FunctionId> = graph.functions
        .iter()
        .map(|f| (f.name, f.id))
        .collect();

    CompileResult { entry_id, modules, context_names, builtin_ids, fn_types, extern_executables }
}

/// Build builtin id mapping from the graph functions.
fn builtin_id_map(
    interner: &Interner,
    modules: &FxHashMap<FunctionId, acvus_mir::ir::MirModule>,
) -> FxHashMap<Astr, FunctionId> {
    // Standard builtins have known names — rebuild them to get name→id.
    let builtins = acvus_mir::builtins::standard_builtins(interner);
    builtins
        .into_iter()
        .map(|f| (f.name, f.id))
        .filter(|(_, id)| !modules.contains_key(id)) // builtins don't have modules
        .collect()
}

/// Parse + compile + execute a template, returning the output string.
pub async fn run(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
) -> String {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_ty(v)))
        .collect();

    let cr = compile(interner, source, &context_types);

    // Debug: dump entry module IR + closures
    if let Some(Executable::Module(module)) = cr.modules.get(&cr.entry_id) {
        let ir = acvus_mir::printer::dump_with(interner, module);
        eprintln!("=== IR for entry ===\n{ir}");
        for (label, closure) in &module.closures {
            eprintln!("=== Closure {label:?} ===");
            for (i, inst) in closure.insts.iter().enumerate() {
                eprintln!("  {i}: {:?}", inst.kind);
            }
        }
    }

    let builtin_handlers = acvus_interpreter::builtins::build_builtins(&cr.builtin_ids, interner);

    // Merge modules + builtins + externs into unified functions map.
    let mut functions = cr.modules;
    for (id, handler) in builtin_handlers {
        functions.insert(id, Executable::Builtin(handler));
    }
    for (id, exec) in cr.extern_executables {
        functions.insert(id, exec);
    }

    // Build context snapshot for overlay.
    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);

    let overlay = ContextOverlay::new(Arc::new(snapshot), interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_id, overlay);
    let result = interp.execute().await.expect("execution failed");

    // Template returns a String.
    match result.value {
        Value::String(s) => s.to_string(),
        Value::Unit => String::new(),
        other => format!("{other:?}"),
    }
}

/// Simple: no context.
pub async fn run_simple(source: &str) -> String {
    let interner = Interner::new();
    run(&interner, source, FxHashMap::default()).await
}

/// Compile and execute a **script**, returning the result Value.
pub async fn run_script(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
) -> Value {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_ty(v)))
        .collect();

    let cr = compile_source(interner, source, &context_types, SourceKind::Script);

    let builtin_handlers = acvus_interpreter::builtins::build_builtins(&cr.builtin_ids, interner);
    let mut functions = cr.modules;
    for (id, handler) in builtin_handlers {
        functions.insert(id, Executable::Builtin(handler));
    }
    for (id, exec) in cr.extern_executables {
        functions.insert(id, exec);
    }

    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);

    let overlay = ContextOverlay::new(Arc::new(snapshot), interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_id, overlay);
    let result = interp.execute().await.expect("execution failed");
    result.value
}

/// Compile and execute a script with ExternFn registries, returning (result, context writes).
pub async fn run_script_with_externs(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
    extern_registries: Vec<ExternRegistry>,
) -> ExecResult {
    let context_types: FxHashMap<Astr, Ty> = context
        .iter()
        .map(|(k, v)| (*k, infer_ty(v)))
        .collect();

    let cr = compile_source_with_externs(
        interner,
        source,
        &context_types,
        SourceKind::Script,
        extern_registries,
    );

    let builtin_handlers =
        acvus_interpreter::builtins::build_builtins(&cr.builtin_ids, interner);
    let mut functions = cr.modules;
    for (id, handler) in builtin_handlers {
        functions.insert(id, Executable::Builtin(handler));
    }
    for (id, exec) in cr.extern_executables {
        functions.insert(id, exec);
    }

    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);

    let overlay = ContextOverlay::new(Arc::new(snapshot), interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_id, overlay);
    interp.execute().await.expect("execution failed")
}

// ── JSON helpers ─────────────────────────────────────────────────

pub fn value_from_json(interner: &Interner, v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Value::Int(i) }
            else { Value::Float(n.as_f64().unwrap()) }
        }
        serde_json::Value::String(s) => Value::string(s.as_str()),
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Array(items) => {
            Value::list(items.iter().map(|v| value_from_json(interner, v)).collect())
        }
        serde_json::Value::Object(fields) => {
            Value::object(fields.iter()
                .map(|(k, v)| (interner.intern(k), value_from_json(interner, v)))
                .collect())
        }
    }
}

/// Infer Ty from a runtime Value (shallow).
fn infer_ty(v: &Value) -> Ty {
    match v {
        Value::Int(_) => Ty::Int,
        Value::Float(_) => Ty::Float,
        Value::Bool(_) => Ty::Bool,
        Value::String(_) => Ty::String,
        Value::Unit => Ty::Unit,
        Value::Byte(_) => Ty::Byte,
        Value::List(items) => {
            let elem = items.first().map(infer_ty).unwrap_or(Ty::Int);
            Ty::List(Box::new(elem))
        }
        Value::Object(fields) => {
            let field_types = fields.iter()
                .map(|(k, v)| (*k, infer_ty(v)))
                .collect();
            Ty::Object(field_types)
        }
        _ => Ty::Unit,
    }
}

// ── Context helpers ──────────────────────────────────────────────

pub fn int_context(interner: &Interner, name: &str, value: i64) -> FxHashMap<Astr, Value> {
    FxHashMap::from_iter([(interner.intern(name), Value::Int(value))])
}

pub fn string_context(interner: &Interner, name: &str, value: &str) -> FxHashMap<Astr, Value> {
    FxHashMap::from_iter([(interner.intern(name), Value::string(value))])
}

pub fn user_context(interner: &Interner) -> FxHashMap<Astr, Value> {
    FxHashMap::from_iter([(
        interner.intern("user"),
        Value::object(FxHashMap::from_iter([
            (interner.intern("name"), Value::string("alice")),
            (interner.intern("age"), Value::Int(30)),
            (interner.intern("email"), Value::string("alice@example.com")),
        ])),
    )])
}

// ── Fixture runner ───────────────────────────────────────────────

/// Run a single `.json` fixture file.
pub async fn run_fixture(path: &std::path::Path) -> Result<(), String> {
    let interner = Interner::new();
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let fixture: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;

    let template = fixture["template"]
        .as_str()
        .ok_or_else(|| format!("{}: missing 'template'", path.display()))?;
    let expected = fixture["expected"]
        .as_str()
        .ok_or_else(|| format!("{}: missing 'expected'", path.display()))?;

    let context: FxHashMap<Astr, Value> = match fixture.get("context") {
        Some(serde_json::Value::Object(fields)) => {
            fields.iter()
                .map(|(k, v)| (interner.intern(k), value_from_json(&interner, v)))
                .collect()
        }
        Some(_) => return Err(format!("{}: 'context' must be an object", path.display())),
        None => FxHashMap::default(),
    };

    let actual = run(&interner, template, context).await;

    if actual != expected {
        Err(format!(
            "output mismatch\n  expected: {expected:?}\n  actual:   {actual:?}"
        ))
    } else {
        Ok(())
    }
}

pub fn items_context(interner: &Interner, items: Vec<i64>) -> FxHashMap<Astr, Value> {
    FxHashMap::from_iter([(
        interner.intern("items"),
        Value::list(items.into_iter().map(Value::Int).collect()),
    )])
}
