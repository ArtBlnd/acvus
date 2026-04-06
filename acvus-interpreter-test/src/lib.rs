use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{
    ExecResult, Executable, ExternRegistry, InMemoryContext, Interpreter, InterpreterContext,
    SequentialExecutor, Value,
};
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower, optimize as graph_optimize};
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, lift_to_poly, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

// ── Core pipeline ───────────────────────────────────────────────

/// Compile a template source → MirModule + context id mapping.
pub struct CompileResult {
    pub entry_qref: QualifiedRef,
    pub modules: FxHashMap<QualifiedRef, Executable>,
    pub context_names: FxHashMap<QualifiedRef, Astr>,
    pub builtin_ids: FxHashMap<Astr, QualifiedRef>,
    pub fn_types: FxHashMap<QualifiedRef, Ty>,
    pub extern_executables: FxHashMap<QualifiedRef, Executable>,
}

fn compile(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast = ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse error"));
    let mut tr = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut tr);
    compile_source_with_externs(interner, ast, context_types, std_regs, tr)
}

fn compile_script(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let mut tr = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut tr);
    compile_source_with_externs(interner, ast, context_types, std_regs, tr)
}

fn compile_script_mode(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast =
        ParsedAst::Script(acvus_ast::parse_script_mode(interner, source).expect("parse error"));
    let mut tr = acvus_mir::ty::TypeRegistry::new();
    let std_regs = acvus_ext::std_registries(interner, &mut tr);
    compile_source_with_externs(interner, ast, context_types, std_regs, tr)
}

pub fn compile_source_with_externs(
    interner: &Interner,
    ast: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<ExternRegistry>,
    type_registry: acvus_mir::ty::TypeRegistry,
) -> CompileResult {
    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_to_poly(ty),
        })
        .collect();

    let entry_qref = QualifiedRef::root(interner.intern("test"));
    let mut pb = PolyBuilder::new();
    let mut functions = Vec::new();
    functions.push(Function {
        qref: entry_qref,
        kind: FnKind::Local(ast),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: pb.fresh_effect_var(),
            hint: None,
        },
        effect_constraint: None,
    });

    // Register ExternFns.
    let mut extern_executables: FxHashMap<QualifiedRef, Executable> = FxHashMap::default();
    let mut fn_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
    for registry in extern_registries {
        let registered = registry.register(interner);
        for func in &registered.functions {
            // Extract concrete Ty from the function's polymorphic type.
            // Polymorphic ExternFns (with Var placeholders) are skipped — only fully concrete ones get metadata.
            if let Some(ty) = try_freeze_poly(&func.ty) {
                fn_types.insert(func.qref, ty);
            }
        }
        functions.extend(registered.functions);
        extern_executables.extend(registered.executables);
    }

    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
    };

    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(type_registry),
        &FxHashMap::default(),
    );

    // Collect all errors: infer (unresolved functions) + lower.
    let mut all_errors: Vec<String> = Vec::new();

    // Report infer-level errors (unresolved functions).
    for (qref, outcome) in &inf.outcomes {
        if let acvus_mir::graph::infer::FnInferOutcome::Incomplete { errors, .. } = outcome
            && !errors.is_empty()
        {
            let fn_name = interner.resolve(qref.name);
            for e in errors {
                all_errors.push(format!("[{fn_name}] {}", e.display(interner)));
            }
        }
    }

    let result = graph_lower::lower(interner, &graph, &ext, &inf, &FxHashMap::default());

    // Report lower-level errors.
    for e in result.errors.iter().flat_map(|e| e.errors.iter()) {
        all_errors.push(format!("{}", e.display(interner)));
    }

    if !all_errors.is_empty() {
        panic!("compile failed:\n  {}", all_errors.join("\n  "));
    }

    // Run full optimization pipeline: SSA → Inline → SpawnSplit → Reorder → SSA → RegColor → Validate.
    let opt_result = graph_optimize::optimize(result.modules.clone(), &inf.context_types, &FxHashSet::default());

    // Report validation errors from optimization.
    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            all_errors.push(format!("[validate:{fn_name}] {:?}", e));
        }
    }
    if !all_errors.is_empty() {
        panic!("optimize validation failed:\n  {}", all_errors.join("\n  "));
    }

    // Collect optimized modules as Executable::Module.
    let modules: FxHashMap<QualifiedRef, Executable> = opt_result
        .modules
        .into_iter()
        .map(|(qref, module)| (qref, Executable::Module(module)))
        .collect();

    // Build context qref → name mapping.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();

    // Build builtin name → qref mapping from the same graph functions.
    let builtin_ids: FxHashMap<Astr, QualifiedRef> = graph
        .functions
        .iter()
        .map(|f| (f.qref.name, f.qref))
        .collect();

    CompileResult {
        entry_qref,
        modules,
        context_names,
        builtin_ids,
        fn_types,
        extern_executables,
    }
}

/// Parse + compile + execute a template, returning the output string.
pub async fn run(interner: &Interner, source: &str, context: FxHashMap<Astr, Value>) -> String {
    let context_types: FxHashMap<Astr, Ty> =
        context.iter().map(|(k, v)| (*k, infer_ty(v))).collect();

    let cr = compile(interner, source, &context_types);

    // Debug: dump entry module IR + closures
    if let Some(Executable::Module(module)) = cr.modules.get(&cr.entry_qref) {
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

    // Build context snapshot for page.
    let snapshot: HashMap<String, Value> = context
        .into_iter()
        .map(|(k, v)| (interner.resolve(k).to_string(), v))
        .collect();

    let executor = Arc::new(SequentialExecutor);
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);

    let page = InMemoryContext::new(snapshot, interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_qref, page);
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
    let context_types: FxHashMap<Astr, Ty> =
        context.iter().map(|(k, v)| (*k, infer_ty(v))).collect();

    let cr = compile_script(interner, source, &context_types);

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

    let page = InMemoryContext::new(snapshot, interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_qref, page);
    let result = interp.execute().await.expect("execution failed");
    result.value
}

/// Compile and execute a **script-mode** (keyword syntax: let/for/while/if), returning the result Value.
pub async fn run_script_mode(
    interner: &Interner,
    source: &str,
    context: FxHashMap<Astr, Value>,
) -> Value {
    let context_types: FxHashMap<Astr, Ty> =
        context.iter().map(|(k, v)| (*k, infer_ty(v))).collect();

    let cr = compile_script_mode(interner, source, &context_types);

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

    let page = InMemoryContext::new(snapshot, interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_qref, page);
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
    let context_types: FxHashMap<Astr, Ty> =
        context.iter().map(|(k, v)| (*k, infer_ty(v))).collect();

    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let cr = compile_source_with_externs(
        interner,
        ast,
        &context_types,
        extern_registries,
        acvus_mir::ty::TypeRegistry::new(),
    );

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

    let page = InMemoryContext::new(snapshot, interner.clone());
    let mut interp = Interpreter::new(shared, cr.entry_qref, page);
    interp.execute().await.expect("execution failed")
}

// ── JSON helpers ─────────────────────────────────────────────────

pub fn value_from_json(interner: &Interner, v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap())
            }
        }
        serde_json::Value::String(s) => Value::string(s.as_str()),
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Array(items) => {
            Value::list(items.iter().map(|v| value_from_json(interner, v)).collect())
        }
        serde_json::Value::Object(fields) => Value::object(
            fields
                .iter()
                .map(|(k, v)| (interner.intern(k), value_from_json(interner, v)))
                .collect(),
        ),
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
            let field_types = fields.iter().map(|(k, v)| (*k, infer_ty(v))).collect();
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
        Some(serde_json::Value::Object(fields)) => fields
            .iter()
            .map(|(k, v)| (interner.intern(k), value_from_json(&interner, v)))
            .collect(),
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
