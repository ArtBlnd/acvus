use std::collections::HashMap;
use std::sync::Arc;

use acvus_extern::{Externs, Registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::{
    ContextWrite, Executable, InMemoryContext, Interpreter, InterpreterContext, SequentialExecutor,
    Value,
};

/// What a run produced: its value, and the final value of every context it
/// assigned.
pub struct Ran {
    pub value: Value,
    pub writes: Vec<ContextWrite>,
}
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower, optimize as graph_optimize};
use acvus_mir::ty::{LenTerm, PolyBuilder, Ty, TyTerm, lift_declaration, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

// -- Typed context values ------------------------------------------

/// A context value with the type the compiler is told for it: the erased
/// value carries none.
pub struct TypedValue {
    pub ty: Ty,
    pub value: Value,
}

pub type Context = FxHashMap<Astr, TypedValue>;

pub fn typed(ty: Ty, value: Value) -> TypedValue {
    TypedValue { ty, value }
}

fn split_context(interner: &Interner, context: Context) -> (FxHashMap<Astr, Ty>, HashMap<String, Value>) {
    let mut types = FxHashMap::default();
    let mut snapshot = HashMap::new();
    for (name, TypedValue { ty, value }) in context {
        types.insert(name, ty);
        snapshot.insert(interner.resolve(name).to_string(), value);
    }
    (types, snapshot)
}

// -- Core pipeline -----------------------------------------------

/// Compile a template source -> MirModule + context id mapping.
pub struct CompileResult {
    pub entry_qref: QualifiedRef,
    pub modules: FxHashMap<QualifiedRef, Executable>,
    pub context_names: FxHashMap<QualifiedRef, Astr>,
    pub fn_types: FxHashMap<QualifiedRef, Ty>,
    pub extern_executables: FxHashMap<QualifiedRef, Executable>,
}

fn compile(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast = ParsedAst::Template(acvus_ast::parse(interner, source).expect("parse error"));
    let std_regs = acvus_ext::std_registries::<AcvusRuntime>();
    compile_source_with_externs(interner, ast, context_types, std_regs)
}

fn compile_script(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let std_regs = acvus_ext::std_registries::<AcvusRuntime>();
    compile_source_with_externs(interner, ast, context_types, std_regs)
}

fn compile_script_mode(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
) -> CompileResult {
    let ast =
        ParsedAst::Script(acvus_ast::parse_script_mode(interner, source).expect("parse error"));
    let std_regs = acvus_ext::std_registries::<AcvusRuntime>();
    compile_source_with_externs(interner, ast, context_types, std_regs)
}

pub fn compile_source_with_externs(
    interner: &Interner,
    ast: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
) -> CompileResult {
    compile_source_with_externs_and_types(interner, ast, context_types, extern_registries, |_| {})
}

/// Compile with the given registries; `declare_types` registers the caller's
/// own type declarations into the combined type registry.
pub fn compile_source_with_externs_and_types<D>(
    interner: &Interner,
    ast: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    declare_types: D,
) -> CompileResult
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    let mut pb = PolyBuilder::new();
    let contexts: Vec<acvus_mir::graph::Context> = context_types
        .iter()
        .map(|(name, ty)| acvus_mir::graph::Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
        })
        .collect();

    let entry_qref = QualifiedRef::root(interner.intern("test"));
    let mut functions = Vec::new();
    functions.push(Function {
        qref: entry_qref,
        kind: FnKind::Local(ast),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(pb.fresh_ty_var()),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    });

    let Externs {
        functions: extern_fns,
        types: mut type_registry,
        handlers,
        ..
    } = Externs::combine(extern_registries, interner).expect("registries combine");
    declare_types(&mut type_registry);
    // Polymorphic ExternFns (with Var placeholders) are skipped - only fully concrete ones get metadata.
    let fn_types: FxHashMap<QualifiedRef, Ty> = extern_fns
        .iter()
        .filter_map(|func| try_freeze_poly(&func.ty).map(|ty| (func.qref, ty)))
        .collect();
    functions.extend(extern_fns);
    let extern_executables: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(qref, h)| (qref, Executable::Extern(h)))
        .collect();

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

    let result = graph_lower::lower(interner, &graph, &ext, &inf);

    // Report lower-level errors.
    for e in result.errors.iter().flat_map(|e| e.errors.iter()) {
        all_errors.push(format!("{}", e.display(interner)));
    }

    if !all_errors.is_empty() {
        panic!("compile failed:\n  {}", all_errors.join("\n  "));
    }

    // Run full optimization pipeline: SSA -> Inline -> SpawnSplit -> Reorder -> SSA -> RegColor -> Validate.
    let opt_result = graph_optimize::optimize(
        result.modules.clone(),
        &inf.context_types,
        &FxHashSet::default(),
    );

    // Report validation errors from optimization.
    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            all_errors.push(format!("[validate:{fn_name}] {:?}", e));
        }
    }
    if !all_errors.is_empty() {
        let dump = match opt_result.modules.get(&entry_qref) {
            Some(m) => acvus_mir::printer::dump_with(interner, m),
            None => "no entry module".to_string(),
        };
        panic!(
            "optimize validation failed:\n  {}\n{dump}",
            all_errors.join("\n  ")
        );
    }

    // Collect optimized modules as Executable::Module.
    let modules: FxHashMap<QualifiedRef, Executable> = opt_result
        .modules
        .into_iter()
        .map(|(qref, module)| (qref, Executable::Module(module)))
        .collect();

    // Build context qref -> name mapping.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();

    CompileResult {
        entry_qref,
        modules,
        context_names,
        fn_types,
        extern_executables,
    }
}

fn execute_compiled(
    interner: &Interner,
    cr: CompileResult,
    snapshot: HashMap<String, Value>,
    executor: Arc<dyn acvus_interpreter::Executor>,
) -> (InterpreterContext, Interpreter) {
    let mut functions = cr.modules;
    for (id, exec) in cr.extern_executables {
        functions.insert(id, exec);
    }
    let shared = InterpreterContext::new(interner, functions, executor)
        .with_fn_types(cr.fn_types)
        .with_context_names(cr.context_names);
    let page = InMemoryContext::new(snapshot);
    let interp = Interpreter::new(shared.clone(), cr.entry_qref, page);
    (shared, interp)
}

/// Parse + compile + execute a template, returning the output string.
pub async fn run(interner: &Interner, source: &str, context: Context) -> String {
    let (context_types, snapshot) = split_context(interner, context);
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

    let (shared, mut interp) = execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    let result = interp.execute().await.expect("execution failed");

    // A template yields a String; an empty one yields unit.
    match &result {
        v if v.is_string() => {
            // SAFETY: the witness is String.
            unsafe { v.as_str() }.to_owned()
        }
        Value::Small(_) => String::new(),
        other => format!("{other:?}"),
    }
}

/// Simple: no context.
pub async fn run_simple(source: &str) -> String {
    let interner = Interner::new();
    run(&interner, source, Context::default()).await
}

/// Compile and execute a **script**, returning the result Value.
pub async fn run_script(interner: &Interner, source: &str, context: Context) -> Value {
    let (context_types, snapshot) = split_context(interner, context);
    let cr = compile_script(interner, source, &context_types);
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    interp.execute().await.expect("execution failed")
}

/// Compile and execute a **script-mode** (keyword syntax: let/for/while/if), returning the result Value.
pub async fn run_script_mode(interner: &Interner, source: &str, context: Context) -> Value {
    let (context_types, snapshot) = split_context(interner, context);
    let cr = compile_script_mode(interner, source, &context_types);
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    interp.execute().await.expect("execution failed")
}

/// Compile and execute a script with ExternFn registries, returning (result, context writes).
pub async fn run_script_with_externs(
    interner: &Interner,
    source: &str,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
) -> Ran {
    run_script_with_externs_and_types(interner, source, context, extern_registries, |_| {}).await
}

/// Run a script with the given registries; `declare_types` registers the
/// caller's own type declarations into the combined type registry.
pub async fn run_script_with_externs_and_types<D>(
    interner: &Interner,
    source: &str,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    declare_types: D,
) -> Ran
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    run_parsed_with_externs(interner, ast, context, extern_registries, declare_types).await
}

/// Run an already parsed script against `context` with the given registries.
pub async fn run_parsed_with_externs<D>(
    interner: &Interner,
    ast: ParsedAst,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    declare_types: D,
) -> Ran
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    run_parsed_on(
        interner,
        ast,
        context,
        extern_registries,
        declare_types,
        Arc::new(SequentialExecutor),
    )
    .await
}

/// Run an already parsed script on the given executor.
pub async fn run_parsed_on<D>(
    interner: &Interner,
    ast: ParsedAst,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    declare_types: D,
    executor: Arc<dyn acvus_interpreter::Executor>,
) -> Ran
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    let (context_types, snapshot) = split_context(interner, context);
    let cr = compile_source_with_externs_and_types(
        interner,
        ast,
        &context_types,
        extern_registries,
        declare_types,
    );
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, executor);
    let value = interp.execute().await.expect("execution failed");
    let writes = interp.take_writes();
    Ran { value, writes }
}

// -- JSON helpers -------------------------------------------------

/// A JSON value with the type it names: the JSON shape is the type.
pub fn value_from_json(interner: &Interner, v: &serde_json::Value) -> TypedValue {
    match v {
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => typed(Ty::Int, Value::int(i)),
            None => typed(Ty::Float, Value::float(n.as_f64().expect("a JSON number is i64 or f64"))),
        },
        serde_json::Value::String(s) => typed(Ty::String, Value::string(s.as_str())),
        serde_json::Value::Bool(b) => typed(Ty::Bool, Value::bool_(*b)),
        serde_json::Value::Null => typed(Ty::Unit, Value::unit()),
        serde_json::Value::Array(items) => {
            let items: Vec<TypedValue> = items.iter().map(|v| value_from_json(interner, v)).collect();
            let elem = items.first().map(|t| t.ty.clone()).unwrap_or(Ty::Int);
            let len = items.len();
            typed(
                Ty::Array(Box::new(elem), LenTerm::Known(len)),
                Value::array(items.into_iter().map(|t| t.value).collect()),
            )
        }
        serde_json::Value::Object(fields) => {
            let mut tys = FxHashMap::default();
            let mut values = FxHashMap::default();
            for (k, v) in fields {
                let key = interner.intern(k);
                let TypedValue { ty, value } = value_from_json(interner, v);
                tys.insert(key, ty);
                values.insert(key, value);
            }
            typed(Ty::Object(tys), Value::object(values))
        }
    }
}

// -- Context helpers ----------------------------------------------

pub fn int_context(interner: &Interner, name: &str, value: i64) -> Context {
    FxHashMap::from_iter([(interner.intern(name), typed(Ty::Int, Value::int(value)))])
}

pub fn string_context(interner: &Interner, name: &str, value: &str) -> Context {
    FxHashMap::from_iter([(interner.intern(name), typed(Ty::String, Value::string(value)))])
}

pub fn user_context(interner: &Interner) -> Context {
    let name = interner.intern("name");
    let age = interner.intern("age");
    let email = interner.intern("email");
    FxHashMap::from_iter([(
        interner.intern("user"),
        typed(
            Ty::Object(FxHashMap::from_iter([
                (name, Ty::String),
                (age, Ty::Int),
                (email, Ty::String),
            ])),
            Value::object(FxHashMap::from_iter([
                (name, Value::string("alice")),
                (age, Value::int(30)),
                (email, Value::string("alice@example.com")),
            ])),
        ),
    )])
}

pub fn items_context(interner: &Interner, items: Vec<i64>) -> Context {
    let len = items.len();
    FxHashMap::from_iter([(
        interner.intern("items"),
        typed(
            Ty::Array(Box::new(Ty::Int), LenTerm::Known(len)),
            Value::array(items.into_iter().map(Value::int).collect()),
        ),
    )])
}

// -- Fixture runner -----------------------------------------------

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

    let context: Context = match fixture.get("context") {
        Some(serde_json::Value::Object(fields)) => fields
            .iter()
            .map(|(k, v)| (interner.intern(k), value_from_json(&interner, v)))
            .collect(),
        Some(_) => return Err(format!("{}: 'context' must be an object", path.display())),
        None => Context::default(),
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
