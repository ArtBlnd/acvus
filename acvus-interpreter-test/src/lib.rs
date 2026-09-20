use std::collections::HashMap;
use std::sync::Arc;

pub mod listing;
pub mod scripts;

use acvus_extern::{Externs, Owned, Registry};
use acvus_interpreter::AcvusRuntime;
use acvus_interpreter::{
    ContextWrite, Executable, InMemoryContext, Interpreter, InterpreterContext, PrepareCtx,
    SequentialExecutor, Value, prepare_module,
};

/// What a run produced: its value, and the final value of every context it
/// assigned.
pub struct Ran {
    pub value: Value,
    pub writes: Vec<ContextWrite>,
}
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::*;
use acvus_mir::graph::{extract, lower as graph_lower, optimize as graph_optimize};
use acvus_mir::ty::{
    LenTerm, ObjectTy, PolyBuilder, PolyParam, Ty, TyTerm, lift_declaration, try_freeze_poly,
};
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

/// Split a context into the types the compiler is told and the values the
/// page is given.
pub fn split_context(
    interner: &Interner,
    context: Context,
) -> (FxHashMap<Astr, Ty>, HashMap<String, Owned<AcvusRuntime>>) {
    let mut types = FxHashMap::default();
    let mut snapshot = HashMap::new();
    for (name, TypedValue { ty, value }) in context {
        types.insert(name, ty);
        snapshot.insert(interner.resolve(name).to_string(), Owned::from_value(value));
    }
    (types, snapshot)
}

// -- Core pipeline -----------------------------------------------

/// Compile a template source -> MirModule + context id mapping.
pub struct CompileResult {
    pub entry_qref: QualifiedRef,
    pub modules: FxHashMap<QualifiedRef, acvus_mir::ir::MirModule>,
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
    compile_source_with_externs(interner, ast, context_types, std_regs, Ty::String)
}

fn compile_script(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    ret: Ty,
) -> CompileResult {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let std_regs = acvus_ext::std_registries::<AcvusRuntime>();
    compile_source_with_externs(interner, ast, context_types, std_regs, ret)
}

/// Parse a script-mode source and compile it: infer, lower, optimize.
pub fn compile_script_mode(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    ret: Ty,
) -> CompileResult {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    let std_regs = acvus_ext::std_registries::<AcvusRuntime>();
    compile_source_with_externs(interner, ast, context_types, std_regs, ret)
}

pub fn compile_source_with_externs(
    interner: &Interner,
    ast: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
) -> CompileResult {
    compile_source_with_externs_and_types(
        interner,
        ast,
        context_types,
        extern_registries,
        ret,
        |_| {},
    )
}

/// Compile with the given registries; `declare_types` registers the caller's
/// own type declarations into the combined type registry.
pub fn compile_source_with_externs_and_types<D>(
    interner: &Interner,
    ast: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
    declare_types: D,
) -> CompileResult
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    match check_source(
        interner,
        ast,
        context_types,
        extern_registries,
        ret,
        Opt::Full,
        declare_types,
    ) {
        Ok(cr) => cr,
        Err(Refusal {
            messages,
            dump: None,
        }) => panic!("compile failed:\n  {}", messages.join("\n  ")),
        Err(Refusal {
            messages,
            dump: Some(dump),
        }) => panic!(
            "optimize validation failed:\n  {}\n{dump}",
            messages.join("\n  ")
        ),
    }
}

/// Why a source did not reach a `CompileResult`. `dump` is the entry module
/// as it stood when validation refused it; the stages before optimization
/// have no module to show.
pub struct Refusal {
    pub messages: Vec<String>,
    pub dump: Option<String>,
}

/// A local function beside `main` in a compiled graph.
///
/// There is no result type here — not an omission, a decision. A helper's
/// result is the thing RFC-0064 step 1 is about, and a declared one would
/// let a test assert a result type the body never produced.
pub struct Helper<'a> {
    pub name: &'a str,
    pub source: &'a str,
    pub params: Vec<PolyParam>,
}

/// Every stage that can refuse a source, at the given optimization level.
pub fn check_source<D>(
    interner: &Interner,
    main: ParsedAst,
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
    opt: Opt,
    declare_types: D,
) -> Result<CompileResult, Refusal>
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    check_graph(
        interner,
        main,
        &[],
        context_types,
        extern_registries,
        ret,
        opt,
        declare_types,
    )
}

pub fn check_graph<D>(
    interner: &Interner,
    main: ParsedAst,
    helpers: &[Helper<'_>],
    context_types: &FxHashMap<Astr, Ty>,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
    opt: Opt,
    declare_types: D,
) -> Result<CompileResult, Refusal>
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

    let entry_qref = QualifiedRef::root(interner.intern("main"));
    let mut functions = Vec::new();
    functions.push(Function {
        qref: entry_qref,
        kind: FnKind::Local(main),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_declaration(&ret, &mut pb)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    });

    for helper in helpers {
        let parsed = match acvus_ast::parse_script(interner, helper.source) {
            Ok(script) => script,
            Err(e) => {
                return Err(Refusal {
                    messages: vec![format!("[{}] parse error: {e:?}", helper.name)],
                    dump: None,
                });
            }
        };
        functions.push(Function {
            qref: QualifiedRef::root(interner.intern(helper.name)),
            kind: FnKind::Local(ParsedAst::Script(parsed)),
            ty: TyTerm::Fn {
                params: helper.params.clone(),
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: acvus_mir::ty::Effect::OPAQUE.into(),
            },
        });
    }

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
        return Err(Refusal {
            messages: all_errors,
            dump: None,
        });
    }

    let opt_result = graph_optimize::optimize(
        result.modules.clone(),
        &inf.context_types,
        &FxHashSet::default(),
        opt,
    );

    // Report validation errors from optimization.
    for (qref, errs) in &opt_result.errors {
        let fn_name = interner.resolve(qref.name);
        for e in errs {
            all_errors.push(format!("[validate:{fn_name}] {}", e.display(interner)));
        }
    }
    if !all_errors.is_empty() {
        let dump = match opt_result.modules.get(&entry_qref) {
            Some(m) => acvus_mir::printer::dump_with(interner, m),
            None => "no entry module".to_string(),
        };
        return Err(Refusal {
            messages: all_errors,
            dump: Some(dump),
        });
    }

    let modules = opt_result.modules;

    // Build context qref -> name mapping.
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|ctx| (ctx.qref, ctx.qref.name))
        .collect();

    Ok(CompileResult {
        entry_qref,
        modules,
        context_names,
        fn_types,
        extern_executables,
    })
}

/// Build the runtime for a `CompileResult`: shared context, page, interpreter.
pub fn execute_compiled(
    interner: &Interner,
    cr: CompileResult,
    snapshot: HashMap<String, Owned<AcvusRuntime>>,
    executor: Arc<dyn acvus_interpreter::Executor>,
) -> (InterpreterContext, Interpreter) {
    let mut functions = cr.extern_executables;
    let ctx = PrepareCtx {
        interner,
        externs: &functions,
        context_names: &cr.context_names,
    };
    let prepared: Vec<(QualifiedRef, Executable)> = cr
        .modules
        .iter()
        .map(|(qref, module)| {
            (
                *qref,
                Executable::Module(Arc::new(prepare_module(module, &ctx))),
            )
        })
        .collect();
    functions.extend(prepared);
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
    if let Some(module) = cr.modules.get(&cr.entry_qref) {
        let ir = acvus_mir::printer::dump_with(interner, module);
        eprintln!("=== IR for entry ===\n{ir}");
        for (label, closure) in &module.closures {
            eprintln!("=== Closure {label:?} ===");
            for (i, inst) in closure.insts.iter().enumerate() {
                eprintln!("  {i}: {:?}", inst.kind);
            }
        }
    }

    let (shared, mut interp) =
        execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    let result = interp.execute().await;

    // A template yields a String; an empty one yields unit.
    match &result {
        v if v.is_string() => {
            // SAFETY: the witness is String.
            unsafe { v.as_str() }.to_owned()
        }
        v if v.kind().is_inline() => String::new(),
        other => format!("{other:?}"),
    }
}

/// Simple: no context.
pub async fn run_simple(source: &str) -> String {
    let interner = Interner::new();
    run(&interner, source, Context::default()).await
}

/// Compile and execute a **script**, returning the result Value.
pub async fn run_script(interner: &Interner, source: &str, context: Context, ret: Ty) -> Value {
    let (context_types, snapshot) = split_context(interner, context);
    let cr = compile_script(interner, source, &context_types, ret);
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    interp.execute().await
}

/// Compile and execute a **script-mode** (keyword syntax: let/for/while/if), returning the result Value.
pub async fn run_script_mode(
    interner: &Interner,
    source: &str,
    context: Context,
    ret: Ty,
) -> Value {
    let (context_types, snapshot) = split_context(interner, context);
    let cr = compile_script_mode(interner, source, &context_types, ret);
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, Arc::new(SequentialExecutor));
    interp.execute().await
}

/// Compile and execute a script with ExternFn registries, returning (result, context writes).
pub async fn run_script_with_externs(
    interner: &Interner,
    source: &str,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
) -> Ran {
    run_script_with_externs_and_types(interner, source, context, extern_registries, ret, |_| {})
        .await
}

/// Run a script with the given registries; `declare_types` registers the
/// caller's own type declarations into the combined type registry.
pub async fn run_script_with_externs_and_types<D>(
    interner: &Interner,
    source: &str,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
    declare_types: D,
) -> Ran
where
    D: FnOnce(&mut acvus_mir::ty::TypeRegistry),
{
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    run_parsed_with_externs(
        interner,
        ast,
        context,
        extern_registries,
        ret,
        declare_types,
    )
    .await
}

/// Run a script-mode source (keyword syntax) with the given registries.
pub async fn run_script_mode_with_externs(
    interner: &Interner,
    source: &str,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
) -> Ran {
    let ast = ParsedAst::Script(acvus_ast::parse_script(interner, source).expect("parse error"));
    run_parsed_with_externs(interner, ast, context, extern_registries, ret, |_| {}).await
}

/// Run an already parsed script against `context` with the given registries.
pub async fn run_parsed_with_externs<D>(
    interner: &Interner,
    ast: ParsedAst,
    context: Context,
    extern_registries: Vec<Registry<AcvusRuntime>>,
    ret: Ty,
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
        ret,
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
    ret: Ty,
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
        ret,
        declare_types,
    );
    let (_, mut interp) = execute_compiled(interner, cr, snapshot, executor);
    let value = interp.execute().await;
    let writes = interp.take_writes();
    Ran { value, writes }
}

// -- JSON helpers -------------------------------------------------

/// A JSON value with the type it names: the JSON shape is the type.
pub fn value_from_json(interner: &Interner, v: &serde_json::Value) -> TypedValue {
    match v {
        serde_json::Value::Number(n) => match n.as_i64() {
            Some(i) => typed(Ty::I64, Value::int(i)),
            None => typed(
                Ty::Float,
                Value::float(n.as_f64().expect("a JSON number is i64 or f64")),
            ),
        },
        serde_json::Value::String(s) => typed(Ty::String, Value::string(s.as_str())),
        serde_json::Value::Bool(b) => typed(Ty::Bool, Value::bool_(*b)),
        serde_json::Value::Null => typed(Ty::Unit, Value::unit()),
        serde_json::Value::Array(items) => {
            let items: Vec<TypedValue> =
                items.iter().map(|v| value_from_json(interner, v)).collect();
            let elem = items.first().map(|t| t.ty.clone()).unwrap_or(Ty::I64);
            let len = items.len();
            typed(
                Ty::Array(Box::new(elem), LenTerm::Known(len)),
                Value::array(
                    items
                        .into_iter()
                        .map(|t| Owned::from_value(t.value))
                        .collect(),
                ),
            )
        }
        serde_json::Value::Object(fields) => {
            let mut tys = FxHashMap::default();
            let mut values = Vec::new();
            for (k, v) in fields {
                let key = interner.intern(k);
                let TypedValue { ty, value } = value_from_json(interner, v);
                tys.insert(key, ty);
                values.push((key, Owned::from_value(value)));
            }
            typed(
                Ty::Object(ObjectTy::written(tys)),
                Value::object_by_name(interner, values),
            )
        }
    }
}

// -- Context helpers ----------------------------------------------

pub fn int_context(interner: &Interner, name: &str, value: i64) -> Context {
    FxHashMap::from_iter([(interner.intern(name), typed(Ty::I64, Value::int(value)))])
}

pub fn string_context(interner: &Interner, name: &str, value: &str) -> Context {
    FxHashMap::from_iter([(
        interner.intern(name),
        typed(Ty::String, Value::string(value)),
    )])
}

pub fn user_context(interner: &Interner) -> Context {
    let name = interner.intern("name");
    let age = interner.intern("age");
    let email = interner.intern("email");
    FxHashMap::from_iter([(
        interner.intern("user"),
        typed(
            Ty::Object(ObjectTy::written(FxHashMap::from_iter([
                (name, Ty::String),
                (age, Ty::I64),
                (email, Ty::String),
            ]))),
            Value::object_by_name(
                interner,
                [
                    (name, Owned::from_value(Value::string("alice"))),
                    (age, Owned::from_value(Value::int(30))),
                    (email, Owned::from_value(Value::string("alice@example.com"))),
                ],
            ),
        ),
    )])
}

pub fn items_context(interner: &Interner, items: Vec<i64>) -> Context {
    let len = items.len();
    FxHashMap::from_iter([(
        interner.intern("items"),
        typed(
            Ty::Array(Box::new(Ty::I64), LenTerm::Known(len)),
            Value::array(
                items
                    .into_iter()
                    .map(|n| Owned::from_value(Value::int(n)))
                    .collect(),
            ),
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

/// The corpus: every script the workspace's tests hand a harness, and one
/// way to compile, prepare and run each of them.
///
/// A registry the harnesses append to at compile time would need a macro at
/// each of the 638 call sites, so the collection is the scanner this module
/// holds: it reads the test sources, finds each harness call by name, and
/// takes the string literal that call receives. What it cannot resolve --
/// a source built by `format!`, a source that arrives as a function
/// parameter -- it reports as a gap rather than dropping.
pub mod corpus {
    use std::collections::{HashMap, HashSet};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use acvus_interpreter::{
        Composite, Executable, InMemoryContext, Interpreter, InterpreterContext, Kind, PrepareCtx,
        SequentialExecutor, Value, prepare_module,
    };
    use acvus_mir::graph::{ParsedAst, QualifiedRef};
    use acvus_mir::ty::{IntTy, Ty};
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;

    use crate::Opt;

    /// One source the corpus holds, with the call site it was written at.
    #[derive(Clone)]
    pub struct Script {
        pub origin: String,
        pub source: String,
    }

    pub struct Collection {
        pub scripts: Vec<Script>,
        /// Harness calls the scanner found, whether or not it recovered a
        /// source from them.
        pub calls: usize,
        /// Each call whose source the scanner could not recover, as
        /// `file:line the_harness(the-reason)`.
        pub gaps: Vec<String>,
    }

    /// Every harness that takes a program's source as a string. A test file's
    /// own function that calls one of these is a harness too, and
    /// [`local_harnesses`] finds those.
    const HARNESSES: &[&str] = &[
        "run",
        "run_simple",
        "run_script",
        "run_script_mode",
        "run_script_with_externs",
        "run_script_with_externs_and_types",
        "run_script_mode_with_externs",
        "compile_to_ir",
        "compile_to_ir_with",
        "compile_simple",
        "compile_script_ir",
        "compile_script_ir_with",
        "compile_script_raw",
        "compile_script_mode_raw",
        "compile_script_mode_ir_with",
        "compile_script_optimized",
        "compile_script_mode_optimized",
        "compile_script_mode",
        "compile_inline_ir",
        "compile_inline_ir_with",
        "compile_multi_fn_raw",
        "compile_multi_fn_optimized",
        "lowered_script_module",
        "declared_script_module",
        "optimized_script_module",
        "refuse_script_mode_optimized",
        "script_listing_with_externs",
    ];

    pub fn test_dirs() -> Vec<PathBuf> {
        let root = workspace_root();
        vec![
            root.join("acvus-mir-test/tests"),
            root.join("acvus-interpreter-test/tests"),
        ]
    }

    pub fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crate sits in the workspace")
            .to_path_buf()
    }

    pub fn collect() -> Collection {
        let mut sources: Vec<PathBuf> = Vec::new();
        for dir in test_dirs() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
                .map(|e| e.expect("a directory entry").path())
                .filter(|p| p.extension().is_some_and(|e| e == "rs"))
                .collect();
            entries.sort();
            sources.append(&mut entries);
        }

        let files: Vec<(PathBuf, Vec<Token>)> = sources
            .iter()
            .map(|path| {
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                (path.clone(), tokens(&text))
            })
            .collect();

        // A `const` a test file holds is visible to every other one through
        // its module path, so the bindings are one table over the corpus.
        let mut bindings: HashMap<String, String> = HashMap::new();
        for (path, toks) in &files {
            bindings.extend(string_bindings(path, toks));
        }
        let shared = workspace_root().join("acvus-interpreter-test/src/scripts.rs");
        let scripts_rs = std::fs::read_to_string(&shared).expect("the shared scripts module");
        bindings.extend(string_bindings(&shared, &tokens(&scripts_rs)));

        let root = workspace_root();
        let mut seen: HashSet<String> = HashSet::new();
        let mut scripts = Vec::new();
        let mut calls = 0;
        let mut gaps = Vec::new();
        for (path, toks) in &files {
            let names = local_harnesses(toks);
            let shown = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .display()
                .to_string();
            for call in harness_calls(toks, &names) {
                calls += 1;
                let origin = format!("{shown}:{} {}", call.line, call.name);
                match call.source(&bindings) {
                    Ok(source) => {
                        if seen.insert(source.clone()) {
                            scripts.push(Script { origin, source });
                        }
                    }
                    Err(reason) => gaps.push(format!("{origin}({reason})")),
                }
            }
        }
        Collection {
            scripts,
            calls,
            gaps,
        }
    }

    // -- The scanner -------------------------------------------------

    #[derive(Clone, PartialEq, Eq, Debug)]
    enum Tok {
        Ident(String),
        Str(String),
        Punct(char),
    }

    #[derive(Clone)]
    pub struct Token {
        tok: Tok,
        line: usize,
    }

    fn tokens(text: &str) -> Vec<Token> {
        let c: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        let mut line = 1;
        while i < c.len() {
            let ch = c[i];
            if ch == '\n' {
                line += 1;
                i += 1;
                continue;
            }
            if ch.is_whitespace() {
                i += 1;
                continue;
            }
            if ch == '/' && c.get(i + 1) == Some(&'/') {
                while i < c.len() && c[i] != '\n' {
                    i += 1;
                }
                continue;
            }
            if ch == '/' && c.get(i + 1) == Some(&'*') {
                let mut depth = 0;
                while i < c.len() {
                    if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                        depth += 1;
                        i += 2;
                        continue;
                    }
                    if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                        continue;
                    }
                    if c[i] == '\n' {
                        line += 1;
                    }
                    i += 1;
                }
                continue;
            }
            if ch == 'r' && matches!(c.get(i + 1), Some('"') | Some('#')) {
                let mut j = i + 1;
                let mut hashes = 0;
                while c.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if c.get(j) == Some(&'"') {
                    j += 1;
                    let start = j;
                    while j < c.len() {
                        if c[j] == '"' {
                            let closing = (1..=hashes).all(|h| c.get(j + h) == Some(&'#'));
                            if closing {
                                break;
                            }
                        }
                        j += 1;
                    }
                    let content: String = c[start..j.min(c.len())].iter().collect();
                    let at = line;
                    line += content.matches('\n').count();
                    out.push(Token {
                        tok: Tok::Str(content),
                        line: at,
                    });
                    i = j + 1 + hashes;
                    continue;
                }
            }
            if ch == '"' {
                let at = line;
                let (content, next) = string_literal(&c, i, &mut line);
                out.push(Token {
                    tok: Tok::Str(content),
                    line: at,
                });
                i = next;
                continue;
            }
            if ch == '\'' {
                if c.get(i + 1) == Some(&'\\') {
                    let mut j = i + 2;
                    while j < c.len() && c[j] != '\'' {
                        j += 1;
                    }
                    i = j + 1;
                    continue;
                }
                if c.get(i + 2) == Some(&'\'') {
                    i += 3;
                    continue;
                }
                i += 1;
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    i += 1;
                }
                continue;
            }
            if ch.is_alphabetic() || ch == '_' {
                let start = i;
                while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    i += 1;
                }
                out.push(Token {
                    tok: Tok::Ident(c[start..i].iter().collect()),
                    line,
                });
                continue;
            }
            out.push(Token {
                tok: Tok::Punct(ch),
                line,
            });
            i += 1;
        }
        out
    }

    /// The text a `"..."` literal stands for, with its escapes spelled out:
    /// the scanner hands the compiler the same bytes the Rust compiler hands
    /// the harness.
    fn string_literal(c: &[char], start: usize, line: &mut usize) -> (String, usize) {
        let mut s = String::new();
        let mut j = start + 1;
        while j < c.len() && c[j] != '"' {
            if c[j] != '\\' {
                if c[j] == '\n' {
                    *line += 1;
                }
                s.push(c[j]);
                j += 1;
                continue;
            }
            j += 1;
            match c.get(j) {
                None => break,
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some('r') => s.push('\r'),
                Some('0') => s.push('\0'),
                Some('\n') => {
                    *line += 1;
                    j += 1;
                    while j < c.len() && c[j].is_whitespace() {
                        if c[j] == '\n' {
                            *line += 1;
                        }
                        j += 1;
                    }
                    continue;
                }
                Some('x') => {
                    let hex: String = c[(j + 1).min(c.len())..(j + 3).min(c.len())]
                        .iter()
                        .collect();
                    if let Ok(n) = u8::from_str_radix(&hex, 16) {
                        s.push(n as char);
                    }
                    j += 2;
                }
                Some('u') => {
                    let mut k = j + 1;
                    if c.get(k) == Some(&'{') {
                        k += 1;
                        let from = k;
                        while k < c.len() && c[k] != '}' {
                            k += 1;
                        }
                        let hex: String = c[from..k].iter().collect();
                        if let Ok(n) = u32::from_str_radix(&hex, 16)
                            && let Some(ch) = char::from_u32(n)
                        {
                            s.push(ch);
                        }
                        j = k;
                    }
                }
                Some(other) => s.push(*other),
            }
            j += 1;
        }
        (s, j + 1)
    }

    /// `let`, `const` and `static` bindings whose value is a string literal or
    /// an `include_str!`, keyed by name.
    fn string_bindings(path: &Path, toks: &[Token]) -> HashMap<String, String> {
        let dir = path.parent().expect("a file sits in a directory");
        let mut out = HashMap::new();
        for i in 0..toks.len() {
            let Tok::Ident(keyword) = &toks[i].tok else {
                continue;
            };
            if !matches!(keyword.as_str(), "let" | "const" | "static") {
                continue;
            }
            let Some(Tok::Ident(name)) = toks.get(i + 1).map(|t| &t.tok) else {
                continue;
            };
            let Some(eq) = (i + 2..toks.len())
                .take_while(|&j| toks[j].tok != Tok::Punct(';'))
                .find(|&j| toks[j].tok == Tok::Punct('='))
            else {
                continue;
            };
            match toks.get(eq + 1).map(|t| &t.tok) {
                Some(Tok::Str(s)) => {
                    out.insert(name.clone(), s.clone());
                }
                Some(Tok::Ident(macro_name)) if macro_name == "include_str" => {
                    if let Some(Tok::Str(rel)) = toks.get(eq + 4).map(|t| &t.tok)
                        && let Ok(text) = std::fs::read_to_string(dir.join(rel))
                    {
                        out.insert(name.clone(), text);
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The file's own functions that reach a harness, to a fixed point: a
    /// test that wraps `run_script` in a local `check` writes its scripts at
    /// the calls of `check`.
    fn local_harnesses(toks: &[Token]) -> HashSet<String> {
        let mut names: HashSet<String> = HARNESSES.iter().map(|h| h.to_string()).collect();
        let bodies: Vec<(String, std::ops::Range<usize>)> = (0..toks.len())
            .filter(|&i| toks[i].tok == Tok::Ident("fn".to_string()))
            .filter_map(|i| {
                let Some(Tok::Ident(name)) = toks.get(i + 1).map(|t| &t.tok) else {
                    return None;
                };
                let open = (i + 2..toks.len()).find(|&j| toks[j].tok == Tok::Punct('{'))?;
                let mut depth = 0;
                let mut close = toks.len();
                for j in open..toks.len() {
                    match toks[j].tok {
                        Tok::Punct('{') => depth += 1,
                        Tok::Punct('}') => {
                            depth -= 1;
                            if depth == 0 {
                                close = j;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                Some((name.clone(), open..close))
            })
            .collect();
        loop {
            let grown = bodies.iter().any(|(name, body)| {
                !names.contains(name)
                    && body.clone().any(|j| {
                        matches!(&toks[j].tok, Tok::Ident(id) if names.contains(id))
                            && toks.get(j + 1).map(|t| &t.tok) == Some(&Tok::Punct('('))
                    })
            });
            if !grown {
                return names;
            }
            for (name, body) in &bodies {
                let calls = body.clone().any(|j| {
                    matches!(&toks[j].tok, Tok::Ident(id) if names.contains(id))
                        && toks.get(j + 1).map(|t| &t.tok) == Some(&Tok::Punct('('))
                });
                if calls {
                    names.insert(name.clone());
                }
            }
        }
    }

    struct Call {
        name: String,
        line: usize,
        args: Vec<Vec<Tok>>,
    }

    impl Call {
        /// The first argument that is a string literal, or a name bound to
        /// one. An argument built by a macro other than `include_str!` is not
        /// a source the scanner can read.
        fn source(&self, bindings: &HashMap<String, String>) -> Result<String, String> {
            for arg in &self.args {
                if arg
                    .iter()
                    .zip(arg.iter().skip(1))
                    .any(|(a, b)| matches!(a, Tok::Ident(_)) && *b == Tok::Punct('!'))
                {
                    continue;
                }
                if let Some(Tok::Str(s)) = arg.iter().find(|t| matches!(t, Tok::Str(_))) {
                    return Ok(s.clone());
                }
                if let Some(s) = arg.iter().rev().find_map(|t| match t {
                    Tok::Ident(name) => bindings.get(name),
                    _ => None,
                }) {
                    return Ok(s.clone());
                }
            }
            Err(
                match self
                    .args
                    .iter()
                    .flatten()
                    .any(|t| matches!(t, Tok::Ident(id) if id == "format" || id == "concat"))
                {
                    true => "a source built by a macro".to_string(),
                    false => "no literal among the arguments".to_string(),
                },
            )
        }
    }

    fn harness_calls(toks: &[Token], names: &HashSet<String>) -> Vec<Call> {
        let mut out = Vec::new();
        for i in 0..toks.len() {
            let Tok::Ident(name) = &toks[i].tok else {
                continue;
            };
            if !names.contains(name) || toks.get(i + 1).map(|t| &t.tok) != Some(&Tok::Punct('(')) {
                continue;
            }
            if matches!(toks.get(i.wrapping_sub(1)).map(|t| &t.tok), Some(Tok::Ident(kw)) if kw == "fn")
            {
                continue;
            }
            let mut args: Vec<Vec<Tok>> = Vec::new();
            let mut arg: Vec<Tok> = Vec::new();
            let mut depth = 0;
            for t in &toks[i + 1..] {
                match &t.tok {
                    Tok::Punct(open @ ('(' | '[' | '{')) => {
                        depth += 1;
                        if depth > 1 {
                            arg.push(Tok::Punct(*open));
                        }
                    }
                    Tok::Punct(close @ (')' | ']' | '}')) => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                        arg.push(Tok::Punct(*close));
                    }
                    Tok::Punct(',') if depth == 1 => args.push(std::mem::take(&mut arg)),
                    other => arg.push(other.clone()),
                }
            }
            if !arg.is_empty() {
                args.push(arg);
            }
            out.push(Call {
                name: name.clone(),
                line: toks[i].line,
                args,
            });
        }
        out
    }

    // -- Compiling, preparing and running one script -----------------

    /// How far [`attempt`] goes.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Stage {
        Prepare,
        Run,
    }

    /// What one attempt at one level produced.
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub enum Outcome {
        /// A stage before the machine refused the source.
        Refused(String),
        /// A stage before the machine panicked instead of refusing, with its
        /// message.
        CompilePanicked(String),
        /// `prepare` panicked, with its message.
        PreparePanicked(String),
        /// `prepare` returned, and [`Stage::Prepare`] asked for no more.
        Prepared,
        /// The run produced this value, rendered as JSON.
        Value(String),
        /// The run panicked, with its message.
        RunPanicked(String),
    }

    /// Why an attempt produced no outcome at all.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Lapse {
        TimedOut,
        /// The attempt took the process down: a signal, or an abort. A thread
        /// cannot hold that, which is why each attempt runs in a process of
        /// its own.
        Crashed,
    }

    /// Compile `source` at `opt`, the way `acvus run` compiles a file: no
    /// context declarations, the standard registries, `!` as the return type.
    pub fn attempt(source: &str, opt: Opt, stage: Stage) -> Outcome {
        let interner = Interner::new();
        let parsed = match acvus_ast::parse_script(&interner, source) {
            Ok(ast) => ParsedAst::Script(ast),
            Err(script) => match acvus_ast::parse(&interner, source) {
                Ok(ast) => ParsedAst::Template(ast),
                Err(_) => return Outcome::Refused(format!("parse: {}", script.kind)),
            },
        };
        let compiled = catch_unwind(AssertUnwindSafe(|| {
            crate::check_source(
                &interner,
                parsed,
                &FxHashMap::default(),
                acvus_ext::std_registries(),
                Ty::Never,
                opt,
                |_| {},
            )
        }));
        let cr = match compiled {
            Ok(Ok(cr)) => cr,
            Ok(Err(refusal)) => return Outcome::Refused(refusal.messages.join("; ")),
            Err(panic) => return Outcome::CompilePanicked(message(panic.as_ref())),
        };

        let mut functions = cr.extern_executables;
        let prepared = {
            let ctx = PrepareCtx {
                interner: &interner,
                externs: &functions,
                context_names: &cr.context_names,
            };
            catch_unwind(AssertUnwindSafe(|| {
                cr.modules
                    .iter()
                    .map(|(qref, module)| {
                        (
                            *qref,
                            Executable::Module(Arc::new(prepare_module(module, &ctx))),
                        )
                    })
                    .collect::<Vec<(QualifiedRef, Executable)>>()
            }))
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(panic) => return Outcome::PreparePanicked(message(panic.as_ref())),
        };
        if let Stage::Prepare = stage {
            return Outcome::Prepared;
        }

        functions.extend(prepared);
        let shared = InterpreterContext::new(&interner, functions, Arc::new(SequentialExecutor))
            .with_fn_types(cr.fn_types)
            .with_context_names(cr.context_names);
        let mut interp =
            Interpreter::new(shared, cr.entry_qref, InMemoryContext::new(HashMap::new()));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a current-thread runtime");
        match catch_unwind(AssertUnwindSafe(|| runtime.block_on(interp.execute()))) {
            Ok(value) => Outcome::Value(render(&interner, &value).to_string()),
            Err(panic) => Outcome::RunPanicked(message(panic.as_ref())),
        }
    }

    const SOURCE: &str = "ACVUS_CORPUS_SOURCE";
    const LEVEL: &str = "ACVUS_CORPUS_OPT";
    const UNTIL: &str = "ACVUS_CORPUS_STAGE";
    const MARK: &str = "acvus-corpus-outcome ";

    /// The child half of [`attempt_within`]: a test binary asked for one
    /// attempt runs it, writes the outcome, and exits. Asked for nothing it
    /// returns, so the test that calls it passes.
    pub fn child() {
        let Ok(source) = std::env::var(SOURCE) else {
            return;
        };
        std::panic::set_hook(Box::new(|_| {}));
        let opt = match std::env::var(LEVEL).as_deref() {
            Ok("none") => Opt::None,
            _ => Opt::Full,
        };
        let stage = match std::env::var(UNTIL).as_deref() {
            Ok("prepare") => Stage::Prepare,
            _ => Stage::Run,
        };
        let outcome = attempt(&source, opt, stage);
        println!("{MARK}{}", encode(&outcome));
        use std::io::Write;
        std::io::stdout()
            .flush()
            .expect("the outcome reaches the pipe");
        std::process::exit(0);
    }

    fn encode(outcome: &Outcome) -> String {
        let (tag, text) = match outcome {
            Outcome::Refused(text) => ("refused", text.as_str()),
            Outcome::CompilePanicked(text) => ("compile-panicked", text.as_str()),
            Outcome::PreparePanicked(text) => ("prepare-panicked", text.as_str()),
            Outcome::Prepared => ("prepared", ""),
            Outcome::Value(text) => ("value", text.as_str()),
            Outcome::RunPanicked(text) => ("run-panicked", text.as_str()),
        };
        serde_json::json!({ "tag": tag, "text": text }).to_string()
    }

    fn decode(line: &str) -> Option<Outcome> {
        let json: serde_json::Value = serde_json::from_str(line).ok()?;
        let text = json["text"].as_str()?.to_string();
        match json["tag"].as_str()? {
            "refused" => Some(Outcome::Refused(text)),
            "compile-panicked" => Some(Outcome::CompilePanicked(text)),
            "prepare-panicked" => Some(Outcome::PreparePanicked(text)),
            "prepared" => Some(Outcome::Prepared),
            "value" => Some(Outcome::Value(text)),
            "run-panicked" => Some(Outcome::RunPanicked(text)),
            _ => None,
        }
    }

    /// [`attempt`] in a process of its own, so that a program which does not
    /// finish costs the sweep `limit`, and one which takes its process down
    /// costs it one outcome.
    pub fn attempt_within(
        source: &str,
        opt: Opt,
        stage: Stage,
        limit: Duration,
    ) -> Result<Outcome, Lapse> {
        let exe = std::env::current_exe().expect("the test binary knows its own path");
        let mut child = std::process::Command::new(exe)
            .args(["corpus_child", "--exact", "--nocapture", "--test-threads=1"])
            .env(SOURCE, source)
            .env(
                LEVEL,
                match opt {
                    Opt::None => "none",
                    Opt::Full => "full",
                },
            )
            .env(
                UNTIL,
                match stage {
                    Stage::Prepare => "prepare",
                    Stage::Run => "run",
                },
            )
            .env("RUST_BACKTRACE", "0")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("the test binary spawns");

        let mut pipe = child.stdout.take().expect("stdout is piped");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut pipe, &mut text);
            let _ = tx.send(text);
        });

        let deadline = std::time::Instant::now() + limit;
        loop {
            match child.try_wait().expect("the child's status") {
                Some(_) => break,
                None if std::time::Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(Lapse::TimedOut);
                }
                None => std::thread::sleep(Duration::from_millis(10)),
            }
        }
        let text = rx.recv_timeout(Duration::from_secs(10)).unwrap_or_default();
        text.lines()
            .filter_map(|line| line.split_once(MARK))
            .find_map(|(_, outcome)| decode(outcome))
            .ok_or(Lapse::Crashed)
    }

    fn message(panic: &(dyn std::any::Any + Send)) -> String {
        if let Some(text) = panic.downcast_ref::<&'static str>() {
            return (*text).to_string();
        }
        match panic.downcast_ref::<String>() {
            Some(text) => text.clone(),
            None => "a panic with a payload that is not a message".to_string(),
        }
    }

    /// A value by its kind, the reading `acvus run` prints (RFC-0054). A
    /// reference reads as what it names, so the two levels are compared on
    /// what the program computed and not on where it put it.
    pub fn render(interner: &Interner, value: &Value) -> serde_json::Value {
        use serde_json::Value as Json;
        match value.kind() {
            Kind::I8 => int(IntTy::I8, value),
            Kind::I16 => int(IntTy::I16, value),
            Kind::I32 => int(IntTy::I32, value),
            Kind::I64 => int(IntTy::I64, value),
            Kind::U8 => int(IntTy::U8, value),
            Kind::U16 => int(IntTy::U16, value),
            Kind::U32 => int(IntTy::U32, value),
            Kind::U64 => int(IntTy::U64, value),
            Kind::F64 => Json::from(value.as_float()),
            Kind::Char => Json::from(
                char::from_u32(value.as_char())
                    .expect("as_char asserted a scalar value")
                    .to_string(),
            ),
            Kind::Bool => Json::from(value.as_bool()),
            Kind::Unit | Kind::None => Json::Null,
            // SAFETY: a reference names a live value for as long as it lives.
            Kind::Ref => render(interner, unsafe { value.target() }),
            Kind::Undef => Json::from("<undef>"),
            Kind::LargeRef => Json::from("<projection>"),
            Kind::Large => composite(interner, value),
        }
    }

    fn int(kind: IntTy, value: &Value) -> serde_json::Value {
        let v = kind.read(value.bits());
        match kind.signed() {
            true => serde_json::Value::from(v as i64),
            false => serde_json::Value::from(v as u64),
        }
    }

    fn composite(interner: &Interner, value: &Value) -> serde_json::Value {
        use serde_json::{Map, Value as Json};
        // SAFETY, every arm: the vtable's `Composite` is the runtime's own
        // witness of the type behind the pointer.
        match value.composite() {
            Some(Composite::String) => Json::from(unsafe { value.as_str() }),
            Some(Composite::Array) => Json::Array(
                unsafe { value.as_array() }
                    .iter()
                    .map(|v| render(interner, v))
                    .collect(),
            ),
            Some(Composite::Tuple) => Json::Array(
                unsafe { value.as_tuple() }
                    .iter()
                    .map(|v| render(interner, v))
                    .collect(),
            ),
            Some(Composite::Object) => {
                let mut fields: Vec<(String, Json)> = unsafe { value.as_shape() }
                    .names()
                    .iter()
                    .zip(unsafe { value.as_object() })
                    .map(|(k, v)| (interner.resolve(*k).to_string(), render(interner, v)))
                    .collect();
                fields.sort_by(|(a, _), (b, _)| a.cmp(b));
                Json::Object(fields.into_iter().collect())
            }
            Some(Composite::Variant) => {
                let variant = unsafe { value.as_variant() };
                // SAFETY: the same witness — a variant's first register is its tag.
                let tag = interner
                    .resolve(unsafe { variant.tag().as_tag() })
                    .to_string();
                match variant.payload().kind() {
                    Kind::Undef => Json::from(tag),
                    _ => Json::Object(Map::from_iter([(tag, render(interner, variant.payload()))])),
                }
            }
            Some(Composite::Fn | Composite::Handle) | None => {
                Json::from(format!("<{}>", (value.vtable().name)()))
            }
        }
    }
}
