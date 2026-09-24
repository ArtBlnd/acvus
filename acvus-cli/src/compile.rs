//! The sources of one run to one compilation graph, through the stages that
//! can refuse it, with every diagnostic collected: parse, typeck, lower,
//! optimize — and validate, which `acvus_mir::graph::optimize` runs over
//! every module it produces.
//!
//! Each source is one entry of the graph (RFC-0054 rule 1), or one
//! context's init. A context is declared by the sources that name it, at a
//! type the graph solves from every one of them and its init (RFC-0090
//! rule 1), so no data and no user states it.

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{CombineError, Externs, Handlers, Registry};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use std::collections::HashMap;

use acvus_interpreter::{
    AcvusRuntime, DeclaredInits, Executable, Executor, GraphParts, InitSource, Inits,
    InterpreterContext, PrepareCtx, Prepared, prepare_module,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Bindings, CompilationGraph, Context, ContextInfo, FnKind, Function, Parsed, ParsedAst,
    QualifiedRef, RecoveredAst, extract, infer, lower, optimize,
};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::{PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Script,
    Template,
    Expr,
}

/// One source of a compilation, and the path its diagnostics are rendered
/// at.
pub struct Unit {
    pub role: Role,
    pub path: String,
    pub mode: Mode,
    pub text: String,
}

pub enum Role {
    Entry(String),
    /// The init of the context of this name.
    Init(String),
}

pub struct Diagnostic {
    /// The unit whose text `span` points into; `None` for a refusal of the
    /// graph as a whole, which no source holds.
    pub unit: Option<usize>,
    pub message: String,
    /// The words the primary marker carries; `None` leaves it repeating
    /// the message where another span is marked.
    pub primary: Option<String>,
    pub span: Option<Span>,
    pub labels: Vec<Label>,
}

#[derive(Clone, Copy)]
pub enum Timed {
    On,
    Off,
}

pub struct Stopwatch(Option<Instant>);

impl Stopwatch {
    pub fn start(timed: Timed) -> Self {
        Stopwatch(match timed {
            Timed::On => Some(Instant::now()),
            Timed::Off => None,
        })
    }

    pub fn stop(self) -> Option<Duration> {
        Some(self.0?.elapsed())
    }
}

/// `typeck` covers two calls, `extract` and `infer`, and they are not
/// reported apart. Extraction builds the tables inference then solves; a
/// reader of the CLI's report has nothing to do with the split, and the
/// stage a script author knows by name is typechecking.
pub struct CompileTimes {
    pub opt: Opt,
    pub parse: Duration,
    pub typeck: Duration,
    pub lower: Duration,
    pub optimize: Duration,
}

struct Stages {
    opt: Opt,
    parse: Option<Duration>,
    typeck: Option<Duration>,
    lower: Option<Duration>,
    optimize: Option<Duration>,
}

impl Stages {
    fn of(opt: Opt) -> Self {
        Stages {
            opt,
            parse: None,
            typeck: None,
            lower: None,
            optimize: None,
        }
    }

    fn times(&self) -> Option<CompileTimes> {
        Some(CompileTimes {
            opt: self.opt,
            parse: self.parse?,
            typeck: self.typeck?,
            lower: self.lower?,
            optimize: self.optimize?,
        })
    }
}

/// A graph every stage that can refuse it has accepted. Running it takes
/// one more stage, `prepare`, which belongs to the interpreter.
pub struct Checked {
    /// The function each unit compiled to, in unit order.
    functions: Vec<QualifiedRef>,
    inputs: FxHashMap<QualifiedRef, Vec<ContextInfo>>,
    contexts: HashMap<String, Ty>,
    inits: Inits,
    modules: FxHashMap<QualifiedRef, MirModule>,
    externs: FxHashMap<QualifiedRef, Executable>,
    instances: acvus_extern::InstanceTable,
    space: acvus_interpreter::SpaceHooksByType,
    fn_types: FxHashMap<QualifiedRef, Ty>,
    context_names: FxHashMap<QualifiedRef, Astr>,
}

/// What a command reads of the one unit it is about.
pub struct Target {
    pub function: QualifiedRef,
    /// A `$` a binding fixed is not among these (RFC-0071 rule 5).
    pub inputs: Vec<ContextInfo>,
    /// The contexts a run fetches before assigning them, its callees'
    /// included (RFC-0025 rule 2), in page-key order.
    pub fetched_first: Vec<String>,
    pub mir: String,
}

impl Checked {
    pub fn target(&self, interner: &Interner, unit: usize) -> Target {
        let function = self.functions[unit];
        let module = &self.modules[&function];
        let mut fetched_first: Vec<String> = module
            .fetched_first
            .iter()
            .map(|context| interner.resolve(context.name).to_owned())
            .collect();
        fetched_first.sort();
        let mut inputs = self.inputs[&function].clone();
        inputs.sort_by_key(|input| input.name.name.bits());
        Target {
            function,
            inputs,
            fetched_first,
            mir: acvus_mir::printer::dump(interner, module),
        }
    }

    /// Every context the graph names, at the type it solved.
    pub fn contexts(&self) -> &HashMap<String, Ty> {
        &self.contexts
    }

    pub fn prepare(self, interner: &Interner) -> Compiled {
        let Checked {
            modules,
            mut externs,
            instances,
            space,
            fn_types,
            context_names,
            inits,
            ..
        } = self;
        let ctx = PrepareCtx {
            interner,
            externs: &externs,
            context_names: &context_names,
            instances: &instances,
        };
        let prepared: Vec<(QualifiedRef, Executable)> = modules
            .iter()
            .map(|(q, m)| (*q, Executable::Module(Arc::new(prepare_module(m, &ctx)))))
            .collect();
        externs.extend(prepared);
        Compiled {
            functions: externs,
            space,
            fn_types,
            context_names,
            inits,
        }
    }
}

pub struct Compiled {
    pub functions: FxHashMap<QualifiedRef, Executable>,
    pub space: acvus_interpreter::SpaceHooksByType,
    pub fn_types: FxHashMap<QualifiedRef, Ty>,
    pub context_names: FxHashMap<QualifiedRef, Astr>,
    pub inits: Inits,
}

/// A compilation ready to run: the runtime's shared state and the inits
/// that fill a page before a run.
pub struct Runnable {
    pub shared: InterpreterContext,
    pub inits: Inits,
}

impl Compiled {
    pub fn runnable(self, interner: &Interner, executor: Arc<dyn Executor>) -> Runnable {
        let Compiled {
            functions,
            space,
            fn_types,
            context_names,
            inits,
        } = self;
        let shared = InterpreterContext::new(interner, functions, executor)
            .with_fn_types(fn_types)
            .with_context_names(context_names)
            .with_space(space);
        Runnable { shared, inits }
    }

    /// A unit's prepared code: the bodies the machine would run.
    /// `prepare` prepares a module for every optimized module and a unit
    /// is a local function, never an extern handler.
    pub fn prepared(&self, function: &QualifiedRef) -> &Prepared {
        match self.functions.get(function) {
            Some(Executable::Module(prepared)) => prepared,
            Some(Executable::Extern(_)) => {
                panic!("a unit is a local function, not an extern handler")
            }
            None => panic!("a unit is prepared with every other module"),
        }
    }
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

pub struct Environment {
    pub graph: CompilationGraph,
    handlers: Handlers<AcvusRuntime>,
    space: acvus_interpreter::SpaceHooksByType,
    instances: acvus_extern::InstanceTable,
    fn_types: FxHashMap<QualifiedRef, Ty>,
}

/// The entry a lone source compiles to.
pub fn entry_ref(interner: &Interner) -> QualifiedRef {
    QualifiedRef::root(interner.intern("main"))
}

/// `!`: this host states no return type and prints whatever comes back
/// (RFC-0054 rule 5, RFC-0090 rule 6).
pub fn entry_ty() -> PolyTy {
    TyTerm::Fn {
        params: vec![],
        ret: Box::new(lift_declaration(&Ty::Never, &mut PolyBuilder::new())),
        captures: vec![],
        effect: acvus_mir::ty::Effect::OPAQUE.into(),
        flows: acvus_mir::ty::Flows::Every.into(),
    }
}

pub fn combine_refusal(error: &CombineError) -> String {
    format!("the registries do not combine: {error}")
}

pub fn unreadable_source(path: &Path, error: &io::Error) -> String {
    format!("{}: {error}", path.display())
}

/// The contexts a parsed source names.
pub fn context_refs(ast: &ParsedAst) -> FxHashSet<QualifiedRef> {
    match ast {
        ParsedAst::Script(script) => acvus_ast::extract_script_context_refs(script),
        ParsedAst::Template(template) => acvus_ast::extract_template_context_refs(template),
        ParsedAst::Recovered(RecoveredAst::Script(script)) => {
            acvus_ast::extract_script_context_refs(script)
        }
        ParsedAst::Recovered(RecoveredAst::Template(template)) => {
            acvus_ast::extract_template_context_refs(template)
        }
    }
}

pub fn parse(interner: &Interner, mode: Mode, text: &str) -> Parsed {
    match mode {
        Mode::Script | Mode::Expr => Parsed::script(acvus_ast::parse_script(interner, text)),
        Mode::Template => Parsed::template(acvus_ast::parse(interner, text)),
    }
}

/// The graph's environment: the registries' functions and types, and each
/// named context at a type variable the whole graph solves (RFC-0090
/// rule 1). `entries` are the graph's entries; the caller adds their
/// functions.
pub fn environment(
    interner: &Interner,
    named: FxHashSet<QualifiedRef>,
    entries: Vec<QualifiedRef>,
    bindings: Bindings,
    registries: Vec<Registry<AcvusRuntime>>,
) -> Result<Environment, CombineError> {
    let mut open = PolyBuilder::new();
    let mut contexts: Vec<Context> = named
        .into_iter()
        .map(|qref| Context {
            qref,
            ty: open.fresh_ty_var(),
            init: None,
        })
        .collect();
    contexts.sort_by_key(|context| context.qref.name.bits());
    let Externs {
        functions,
        types,
        handlers,
        space,
        instances,
    } = Externs::combine(registries, interner)?;
    let fn_types: FxHashMap<QualifiedRef, Ty> = functions
        .iter()
        .filter_map(|f| try_freeze_poly(&f.ty).map(|ty| (f.qref, ty)))
        .collect();
    Ok(Environment {
        graph: CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            types: Freeze::new(types),
            bindings,
            entries,
        },
        handlers,
        space,
        instances,
        fn_types,
    })
}

/// Compile `units` as one graph.
pub fn check(
    interner: &Interner,
    units: &[Unit],
    bindings: Bindings,
    registries: Vec<Registry<AcvusRuntime>>,
    timed: Timed,
    opt: Opt,
) -> Result<(Checked, Option<CompileTimes>), Vec<Diagnostic>> {
    let mut stages = Stages::of(opt);
    let refused = |unit: Option<usize>, message: String| Diagnostic {
        unit,
        message,
        primary: None,
        span: None,
        labels: Vec::new(),
    };

    let watch = Stopwatch::start(timed);
    let parsed: Vec<Parsed> = units
        .iter()
        .map(|unit| parse(interner, unit.mode, &unit.text))
        .collect();
    stages.parse = watch.stop();

    let entry_refs: Vec<QualifiedRef> = units
        .iter()
        .filter_map(|unit| match &unit.role {
            Role::Entry(name) => Some(QualifiedRef::root(interner.intern(name))),
            Role::Init(_) => None,
        })
        .collect();
    let named: FxHashSet<QualifiedRef> = parsed
        .iter()
        .zip(units)
        .filter(|(_, unit)| matches!(unit.role, Role::Entry(_)))
        .flat_map(|(parsed, _)| context_refs(&parsed.ast))
        .collect();
    let init_unit = |key: &str| {
        units
            .iter()
            .position(|unit| matches!(&unit.role, Role::Init(held) if held == key))
    };

    let Environment {
        graph: environment,
        handlers,
        space,
        instances,
        fn_types,
    } = environment(interner, named, entry_refs, bindings, registries)
        .map_err(|e| vec![refused(None, combine_refusal(&e))])?;

    // A recovered tree is checked for what parsed and never lowered, so its
    // parse errors and those refusals are reported together (RFC-0078).
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    for (unit, parsed) in parsed.iter().enumerate() {
        diagnostics.extend(parsed.errors.iter().map(|e| Diagnostic {
            unit: Some(unit),
            message: e.kind.to_string(),
            primary: None,
            span: span_of(e.span),
            labels: Vec::new(),
        }));
    }
    let clashes: Vec<Diagnostic> = units
        .iter()
        .enumerate()
        .filter_map(|(unit, held)| match &held.role {
            Role::Entry(name) => Some((unit, name)),
            Role::Init(_) => None,
        })
        .filter(|(_, name)| {
            let qref = QualifiedRef::root(interner.intern(name));
            environment.functions.iter().any(|f| f.qref == qref)
        })
        .map(|(unit, name)| {
            refused(
                Some(unit),
                format!(
                    "the script `{name}` has the name of an extern function; `acvus ctl space rm-script` and add it under another name"
                ),
            )
        })
        .collect();
    if !clashes.is_empty() {
        diagnostics.extend(clashes);
        return Err(diagnostics);
    }

    let mut functions: Vec<Function> = Vec::with_capacity(units.len() + environment.functions.len());
    let mut inits: Vec<InitSource> = Vec::new();
    for (parsed, unit) in parsed.into_iter().zip(units) {
        match &unit.role {
            Role::Entry(name) => functions.push(Function {
                qref: QualifiedRef::root(interner.intern(name)),
                kind: FnKind::Local(parsed.ast, acvus_mir::graph::Inputs::FromReads),
                ty: entry_ty(),
            }),
            Role::Init(key) => inits.push(InitSource {
                key: key.clone(),
                ast: parsed.ast,
            }),
        }
    }
    functions.extend(environment.functions.iter().cloned());
    let mut parts = GraphParts {
        open: PolyBuilder::new(),
        contexts: environment.contexts.to_vec(),
        functions,
        entries: environment.entries.clone(),
    };
    let declared = match DeclaredInits::declare(interner, inits, &mut parts) {
        Ok(declared) => declared,
        Err(refusals) => {
            diagnostics.extend(
                refusals
                    .iter()
                    .map(|refusal| refused(init_unit(refusal.key()), refusal.to_string())),
            );
            return Err(diagnostics);
        }
    };
    let GraphParts {
        contexts,
        functions,
        entries,
        ..
    } = parts;
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        entries,
        ..environment
    };
    let unit_functions: Vec<QualifiedRef> = units
        .iter()
        .map(|unit| match &unit.role {
            Role::Entry(name) => QualifiedRef::root(interner.intern(name)),
            Role::Init(key) => declared
                .function(key)
                .expect("`declare` declared every init it was given"),
        })
        .collect();
    let at = |qref: &QualifiedRef| unit_functions.iter().position(|held| held == qref);

    let watch = Stopwatch::start(timed);
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);
    stages.typeck = watch.stop();

    diagnostics.extend(inf.errors().into_iter().flat_map(|(qref, errs)| {
        let unit = at(&qref);
        errs.iter().map(move |e| Diagnostic {
            unit,
            message: e.display(interner).to_string(),
            primary: e.primary(),
            span: span_of(e.span),
            labels: e.labels.clone(),
        })
    }));

    let watch = Stopwatch::start(timed);
    let lowered = lower::lower(interner, &graph, &ext.view(), &inf);
    stages.lower = watch.stop();

    diagnostics.extend(lowered.errors.iter().flat_map(|le| {
        let unit = at(&le.fn_id);
        le.errors.iter().map(move |e| Diagnostic {
            unit,
            message: e.display(interner).to_string(),
            primary: e.primary(),
            span: span_of(e.span),
            labels: e.labels.clone(),
        })
    }));
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let watch = Stopwatch::start(timed);
    let optimized = optimize::optimize(
        interner,
        &acvus_mir::laws::LawTable::of(graph.functions.iter()),
        lowered.modules,
        opt,
    );
    stages.optimize = watch.stop();

    diagnostics.extend(optimized.errors.into_iter().flat_map(|(qref, errs)| {
        let unit = at(&qref);
        errs.into_iter().map(move |e| Diagnostic {
            unit,
            message: e.display(interner).to_string(),
            primary: None,
            span: span_of(e.span),
            labels: e.labels().to_vec(),
        })
    }));
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let name = |qref: &QualifiedRef| interner.resolve(qref.name).to_owned();
    let contexts: HashMap<String, Ty> = graph
        .contexts
        .iter()
        .map(|c| {
            let Some(ty) = inf.context_types.get(&c.qref) else {
                panic!("inference settles a type for every context of the graph")
            };
            (name(&c.qref), ty.clone())
        })
        .collect();
    let externs: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(q, h)| (q, Executable::Extern(h)))
        .collect();
    let context_names: FxHashMap<QualifiedRef, Astr> = graph
        .contexts
        .iter()
        .map(|c| (c.qref, c.qref.name))
        .collect();
    Ok((
        Checked {
            functions: unit_functions,
            inputs: optimized.inputs,
            inits: declared.solved(&contexts),
            contexts,
            modules: optimized.modules,
            externs,
            instances,
            space,
            fn_types,
            context_names,
        },
        stages.times(),
    ))
}
