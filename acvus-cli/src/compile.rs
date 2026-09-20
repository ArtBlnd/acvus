//! One file to one function, through the stages that can refuse it, with
//! every diagnostic collected: parse, typeck, lower, optimize — and
//! validate, which `acvus_mir::graph::optimize` runs over every module it
//! produces.

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{Externs, Registry};
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_interpreter::{AcvusRuntime, Executable, PrepareCtx, Prepared, prepare_module};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    CompilationGraph, Context, FnKind, Function, ParsedAst, QualifiedRef, extract, infer, lower,
    optimize,
};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::{PolyBuilder, Ty, TyTerm, lift_declaration, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Script,
    Template,
    Expr,
}

pub struct Diagnostic {
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

impl Timed {
    pub fn of(time: bool) -> Self {
        match time {
            true => Timed::On,
            false => Timed::Off,
        }
    }
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

/// A source every stage that can refuse it has accepted. Running it takes
/// one more stage, `prepare`, which belongs to the interpreter.
pub struct Checked {
    entry: QualifiedRef,
    modules: FxHashMap<QualifiedRef, MirModule>,
    externs: FxHashMap<QualifiedRef, Executable>,
    space: acvus_interpreter::SpaceHooksByType,
    fn_types: FxHashMap<QualifiedRef, Ty>,
    context_names: FxHashMap<QualifiedRef, Astr>,
    mir: String,
}

impl Checked {
    pub fn mir_dump(&self) -> &str {
        &self.mir
    }

    pub fn prepare(self, interner: &Interner) -> Compiled {
        let Checked {
            entry,
            modules,
            mut externs,
            space,
            fn_types,
            context_names,
            mir: _,
        } = self;
        let ctx = PrepareCtx {
            interner,
            externs: &externs,
            context_names: &context_names,
        };
        let prepared: Vec<(QualifiedRef, Executable)> = modules
            .iter()
            .map(|(q, m)| (*q, Executable::Module(Arc::new(prepare_module(m, &ctx)))))
            .collect();
        externs.extend(prepared);
        Compiled {
            entry,
            functions: externs,
            space,
            fn_types,
            context_names,
        }
    }
}

pub struct Compiled {
    pub entry: QualifiedRef,
    pub functions: FxHashMap<QualifiedRef, Executable>,
    pub space: acvus_interpreter::SpaceHooksByType,
    pub fn_types: FxHashMap<QualifiedRef, Ty>,
    pub context_names: FxHashMap<QualifiedRef, Astr>,
}

impl Compiled {
    /// The entry's prepared code: the bodies the machine would run.
    /// `prepare` prepares a module for every optimized module and the entry
    /// is a local function, never an extern handler.
    pub fn entry_prepared(&self) -> &Prepared {
        match self.functions.get(&self.entry) {
            Some(Executable::Module(prepared)) => prepared,
            Some(Executable::Extern(_)) => {
                panic!("the entry is a local function, not an extern handler")
            }
            None => panic!("the entry is prepared with every other module"),
        }
    }
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

pub fn check(
    interner: &Interner,
    source: &str,
    mode: Mode,
    context_types: &FxHashMap<Astr, Ty>,
    registries: Vec<Registry<AcvusRuntime>>,
    timed: Timed,
    opt: Opt,
) -> Result<(Checked, Option<CompileTimes>), Vec<Diagnostic>> {
    let mut stages = Stages::of(opt);
    let watch = Stopwatch::start(timed);
    let parsed = match mode {
        Mode::Script => acvus_ast::parse_script(interner, source).map(ParsedAst::Script),
        Mode::Expr => acvus_ast::parse_script(interner, source).map(ParsedAst::Script),
        Mode::Template => acvus_ast::parse(interner, source).map(ParsedAst::Template),
    };
    let parsed = parsed.map_err(|e| {
        vec![Diagnostic {
            message: e.kind.to_string(),
            primary: None,
            span: span_of(e.span),
            labels: Vec::new(),
        }]
    })?;
    stages.parse = watch.stop();

    let mut pb = PolyBuilder::new();
    let contexts: Vec<Context> = context_types
        .iter()
        .map(|(name, ty)| Context {
            qref: QualifiedRef::root(*name),
            ty: lift_declaration(ty, &mut pb),
        })
        .collect();
    let entry = QualifiedRef::root(interner.intern("main"));
    let mut functions = vec![Function {
        qref: entry,
        kind: FnKind::Local(parsed),
        ty: TyTerm::Fn {
            params: vec![],
            ret: Box::new(lift_declaration(&Ty::Never, &mut pb)),
            captures: vec![],
            effect: acvus_mir::ty::Effect::OPAQUE.into(),
        },
    }];
    let Externs {
        functions: extern_fns,
        types,
        handlers,
        space,
    } = Externs::combine(registries, interner).map_err(|e| {
        vec![Diagnostic {
            message: format!("the registries do not combine: {e}"),
            primary: None,
            span: None,
            labels: Vec::new(),
        }]
    })?;
    let fn_types: FxHashMap<QualifiedRef, Ty> = extern_fns
        .iter()
        .filter_map(|f| try_freeze_poly(&f.ty).map(|ty| (f.qref, ty)))
        .collect();
    functions.extend(extern_fns);
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        entry: Some(entry),
    };

    let watch = Stopwatch::start(timed);
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(
        interner,
        &graph,
        &ext,
        &FxHashMap::default(),
        Freeze::new(types),
    );
    stages.typeck = watch.stop();

    let mut diagnostics: Vec<Diagnostic> = inf
        .errors()
        .into_iter()
        .flat_map(|(_, errs)| errs.iter())
        .map(|e| Diagnostic {
            message: e.display(interner).to_string(),
            primary: e.primary(),
            span: span_of(e.span),
            labels: e.labels.clone(),
        })
        .collect();

    let watch = Stopwatch::start(timed);
    let lowered = lower::lower(interner, &graph, &ext, &inf);
    stages.lower = watch.stop();

    diagnostics.extend(
        lowered
            .errors
            .iter()
            .flat_map(|le| le.errors.iter())
            .map(|e| Diagnostic {
                message: e.display(interner).to_string(),
                primary: e.primary(),
                span: span_of(e.span),
                labels: e.labels.clone(),
            }),
    );
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }

    let watch = Stopwatch::start(timed);
    let optimized = optimize::optimize(lowered.modules, opt);
    stages.optimize = watch.stop();

    diagnostics.extend(
        optimized
            .errors
            .into_iter()
            .flat_map(|(_, errs)| errs)
            .map(|e| Diagnostic {
                message: e.display(interner).to_string(),
                primary: None,
                span: span_of(e.span),
                labels: e.labels().to_vec(),
            }),
    );
    if !diagnostics.is_empty() {
        return Err(diagnostics);
    }
    let mir = acvus_mir::printer::dump(
        interner,
        optimized
            .modules
            .get(&entry)
            .expect("the entry function lowers to a module"),
    );
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
            entry,
            modules: optimized.modules,
            externs,
            space,
            fn_types,
            context_names,
            mir,
        },
        stages.times(),
    ))
}
