//! A host compiles scripts with `Host`, and inside `Program::scope` opens a
//! page over a storage and runs entries on it (RFC-0090).
//!
//! A page and an entry belong to the one scope that made them, so a page
//! opened by one program never serves another's entry:
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Program};
//! async fn cross(a: &Program, b: &Program) {
//!     a.scope(async |a| {
//!         let mut page = a.open(MemoryStorage::new()).await.unwrap();
//!         b.scope(async |b| {
//!             let entry = b.entry::<(), i64>("main").unwrap();
//!             let _ = entry.run(&mut page, ()).await;
//!         })
//!         .await;
//!     })
//!     .await;
//! }
//! ```
//!
//! ```
//! use acvus_interpreter::{HostError, MemoryStorage, Program};
//! async fn same(a: &Program) -> Result<i64, HostError> {
//!     a.scope(async |a| {
//!         let mut page = a.open(MemoryStorage::new()).await?;
//!         let entry = a.entry::<(), i64>("main")?;
//!         entry.run(&mut page, ()).await?.with(|n: &i64| *n)
//!     })
//!     .await
//! }
//! ```
//!
//! and neither leaves the scope:
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Program};
//! async fn kept(a: &Program) {
//!     let _page = a.scope(async |a| a.open(MemoryStorage::new()).await).await;
//! }
//! ```
//!
//! A host lends a result or a context to a closure, and nothing the closure
//! is lent outlives it (RFC-0090 rule 5):
//!
//! ```compile_fail
//! use acvus_interpreter::{HostError, Output};
//! fn kept<'o>(output: &'o Output<'_, String>) -> Result<&'o str, HostError> {
//!     output.with(|s: &str| s)
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::{MemoryStorage, Page};
//! fn kept<'a>(page: &'a Page<'_, MemoryStorage>) -> Vec<&'a str> {
//!     let mut kept = Vec::new();
//!     let _ = page.with("name", |s: &str| kept.push(s));
//!     kept
//! }
//! ```
//!
//! A value the host keeps is a copy it makes in Rust, and `with` lends
//! shared alone, so a parameter that writes is lent by `with_mut`:
//!
//! ```
//! use acvus_interpreter::{HostError, MemoryStorage, Output, Page};
//! fn kept(output: &Output<'_, String>) -> Result<String, HostError> {
//!     output.with(|s: &str| s.to_owned())
//! }
//! fn pushed(page: &mut Page<'_, MemoryStorage>) -> Result<(), HostError> {
//!     page.with_mut("name", |s: &mut String| s.push('!'))
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::{HostError, Output};
//! fn pushed(output: &Output<'_, String>) -> Result<(), HostError> {
//!     output.with(|s: &mut String| s.push('!'))
//! }
//! ```
//!
//! A storage moves holders it cannot open, so implementing one is safe and
//! names no runtime value:
//!
//! ```
//! use std::collections::HashMap;
//! use acvus_interpreter::{Codec, Held, Storage, StorageError};
//! #[derive(Default)]
//! struct Shelf(HashMap<String, Held>);
//! impl Storage for Shelf {
//!     fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
//!         Ok(self.0.remove(key))
//!     }
//!     fn store(&mut self, key: &str, held: Held) {
//!         self.0.insert(key.to_owned(), held);
//!     }
//!     fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
//!         Ok(())
//!     }
//! }
//! ```

#![cfg_attr(
    not(feature = "tooling"),
    doc = r#"
A crate without the `tooling` feature cannot reach the runtime's value
word (RFC-0090 rule 6): not the machine's entry,

```compile_fail,E0603
use acvus_interpreter::machine::call_module;
```

not a space's raw load,

```compile_fail,E0624
fn load(space: &acvus_interpreter::Space, rt: &acvus_interpreter::AcvusRuntime, ty: &acvus_mir::ty::Ty) {
    let _ = space.load(rt, "x", ty);
}
```

not the word inside a holder a page moves,

```compile_fail,E0616
fn open(held: acvus_interpreter::Held) {
    let _ = held.value;
}
```

```compile_fail,E0624
fn into(held: acvus_interpreter::Held) {
    let _ = held.into_value();
}
```

not a raw read of an entry's result,

```compile_fail,E0599
fn raw(scope: acvus_interpreter::Scope<'_>) {
    let _ = scope.untyped_entry("main");
}
```

and not the value word's constructors or readers, through the runtime's
associated type,

```compile_fail,E0624
use acvus_extern::Runtime;
let _ = <acvus_interpreter::AcvusRuntime as Runtime>::Value::int(3);
```

```compile_fail,E0616
fn page(rt: &acvus_interpreter::AcvusRuntime) {
    let _ = &rt.page;
}
```
"#
)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_ast::Span;
use acvus_ast::report::Label;
use acvus_extern::{
    Borrows, CombineError, Cross, Crossing, Declared, Externs, Form, FormKind, Gives, Handlers,
    Holding, InstanceTable, Lendable, ObjectShape, OneValue, Owned, Registry, Returned, Shared,
    SpaceError, Uniform, Val,
};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Bindings, BoundValue, CompilationGraph, Context, ContextInfo, FnKind, Function, Inputs,
    NotABoundValue, Parsed, ParsedAst, QualifiedRef, RecoveredAst, extract, infer, lower,
    optimize,
};
use acvus_mir::ir::MirModule;
use acvus_mir::ty::{
    Effect, Flows, ParamTerm, Poly, PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration,
    try_freeze_poly,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::executor::Executor;
use crate::init::{DeclaredInits, GraphParts, InitSource, Inits};
use crate::interpreter::{Absent, Executable, Interpreter, InterpreterContext, lookup_module};
use crate::journal::{Held, RuntimeContext};
use crate::prepare::{PrepareCtx, prepare_module};
use crate::runtime::AcvusRuntime;
use crate::value::Value;

type Brand<'p> = PhantomData<fn(&'p ()) -> &'p ()>;

// -- Sources and refusals ------------------------------------------------

pub enum Source<'a> {
    Script(&'a str),
    Template(&'a str),
    Expr(&'a str),
}

macro_rules! source_parse {
    ($v:vis) => {
        impl Source<'_> {
            $v fn parse(&self, interner: &Interner) -> Parsed {
                match self {
                    Source::Script(text) | Source::Expr(text) => {
                        Parsed::script(acvus_ast::parse_script(interner, text))
                    }
                    Source::Template(text) => Parsed::template(acvus_ast::parse(interner, text)),
                }
            }
        }
    };
}
tooling_vis!(source_parse);

/// The body or binding whose source a refusal's span points into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    Entry(String),
    Init(String),
    Binding(String),
}

#[derive(Debug, Clone)]
pub struct Refusal {
    /// `None` for a refusal of the compilation as a whole.
    pub origin: Option<Origin>,
    pub message: String,
    pub span: Option<Span>,
    /// The words the primary marker carries; `None` repeats the message.
    pub primary: Option<String>,
    pub labels: Vec<Label>,
}

impl Refusal {
    fn of(origin: Option<Origin>, message: String) -> Self {
        Refusal {
            origin,
            message,
            span: None,
            primary: None,
            labels: Vec::new(),
        }
    }
}

// -- Errors --------------------------------------------------------------

/// What a host asked for that the compilation does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Named {
    Entry(String),
    Context(String),
}

/// Where a declared type and an asked one differ.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    Context(String),
    /// The inputs of the named entry.
    Inputs(String),
    /// The result of the named entry.
    Result(String),
}

#[derive(Debug)]
pub enum HostError {
    Refused(Vec<Refusal>),
    NotInGraph { what: Named },
    Mismatched { what: Part, held: String, asked: String },
    /// A context an entry fetches before assigning it that the page lacks,
    /// with no init to fill it.
    Unfilled { key: String },
    /// A context no init filled and no run has stored yet (RFC-0090 rule 4).
    Unstored { key: String },
    Storage(StorageError),
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostError::Refused(refusals) => {
                write!(f, "the compilation is refused")?;
                for refusal in refusals {
                    write!(f, "; {}", refusal.message)?;
                }
                Ok(())
            }
            HostError::NotInGraph {
                what: Named::Entry(name),
            } => write!(f, "the compilation has no entry `{name}`"),
            HostError::NotInGraph {
                what: Named::Context(key),
            } => write!(f, "the compilation has no `@{key}`"),
            HostError::Mismatched {
                what: Part::Context(key),
                held,
                asked,
            } => write!(f, "`@{key}` is held as {held}, and was asked for as {asked}"),
            HostError::Mismatched {
                what: Part::Inputs(name),
                held,
                asked,
            } => write!(
                f,
                "the entry `{name}` was declared to take {held}, and was asked for taking {asked}"
            ),
            HostError::Mismatched {
                what: Part::Result(name),
                held,
                asked,
            } => write!(
                f,
                "the entry `{name}` was declared to return {held}, and was asked for as {asked}"
            ),
            HostError::Unfilled { key } => write!(
                f,
                "the storage holds no value for `@{key}`, which an entry fetches first, and `@{key}` has no init"
            ),
            HostError::Unstored { key } => {
                write!(f, "the page holds no value for `@{key}`: no init filled it and no run stored it")
            }
            HostError::Storage(error) => write!(f, "the storage refused: {error}"),
        }
    }
}

impl std::error::Error for HostError {}

impl From<StorageError> for HostError {
    fn from(error: StorageError) -> Self {
        HostError::Storage(error)
    }
}

impl From<Absent> for HostError {
    fn from(Absent { key }: Absent) -> Self {
        HostError::Unfilled { key }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageError {
    message: String,
}

impl StorageError {
    pub fn new(message: String) -> Self {
        StorageError { message }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StorageError {}

impl From<SpaceError> for StorageError {
    fn from(error: SpaceError) -> Self {
        StorageError::new(error.to_string())
    }
}

// -- Storage -------------------------------------------------------------

/// The runtime a storage that encodes holders reaches the encoding through.
/// A storage that keeps holders whole ignores it.
pub struct Codec<'a> {
    rt: &'a AcvusRuntime,
}

impl<'a> Codec<'a> {
    pub(crate) fn rt(&self) -> &'a AcvusRuntime {
        self.rt
    }
}

/// Where a page's contexts persist between pages. A page loads every
/// context of its compilation when it opens, and hands a storage each
/// context it changed when it commits, taking the holder back after the
/// storage's `commit`; a storage's errors reach the host at those two
/// points and never inside a run (RFC-0090 rule 6).
pub trait Storage {
    /// Move out the holder stored for `key`.
    fn load(&mut self, key: &str, codec: &Codec<'_>) -> Result<Option<Held>, StorageError>;
    fn store(&mut self, key: &str, held: Held);
    fn commit(&mut self, codec: &Codec<'_>) -> Result<(), StorageError>;
}

/// A storage that keeps holders in memory for the life of the page.
#[derive(Default)]
pub struct MemoryStorage {
    holders: HashMap<String, Held>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        MemoryStorage::default()
    }
}

impl Storage for MemoryStorage {
    fn load(&mut self, key: &str, _: &Codec<'_>) -> Result<Option<Held>, StorageError> {
        Ok(self.holders.remove(key))
    }

    fn store(&mut self, key: &str, held: Held) {
        self.holders.insert(key.to_owned(), held);
    }

    fn commit(&mut self, _: &Codec<'_>) -> Result<(), StorageError> {
        Ok(())
    }
}

// -- The graph environment ----------------------------------------------

struct Environment {
    functions: Vec<Function>,
    types: acvus_mir::ty::TypeRegistry,
    handlers: Handlers<AcvusRuntime>,
    space: crate::layout::Hooks,
    instances: InstanceTable,
    fn_types: FxHashMap<QualifiedRef, Ty>,
}

impl Environment {
    fn combine(interner: &Interner, registries: Vec<Registry<AcvusRuntime>>) -> Result<Self, CombineError> {
        let Externs {
            functions,
            types,
            handlers,
            space,
            instances,
        } = Externs::combine(registries, interner)?;
        let fn_types = functions
            .iter()
            .filter_map(|f| try_freeze_poly(&f.ty).map(|ty| (f.qref, ty)))
            .collect();
        Ok(Environment {
            functions,
            types,
            handlers,
            space,
            instances,
            fn_types,
        })
    }
}

/// Each named context at a type variable the whole graph solves (RFC-0090
/// rule 1), in name order.
fn open_contexts(open: &mut PolyBuilder, named: FxHashSet<QualifiedRef>) -> Vec<Context> {
    let mut contexts: Vec<Context> = named
        .into_iter()
        .map(|qref| Context {
            qref,
            ty: open.fresh_ty_var(),
            init: None,
        })
        .collect();
    contexts.sort_by_key(|context| context.qref.name.bits());
    contexts
}

macro_rules! tooling_graph {
    ($v:vis) => {
        /// The environment an editor checks a source in: the registries'
        /// functions and types and each named context, the graph `compile`
        /// builds before it adds the entries' functions.
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        $v fn environment(
            interner: &Interner,
            named: FxHashSet<QualifiedRef>,
            entries: Vec<QualifiedRef>,
            registries: Vec<Registry<AcvusRuntime>>,
        ) -> Result<CompilationGraph, CombineError> {
            let environment = Environment::combine(interner, registries)?;
            let contexts = open_contexts(&mut PolyBuilder::new(), named);
            Ok(CompilationGraph {
                functions: Freeze::new(environment.functions),
                contexts: Freeze::new(contexts),
                types: Freeze::new(environment.types),
                bindings: Bindings::default(),
                entries,
            })
        }

        /// The type of an entry that declares `!` and reads its `$` inputs
        /// from its body (RFC-0054 rule 5, RFC-0090 rule 6).
        $v fn untyped_entry_ty() -> PolyTy {
            TyTerm::Fn {
                params: vec![],
                ret: Box::new(lift_declaration(&Ty::Never, &mut PolyBuilder::new())),
                captures: vec![],
                effect: Effect::OPAQUE.into(),
                flows: Flows::Every.into(),
            }
        }

        $v fn context_refs(ast: &ParsedAst) -> FxHashSet<QualifiedRef> {
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
    };
}
tooling_vis!(tooling_graph);

// -- Host ----------------------------------------------------------------

/// The entry's `$` inputs as `I` declares them (RFC-0090 rule 2).
struct DeclaredInputs {
    ty: PolyTy,
    params: Vec<ParamTerm<Poly>>,
}

impl DeclaredInputs {
    fn of(interner: &Interner, ty: PolyTy) -> Option<Self> {
        let params = match &ty {
            TyTerm::Unit => Vec::new(),
            TyTerm::Object(fields) => {
                let order = ObjectShape::of(interner, fields.keys().copied());
                order
                    .names()
                    .iter()
                    .map(|name| ParamTerm::new(*name, fields[name].clone()))
                    .collect()
            }
            _ => return None,
        };
        Some(DeclaredInputs { ty, params })
    }
}

enum EntryDeclaration {
    Typed { inputs: DeclaredInputs, declared: PolyTy },
    Untyped,
}

struct EntryDecl {
    name: String,
    ast: ParsedAst,
    declaration: EntryDeclaration,
}

/// The entries, inits and bindings of one compilation graph, so a context's
/// type is solved from every body that stores, reads or initializes it
/// (RFC-0090 rule 1).
pub struct Host {
    interner: Interner,
    registries: Vec<Registry<AcvusRuntime>>,
    bindings: Bindings,
    entries: Vec<EntryDecl>,
    inits: Vec<InitSource>,
    parse_refusals: Vec<Refusal>,
    refusals: Vec<Refusal>,
    opt: Opt,
    parse: Duration,
}

impl Host {
    pub fn new(registries: Vec<Registry<AcvusRuntime>>) -> Self {
        Host {
            interner: Interner::new(),
            registries,
            bindings: Bindings::default(),
            entries: Vec::new(),
            inits: Vec::new(),
            parse_refusals: Vec::new(),
            refusals: Vec::new(),
            opt: Opt::Full,
            parse: Duration::ZERO,
        }
    }

    fn parsed(&mut self, origin: Origin, source: Source<'_>) -> ParsedAst {
        let started = Instant::now();
        let Parsed { ast, errors } = source.parse(&self.interner);
        self.parse += started.elapsed();
        self.parse_refusals.extend(errors.iter().map(|e| Refusal {
            span: span_of(e.span),
            ..Refusal::of(Some(origin.clone()), e.kind.to_string())
        }));
        ast
    }

    pub fn init(mut self, key: &str, source: Source<'_>) -> Self {
        let ast = self.parsed(Origin::Init(key.to_owned()), source);
        self.inits.push(InitSource {
            key: key.to_owned(),
            ast,
        });
        self
    }

    pub fn entry<I, R>(mut self, name: &str, source: Source<'_>) -> Self
    where
        I: Declared,
        R: Declared,
    {
        let origin = Some(Origin::Entry(name.to_owned()));
        let asked = I::declared(&self.interner);
        let Some(inputs) = DeclaredInputs::of(&self.interner, asked.clone()) else {
            let message = format!(
                "the inputs of the entry `{name}` are declared as {}, which is neither `()` \
                 nor a struct of named fields",
                asked.display(&self.interner)
            );
            self.refusals.push(Refusal::of(origin, message));
            return self;
        };
        let declared = R::declared(&self.interner);
        self.declare_entry(name, source, EntryDeclaration::Typed { inputs, declared })
    }

    fn declare_entry(mut self, name: &str, source: Source<'_>, declaration: EntryDeclaration) -> Self {
        let origin = Origin::Entry(name.to_owned());
        if self.entries.iter().any(|entry| entry.name == name) {
            let message = format!("the entry `{name}` is given twice");
            self.refusals.push(Refusal::of(Some(origin.clone()), message));
        }
        let ast = self.parsed(origin, source);
        self.entries.push(EntryDecl {
            name: name.to_owned(),
            ast,
            declaration,
        });
        self
    }

    /// Fix the input `$name` to the value `literal` writes, in the script's
    /// own syntax (RFC-0087 rule 1).
    pub fn bind(mut self, name: &str, literal: &str) -> Result<Self, HostError> {
        let refused = |message: String| {
            HostError::Refused(vec![Refusal::of(Some(Origin::Binding(name.to_owned())), message)])
        };
        let interner = &self.interner;
        let expr = acvus_ast::parse_expr(interner, literal)
            .map_err(|e| refused(format!("${name}: `{literal}` does not parse: {e:?}")))?;
        let written = |span: Span| &literal[span.start..span.end];
        let value = BoundValue::from_expr(interner, &expr).map_err(|e| match e {
            NotABoundValue::NotALiteral { form } => refused(format!(
                "${name}: `{}` is not a value a literal writes",
                written(form)
            )),
            NotABoundValue::RepeatedField { key, second } => refused(format!(
                "${name}: `{}` in `{}` is a field an object holds once",
                interner.resolve(key),
                written(second)
            )),
        })?;
        self.bindings
            .bind(interner.intern(name), value)
            .map_err(|e| refused(format!("${name}: {e}")))?;
        Ok(self)
    }

    pub fn compile<E>(self, executor: E) -> Result<Program, HostError>
    where
        E: Executor + 'static,
    {
        compile(self, Arc::new(executor)).map_err(HostError::Refused)
    }
}

macro_rules! host_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl Host {
            $v fn opt(self, opt: Opt) -> Self {
                Host { opt, ..self }
            }

            /// An entry that declares `!` and whose `$` inputs are the ones
            /// its body reads: the CLI's, which prints whatever comes back
            /// (RFC-0054 rule 5, RFC-0090 rule 6).
            $v fn untyped_entry(self, name: &str, source: Source<'_>) -> Self {
                self.declare_entry(name, source, EntryDeclaration::Untyped)
            }
        }
    };
}
tooling_vis!(host_tooling);

enum EntryShape {
    Typed { inputs: PolyTy, declared: PolyTy },
    Untyped,
}

struct Declaration {
    name: String,
    shape: EntryShape,
}

struct LocalFunction {
    kind: FnKind,
    ty: PolyTy,
    shape: EntryShape,
}

struct CompiledEntry {
    name: String,
    qref: QualifiedRef,
    shape: EntryShape,
    module_params: Vec<Astr>,
    ret: Ty,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    module: MirModule,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    required: Vec<ContextInfo>,
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

/// How long each stage of a compilation took.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub struct CompileTimes {
    pub opt: Opt,
    pub parse: Duration,
    /// `extract` and `infer`, which are not reported apart.
    pub typeck: Duration,
    pub lower: Duration,
    pub optimize: Duration,
    pub prepare: Duration,
}

#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
impl CompileTimes {
    /// Every stage before `prepare`, which belongs to the interpreter.
    pub fn check(&self) -> Duration {
        self.parse + self.typeck + self.lower + self.optimize
    }
}

fn compile(host: Host, executor: Arc<dyn Executor>) -> Result<Program, Vec<Refusal>> {
    let Host {
        interner,
        registries,
        bindings,
        entries,
        inits,
        parse_refusals,
        refusals: structural,
        opt,
        parse,
    } = host;
    let interner = &interner;
    // A recovered tree is checked for what parsed and never lowered, so its
    // parse errors and those refusals are reported together (RFC-0078
    // rule 5); a structural refusal ends the compilation before typeck.
    let mut refusals = parse_refusals;
    let structural_at = refusals.len();
    refusals.extend(structural);

    let environment = match Environment::combine(interner, registries) {
        Ok(environment) => environment,
        Err(error) => {
            let message = format!("the registries do not combine: {error}");
            refusals.push(Refusal::of(None, message));
            return Err(refusals);
        }
    };
    let Environment {
        functions: extern_fns,
        types,
        handlers,
        space,
        instances,
        fn_types,
    } = environment;

    let mut named: FxHashSet<QualifiedRef> = FxHashSet::default();
    let mut declared: FxHashMap<QualifiedRef, Declaration> = FxHashMap::default();
    let mut functions: Vec<Function> = Vec::with_capacity(entries.len() + extern_fns.len());
    for EntryDecl {
        name,
        ast,
        declaration,
    } in entries
    {
        let qref = QualifiedRef::root(interner.intern(&name));
        let origin = Some(Origin::Entry(name.clone()));
        if extern_fns.iter().any(|f| f.qref == qref) {
            let message = format!("the entry `{name}` has the name of an extern function");
            refusals.push(Refusal::of(origin.clone(), message));
        }
        named.extend(context_refs(&ast));
        let local = match declaration {
            EntryDeclaration::Typed { inputs, declared } => {
                refusals.extend(
                    inputs
                        .params
                        .iter()
                        .filter(|param| bindings.get(param.name).is_some())
                        .map(|param| {
                            let message = format!(
                                "the input `${}` of the entry `{name}` is already fixed by a binding",
                                interner.resolve(param.name)
                            );
                            Refusal::of(origin.clone(), message)
                        }),
                );
                let ty = TyTerm::Fn {
                    params: inputs.params,
                    ret: Box::new(declared.clone()),
                    captures: vec![],
                    effect: Effect::OPAQUE.into(),
                    flows: Flows::Every.into(),
                };
                let shape = EntryShape::Typed {
                    inputs: inputs.ty,
                    declared,
                };
                LocalFunction {
                    kind: FnKind::Local(ast, Inputs::Declared),
                    ty,
                    shape,
                }
            }
            EntryDeclaration::Untyped => LocalFunction {
                kind: FnKind::Local(ast, Inputs::FromReads),
                ty: untyped_entry_ty(),
                shape: EntryShape::Untyped,
            },
        };
        let LocalFunction { kind, ty, shape } = local;
        declared.insert(qref, Declaration { name, shape });
        functions.push(Function { qref, kind, ty });
    }
    let entry_refs: Vec<QualifiedRef> = functions.iter().map(|f| f.qref).collect();
    functions.extend(extern_fns);

    let mut open = PolyBuilder::new();
    let contexts = open_contexts(&mut open, named);
    let mut parts = GraphParts {
        open,
        contexts,
        functions,
        entries: entry_refs,
    };
    let declared_inits = match DeclaredInits::declare(interner, inits, &mut parts) {
        Ok(declared) => declared,
        Err(refused) => {
            refusals.extend(refused.into_iter().map(|refusal| {
                Refusal::of(Some(Origin::Init(refusal.key().to_owned())), refusal.to_string())
            }));
            return Err(refusals);
        }
    };
    if refusals.len() > structural_at {
        return Err(refusals);
    }
    let GraphParts {
        contexts,
        functions,
        entries: entry_refs,
        ..
    } = parts;
    let graph = CompilationGraph {
        functions: Freeze::new(functions),
        contexts: Freeze::new(contexts),
        types: Freeze::new(types),
        bindings,
        entries: entry_refs,
    };
    let origin_of = |qref: &QualifiedRef| match declared.get(qref) {
        Some(declaration) => Some(Origin::Entry(declaration.name.clone())),
        None => declared_inits
            .key_of(qref)
            .map(|key| Origin::Init(key.to_owned())),
    };

    let started = Instant::now();
    let ext = extract::extract(interner, &graph);
    let inf = infer::infer(interner, &graph, &ext);
    let typeck = started.elapsed();
    refusals.extend(inf.errors().into_iter().flat_map(|(qref, errs)| {
        let origin = origin_of(&qref);
        errs.iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: e.primary(),
            labels: e.labels.clone(),
        })
    }));

    let started = Instant::now();
    let lowered = lower::lower(interner, &graph, &ext.view(), &inf);
    let lower = started.elapsed();
    refusals.extend(lowered.errors.iter().flat_map(|le| {
        let origin = origin_of(&le.fn_id);
        le.errors.iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: e.primary(),
            labels: e.labels.clone(),
        })
    }));
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let started = Instant::now();
    let optimized = optimize::optimize(
        interner,
        &acvus_mir::laws::LawTable::of(graph.functions.iter()),
        lowered.modules,
        opt,
    );
    let optimize = started.elapsed();
    refusals.extend(optimized.errors.into_iter().flat_map(|(qref, errs)| {
        let origin = origin_of(&qref);
        errs.into_iter().map(move |e| Refusal {
            origin: origin.clone(),
            message: e.display(interner).to_string(),
            span: span_of(e.span),
            primary: None,
            labels: e.labels().to_vec(),
        })
    }));
    if !refusals.is_empty() {
        return Err(refusals);
    }

    let context_names: FxHashMap<QualifiedRef, Astr> =
        graph.contexts.iter().map(|c| (c.qref, c.qref.name)).collect();
    let mut executables: FxHashMap<QualifiedRef, Executable> = handlers
        .into_iter()
        .map(|(q, h)| (q, Executable::Extern(h)))
        .collect();
    let started = Instant::now();
    let prepared: Vec<(QualifiedRef, Executable)> = {
        let ctx = PrepareCtx {
            interner,
            externs: &executables,
            context_names: &context_names,
            instances: &instances,
        };
        optimized
            .modules
            .iter()
            .map(|(q, m)| (*q, Executable::Module(Arc::new(prepare_module(m, &ctx)))))
            .collect()
    };
    let prepare = started.elapsed();
    executables.extend(prepared);
    let mut modules = optimized.modules;
    let mut required = optimized.inputs;
    let compiled_entries: HashMap<String, CompiledEntry> = declared
        .into_iter()
        .map(|(qref, Declaration { name, shape })| {
            let (Some(module), Some(required)) = (modules.remove(&qref), required.remove(&qref)) else {
                panic!("`optimize` keeps a module and its inputs for an entry no stage refused")
            };
            let entry = CompiledEntry {
                name: name.clone(),
                qref,
                shape,
                module_params: module.main.params.iter().map(|(name, _)| *name).collect(),
                ret: module.ret.clone(),
                module,
                required,
            };
            (name, entry)
        })
        .collect();
    let shared = InterpreterContext::new(interner, executables, executor)
        .with_fn_types(fn_types)
        .with_context_names(context_names)
        .with_space(space);
    let solved: BTreeMap<String, Arc<Ty>> = graph
        .contexts
        .iter()
        .map(|c| {
            let Some(ty) = inf.context_types.get(&c.qref) else {
                panic!("inference settles a type for every context of the graph")
            };
            (interner.resolve(c.qref.name).to_owned(), Arc::new(ty.clone()))
        })
        .collect();
    let fetched_first: BTreeSet<String> = compiled_entries
        .values()
        .flat_map(|entry| lookup_module(&shared, &entry.qref).fetched_first.iter())
        .map(|key| key.to_string())
        .collect();
    let rt = shared.runtime_over_an_empty_page();
    Ok(Program {
        inits: declared_inits.solved(&solved),
        solved,
        fetched_first,
        rt,
        shared,
        entries: compiled_entries,
        times: CompileTimes {
            opt,
            parse,
            typeck,
            lower,
            optimize,
            prepare,
        },
    })
}

enum InputsCrossing {
    Nothing,
    Fields { width: usize },
}

impl InputsCrossing {
    fn of<I>(interner: &Interner, asked: &PolyTy, params: &[Astr]) -> Option<Self>
    where
        I: Cross<AcvusRuntime>,
    {
        match asked {
            TyTerm::Unit => params.is_empty().then_some(InputsCrossing::Nothing),
            TyTerm::Object(fields) => {
                let form = (
                    <I::ReturnForm as Form>::KIND,
                    <I::ReturnForm as Form>::WIDTH,
                );
                let order = ObjectShape::of(interner, fields.keys().copied());
                match form {
                    (FormKind::Components, width)
                        if width == params.len() && order.names() == params =>
                    {
                        Some(InputsCrossing::Fields { width })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// -- Program -------------------------------------------------------------

pub struct Program {
    shared: InterpreterContext,
    rt: AcvusRuntime,
    entries: HashMap<String, CompiledEntry>,
    inits: Inits,
    solved: BTreeMap<String, Arc<Ty>>,
    fetched_first: BTreeSet<String>,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    times: CompileTimes,
}

impl Program {
    /// Every context of the compilation, in key order.
    pub fn contexts(&self) -> impl Iterator<Item = &str> {
        self.solved.keys().map(String::as_str)
    }

    /// The type the compilation solved for `key`, as the language writes it.
    pub fn context_type(&self, key: &str) -> Option<String> {
        let ty = self.solved.get(key)?;
        Some(ty.display(&self.shared.interner).to_string())
    }

    /// Run `f` with this program's pages and entries, which are good for
    /// that call alone.
    pub async fn scope<F, T>(&self, f: F) -> T
    where
        F: for<'p> AsyncFnOnce(Scope<'p>) -> T,
    {
        f(Scope {
            program: self,
            brand: PhantomData,
        })
        .await
    }

    fn solved_type(&self, key: &str) -> Result<&Arc<Ty>, HostError> {
        self.solved.get(key).ok_or_else(|| HostError::NotInGraph {
            what: Named::Context(key.to_owned()),
        })
    }

    fn mismatched(&self, what: Part, held: &Ty, asked: &PolyTy) -> HostError {
        HostError::Mismatched {
            what,
            held: held.display(self.interner()).to_string(),
            asked: asked.display(self.interner()).to_string(),
        }
    }

    /// Refuse a holder a storage gives for `key` at another type than the
    /// solved one, before its word is read (RFC-0090 rule 4).
    fn accept_held(&self, key: &str, held: &Held) -> Result<(), HostError> {
        if held.made_by() != self.shared.compilation {
            let message = format!("the storage gave `@{key}` a holder another compilation made");
            return Err(HostError::Storage(StorageError::new(message)));
        }
        let solved = self.solved_type(key)?;
        if held.ty().same_erased(solved) {
            return Ok(());
        }
        Err(HostError::Mismatched {
            what: Part::Context(key.to_owned()),
            held: held.ty().display(self.interner()).to_string(),
            asked: solved.display(self.interner()).to_string(),
        })
    }

    fn compiled(&self, name: &str) -> Result<&CompiledEntry, HostError> {
        self.entries.get(name).ok_or_else(|| HostError::NotInGraph {
            what: Named::Entry(name.to_owned()),
        })
    }
}

/// A listing of one entry for the runtime's tooling.
#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub struct Listing<'p> {
    /// The `$` inputs the entry still requires, in name order; a binding's
    /// input is not among them (RFC-0071 rule 5).
    pub inputs: Vec<InputListing>,
    pub mir: String,
    pub prepared: &'p crate::code::Prepared,
}

#[cfg_attr(not(feature = "tooling"), allow(dead_code))]
pub struct InputListing {
    pub name: String,
    pub ty: String,
}

macro_rules! program_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl Program {
            $v fn interner(&self) -> &Interner {
                &self.shared.interner
            }

            $v fn times(&self) -> &CompileTimes {
                &self.times
            }

            $v fn listing(&self, name: &str) -> Result<Listing<'_>, HostError> {
                let compiled = self.compiled(name)?;
                let interner = self.interner();
                let mut required: Vec<&ContextInfo> = compiled.required.iter().collect();
                required.sort_by_key(|input| input.name.name.bits());
                let inputs = required
                    .into_iter()
                    .map(|input| InputListing {
                        name: interner.resolve(input.name.name).to_owned(),
                        ty: input.ty.display(interner).to_string(),
                    })
                    .collect();
                Ok(Listing {
                    inputs,
                    mir: acvus_mir::printer::dump(interner, &compiled.module),
                    prepared: lookup_module(&self.shared, &compiled.qref),
                })
            }
        }
    };
}
tooling_vis!(program_tooling);

// -- Scope ---------------------------------------------------------------

/// One `Program::scope` call's view of its program. The lifetime brands
/// every page and entry the scope makes.
pub struct Scope<'p> {
    program: &'p Program,
    brand: Brand<'p>,
}

impl Clone for Scope<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Scope<'_> {}

impl<'p> Scope<'p> {
    /// A page over `storage`: every context the storage holds is loaded and
    /// checked against its solved type, and the init of each context an
    /// entry fetches before assigning that the storage lacks runs (RFC-0090
    /// rules 1, 3).
    pub async fn open<S>(self, mut storage: S) -> Result<Page<'p, S>, HostError>
    where
        S: Storage,
    {
        let program = self.program;
        let codec = Codec { rt: &program.rt };
        let journal = RuntimeContext::empty();
        for key in program.solved.keys() {
            let Some(held) = storage.load(key, &codec)? else {
                continue;
            };
            program.accept_held(key, &held)?;
            journal.set_unchanged(key, held);
        }
        let unfilled = program
            .fetched_first
            .iter()
            .find(|key| !journal.holds(key) && !program.inits.has(key));
        if let Some(key) = unfilled {
            return Err(HostError::Unfilled { key: key.clone() });
        }
        let journal = Arc::new(journal);
        let fetched_first = |key: &str| program.fetched_first.contains(key);
        let filled = program
            .inits
            .fill_lacking(&program.shared, &journal, fetched_first)
            .await?;
        Ok(Page {
            program,
            storage,
            journal,
            filled,
            brand: PhantomData,
        })
    }

    pub fn entry<I, R>(self, name: &str) -> Result<Entry<'p, I, R>, HostError>
    where
        I: Declared + Cross<AcvusRuntime>,
        R: Declared,
    {
        let program = self.program;
        let compiled = program.compiled(name)?;
        let interner = program.interner();
        let asked_inputs = I::declared(interner);
        let asked = R::declared(interner);
        let EntryShape::Typed { inputs, declared } = &compiled.shape else {
            return Err(program.mismatched(Part::Result(name.to_owned()), &Ty::Never, &asked));
        };
        let refused_inputs = || HostError::Mismatched {
            what: Part::Inputs(name.to_owned()),
            held: inputs.display(interner).to_string(),
            asked: asked_inputs.display(interner).to_string(),
        };
        if !asked_inputs.same_erased(inputs) {
            return Err(refused_inputs());
        }
        let Some(crossed) = InputsCrossing::of::<I>(interner, &asked_inputs, &compiled.module_params)
        else {
            return Err(refused_inputs());
        };
        if !asked.same_erased(declared) {
            return Err(HostError::Mismatched {
                what: Part::Result(name.to_owned()),
                held: declared.display(interner).to_string(),
                asked: asked.display(interner).to_string(),
            });
        }
        Ok(Entry {
            program,
            compiled,
            crossed,
            declared: PhantomData,
            brand: PhantomData,
        })
    }
}

macro_rules! scope_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<'p> Scope<'p> {
            /// An entry the tooling runs whatever it declares, whose result
            /// it reads by the settled type; one that still requires a `$`
            /// input is refused.
            $v fn untyped_entry(self, name: &str) -> Result<UntypedEntry<'p>, HostError> {
                let program = self.program;
                let compiled = program.compiled(name)?;
                if !compiled.required.is_empty() {
                    let interner = program.interner();
                    let held: Vec<String> = compiled
                        .required
                        .iter()
                        .map(|input| format!("${}", interner.resolve(input.name.name)))
                        .collect();
                    return Err(HostError::Mismatched {
                        what: Part::Inputs(name.to_owned()),
                        held: held.join(", "),
                        asked: "()".to_owned(),
                    });
                }
                Ok(UntypedEntry {
                    program,
                    compiled,
                    brand: PhantomData,
                })
            }
        }
    };
}
tooling_vis!(scope_tooling);

// -- Page ----------------------------------------------------------------

/// A program's view of one storage (RFC-0090 rule 3), made by
/// `Scope::open`.
pub struct Page<'p, S> {
    program: &'p Program,
    storage: S,
    journal: Arc<RuntimeContext>,
    #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
    filled: Vec<String>,
    brand: Brand<'p>,
}

impl<'p, S> Page<'p, S>
where
    S: Storage,
{
    /// Lend `key`'s value to `f`, whose parameter crosses as a handler's
    /// shared one does (RFC-0090 rule 3).
    pub fn with<Q, F, O>(&self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let program = self.program;
        program.solved_type(key)?;
        let holders = self.journal.read();
        let Some(held) = holders.get(key) else {
            return Err(HostError::Unstored { key: key.to_owned() });
        };
        held.lend(&program.rt, program.interner(), f)
            .map_err(|asked| program.mismatched(Part::Context(key.to_owned()), held.ty(), &asked))
    }

    /// Lend `key`'s value to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive.
    pub fn with_mut<Q, F, O>(&mut self, key: &str, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let program = self.program;
        program.solved_type(key)?;
        let lent = {
            let mut holders = self.journal.write();
            let Some(held) = holders.get_mut(key) else {
                return Err(HostError::Unstored { key: key.to_owned() });
            };
            held.lend_mut(&program.rt, program.interner(), f).map_err(|asked| {
                program.mismatched(Part::Context(key.to_owned()), held.ty(), &asked)
            })?
        };
        self.journal.mark_changed(key);
        Ok(lent)
    }

    pub fn insert<T>(&mut self, key: &str, value: T) -> Result<(), HostError>
    where
        T: Declared + OneValue<AcvusRuntime>,
    {
        let program = self.program;
        let solved = program.solved_type(key)?;
        let asked = T::declared(program.interner());
        if !lift_declaration(solved, &mut PolyBuilder::new()).same_erased(&asked) {
            return Err(program.mismatched(Part::Context(key.to_owned()), solved, &asked));
        }
        // SAFETY: `value` crosses at `T`, the type the page holds `key` at.
        let value = Owned::erased(unsafe { Crossing::new(&program.rt) }, value);
        let made_by = program.shared.compilation;
        self.journal.set_changed(key, Held::new(value, Arc::clone(solved), made_by));
        Ok(())
    }

    /// Hand the storage every context changed since the last commit, commit
    /// the storage, and take each holder back.
    pub fn commit(&mut self) -> Result<(), HostError> {
        let program = self.program;
        let codec = Codec { rt: &program.rt };
        let mut handed = Vec::new();
        for key in self.journal.take_changed() {
            let Some(held) = self.journal.take(&key) else {
                continue;
            };
            self.storage.store(&key, held);
            handed.push(key);
        }
        let committed = self.storage.commit(&codec);
        for key in &handed {
            if committed.is_err() {
                self.journal.mark_changed(key);
            }
            let Some(held) = self.storage.load(key, &codec)? else {
                let message = format!("the storage gave back nothing for `@{key}`, which the page handed it");
                return Err(HostError::Storage(StorageError::new(message)));
            };
            program.accept_held(key, &held)?;
            self.journal.set_unchanged(key, held);
        }
        committed.map_err(HostError::Storage)
    }

    pub fn storage(&self) -> &S {
        &self.storage
    }
}

macro_rules! page_tooling {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<S> Page<'_, S>
        where
            S: Storage,
        {
            /// The keys whose init ran when the page opened, in key order.
            $v fn filled(&self) -> &[String] {
                &self.filled
            }

            /// Run every init whose key the page lacks; the keys filled, in
            /// key order.
            $v async fn fill(&mut self) -> Result<Vec<String>, HostError> {
                let program = self.program;
                let every = |_: &str| true;
                let filled = program.inits.fill_lacking(&program.shared, &self.journal, every).await?;
                Ok(filled)
            }
        }
    };
}
tooling_vis!(page_tooling);

// -- Entry and Output ----------------------------------------------------

pub struct Entry<'p, I, R> {
    program: &'p Program,
    compiled: &'p CompiledEntry,
    crossed: InputsCrossing,
    declared: PhantomData<fn(I) -> R>,
    brand: Brand<'p>,
}

impl<'p, I, R> Entry<'p, I, R>
where
    I: Cross<AcvusRuntime>,
    I::ReturnForm: Returned<Verdict = ()>,
{
    pub async fn run<S>(&self, page: &mut Page<'p, S>, inputs: I) -> Result<Output<'p, R>, HostError>
    where
        S: Storage,
    {
        let program = self.program;
        let mut interpreter =
            Interpreter::on_page(program.shared.clone(), self.compiled.qref, Arc::clone(&page.journal));
        let accepted = interpreter.accept_page()?;
        let args = match self.crossed {
            InputsCrossing::Nothing => Vec::new(),
            InputsCrossing::Fields { width } => {
                let mut run: Vec<Value> = std::iter::repeat_with(Value::unit).take(width).collect();
                // SAFETY: `Scope::entry` compared `I`'s declaration with the
                // one the entry was compiled against, and `I`'s crossing
                // writes one value per parameter, at that parameter's type and
                // in the order the module takes them.
                let crossing = unsafe { Crossing::new(&program.rt) };
                <I as Gives<Val<I, Uniform>, AcvusRuntime>>::give(inputs, crossing, &mut run);
                run
            }
        };
        let value = accepted.run(args).await;
        Ok(Output {
            // SAFETY: the run moved its result out to this caller, and no
            // other holder owns it.
            value: unsafe { Owned::from_value(Holding::new(), value) },
            program,
            compiled: self.compiled,
            result: PhantomData,
            brand: PhantomData,
        })
    }
}

/// An entry's result, owned until dropped, at the type the checker settled
/// for `R` (RFC-0090 rule 3).
pub struct Output<'p, R> {
    value: Owned<AcvusRuntime>,
    program: &'p Program,
    compiled: &'p CompiledEntry,
    result: PhantomData<fn() -> R>,
    brand: Brand<'p>,
}

impl<R> Output<'_, R> {
    /// Lend the result to `f`, whose parameter crosses as a handler's shared
    /// one does.
    pub fn with<Q, F, O>(&self, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let program = self.program;
        // SAFETY: the run crossed the result at `ret` and `self` owns it for
        // the borrow; a shared parameter writes nothing.
        unsafe { acvus_extern::lend(&program.rt, program.interner(), &self.value, &self.compiled.ret, f) }
            .map_err(|asked| self.mismatched(&asked))
    }

    /// Lend the result to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive.
    pub fn with_mut<Q, F, O>(&mut self, f: F) -> Result<O, HostError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let program = self.program;
        // SAFETY: `&mut self` names the result exclusively, and a lent
        // parameter writes inside the storage the word names or the word in
        // place, never another holder's.
        let word = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: the run crossed the result at `ret`, and `word` is the only
        // live name of it for the call.
        unsafe { acvus_extern::lend(&program.rt, program.interner(), word, &self.compiled.ret, f) }
            .map_err(|asked| self.mismatched(&asked))
    }

    fn mismatched(&self, asked: &PolyTy) -> HostError {
        let what = Part::Result(self.compiled.name.clone());
        self.program.mismatched(what, &self.compiled.ret, asked)
    }
}

/// An entry the runtime's tooling runs and reads by the settled type.
pub struct UntypedEntry<'p> {
    program: &'p Program,
    compiled: &'p CompiledEntry,
    brand: Brand<'p>,
}

pub struct UntypedOutput<'p> {
    value: Owned<AcvusRuntime>,
    compiled: &'p CompiledEntry,
    brand: Brand<'p>,
}

macro_rules! untyped_run {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl<'p> UntypedEntry<'p> {
            $v async fn run<S>(&self, page: &mut Page<'p, S>) -> Result<UntypedOutput<'p>, HostError>
            where
                S: Storage,
            {
                let program = self.program;
                let mut interpreter = Interpreter::on_page(
                    program.shared.clone(),
                    self.compiled.qref,
                    Arc::clone(&page.journal),
                );
                let value = interpreter.accept_page()?.run(Vec::new()).await;
                Ok(UntypedOutput {
                    // SAFETY: the run moved its result out to this caller,
                    // and no other holder owns it.
                    value: unsafe { Owned::from_value(Holding::new(), value) },
                    compiled: self.compiled,
                    brand: PhantomData,
                })
            }
        }

        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl UntypedOutput<'_> {
            $v fn with_value<O, F>(&self, f: F) -> O
            where
                F: FnOnce(&Value, &Ty) -> O,
            {
                f(&self.value, &self.compiled.ret)
            }
        }
    };
}
tooling_vis!(untyped_run);
