//! A host lends a result or a context to a closure, and nothing the closure
//! is lent outlives it (RFC-0090 rule 5):
//!
//! ```compile_fail
//! use acvus_interpreter::{Output, OutputError};
//! fn kept(output: &Output<String>) -> Result<&str, OutputError> {
//!     output.with(|s: &str| s)
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::InMemoryContext;
//! fn kept(page: &mut InMemoryContext) -> Vec<&str> {
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
//! use acvus_interpreter::{InMemoryContext, Output, OutputError, PageError};
//! fn kept(output: &Output<String>) -> Result<String, OutputError> {
//!     output.with(|s: &str| s.to_owned())
//! }
//! fn pushed(page: &mut InMemoryContext) -> Result<(), PageError> {
//!     page.with_mut("name", |s: &mut String| s.push('!'))
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::{Output, OutputError};
//! fn pushed(output: &Output<String>) -> Result<(), OutputError> {
//!     output.with(|s: &mut String| s.push('!'))
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

not a space's raw load or commit,

```compile_fail,E0624
fn load(space: &acvus_interpreter::Space, rt: &acvus_interpreter::AcvusRuntime, ty: &acvus_mir::ty::Ty) {
    let _ = space.load(rt, "x", ty);
}
```

```compile_fail,E0624
fn commit(page: &acvus_interpreter::SpacePage, rt: &acvus_interpreter::AcvusRuntime) {
    let _ = page.commit(rt);
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

not a run's raw writes,

```compile_fail,E0599
use acvus_interpreter::RuntimeContext;
fn drain(page: &acvus_interpreter::InMemoryContext) {
    let _ = page.take_writes();
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

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use acvus_ast::Span;
use acvus_extern::{Borrows, Declared, Externs, Holding, Lendable, Owned, Registry, Shared, SpaceError};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Bindings, CompilationGraph, Context, FnKind, Function, Parsed, ParsedAst, QualifiedRef,
    RecoveredAst, extract, infer, lower, optimize,
};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::executor::Executor;
use crate::interpreter::{Executable, Interpreter, InterpreterContext};
use crate::journal::RuntimeContext;
use crate::prepare::{PrepareCtx, prepare_module};
use crate::runtime::AcvusRuntime;

pub enum Source<'a> {
    Script(&'a str),
    Template(&'a str),
}

#[derive(Debug)]
pub struct Refusal {
    /// The entry whose source `span` points into; `None` for a refusal of
    /// the compilation as a whole.
    pub entry: Option<String>,
    pub message: String,
    pub span: Option<Span>,
}

struct EntryDecl {
    name: String,
    ast: ParsedAst,
    declared: PolyTy,
}

/// The entries of one compilation graph. An initializer that stores a
/// context and the scripts that read it are entries of the same graph, so
/// the context's type is solved from all of them (RFC-0090 rule 1).
pub struct Host {
    interner: Interner,
    registries: Vec<Registry<AcvusRuntime>>,
    bindings: Bindings,
    entries: Vec<EntryDecl>,
    refusals: Vec<Refusal>,
}

impl Host {
    pub fn new(interner: &Interner, registries: Vec<Registry<AcvusRuntime>>) -> Self {
        Host {
            interner: interner.clone(),
            registries,
            bindings: Bindings::default(),
            entries: Vec::new(),
            refusals: Vec::new(),
        }
    }

    pub fn bindings(self, bindings: Bindings) -> Self {
        Host { bindings, ..self }
    }

    pub fn entry<R>(mut self, name: &str, source: Source<'_>) -> Self
    where
        R: Declared,
    {
        if self.entries.iter().any(|entry| entry.name == name) {
            self.refusals.push(Refusal {
                entry: Some(name.to_owned()),
                message: format!("the entry `{name}` is given twice"),
                span: None,
            });
        }
        let Parsed { ast, errors } = match source {
            Source::Script(text) => Parsed::script(acvus_ast::parse_script(&self.interner, text)),
            Source::Template(text) => Parsed::template(acvus_ast::parse(&self.interner, text)),
        };
        self.refusals.extend(errors.iter().map(|e| Refusal {
            entry: Some(name.to_owned()),
            message: e.kind.to_string(),
            span: span_of(e.span),
        }));
        self.entries.push(EntryDecl {
            name: name.to_owned(),
            ast,
            declared: R::declared(&self.interner),
        });
        self
    }

    pub fn compile(self, executor: Arc<dyn Executor>) -> Result<Program, Vec<Refusal>> {
        let Host {
            interner,
            registries,
            bindings,
            entries,
            mut refusals,
        } = self;
        let interner = &interner;

        let Externs {
            functions: extern_fns,
            types,
            handlers,
            space,
            instances,
        } = match Externs::combine(registries, interner) {
            Ok(externs) => externs,
            Err(error) => {
                refusals.push(Refusal {
                    entry: None,
                    message: format!("the registries do not combine: {error}"),
                    span: None,
                });
                return Err(refusals);
            }
        };
        let fn_types: FxHashMap<QualifiedRef, Ty> = extern_fns
            .iter()
            .filter_map(|f| try_freeze_poly(&f.ty).map(|ty| (f.qref, ty)))
            .collect();

        let mut named: FxHashSet<QualifiedRef> = FxHashSet::default();
        let mut declared: FxHashMap<QualifiedRef, Declaration> = FxHashMap::default();
        let mut functions: Vec<Function> = Vec::with_capacity(entries.len() + extern_fns.len());
        for EntryDecl {
            name,
            ast,
            declared: ret,
        } in entries
        {
            let qref = QualifiedRef::root(interner.intern(&name));
            if extern_fns.iter().any(|f| f.qref == qref) {
                refusals.push(Refusal {
                    entry: Some(name.clone()),
                    message: format!("the entry `{name}` has the name of an extern function"),
                    span: None,
                });
            }
            named.extend(context_refs(&ast));
            declared.insert(
                qref,
                Declaration {
                    name,
                    declared: ret.clone(),
                },
            );
            functions.push(Function {
                qref,
                kind: FnKind::Local(ast),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: Effect::OPAQUE.into(),
                    flows: Flows::Every.into(),
                },
            });
        }
        let entry_refs: Vec<QualifiedRef> = functions.iter().map(|f| f.qref).collect();
        functions.extend(extern_fns);
        if !refusals.is_empty() {
            return Err(refusals);
        }

        let mut open = PolyBuilder::new();
        let mut contexts: Vec<Context> = named
            .into_iter()
            .map(|qref| Context {
                qref,
                ty: open.fresh_ty_var(),
            })
            .collect();
        contexts.sort_by_key(|context| context.qref.name.bits());
        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            types: Freeze::new(types),
            bindings,
            entries: entry_refs,
        };
        let entry_of = |qref: &QualifiedRef| declared.get(qref).map(|d| d.name.clone());

        let ext = extract::extract(&interner, &graph);
        let inf = infer::infer(&interner, &graph, &ext);
        refusals.extend(inf.errors().into_iter().flat_map(|(qref, errs)| {
            errs.iter().map(move |e| Refusal {
                entry: entry_of(&qref),
                message: e.display(&interner).to_string(),
                span: span_of(e.span),
            })
        }));
        let lowered = lower::lower(&interner, &graph, &ext.view(), &inf);
        refusals.extend(lowered.errors.iter().flat_map(|le| {
            le.errors.iter().map(|e| Refusal {
                entry: entry_of(&le.fn_id),
                message: e.display(&interner).to_string(),
                span: span_of(e.span),
            })
        }));
        if !refusals.is_empty() {
            return Err(refusals);
        }

        let optimized = optimize::optimize(
            &interner,
            &acvus_mir::laws::LawTable::of(graph.functions.iter()),
            lowered.modules,
            Opt::Full,
        );
        refusals.extend(optimized.errors.into_iter().flat_map(|(qref, errs)| {
            let entry = entry_of(&qref);
            errs.into_iter().map(move |e| Refusal {
                entry: entry.clone(),
                message: e.display(&interner).to_string(),
                span: span_of(e.span),
            })
        }));
        if !refusals.is_empty() {
            return Err(refusals);
        }

        let compiled_entries: HashMap<String, CompiledEntry> = declared
            .into_iter()
            .map(|(qref, Declaration { name, declared })| {
                let Some(module) = optimized.modules.get(&qref) else {
                    panic!("an entry lowers to a module whenever no stage refused it")
                };
                let entry = CompiledEntry {
                    qref,
                    declared,
                    ret: module.ret.clone(),
                };
                (name, entry)
            })
            .collect();
        let context_names: FxHashMap<QualifiedRef, Astr> =
            graph.contexts.iter().map(|c| (c.qref, c.qref.name)).collect();
        let mut executables: FxHashMap<QualifiedRef, Executable> = handlers
            .into_iter()
            .map(|(q, h)| (q, Executable::Extern(h)))
            .collect();
        let prepared: Vec<(QualifiedRef, Executable)> = {
            let ctx = PrepareCtx {
                interner: &interner,
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
        executables.extend(prepared);
        let shared = InterpreterContext::new(&interner, executables, executor)
            .with_fn_types(fn_types)
            .with_context_names(context_names)
            .with_space(space);
        let solved: HashMap<String, Ty> = graph
            .contexts
            .iter()
            .map(|c| {
                let Some(ty) = inf.context_types.get(&c.qref) else {
                    panic!("inference settles a type for every context of the graph")
                };
                (interner.resolve(c.qref.name).to_owned(), ty.clone())
            })
            .collect();
        Ok(Program {
            contexts: Contexts {
                rt: shared.runtime_over_an_empty_page(),
                solved,
            },
            shared,
            entries: compiled_entries,
        })
    }
}

fn context_refs(ast: &ParsedAst) -> FxHashSet<QualifiedRef> {
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

struct Declaration {
    name: String,
    declared: PolyTy,
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

struct CompiledEntry {
    qref: QualifiedRef,
    declared: PolyTy,
    ret: Ty,
}

pub struct Program {
    shared: InterpreterContext,
    entries: HashMap<String, CompiledEntry>,
    contexts: Contexts,
}

impl Program {
    pub fn contexts(&self) -> &Contexts {
        &self.contexts
    }

    pub fn entry<R>(&self, name: &str) -> Result<Entry<'_, R>, EntryError>
    where
        R: Declared,
    {
        let Some(compiled) = self.entries.get(name) else {
            return Err(EntryError::NotInGraph {
                name: name.to_owned(),
            });
        };
        let interner = &self.shared.interner;
        let asked = R::declared(interner);
        if !asked.same_erased(&compiled.declared) {
            return Err(EntryError::Mismatched {
                name: name.to_owned(),
                declared: compiled.declared.display(interner).to_string(),
                asked: asked.display(interner).to_string(),
            });
        }
        Ok(Entry {
            program: self,
            compiled,
            result: PhantomData,
        })
    }
}

#[derive(Debug)]
pub enum EntryError {
    NotInGraph {
        name: String,
    },
    Mismatched {
        name: String,
        declared: String,
        asked: String,
    },
}

impl fmt::Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntryError::NotInGraph { name } => write!(f, "the compilation has no entry `{name}`"),
            EntryError::Mismatched {
                name,
                declared,
                asked,
            } => write!(
                f,
                "the entry `{name}` was declared to return {declared}, and was asked for as {asked}"
            ),
        }
    }
}

impl std::error::Error for EntryError {}

pub struct Entry<'p, R> {
    program: &'p Program,
    compiled: &'p CompiledEntry,
    result: PhantomData<fn() -> R>,
}

impl<R> Entry<'_, R> {
    pub async fn run<P>(&self, page: &Arc<P>) -> Result<Output<R>, PageError>
    where
        P: Page + 'static,
    {
        let program = self.program;
        let interner = &program.shared.interner;
        for (key, solved) in &program.contexts.solved {
            let Some(held) = page.held_type(key) else {
                return Err(PageError::NotInGraph { key: key.clone() });
            };
            if !held.same_erased(solved) {
                return Err(PageError::Mismatched {
                    key: key.clone(),
                    held: held.display(interner).to_string(),
                    asked: solved.display(interner).to_string(),
                });
            }
        }
        let page: Arc<dyn RuntimeContext> = Arc::<P>::clone(page);
        let mut interpreter =
            Interpreter::on_page(program.shared.clone(), self.compiled.qref, page);
        let value = interpreter.execute().await?;
        Ok(Output {
            // SAFETY: the run moved its result out to this caller, and no
            // other holder owns it.
            value: unsafe { Owned::from_value(Holding::new(), value) },
            ty: self.compiled.ret.clone(),
            rt: program.shared.runtime_over_an_empty_page(),
            result: PhantomData,
        })
    }
}

/// An entry's result, owned until dropped, at the type the checker settled
/// for `R` (RFC-0090 rule 3).
pub struct Output<R> {
    value: Owned<AcvusRuntime>,
    ty: Ty,
    rt: AcvusRuntime,
    result: PhantomData<fn() -> R>,
}

impl<R> Output<R> {
    /// Lend the result to `f`, whose parameter crosses as a handler's shared
    /// one does.
    pub fn with<Q, F, O>(&self, f: F) -> Result<O, OutputError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
        F::Marker: Lendable<AcvusRuntime, Loan = Shared>,
    {
        let interner = &self.rt.shared.interner;
        // SAFETY: the run crossed the result at `ty` and `self` owns it for
        // the borrow; a shared parameter writes nothing.
        unsafe { acvus_extern::lend(&self.rt, interner, &self.value, &self.ty, f) }
            .map_err(|asked| OutputError::mismatched(interner, &self.ty, &asked))
    }

    /// Lend the result to `f` exclusively, whose parameter crosses as a
    /// handler's does, shared or exclusive.
    pub fn with_mut<Q, F, O>(&mut self, f: F) -> Result<O, OutputError>
    where
        F: Borrows<AcvusRuntime, Q, O>,
    {
        let interner = &self.rt.shared.interner;
        // SAFETY: `&mut self` names the result exclusively, and a lent
        // parameter writes inside the storage the word names or the word in
        // place, never another holder's.
        let word = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: the run crossed the result at `ty`, and `word` is the only
        // live name of it for the call.
        unsafe { acvus_extern::lend(&self.rt, interner, word, &self.ty, f) }
            .map_err(|asked| OutputError::mismatched(interner, &self.ty, &asked))
    }
}

#[derive(Debug)]
pub enum OutputError {
    Mismatched { held: String, asked: String },
}

impl OutputError {
    fn mismatched(interner: &Interner, held: &Ty, asked: &PolyTy) -> Self {
        OutputError::Mismatched {
            held: held.display(interner).to_string(),
            asked: asked.display(interner).to_string(),
        }
    }
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputError::Mismatched { held, asked } => {
                write!(f, "the result is {held}, and was lent to a parameter of {asked}")
            }
        }
    }
}

impl std::error::Error for OutputError {}

#[derive(Clone)]
pub struct Contexts {
    rt: AcvusRuntime,
    solved: HashMap<String, Ty>,
}

impl Contexts {
    pub fn solved(&self) -> &HashMap<String, Ty> {
        &self.solved
    }

    pub(crate) fn rt(&self) -> &AcvusRuntime {
        &self.rt
    }
}

/// A storage typed by the compilation it was opened for (RFC-0090 rule 4).
/// `held_type` is what `Entry::run` refuses a mismatched page by before the
/// run starts; the run itself reads each holder at the type the holder
/// carries, so a wrong answer here is a refusal at the fetch and not a read
/// at another type.
pub trait Page: RuntimeContext {
    fn held_type(&self, key: &str) -> Option<&Ty>;
}

#[derive(Debug)]
pub enum PageError {
    NotInGraph { key: String },
    Mismatched { key: String, held: String, asked: String },
    Absent { key: String },
    Space(SpaceError),
}

impl fmt::Display for PageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageError::NotInGraph { key } => write!(f, "the compilation has no `@{key}`"),
            PageError::Mismatched { key, held, asked } => {
                write!(f, "`@{key}` is held as {held}, and was asked for as {asked}")
            }
            PageError::Absent { key } => write!(f, "the page holds no value for `@{key}`"),
            PageError::Space(error) => write!(f, "the space refused: {error}"),
        }
    }
}

impl std::error::Error for PageError {}

pub(crate) fn held_as<'t, T>(
    interner: &Interner,
    key: &str,
    held: Option<&'t Ty>,
) -> Result<&'t Ty, PageError>
where
    T: Declared,
{
    let Some(held) = held else {
        return Err(PageError::NotInGraph {
            key: key.to_owned(),
        });
    };
    let asked = T::declared(interner);
    if !lift_declaration(held, &mut PolyBuilder::new()).same_erased(&asked) {
        return Err(PageError::Mismatched {
            key: key.to_owned(),
            held: held.display(interner).to_string(),
            asked: asked.display(interner).to_string(),
        });
    }
    Ok(held)
}
