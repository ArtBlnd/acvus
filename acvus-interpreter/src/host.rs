//! ```compile_fail
//! use acvus_interpreter::Output;
//! fn kept(output: Output<String>) -> &'static String {
//!     output.get()
//! }
//! ```
//!
//! ```compile_fail
//! use acvus_interpreter::InMemoryContext;
//! fn kept(page: &mut InMemoryContext) -> &'static String {
//!     page.read::<String>("name").unwrap()
//! }
//! ```
//!
//! ```
//! use acvus_interpreter::{InMemoryContext, Output, PageError};
//! fn read(output: &Output<String>) -> &String {
//!     output.get()
//! }
//! fn read_page(page: &mut InMemoryContext) -> Result<&String, PageError> {
//!     page.read::<String>("name")
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use acvus_ast::Span;
use acvus_extern::{ArgAt, Declared, Externs, Holding, Owned, Project, Registry, SpaceError};
use acvus_mir::graph::optimize::Opt;
use acvus_mir::graph::{
    Bindings, CompilationGraph, Context, FnKind, Function, Parsed, QualifiedRef, extract, infer,
    lower, optimize,
};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration, try_freeze_poly};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

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
    pub message: String,
    pub span: Option<Span>,
}

pub struct Host {
    interner: Interner,
    registries: Vec<Registry<AcvusRuntime>>,
    bindings: Bindings,
    contexts: Vec<(String, PolyTy)>,
}

impl Host {
    pub fn new(interner: &Interner, registries: Vec<Registry<AcvusRuntime>>) -> Self {
        Host {
            interner: interner.clone(),
            registries,
            bindings: Bindings::default(),
            contexts: Vec::new(),
        }
    }

    pub fn bindings(self, bindings: Bindings) -> Self {
        Host { bindings, ..self }
    }

    pub fn context<T>(mut self, key: &str) -> Self
    where
        T: Declared,
    {
        let declared = T::declared(&self.interner);
        self.contexts.push((key.to_owned(), declared));
        self
    }

    pub fn compile<R>(
        self,
        source: Source<'_>,
        executor: Arc<dyn Executor>,
    ) -> Result<Program<R>, Vec<Refusal>>
    where
        R: Declared,
    {
        let Host {
            interner,
            registries,
            bindings,
            contexts,
        } = self;
        let mut refusals = Vec::new();
        let mut keys: Vec<&str> = Vec::new();
        for (key, _) in &contexts {
            if keys.contains(&key.as_str()) {
                refusals.push(Refusal {
                    message: format!("the context `{key}` is declared twice"),
                    span: None,
                });
            }
            keys.push(key);
        }

        let Parsed {
            ast,
            errors: parse_errors,
        } = match source {
            Source::Script(text) => Parsed::script(acvus_ast::parse_script(&interner, text)),
            Source::Template(text) => Parsed::template(acvus_ast::parse(&interner, text)),
        };
        refusals.extend(parse_errors.iter().map(|e| Refusal {
            message: e.kind.to_string(),
            span: span_of(e.span),
        }));

        let Externs {
            functions: extern_fns,
            types,
            handlers,
            space,
            instances,
        } = match Externs::combine(registries, &interner) {
            Ok(externs) => externs,
            Err(error) => {
                refusals.push(Refusal {
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
        let entry = QualifiedRef::root(interner.intern("main"));
        let functions: Vec<Function> = std::iter::once(Function {
            qref: entry,
            kind: FnKind::Local(ast),
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(R::declared(&interner)),
                captures: vec![],
                effect: Effect::OPAQUE.into(),
                flows: Flows::Every.into(),
            },
        })
        .chain(extern_fns)
        .collect();
        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(
                contexts
                    .into_iter()
                    .map(|(key, ty)| Context {
                        qref: QualifiedRef::root(interner.intern(&key)),
                        ty,
                    })
                    .collect(),
            ),
            types: Freeze::new(types),
            bindings,
            entry: Some(entry),
        };

        let ext = extract::extract(&interner, &graph);
        let inf = infer::infer(&interner, &graph, &ext);
        refusals.extend(
            inf.errors()
                .into_iter()
                .flat_map(|(_, errs)| errs.iter())
                .map(|e| Refusal {
                    message: e.display(&interner).to_string(),
                    span: span_of(e.span),
                }),
        );
        let lowered = lower::lower(&interner, &graph, &ext.view(), &inf);
        refusals.extend(
            lowered
                .errors
                .iter()
                .flat_map(|le| le.errors.iter())
                .map(|e| Refusal {
                    message: e.display(&interner).to_string(),
                    span: span_of(e.span),
                }),
        );
        if !refusals.is_empty() {
            return Err(refusals);
        }

        let optimized = optimize::optimize(
            &interner,
            &acvus_mir::laws::LawTable::of(graph.functions.iter()),
            lowered.modules,
            Opt::Full,
        );
        refusals.extend(
            optimized
                .errors
                .into_iter()
                .flat_map(|(_, errs)| errs)
                .map(|e| Refusal {
                    message: e.display(&interner).to_string(),
                    span: span_of(e.span),
                }),
        );
        if !refusals.is_empty() {
            return Err(refusals);
        }

        let Some(ret) = optimized.modules.get(&entry).map(|module| module.ret.clone()) else {
            panic!("the entry lowers to a module whenever no stage refused it")
        };
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
        let settled: HashMap<String, Ty> = graph
            .contexts
            .iter()
            .map(|c| {
                let Some(ty) = inf.context_types.get(&c.qref) else {
                    panic!("inference settles a type for every declared context")
                };
                (interner.resolve(c.qref.name).to_owned(), ty.clone())
            })
            .collect();
        Ok(Program {
            contexts: Contexts {
                rt: shared.runtime_over_an_empty_page(),
                settled,
            },
            shared,
            entry,
            ret,
            result: PhantomData,
        })
    }
}

fn span_of(span: Span) -> Option<Span> {
    (span.start != 0 || span.end != 0).then_some(span)
}

pub struct Program<R> {
    shared: InterpreterContext,
    entry: QualifiedRef,
    ret: Ty,
    contexts: Contexts,
    result: PhantomData<fn() -> R>,
}

impl<R> Program<R> {
    pub fn contexts(&self) -> &Contexts {
        &self.contexts
    }
}

impl<R> Program<R>
where
    R: Project<AcvusRuntime>,
{
    pub async fn run<P>(&self, page: &Arc<P>) -> Result<Output<R>, PageError>
    where
        P: Page + 'static,
    {
        let interner = &self.shared.interner;
        for (key, declared) in &self.contexts.settled {
            let Some(held) = page.held_type(key) else {
                return Err(PageError::Undeclared { key: key.clone() });
            };
            if !held.same_erased(declared) {
                return Err(PageError::Mismatched {
                    key: key.clone(),
                    held: held.display(interner).to_string(),
                    asked: declared.display(interner).to_string(),
                });
            }
        }
        let page: Arc<dyn RuntimeContext> = Arc::<P>::clone(page);
        let mut interpreter = Interpreter::on_page(self.shared.clone(), self.entry, page);
        let value = interpreter.execute().await;
        let table = R::table(ArgAt {
            interner,
            ty: &self.ret,
        });
        Ok(Output {
            // SAFETY: the run moved its result out to this caller, and no
            // other holder owns it.
            value: unsafe { Owned::from_value(Holding::new(), value) },
            rt: self.shared.runtime_over_an_empty_page(),
            table,
        })
    }
}

pub struct Output<R>
where
    R: Project<AcvusRuntime>,
{
    value: Owned<AcvusRuntime>,
    rt: AcvusRuntime,
    table: R::Table,
}

impl<R> Output<R>
where
    R: Project<AcvusRuntime>,
{
    pub fn get(&self) -> R::Ref<'_> {
        // SAFETY: the checker held the entry's result to `R`, whose table
        // this is, and the value lives as long as `self`.
        unsafe { R::project(&self.rt, &self.value, &self.table) }
    }

    pub fn get_mut(&mut self) -> R::Mut<'_> {
        // SAFETY: as `get`; `&mut self` names the value exclusively, and a
        // projection writes inside the storage the word names, never the
        // word itself.
        let value = unsafe { self.value.value_mut(Holding::new()) };
        // SAFETY: as `get`, exclusively.
        unsafe { R::project_mut(&self.rt, value, &self.table) }
    }
}

#[derive(Clone)]
pub struct Contexts {
    rt: AcvusRuntime,
    settled: HashMap<String, Ty>,
}

impl Contexts {
    pub fn settled(&self) -> &HashMap<String, Ty> {
        &self.settled
    }

    pub(crate) fn rt(&self) -> &AcvusRuntime {
        &self.rt
    }
}

pub trait Page: RuntimeContext {
    fn held_type(&self, key: &str) -> Option<&Ty>;
}

#[derive(Debug)]
pub enum PageError {
    Undeclared { key: String },
    Mismatched { key: String, held: String, asked: String },
    Absent { key: String },
    Space(SpaceError),
}

impl fmt::Display for PageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageError::Undeclared { key } => write!(f, "the page declares no `@{key}`"),
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
        return Err(PageError::Undeclared {
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
