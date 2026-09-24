//! A context's first value is its init: a body the host gives for that one
//! key, compiled into the graph beside the entries and run before a run
//! that fetches the context from a page that lacks it (RFC-0090 rule 1).

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use acvus_extern::{Holding, Owned};
use acvus_mir::graph::{Context, FnKind, Function, Inputs, ParsedAst, QualifiedRef};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, Ty, TyTerm};
use acvus_utils::Interner;

use crate::host::PageError;
use crate::interpreter::{Interpreter, InterpreterContext};
use crate::journal::{Held, RuntimeContext};

pub struct InitSource {
    pub key: String,
    pub ast: ParsedAst,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitRefusal {
    Twice { key: String },
    NamesAContext { key: String, context: String },
    NameTaken { key: String },
}

impl InitRefusal {
    pub fn key(&self) -> &str {
        match self {
            InitRefusal::Twice { key }
            | InitRefusal::NamesAContext { key, .. }
            | InitRefusal::NameTaken { key } => key,
        }
    }
}

impl fmt::Display for InitRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InitRefusal::Twice { key } => write!(f, "`@{key}` is given two inits"),
            InitRefusal::NamesAContext { key, context } => write!(
                f,
                "the init of `@{key}` names `@{context}`, and an init names no context"
            ),
            InitRefusal::NameTaken { key } => write!(
                f,
                "the init of `@{key}` is compiled as the function `@{key}`, which the graph already holds"
            ),
        }
    }
}

impl std::error::Error for InitRefusal {}

/// A source reads `@key` as the context, so no call a source writes reaches
/// the init.
fn init_ref(interner: &Interner, key: &str) -> QualifiedRef {
    QualifiedRef::root(interner.intern(&format!("@{key}")))
}

pub struct GraphParts {
    pub open: PolyBuilder,
    pub contexts: Vec<Context>,
    pub functions: Vec<Function>,
    pub entries: Vec<QualifiedRef>,
}

pub struct DeclaredInits {
    by_key: BTreeMap<String, QualifiedRef>,
}

macro_rules! declare_inits {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl DeclaredInits {
            $v fn declare(
                interner: &Interner,
                inits: Vec<InitSource>,
                graph: &mut GraphParts,
            ) -> Result<DeclaredInits, Vec<InitRefusal>> {
                let mut refusals = Vec::new();
                let mut by_key = BTreeMap::new();
                for InitSource { key, ast } in inits {
                    let qref = init_ref(interner, &key);
                    if by_key.contains_key(&key) {
                        refusals.push(InitRefusal::Twice { key });
                        continue;
                    }
                    let mut named: Vec<String> = crate::host::context_refs(&ast)
                        .into_iter()
                        .map(|context| interner.resolve(context.name).to_owned())
                        .collect();
                    named.sort();
                    if let Some(context) = named.into_iter().next() {
                        refusals.push(InitRefusal::NamesAContext { key, context });
                        continue;
                    }
                    if graph.functions.iter().any(|function| function.qref == qref) {
                        refusals.push(InitRefusal::NameTaken { key });
                        continue;
                    }
                    let context_ref = QualifiedRef::root(interner.intern(&key));
                    let at = match graph.contexts.iter().position(|c| c.qref == context_ref) {
                        Some(at) => at,
                        None => {
                            graph.contexts.push(Context {
                                qref: context_ref,
                                ty: graph.open.fresh_ty_var(),
                                init: None,
                            });
                            graph.contexts.len() - 1
                        }
                    };
                    let context = &mut graph.contexts[at];
                    context.init = Some(qref);
                    graph.functions.push(Function {
                        qref,
                        kind: FnKind::Local(ast, Inputs::Declared),
                        ty: TyTerm::Fn {
                            params: Vec::new(),
                            ret: Box::new(context.ty.clone()),
                            captures: vec![],
                            effect: Effect::OPAQUE.into(),
                            flows: Flows::Every.into(),
                        },
                    });
                    graph.entries.push(qref);
                    by_key.insert(key, qref);
                }
                match refusals.is_empty() {
                    true => Ok(DeclaredInits { by_key }),
                    false => Err(refusals),
                }
            }

            $v fn function(&self, key: &str) -> Option<QualifiedRef> {
                self.by_key.get(key).copied()
            }

            $v fn key_of(&self, qref: &QualifiedRef) -> Option<&str> {
                self.by_key
                    .iter()
                    .find(|(_, init)| *init == qref)
                    .map(|(key, _)| key.as_str())
            }

            $v fn solved(self, solved: &HashMap<String, Ty>) -> Inits {
                let by_key = self
                    .by_key
                    .into_iter()
                    .map(|(key, function)| {
                        let Some(ty) = solved.get(&key) else {
                            panic!("`declare` made `@{key}` a context of the graph")
                        };
                        let init = Init {
                            function,
                            ty: Arc::new(ty.clone()),
                        };
                        (key, init)
                    })
                    .collect();
                Inits { by_key }
            }
        }
    };
}
tooling_vis!(declare_inits);

struct Init {
    function: QualifiedRef,
    ty: Arc<Ty>,
}

struct Absent<'i> {
    key: &'i str,
    init: &'i Init,
}

pub struct Inits {
    by_key: BTreeMap<String, Init>,
}

macro_rules! fill_page {
    ($v:vis) => {
        #[cfg_attr(not(feature = "tooling"), allow(dead_code))]
        impl Inits {
            $v fn keys(&self) -> impl Iterator<Item = &str> {
                self.by_key.keys().map(String::as_str)
            }

            /// The keys filled, in the order given.
            $v async fn fill<'k, K>(
                &self,
                shared: &InterpreterContext,
                page: &Arc<dyn RuntimeContext>,
                keys: K,
            ) -> Result<Vec<String>, PageError>
            where
                K: IntoIterator<Item = &'k str>,
            {
                let mut absent: Vec<Absent<'_>> = Vec::new();
                for key in keys {
                    if page.holds(key) {
                        continue;
                    }
                    let Some((key, init)) = self.by_key.get_key_value(key) else {
                        return Err(PageError::Absent {
                            key: key.to_owned(),
                        });
                    };
                    absent.push(Absent { key, init });
                }
                let mut filled = Vec::with_capacity(absent.len());
                for Absent { key, init } in absent {
                    let mut run = Interpreter::on_page(shared.clone(), init.function, Arc::clone(page));
                    let value = run.accept_page()?.run(Vec::new()).await;
                    // SAFETY: the run moved its result out to this caller, and
                    // no other holder owns it.
                    let value = unsafe { Owned::from_value(Holding::new(), value) };
                    page.set(key, Held::new(value, Arc::clone(&init.ty)));
                    filled.push(key.to_owned());
                }
                Ok(filled)
            }

            $v async fn fill_absent(
                &self,
                shared: &InterpreterContext,
                page: &Arc<dyn RuntimeContext>,
            ) -> Result<Vec<String>, PageError> {
                self.fill(shared, page, self.keys()).await
            }
        }
    };
}
tooling_vis!(fill_page);
