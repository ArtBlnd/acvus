//! A context's first value is its init: a body the host gives for that one
//! key, compiled into the graph beside the entries and run when a page opens
//! over a storage that lacks a context an entry fetches first (RFC-0090
//! rule 1).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use acvus_extern::{Holding, Owned};
use acvus_mir::graph::{Context, FnKind, Function, Inputs, ParsedAst, QualifiedRef};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, Ty, TyTerm};
use acvus_utils::Interner;

use crate::interpreter::{Absent, Interpreter, InterpreterContext};
use crate::journal::{Held, RuntimeContext};

pub(crate) struct InitSource {
    pub(crate) key: String,
    pub(crate) ast: ParsedAst,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InitRefusal {
    Twice { key: String },
    NamesAContext { key: String, context: String },
    NameTaken { key: String },
}

impl InitRefusal {
    pub(crate) fn key(&self) -> &str {
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

/// A source reads `@key` as the context, so no call a source writes reaches
/// the init.
fn init_ref(interner: &Interner, key: &str) -> QualifiedRef {
    QualifiedRef::root(interner.intern(&format!("@{key}")))
}

pub(crate) struct GraphParts {
    pub(crate) open: PolyBuilder,
    pub(crate) contexts: Vec<Context>,
    pub(crate) functions: Vec<Function>,
    pub(crate) entries: Vec<QualifiedRef>,
}

pub(crate) struct DeclaredInits {
    by_key: BTreeMap<String, QualifiedRef>,
}

impl DeclaredInits {
    pub(crate) fn declare(
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

    pub(crate) fn key_of(&self, qref: &QualifiedRef) -> Option<&str> {
        self.by_key
            .iter()
            .find(|(_, init)| *init == qref)
            .map(|(key, _)| key.as_str())
    }

    pub(crate) fn solved(self, solved: &BTreeMap<String, Arc<Ty>>) -> Inits {
        let by_key = self
            .by_key
            .into_iter()
            .map(|(key, function)| {
                let Some(ty) = solved.get(&key) else {
                    panic!("`declare` made `@{key}` a context of the graph")
                };
                let ty = Arc::clone(ty);
                (key, Init { function, ty })
            })
            .collect();
        Inits { by_key }
    }
}

struct Init {
    function: QualifiedRef,
    ty: Arc<Ty>,
}

pub(crate) struct Inits {
    by_key: BTreeMap<String, Init>,
}

impl Inits {
    pub(crate) fn has(&self, key: &str) -> bool {
        self.by_key.contains_key(key)
    }

    pub(crate) async fn fill_lacking<W>(
        &self,
        shared: &InterpreterContext,
        page: &Arc<RuntimeContext>,
        wanted: W,
    ) -> Result<Vec<String>, Absent>
    where
        W: Fn(&str) -> bool,
    {
        let mut filled = Vec::new();
        for (key, init) in &self.by_key {
            if !wanted(key) || page.holds(key) {
                continue;
            }
            let mut run = Interpreter::on_page(shared.clone(), init.function, Arc::clone(page));
            let value = run.accept_page()?.run(Vec::new()).await;
            // SAFETY: the run moved its result out to this caller, and no
            // other holder owns it.
            let value = unsafe { Owned::from_value(Holding::new(), value) };
            page.set_changed(key, Held::new(value, Arc::clone(&init.ty), shared.compilation));
            filled.push(key.to_owned());
        }
        Ok(filled)
    }
}
