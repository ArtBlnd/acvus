//! A context's first value is its init: a body the host gives for that one
//! key, compiled into the graph beside the entries and run by a `Fetch` that
//! finds the storage lacks the key (RFC-0090 rule 1).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use acvus_mir::graph::{Context, FnKind, Function, Inputs, ParsedAst, QualifiedRef};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, Ty, TyTerm};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::interpreter::Init;

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

    pub(crate) fn functions(&self) -> impl Iterator<Item = (&str, QualifiedRef)> {
        self.by_key.iter().map(|(key, function)| (key.as_str(), *function))
    }

    pub(crate) fn solved(self, solved: &BTreeMap<String, Arc<Ty>>) -> FxHashMap<Box<str>, Init> {
        self.by_key
            .into_iter()
            .map(|(key, function)| {
                let Some(ty) = solved.get(&key) else {
                    panic!("`declare` made `@{key}` a context of the graph")
                };
                let ty = Arc::clone(ty);
                (key.into_boxed_str(), Init { function, ty })
            })
            .collect()
    }
}
