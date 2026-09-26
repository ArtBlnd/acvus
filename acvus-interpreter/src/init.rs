//! A context's first value is its init: a body the host gives for that one
//! key, compiled into the graph beside the entries, or a Rust function whose
//! declared type joins the graph; a load that finds the storage lacking the
//! key runs it (RFC-0090 rule 1).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use acvus_extern::{Crossing, Owned};
use acvus_mir::graph::{Context, ContextInit, FnKind, Function, Inputs, ParsedAst, QualifiedRef};
use acvus_mir::ty::{Effect, Flows, PolyBuilder, PolyTy, Ty, TyTerm, lift_declaration};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::interpreter::Init;
use crate::runtime::AcvusRuntime;

pub(crate) struct InitSource {
    pub(crate) key: String,
    pub(crate) given: InitGiven,
}

pub(crate) enum InitGiven {
    Source(ParsedAst),
    Rust(RustInit),
}

type MakeInit = Box<dyn Fn(Crossing<'_, AcvusRuntime>) -> Owned<AcvusRuntime> + Send + Sync>;

pub(crate) struct RustInit {
    declared: PolyTy,
    make: MakeInit,
}

impl RustInit {
    pub(crate) fn new(declared: PolyTy, make: MakeInit) -> Self {
        RustInit { declared, make }
    }
}

pub(crate) struct SolvedRustInit {
    make: MakeInit,
}

impl SolvedRustInit {
    pub(crate) fn make(&self, rt: &AcvusRuntime) -> Owned<AcvusRuntime> {
        // SAFETY: `DeclaredInits::solved` makes a `SolvedRustInit` only where
        // the key's solved type is the one `make`'s value is declared at, and
        // `Init` holds the value at that solved type.
        (self.make)(unsafe { Crossing::new(rt) })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InitRefusal {
    Twice { key: String },
    NamesAContext { key: String, context: String },
    NameTaken { key: String },
    SolvedElsewhere { key: String, declared: String, solved: String },
}

impl InitRefusal {
    pub(crate) fn key(&self) -> &str {
        match self {
            InitRefusal::Twice { key }
            | InitRefusal::NamesAContext { key, .. }
            | InitRefusal::NameTaken { key }
            | InitRefusal::SolvedElsewhere { key, .. } => key,
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
            InitRefusal::SolvedElsewhere { key, declared, solved } => write!(
                f,
                "the init of `@{key}` makes a {declared}, and the scripts solve `@{key}` to {solved}"
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

enum DeclaredInit {
    Script(QualifiedRef),
    Rust(RustInit),
}

pub(crate) struct DeclaredInits {
    by_key: BTreeMap<String, DeclaredInit>,
}

impl DeclaredInits {
    pub(crate) fn declare(
        interner: &Interner,
        inits: Vec<InitSource>,
        graph: &mut GraphParts,
    ) -> Result<DeclaredInits, Vec<InitRefusal>> {
        let mut refusals = Vec::new();
        let mut by_key = BTreeMap::new();
        for InitSource { key, given } in inits {
            if by_key.contains_key(&key) {
                refusals.push(InitRefusal::Twice { key });
                continue;
            }
            let declared = match given {
                InitGiven::Source(ast) => match script_init(interner, &key, ast, graph) {
                    Ok(function) => DeclaredInit::Script(function),
                    Err(refused) => {
                        refusals.push(refused);
                        continue;
                    }
                },
                InitGiven::Rust(rust) => {
                    context_of(interner, &key, graph).init = Some(ContextInit::Declared(rust.declared.clone()));
                    DeclaredInit::Rust(rust)
                }
            };
            by_key.insert(key, declared);
        }
        match refusals.is_empty() {
            true => Ok(DeclaredInits { by_key }),
            false => Err(refusals),
        }
    }

    pub(crate) fn key_of(&self, qref: &QualifiedRef) -> Option<&str> {
        self.functions()
            .find(|(_, function)| function == qref)
            .map(|(key, _)| key)
    }

    pub(crate) fn functions(&self) -> impl Iterator<Item = (&str, QualifiedRef)> {
        self.by_key.iter().filter_map(|(key, init)| match init {
            DeclaredInit::Script(function) => Some((key.as_str(), *function)),
            DeclaredInit::Rust(_) => None,
        })
    }

    pub(crate) fn solved(
        self,
        interner: &Interner,
        solved: &BTreeMap<String, Arc<Ty>>,
    ) -> Result<FxHashMap<Box<str>, Init>, Vec<InitRefusal>> {
        let mut inits = FxHashMap::default();
        let mut refusals = Vec::new();
        for (key, init) in self.by_key {
            let Some(ty) = solved.get(&key) else {
                panic!("`declare` made `@{key}` a context of the graph")
            };
            let ty = Arc::clone(ty);
            let init = match init {
                DeclaredInit::Script(function) => Init::script(function, ty),
                DeclaredInit::Rust(RustInit { declared, make }) => {
                    if !lift_declaration(&ty, &mut PolyBuilder::new()).same_erased(&declared) {
                        refusals.push(InitRefusal::SolvedElsewhere {
                            declared: declared.display(interner).to_string(),
                            solved: ty.display(interner).to_string(),
                            key,
                        });
                        continue;
                    }
                    Init::rust(SolvedRustInit { make }, ty)
                }
            };
            inits.insert(key.into_boxed_str(), init);
        }
        match refusals.is_empty() {
            true => Ok(inits),
            false => Err(refusals),
        }
    }
}

fn context_of<'g>(interner: &Interner, key: &str, graph: &'g mut GraphParts) -> &'g mut Context {
    let context_ref = QualifiedRef::root(interner.intern(key));
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
    &mut graph.contexts[at]
}

fn script_init(
    interner: &Interner,
    key: &str,
    ast: ParsedAst,
    graph: &mut GraphParts,
) -> Result<QualifiedRef, InitRefusal> {
    let qref = init_ref(interner, key);
    let mut named: Vec<String> = crate::host::context_refs(&ast)
        .into_iter()
        .map(|context| interner.resolve(context.name).to_owned())
        .collect();
    named.sort();
    if let Some(context) = named.into_iter().next() {
        return Err(InitRefusal::NamesAContext {
            key: key.to_owned(),
            context,
        });
    }
    if graph.functions.iter().any(|function| function.qref == qref) {
        return Err(InitRefusal::NameTaken { key: key.to_owned() });
    }
    let context = context_of(interner, key, graph);
    context.init = Some(ContextInit::Body(qref));
    let ret = Box::new(context.ty.clone());
    graph.functions.push(Function {
        qref,
        kind: FnKind::Local(ast, Inputs::Declared),
        ty: TyTerm::Fn {
            params: Vec::new(),
            ret,
            captures: vec![],
            effect: Effect::OPAQUE.into(),
            flows: Flows::Every.into(),
        },
    });
    graph.entries.push(qref);
    Ok(qref)
}
