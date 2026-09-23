//! The open documents of one compilation, each the local body of a
//! `Function` in an `IncrementalGraph`, which checks them. This layer
//! reads the checker's answers at a place in a document's source:
//! - `DocId` to the document's function, its source, its parse errors and
//!   the nodes of the tree the graph holds for it
//! - Refusals and parse errors as `LspError`s
//! - Hover, the type the checker's view (`BodyView`) gives a node
//! - Definition, references and rename, from what the view records each
//!   name resolves to; a rename is checked by the checker again
//!   (`IncrementalGraph::view_as`)
//! - Where a completion is asked and how its items read; what they are is
//!   the checker's answer (`IncrementalGraph::probe`)

use std::collections::BTreeMap;
use std::fmt;

use acvus_ast::lexer::{ExprTokenizer, Line, Piece, scan_template};
use acvus_ast::locate::{Name, Nodes};
use acvus_ast::report::Label;
use acvus_ast::token::{KEYWORDS, Token};
use acvus_ast::{AstId, Span};
use acvus_mir::error::Refusal;
use acvus_mir::graph::ContextInfo;
use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::types::*;
use acvus_mir::ty::{PolyTy, TyTerm};
use acvus_mir::typeck::{BodyView, DeclarationFit, ProbeProduct, Resolved};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

// -- Public types ----------------------------------------------------

/// Opaque document identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DocId(u32);

impl DocId {
    pub fn from_raw(id: u32) -> Self {
        Self(id)
    }
    pub fn raw(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Script,
    Template,
}

impl Mode {
    pub(crate) fn parse(self, interner: &Interner, source: &str) -> Parsed {
        match self {
            Mode::Script => Parsed::script(acvus_ast::parse_script(interner, source)),
            Mode::Template => Parsed::template(acvus_ast::parse(interner, source)),
        }
    }
}

/// `ty` must be the `Fn` type the host's batch path gives this source's
/// function; a different one checks a program the host never compiles.
#[derive(Debug, Clone)]
pub struct Document {
    pub qref: QualifiedRef,
    pub mode: Mode,
    pub ty: PolyTy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LspError {
    pub category: LspErrorCategory,
    pub message: String,
    pub span: Option<(usize, usize)>,
    /// The other places the refusal points at, which a protocol layer sends
    /// as `relatedInformation`.
    pub related: Vec<Label>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspErrorCategory {
    Parse,
    Type,
    Unreadable,
    Host,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: String,
    /// What replaces the identifier the cursor is in, which never holds the
    /// `$`, `@`, `ns::` or `.` before it.
    pub insert_text: String,
    /// Whether the checker joins the item, or for a function or a method a
    /// call of it, with the type the solve gave the cursor's position.
    /// `false` where the solve gave that position none.
    pub fits: bool,
    /// One per declaration a `Function` or `Method` item names, in the
    /// order `detail` shows them; empty for every other kind.
    pub calls: Vec<CallShape>,
}

/// The arguments a call of one declaration writes: for a method, those
/// after the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallShape {
    pub params: Vec<ParamHint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamHint {
    /// `None` where the declaration names the parameter `_`, the name a
    /// parameter without one is given.
    pub name: Option<String>,
    pub ty: String,
}

/// `completions` lists fitting items first, then items in this order, then
/// by label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CompletionKind {
    Local,
    Param,
    Field,
    Method,
    Function,
    Context,
    Keyword,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hover {
    pub span: (usize, usize),
    pub ty: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Definition {
    Local {
        span: (usize, usize),
    },
    /// A function whose body is in the graph.
    Function(QualifiedRef),
    Context(QualifiedRef),
    Input(Astr),
}

/// Replace the source at `span` with `text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edit {
    pub span: (usize, usize),
    pub text: String,
}

/// Why `rename` wrote no edits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameRefusal {
    /// The offset is on no name the checker resolved.
    NotAName,
    /// A function is named by the path of the file that defines it.
    Function,
    /// A context is declared by the host's context file.
    Context,
    /// An input is bound by the host.
    Input,
    Keyword(String),
    NotAnIdentifier(String),
    /// Checked with the new name, some name of the source resolves to a
    /// binding other than the one it resolved to before.
    ResolutionChanged(String),
}

impl fmt::Display for RenameRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenameRefusal::NotAName => write!(f, "there is no local binding to rename here"),
            RenameRefusal::Function => write!(
                f,
                "a function is named by the path of its file, which a rename here does not change"
            ),
            RenameRefusal::Context => write!(
                f,
                "a context is declared by the host's context file, which a rename here does not change"
            ),
            RenameRefusal::Input => write!(
                f,
                "an input is bound by the host, which a rename here does not change"
            ),
            RenameRefusal::Keyword(name) => write!(f, "`{name}` is a keyword"),
            RenameRefusal::NotAnIdentifier(name) => write!(f, "`{name}` is not an identifier"),
            RenameRefusal::ResolutionChanged(name) => write!(
                f,
                "renaming to `{name}` changes which binding a name of this document refers to"
            ),
        }
    }
}

impl std::error::Error for RenameRefusal {}

/// Why `open` opened no document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenRefusal {
    /// The document's function is the body of the open document `holder`;
    /// the graph holds one body per function, so a second would hide the
    /// first's tree from every query on it.
    FunctionHeld {
        /// As a script writes it: `ns::name`, or the bare name.
        function: String,
        holder: DocId,
    },
}

impl fmt::Display for OpenRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenRefusal::FunctionHeld { function, .. } => write!(
                f,
                "function `{function}` is already the body of an open document"
            ),
        }
    }
}

impl std::error::Error for OpenRefusal {}

/// The names a rename writes anew, which every compilation holding the
/// document must resolve as before.
#[derive(Debug, Clone)]
pub(crate) struct RenamePlan {
    new_name: String,
    /// In source order.
    renamed: Vec<Name>,
    /// One per renamed name, in the same order.
    edits: Vec<Edit>,
}

impl RenamePlan {
    /// One edit per name: the new name, or for a shorthand field `{ a }`
    /// the key kept, `a: b`.
    fn new(source: &str, new_name: &str, renamed: Vec<Name>) -> Self {
        let edits = renamed
            .iter()
            .map(|name| Edit {
                span: (name.span.start, name.span.end),
                text: match name.shorthand {
                    true => format!("{}: {new_name}", &source[name.span.start..name.span.end]),
                    false => new_name.to_string(),
                },
            })
            .collect();
        Self {
            new_name: new_name.to_string(),
            renamed,
            edits,
        }
    }

    pub(crate) fn edits(&self) -> &[Edit] {
        &self.edits
    }

    /// Where a span of the source is after the edits: a renamed name at the
    /// new name, anything else moved by the edits before its ends.
    fn moved(&self, span: Span) -> Span {
        let shift = |at: usize| -> usize {
            let grown: usize = self
                .edits
                .iter()
                .filter(|edit| edit.span.1 <= at)
                .map(|edit| edit.text.len())
                .sum();
            let shrunk: usize = self
                .edits
                .iter()
                .filter(|edit| edit.span.1 <= at)
                .map(|edit| edit.span.1 - edit.span.0)
                .sum();
            at + grown - shrunk
        };
        match self.renamed.iter().find(|name| name.span == span) {
            Some(name) => {
                let key = match name.shorthand {
                    true => span.end - span.start + ": ".len(),
                    false => 0,
                };
                let start = shift(span.start) + key;
                Span::new(start, start + self.new_name.len())
            }
            None => Span::new(shift(span.start), shift(span.end)),
        }
    }
}

// -- LspSession ------------------------------------------------------

pub struct LspSession {
    graph: IncrementalGraph,
    documents: FxHashMap<DocId, Open>,
    next_doc_id: u32,
}

/// A document as its last parse left it.
struct Open {
    document: Document,
    source: String,
    parse_errors: Vec<acvus_ast::ParseError>,
    /// The nodes of the tree the graph holds as the document's body, whose
    /// ids the checker's view is keyed by.
    nodes: Nodes,
}

impl LspSession {
    pub fn new(interner: &Interner, environment: CompilationGraph) -> Self {
        Self {
            graph: IncrementalGraph::new(interner, environment),
            documents: FxHashMap::default(),
            next_doc_id: 0,
        }
    }

    pub fn interner(&self) -> &Interner {
        self.graph.interner()
    }

    pub fn graph(&self) -> &IncrementalGraph {
        &self.graph
    }

    pub fn set_environment(&mut self, environment: CompilationGraph) {
        let interner = self.graph.interner().clone();
        self.graph = IncrementalGraph::new(&interner, environment);
        let open: Vec<DocId> = self.documents.keys().copied().collect();
        for id in open {
            let open = &self.documents[&id];
            let document = open.document.clone();
            let source = open.source.clone();
            self.load(id, document, &source);
        }
    }

    // -- Document lifecycle ------------------------------------------

    /// Refused, leaving the session as it was, when an open document is
    /// already the body of `document`'s function: one function has at most
    /// one open document.
    pub fn open(&mut self, document: Document, source: &str) -> Result<DocId, OpenRefusal> {
        if let Some(holder) = self.holder(document.qref) {
            return Err(OpenRefusal::FunctionHeld {
                function: qualified(self.graph.interner(), document.qref),
                holder,
            });
        }
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;
        self.load(doc_id, document, source);
        Ok(doc_id)
    }

    /// The open document whose body is `qref`'s function.
    fn holder(&self, qref: QualifiedRef) -> Option<DocId> {
        self.documents
            .iter()
            .find(|(_, open)| open.document.qref == qref)
            .map(|(id, _)| *id)
    }

    pub fn update_source(&mut self, id: DocId, source: &str) {
        if let Some(open) = self.documents.get(&id) {
            let document = open.document.clone();
            self.load(id, document, source);
        }
    }

    /// Parse `source` as `document`'s body, hand the tree to the graph, and
    /// keep its nodes beside the document.
    fn load(&mut self, id: DocId, document: Document, source: &str) {
        let interner = self.graph.interner().clone();
        let Parsed { ast, errors } = document.mode.parse(&interner, source);
        let nodes = nodes_of(&ast);
        match self.graph.function(document.qref) {
            Some(_) => self.graph.update_ast(document.qref, ast),
            None => self.graph.add_function(Function {
                qref: document.qref,
                kind: FnKind::Local(ast),
                ty: document.ty.clone(),
            }),
        }
        self.documents.insert(
            id,
            Open {
                document,
                source: source.to_string(),
                parse_errors: errors,
                nodes,
            },
        );
    }

    /// Close a document and remove its function from the graph. `open`
    /// refuses a second document of one function, so the function removed
    /// is the body of no other open document.
    pub fn close(&mut self, id: DocId) {
        if let Some(open) = self.documents.remove(&id) {
            self.graph.remove_function(open.document.qref);
        }
    }

    // -- Inputs ------------------------------------------------------

    /// The inputs a run starting at this document requires: its own and those
    /// of every function it calls, since one host injects the `$` names of
    /// the whole graph (RFC-0071 rule 4).
    pub fn required_inputs(&self, id: DocId) -> Vec<ContextInfo> {
        let Some(qref) = self.function_ref(id) else {
            return vec![];
        };
        self.graph.required_inputs(qref)
    }

    /// Get the QualifiedRef for a document.
    pub fn function_ref(&self, id: DocId) -> Option<QualifiedRef> {
        self.documents.get(&id).map(|open| open.document.qref)
    }

    // -- Queries -----------------------------------------------------

    /// Diagnostics for a document.
    pub fn diagnostics(&self, id: DocId) -> Vec<LspError> {
        let Some(qref) = self.function_ref(id) else {
            return vec![];
        };
        let interner = self.graph.interner();
        let parse_errors = self
            .documents
            .get(&id)
            .into_iter()
            .flat_map(|open| &open.parse_errors)
            .map(parse_error_to_lsp);
        let refusals = self
            .graph
            .diagnostics(qref)
            .iter()
            .map(|refusal| refusal_to_lsp(refusal, interner));
        let mut diagnostics: Vec<LspError> = Vec::new();
        for diagnostic in parse_errors.chain(refusals) {
            if !diagnostics.contains(&diagnostic) {
                diagnostics.push(diagnostic);
            }
        }
        diagnostics
    }

    /// Context/param info for a document.
    pub fn context_info(&self, id: DocId) -> Vec<ContextInfo> {
        let Some(qref) = self.function_ref(id) else {
            return vec![];
        };
        self.graph.context_info(qref)
    }

    /// The names that can be written where the identifier at `cursor`
    /// is, starting with what is typed of it up to `cursor`. In a template
    /// only a `%` line and a `{{ }}` tag hold names.
    pub fn completions(&self, id: DocId, cursor: usize) -> Vec<CompletionItem> {
        let Some(Open {
            document, source, ..
        }) = self.documents.get(&id)
        else {
            return vec![];
        };
        let Some(site) = CompletionSite::at(source, cursor, document.mode) else {
            return vec![];
        };
        let interner = self.graph.interner();
        let mut items = match site.after {
            Sigil::None => match self.probe(document, source, &site) {
                Some(ProbeProduct::Value {
                    scope, functions, ..
                }) => {
                    let mut items = self.function_items(None, &fitting(&functions));
                    items.extend(keyword_items());
                    items.extend(scope.iter().map(|visible| CompletionItem {
                        label: interner.resolve(visible.name).to_string(),
                        kind: CompletionKind::Local,
                        detail: visible.ty.display(interner).to_string(),
                        insert_text: interner.resolve(visible.name).to_string(),
                        fits: visible.fits,
                        calls: vec![],
                    }));
                    items
                }
                Some(ProbeProduct::Member { .. }) | None => {
                    let mut items = self.function_items(None, &FxHashSet::default());
                    items.extend(keyword_items());
                    items
                }
            },
            Sigil::Input => match self.probe(document, source, &site) {
                Some(ProbeProduct::Value { inputs, .. }) => inputs
                    .iter()
                    .map(|input| CompletionItem {
                        label: format!("${}", interner.resolve(input.name)),
                        kind: CompletionKind::Param,
                        detail: input.ty.display(interner).to_string(),
                        insert_text: interner.resolve(input.name).to_string(),
                        fits: input.fits,
                        calls: vec![],
                    })
                    .collect(),
                Some(ProbeProduct::Member { .. }) | None => vec![],
            },
            Sigil::Context => {
                let fitting = match self.probe(document, source, &site) {
                    Some(ProbeProduct::Value { contexts, .. }) => fitting(&contexts),
                    Some(ProbeProduct::Member { .. }) | None => FxHashSet::default(),
                };
                self.graph
                    .visible_contexts()
                    .map(|context| CompletionItem {
                        label: format!("@{}", interner.resolve(context.qref.name)),
                        kind: CompletionKind::Context,
                        detail: context.ty.display(interner).to_string(),
                        insert_text: interner.resolve(context.qref.name).to_string(),
                        fits: fitting.contains(&context.qref),
                        calls: vec![],
                    })
                    .collect()
            }
            Sigil::Qualifier(namespace) => {
                let fitting = match self.probe(document, source, &site) {
                    Some(ProbeProduct::Value { functions, .. }) => fitting(&functions),
                    Some(ProbeProduct::Member { .. }) | None => FxHashSet::default(),
                };
                self.function_items(Some(namespace), &fitting)
            }
            Sigil::Member => match self.probe(document, source, &site) {
                Some(ProbeProduct::Member {
                    fields, methods, ..
                }) => {
                    let fields = fields.iter().map(|field| CompletionItem {
                        label: interner.resolve(field.name).to_string(),
                        kind: CompletionKind::Field,
                        detail: field.ty.display(interner).to_string(),
                        insert_text: interner.resolve(field.name).to_string(),
                        fits: field.fits,
                        calls: vec![],
                    });
                    let methods = methods.iter().map(|method| {
                        let types: Vec<&PolyTy> = method
                            .admitted
                            .iter()
                            .map(|declaration| &declaration.ty)
                            .collect();
                        CompletionItem {
                            label: interner.resolve(method.name).to_string(),
                            kind: CompletionKind::Method,
                            detail: shown_types(&types, interner),
                            insert_text: interner.resolve(method.name).to_string(),
                            fits: method.fits,
                            calls: types
                                .iter()
                                .map(|ty| call_shape(ty, Arguments::AfterReceiver, interner))
                                .collect(),
                        }
                    });
                    fields.chain(methods).collect()
                }
                Some(ProbeProduct::Value { .. }) | None => vec![],
            },
        };
        items.retain(|item| item.insert_text.starts_with(site.typed));
        items.sort_by(|a, b| {
            b.fits
                .cmp(&a.fits)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.label.cmp(&b.label))
        });
        items
    }

    /// The functions a bare name reaches, one item per name, or the ones
    /// `namespace::` qualifies. An item fits where a function it names is
    /// in `fitting`.
    fn function_items(
        &self,
        namespace: Option<&str>,
        fitting: &FxHashSet<QualifiedRef>,
    ) -> Vec<CompletionItem> {
        let interner = self.graph.interner();
        let mut by_name: BTreeMap<&str, Vec<&Function>> = BTreeMap::new();
        for function in self.graph.functions() {
            let qualified_as = function.qref.namespace.map(|ns| interner.resolve(ns));
            if namespace.is_some() && qualified_as != namespace {
                continue;
            }
            by_name
                .entry(interner.resolve(function.qref.name))
                .or_default()
                .push(function);
        }
        by_name
            .into_iter()
            .map(|(name, functions)| {
                let types: Vec<&PolyTy> = functions.iter().map(|function| &function.ty).collect();
                CompletionItem {
                    label: name.to_string(),
                    kind: CompletionKind::Function,
                    detail: shown_types(&types, interner),
                    insert_text: name.to_string(),
                    fits: functions
                        .iter()
                        .any(|function| fitting.contains(&function.qref)),
                    calls: types
                        .iter()
                        .map(|ty| call_shape(ty, Arguments::All, interner))
                        .collect(),
                }
            })
            .collect()
    }

    /// What the checker sees with the identifier at `site` replaced by a
    /// name no binding of the source has.
    fn probe(
        &self,
        document: &Document,
        source: &str,
        site: &CompletionSite<'_>,
    ) -> Option<ProbeProduct> {
        let interner = self.graph.interner();
        let marker = unwritten_name(source);
        let probed = format!(
            "{}{marker}{}",
            &source[..site.word.start],
            &source[site.word.end..]
        );
        let Parsed { ast, .. } = document.mode.parse(interner, &probed);
        let marked: AstId = nodes_of(&ast).at(site.word.start).first()?.id;
        let body = Function {
            qref: document.qref,
            kind: FnKind::Local(ast),
            ty: document.ty.clone(),
        };
        self.graph.probe(body, marked)
    }

    /// The type of the innermost node at `offset` that has one.
    pub fn hover(&self, id: DocId, offset: usize) -> Option<Hover> {
        let (nodes, view) = self.checked_body(id)?;
        let interner = self.graph.interner();
        nodes.at(offset).into_iter().find_map(|node| {
            let ty = view.types.get(&node.id)?;
            Some(Hover {
                span: (node.span.start, node.span.end),
                ty: ty.display(interner).to_string(),
            })
        })
    }

    /// Where the name at `offset` is defined: the binder of a local, the
    /// binder itself included, the function a call settled on where the
    /// graph holds its body, or the context or input it reads.
    pub fn definition(&self, id: DocId, offset: usize) -> Option<Definition> {
        let (nodes, view) = self.checked_body(id)?;
        let name = nodes.name_at(offset)?;
        match *view.resolved.get(&name.id)? {
            Resolved::Local(binder) => {
                let span = nodes
                    .names()
                    .iter()
                    .find(|name| name.id == binder)
                    .expect("a binder is a name of the body its uses are in")
                    .span;
                Some(Definition::Local {
                    span: (span.start, span.end),
                })
            }
            Resolved::Function(qref) => match self.graph.function(qref)?.kind {
                FnKind::Local(_) => Some(Definition::Function(qref)),
                FnKind::Extern { .. } => None,
            },
            Resolved::Context(qref) => Some(Definition::Context(qref)),
            Resolved::Input(name) => Some(Definition::Input(name)),
        }
    }

    /// The places in this document that refer to what the name at `offset`
    /// refers to: a local's uses, and with `include_declaration` its
    /// binder; a function's call sites, and its declaration, the start of
    /// its document, where that is this one; a context's or an input's
    /// references. `None` where no name at `offset` resolved.
    pub fn references(
        &self,
        id: DocId,
        offset: usize,
        include_declaration: bool,
    ) -> Option<Vec<(usize, usize)>> {
        let referent = self.referent(id, offset)?;
        Some(self.references_to(id, referent, include_declaration))
    }

    /// What the name at `offset` refers to.
    pub(crate) fn referent(&self, id: DocId, offset: usize) -> Option<Resolved> {
        let (nodes, view) = self.checked_body(id)?;
        let name = nodes.name_at(offset)?;
        view.resolved.get(&name.id).copied()
    }

    /// The names of this document that refer to `referent`, in source
    /// order.
    pub(crate) fn references_to(
        &self,
        id: DocId,
        referent: Resolved,
        include_declaration: bool,
    ) -> Vec<(usize, usize)> {
        let Some((nodes, view)) = self.checked_body(id) else {
            return vec![];
        };
        let refers = |name: &Name| {
            let declares = referent == Resolved::Local(name.id);
            view.resolved.get(&name.id) == Some(&referent) && (include_declaration || !declares)
        };
        let declared_here = match referent {
            Resolved::Function(qref) => include_declaration && self.function_ref(id) == Some(qref),
            Resolved::Local(_) | Resolved::Context(_) | Resolved::Input(_) => false,
        };
        let mut spans: Vec<(usize, usize)> = nodes
            .names()
            .iter()
            .filter(|name| refers(name))
            .map(|name| (name.span.start, name.span.end))
            .chain(declared_here.then_some((0, 0)))
            .collect();
        spans.sort_unstable();
        spans.dedup();
        spans
    }

    /// The edits that rename the local at `offset` to `new_name`: its
    /// binder and every use. Refused unless the source with the edits,
    /// checked beside the graph as it stands, resolves every name as this
    /// one does, the renamed ones to the renamed binder.
    pub fn rename(
        &self,
        id: DocId,
        offset: usize,
        new_name: &str,
    ) -> Result<Vec<Edit>, RenameRefusal> {
        let plan = self.plan_rename(id, offset, new_name)?;
        self.check_rename(id, &plan)?;
        Ok(plan.edits)
    }

    pub(crate) fn plan_rename(
        &self,
        id: DocId,
        offset: usize,
        new_name: &str,
    ) -> Result<RenamePlan, RenameRefusal> {
        let binder = match self.referent(id, offset) {
            Some(Resolved::Local(binder)) => binder,
            Some(Resolved::Function(_)) => return Err(RenameRefusal::Function),
            Some(Resolved::Context(_)) => return Err(RenameRefusal::Context),
            Some(Resolved::Input(_)) => return Err(RenameRefusal::Input),
            None => return Err(RenameRefusal::NotAName),
        };
        if KEYWORDS.contains(&new_name) {
            return Err(RenameRefusal::Keyword(new_name.to_string()));
        }
        let tokens: Vec<(usize, Token, usize)> =
            ExprTokenizer::new(new_name, 0, self.graph.interner()).collect();
        let [(0, Token::Ident(_), end)] = tokens.as_slice() else {
            return Err(RenameRefusal::NotAnIdentifier(new_name.to_string()));
        };
        if *end != new_name.len() {
            return Err(RenameRefusal::NotAnIdentifier(new_name.to_string()));
        }
        let (nodes, view) = self
            .checked_body(id)
            .expect("a document with a referent is checked");
        let source = &self.documents[&id].source;
        let renamed = nodes
            .names()
            .iter()
            .filter(|name| view.resolved.get(&name.id) == Some(&Resolved::Local(binder)))
            .copied()
            .collect();
        Ok(RenamePlan::new(source, new_name, renamed))
    }

    /// Whether this open document with `plan`'s edits, checked as its
    /// function's body on a copy of the graph's sources, resolves each name
    /// as it does now, compared by where the names are after the edits.
    pub(crate) fn check_rename(&self, id: DocId, plan: &RenamePlan) -> Result<(), RenameRefusal> {
        let Open {
            document, source, ..
        } = &self.documents[&id];
        let (nodes, view) = self
            .checked_body(id)
            .expect("an open document's body is checked");
        let renamed_source = apply(source, &plan.edits);
        let interner = self.graph.interner();
        let Parsed { ast, .. } = document.mode.parse(interner, &renamed_source);
        let renamed_nodes = nodes_of(&ast);
        let renamed_view = self
            .graph
            .view_as(Function {
                qref: document.qref,
                kind: FnKind::Local(ast),
                ty: document.ty.clone(),
            })
            .expect("a document's function is local, so its body is checked");
        let expected = Resolution::of(nodes, &view, |span| plan.moved(span));
        let found = Resolution::of(&renamed_nodes, &renamed_view, |span| span);
        match expected == found {
            true => Ok(()),
            false => Err(RenameRefusal::ResolutionChanged(plan.new_name.clone())),
        }
    }

    /// The nodes of the document's body and the checker's view of it.
    fn checked_body(&self, id: DocId) -> Option<(&Nodes, Freeze<BodyView>)> {
        let open = self.documents.get(&id)?;
        Some((&open.nodes, self.graph.view(open.document.qref)?))
    }
}

/// What the names of a body resolve to, by where the names are written,
/// a local by where its binder is. A name the checker records no
/// resolution for is not in it, so a name that comes to resolve is a
/// difference as much as one that stops.
#[derive(Debug, PartialEq, Eq)]
struct Resolution(FxHashMap<Span, Resolved<Span>>);

impl Resolution {
    fn of<F>(nodes: &Nodes, view: &BodyView, at: F) -> Self
    where
        F: Fn(Span) -> Span,
    {
        let spans: FxHashMap<AstId, Span> = nodes
            .names()
            .iter()
            .map(|name| (name.id, name.span))
            .collect();
        let resolved = view
            .resolved
            .iter()
            .filter_map(|(id, resolved)| {
                let Some(&written) = spans.get(id) else {
                    return match resolved {
                        // An index's or a `for`'s callee is written as no
                        // name: the instance it settles on follows from
                        // the types of what the names resolve to, so it is
                        // compared through them.
                        Resolved::Function(_) => None,
                        Resolved::Local(_) | Resolved::Context(_) | Resolved::Input(_) => panic!(
                            "the checker resolves a local, a context or an input only where a name is written"
                        ),
                    };
                };
                let by_span = match *resolved {
                    Resolved::Local(binder) => Resolved::Local(at(*spans
                        .get(&binder)
                        .expect("a binder is a name of the body its uses are in"))),
                    Resolved::Function(qref) => Resolved::Function(qref),
                    Resolved::Context(qref) => Resolved::Context(qref),
                    Resolved::Input(input) => Resolved::Input(input),
                };
                Some((at(written), by_span))
            })
            .collect();
        Self(resolved)
    }
}

/// `edits` are in source order and do not overlap.
/// A function's name as a script writes it: `ns::name`, or the bare name.
pub(crate) fn qualified(interner: &Interner, qref: QualifiedRef) -> String {
    match qref.namespace {
        Some(ns) => format!("{}::{}", interner.resolve(ns), interner.resolve(qref.name)),
        None => interner.resolve(qref.name).to_string(),
    }
}

fn apply(source: &str, edits: &[Edit]) -> String {
    let mut applied = String::with_capacity(source.len());
    let mut copied = 0;
    for edit in edits {
        applied.push_str(&source[copied..edit.span.0]);
        applied.push_str(&edit.text);
        copied = edit.span.1;
    }
    applied.push_str(&source[copied..]);
    applied
}

fn nodes_of(ast: &ParsedAst) -> Nodes {
    match ast {
        ParsedAst::Script(script) => Nodes::of_script(script),
        ParsedAst::Template(template) => Nodes::of_template(template),
        ParsedAst::Recovered(RecoveredAst::Script(script)) => Nodes::of_script(script),
        ParsedAst::Recovered(RecoveredAst::Template(template)) => Nodes::of_template(template),
    }
}

// -- Completion site -------------------------------------------------

/// What precedes the identifier completion replaces, which says what kind
/// of name it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sigil<'s> {
    None,
    Input,
    Context,
    Qualifier(&'s str),
    Member,
}

struct CompletionSite<'s> {
    word: Span,
    typed: &'s str,
    after: Sigil<'s>,
}

impl<'s> CompletionSite<'s> {
    fn at(source: &'s str, cursor: usize, mode: Mode) -> Option<Self> {
        if !source.is_char_boundary(cursor) {
            return None;
        }
        let code = code_around(source, cursor, mode)?;
        let start = source[code.start..cursor]
            .char_indices()
            .rev()
            .take_while(|&(_, c)| is_identifier_char(c))
            .last()
            .map_or(cursor, |(at, _)| code.start + at);
        let end = source[cursor..code.end]
            .char_indices()
            .find(|&(_, c)| !is_identifier_char(c))
            .map_or(code.end, |(at, _)| cursor + at);
        if source[start..end].starts_with(|c: char| c.is_numeric()) {
            return None;
        }
        let before = &source[code.start..start];
        let after = if before.ends_with('$') {
            Sigil::Input
        } else if before.ends_with('@') {
            Sigil::Context
        } else if let Some(qualified) = before.strip_suffix("::") {
            let namespace = trailing_identifier(qualified);
            if namespace.is_empty() {
                return None;
            }
            Sigil::Qualifier(namespace)
        } else if before.ends_with('.') && !before.ends_with("..") {
            Sigil::Member
        } else {
            Sigil::None
        };
        Some(Self {
            word: Span::new(start, end),
            typed: &source[start..cursor],
            after,
        })
    }
}

/// The source that holds code around `cursor`: all of a script; in a
/// template, the `%` line or the `{{ }}` tag the cursor is in.
fn code_around(source: &str, cursor: usize, mode: Mode) -> Option<Span> {
    let holds = |span: Span| span.start <= cursor && cursor <= span.end;
    match mode {
        Mode::Script => Some(Span::new(0, source.len())),
        Mode::Template => scan_template(source)
            .into_iter()
            .find_map(|line| match line {
                Line::Stmt { span, .. } => holds(span).then_some(span),
                Line::Text { pieces, .. } => pieces.into_iter().find_map(|piece| match piece {
                    Piece::Tag { inner_span, .. } => holds(inner_span).then_some(inner_span),
                    Piece::Text { .. } => None,
                }),
            }),
    }
}

/// The characters of `[\p{L}_][\p{L}\p{N}_]*`, the lexer's identifier.
fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn trailing_identifier(text: &str) -> &str {
    let start = text
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_identifier_char(c))
        .last()
        .map_or(text.len(), |(at, _)| at);
    &text[start..]
}

/// An identifier the source does not contain, so no binding, input or
/// field of the source is named by it.
fn unwritten_name(source: &str) -> String {
    (0..)
        .map(|n| format!("__acvus_completion_{n}"))
        .find(|name| !source.contains(name.as_str()))
        .expect("a source holds finitely many names")
}

/// A keyword is no value, so it fits no type.
fn keyword_items() -> impl Iterator<Item = CompletionItem> {
    KEYWORDS.iter().map(|keyword| CompletionItem {
        label: keyword.to_string(),
        kind: CompletionKind::Keyword,
        detail: "keyword".to_string(),
        insert_text: keyword.to_string(),
        fits: false,
        calls: vec![],
    })
}

fn fitting(declarations: &[DeclarationFit]) -> FxHashSet<QualifiedRef> {
    declarations
        .iter()
        .filter(|declaration| declaration.fits)
        .map(|declaration| declaration.qref)
        .collect()
}

/// Which parameters of a declaration a call written at the cursor still
/// writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arguments {
    All,
    /// A method call's receiver is written before the `.`.
    AfterReceiver,
}

fn call_shape(ty: &PolyTy, arguments: Arguments, interner: &Interner) -> CallShape {
    let TyTerm::Fn { params, .. } = ty else {
        panic!("a declared signature is a function type");
    };
    let written = match arguments {
        Arguments::All => params.as_slice(),
        Arguments::AfterReceiver => {
            let (_receiver, after) = params
                .split_first()
                .expect("a method's declaration takes the receiver first");
            after
        }
    };
    CallShape {
        params: written
            .iter()
            .map(|param| {
                let name = interner.resolve(param.name);
                ParamHint {
                    name: (name != "_").then(|| name.to_string()),
                    ty: param.ty.display(interner).to_string(),
                }
            })
            .collect(),
    }
}

/// The types of the declarations one name reaches.
fn shown_types(types: &[&PolyTy], interner: &Interner) -> String {
    types
        .iter()
        .map(|ty| ty.display(interner).to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

// -- Refusal -> LspError ---------------------------------------------

fn refusal_to_lsp(refusal: &Refusal, interner: &Interner) -> LspError {
    LspError {
        category: LspErrorCategory::Type,
        message: format!("{}", refusal.display(interner)),
        span: {
            let s = refusal.span();
            if s.start != 0 || s.end != 0 {
                Some((s.start, s.end))
            } else {
                None
            }
        },
        related: refusal.labels().to_vec(),
    }
}

pub(crate) fn parse_error_to_lsp(error: &acvus_ast::ParseError) -> LspError {
    LspError {
        category: LspErrorCategory::Parse,
        message: error.kind.to_string(),
        span: (error.span.start != 0 || error.span.end != 0)
            .then_some((error.span.start, error.span.end)),
        related: Vec::new(),
    }
}
