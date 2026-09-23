//! LSP session - thin wrapper over `IncrementalGraph`.
//!
//! Each document maps to a `Function` in the graph.
//! Namespace scoping, caching, and incremental recompilation are all
//! handled by `IncrementalGraph`. This layer only provides:
//! - DocId <-> FunctionId mapping
//! - MirError -> LspError conversion
//! - Where a completion is asked and how its items read; what they are is
//!   the checker's answer (`IncrementalGraph::probe`)

use std::collections::BTreeMap;

use acvus_ast::lexer::{Line, Piece, scan_template};
use acvus_ast::locate::Nodes;
use acvus_ast::report::Label;
use acvus_ast::token::KEYWORDS;
use acvus_ast::{AstId, Span};
use acvus_mir::error::Refusal;
use acvus_mir::graph::ContextInfo;
use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::types::*;
use acvus_mir::ty::PolyTy;
use acvus_mir::typeck::{BodyView, ProbeProduct};
use acvus_utils::{Freeze, Interner};
use rustc_hash::FxHashMap;

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
    pub(crate) fn parse(
        self,
        interner: &Interner,
        source: &str,
    ) -> Result<ParsedAst, acvus_ast::ParseError> {
        match self {
            Mode::Script => acvus_ast::parse_script(interner, source).map(ParsedAst::Script),
            Mode::Template => acvus_ast::parse(interner, source).map(ParsedAst::Template),
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
}

/// `completions` lists items in this order, then by label.
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
}

// -- LspSession ------------------------------------------------------

pub struct LspSession {
    graph: IncrementalGraph,
    documents: FxHashMap<DocId, Document>,
    doc_sources: FxHashMap<DocId, String>,
    doc_parse_errors: FxHashMap<DocId, acvus_ast::ParseError>,
    next_doc_id: u32,
}

impl LspSession {
    pub fn new(interner: &Interner, environment: CompilationGraph) -> Self {
        Self {
            graph: IncrementalGraph::new(interner, environment),
            documents: FxHashMap::default(),
            doc_sources: FxHashMap::default(),
            doc_parse_errors: FxHashMap::default(),
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
            let source = self.doc_sources[&id].clone();
            self.reparse(id, &source);
        }
    }

    // -- Document lifecycle ------------------------------------------

    pub fn open(&mut self, document: Document, source: &str) -> DocId {
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;
        self.documents.insert(doc_id, document);
        self.reparse(doc_id, source);
        doc_id
    }

    pub fn update_source(&mut self, id: DocId, source: &str) {
        if self.documents.contains_key(&id) {
            self.reparse(id, source);
        }
    }

    fn reparse(&mut self, id: DocId, source: &str) {
        let interner = self.graph.interner().clone();
        let document = self.documents[&id].clone();
        self.doc_sources.insert(id, source.to_string());
        let ast = match document.mode.parse(&interner, source) {
            Ok(ast) => ast,
            Err(error) => {
                self.doc_parse_errors.insert(id, error);
                self.graph.remove_function(document.qref);
                return;
            }
        };
        self.doc_parse_errors.remove(&id);
        match self.graph.function(document.qref) {
            Some(_) => self.graph.update_ast(document.qref, ast),
            None => self.graph.add_function(Function {
                qref: document.qref,
                kind: FnKind::Local(ast),
                ty: document.ty,
            }),
        }
    }

    /// Close a document. Removes the Function from the graph.
    pub fn close(&mut self, id: DocId) {
        if let Some(document) = self.documents.remove(&id) {
            self.graph.remove_function(document.qref);
        }
        self.doc_sources.remove(&id);
        self.doc_parse_errors.remove(&id);
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
        self.documents.get(&id).map(|document| document.qref)
    }

    // -- Queries -----------------------------------------------------

    /// Diagnostics for a document.
    pub fn diagnostics(&self, id: DocId) -> Vec<LspError> {
        if let Some(error) = self.doc_parse_errors.get(&id) {
            return vec![parse_error_to_lsp(error)];
        }
        let Some(qref) = self.function_ref(id) else {
            return vec![];
        };
        let interner = self.graph.interner();
        self.graph
            .diagnostics(qref)
            .iter()
            .map(|refusal| refusal_to_lsp(refusal, interner))
            .collect()
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
        let (Some(source), Some(document)) = (self.doc_sources.get(&id), self.documents.get(&id))
        else {
            return vec![];
        };
        let Some(site) = CompletionSite::at(source, cursor, document.mode) else {
            return vec![];
        };
        let interner = self.graph.interner();
        let mut items = match site.after {
            Sigil::None => {
                let mut items = self.function_items(None);
                items.extend(keyword_items());
                if let Some(ProbeProduct::Value { scope, .. }) = self.probe(document, source, &site)
                {
                    items.extend(scope.iter().map(|visible| CompletionItem {
                        label: interner.resolve(visible.name).to_string(),
                        kind: CompletionKind::Local,
                        detail: visible.ty.display(interner).to_string(),
                        insert_text: interner.resolve(visible.name).to_string(),
                    }));
                }
                items
            }
            Sigil::Input => match self.probe(document, source, &site) {
                Some(ProbeProduct::Value { inputs, .. }) => inputs
                    .iter()
                    .map(|input| CompletionItem {
                        label: format!("${}", interner.resolve(input.name)),
                        kind: CompletionKind::Param,
                        detail: input.ty.display(interner).to_string(),
                        insert_text: interner.resolve(input.name).to_string(),
                    })
                    .collect(),
                Some(ProbeProduct::Member { .. }) | None => vec![],
            },
            Sigil::Context => self
                .graph
                .visible_contexts()
                .map(|context| CompletionItem {
                    label: format!("@{}", interner.resolve(context.qref.name)),
                    kind: CompletionKind::Context,
                    detail: context.ty.display(interner).to_string(),
                    insert_text: interner.resolve(context.qref.name).to_string(),
                })
                .collect(),
            Sigil::Qualifier(namespace) => self.function_items(Some(namespace)),
            Sigil::Member => match self.probe(document, source, &site) {
                Some(ProbeProduct::Member {
                    fields, methods, ..
                }) => {
                    let fields = fields.iter().map(|field| CompletionItem {
                        label: interner.resolve(field.name).to_string(),
                        kind: CompletionKind::Field,
                        detail: field.ty.display(interner).to_string(),
                        insert_text: interner.resolve(field.name).to_string(),
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
                        }
                    });
                    fields.chain(methods).collect()
                }
                Some(ProbeProduct::Value { .. }) | None => vec![],
            },
        };
        items.retain(|item| item.insert_text.starts_with(site.typed));
        items.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.label.cmp(&b.label)));
        items
    }

    /// The functions a bare name reaches, one item per name, or the ones
    /// `namespace::` qualifies.
    fn function_items(&self, namespace: Option<&str>) -> Vec<CompletionItem> {
        let interner = self.graph.interner();
        let mut by_name: BTreeMap<&str, Vec<&PolyTy>> = BTreeMap::new();
        for function in self.graph.functions() {
            let qualified_as = function.qref.namespace.map(|ns| interner.resolve(ns));
            if namespace.is_some() && qualified_as != namespace {
                continue;
            }
            by_name
                .entry(interner.resolve(function.qref.name))
                .or_default()
                .push(&function.ty);
        }
        by_name
            .into_iter()
            .map(|(name, types)| CompletionItem {
                label: name.to_string(),
                kind: CompletionKind::Function,
                detail: shown_types(&types, interner),
                insert_text: name.to_string(),
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
        let ast = document.mode.parse(interner, &probed).ok()?;
        let marked: AstId = match &ast {
            ParsedAst::Script(script) => Nodes::of_script(script),
            ParsedAst::Template(template) => Nodes::of_template(template),
        }
        .at(site.word.start)
        .first()?
        .id;
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

    /// Where the name at `offset` is defined: the binder of a local name,
    /// or the function a call settled on. Only the innermost node at
    /// `offset` is read, so an offset inside a call's argument does not
    /// answer with the call's callee.
    pub fn definition(&self, id: DocId, offset: usize) -> Option<Definition> {
        let (nodes, view) = self.checked_body(id)?;
        let node = *nodes.at(offset).first()?;
        let keys = std::iter::once(node.id).chain(node.callee_id);
        if let Some(binder) = keys.clone().find_map(|key| view.binder_of.get(&key)) {
            let span = nodes
                .spans()
                .get(binder)
                .copied()
                .expect("a binder is a node of the body its uses are in");
            return Some(Definition::Local {
                span: (span.start, span.end),
            });
        }
        let qref = *keys
            .into_iter()
            .find_map(|key| view.declaration_of.get(&key))?;
        match self.graph.function(qref)?.kind {
            FnKind::Local(_) => Some(Definition::Function(qref)),
            FnKind::Extern { .. } => None,
        }
    }

    fn checked_body(&self, id: DocId) -> Option<(Nodes, Freeze<BodyView>)> {
        let qref = self.function_ref(id)?;
        let FnKind::Local(ast) = &self.graph.function(qref)?.kind else {
            return None;
        };
        let nodes = match ast {
            ParsedAst::Script(script) => Nodes::of_script(script),
            ParsedAst::Template(template) => Nodes::of_template(template),
        };
        Some((nodes, self.graph.view(qref)?))
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
            .ok()?
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

fn keyword_items() -> impl Iterator<Item = CompletionItem> {
    KEYWORDS.iter().map(|keyword| CompletionItem {
        label: keyword.to_string(),
        kind: CompletionKind::Keyword,
        detail: "keyword".to_string(),
        insert_text: keyword.to_string(),
    })
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
