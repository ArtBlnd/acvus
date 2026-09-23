//! LSP session - thin wrapper over `IncrementalGraph`.
//!
//! Each document maps to a `Function` in the graph.
//! Namespace scoping, caching, and incremental recompilation are all
//! handled by `IncrementalGraph`. This layer only provides:
//! - DocId <-> FunctionId mapping
//! - MirError -> LspError conversion
//! - Completion logic (context, pipe, keyword)

use acvus_ast::report::Label;
use acvus_mir::error::Refusal;
use acvus_mir::graph::ContextInfo;
use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::types::*;
use acvus_mir::ty::PolyTy;
use acvus_utils::{Astr, Interner};
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
    pub(crate) fn parse(self, interner: &Interner, source: &str) -> Result<ParsedAst, acvus_ast::ParseError> {
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

#[derive(Debug, Clone)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: String,
    pub insert_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionKind {
    Context,
    Function,
    Keyword,
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

    /// Completions at cursor position.
    pub fn completions(&self, id: DocId, cursor: usize) -> Vec<CompletionItem> {
        let Some(source) = self.doc_sources.get(&id) else {
            return vec![];
        };
        let before = &source[..cursor.min(source.len())];
        let interner = self.graph.interner();
        let ns = self.function_ref(id).and_then(|qref| qref.namespace);

        match detect_trigger(before) {
            Trigger::Context { prefix } => self.context_completions(ns, &prefix, interner),
            Trigger::Pipe => self.pipe_completions(interner),
            Trigger::Keyword { prefix } => keyword_completions(&prefix),
            Trigger::None => vec![],
        }
    }

    // -- Completion helpers ------------------------------------------

    fn context_completions(
        &self,
        ns: Option<Astr>,
        prefix: &str,
        interner: &Interner,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        for (ctx_ns, name, ctx) in self.graph.visible_contexts(ns) {
            let name_str = interner.resolve(name);
            let label = match ctx_ns {
                None => format!("@{name_str}"),
                Some(ns_name) => {
                    let ns_str = interner.resolve(ns_name);
                    format!("@{ns_str}:{name_str}")
                }
            };
            if !label[1..].starts_with(prefix) {
                continue;
            }
            let ty = format!("{:?}", ctx.ty);
            items.push(CompletionItem {
                label: label.clone(),
                kind: CompletionKind::Context,
                detail: ty,
                insert_text: label[1..].to_string(), // strip @
            });
        }
        items.sort_by(|a, b| a.label.cmp(&b.label));
        items
    }

    fn pipe_completions(&self, interner: &Interner) -> Vec<CompletionItem> {
        // All root functions as pipe candidates.
        let mut items = Vec::new();
        for (_, name, _func) in self.graph.visible_functions(None) {
            let name_str = interner.resolve(name);
            items.push(CompletionItem {
                label: name_str.to_string(),
                kind: CompletionKind::Function,
                detail: String::new(),
                insert_text: format!(" {name_str}"),
            });
        }
        items.sort_by(|a, b| a.label.cmp(&b.label));
        items
    }
}

// -- Trigger detection -----------------------------------------------

enum Trigger {
    Context { prefix: String },
    Pipe,
    Keyword { prefix: String },
    None,
}

fn detect_trigger(before: &str) -> Trigger {
    let trimmed = before.trim_end();
    if trimmed.is_empty() {
        return Trigger::None;
    }
    if trimmed.ends_with('|') {
        return Trigger::Pipe;
    }
    // @prefix or @ns:prefix
    if let Some(at_pos) = before.rfind('@') {
        let after_at = &before[at_pos + 1..];
        if after_at
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == ':')
        {
            return Trigger::Context {
                prefix: after_at.to_string(),
            };
        }
    }
    let last_word = before
        .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or("");
    if !last_word.is_empty() {
        return Trigger::Keyword {
            prefix: last_word.to_string(),
        };
    }
    Trigger::None
}

fn keyword_completions(prefix: &str) -> Vec<CompletionItem> {
    let keywords = [
        "true", "false", "in", "Some", "None", "let", "if", "else", "for", "while",
    ];
    keywords
        .iter()
        .filter(|kw| kw.starts_with(prefix) && **kw != prefix)
        .map(|kw| CompletionItem {
            label: kw.to_string(),
            kind: CompletionKind::Keyword,
            detail: "keyword".to_string(),
            insert_text: kw.to_string(),
        })
        .collect()
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
