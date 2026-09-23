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
use acvus_mir::ty::{PolyBuilder, PolyTy, TyTerm};
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

#[derive(Debug, Clone)]
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
    doc_to_fn: FxHashMap<DocId, QualifiedRef>,
    fn_to_doc: FxHashMap<QualifiedRef, DocId>,
    doc_sources: FxHashMap<DocId, String>,
    doc_parse_errors: FxHashMap<DocId, acvus_ast::ParseError>,
    next_doc_id: u32,
}

impl LspSession {
    pub fn new(interner: &Interner) -> Self {
        Self {
            graph: IncrementalGraph::new(interner),
            doc_to_fn: FxHashMap::default(),
            fn_to_doc: FxHashMap::default(),
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

    pub fn graph_mut(&mut self) -> &mut IncrementalGraph {
        &mut self.graph
    }

    // -- Namespace management (delegate) -----------------------------

    pub fn add_namespace(&mut self, name: &str) -> Astr {
        self.graph.interner().intern(name)
    }

    pub fn remove_namespace(&mut self, ns_name: Astr) {
        // Remove docs bound to functions in this namespace.
        let fn_qrefs: Vec<QualifiedRef> = self
            .fn_to_doc
            .keys()
            .filter(|qref| {
                self.graph
                    .function(**qref)
                    .is_some_and(|f| f.qref.namespace == Some(ns_name))
            })
            .copied()
            .collect();
        for qref in fn_qrefs {
            if let Some(doc_id) = self.fn_to_doc.remove(&qref) {
                self.doc_to_fn.remove(&doc_id);
            }
        }
        self.graph.remove_namespace(ns_name);
    }

    // -- Context management (delegate) -------------------------------

    pub fn add_context(&mut self, name: &str, namespace: Option<Astr>, ty: PolyTy) -> QualifiedRef {
        let interned = self.graph.interner().intern(name);
        let qref = match namespace {
            Some(ns) => QualifiedRef::qualified(ns, interned),
            None => QualifiedRef::root(interned),
        };
        self.graph.add_context(Context { qref, ty });
        qref
    }

    pub fn remove_context(&mut self, qref: QualifiedRef) {
        self.graph.remove_context(qref);
    }

    // -- Document lifecycle ------------------------------------------

    /// A source that does not parse is registered all the same, with the
    /// parse error as its only diagnostic, no function in the graph and no
    /// inputs, until an update parses.
    pub fn open(&mut self, name: &str, source: &str, namespace: Option<Astr>) -> DocId {
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;

        let interner = self.graph.interner().clone();
        let fn_name = interner.intern(name);
        let qref = match namespace {
            Some(ns) => QualifiedRef::qualified(ns, fn_name),
            None => QualifiedRef::root(fn_name),
        };

        self.doc_to_fn.insert(doc_id, qref);
        self.fn_to_doc.insert(qref, doc_id);
        self.reparse(doc_id, qref, source);
        doc_id
    }

    /// A source that stops parsing leaves the document registered with the
    /// parse error as its only diagnostic and takes its function out of the
    /// graph; the next source that parses puts it back.
    pub fn update_source(&mut self, id: DocId, source: &str) {
        let Some(&qref) = self.doc_to_fn.get(&id) else {
            return;
        };
        self.reparse(id, qref, source);
    }

    /// Always parsed as a template: that is the document kind an editor opens.
    fn reparse(&mut self, id: DocId, qref: QualifiedRef, source: &str) {
        let interner = self.graph.interner().clone();
        self.doc_sources.insert(id, source.to_string());
        let ast = match acvus_ast::parse(&interner, source) {
            Ok(ast) => ast,
            Err(error) => {
                self.doc_parse_errors.insert(id, error);
                self.graph.remove_function(qref);
                return;
            }
        };
        self.doc_parse_errors.remove(&id);
        match self.graph.function(qref) {
            Some(_) => self.graph.update_ast(qref, ParsedAst::Template(ast)),
            None => {
                let mut pb = PolyBuilder::new();
                self.graph.add_function(Function {
                    qref,
                    kind: FnKind::Local(ParsedAst::Template(ast)),
                    ty: TyTerm::Fn {
                        params: vec![],
                        ret: Box::new(pb.fresh_ty_var()),
                        captures: vec![],
                        effect: acvus_mir::ty::Effect::OPAQUE.into(),
                    },
                });
            }
        }
    }

    /// Close a document. Removes the Function from the graph.
    pub fn close(&mut self, id: DocId) {
        if let Some(qref) = self.doc_to_fn.remove(&id) {
            self.fn_to_doc.remove(&qref);
            self.graph.remove_function(qref);
        }
        self.doc_sources.remove(&id);
        self.doc_parse_errors.remove(&id);
    }

    // -- Inputs ------------------------------------------------------

    /// The binding is the whole graph's, not this document's (RFC-0071
    /// rule 4).
    pub fn bind_input(&mut self, name: &str, value: acvus_ast::Literal) {
        let interned = self.graph.interner().intern(name);
        self.graph.bind_input(interned, value);
    }

    pub fn unbind_input(&mut self, name: &str) {
        let interned = self.graph.interner().intern(name);
        self.graph.unbind_input(interned);
    }

    /// The inputs a run starting at this document requires: its own and those
    /// of every function it calls, since one host injects the `$` names of
    /// the whole graph (RFC-0071 rule 4).
    pub fn required_inputs(&self, id: DocId) -> Vec<ContextInfo> {
        let Some(&qref) = self.doc_to_fn.get(&id) else {
            return vec![];
        };
        self.graph.required_inputs(qref)
    }

    /// Get the QualifiedRef for a document.
    pub fn function_ref(&self, id: DocId) -> Option<QualifiedRef> {
        self.doc_to_fn.get(&id).copied()
    }

    // -- Queries -----------------------------------------------------

    /// Diagnostics for a document.
    pub fn diagnostics(&self, id: DocId) -> Vec<LspError> {
        if let Some(error) = self.doc_parse_errors.get(&id) {
            return vec![parse_error_to_lsp(error)];
        }
        let Some(&qref) = self.doc_to_fn.get(&id) else {
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
        let Some(&qref) = self.doc_to_fn.get(&id) else {
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
        let ns = self.doc_to_fn.get(&id).and_then(|qref| qref.namespace);

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

fn parse_error_to_lsp(error: &acvus_ast::ParseError) -> LspError {
    LspError {
        category: LspErrorCategory::Parse,
        message: error.kind.to_string(),
        span: (error.span.start != 0 || error.span.end != 0)
            .then_some((error.span.start, error.span.end)),
        related: Vec::new(),
    }
}
