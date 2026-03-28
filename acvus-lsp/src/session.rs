//! LSP session — thin wrapper over `IncrementalGraph`.
//!
//! Each document maps to a `Function` in the graph.
//! Namespace scoping, caching, and incremental recompilation are all
//! handled by `IncrementalGraph`. This layer only provides:
//! - DocId ↔ FunctionId mapping
//! - MirError → LspError conversion
//! - Completion logic (context, pipe, keyword)

use acvus_mir::error::MirError;
use acvus_mir::graph::incremental::{ContextInfo, IncrementalGraph};
use acvus_mir::graph::types::*;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

// ── Public types ────────────────────────────────────────────────────

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

// ── LspSession ──────────────────────────────────────────────────────

pub struct LspSession {
    graph: IncrementalGraph,
    doc_to_fn: FxHashMap<DocId, QualifiedRef>,
    fn_to_doc: FxHashMap<QualifiedRef, DocId>,
    next_doc_id: u32,
}

impl LspSession {
    pub fn new(interner: &Interner) -> Self {
        Self {
            graph: IncrementalGraph::new(interner),
            doc_to_fn: FxHashMap::default(),
            fn_to_doc: FxHashMap::default(),
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

    // ── Namespace management (delegate) ─────────────────────────────

    pub fn add_namespace(&mut self, name: &str) -> Astr {
        let ns_name = self.graph.interner().intern(name);
        ns_name
    }

    pub fn remove_namespace(&mut self, ns_name: Astr) {
        // Remove docs bound to functions in this namespace.
        let fn_qrefs: Vec<QualifiedRef> = self
            .fn_to_doc
            .keys()
            .filter(|qref| {
                self.graph
                    .function(**qref)
                    .map_or(false, |f| f.qref.namespace == Some(ns_name))
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

    // ── Context management (delegate) ───────────────────────────────

    pub fn add_context(
        &mut self,
        name: &str,
        namespace: Option<Astr>,
        constraint: Constraint,
    ) -> QualifiedRef {
        let interned = self.graph.interner().intern(name);
        let qref = match namespace {
            Some(ns) => QualifiedRef::qualified(ns, interned),
            None => QualifiedRef::root(interned),
        };
        self.graph.add_context(Context {
            qref,
            constraint,
        });
        qref
    }

    pub fn remove_context(&mut self, qref: QualifiedRef) {
        self.graph.remove_context(qref);
    }

    // ── Document lifecycle ──────────────────────────────────────────

    /// Open a document. Creates a Function in the graph.
    pub fn open(
        &mut self,
        name: &str,
        source: &str,
        namespace: Option<Astr>,
    ) -> DocId {
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;

        let interner = self.graph.interner().clone();
        let fn_name = interner.intern(name);
        let qref = match namespace {
            Some(ns) => QualifiedRef::qualified(ns, fn_name),
            None => QualifiedRef::root(fn_name),
        };

        // Always parse as template for LSP (templates are the primary document type).
        let ast = acvus_ast::parse(&interner, source)
            .expect("parse error in LSP open");
        let func = Function {
            qref,
            kind: FnKind::Local(ParsedAst::Template(ast)),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        };

        self.graph.add_function(func);
        self.doc_to_fn.insert(doc_id, qref);
        self.fn_to_doc.insert(qref, doc_id);
        doc_id
    }

    /// Update a document's source.
    pub fn update_source(&mut self, id: DocId, source: &str) {
        let Some(&qref) = self.doc_to_fn.get(&id) else {
            return;
        };
        let interner = self.graph.interner().clone();
        let ast = acvus_ast::parse(&interner, source)
            .expect("parse error in LSP update_source");
        self.graph.update_ast(qref, ParsedAst::Template(ast));
    }

    /// Close a document. Removes the Function from the graph.
    pub fn close(&mut self, id: DocId) {
        if let Some(qref) = self.doc_to_fn.remove(&id) {
            self.fn_to_doc.remove(&qref);
            self.graph.remove_function(qref);
        }
    }

    /// Get the QualifiedRef for a document.
    pub fn function_ref(&self, id: DocId) -> Option<QualifiedRef> {
        self.doc_to_fn.get(&id).copied()
    }

    // ── Queries ─────────────────────────────────────────────────────

    /// Diagnostics for a document.
    pub fn diagnostics(&self, id: DocId) -> Vec<LspError> {
        let Some(&qref) = self.doc_to_fn.get(&id) else {
            return vec![];
        };
        let interner = self.graph.interner();
        self.graph
            .diagnostics(qref)
            .iter()
            .map(|e| mir_error_to_lsp(e, interner))
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
    /// Completions at cursor position.
    ///
    /// TODO: LSP should store source text separately now that FnKind::Local
    /// contains ParsedAst instead of source strings. Completions require
    /// source text which is not available from ParsedAst.
    /// Return empty until source text storage is added to LspSession.
    pub fn completions(&self, _id: DocId, _cursor: usize) -> Vec<CompletionItem> {
        vec![]
    }

    // ── Completion helpers ──────────────────────────────────────────

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
            let ty = match &ctx.constraint {
                Constraint::Exact(ty) => format!("{}", ty.display(interner)),
                _ => "inferred".to_string(),
            };
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

// ── Trigger detection ───────────────────────────────────────────────

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
    let keywords = ["true", "false", "in", "Some", "None"];
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

// ── MirError → LspError ────────────────────────────────────────────

fn mir_error_to_lsp(error: &MirError, interner: &Interner) -> LspError {
    LspError {
        category: LspErrorCategory::Type,
        message: format!("{}", error.display(interner)),
        span: {
            let s = error.span;
            if s.start != 0 || s.end != 0 {
                Some((s.start, s.end))
            } else {
                None
            }
        },
    }
}
