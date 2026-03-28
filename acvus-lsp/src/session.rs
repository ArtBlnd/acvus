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
use acvus_utils::Interner;
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
    doc_to_fn: FxHashMap<DocId, FunctionId>,
    fn_to_doc: FxHashMap<FunctionId, DocId>,
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

    pub fn add_namespace(&mut self, name: &str) -> NamespaceId {
        let id = NamespaceId::alloc();
        let ns = Namespace {
            id,
            name: self.graph.interner().intern(name),
        };
        self.graph.add_namespace(ns);
        id
    }

    pub fn remove_namespace(&mut self, id: NamespaceId) {
        // Remove docs bound to functions in this namespace.
        let fn_ids: Vec<FunctionId> = self
            .fn_to_doc
            .keys()
            .filter(|fid| {
                self.graph
                    .function(**fid)
                    .map_or(false, |f| f.namespace == Some(id))
            })
            .copied()
            .collect();
        for fid in fn_ids {
            if let Some(doc_id) = self.fn_to_doc.remove(&fid) {
                self.doc_to_fn.remove(&doc_id);
            }
        }
        self.graph.remove_namespace(id);
    }

    // ── Context management (delegate) ───────────────────────────────

    pub fn add_context(
        &mut self,
        name: &str,
        namespace: Option<NamespaceId>,
        constraint: Constraint,
    ) -> QualifiedRef {
        let interned = self.graph.interner().intern(name);
        let qref = QualifiedRef::root(interned);
        self.graph.add_context(Context {
            name: interned,
            namespace,
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
        kind: SourceKind,
        namespace: Option<NamespaceId>,
    ) -> DocId {
        let doc_id = DocId(self.next_doc_id);
        self.next_doc_id += 1;

        let interner = self.graph.interner().clone();
        let fn_name = interner.intern(name);
        let fn_id = FunctionId::alloc();

        let func = Function {
            id: fn_id,
            name: fn_name,
            namespace,
            kind: FnKind::Local(SourceCode {
                name: fn_name,
                source: interner.intern(source),
                kind,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        };

        self.graph.add_function(func);
        self.doc_to_fn.insert(doc_id, fn_id);
        self.fn_to_doc.insert(fn_id, doc_id);
        doc_id
    }

    /// Update a document's source.
    pub fn update_source(&mut self, id: DocId, source: &str) {
        let Some(&fn_id) = self.doc_to_fn.get(&id) else {
            return;
        };
        let interned = self.graph.interner().intern(source);
        self.graph.update_source(fn_id, interned);
    }

    /// Close a document. Removes the Function from the graph.
    pub fn close(&mut self, id: DocId) {
        if let Some(fn_id) = self.doc_to_fn.remove(&id) {
            self.fn_to_doc.remove(&fn_id);
            self.graph.remove_function(fn_id);
        }
    }

    /// Get the FunctionId for a document.
    pub fn function_id(&self, id: DocId) -> Option<FunctionId> {
        self.doc_to_fn.get(&id).copied()
    }

    // ── Queries ─────────────────────────────────────────────────────

    /// Diagnostics for a document.
    pub fn diagnostics(&self, id: DocId) -> Vec<LspError> {
        let Some(&fn_id) = self.doc_to_fn.get(&id) else {
            return vec![];
        };
        let interner = self.graph.interner();
        self.graph
            .diagnostics(fn_id)
            .iter()
            .map(|e| mir_error_to_lsp(e, interner))
            .collect()
    }

    /// Context/param info for a document.
    pub fn context_info(&self, id: DocId) -> Vec<ContextInfo> {
        let Some(&fn_id) = self.doc_to_fn.get(&id) else {
            return vec![];
        };
        self.graph.context_info(fn_id)
    }

    /// Completions at cursor position.
    pub fn completions(&self, id: DocId, cursor: usize) -> Vec<CompletionItem> {
        let Some(&fn_id) = self.doc_to_fn.get(&id) else {
            return vec![];
        };
        let Some(func) = self.graph.function(fn_id) else {
            return vec![];
        };
        let source = match &func.kind {
            FnKind::Local(src) => self.graph.interner().resolve(src.source),
            FnKind::Extern => return vec![],
        };

        if cursor == 0 || cursor > source.len() || !source.is_char_boundary(cursor) {
            return vec![];
        }

        let before = &source[..cursor];
        let ns = func.namespace;
        let interner = self.graph.interner();

        match detect_trigger(before) {
            Trigger::Context { prefix } => self.context_completions(ns, &prefix, interner),
            Trigger::Pipe => self.pipe_completions(interner),
            Trigger::Keyword { prefix } => keyword_completions(&prefix),
            Trigger::None => vec![],
        }
    }

    // ── Completion helpers ──────────────────────────────────────────

    fn context_completions(
        &self,
        ns: Option<NamespaceId>,
        prefix: &str,
        interner: &Interner,
    ) -> Vec<CompletionItem> {
        let mut items = Vec::new();
        for (ctx_ns, name, ctx) in self.graph.visible_contexts(ns) {
            let name_str = interner.resolve(name);
            let label = match ctx_ns {
                None => format!("@{name_str}"),
                Some(ns_id) => {
                    let ns_name = self
                        .graph
                        .namespace(ns_id)
                        .map(|n| interner.resolve(n.name))
                        .unwrap_or("?");
                    format!("@{ns_name}:{name_str}")
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
