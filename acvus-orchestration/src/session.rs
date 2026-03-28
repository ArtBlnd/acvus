//! Session — the orchestration runtime.
//!
//! Manages namespace specs, incremental compilation, and turn execution.
//! Stateless with respect to conversation history — caller provides
//! journal entries for each operation.
//!
//! Uses `IncrementalGraph` from acvus-mir for incremental compilation.
//! Spec changes (add/update item) are lowered to graph mutations,
//! and only affected functions are recompiled.

use std::ops::Range;

use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::{Function, FunctionId};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::lower::{self, FieldError, SpanMap};
use crate::spec::{Item, Namespace};

// ── Turn result ────────────────────────────────────────────────────

/// Result of executing a single turn.
pub struct TurnResult {
    // TODO: response value, tool calls, etc.
}

// ── Session ────────────────────────────────────────────────────────

pub struct Session {
    interner: Interner,
    /// Incremental compilation graph.
    graph: IncrementalGraph,
    /// Spec-level field errors from lowering (not from typeck).
    field_errors: Vec<FieldError>,
    /// Span mapping for type error → spec field resolution.
    span_map: SpanMap,
    /// Mapping from spec item name → FunctionId(s) in the graph.
    item_functions: FxHashMap<String, Vec<FunctionId>>,
}

impl Session {
    pub fn new() -> Self {
        let interner = Interner::new();
        let graph = IncrementalGraph::new(&interner);
        Self {
            interner,
            graph,
            field_errors: Vec::new(),
            span_map: SpanMap::default(),
            item_functions: FxHashMap::default(),
        }
    }

    pub fn with_interner(interner: Interner) -> Self {
        let graph = IncrementalGraph::new(&interner);
        Self {
            interner,
            graph,
            field_errors: Vec::new(),
            span_map: SpanMap::default(),
            item_functions: FxHashMap::default(),
        }
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    // ── Spec management ────────────────────────────────────────────

    /// Add a namespace spec. Lowers all items and registers them in the graph.
    pub fn add_namespace(&mut self, ns: &Namespace) {
        let lowered = lower::lower_namespace(&self.interner, ns, &[]);

        // Track field errors and span map from lowering.
        self.field_errors.extend(lowered.field_errors);
        self.span_map.entries.extend(lowered.span_map.entries);

        // Register each function in the incremental graph.
        for func in lowered.graph.functions.iter() {
            let name = self.interner.resolve(func.name).to_string();
            let fid = func.id;
            self.graph.add_function(func.clone());
            self.item_functions.entry(name).or_default().push(fid);
        }

        // Register contexts.
        for ctx in lowered.graph.contexts.iter() {
            self.graph.add_context(ctx.clone());
        }
    }

    /// Register externally provided functions (config-bound LLM callables, tools, etc.).
    pub fn register_extern(&mut self, fns: Vec<Function>) {
        for func in fns {
            let name = self.interner.resolve(func.name).to_string();
            let fid = func.id;
            self.graph.add_function(func);
            self.item_functions.entry(name).or_default().push(fid);
        }
    }

    // ── Error queries ──────────────────────────────────────────────

    /// Get field-level parse errors from lowering.
    pub fn field_errors(&self) -> &[FieldError] {
        &self.field_errors
    }

    /// Get the span map for type error → spec field resolution.
    pub fn span_map(&self) -> &SpanMap {
        &self.span_map
    }

    /// Get type-checking diagnostics for a specific function.
    pub fn diagnostics(&self, id: FunctionId) -> &[acvus_mir::error::MirError] {
        self.graph.diagnostics(id)
    }

    /// Get all diagnostics across all functions.
    pub fn all_diagnostics(&self) -> impl Iterator<Item = (FunctionId, &[acvus_mir::error::MirError])> {
        self.graph.all_diagnostics()
    }

    /// Whether any function has errors (field errors or type errors).
    pub fn has_errors(&self) -> bool {
        !self.field_errors.is_empty()
            || self.graph.all_diagnostics().any(|(_, errs)| !errs.is_empty())
    }

    /// Look up function ID by name.
    pub fn function_id(&self, name: &str) -> Option<FunctionId> {
        self.item_functions.get(name)?.first().copied()
    }

    // ── Execution ──────────────────────────────────────────────────

    /// Execute a single turn.
    ///
    /// `entry`: mutable journal entry for context read/write.
    pub async fn execute_turn(
        &self,
        _entry: &mut dyn acvus_interpreter::journal::EntryMut,
    ) -> TurnResult {
        // TODO: find entrypoint, create interpreter, execute
        todo!("execute_turn")
    }

    /// Render past turns from history.
    ///
    /// `entry`: read-only journal entry for context access.
    /// `range`: which history items to render (start..end).
    pub async fn display_history(
        &self,
        _entry: &dyn acvus_interpreter::journal::EntryRef,
        _range: Range<usize>,
    ) -> Vec<String> {
        // TODO: get history display function, execute with range
        todo!("display_history")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::*;

    #[test]
    fn session_compiles_incrementally() {
        let mut session = Session::new();
        session.add_namespace(&Namespace {
            name: "test".into(),
            items: vec![Item::Block(Block {
                name: "hello".into(),
                source: "hello world".into(),
                mode: BlockMode::Template,
            })],
        });

        assert!(!session.has_errors());
        assert!(session.function_id("hello").is_some());
    }

    #[test]
    fn session_detects_field_errors() {
        let mut session = Session::new();
        session.add_namespace(&Namespace {
            name: "test".into(),
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(),
                    api_key: "k".into(),
                    model: "m".into(),
                    temperature: None, top_p: None, top_k: None, max_tokens: None,
                    system: None,
                    messages: vec![GoogleMessage {
                        role: GoogleRole::User,
                        content: Content::Inline("{{bad".into()),
                    }],
                }),
            })],
        });

        assert!(session.has_errors());
        assert!(!session.field_errors().is_empty());
    }

    #[test]
    fn session_multiple_namespaces() {
        let mut session = Session::new();
        session.add_namespace(&Namespace {
            name: "ns1".into(),
            items: vec![Item::Block(Block {
                name: "a".into(),
                source: "1".into(),
                mode: BlockMode::Script,
            })],
        });
        session.add_namespace(&Namespace {
            name: "ns2".into(),
            items: vec![Item::Block(Block {
                name: "b".into(),
                source: "2".into(),
                mode: BlockMode::Script,
            })],
        });

        assert!(!session.has_errors());
        assert!(session.function_id("a").is_some());
        assert!(session.function_id("b").is_some());
    }

    #[test]
    fn session_diagnostics_for_function() {
        let mut session = Session::new();
        session.add_namespace(&Namespace {
            name: "test".into(),
            items: vec![Item::Block(Block {
                name: "good".into(),
                source: "42".into(),
                mode: BlockMode::Script,
            })],
        });

        let fid = session.function_id("good").unwrap();
        let diags = session.diagnostics(fid);
        assert!(diags.is_empty(), "valid block should have no diagnostics");
    }
}
