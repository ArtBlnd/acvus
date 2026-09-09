//! Session - the orchestration runtime.
//!
//! Manages namespace specs, incremental compilation, and turn execution.
//! Stateless with respect to conversation history - caller provides
//! journal entries for each operation.
//!
//! Uses `IncrementalGraph` from acvus-mir for incremental compilation.
//! Spec changes (add/update item) are lowered to graph mutations,
//! and only affected functions are recompiled.

use acvus_mir::graph::incremental::IncrementalGraph;
use acvus_mir::graph::{Function, QualifiedRef};
use acvus_utils::Interner;
use rustc_hash::FxHashMap;

use crate::lower::{self, FieldError, SpanMap};
use crate::spec::Namespace;

// -- Turn result ----------------------------------------------------

/// Result of executing a single turn.
pub struct TurnResult {
    // TODO: response value, tool calls, etc.
}

// -- Session --------------------------------------------------------

pub struct Session {
    interner: Interner,
    /// Incremental compilation graph.
    graph: IncrementalGraph,
    /// Spec-level field errors from lowering (not from typeck).
    field_errors: Vec<FieldError>,
    /// Span mapping for type error -> spec field resolution.
    span_map: SpanMap,
    /// Mapping from spec item name -> QualifiedRef(s) in the graph.
    item_functions: FxHashMap<String, Vec<QualifiedRef>>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
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

    // -- Spec management --------------------------------------------

    /// Add a namespace spec. Lowers all items and registers them in the graph.
    pub fn add_namespace(&mut self, ns: &Namespace) {
        let lowered = lower::lower_namespace(&self.interner, ns, &[]);

        // Track field errors and span map from lowering.
        self.field_errors.extend(lowered.field_errors);
        self.span_map.entries.extend(lowered.span_map.entries);

        // Register each function in the incremental graph.
        for func in lowered.graph.functions.iter() {
            let name = self.interner.resolve(func.qref.name).to_string();
            let qref = func.qref;
            self.graph.add_function(func.clone());
            self.item_functions.entry(name).or_default().push(qref);
        }

        // Register contexts.
        for ctx in lowered.graph.contexts.iter() {
            self.graph.add_context(ctx.clone());
        }
    }

    /// Register externally provided functions (config-bound LLM callables, tools, etc.).
    pub fn register_extern(&mut self, fns: Vec<Function>) {
        for func in fns {
            let name = self.interner.resolve(func.qref.name).to_string();
            let qref = func.qref;
            self.graph.add_function(func);
            self.item_functions.entry(name).or_default().push(qref);
        }
    }

    // -- Error queries ----------------------------------------------

    /// Get field-level parse errors from lowering.
    pub fn field_errors(&self) -> &[FieldError] {
        &self.field_errors
    }

    /// Get the span map for type error -> spec field resolution.
    pub fn span_map(&self) -> &SpanMap {
        &self.span_map
    }

    /// Get type-checking diagnostics for a specific function.
    pub fn diagnostics(&self, qref: QualifiedRef) -> &[acvus_mir::error::MirError] {
        self.graph.diagnostics(qref)
    }

    /// Get all diagnostics across all functions.
    pub fn all_diagnostics(
        &self,
    ) -> impl Iterator<Item = (QualifiedRef, &[acvus_mir::error::MirError])> {
        self.graph.all_diagnostics()
    }

    /// Whether any function has errors (field errors or type errors).
    pub fn has_errors(&self) -> bool {
        !self.field_errors.is_empty()
            || self
                .graph
                .all_diagnostics()
                .any(|(_, errs)| !errs.is_empty())
    }

    /// Look up function ID by name.
    pub fn function_id(&self, name: &str) -> Option<QualifiedRef> {
        self.item_functions.get(name)?.first().copied()
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
            defaults: vec![],
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
            defaults: vec![],
            name: "test".into(),
            items: vec![Item::Llm(LlmSpec {
                name: "chat".into(),
                provider: Provider::Google(GoogleSpec {
                    endpoint: "e".into(),
                    api_key: "k".into(),
                    model: "m".into(),
                    temperature: None,
                    top_p: None,
                    top_k: None,
                    max_tokens: None,
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
            defaults: vec![],
            name: "ns1".into(),
            items: vec![Item::Block(Block {
                name: "a".into(),
                source: "1".into(),
                mode: BlockMode::Script,
            })],
        });
        session.add_namespace(&Namespace {
            defaults: vec![],
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
            defaults: vec![],
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
