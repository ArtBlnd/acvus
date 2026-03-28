//! Graph clustering: compute access ordering constraints for mutable contexts.
//!
//! Analyzes the compilation graph to determine which contexts are mutable
//! and produces ordering constraints that the runtime Executor uses to
//! enforce race-free access via Reader-Writer Lock semantics.
//!
//! Soundness guarantee: if the Executor respects these constraints,
//! no two functions will have conflicting access to the same mutable context.

use rustc_hash::FxHashSet;

use super::extract::ExtractResult;
use super::types::*;

// ── Access types ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct Access {
    pub fn_id: FunctionId,
    pub kind: AccessKind,
}

// ── Cluster output ──────────────────────────────────────────────────

/// Per-context ordering constraint.
/// Functions accessing this context must respect this sequence.
#[derive(Debug, Clone)]
pub struct ContextOrdering {
    pub context_name: QualifiedRef,
    /// Ordered access sequence. The Executor must ensure:
    /// - Reads can be concurrent with other reads.
    /// - Write must be exclusive (no concurrent reads or writes).
    /// - Sequence order must be respected for writes.
    pub accesses: Vec<Access>,
}

/// Cluster analysis result.
#[derive(Debug)]
pub struct ClusterResult {
    /// Mutable contexts and their access orderings.
    pub mutable_orderings: Vec<ContextOrdering>,
    /// Set of mutable context names (convenience for quick lookup).
    pub mutable_contexts: FxHashSet<QualifiedRef>,
    /// Function dependency edges derived from context data flow.
    /// (fn_a, fn_b) means fn_a must complete before fn_b starts
    /// (because fn_a writes a context that fn_b reads).
    pub dependencies: Vec<(FunctionId, FunctionId)>,
}

// ── Clustering ──────────────────────────────────────────────────────

/// Run graph clustering analysis.
pub fn cluster(graph: &CompilationGraph, extract: &ExtractResult) -> ClusterResult {
    // 1. Determine mutable contexts: any context written by at least one function.
    let mut mutable_contexts = FxHashSet::default();
    for fn_ref in extract.fn_refs.values() {
        for r in &fn_ref.context_writes {
            mutable_contexts.insert(*r);
        }
    }

    // 2. For each mutable context, collect accesses from all functions.
    let mut orderings = Vec::new();
    let mut dependencies = Vec::new();

    for &ctx_qref in &mutable_contexts {
        let mut accesses = Vec::new();
        let mut writers: Vec<FunctionId> = Vec::new();
        let mut readers: Vec<FunctionId> = Vec::new();

        for func in graph.functions.iter() {
            let Some(fn_ref) = extract.fn_refs.get(&func.id) else {
                continue;
            };

            let is_write = fn_ref.context_writes.contains(&ctx_qref);
            let is_read = fn_ref.context_reads.contains(&ctx_qref);

            if is_write {
                // Writer also reads (read-modify-write pattern).
                accesses.push(Access {
                    fn_id: func.id,
                    kind: AccessKind::Write,
                });
                writers.push(func.id);
            } else if is_read {
                accesses.push(Access {
                    fn_id: func.id,
                    kind: AccessKind::Read,
                });
                readers.push(func.id);
            }
        }

        // 3. Generate dependency edges:
        // - All readers of the initial version must complete before writer starts.
        // - Writer must complete before readers of the next version start.
        //
        // For now, simple model: readers → writer → next readers.
        // With SSA versioning, this becomes version-aware.
        for &reader in &readers {
            for &writer in &writers {
                // Reader must see the value before writer changes it,
                // OR reader must see the value after writer changes it.
                // This is determined by the SSA version — for now, we conservatively
                // add a dependency edge: writer depends on no pre-existing reader
                // (readers get the initial value), and post-write readers depend on writer.
                //
                // Without SSA version info, we add writer→reader dependency
                // (reader sees the written value).
                dependencies.push((writer, reader));
            }
        }

        orderings.push(ContextOrdering {
            context_name: ctx_qref,
            accesses,
        });
    }

    ClusterResult {
        mutable_orderings: orderings,
        mutable_contexts,
        dependencies,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use acvus_utils::{Freeze, Interner};

    fn make_graph(
        interner: &Interner,
        units: &[(&str, &str)], // (name, source)
    ) -> CompilationGraph {
        let unit_list: Vec<Function> = units
            .iter()
            .map(|(name, source)| Function {
                id: FunctionId::alloc(),
                name: interner.intern(name),
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: interner.intern(name),
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                },
            })
            .collect();
        CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(unit_list),
            contexts: Freeze::new(vec![]),
        }
    }

    // -- Soundness: mutable contexts detected --

    #[test]
    fn cluster_detects_mutable_context() {
        let i = Interner::new();
        let graph = make_graph(&i, &[("writer", "@x = 42; @x"), ("reader", "@x + 1")]);
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        assert!(
            result
                .mutable_contexts
                .contains(&QualifiedRef::root(i.intern("x")))
        );
    }

    #[test]
    fn cluster_immutable_no_ordering() {
        let i = Interner::new();
        let graph = make_graph(&i, &[("a", "@x + 1"), ("b", "@x + 2")]);
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        assert!(result.mutable_contexts.is_empty());
        assert!(result.mutable_orderings.is_empty());
        assert!(result.dependencies.is_empty());
    }

    // -- Completeness: ordering constraints generated --

    #[test]
    fn cluster_writer_reader_dependency() {
        let i = Interner::new();
        let graph = make_graph(&i, &[("writer", "@x = 42; @x"), ("reader", "@x + 1")]);
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        // Should have a dependency: writer → reader.
        assert!(
            !result.dependencies.is_empty(),
            "should have dependency edges"
        );

        let writer_id = graph.functions[0].id;
        let reader_id = graph.functions[1].id;
        assert!(
            result
                .dependencies
                .iter()
                .any(|(from, to)| *from == writer_id && *to == reader_id),
            "writer should come before reader"
        );
    }

    #[test]
    fn cluster_mutable_ordering_has_accesses() {
        let i = Interner::new();
        let graph = make_graph(&i, &[("writer", "@x = 42; @x"), ("reader", "@x + 1")]);
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        assert_eq!(result.mutable_orderings.len(), 1);
        let ordering = &result.mutable_orderings[0];
        assert_eq!(ordering.context_name, QualifiedRef::root(i.intern("x")));
        assert_eq!(ordering.accesses.len(), 2); // writer + reader
    }

    // -- Soundness: read-only contexts produce no constraints --

    #[test]
    fn cluster_multiple_readers_no_constraints() {
        let i = Interner::new();
        let graph = make_graph(&i, &[("a", "@x + 1"), ("b", "@x + 2"), ("c", "@x + 3")]);
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        assert!(result.mutable_contexts.is_empty());
        assert!(result.dependencies.is_empty());
    }

    // -- Multiple mutable contexts --

    #[test]
    fn cluster_multiple_mutable_contexts() {
        let i = Interner::new();
        let graph = make_graph(
            &i,
            &[
                ("w1", "@x = 1; @x"),
                ("w2", "@y = 2; @y"),
                ("reader", "@x + @y"),
            ],
        );
        let ext = extract::extract(&i, &graph);
        let result = cluster(&graph, &ext);

        assert!(
            result
                .mutable_contexts
                .contains(&QualifiedRef::root(i.intern("x")))
        );
        assert!(
            result
                .mutable_contexts
                .contains(&QualifiedRef::root(i.intern("y")))
        );
        assert_eq!(result.mutable_orderings.len(), 2);
    }
}
