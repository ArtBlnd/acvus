use std::collections::VecDeque;

use acvus_utils::Interner;
use rustc_hash::FxHashSet;

use crate::compile::{CompiledNodeGraph, NodeId};
use crate::error::{OrchError, OrchErrorKind};

/// Dependency graph for orchestration nodes.
#[derive(Debug, Clone)]
pub struct Dag {
    pub deps: Vec<FxHashSet<NodeId>>,
    pub rdeps: Vec<FxHashSet<NodeId>>,
    pub topo_order: Vec<NodeId>,
}

/// Build a DAG from a compiled node graph.
///
/// Dependencies are inferred from context keys: if node A references a context
/// key that matches a primary name mapping to node B, then A depends on B.
/// External keys (not produced by any node) are allowed.
/// Bind→body edges are NOT added (lazy, not a DAG dependency).
///
/// Uses Kahn's algorithm for topological sort + cycle detection.
pub fn build_dag(interner: &Interner, graph: &CompiledNodeGraph) -> Result<Dag, Vec<OrchError>> {
    let n = graph.nodes.len();
    let mut deps: Vec<FxHashSet<NodeId>> = vec![FxHashSet::default(); n];
    let mut rdeps: Vec<FxHashSet<NodeId>> = vec![FxHashSet::default(); n];

    for node in &graph.nodes {
        let i = node.id;
        for key in &node.all_context_keys {
            if let Some(&dep_id) = graph.name_to_primary.get(key) {
                if dep_id == i {
                    // Skip self-references
                    continue;
                }
                deps[i.index()].insert(dep_id);
                rdeps[dep_id.index()].insert(i);
            }
        }
    }

    // Kahn's algorithm
    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    let mut topo_order = Vec::new();

    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(NodeId(i));
        }
    }

    while let Some(u) = queue.pop_front() {
        topo_order.push(u);
        for &v in &rdeps[u.index()] {
            in_degree[v.index()] -= 1;
            if in_degree[v.index()] == 0 {
                queue.push_back(v);
            }
        }
    }

    if topo_order.len() != n {
        let in_cycle: Vec<String> = (0..n)
            .filter(|i| in_degree[*i] > 0)
            .map(|i| interner.resolve(graph.nodes[i].name).to_string())
            .collect();
        return Err(vec![OrchError::new(OrchErrorKind::CycleDetected {
            nodes: in_cycle,
        })]);
    }

    Ok(Dag {
        deps,
        rdeps,
        topo_order,
    })
}

#[cfg(test)]
mod tests {
    use acvus_mir::ty::Ty;
    use crate::compile::{CompiledExecution, CompiledStrategy, CompiledNode, CompiledNodeGraph, NodeId};
    use crate::{CompiledOpenAICompatible, CompiledNodeKind, MaxTokens};
    use acvus_utils::{Astr, Interner};
    use rustc_hash::FxHashMap;

    use super::*;

    fn make_node(interner: &Interner, name: &str, id: usize, context_keys: Vec<&str>) -> CompiledNode {
        CompiledNode {
            id: NodeId(id),
            name: interner.intern(name),
            kind: CompiledNodeKind::OpenAICompatible(CompiledOpenAICompatible {
                endpoint: String::new(),
                api_key: String::new(),
                model: "m".into(),
                messages: vec![],
                tools: vec![],
                temperature: None,
                top_p: None,
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            all_context_keys: context_keys
                .into_iter()
                .map(|k| interner.intern(k))
                .collect(),
            strategy: CompiledStrategy {
                execution: CompiledExecution::Always,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
            output_ty: Ty::String,
            role: crate::compile::NodeRole::Standalone,
        }
    }

    fn make_graph(nodes: Vec<CompiledNode>) -> CompiledNodeGraph {
        let name_to_primary: FxHashMap<Astr, NodeId> = nodes
            .iter()
            .map(|n| (n.name, n.id))
            .collect();
        CompiledNodeGraph {
            nodes,
            name_to_primary,
        }
    }

    #[test]
    fn linear_dag() {
        let interner = Interner::new();
        // A -> B -> C
        let nodes = vec![
            make_node(&interner, "A", 0, vec![]),
            make_node(&interner, "B", 1, vec!["A"]),
            make_node(&interner, "C", 2, vec!["B"]),
        ];
        let graph = make_graph(nodes);
        let dag = build_dag(&interner, &graph).unwrap();
        assert_eq!(dag.topo_order.len(), 3);
        let pos_a = dag.topo_order.iter().position(|&i| i == NodeId(0)).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == NodeId(1)).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == NodeId(2)).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn diamond_dag() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", 0, vec![]),
            make_node(&interner, "B", 1, vec!["A"]),
            make_node(&interner, "C", 2, vec!["A"]),
            make_node(&interner, "D", 3, vec!["B", "C"]),
        ];
        let graph = make_graph(nodes);
        let dag = build_dag(&interner, &graph).unwrap();
        assert_eq!(dag.topo_order.len(), 4);
        let pos_a = dag.topo_order.iter().position(|&i| i == NodeId(0)).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == NodeId(1)).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == NodeId(2)).unwrap();
        let pos_d = dag.topo_order.iter().position(|&i| i == NodeId(3)).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn cycle_detected() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", 0, vec!["B"]),
            make_node(&interner, "B", 1, vec!["A"]),
        ];
        let graph = make_graph(nodes);
        let err = build_dag(&interner, &graph).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::CycleDetected { .. }));
    }

    #[test]
    fn no_deps() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", 0, vec![]),
            make_node(&interner, "B", 1, vec![]),
        ];
        let graph = make_graph(nodes);
        let dag = build_dag(&interner, &graph).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
    }

    #[test]
    fn external_key_ignored() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", 0, vec![]),
            make_node(&interner, "B", 1, vec!["ext"]),
        ];
        let graph = make_graph(nodes);
        let dag = build_dag(&interner, &graph).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
        assert!(dag.deps[NodeId(1).index()].is_empty());
    }

    #[test]
    fn self_reference_skipped() {
        let interner = Interner::new();
        // Self-reference is now skipped (not a cycle), since name_to_primary
        // maps to the node's own id.
        let nodes = vec![
            make_node(&interner, "A", 0, vec!["A"]),
        ];
        let graph = make_graph(nodes);
        let dag = build_dag(&interner, &graph).unwrap();
        assert_eq!(dag.topo_order.len(), 1);
        assert!(dag.deps[NodeId(0).index()].is_empty());
    }
}
