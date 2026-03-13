use std::collections::VecDeque;

use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::compile::CompiledNode;
use crate::error::{OrchError, OrchErrorKind};

/// Dependency graph for orchestration nodes.
#[derive(Debug, Clone)]
pub struct Dag {
    pub name_to_idx: FxHashMap<Astr, usize>,
    pub deps: Vec<FxHashSet<usize>>,
    pub rdeps: Vec<FxHashSet<usize>>,
    pub topo_order: Vec<usize>,
}

/// Build a DAG from compiled nodes.
///
/// Dependencies are inferred from context keys: if node A references a context
/// key that matches node B's name, then A depends on B.
/// External keys (not produced by any node) are allowed.
///
/// Uses Kahn's algorithm for topological sort + cycle detection.
pub fn build_dag(interner: &Interner, nodes: &[CompiledNode]) -> Result<Dag, Vec<OrchError>> {
    let name_to_idx: FxHashMap<Astr, usize> =
        nodes.iter().enumerate().map(|(i, n)| (n.name, i)).collect();

    let n = nodes.len();
    let mut deps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
    let mut rdeps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

    for (i, node) in nodes.iter().enumerate() {
        for key in &node.all_context_keys {
            if let Some(&j) = name_to_idx.get(key) {
                deps[i].insert(j);
                rdeps[j].insert(i);
            }
        }
    }

    // Kahn's algorithm
    let mut in_degree: Vec<usize> = deps.iter().map(|d| d.len()).collect();
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut topo_order = Vec::new();

    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    while let Some(u) = queue.pop_front() {
        topo_order.push(u);
        for &v in &rdeps[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if topo_order.len() != n {
        let in_cycle: Vec<String> = (0..n)
            .filter(|i| in_degree[*i] > 0)
            .map(|i| interner.resolve(nodes[i].name).to_string())
            .collect();
        return Err(vec![OrchError::new(OrchErrorKind::CycleDetected {
            nodes: in_cycle,
        })]);
    }

    Ok(Dag {
        name_to_idx,
        deps,
        rdeps,
        topo_order,
    })
}

#[cfg(test)]
mod tests {
    use crate::compile::{CompiledExecution, CompiledStrategy};
    use crate::{ApiKind, CompiledLlm, CompiledPersistency, CompiledNodeKind};
    use acvus_utils::Interner;

    use super::*;

    fn make_node(interner: &Interner, name: &str, context_keys: Vec<&str>) -> CompiledNode {
        CompiledNode {
            name: interner.intern(name),
            kind: CompiledNodeKind::Llm(CompiledLlm {
                api: ApiKind::OpenAI,
                provider: "test".into(),
                model: "m".into(),
                messages: vec![],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
                max_tokens: Default::default(),
            }),
            all_context_keys: context_keys
                .into_iter()
                .map(|k| interner.intern(k))
                .collect(),
            strategy: CompiledStrategy {
                execution: CompiledExecution::Always,
                persistency: CompiledPersistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        }
    }

    #[test]
    fn linear_dag() {
        let interner = Interner::new();
        // A -> B -> C
        let nodes = vec![
            make_node(&interner, "A", vec![]),
            make_node(&interner, "B", vec!["A"]),
            make_node(&interner, "C", vec!["B"]),
        ];
        let dag = build_dag(&interner, &nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 3);
        let pos_a = dag.topo_order.iter().position(|&i| i == 0).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == 1).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn diamond_dag() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", vec![]),
            make_node(&interner, "B", vec!["A"]),
            make_node(&interner, "C", vec!["A"]),
            make_node(&interner, "D", vec!["B", "C"]),
        ];
        let dag = build_dag(&interner, &nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 4);
        let pos_a = dag.topo_order.iter().position(|&i| i == 0).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == 1).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == 2).unwrap();
        let pos_d = dag.topo_order.iter().position(|&i| i == 3).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn cycle_detected() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", vec!["B"]),
            make_node(&interner, "B", vec!["A"]),
        ];
        let err = build_dag(&interner, &nodes).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::CycleDetected { .. }));
    }

    #[test]
    fn no_deps() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", vec![]),
            make_node(&interner, "B", vec![]),
        ];
        let dag = build_dag(&interner, &nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
    }

    #[test]
    fn external_key_ignored() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", vec![]),
            make_node(&interner, "B", vec!["ext"]),
        ];
        let dag = build_dag(&interner, &nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
        assert!(dag.deps[1].is_empty());
    }

    #[test]
    fn self_reference_detected() {
        let interner = Interner::new();
        let nodes = vec![
            make_node(&interner, "A", vec!["A"]),
        ];
        let err = build_dag(&interner, &nodes).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::CycleDetected { .. }));
    }
}
