use std::collections::{HashMap, HashSet, VecDeque};

use crate::compile::CompiledNode;
use crate::error::{OrchError, OrchErrorKind};

/// Dependency graph for orchestration nodes.
#[derive(Debug, Clone)]
pub struct Dag {
    pub name_to_idx: HashMap<String, usize>,
    pub deps: Vec<HashSet<usize>>,
    pub rdeps: Vec<HashSet<usize>>,
    pub topo_order: Vec<usize>,
}

/// Build a DAG from compiled nodes.
///
/// Dependencies are inferred from context keys: if node A references a context
/// key that matches node B's name, then A depends on B.
/// External keys (not produced by any node) are allowed.
///
/// Uses Kahn's algorithm for topological sort + cycle detection.
pub fn build_dag(nodes: &[CompiledNode]) -> Result<Dag, Vec<OrchError>> {
    let name_to_idx: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.clone(), i))
        .collect();

    let n = nodes.len();
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut rdeps: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for (i, node) in nodes.iter().enumerate() {
        for key in &node.all_context_keys {
            if let Some(&j) = name_to_idx.get(key) {
                if j != i {
                    deps[i].insert(j);
                    rdeps[j].insert(i);
                }
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
            .map(|i| nodes[i].name.clone())
            .collect();
        return Err(vec![OrchError::new(OrchErrorKind::CycleDetected { nodes: in_cycle })]);
    }

    Ok(Dag { name_to_idx, deps, rdeps, topo_order })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(name: &str, context_keys: Vec<&str>) -> CompiledNode {
        CompiledNode {
            name: name.into(),
            kind: crate::compile::CompiledNodeKind::Llm {
                provider: "test".into(),
                model: "m".into(),
                messages: vec![],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
            },
            all_context_keys: context_keys.into_iter().map(Into::into).collect(),
            strategy: Default::default(),
            key_module: None,
        }
    }

    #[test]
    fn linear_dag() {
        // A -> B -> C
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec!["A"]),
            make_node("C", vec!["B"]),
        ];
        let dag = build_dag(&nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 3);
        let pos_a = dag.topo_order.iter().position(|&i| i == 0).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == 1).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn diamond_dag() {
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec!["A"]),
            make_node("C", vec!["A"]),
            make_node("D", vec!["B", "C"]),
        ];
        let dag = build_dag(&nodes).unwrap();
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
        let nodes = vec![
            make_node("A", vec!["B"]),
            make_node("B", vec!["A"]),
        ];
        let err = build_dag(&nodes).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::CycleDetected { .. }));
    }

    #[test]
    fn no_deps() {
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec![]),
        ];
        let dag = build_dag(&nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
    }

    #[test]
    fn external_key_ignored() {
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec!["ext"]),
        ];
        let dag = build_dag(&nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
        assert!(dag.deps[1].is_empty());
    }
}
