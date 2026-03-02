use std::collections::{HashMap, HashSet, VecDeque};

use crate::compile::CompiledNode;
use crate::error::{OrchError, OrchErrorKind};

/// Dependency graph for orchestration nodes.
#[derive(Debug)]
pub struct Dag {
    pub name_to_idx: HashMap<String, usize>,
    pub deps: Vec<HashSet<usize>>,
    pub rdeps: Vec<HashSet<usize>>,
    pub topo_order: Vec<usize>,
}

/// Build a DAG from compiled nodes.
///
/// Dependencies are inferred from input mappings: if node A's input references
/// a storage key that matches node B's name, then A depends on B.
///
/// Uses Kahn's algorithm for topological sort + cycle detection.
pub fn build_dag(nodes: &[CompiledNode]) -> Result<Dag, Vec<OrchError>> {
    let name_to_idx: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.config.name.clone(), i))
        .collect();

    let n = nodes.len();
    let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    let mut rdeps: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    for (i, node) in nodes.iter().enumerate() {
        for (_, storage_ref) in &node.config.inputs {
            let key = storage_ref.strip_prefix('@').unwrap_or(storage_ref);
            if let Some(&j) = name_to_idx.get(key) {
                if j != i {
                    deps[i].insert(j);
                    rdeps[j].insert(i);
                }
            }
            // External keys (not produced by any node) are fine — caller seeds them
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
            .map(|i| nodes[i].config.name.clone())
            .collect();
        return Err(vec![OrchError::new(OrchErrorKind::CycleDetected { nodes: in_cycle })]);
    }

    Ok(Dag { name_to_idx, deps, rdeps, topo_order })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::ConfigBlock;
    use std::collections::HashSet;

    fn make_node(name: &str, inputs: Vec<(&str, &str)>) -> CompiledNode {
        CompiledNode {
            config: ConfigBlock {
                name: name.into(),
                model: "m".into(),
                inputs: inputs.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
                tools: vec![],
            },
            blocks: vec![],
            all_context_keys: HashSet::new(),
        }
    }

    #[test]
    fn linear_dag() {
        // A -> B -> C
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec![("a", "@A")]),
            make_node("C", vec![("b", "@B")]),
        ];
        let dag = build_dag(&nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 3);
        // A must come before B, B before C
        let pos_a = dag.topo_order.iter().position(|&i| i == 0).unwrap();
        let pos_b = dag.topo_order.iter().position(|&i| i == 1).unwrap();
        let pos_c = dag.topo_order.iter().position(|&i| i == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn diamond_dag() {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec![("a", "@A")]),
            make_node("C", vec![("a", "@A")]),
            make_node("D", vec![("b", "@B"), ("c", "@C")]),
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
        // A -> B -> A
        let nodes = vec![
            make_node("A", vec![("b", "@B")]),
            make_node("B", vec![("a", "@A")]),
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
        // B depends on "@ext" which is not a node — should not cause error
        let nodes = vec![
            make_node("A", vec![]),
            make_node("B", vec![("ext", "@ext")]),
        ];
        let dag = build_dag(&nodes).unwrap();
        assert_eq!(dag.topo_order.len(), 2);
        assert!(dag.deps[1].is_empty()); // B has no deps on other nodes
    }
}
