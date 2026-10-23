//! Whether a node of a directed graph reaches another along one or more
//! edges, as the transitive closure of the waiting relation asks (type
//! system: Lifecycle, Join).

/// The nodes that reach `target` along one or more edges of the graph in
/// which node `n` has an edge to each node of `successors[n]`: `result[n]`
/// holds exactly when such a path leads from `n` to `target`, so `target`
/// is among them only where it lies on a cycle. Cost: O(nodes + edges).
///
/// # Panics
/// `target` or the end of an edge is no node of the graph.
pub(crate) fn reaching(successors: &[Vec<usize>], target: usize) -> Vec<bool> {
    let mut predecessors = vec![Vec::new(); successors.len()];
    for (node, ends) in successors.iter().enumerate() {
        for &end in ends {
            predecessors[end].push(node);
        }
    }

    let mut reaches = vec![false; successors.len()];
    let mut pending = vec![target];
    while let Some(node) = pending.pop() {
        for &predecessor in &predecessors[node] {
            if !reaches[predecessor] {
                reaches[predecessor] = true;
                pending.push(predecessor);
            }
        }
    }
    reaches
}

#[cfg(test)]
mod tests {
    use super::reaching;

    /// The graph on `count` nodes whose edge from `i` to `j` is bit
    /// `i * count + j` of `edges`.
    fn graph(count: usize, edges: u32) -> Vec<Vec<usize>> {
        (0..count).map(|i| (0..count).filter(|j| edges >> (i * count + j) & 1 == 1).collect()).collect()
    }

    /// Paths of one or more edges, by Warshall's algorithm.
    fn closure(count: usize, edges: u32) -> Vec<Vec<bool>> {
        let mut path: Vec<Vec<bool>> = (0..count).map(|i| (0..count).map(|j| edges >> (i * count + j) & 1 == 1).collect()).collect();
        for k in 0..count {
            for i in 0..count {
                for j in 0..count {
                    path[i][j] |= path[i][k] && path[k][j];
                }
            }
        }
        path
    }

    /// Every directed graph on up to four nodes, self-loops included, and
    /// every target in it.
    #[test]
    fn every_small_graph_agrees_with_its_closure() {
        for count in 1..=4 {
            for edges in 0..1u32 << (count * count) {
                let successors = graph(count, edges);
                let path = closure(count, edges);
                for target in 0..count {
                    let expected: Vec<bool> = path.iter().map(|from| from[target]).collect();
                    assert_eq!(reaching(&successors, target), expected, "target {target} in {successors:?}");
                }
            }
        }
    }
}
