//! The strongly connected components of a directed graph, by Tarjan's
//! algorithm, by which the solver groups the acvus fns that call one
//! another (type system: Instantiate 1).
//!
//! This is a copy of the walk in `passes::analysis::call_graph`, kept apart
//! so that either can change without the other.

/// The strongly connected components of the graph in which node `n` has an
/// edge to each node of `successors[n]`, each after every component it
/// reaches (Tarjan 1972, with an explicit stack). Cost: O(nodes + edges).
///
/// # Panics
/// The end of an edge is no node of the graph.
pub(crate) fn components_callees_first(successors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut tarjan = Tarjan::new(successors.len());
    for root in 0..successors.len() {
        if tarjan.visited[root].is_none() {
            tarjan.walk_from(root, successors);
        }
    }
    tarjan.components
}

/// Tarjan's walk: each node's visit order and the lowest visit order it
/// reaches on the stack, and the components completed so far.
struct Tarjan {
    visited: Vec<Option<usize>>,
    low: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    next: usize,
    components: Vec<Vec<usize>>,
}

/// A node on the walk, and the index of its next edge to follow.
struct Frame {
    node: usize,
    next_edge: usize,
}

impl Tarjan {
    fn new(count: usize) -> Tarjan {
        Tarjan {
            visited: vec![None; count],
            low: vec![0; count],
            on_stack: vec![false; count],
            stack: Vec::new(),
            next: 0,
            components: Vec::new(),
        }
    }

    fn enter(&mut self, node: usize) -> Frame {
        self.visited[node] = Some(self.next);
        self.low[node] = self.next;
        self.next += 1;
        self.stack.push(node);
        self.on_stack[node] = true;
        Frame { node, next_edge: 0 }
    }

    fn walk_from(&mut self, root: usize, edges: &[Vec<usize>]) {
        let mut walk = vec![self.enter(root)];
        while let Some(frame) = walk.last_mut() {
            let node = frame.node;
            if let Some(successor) = edges[node].get(frame.next_edge).copied() {
                frame.next_edge += 1;
                match self.visited[successor] {
                    None => {
                        let entered = self.enter(successor);
                        walk.push(entered);
                    }
                    Some(order) if self.on_stack[successor] => self.low[node] = self.low[node].min(order),
                    Some(_) => {}
                }
                continue;
            }
            walk.pop();
            if let Some(predecessor) = walk.last() {
                self.low[predecessor.node] = self.low[predecessor.node].min(self.low[node]);
            }
            if self.visited[node] == Some(self.low[node]) {
                self.complete(node);
            }
        }
    }

    /// Pops the component whose root is `root`.
    fn complete(&mut self, root: usize) {
        let mut component = Vec::new();
        loop {
            let member = self.stack.pop().expect("a component's root is on the stack below its members");
            self.on_stack[member] = false;
            component.push(member);
            if member == root {
                break;
            }
        }
        self.components.push(component);
    }
}

#[cfg(test)]
mod tests {
    use super::components_callees_first;

    /// The graph on `count` nodes whose edge from `i` to `j` is bit
    /// `i * count + j` of `edges`.
    fn graph(count: usize, edges: u32) -> Vec<Vec<usize>> {
        (0..count).map(|i| (0..count).filter(|j| edges >> (i * count + j) & 1 == 1).collect()).collect()
    }

    /// Paths of zero or more edges, by Warshall's algorithm.
    fn reflexive_closure(count: usize, edges: u32) -> Vec<Vec<bool>> {
        let mut path: Vec<Vec<bool>> = (0..count).map(|i| (0..count).map(|j| i == j || edges >> (i * count + j) & 1 == 1).collect()).collect();
        for k in 0..count {
            for i in 0..count {
                for j in 0..count {
                    path[i][j] |= path[i][k] && path[k][j];
                }
            }
        }
        path
    }

    /// Every directed graph on up to four nodes, self-loops included: the
    /// components partition the nodes, two nodes share one exactly when
    /// each reaches the other, and a component comes after every other
    /// component it reaches.
    #[test]
    fn every_small_graph_agrees_with_its_closure() {
        for count in 1..=4 {
            for edges in 0..1u32 << (count * count) {
                let successors = graph(count, edges);
                let path = reflexive_closure(count, edges);
                let components = components_callees_first(&successors);

                let mut position = vec![None; count];
                for (at, component) in components.iter().enumerate() {
                    for &node in component {
                        assert_eq!(position[node], None, "node {node} in two components of {successors:?}");
                        position[node] = Some(at);
                    }
                }
                let position: Vec<usize> = position
                    .into_iter()
                    .enumerate()
                    .map(|(node, at)| at.unwrap_or_else(|| panic!("node {node} in no component of {successors:?}")))
                    .collect();

                for a in 0..count {
                    for b in 0..count {
                        let mutual = path[a][b] && path[b][a];
                        assert_eq!(position[a] == position[b], mutual, "nodes {a} and {b} of {successors:?}");
                        if path[a][b] && !mutual {
                            assert!(position[b] < position[a], "{a} reaches {b} but comes first in {successors:?}");
                        }
                    }
                }
            }
        }
    }
}
