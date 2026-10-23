//! The nearest common ancestor of two nodes of a tree given by each node's
//! parent, by which two types of different constructors join along the
//! cast tree (type system: Join).

use std::fmt::Debug;
use std::hash::Hash;

use rustc_hash::FxHashSet;

/// The nearest common ancestor of `a` and `b` in the forest where `node`'s
/// parent is `parent(node)`, every node an ancestor of itself; `None` where
/// they have none in common. Cost: O(the depths of `a` and `b`).
///
/// # Panics
/// The parent chain from `a` or `b` returns to a node on it.
pub(crate) fn nearest_common_ancestor<N, P>(a: N, b: N, parent: P) -> Option<N>
where
    N: Copy + Eq + Hash + Debug,
    P: Fn(N) -> Option<N>,
{
    let ancestors_of_a = ancestors(a, &parent);
    let mut seen = FxHashSet::default();
    let mut at = Some(b);
    while let Some(node) = at {
        if ancestors_of_a.contains(&node) {
            return Some(node);
        }
        assert!(seen.insert(node), "nearest_common_ancestor: the parent chain from {b:?} returns to {node:?}");
        at = parent(node);
    }
    None
}

/// `node` and every ancestor of it.
fn ancestors<N, P>(node: N, parent: &P) -> FxHashSet<N>
where
    N: Copy + Eq + Hash + Debug,
    P: Fn(N) -> Option<N>,
{
    let mut ancestors = FxHashSet::default();
    let mut at = Some(node);
    while let Some(ancestor) = at {
        assert!(ancestors.insert(ancestor), "nearest_common_ancestor: the parent chain from {node:?} returns to {ancestor:?}");
        at = parent(ancestor);
    }
    ancestors
}

#[cfg(test)]
mod tests {
    use super::nearest_common_ancestor;

    /// Every ancestor of `node`, itself included, or `None` where its parent
    /// chain returns to a node on it.
    fn ancestors(parent: &[Option<usize>], node: usize) -> Option<Vec<usize>> {
        let mut chain = vec![node];
        let mut at = node;
        while let Some(next) = parent[at] {
            if chain.contains(&next) {
                return None;
            }
            chain.push(next);
            at = next;
        }
        Some(chain)
    }

    /// The definition read directly, in a forest without cycles: the common
    /// ancestor of which every other common ancestor is an ancestor.
    fn reference(parent: &[Option<usize>], a: usize, b: usize) -> Option<usize> {
        let chain = |node| ancestors(parent, node).unwrap_or_else(|| panic!("the reference is asked about {node} in {parent:?}, which has a cycle"));
        let of_b = chain(b);
        let common: Vec<usize> = chain(a).into_iter().filter(|node| of_b.contains(node)).collect();
        let nearest: Vec<usize> = common.iter().copied().filter(|&candidate| common.iter().all(|other| chain(candidate).contains(other))).collect();
        assert!(nearest.len() <= 1, "two nearest common ancestors of {a} and {b} in {parent:?}");
        nearest.first().copied()
    }

    /// Every parent function on up to five nodes whose chains have no cycle,
    /// and every pair of nodes in it.
    #[test]
    fn every_small_forest_agrees_with_the_definition() {
        for count in 1..=5usize {
            let choices = count + 1;
            for code in 0..choices.pow(count as u32) {
                let parent: Vec<Option<usize>> = (0..count)
                    .map(|node| match code / choices.pow(node as u32) % choices {
                        0 => None,
                        chosen => Some(chosen - 1),
                    })
                    .collect();
                if (0..count).any(|node| ancestors(&parent, node).is_none()) {
                    continue;
                }
                for a in 0..count {
                    for b in 0..count {
                        let got = nearest_common_ancestor(a, b, |node| parent[node]);
                        assert_eq!(got, reference(&parent, a, b), "{a} and {b} in {parent:?}");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "returns to")]
    fn a_cycle_above_the_first_node_panics() {
        let parent = [Some(1), Some(0), None];
        nearest_common_ancestor(0, 2, |node| parent[node]);
    }

    #[test]
    #[should_panic(expected = "returns to")]
    fn a_cycle_above_the_second_node_panics() {
        let parent = [Some(1), Some(0), None];
        nearest_common_ancestor(2, 0, |node| parent[node]);
    }
}
