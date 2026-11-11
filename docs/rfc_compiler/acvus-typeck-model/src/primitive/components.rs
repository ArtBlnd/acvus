//! Strongly connected components: which nodes of a graph reach one another.

use std::collections::BTreeMap;

/// The strongly connected component of each node that `leads` leads from, as a
/// number: two nodes have one number where each reaches the other. A node that
/// is only led to, and leads nowhere, is in a component of its own and is given
/// its number too.
pub fn components<I>(leads: &BTreeMap<I, Vec<I>>) -> BTreeMap<I, usize>
where
    I: Copy + Ord,
{
    // Tarjan's algorithm, with the stack of the walk kept by hand: each frame is
    // a node and how many of the nodes it leads to have been walked.
    let nowhere = Vec::new();
    let leads_to = |node: &I| leads.get(node).unwrap_or(&nowhere);
    let mut index = BTreeMap::new();
    let mut lowest = BTreeMap::new();
    let mut component = BTreeMap::new();
    let mut open = Vec::new();
    for start in leads.keys() {
        if index.contains_key(start) {
            continue;
        }
        let mut walk = vec![(*start, 0)];
        while let Some((node, walked)) = walk.pop() {
            if walked == 0 {
                index.insert(node, index.len());
                lowest.insert(node, index[&node]);
                open.push(node);
            }
            if let Some(next) = leads_to(&node).get(walked) {
                walk.push((node, walked + 1));
                if !index.contains_key(next) {
                    walk.push((*next, 0));
                }
                continue;
            }
            // Every node it leads to is walked: the lowest index it reaches is
            // its own, or that of a node it leads to that is still open.
            for next in leads_to(&node) {
                if !component.contains_key(next) {
                    let reached = lowest[next].min(lowest[&node]);
                    lowest.insert(node, reached);
                }
            }
            if lowest[&node] == index[&node] {
                let number = component.len();
                while let Some(member) = open.pop() {
                    component.insert(member, number);
                    if member == node {
                        break;
                    }
                }
            }
        }
    }
    component
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(edges: &[(u32, u32)]) -> BTreeMap<u32, usize> {
        let mut leads: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (from, to) in edges {
            leads.entry(*from).or_default().push(*to);
        }
        components(&leads)
    }

    #[test]
    fn nodes_on_one_cycle_are_one_component() {
        let component = of(&[(0, 1), (1, 2), (2, 0), (2, 3)]);
        assert_eq!(component[&0], component[&1]);
        assert_eq!(component[&1], component[&2]);
        assert_ne!(component[&2], component[&3]);
    }

    #[test]
    fn a_node_that_reaches_another_and_is_not_reached_is_another_component() {
        let component = of(&[(0, 1), (1, 2)]);
        assert_ne!(component[&0], component[&1]);
        assert_ne!(component[&1], component[&2]);
    }

    #[test]
    fn two_cycles_joined_one_way_are_two_components() {
        let component = of(&[(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)]);
        assert_eq!(component[&0], component[&1]);
        assert_eq!(component[&2], component[&3]);
        assert_ne!(component[&0], component[&2]);
    }

    #[test]
    fn a_long_path_is_walked_without_recursion() {
        let edges: Vec<(u32, u32)> = (0..100_000).map(|node| (node, node + 1)).collect();
        assert_eq!(of(&edges).len(), 100_001);
    }
}
