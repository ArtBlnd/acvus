//! Homeomorphic embedding of finite trees, by which a recursion whose type
//! arguments grow is refused (type system: Instantiate 3).

use std::fmt::Debug;
use std::hash::Hash;

use rustc_hash::{FxHashMap, FxHashSet};

/// Finite labelled trees, read through their nodes. A node that several
/// parents share is read as a copy under each.
pub(crate) trait Tree {
    type Node: Copy + Eq + Hash + Debug;
    type Label: Eq;

    fn label(&self, node: Self::Node) -> Self::Label;
    fn children(&self, node: Self::Node) -> impl Iterator<Item = Self::Node>;
}

/// Whether `big` homeomorphically embeds `small`, `small ⊴ big` (Kruskal):
/// `small` embeds into a child of `big`, or the two have one label and as
/// many children, and each child of `small` embeds into the child of `big`
/// at its position. Cost: O(|small| · |big| · arity), each pair of nodes
/// decided once.
///
/// # Panics
/// A node lies under itself.
pub(crate) fn embeds<T>(tree: &T, small: T::Node, big: T::Node) -> bool
where
    T: Tree,
{
    let small = Nodes::under(tree, small);
    let big = Nodes::under(tree, big);
    let width = big.labels.len();
    let mut embedded = vec![false; small.labels.len() * width];

    for b in 0..big.labels.len() {
        for s in 0..small.labels.len() {
            let diving = big.children[b].iter().any(|&child| embedded[s * width + child]);
            let coupling = || {
                small.labels[s] == big.labels[b]
                    && small.children[s].len() == big.children[b].len()
                    && small.children[s].iter().zip(&big.children[b]).all(|(&sc, &bc)| embedded[sc * width + bc])
            };
            embedded[s * width + b] = diving || coupling();
        }
    }
    embedded[small.root * width + big.root]
}

/// The distinct nodes under a root, numbered children before parents, each
/// with its label and its children's numbers in order. Deciding pairs in
/// this order finds every pair a pair reads already decided.
struct Nodes<L> {
    root: usize,
    labels: Vec<L>,
    children: Vec<Vec<usize>>,
}

/// A node on the walk, its children, and the index of the next to visit.
struct Frame<N> {
    node: N,
    children: Vec<N>,
    next: usize,
}

impl<L> Nodes<L> {
    fn under<T>(tree: &T, root: T::Node) -> Nodes<L>
    where
        T: Tree<Label = L>,
    {
        let mut number: FxHashMap<T::Node, usize> = FxHashMap::default();
        let mut entered = FxHashSet::default();
        let mut labels = Vec::new();
        let mut children = Vec::new();
        let mut walk = Vec::new();

        let mut enter = |node: T::Node, walk: &mut Vec<Frame<T::Node>>| {
            assert!(entered.insert(node), "embeds: {node:?} lies under itself");
            walk.push(Frame { node, children: tree.children(node).collect(), next: 0 });
        };
        enter(root, &mut walk);
        while let Some(mut frame) = walk.pop() {
            if let Some(&child) = frame.children.get(frame.next) {
                frame.next += 1;
                walk.push(frame);
                if !number.contains_key(&child) {
                    enter(child, &mut walk);
                }
                continue;
            }
            children.push(frame.children.iter().map(|child| number[child]).collect());
            labels.push(tree.label(frame.node));
            number.insert(frame.node, labels.len() - 1);
        }
        Nodes { root: number[&root], labels, children }
    }
}

#[cfg(test)]
mod tests {
    use super::{Tree, embeds};

    /// A tree written out, for the reference.
    #[derive(Debug, Clone)]
    struct Term {
        label: u8,
        children: Vec<Term>,
    }

    struct Node {
        label: u8,
        children: Vec<usize>,
    }

    #[derive(Default)]
    struct Arena {
        nodes: Vec<Node>,
    }

    impl Arena {
        fn add(&mut self, label: u8, children: Vec<usize>) -> usize {
            self.nodes.push(Node { label, children });
            self.nodes.len() - 1
        }

        fn insert(&mut self, term: &Term) -> usize {
            let children = term.children.iter().map(|child| self.insert(child)).collect();
            self.add(term.label, children)
        }
    }

    impl Tree for Arena {
        type Node = usize;
        type Label = u8;

        fn label(&self, node: usize) -> u8 {
            self.nodes[node].label
        }

        fn children(&self, node: usize) -> impl Iterator<Item = usize> {
            self.nodes[node].children.iter().copied()
        }
    }

    /// The definition, read directly.
    fn reference(small: &Term, big: &Term) -> bool {
        big.children.iter().any(|child| reference(small, child))
            || (small.label == big.label
                && small.children.len() == big.children.len()
                && small.children.iter().zip(&big.children).all(|(s, b)| reference(s, b)))
    }

    /// Every ordered tree of `size` nodes labelled 0 or 1.
    fn trees(size: usize) -> Vec<Term> {
        (0..2).flat_map(|label| forests(size - 1).into_iter().map(move |children| Term { label, children })).collect()
    }

    /// Every ordered sequence of trees of `size` nodes in all.
    fn forests(size: usize) -> Vec<Vec<Term>> {
        if size == 0 {
            return vec![Vec::new()];
        }
        let mut every = Vec::new();
        for first in 1..=size {
            for tree in trees(first) {
                for rest in forests(size - first) {
                    let mut forest = vec![tree.clone()];
                    forest.extend(rest);
                    every.push(forest);
                }
            }
        }
        every
    }

    /// Every pair of trees of up to five nodes labelled 0 or 1.
    #[test]
    fn every_small_pair_agrees_with_the_definition() {
        let terms: Vec<Term> = (1..=5).flat_map(trees).collect();
        let mut arena = Arena::default();
        let nodes: Vec<usize> = terms.iter().map(|term| arena.insert(term)).collect();
        for (small, &small_node) in terms.iter().zip(&nodes) {
            for (big, &big_node) in terms.iter().zip(&nodes) {
                assert_eq!(embeds(&arena, small_node, big_node), reference(small, big), "{small:?} into {big:?}");
            }
        }
    }

    /// A node two parents share is read as a copy under each.
    #[test]
    fn a_shared_node_is_read_under_each_parent() {
        const F: u8 = 0;
        const G: u8 = 1;
        const A: u8 = 2;
        let mut arena = Arena::default();
        let leaf = arena.add(A, vec![]);
        let shared = arena.add(G, vec![leaf]);
        let big = arena.add(F, vec![shared, shared]);
        let leaf_term = Term { label: A, children: vec![] };
        let shared_term = Term { label: G, children: vec![leaf_term.clone()] };
        let big_term = Term { label: F, children: vec![shared_term.clone(), shared_term.clone()] };
        for small_term in [
            Term { label: F, children: vec![leaf_term.clone(), leaf_term.clone()] },
            Term { label: F, children: vec![shared_term.clone(), leaf_term.clone()] },
            Term { label: F, children: vec![shared_term.clone(), shared_term.clone(), leaf_term.clone()] },
            Term { label: G, children: vec![shared_term.clone()] },
        ] {
            let small = arena.insert(&small_term);
            assert_eq!(embeds(&arena, small, big), reference(&small_term, &big_term), "{small_term:?}");
        }
    }

    /// Example 2 of the type system: `nest`'s inner use grows its type
    /// arguments, and `swap`'s does not.
    #[test]
    fn the_growing_type_arguments_of_example_two() {
        const TYPE_ARGUMENTS: u8 = 0;
        const PAIR: u8 = 1;
        const STR: u8 = 2;
        const U8: u8 = 3;
        const I32: u8 = 4;
        let mut arena = Arena::default();
        let pair = |arena: &mut Arena, l: u8, r: u8| {
            let (l, r) = (arena.add(l, vec![]), arena.add(r, vec![]));
            arena.add(PAIR, vec![l, r])
        };
        let made_pair = pair(&mut arena, STR, U8);
        let swapped_pair = pair(&mut arena, U8, STR);
        let type_arguments = |arena: &mut Arena, first: usize| {
            let second = arena.add(I32, vec![]);
            arena.add(TYPE_ARGUMENTS, vec![first, second])
        };
        let made = type_arguments(&mut arena, made_pair);
        let nested = {
            let r = arena.add(U8, vec![]);
            arena.add(PAIR, vec![made_pair, r])
        };
        let grown = type_arguments(&mut arena, nested);
        let swapped = type_arguments(&mut arena, swapped_pair);
        assert!(embeds(&arena, made, grown));
        assert!(!embeds(&arena, grown, made));
        assert!(!embeds(&arena, made, swapped));
        assert!(!embeds(&arena, swapped, made));
    }

    #[test]
    #[should_panic(expected = "lies under itself")]
    fn a_cycle_panics() {
        let mut arena = Arena::default();
        arena.nodes.push(Node { label: 0, children: vec![0] });
        embeds(&arena, 0, 0);
    }
}
