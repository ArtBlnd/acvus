//! A persistent vector: a radix tree of reference-counted nodes, so a copy
//! is one reference count and a write copies the path from the root to the
//! element it writes.

use std::fmt;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

const BITS: u32 = 5;
const WIDTH: usize = 1 << BITS;
const MASK: usize = WIDTH - 1;

#[derive(Clone)]
enum Node<T> {
    Branch(Vec<Arc<Node<T>>>),
    Leaf(Vec<T>),
}

impl<T> Node<T> {
    fn empty(shift: u32) -> Node<T> {
        match shift {
            0 => Node::Leaf(Vec::with_capacity(WIDTH)),
            _ => Node::Branch(Vec::with_capacity(WIDTH)),
        }
    }
}

/// A sequence indexed from zero, grown at its end, as `Vec` is.
pub(crate) struct PVec<T> {
    root: Arc<Node<T>>,
    /// The tree holds `WIDTH << shift` elements before it needs another
    /// level.
    shift: u32,
    len: usize,
}

impl<T> Clone for PVec<T> {
    fn clone(&self) -> Self {
        PVec {
            root: Arc::clone(&self.root),
            shift: self.shift,
            len: self.len,
        }
    }
}

impl<T> Default for PVec<T> {
    fn default() -> Self {
        PVec::new()
    }
}

impl<T> PVec<T> {
    pub(crate) fn new() -> Self {
        PVec {
            root: Arc::new(Node::empty(0)),
            shift: 0,
            len: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let mut node = &*self.root;
        let mut shift = self.shift;
        loop {
            match node {
                Node::Branch(children) => {
                    node = &children[(index >> shift) & MASK];
                    shift -= BITS;
                }
                Node::Leaf(items) => return items.get(index & MASK),
            }
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len).map(move |index| &self[index])
    }
}

impl<T: Clone> PVec<T> {
    pub(crate) fn push(&mut self, value: T) {
        if self.len == WIDTH << self.shift {
            let old = std::mem::replace(&mut self.root, Arc::new(Node::Branch(Vec::new())));
            self.root = Arc::new(Node::Branch(vec![old]));
            self.shift += BITS;
        }
        let index = self.len;
        let mut node = &mut self.root;
        let mut shift = self.shift;
        loop {
            match Arc::make_mut(node) {
                Node::Branch(children) => {
                    let at = (index >> shift) & MASK;
                    shift -= BITS;
                    if at == children.len() {
                        children.push(Arc::new(Node::empty(shift)));
                    }
                    node = &mut children[at];
                }
                Node::Leaf(items) => {
                    items.push(value);
                    break;
                }
            }
        }
        self.len += 1;
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let mut node = &mut self.root;
        let mut shift = self.shift;
        loop {
            match Arc::make_mut(node) {
                Node::Branch(children) => {
                    node = &mut children[(index >> shift) & MASK];
                    shift -= BITS;
                }
                Node::Leaf(items) => return items.get_mut(index & MASK),
            }
        }
    }
}

impl<T> Index<usize> for PVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        let len = self.len;
        self.get(index)
            .unwrap_or_else(|| panic!("index {index} out of range for a vector of {len}"))
    }
}

impl<T: Clone> IndexMut<usize> for PVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let len = self.len;
        self.get_mut(index)
            .unwrap_or_else(|| panic!("index {index} out of range for a vector of {len}"))
    }
}

impl<T: fmt::Debug> fmt::Debug for PVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::PVec;

    #[test]
    fn a_copy_written_leaves_the_original() {
        let mut held: PVec<usize> = PVec::new();
        for value in 0..40_000 {
            held.push(value);
            assert_eq!(held.len(), value + 1);
        }
        assert!(held.iter().copied().eq(0..40_000));
        let mut copy = held.clone();
        for index in (0..40_000).step_by(7) {
            copy[index] = index + 1;
        }
        copy.push(7);
        assert!(held.iter().copied().eq(0..40_000));
        assert_eq!(copy.len(), 40_001);
        for index in 0..40_000 {
            let want = match index % 7 {
                0 => index + 1,
                _ => index,
            };
            assert_eq!(copy[index], want);
        }
        assert_eq!(copy.get(40_001), None);
    }
}
