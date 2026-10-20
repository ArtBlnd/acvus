//! Disjoint sets under union and find, each set carrying a descriptor that
//! the caller merges when two sets are joined.

use std::cell::Cell;
use std::cmp::Ordering;
use std::marker::PhantomData;

/// A key of one `UnionFind`, made only by its `push`.
pub(crate) trait Key: Copy + Eq {
    fn index(self) -> usize;
    fn from_index(index: usize) -> Self;
}

/// What a class knows. `merge` is a partial join: commutative, associative
/// and idempotent wherever it is defined. A class's descriptor is then the
/// join of every descriptor that entered it, whatever the order of the
/// unions and refinements that brought them in.
pub(crate) trait Descriptor: Sized {
    type Conflict;

    fn merge(&self, other: &Self) -> Result<Self, Self::Conflict>;
}

/// What a union did; only `Merged` changes the partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Union<K> {
    /// The keys were already in one class.
    Same,
    /// Two classes became one, represented by `kept`; `absorbed`, which
    /// represented the other, represents nothing any more.
    Merged { kept: K, absorbed: K },
}

/// A partition of the keys `push` made, each class with its descriptor.
/// Union by rank and path compression; `find` compresses through `Cell`s,
/// so a class's descriptor can stay borrowed while its parts are found.
pub(crate) struct UnionFind<K, D> {
    entries: Vec<Entry<D>>,
    key: PhantomData<K>,
}

enum Entry<D> {
    Root(Class<D>),
    Child { parent: Cell<usize> },
}

struct Class<D> {
    rank: u8,
    descriptor: D,
}

/// A class's representative, as `UnionFind::root` finds it.
struct Root<'a, D> {
    index: usize,
    class: &'a Class<D>,
}

impl<K, D> UnionFind<K, D>
where
    K: Key,
    D: Descriptor,
{
    pub(crate) fn new() -> Self {
        UnionFind { entries: Vec::new(), key: PhantomData }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// A new class of one new key.
    pub(crate) fn push(&mut self, descriptor: D) -> K {
        let index = self.entries.len();
        self.entries.push(Entry::Root(Class { rank: 0, descriptor }));
        K::from_index(index)
    }

    /// The representative of `key`'s class, the same for every key of it
    /// until the class is united with another.
    pub(crate) fn find(&self, key: K) -> K {
        K::from_index(self.root(key).index)
    }

    pub(crate) fn same(&self, a: K, b: K) -> bool {
        self.root(a).index == self.root(b).index
    }

    pub(crate) fn descriptor(&self, key: K) -> &D {
        &self.root(key).class.descriptor
    }

    /// Unites the classes of `a` and `b`, whose descriptor becomes `a`'s
    /// merged with `b`'s. Where the merge conflicts, nothing changes.
    pub(crate) fn union(&mut self, a: K, b: K) -> Result<Union<K>, D::Conflict> {
        let (root_a, root_b) = (self.root(a), self.root(b));
        if root_a.index == root_b.index {
            return Ok(Union::Same);
        }
        let descriptor = root_a.class.descriptor.merge(&root_b.class.descriptor)?;
        let (kept, absorbed) = match root_a.class.rank.cmp(&root_b.class.rank) {
            Ordering::Less => (root_b, root_a),
            Ordering::Greater | Ordering::Equal => (root_a, root_b),
        };
        let rank = if kept.class.rank == absorbed.class.rank { kept.class.rank + 1 } else { kept.class.rank };
        let (kept, absorbed) = (kept.index, absorbed.index);

        self.entries[absorbed] = Entry::Child { parent: Cell::new(kept) };
        self.entries[kept] = Entry::Root(Class { rank, descriptor });
        Ok(Union::Merged { kept: K::from_index(kept), absorbed: K::from_index(absorbed) })
    }

    /// Merges `descriptor` into `key`'s class, as a union with a new class
    /// holding it would, and answers whether the class's descriptor changed.
    /// Where the merge conflicts, nothing changes.
    pub(crate) fn refine(&mut self, key: K, descriptor: D) -> Result<bool, D::Conflict>
    where
        D: Eq,
    {
        let root = self.root(key);
        let merged = root.class.descriptor.merge(&descriptor)?;
        if merged == root.class.descriptor {
            return Ok(false);
        }
        let (index, rank) = (root.index, root.class.rank);

        self.entries[index] = Entry::Root(Class { rank, descriptor: merged });
        Ok(true)
    }

    /// `key`'s representative, every key on the way pointed at it.
    ///
    /// # Panics
    /// `key` was not made by this table's `push`.
    fn root(&self, key: K) -> Root<'_, D> {
        let index = key.index();
        let len = self.entries.len();
        let entry = |at: usize| {
            self.entries
                .get(at)
                .unwrap_or_else(|| panic!("UnionFind: key {index} was not made by this table, which holds {len} keys"))
        };
        let mut root = index;
        let class = loop {
            match entry(root) {
                Entry::Root(class) => break class,
                Entry::Child { parent } => root = parent.get(),
            }
        };

        let mut at = index;
        while let Entry::Child { parent } = &self.entries[at] {
            at = parent.replace(root);
        }
        Root { index: root, class }
    }
}

#[cfg(test)]
mod tests {
    use super::{Descriptor, Key, Union, UnionFind};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestKey(usize);

    impl Key for TestKey {
        fn index(self) -> usize {
            self.0
        }

        fn from_index(index: usize) -> Self {
            TestKey(index)
        }
    }

    /// A product of two semilattices: a set of bits under union, which
    /// never conflicts, and a flat lattice of constants, where two
    /// different constants conflict.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Model {
        bits: u8,
        constant: Option<u8>,
    }

    impl Descriptor for Model {
        type Conflict = ();

        fn merge(&self, other: &Self) -> Result<Self, ()> {
            let constant = match (self.constant, other.constant) {
                (Some(a), Some(b)) if a != b => return Err(()),
                (a, b) => a.or(b),
            };
            Ok(Model { bits: self.bits | other.bits, constant })
        }
    }

    const KEYS: usize = 4;

    const INITIAL: [Model; KEYS] = [
        Model { bits: 0b0001, constant: None },
        Model { bits: 0b0010, constant: Some(0) },
        Model { bits: 0b0100, constant: Some(1) },
        Model { bits: 0b1000, constant: None },
    ];

    const REFINEMENTS: [Model; 3] = [
        Model { bits: 0b1_0000, constant: None },
        Model { bits: 0, constant: Some(0) },
        Model { bits: 0, constant: Some(1) },
    ];

    #[derive(Debug, Clone, Copy)]
    enum Op {
        Union(usize, usize),
        Refine(usize, Model),
    }

    fn ops() -> Vec<Op> {
        let unions = (0..KEYS).flat_map(|a| (0..KEYS).map(move |b| Op::Union(a, b)));
        let refines = (0..KEYS).flat_map(|k| REFINEMENTS.into_iter().map(move |d| Op::Refine(k, d)));
        unions.chain(refines).collect()
    }

    /// The partition as a class label per key, and each label's descriptor.
    struct Reference {
        label: [usize; KEYS],
        descriptor: [Model; KEYS],
    }

    impl Reference {
        fn new() -> Self {
            Reference { label: [0, 1, 2, 3], descriptor: INITIAL }
        }

        fn union(&mut self, a: usize, b: usize) -> Result<bool, ()> {
            let (la, lb) = (self.label[a], self.label[b]);
            if la == lb {
                return Ok(false);
            }
            let merged = self.descriptor[la].merge(&self.descriptor[lb])?;
            for label in &mut self.label {
                if *label == lb {
                    *label = la;
                }
            }
            self.descriptor[la] = merged;
            Ok(true)
        }

        fn refine(&mut self, k: usize, d: Model) -> Result<bool, ()> {
            let l = self.label[k];
            let merged = self.descriptor[l].merge(&d)?;
            let changed = merged != self.descriptor[l];
            self.descriptor[l] = merged;
            Ok(changed)
        }
    }

    fn initial_table() -> UnionFind<TestKey, Model> {
        let mut table = UnionFind::new();
        for (index, descriptor) in INITIAL.into_iter().enumerate() {
            assert_eq!(table.push(descriptor), TestKey(index));
        }
        table
    }

    /// The table agrees with the reference on every key the reference has.
    fn agrees(table: &UnionFind<TestKey, Model>, reference: &Reference, trace: &[Op]) {
        for a in 0..KEYS {
            let rep = table.find(TestKey(a));
            assert_eq!(reference.label[rep.0], reference.label[a], "find leaves the class after {trace:?}");
            assert_eq!(*table.descriptor(TestKey(a)), reference.descriptor[reference.label[a]], "descriptor after {trace:?}");
            for b in 0..KEYS {
                let same = reference.label[a] == reference.label[b];
                assert_eq!(table.same(TestKey(a), TestKey(b)), same, "same({a}, {b}) after {trace:?}");
                assert_eq!(table.find(TestKey(a)) == table.find(TestKey(b)), same, "find({a}), find({b}) after {trace:?}");
            }
        }
    }

    /// Every sequence of four operations over four keys, checked after each
    /// step against the reference; a second table takes each refinement as a
    /// union with a new class, and must agree on the four keys as well.
    #[test]
    fn every_sequence_agrees_with_the_reference() {
        let ops = ops();
        let length = 4;
        let mut sequence = vec![0usize; length];
        loop {
            let mut table = initial_table();
            let mut by_union = initial_table();
            let mut reference = Reference::new();
            let mut trace = Vec::with_capacity(length);
            for &at in &sequence {
                let op = ops[at];
                trace.push(op);
                match op {
                    Op::Union(a, b) => {
                        let (before_a, before_b) = (table.find(TestKey(a)), table.find(TestKey(b)));
                        let expected = reference.union(a, b);
                        let got = table.union(TestKey(a), TestKey(b));
                        match (expected, got) {
                            (Ok(false), Ok(Union::Same)) | (Err(()), Err(())) => {}
                            (Ok(true), Ok(Union::Merged { kept, absorbed })) => {
                                let roots = [before_a, before_b];
                                assert!(roots.contains(&kept) && roots.contains(&absorbed) && kept != absorbed, "kept and absorbed after {trace:?}");
                                assert_eq!(table.find(absorbed), kept, "absorbed is found as kept after {trace:?}");
                            }
                            (expected, got) => panic!("union gave {got:?}, the reference {expected:?}, after {trace:?}"),
                        }
                        let by_union_got = by_union.union(TestKey(a), TestKey(b));
                        assert_eq!(by_union_got.is_ok(), expected.is_ok(), "the second table's union after {trace:?}");
                    }
                    Op::Refine(k, d) => {
                        let expected = reference.refine(k, d);
                        assert_eq!(table.refine(TestKey(k), d), expected, "refine after {trace:?}");
                        let fresh = by_union.push(d);
                        let united = by_union.union(TestKey(k), fresh);
                        assert_eq!(united.is_ok(), expected.is_ok(), "refine as a union after {trace:?}");
                    }
                }
                agrees(&table, &reference, &trace);
                agrees(&by_union, &reference, &trace);
            }

            let Some(position) = sequence.iter().rposition(|&at| at + 1 < ops.len()) else {
                break;
            };
            sequence[position] += 1;
            for later in &mut sequence[position + 1..] {
                *later = 0;
            }
        }
    }

    impl Descriptor for () {
        type Conflict = ();

        fn merge(&self, _: &Self) -> Result<Self, ()> {
            Ok(())
        }
    }

    #[test]
    fn many_keys_united_in_a_chain_are_one_class() {
        let count = 100_000;
        let mut table = UnionFind::<TestKey, ()>::new();
        let keys: Vec<TestKey> = (0..count).map(|_| table.push(())).collect();
        for pair in keys.windows(2) {
            assert!(matches!(table.union(pair[1], pair[0]), Ok(Union::Merged { .. })));
        }
        let rep = table.find(keys[0]);
        assert!(keys.iter().all(|&key| table.find(key) == rep));
        assert_eq!(table.len(), count);
    }

    #[test]
    #[should_panic(expected = "was not made by this table")]
    fn a_foreign_key_panics() {
        initial_table().find(TestKey(KEYS));
    }
}
