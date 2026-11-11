//! The owners: each undecided class with the parts named on it, and each part
//! with the undecided classes it is named on.
//!
//! A part is decided no earlier than a class it is named on, so deciding asks
//! of a class which undecided classes own it. That is kept here both ways, by
//! owner and by part, and the two are written together in one place
//! (`Owners::put`, `Owners::take`): nothing else writes either, so neither can
//! say what the other does not.
//!
//! What is kept is, of each undecided class, the parts it has now, each by its
//! root. A decided class owns nothing. This knows no class but by its id:
//! whoever holds the classes tells it when a part is named, when a class is
//! written whole, and when two are merged, a decision being a merge.

use std::collections::{BTreeMap, BTreeSet};

/// Each class of one side, with the classes of the other.
pub(super) type Rows<I> = BTreeMap<I, BTreeSet<I>>;

#[derive(Clone, Debug)]
pub(super) struct Owners<I> {
    parts_of: Rows<I>,
    /// The same the other way round.
    owners_of: Rows<I>,
}

// Written out, as a derive would ask an id to have a default, and none has.
impl<I> Default for Owners<I> {
    fn default() -> Self {
        Self {
            parts_of: BTreeMap::new(),
            owners_of: BTreeMap::new(),
        }
    }
}

impl<I> Owners<I>
where
    I: Copy + Ord,
{
    fn put(&mut self, owner: I, part: I) {
        self.parts_of.entry(owner).or_default().insert(part);
        self.owners_of.entry(part).or_default().insert(owner);
    }

    fn take(&mut self, owner: I, part: I) {
        for (map, key, value) in [(&mut self.parts_of, owner, part), (&mut self.owners_of, part, owner)] {
            if let Some(set) = map.get_mut(&key) {
                set.remove(&value);
                if set.is_empty() {
                    map.remove(&key);
                }
            }
        }
    }

    /// The undecided class `owner` has come to have the part `part`.
    pub(super) fn named(&mut self, owner: I, part: I) {
        self.put(owner, part);
    }

    /// The class `owner` is written whole: it owns `parts` and nothing else,
    /// which is nothing where it is decided.
    pub(super) fn written<P>(&mut self, owner: I, parts: P)
    where
        P: IntoIterator<Item = I>,
    {
        for part in self.parts_of.get(&owner).cloned().unwrap_or_default() {
            self.take(owner, part);
        }
        for part in parts {
            self.put(owner, part);
        }
    }

    /// The class `gone` has been merged into `kept`, which has the parts `parts`
    /// where it is undecided and none where it is decided. What owned `gone`
    /// owns `kept`, and `kept` owns what it has now.
    pub(super) fn merged<P>(&mut self, kept: I, gone: I, parts: P)
    where
        P: IntoIterator<Item = I>,
    {
        self.written(gone, []);
        for owner in self.owners_of.get(&gone).cloned().unwrap_or_default() {
            self.take(owner, gone);
            self.put(owner, kept);
        }
        self.written(kept, parts);
    }

    /// Every class an undecided class owns.
    pub(super) fn owned(&self) -> impl Iterator<Item = I> {
        self.owners_of.keys().copied()
    }

    /// The undecided classes that own `part`.
    pub(super) fn of(&self, part: I) -> impl Iterator<Item = I> {
        self.owners_of.get(&part).into_iter().flatten().copied()
    }

    /// Each owner with its parts, and each part with its owners, for these to
    /// be checked against one another and against the classes.
    pub(super) fn both_ways(&self) -> (&Rows<I>, &Rows<I>) {
        (&self.parts_of, &self.owners_of)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Listed = Vec<(u32, Vec<u32>)>;

    fn rows(owners: &Owners<u32>) -> (Listed, Listed) {
        let listed = |map: &Rows<u32>| map.iter().map(|(key, set)| (*key, set.iter().copied().collect())).collect();
        let (parts_of, owners_of) = owners.both_ways();
        (listed(parts_of), listed(owners_of))
    }

    #[test]
    fn a_part_named_is_kept_by_its_owner_and_by_itself() {
        let mut owners = Owners::default();
        owners.named(1, 2);
        owners.named(1, 3);
        owners.named(4, 3);
        assert_eq!(rows(&owners), (vec![(1, vec![2, 3]), (4, vec![3])], vec![(2, vec![1]), (3, vec![1, 4])]));
        assert_eq!(owners.of(3).collect::<Vec<_>>(), [1, 4]);
        assert_eq!(owners.owned().collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn a_class_written_whole_owns_what_it_is_written_with_and_no_more() {
        let mut owners = Owners::default();
        owners.named(1, 2);
        owners.named(1, 3);
        owners.written(1, [3, 5]);
        assert_eq!(rows(&owners), (vec![(1, vec![3, 5])], vec![(3, vec![1]), (5, vec![1])]));
        owners.written(1, []);
        assert_eq!(rows(&owners), (vec![], vec![]));
    }

    #[test]
    fn a_class_merged_away_is_owned_and_owns_as_the_class_kept() {
        let mut owners = Owners::default();
        owners.named(1, 7);
        owners.named(7, 8);
        owners.named(5, 9);
        // 7 is merged into 5, which then has the parts of both.
        owners.merged(5, 7, [8, 9]);
        assert_eq!(rows(&owners), (vec![(1, vec![5]), (5, vec![8, 9])], vec![(5, vec![1]), (8, vec![5]), (9, vec![5])]));
        // 5 is decided, by a merge with a decided class 3: it owns nothing.
        owners.merged(3, 5, []);
        assert_eq!(rows(&owners), (vec![(1, vec![3])], vec![(3, vec![1])]));
    }
}
