use std::hash::{BuildHasher, Hash, Hasher};

/// A deque with per-turn diff tracking and origin checksum.
///
/// Supports four operations:
/// - `push` / `extend`: add items to the back
/// - `pop`: remove one item from the back
/// - `consume(n)`: remove `n` items from the front
///
/// The visible window is `items[head..]`.
/// Diff tracking records what changed since the last checkpoint.
///
/// Each deque is born with a random checksum. The checksum evolves
/// through [`into_diff`](Self::into_diff), which squashes the origin
/// checksum with the diff metadata. To extract a diff, the caller must
/// provide the original deque (`origin: &Self`) — if the checksums
/// diverge, the call panics.
///
/// This type is intentionally NOT Serialize/Deserialize.
/// Use [`into_diff`](Self::into_diff) to extract a storable diff,
/// and [`OwnedDequeDiff::apply`] to reconstruct from a previous state.
#[derive(Debug, Clone)]
pub struct TrackedDeque<T> {
    items: Vec<T>,
    head: usize,
    checkpoint: Option<DequeCheckpoint>,
    checksum: u64,
}

#[derive(Debug, Clone, Copy)]
struct DequeCheckpoint {
    head: usize,
    tail: usize,
    /// Minimum `items.len()` reached since checkpoint.
    /// Tracks the "low water mark" to distinguish
    /// "popped 3, pushed 2" from "popped 1, pushed 0".
    low_water_mark: usize,
}

/// Borrowed diff between a checkpoint and the current state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DequeDiff<'a, T> {
    pub consumed: usize,
    pub removed_back: usize,
    pub pushed: &'a [T],
}

/// Owned diff — suitable for serialization and storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedDequeDiff<T> {
    pub consumed: usize,
    pub removed_back: usize,
    /// The new items pushed since the checkpoint.
    pub pushed: Vec<T>,
}

fn random_checksum() -> u64 {
    std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish()
}

fn squash_checksum(origin: u64, consumed: usize, removed_back: usize, pushed_len: usize) -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    origin.hash(&mut hasher);
    consumed.hash(&mut hasher);
    removed_back.hash(&mut hasher);
    pushed_len.hash(&mut hasher);
    hasher.finish()
}

impl<T> TrackedDeque<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            head: 0,
            checkpoint: None,
            checksum: random_checksum(),
        }
    }

    pub fn from_vec(items: Vec<T>) -> Self {
        Self {
            items,
            head: 0,
            checkpoint: None,
            checksum: random_checksum(),
        }
    }

    // ── Accessors ──────────────────────────────────────────────

    pub fn checksum(&self) -> u64 {
        self.checksum
    }

    pub fn as_slice(&self) -> &[T] {
        &self.items[self.head..]
    }

    pub fn len(&self) -> usize {
        self.items.len() - self.head
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consume the deque and return the visible items.
    pub fn into_vec(self) -> Vec<T> {
        if self.head == 0 {
            self.items
        } else {
            self.items.into_iter().skip(self.head).collect()
        }
    }

    // ── Mutators ───────────────────────────────────────────────

    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn extend(&mut self, items: impl IntoIterator<Item = T>) {
        self.items.extend(items);
    }

    /// Pop one item from the back. Returns `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = self.items.pop();
        if let Some(cp) = &mut self.checkpoint {
            cp.low_water_mark = cp.low_water_mark.min(self.items.len());
        }
        item
    }

    /// Consume `n` items from the front.
    ///
    /// # Panics
    ///
    /// Panics if `n > self.len()`.
    pub fn consume(&mut self, n: usize) {
        assert!(
            n <= self.len(),
            "consume({n}) but only {} items visible",
            self.len()
        );
        self.head += n;
    }

    // ── Diff tracking ──────────────────────────────────────────

    /// Take a checkpoint of the current state.
    pub fn checkpoint(&mut self) {
        self.checkpoint = Some(DequeCheckpoint {
            head: self.head,
            tail: self.items.len(),
            low_water_mark: self.items.len(),
        });
    }

    /// Compute the diff since the last checkpoint (borrowed).
    ///
    /// # Panics
    ///
    /// Panics if `origin.checksum` does not match this deque's checksum.
    /// Panics if no checkpoint has been taken.
    pub fn diff(&self, origin: &Self) -> DequeDiff<'_, T> {
        assert_eq!(
            self.checksum,
            origin.checksum,
            "checksum mismatch: origin {:#x} vs self {:#x}",
            origin.checksum,
            self.checksum,
        );
        let cp = self
            .checkpoint
            .as_ref()
            .expect("diff called without checkpoint");
        DequeDiff {
            consumed: self.head - cp.head,
            removed_back: cp.tail - cp.low_water_mark,
            pushed: &self.items[cp.low_water_mark..],
        }
    }

    /// Consume the deque, verify origin, and return the squashed deque + diff.
    ///
    /// The returned `TrackedDeque` has:
    /// - items = current visible items (compacted)
    /// - checksum = squash(origin.checksum, diff metadata)
    /// - no checkpoint
    ///
    /// # Panics
    ///
    /// Panics if `origin.checksum` does not match this deque's checksum.
    /// Panics if no checkpoint has been taken.
    pub fn into_diff(self, origin: &Self) -> (TrackedDeque<T>, OwnedDequeDiff<T>)
    where
        T: Clone,
    {
        assert_eq!(
            self.checksum,
            origin.checksum,
            "checksum mismatch: origin {:#x} vs self {:#x}",
            origin.checksum,
            self.checksum,
        );
        let cp = self
            .checkpoint
            .expect("into_diff called without checkpoint");
        let consumed = self.head - cp.head;
        let removed_back = cp.tail - cp.low_water_mark;

        // items[low_water_mark..] are the pushed items
        let mut items = self.items;
        let pushed = items.split_off(cp.low_water_mark);

        // visible = items[head..] (the preserved part)
        let visible: Vec<T> = items.into_iter().skip(self.head).collect();

        // squashed items = preserved ++ pushed
        let mut squashed_items = visible;
        squashed_items.extend(pushed.clone());

        let new_checksum = squash_checksum(origin.checksum, consumed, removed_back, pushed.len());

        let squashed = TrackedDeque {
            items: squashed_items,
            head: 0,
            checkpoint: None,
            checksum: new_checksum,
        };

        let diff = OwnedDequeDiff {
            consumed,
            removed_back,
            pushed,
        };

        (squashed, diff)
    }

    /// Check whether the deque has been modified since the last checkpoint.
    pub fn is_dirty(&self) -> bool {
        match &self.checkpoint {
            None => false,
            Some(cp) => {
                self.head != cp.head
                    || self.items.len() != cp.tail
                    || cp.low_water_mark != cp.tail
            }
        }
    }
}

impl<T> OwnedDequeDiff<T> {
    /// Apply this diff to a previous visible state, producing a new visible state.
    pub fn apply(self, mut prev: Vec<T>) -> Vec<T> {
        if self.consumed > 0 {
            prev.drain(..self.consumed);
        }
        if self.removed_back > 0 {
            let new_len = prev.len() - self.removed_back;
            prev.truncate(new_len);
        }
        prev.extend(self.pushed);
        prev
    }

    /// Returns `true` if this diff represents no change.
    pub fn is_noop(&self) -> bool {
        self.consumed == 0 && self.removed_back == 0 && self.pushed.is_empty()
    }
}

impl<T: PartialEq> PartialEq for TrackedDeque<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T> Default for TrackedDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic operations ───────────────────────────────────────

    #[test]
    fn empty_deque() {
        let deque: TrackedDeque<i32> = TrackedDeque::new();
        assert!(deque.is_empty());
        assert_eq!(deque.len(), 0);
        assert_eq!(deque.as_slice(), &[] as &[i32]);
    }

    #[test]
    fn new_deque_has_nonzero_checksum() {
        // Random checksum — extremely unlikely to be 0
        let d1 = TrackedDeque::<i32>::new();
        let d2 = TrackedDeque::<i32>::new();
        // Two deques should have different checksums
        assert_ne!(d1.checksum(), d2.checksum());
    }

    #[test]
    fn push_and_view() {
        let mut deque = TrackedDeque::new();
        deque.push(1);
        deque.push(2);
        deque.push(3);
        assert_eq!(deque.as_slice(), &[1, 2, 3]);
        assert_eq!(deque.len(), 3);
    }

    #[test]
    fn pop_returns_last() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        assert_eq!(deque.pop(), Some(3));
        assert_eq!(deque.as_slice(), &[1, 2]);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut deque: TrackedDeque<i32> = TrackedDeque::new();
        assert_eq!(deque.pop(), None);
    }

    #[test]
    fn consume_from_front() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        deque.consume(2);
        assert_eq!(deque.as_slice(), &[3, 4, 5]);
        assert_eq!(deque.len(), 3);
    }

    #[test]
    #[should_panic(expected = "consume(4) but only 3 items visible")]
    fn consume_too_many_panics() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.consume(4);
    }

    #[test]
    fn extend_items() {
        let mut deque = TrackedDeque::new();
        deque.extend(vec![1, 2, 3]);
        deque.extend(vec![4, 5]);
        assert_eq!(deque.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn into_vec_no_consume() {
        let deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        assert_eq!(deque.into_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn into_vec_after_consume() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        deque.consume(2);
        assert_eq!(deque.into_vec(), vec![3, 4, 5]);
    }

    // ── Checksum preserved through mutations ─────────────────

    #[test]
    fn checksum_preserved_through_push() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2]);
        let cs = deque.checksum();
        deque.push(3);
        assert_eq!(deque.checksum(), cs);
    }

    #[test]
    fn checksum_preserved_through_pop() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let cs = deque.checksum();
        deque.pop();
        assert_eq!(deque.checksum(), cs);
    }

    #[test]
    fn checksum_preserved_through_consume() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let cs = deque.checksum();
        deque.consume(1);
        assert_eq!(deque.checksum(), cs);
    }

    #[test]
    fn checksum_preserved_through_extend() {
        let mut deque = TrackedDeque::new();
        let cs = deque.checksum();
        deque.extend(vec![1, 2, 3]);
        assert_eq!(deque.checksum(), cs);
    }

    #[test]
    fn checksum_preserved_through_checkpoint() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2]);
        let cs = deque.checksum();
        deque.checkpoint();
        assert_eq!(deque.checksum(), cs);
    }

    #[test]
    fn clone_preserves_checksum() {
        let deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let cloned = deque.clone();
        assert_eq!(deque.checksum(), cloned.checksum());
    }

    // ── Borrowed diff ──────────────────────────────────────────

    #[test]
    #[should_panic(expected = "diff called without checkpoint")]
    fn diff_without_checkpoint_panics() {
        let deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.diff(&origin);
    }

    #[test]
    fn diff_append_only() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.push(4);
        deque.push(5);

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 0);
        assert_eq!(diff.pushed, &[4, 5]);
    }

    #[test]
    fn diff_consume_only() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.consume(2);

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 2);
        assert_eq!(diff.removed_back, 0);
        assert_eq!(diff.pushed, &[] as &[i32]);
    }

    #[test]
    fn diff_pop_only() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.pop();
        deque.pop();

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 2);
        assert_eq!(diff.pushed, &[] as &[i32]);
    }

    #[test]
    fn diff_pop_then_push() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.pop();
        deque.pop();
        deque.push(10);
        deque.push(20);
        deque.push(30);

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 2);
        assert_eq!(diff.pushed, &[10, 20, 30]);
        assert_eq!(deque.as_slice(), &[1, 2, 3, 10, 20, 30]);
    }

    #[test]
    fn diff_consume_pop_push() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.consume(2);
        deque.pop();
        deque.push(10);
        deque.push(20);

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 2);
        assert_eq!(diff.removed_back, 1);
        assert_eq!(diff.pushed, &[10, 20]);
        assert_eq!(deque.as_slice(), &[3, 4, 10, 20]);
    }

    #[test]
    fn diff_no_changes() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.checkpoint();

        let diff = deque.diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 0);
        assert_eq!(diff.pushed, &[] as &[i32]);
        assert!(!deque.is_dirty());
    }

    // ── Checksum mismatch panics ─────────────────────────────

    #[test]
    #[should_panic(expected = "checksum mismatch")]
    fn diff_checksum_mismatch_panics() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.checkpoint();
        deque.push(4);

        let hijacked = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.diff(&hijacked);
    }

    #[test]
    #[should_panic(expected = "checksum mismatch")]
    fn into_diff_checksum_mismatch_panics() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.checkpoint();
        deque.push(4);

        let hijacked = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.into_diff(&hijacked);
    }

    #[test]
    #[should_panic(expected = "checksum mismatch")]
    fn fresh_deque_cannot_diff_against_different_origin() {
        let mut original = TrackedDeque::from_vec(vec![1, 2, 3]);
        original.checkpoint();
        original.push(4);

        // Someone creates a new [] and tries to pass it off
        let mut fake = TrackedDeque::new();
        fake.checkpoint();
        fake.extend(vec![1, 2, 3, 4]);

        fake.into_diff(&original);
    }

    // ── is_dirty ───────────────────────────────────────────────

    #[test]
    fn is_dirty_after_push() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2]);
        deque.checkpoint();
        assert!(!deque.is_dirty());
        deque.push(3);
        assert!(deque.is_dirty());
    }

    #[test]
    fn is_dirty_after_consume() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.checkpoint();
        deque.consume(1);
        assert!(deque.is_dirty());
    }

    #[test]
    fn is_dirty_pop_then_push_same_len() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        deque.checkpoint();
        deque.pop();
        deque.push(99);
        assert!(deque.is_dirty());
    }

    // ── into_diff (owned) ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "into_diff called without checkpoint")]
    fn into_diff_without_checkpoint_panics() {
        let deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.into_diff(&origin);
    }

    #[test]
    fn into_diff_append_only() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.push(4);
        deque.push(5);

        let (squashed, diff) = deque.into_diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 0);
        assert_eq!(diff.pushed, vec![4, 5]);
        assert_eq!(squashed.as_slice(), &[1, 2, 3, 4, 5]);
        assert_ne!(squashed.checksum(), origin.checksum());
    }

    #[test]
    fn into_diff_pop_then_push() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.pop();
        deque.pop();
        deque.push(10);

        let (squashed, diff) = deque.into_diff(&origin);
        assert_eq!(diff.consumed, 0);
        assert_eq!(diff.removed_back, 2);
        assert_eq!(diff.pushed, vec![10]);
        assert_eq!(squashed.as_slice(), &[1, 2, 3, 10]);
    }

    #[test]
    fn into_diff_squashed_checksum_evolves() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.push(4);

        let (squashed, _) = deque.into_diff(&origin);
        // Squashed checksum differs from origin
        assert_ne!(squashed.checksum(), origin.checksum());
        // Squashed can be used as next origin
        let mut next = squashed.clone();
        next.checkpoint();
        next.push(5);
        let (squashed2, diff2) = next.into_diff(&squashed);
        assert_eq!(diff2.pushed, vec![5]);
        assert_ne!(squashed2.checksum(), squashed.checksum());
    }

    // ── apply_diff ─────────────────────────────────────────────

    #[test]
    fn apply_diff_append_only() {
        let prev = vec![1, 2, 3];
        let diff = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![4, 5],
        };
        assert_eq!(diff.apply(prev), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn apply_diff_consume_and_push() {
        let prev = vec![1, 2, 3, 4, 5];
        let diff = OwnedDequeDiff {
            consumed: 2,
            removed_back: 1,
            pushed: vec![10, 20],
        };
        assert_eq!(diff.apply(prev), vec![3, 4, 10, 20]);
    }

    #[test]
    fn apply_diff_noop() {
        let prev = vec![1, 2, 3];
        let diff: OwnedDequeDiff<i32> = OwnedDequeDiff {
            consumed: 0,
            removed_back: 0,
            pushed: vec![],
        };
        assert!(diff.is_noop());
        assert_eq!(diff.apply(prev), vec![1, 2, 3]);
    }

    #[test]
    fn diff_then_apply_roundtrip() {
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3, 4, 5]);
        let origin = deque.clone();
        let prev = deque.as_slice().to_vec();

        deque.checkpoint();
        deque.consume(1);
        deque.pop();
        deque.push(10);
        deque.push(20);

        let expected = deque.as_slice().to_vec();
        let (_, diff) = deque.into_diff(&origin);
        let reconstructed = diff.apply(prev);
        assert_eq!(reconstructed, expected);
    }

    // ── Multi-turn roundtrip with checksum evolution ──────────

    #[test]
    fn multi_turn_roundtrip() {
        // Turn 0: start fresh
        let mut deque = TrackedDeque::<i32>::new();
        let origin0 = deque.clone();
        deque.checkpoint();
        deque.push(1);
        deque.push(2);

        let (squashed0, diff0) = deque.into_diff(&origin0);
        assert_eq!(diff0.pushed, vec![1, 2]);
        assert_eq!(squashed0.as_slice(), &[1, 2]);

        // Turn 1: use squashed0 as the stored deque
        let mut deque = squashed0.clone();
        deque.checkpoint();
        deque.push(3);
        deque.consume(1);

        let (squashed1, diff1) = deque.into_diff(&squashed0);
        assert_eq!(diff1.consumed, 1);
        assert_eq!(diff1.pushed, vec![3]);
        assert_eq!(squashed1.as_slice(), &[2, 3]);

        // Verify apply gives same result
        let reconstructed = diff1.apply(squashed0.as_slice().to_vec());
        assert_eq!(reconstructed, squashed1.as_slice());

        // Turn 2: use squashed1
        let mut deque = squashed1.clone();
        deque.checkpoint();
        deque.pop();
        deque.push(30);
        deque.push(40);

        let (squashed2, diff2) = deque.into_diff(&squashed1);
        assert_eq!(diff2.removed_back, 1);
        assert_eq!(diff2.pushed, vec![30, 40]);
        assert_eq!(squashed2.as_slice(), &[2, 30, 40]);

        let reconstructed = diff2.apply(squashed1.as_slice().to_vec());
        assert_eq!(reconstructed, squashed2.as_slice());

        // Checksums evolved each turn
        assert_ne!(origin0.checksum(), squashed0.checksum());
        assert_ne!(squashed0.checksum(), squashed1.checksum());
        assert_ne!(squashed1.checksum(), squashed2.checksum());
    }

    // ── Hijacking detection across turns ─────────────────────

    #[test]
    #[should_panic(expected = "checksum mismatch")]
    fn hijack_after_squash_panics() {
        // Turn 0
        let mut deque = TrackedDeque::from_vec(vec![1, 2, 3]);
        let origin = deque.clone();
        deque.checkpoint();
        deque.push(4);
        let (squashed, _) = deque.into_diff(&origin);

        // Turn 1: attacker creates a fake deque with same content
        let mut fake = TrackedDeque::from_vec(squashed.as_slice().to_vec());
        fake.checkpoint();
        fake.push(5);

        // This must panic — fake has different checksum than squashed
        fake.into_diff(&squashed);
    }
}
