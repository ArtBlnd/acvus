use rustc_hash::FxHashMap;

// ── BlobHash ────────────────────────────────────────────────────────

/// 32-byte content hash (blake3).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlobHash([u8; 32]);

impl BlobHash {
    /// Compute the hash of `data`.
    pub fn of(data: &[u8]) -> Self {
        Self(*blake3::hash(data).as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl std::fmt::Debug for BlobHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlobHash(")?;
        for b in &self.0[..4] {
            write!(f, "{b:02x}")?;
        }
        write!(f, "..)")
    }
}

impl std::fmt::Display for BlobHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

// ── BlobStore trait ─────────────────────────────────────────────────

/// Content-addressed blob store with named refs and CAS.
///
/// Two independent namespaces:
/// - **Blobs**: immutable, addressed by content hash. `put` is idempotent.
/// - **Refs**: mutable named pointers to blob hashes. `ref_cas` is atomic.
///
/// The store does NOT enforce referential integrity — a ref may point to
/// a removed blob. Integrity is the caller's responsibility.
#[trait_variant::make(Send)]
pub trait BlobStore: Sync {
    // ── Blob operations (content-addressed) ──

    /// Store a blob. Returns its content hash. Idempotent: same content = same hash.
    async fn put(&mut self, data: Vec<u8>) -> BlobHash;

    /// Retrieve a blob by hash. Returns `None` if not found.
    async fn get(&self, hash: &BlobHash) -> Option<Vec<u8>>;

    /// Remove a blob by hash. No-op if not found.
    async fn remove(&mut self, hash: &BlobHash);

    // ── Ref operations (named pointers with CAS) ──

    /// Get the current hash that a named ref points to.
    async fn ref_get(&self, name: &str) -> Option<BlobHash>;

    /// Atomic compare-and-swap on a named ref.
    ///
    /// - `expected = None`: create — fails if ref already exists.
    /// - `expected = Some(h)`: update — fails if current hash ≠ `h`.
    ///
    /// Returns `Ok(())` on success.
    /// Returns `Err(actual)` on conflict, where `actual` is the current value.
    async fn ref_cas(
        &mut self,
        name: &str,
        expected: Option<BlobHash>,
        new: BlobHash,
    ) -> Result<(), Option<BlobHash>>;

    /// Remove a named ref. No-op if not found.
    async fn ref_remove(&mut self, name: &str);

    // ── Batch operations ──

    /// Batch put. Returns hashes in the same order as input.
    async fn batch_put(&mut self, blobs: Vec<Vec<u8>>) -> Vec<BlobHash>;

    /// Batch get. Returns values in the same order as input. `None` for missing blobs.
    async fn batch_get(&self, hashes: &[BlobHash]) -> Vec<Option<Vec<u8>>>;

    /// Batch remove.
    async fn batch_remove(&mut self, hashes: Vec<BlobHash>);
}

// ── MemBlobStore ────────────────────────────────────────────────────

/// In-memory blob store for testing.
#[derive(Debug, Default)]
pub struct MemBlobStore {
    blobs: FxHashMap<BlobHash, Vec<u8>>,
    refs: FxHashMap<String, BlobHash>,
}

impl MemBlobStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of blobs currently stored.
    pub fn blob_count(&self) -> usize {
        self.blobs.len()
    }

    /// Number of refs currently stored.
    pub fn ref_count(&self) -> usize {
        self.refs.len()
    }

    /// All blob hashes currently stored.
    pub fn blob_hashes(&self) -> Vec<BlobHash> {
        self.blobs.keys().copied().collect()
    }
}

impl BlobStore for MemBlobStore {
    async fn put(&mut self, data: Vec<u8>) -> BlobHash {
        let hash = BlobHash::of(&data);
        self.blobs.entry(hash).or_insert(data);
        hash
    }

    async fn get(&self, hash: &BlobHash) -> Option<Vec<u8>> {
        self.blobs.get(hash).cloned()
    }

    async fn remove(&mut self, hash: &BlobHash) {
        self.blobs.remove(hash);
    }

    async fn ref_get(&self, name: &str) -> Option<BlobHash> {
        self.refs.get(name).copied()
    }

    async fn ref_cas(
        &mut self,
        name: &str,
        expected: Option<BlobHash>,
        new: BlobHash,
    ) -> Result<(), Option<BlobHash>> {
        let current = self.refs.get(name).copied();
        if current == expected {
            self.refs.insert(name.to_string(), new);
            Ok(())
        } else {
            Err(current)
        }
    }

    async fn ref_remove(&mut self, name: &str) {
        self.refs.remove(name);
    }

    async fn batch_put(&mut self, blobs: Vec<Vec<u8>>) -> Vec<BlobHash> {
        blobs
            .into_iter()
            .map(|data| {
                let hash = BlobHash::of(&data);
                self.blobs.entry(hash).or_insert(data);
                hash
            })
            .collect()
    }

    async fn batch_get(&self, hashes: &[BlobHash]) -> Vec<Option<Vec<u8>>> {
        hashes.iter().map(|h| self.blobs.get(h).cloned()).collect()
    }

    async fn batch_remove(&mut self, hashes: Vec<BlobHash>) {
        for h in hashes {
            self.blobs.remove(&h);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Blob: basic put/get/remove ──

    #[tokio::test]
    async fn put_and_get() {
        let mut s = MemBlobStore::new();
        let data = b"hello world".to_vec();
        let hash = s.put(data.clone()).await;
        assert_eq!(s.get(&hash).await, Some(data));
    }

    #[tokio::test]
    async fn put_idempotent_same_hash() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"abc".to_vec()).await;
        let h2 = s.put(b"abc".to_vec()).await;
        assert_eq!(h1, h2);
        assert_eq!(s.blob_count(), 1);
    }

    #[tokio::test]
    async fn put_different_content_different_hash() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"aaa".to_vec()).await;
        let h2 = s.put(b"bbb".to_vec()).await;
        assert_ne!(h1, h2);
        assert_eq!(s.blob_count(), 2);
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let s = MemBlobStore::new();
        let fake = BlobHash::of(b"nonexistent");
        assert_eq!(s.get(&fake).await, None);
    }

    #[tokio::test]
    async fn remove_existing() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"data".to_vec()).await;
        s.remove(&hash).await;
        assert_eq!(s.get(&hash).await, None);
        assert_eq!(s.blob_count(), 0);
    }

    #[tokio::test]
    async fn remove_missing_noop() {
        let mut s = MemBlobStore::new();
        let fake = BlobHash::of(b"ghost");
        s.remove(&fake).await; // should not panic
        assert_eq!(s.blob_count(), 0);
    }

    #[tokio::test]
    async fn put_after_remove_restores() {
        let mut s = MemBlobStore::new();
        let data = b"comeback".to_vec();
        let h1 = s.put(data.clone()).await;
        s.remove(&h1).await;
        assert_eq!(s.get(&h1).await, None);
        let h2 = s.put(data.clone()).await;
        assert_eq!(h1, h2);
        assert_eq!(s.get(&h2).await, Some(data));
    }

    #[tokio::test]
    async fn empty_blob() {
        let mut s = MemBlobStore::new();
        let hash = s.put(vec![]).await;
        assert_eq!(s.get(&hash).await, Some(vec![]));
    }

    #[tokio::test]
    async fn large_blob() {
        let mut s = MemBlobStore::new();
        let data = vec![0xABu8; 1_000_000];
        let hash = s.put(data.clone()).await;
        assert_eq!(s.get(&hash).await, Some(data));
    }

    // ── Blob: hash determinism ──

    #[test]
    fn hash_deterministic() {
        let h1 = BlobHash::of(b"deterministic");
        let h2 = BlobHash::of(b"deterministic");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_single_bit_difference() {
        let h1 = BlobHash::of(&[0x00]);
        let h2 = BlobHash::of(&[0x01]);
        assert_ne!(h1, h2);
    }

    // ── Ref: basic operations ──

    #[tokio::test]
    async fn ref_get_missing_returns_none() {
        let s = MemBlobStore::new();
        assert_eq!(s.ref_get("head").await, None);
    }

    #[tokio::test]
    async fn ref_cas_create() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"data".to_vec()).await;
        assert!(s.ref_cas("head", None, hash).await.is_ok());
        assert_eq!(s.ref_get("head").await, Some(hash));
    }

    #[tokio::test]
    async fn ref_cas_update() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"v1".to_vec()).await;
        let h2 = s.put(b"v2".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        assert!(s.ref_cas("head", Some(h1), h2).await.is_ok());
        assert_eq!(s.ref_get("head").await, Some(h2));
    }

    #[tokio::test]
    async fn ref_cas_conflict_on_create() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"v1".to_vec()).await;
        let h2 = s.put(b"v2".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        // Try to create again — should fail
        let err = s.ref_cas("head", None, h2).await.unwrap_err();
        assert_eq!(err, Some(h1));
        // Original untouched
        assert_eq!(s.ref_get("head").await, Some(h1));
    }

    #[tokio::test]
    async fn ref_cas_conflict_on_update_wrong_expected() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"v1".to_vec()).await;
        let h2 = s.put(b"v2".to_vec()).await;
        let h3 = s.put(b"v3".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        // expected h2, actual h1 → conflict
        let err = s.ref_cas("head", Some(h2), h3).await.unwrap_err();
        assert_eq!(err, Some(h1));
        assert_eq!(s.ref_get("head").await, Some(h1));
    }

    #[tokio::test]
    async fn ref_cas_conflict_on_update_missing_ref() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"v1".to_vec()).await;
        let h2 = s.put(b"v2".to_vec()).await;
        // expected Some(h1) but ref doesn't exist → conflict
        let err = s.ref_cas("head", Some(h1), h2).await.unwrap_err();
        assert_eq!(err, None);
    }

    #[tokio::test]
    async fn ref_remove_existing() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"data".to_vec()).await;
        s.ref_cas("head", None, hash).await.unwrap();
        s.ref_remove("head").await;
        assert_eq!(s.ref_get("head").await, None);
        assert_eq!(s.ref_count(), 0);
    }

    #[tokio::test]
    async fn ref_remove_missing_noop() {
        let mut s = MemBlobStore::new();
        s.ref_remove("ghost").await; // no panic
        assert_eq!(s.ref_count(), 0);
    }

    #[tokio::test]
    async fn ref_cas_create_after_remove() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"v1".to_vec()).await;
        let h2 = s.put(b"v2".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        s.ref_remove("head").await;
        // Now create again with None expected
        assert!(s.ref_cas("head", None, h2).await.is_ok());
        assert_eq!(s.ref_get("head").await, Some(h2));
    }

    // ── Ref: multiple independent refs ──

    #[tokio::test]
    async fn multiple_refs_independent() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"a".to_vec()).await;
        let h2 = s.put(b"b".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        s.ref_cas("branch", None, h2).await.unwrap();
        assert_eq!(s.ref_get("head").await, Some(h1));
        assert_eq!(s.ref_get("branch").await, Some(h2));
        assert_eq!(s.ref_count(), 2);
    }

    #[tokio::test]
    async fn ref_cas_on_one_does_not_affect_other() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"a".to_vec()).await;
        let h2 = s.put(b"b".to_vec()).await;
        let h3 = s.put(b"c".to_vec()).await;
        s.ref_cas("head", None, h1).await.unwrap();
        s.ref_cas("branch", None, h2).await.unwrap();
        // Update head, branch untouched
        s.ref_cas("head", Some(h1), h3).await.unwrap();
        assert_eq!(s.ref_get("head").await, Some(h3));
        assert_eq!(s.ref_get("branch").await, Some(h2));
    }

    // ── Ref: two refs can point to same hash ──

    #[tokio::test]
    async fn two_refs_same_hash() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"shared".to_vec()).await;
        s.ref_cas("a", None, hash).await.unwrap();
        s.ref_cas("b", None, hash).await.unwrap();
        assert_eq!(s.ref_get("a").await, Some(hash));
        assert_eq!(s.ref_get("b").await, Some(hash));
    }

    // ── Cross-concern: blob and ref independence ──

    #[tokio::test]
    async fn removing_blob_does_not_remove_ref() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"data".to_vec()).await;
        s.ref_cas("head", None, hash).await.unwrap();
        s.remove(&hash).await;
        // Ref still exists (dangling — that's fine, GC is caller's job)
        assert_eq!(s.ref_get("head").await, Some(hash));
        // But blob is gone
        assert_eq!(s.get(&hash).await, None);
    }

    #[tokio::test]
    async fn removing_ref_does_not_remove_blob() {
        let mut s = MemBlobStore::new();
        let hash = s.put(b"data".to_vec()).await;
        s.ref_cas("head", None, hash).await.unwrap();
        s.ref_remove("head").await;
        // Blob still exists
        assert_eq!(s.get(&hash).await, Some(b"data".to_vec()));
    }

    // ── Batch: put ──

    #[tokio::test]
    async fn batch_put_multiple() {
        let mut s = MemBlobStore::new();
        let hashes = s.batch_put(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]).await;
        assert_eq!(hashes.len(), 3);
        for (h, data) in hashes.iter().zip([b"a", b"b", b"c"]) {
            assert_eq!(s.get(h).await, Some(data.to_vec()));
        }
    }

    #[tokio::test]
    async fn batch_put_with_duplicates() {
        let mut s = MemBlobStore::new();
        let hashes = s.batch_put(vec![b"same".to_vec(), b"same".to_vec()]).await;
        assert_eq!(hashes[0], hashes[1]);
        assert_eq!(s.blob_count(), 1);
    }

    #[tokio::test]
    async fn batch_put_empty() {
        let mut s = MemBlobStore::new();
        let hashes = s.batch_put(vec![]).await;
        assert!(hashes.is_empty());
        assert_eq!(s.blob_count(), 0);
    }

    // ── Batch: get ──

    #[tokio::test]
    async fn batch_get_multiple() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"a".to_vec()).await;
        let h2 = s.put(b"b".to_vec()).await;
        let h3 = s.put(b"c".to_vec()).await;
        let results = s.batch_get(&[h1, h2, h3]).await;
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(b"a".to_vec()));
        assert_eq!(results[1], Some(b"b".to_vec()));
        assert_eq!(results[2], Some(b"c".to_vec()));
    }

    #[tokio::test]
    async fn batch_get_with_missing() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"exists".to_vec()).await;
        let fake = BlobHash::of(b"missing");
        let results = s.batch_get(&[h1, fake]).await;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Some(b"exists".to_vec()));
        assert_eq!(results[1], None);
    }

    #[tokio::test]
    async fn batch_get_empty() {
        let s = MemBlobStore::new();
        let results = s.batch_get(&[]).await;
        assert!(results.is_empty());
    }

    // ── Batch: remove ──

    #[tokio::test]
    async fn batch_remove_multiple() {
        let mut s = MemBlobStore::new();
        let h1 = s.put(b"a".to_vec()).await;
        let h2 = s.put(b"b".to_vec()).await;
        let h3 = s.put(b"c".to_vec()).await;
        s.batch_remove(vec![h1, h3]).await;
        assert_eq!(s.get(&h1).await, None);
        assert_eq!(s.get(&h2).await, Some(b"b".to_vec()));
        assert_eq!(s.get(&h3).await, None);
    }

    #[tokio::test]
    async fn batch_remove_empty() {
        let mut s = MemBlobStore::new();
        s.put(b"survive".to_vec()).await;
        s.batch_remove(vec![]).await;
        assert_eq!(s.blob_count(), 1);
    }

    #[tokio::test]
    async fn batch_remove_with_missing() {
        let mut s = MemBlobStore::new();
        let real = s.put(b"real".to_vec()).await;
        let fake = BlobHash::of(b"fake");
        s.batch_remove(vec![real, fake]).await; // no panic
        assert_eq!(s.blob_count(), 0);
    }

    // ── BlobHash: Display / Debug ──

    #[test]
    fn hash_display_is_64_hex_chars() {
        let h = BlobHash::of(b"test");
        let display = format!("{h}");
        assert_eq!(display.len(), 64);
        assert!(display.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn hash_debug_is_short() {
        let h = BlobHash::of(b"test");
        let debug = format!("{h:?}");
        assert!(debug.starts_with("BlobHash("));
        assert!(debug.ends_with("..)"));
    }

    // ── BlobHash: from_bytes round-trip ──

    #[test]
    fn hash_from_bytes_round_trip() {
        let h = BlobHash::of(b"round-trip");
        let bytes = *h.as_bytes();
        let h2 = BlobHash::from_bytes(bytes);
        assert_eq!(h, h2);
    }

    // ── Stress: many keys don't collide ──

    #[tokio::test]
    async fn many_unique_blobs() {
        let mut s = MemBlobStore::new();
        let n = 10_000;
        let mut hashes = Vec::with_capacity(n);
        for i in 0..n {
            hashes.push(s.put(format!("blob-{i}").into_bytes()).await);
        }

        // All hashes unique
        let unique: std::collections::HashSet<BlobHash> = hashes.iter().copied().collect();
        assert_eq!(unique.len(), n);

        // All retrievable
        for (i, h) in hashes.iter().enumerate() {
            assert_eq!(s.get(h).await, Some(format!("blob-{i}").into_bytes()));
        }
    }

    // ── Concurrent CAS pattern: simulate two writers ──

    #[tokio::test]
    async fn cas_serializes_concurrent_writers() {
        let mut s = MemBlobStore::new();
        let v1 = s.put(b"v1".to_vec()).await;
        let v2 = s.put(b"v2".to_vec()).await;
        let v3 = s.put(b"v3".to_vec()).await;

        // Both writers read current state: None
        let writer_a_sees = s.ref_get("head").await; // None
        let writer_b_sees = s.ref_get("head").await; // None

        // Writer A wins
        assert!(s.ref_cas("head", writer_a_sees, v1).await.is_ok());

        // Writer B fails — stale expected
        let err = s.ref_cas("head", writer_b_sees, v2).await.unwrap_err();
        assert_eq!(err, Some(v1));

        // Writer B retries with fresh read
        let fresh = s.ref_get("head").await;
        assert!(s.ref_cas("head", fresh, v3).await.is_ok());
        assert_eq!(s.ref_get("head").await, Some(v3));
    }
}
