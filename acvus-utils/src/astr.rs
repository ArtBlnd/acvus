use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Global interner ID counter
// ---------------------------------------------------------------------------

static NEXT_INTERNER_ID: AtomicU32 = AtomicU32::new(1);

// ---------------------------------------------------------------------------
// Astr — interned string handle (8 bytes, Copy)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Astr {
    interner_id: u32,
    id: u32,
}

impl PartialEq for Astr {
    fn eq(&self, other: &Self) -> bool {
        self.interner_id == other.interner_id && self.id == other.id
    }
}

impl Eq for Astr {}

impl Hash for Astr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.interner_id.hash(state);
        self.id.hash(state);
    }
}

impl std::fmt::Debug for Astr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        THREAD_INTERNER.with(|c| {
            let opt = c.take();
            let result = match &opt {
                Some(interner) if interner.id == self.interner_id => {
                    write!(f, "{:?}", interner.resolve(*self))
                }
                _ => write!(f, "Astr({}:{})", self.interner_id, self.id),
            };
            c.set(opt);
            result
        })
    }
}

impl std::fmt::Display for Astr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        THREAD_INTERNER.with(|c| {
            let opt = c.take();
            let result = match &opt {
                Some(interner) if interner.id == self.interner_id => {
                    write!(f, "{}", interner.resolve(*self))
                }
                _ => write!(f, "#{}", self.id),
            };
            c.set(opt);
            result
        })
    }
}

// ---------------------------------------------------------------------------
// Serde support via thread_local interner
// ---------------------------------------------------------------------------

thread_local! {
    static THREAD_INTERNER: Cell<Option<Interner>> = const { Cell::new(None) };
}

/// Run a closure with the given interner set as thread-local context.
/// Used by Astr's Display, Serialize, and Deserialize impls.
pub fn with_interner_context<T>(interner: &Interner, f: impl FnOnce() -> T) -> T {
    THREAD_INTERNER.with(|c| c.set(Some(interner.clone())));
    let result = f();
    THREAD_INTERNER.with(|c| c.set(None));
    result
}

/// Set the thread-local interner context (for use across `.await` points).
/// Call `clear_thread_interner()` when done.
pub fn set_thread_interner(interner: &Interner) {
    THREAD_INTERNER.with(|c| c.set(Some(interner.clone())));
}

/// Clear the thread-local interner context.
pub fn clear_thread_interner() {
    THREAD_INTERNER.with(|c| c.set(None));
}

pub fn get_thread_interner() -> Option<Interner> {
    THREAD_INTERNER.with(|c| {
        let i = c.take();
        let cloned = i.clone();
        c.set(i);
        cloned
    })
}

impl Serialize for Astr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let interner = get_thread_interner()
            .expect("Astr::serialize requires with_interner_context()");
        let s = interner.resolve(*self);
        s.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Astr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let interner = get_thread_interner()
            .expect("Astr::deserialize requires with_interner_context()");
        Ok(interner.intern(&s))
    }
}

// ---------------------------------------------------------------------------
// Shard — single partition of the interner
// ---------------------------------------------------------------------------

const SHARD_COUNT: usize = 64;

struct Shard {
    strings: Vec<String>,
    lookup: HashMap<String, u32>,
}

impl Shard {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            lookup: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ShardedInner
// ---------------------------------------------------------------------------

struct ShardedInner {
    shards: Vec<RwLock<Shard>>,
}

impl ShardedInner {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(SHARD_COUNT);
        for _ in 0..SHARD_COUNT {
            shards.push(RwLock::new(Shard::new()));
        }
        Self { shards }
    }

    fn shard_for(s: &str) -> usize {
        // FNV-1a style hash for shard selection
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % SHARD_COUNT
    }

    /// Encode (shard_index, local_index) into a single u32 id.
    /// Layout: upper bits = shard, lower bits = local index.
    /// Max local index per shard: 2^26 = 67M entries.
    fn encode_id(shard_idx: usize, local_idx: u32) -> u32 {
        debug_assert!(shard_idx < SHARD_COUNT);
        debug_assert!(local_idx < (1 << 26));
        ((shard_idx as u32) << 26) | local_idx
    }

    fn decode_id(id: u32) -> (usize, u32) {
        let shard_idx = (id >> 26) as usize;
        let local_idx = id & ((1 << 26) - 1);
        (shard_idx, local_idx)
    }
}

// ---------------------------------------------------------------------------
// Interner — Arc-based, cloneable, Send + Sync
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Interner {
    id: u32,
    inner: Arc<ShardedInner>,
}

impl Interner {
    pub fn new() -> Self {
        let id = NEXT_INTERNER_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id,
            inner: Arc::new(ShardedInner::new()),
        }
    }

    /// Intern a string, returning its Astr handle.
    pub fn intern(&self, s: &str) -> Astr {
        let shard_idx = ShardedInner::shard_for(s);
        let shard = &self.inner.shards[shard_idx];

        // Fast path: read lock
        {
            let guard = shard.read();
            if let Some(&local_id) = guard.lookup.get(s) {
                return Astr {
                    interner_id: self.id,
                    id: ShardedInner::encode_id(shard_idx, local_id),
                };
            }
        }

        // Slow path: write lock
        let mut guard = shard.write();
        // Double-check after acquiring write lock
        if let Some(&local_id) = guard.lookup.get(s) {
            return Astr {
                interner_id: self.id,
                id: ShardedInner::encode_id(shard_idx, local_id),
            };
        }

        let local_id = guard.strings.len() as u32;
        guard.strings.push(s.to_owned());
        guard.lookup.insert(s.to_owned(), local_id);

        Astr {
            interner_id: self.id,
            id: ShardedInner::encode_id(shard_idx, local_id),
        }
    }

    /// Resolve an Astr back to a string slice.
    ///
    /// # Panics
    /// Panics if the Astr was created by a different Interner.
    pub fn resolve(&self, astr: Astr) -> &str {
        assert_eq!(
            astr.interner_id, self.id,
            "Astr interner mismatch: expected {}, got {}",
            self.id, astr.interner_id
        );
        let (shard_idx, local_idx) = ShardedInner::decode_id(astr.id);
        let shard = &self.inner.shards[shard_idx];
        let guard = shard.read();
        let s: &str = &guard.strings[local_idx as usize];
        // SAFETY: the string is never removed from the shard,
        // and the Arc keeps the inner alive as long as any Interner clone lives.
        // We extend the lifetime from the guard to self.
        unsafe { &*(s as *const str) }
    }

    /// Get the interner's unique id.
    pub fn id(&self) -> u32 {
        self.id
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_and_resolve() {
        let interner = Interner::new();
        let a = interner.intern("hello");
        let b = interner.intern("world");
        let c = interner.intern("hello");

        assert_eq!(a, c);
        assert_ne!(a, b);
        assert_eq!(interner.resolve(a), "hello");
        assert_eq!(interner.resolve(b), "world");
    }

    #[test]
    fn copy_semantics() {
        let interner = Interner::new();
        let a = interner.intern("test");
        let b = a; // Copy
        assert_eq!(a, b);
        assert_eq!(interner.resolve(a), interner.resolve(b));
    }

    #[test]
    fn hash_consistency() {
        use std::collections::HashMap;
        let interner = Interner::new();
        let a = interner.intern("key");
        let mut map = HashMap::new();
        map.insert(a, 42);
        assert_eq!(map.get(&a), Some(&42));
    }

    #[test]
    fn different_interners_not_equal() {
        let i1 = Interner::new();
        let i2 = Interner::new();
        let a = i1.intern("same");
        let b = i2.intern("same");
        assert_ne!(a, b);
    }

    #[test]
    #[should_panic(expected = "Astr interner mismatch")]
    fn resolve_wrong_interner_panics() {
        let i1 = Interner::new();
        let i2 = Interner::new();
        let a = i1.intern("hello");
        i2.resolve(a);
    }

    #[test]
    fn clone_interner_shares_data() {
        let i1 = Interner::new();
        let a = i1.intern("shared");
        let i2 = i1.clone();
        assert_eq!(i2.resolve(a), "shared");
        let b = i2.intern("new");
        assert_eq!(i1.resolve(b), "new");
    }

    #[test]
    fn serde_roundtrip() {
        let interner = Interner::new();
        let a = interner.intern("hello");

        let json = with_interner_context(&interner, || {
            serde_json::to_string(&a).unwrap()
        });
        assert_eq!(json, "\"hello\"");

        let b: Astr = with_interner_context(&interner, || {
            serde_json::from_str(&json).unwrap()
        });
        assert_eq!(a, b);
    }

    #[test]
    fn size_is_8_bytes() {
        assert_eq!(std::mem::size_of::<Astr>(), 8);
    }
}
