use std::marker::PhantomData;

// ── Global unique Id ─────────────────────────────────────────────────

/// Declares a globally-unique, opaque Id type with its own atomic counter.
///
/// `Id::new()` is the only way to create a valid Id — guaranteed unique
/// within the process lifetime. No way to extract or forge the inner value.
///
/// ```ignore
/// acvus_utils::declare_id!(pub NodeId);
///
/// let a = NodeId::new();
/// let b = NodeId::new();
/// assert_ne!(a, b);
/// ```
#[macro_export]
macro_rules! declare_id {
    ($vis:vis $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
        $vis struct $name(usize);

        impl $name {
            pub const UNKNOWN: $name = $name(usize::MAX);

            pub fn alloc() -> Self {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static NEXT: AtomicUsize = AtomicUsize::new(0);
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                assert!(id < usize::MAX, "Id space exhausted");
                $name(id)
            }

            /// Raw numeric index for display purposes only.
            pub fn index(self) -> usize {
                self.0
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                if *self == $name::UNKNOWN {
                    write!(f, "{}(UNKNOWN)", stringify!($name))
                } else {
                    write!(f, "{}({})", stringify!($name), self.0)
                }
            }
        }
    };
}

// ── Local indexed Id ─────────────────────────────────────────────────

/// Declares a local, sequential Id type usable as an index.
///
/// Unlike `declare_id!`, these Ids are not globally unique — they are
/// sequential within a single `LocalFactory` instance. The factory is
/// consumed to produce a `LocalVec` that can only be indexed by this Id type.
///
/// ```ignore
/// acvus_utils::declare_local_id!(pub ValueId);
///
/// let mut factory = LocalFactory::<ValueId>::new();
/// let v0 = factory.next();
/// let v1 = factory.next();
/// let mut vec = factory.into_vec(|| 0i32);
/// vec[v0] = 42;
/// vec[v1] = 99;
/// assert_eq!(vec[v0], 42);
/// ```
#[macro_export]
macro_rules! declare_local_id {
    ($vis:vis $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        $vis struct $name(u32);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl $crate::LocalIdOps for $name {
            fn from_raw(index: usize) -> Self { Self(index as u32) }
            fn to_raw(self) -> usize { self.0 as usize }
        }
    };
}

/// Sealed trait for local id types. Only implementable via `declare_local_id!`.
///
/// These methods are intentionally not meant for direct use — use
/// `LocalFactory` and `LocalVec` instead.
pub trait LocalIdOps: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    #[doc(hidden)]
    fn from_raw(index: usize) -> Self;
    #[doc(hidden)]
    fn to_raw(self) -> usize;
}

/// Sequential allocator for local ids. Consume with `into_vec` to get
/// an indexable collection.
#[derive(Debug, Clone)]
pub struct LocalFactory<I: LocalIdOps> {
    next: usize,
    _phantom: PhantomData<I>,
}

impl<I: LocalIdOps> LocalFactory<I> {
    pub fn new() -> Self {
        Self { next: 0, _phantom: PhantomData }
    }

    /// Allocate the next sequential id.
    pub fn next(&mut self) -> I {
        let id = I::from_raw(self.next);
        self.next += 1;
        id
    }

    /// How many ids have been allocated.
    pub fn len(&self) -> usize {
        self.next
    }

    /// Produce a `LocalVec` sized to hold all allocated ids, initialized with `default`.
    pub fn build_vec<V>(&self, default: impl Fn() -> V) -> LocalVec<I, V> {
        LocalVec {
            data: (0..self.next).map(|_| default()).collect(),
            _phantom: PhantomData,
        }
    }

    /// Produce a `LocalVec` initialized with `V::default()`.
    pub fn build_default<V: Default>(&self) -> LocalVec<I, V> {
        self.build_vec(V::default)
    }
}

/// Vec-like container indexed exclusively by a local id type.
/// Can only be created from a `LocalFactory`.
pub struct LocalVec<I: LocalIdOps, V> {
    data: Vec<V>,
    _phantom: PhantomData<I>,
}

impl<I: LocalIdOps, V> LocalVec<I, V> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.data.iter_mut()
    }
}

impl<I: LocalIdOps, V> std::ops::Index<I> for LocalVec<I, V> {
    type Output = V;
    fn index(&self, id: I) -> &V {
        &self.data[id.to_raw()]
    }
}

impl<I: LocalIdOps, V> std::ops::IndexMut<I> for LocalVec<I, V> {
    fn index_mut(&mut self, id: I) -> &mut V {
        &mut self.data[id.to_raw()]
    }
}

impl<I: LocalIdOps, V: std::fmt::Debug> std::fmt::Debug for LocalVec<I, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalVec")
            .field("len", &self.data.len())
            .finish()
    }
}

impl<I: LocalIdOps, V: Clone> Clone for LocalVec<I, V> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), _phantom: PhantomData }
    }
}
