use std::marker::PhantomData;

// -- Global unique Id -------------------------------------------------

/// Declares a globally-unique, opaque Id type with its own atomic counter.
///
/// `Id::new()` is the only way to create a valid Id - guaranteed unique
/// within the process lifetime. No way to extract or forge the inner value.
///
/// Internally stores index + 1 as `NonZero<usize>` for niche optimization
/// (`Option<Id>` is the same size as `Id`).
///
/// ```ignore
/// acvus_utils::declare_id!(pub NodeId);
///
/// let a = NodeId::alloc();
/// let b = NodeId::alloc();
/// assert_ne!(a, b);
/// ```
#[macro_export]
macro_rules! declare_id {
    ($vis:vis $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
        $vis struct $name(std::num::NonZero<usize>);

        impl $name {
            pub fn alloc() -> Self {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static NEXT: AtomicUsize = AtomicUsize::new(0);
                let id = NEXT.fetch_add(1, Ordering::Relaxed);
                // SAFETY: id + 1 is always >= 1 (id < usize::MAX by assertion).
                assert!(id < usize::MAX, "Id space exhausted");
                $name(unsafe { std::num::NonZero::new_unchecked(id + 1) })
            }

            /// Raw numeric index for display purposes only.
            pub fn index(self) -> usize {
                self.0.get() - 1
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0.get() - 1)
            }
        }
    };
}

// -- Local indexed Id -------------------------------------------------

/// Declares a local, sequential Id type usable as an index.
///
/// Unlike `declare_id!`, these Ids are not globally unique - they are
/// sequential within a single `LocalFactory` instance. The factory is
/// consumed to produce a `LocalVec` that can only be indexed by this Id type.
///
/// Internally stores index + 1 as `NonZero<u32>` for niche optimization
/// (`Option<Id>` is the same size as `Id`), so the indices it holds are
/// `0..=u32::MAX - 1`. `LocalIdOps::try_from_raw` refuses an index outside
/// them; `LocalIdOps::from_raw` panics on one.
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
        $vis struct $name(std::num::NonZero<u32>);

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0.get() - 1)
            }
        }

        impl $crate::LocalIdOps for $name {
            fn try_from_raw(index: usize) -> Option<Self> {
                let index = u32::try_from(index).ok()?;
                std::num::NonZero::<u32>::MIN.checked_add(index).map(Self)
            }
            fn from_raw(index: usize) -> Self {
                match <Self as $crate::LocalIdOps>::try_from_raw(index) {
                    Some(id) => id,
                    None => panic!(
                        "{} index {} exceeds the local id space (at most u32::MAX - 1)",
                        stringify!($name),
                        index,
                    ),
                }
            }
            fn to_raw(self) -> usize { (self.0.get() - 1) as usize }
        }
    };
}

/// Sealed trait for local id types. Only implementable via `declare_local_id!`.
///
/// These methods are intentionally not meant for direct use - use
/// `LocalFactory` and `LocalVec` instead.
pub trait LocalIdOps: Copy + Eq + std::hash::Hash + std::fmt::Debug {
    /// The id of `index`, or `None` when `index` is above `u32::MAX - 1`.
    /// For an index read from outside the process, such as a recorded
    /// identity.
    #[doc(hidden)]
    fn try_from_raw(index: usize) -> Option<Self>;
    /// The id of `index`; panics when `index` is above `u32::MAX - 1`. For
    /// an index the process produced itself: a factory's next id, or an
    /// index below a factory's `len`.
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

impl<I: LocalIdOps> Default for LocalFactory<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: LocalIdOps> LocalFactory<I> {
    pub fn new() -> Self {
        Self {
            next: 0,
            _phantom: PhantomData,
        }
    }

    /// Allocate the next sequential id. Panics once the id type's space
    /// (`u32::MAX` ids) is exhausted.
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
        Self {
            data: self.data.clone(),
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LocalIdOps;

    crate::declare_local_id!(TestId);

    #[test]
    fn the_last_index_below_the_boundary_is_an_id() {
        let last = (u32::MAX - 1) as usize;
        let id = TestId::try_from_raw(last).expect("u32::MAX - 1 is in the space");
        assert_eq!(id.to_raw(), last);
        assert_eq!(TestId::from_raw(last).to_raw(), last);
    }

    #[test]
    fn the_boundary_index_is_refused() {
        assert_eq!(TestId::try_from_raw(u32::MAX as usize), None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn an_index_above_u32_is_refused() {
        assert_eq!(TestId::try_from_raw(u32::MAX as usize + 1), None);
        assert_eq!(TestId::try_from_raw(1 << 32), None);
    }

    #[test]
    #[should_panic(expected = "TestId index 4294967295 exceeds the local id space")]
    fn from_raw_panics_at_the_boundary() {
        TestId::from_raw(u32::MAX as usize);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    #[should_panic(expected = "TestId index 4294967296 exceeds the local id space")]
    fn from_raw_panics_above_u32_rather_than_colliding() {
        TestId::from_raw(1 << 32);
    }
}
