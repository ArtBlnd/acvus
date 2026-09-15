//! The vtable a `Large` value carries in its allocation header: how to
//! drop, clone, and print the payload behind the pointer. The interpreter's
//! own composites are process-wide statics; an extension type erased for
//! the first time registers a drop-only vtable through the `VtableRegistry`, which
//! leaks it for the life of the process.

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::Mutex;

/// The first word of every `Large` allocation.
#[repr(C)]
pub struct Header {
    pub vtable: &'static Vtable,
}

/// A `Large` payload of type `T` behind its header.
#[repr(C)]
pub struct Slot<T> {
    pub header: Header,
    pub value: T,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Composite {
    String,
    Array,
    Tuple,
    Object,
    Variant,
    Fn,
    Handle,
}

pub type DropFn = unsafe fn(NonNull<Header>);
pub type CloneFn = unsafe fn(NonNull<Header>) -> NonNull<Header>;
pub type DebugFn = unsafe fn(NonNull<Header>, &mut fmt::Formatter<'_>) -> fmt::Result;

pub struct Vtable {
    pub type_id: TypeId,
    pub name: &'static str,
    pub composite: Option<Composite>,
    pub drop: DropFn,
    pub clone: Option<CloneFn>,
    pub debug: Option<DebugFn>,
}

impl Vtable {
    pub fn drop_only<T: 'static>(name: &'static str) -> Self {
        Vtable {
            type_id: TypeId::of::<T>(),
            name,
            composite: None,
            drop: drop_slot::<T>,
            clone: None,
            debug: None,
        }
    }
}

/// # Safety
/// `p` is the header of a live `Box<Slot<T>>` that is not used again.
pub unsafe fn drop_slot<T>(p: NonNull<Header>) {
    drop(unsafe { Box::from_raw(p.cast::<Slot<T>>().as_ptr()) });
}

/// Witnesses of extension types, one per Rust type, leaked on registration.
#[derive(Default)]
pub struct VtableRegistry {
    by_type: Mutex<HashMap<TypeId, &'static Vtable>>,
}

impl VtableRegistry {
    pub fn register(&self, vtable: Vtable) -> &'static Vtable {
        let mut by_type = self.by_type.lock().expect("table poisoned");
        if let Some(&existing) = by_type.get(&vtable.type_id) {
            return existing;
        }
        let leaked: &'static Vtable = Box::leak(Box::new(vtable));
        by_type.insert(leaked.type_id, leaked);
        leaked
    }

    pub fn vtable_of<T: 'static>(&self) -> &'static Vtable {
        if let Some(&existing) = self
            .by_type
            .lock()
            .expect("table poisoned")
            .get(&TypeId::of::<T>())
        {
            return existing;
        }
        self.register(Vtable::drop_only::<T>(std::any::type_name::<T>()))
    }
}
