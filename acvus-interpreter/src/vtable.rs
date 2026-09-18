//! The vtable a `Large` value carries in its allocation header: how to
//! drop and print the payload behind the pointer.
//!
//! A vtable is a constant of the type it describes (RFC-0048 §7), so the
//! address a header holds is a promoted constant, and a promoted constant
//! may have more than one address across codegen units.

use std::any::TypeId;
use std::fmt;
use std::ptr::NonNull;

#[repr(C)]
pub struct Header {
    pub vtable: &'static Vtable,
}

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
    Result,
    Fn,
    Handle,
}

pub type DropFn = unsafe fn(NonNull<Header>);
pub type DebugFn = unsafe fn(NonNull<Header>, &mut fmt::Formatter<'_>) -> fmt::Result;
/// `type_name` is not a `const fn` on 1.97.1, so a vtable that is itself a
/// constant carries the call that produces the name rather than the name.
pub type NameFn = fn() -> &'static str;

pub struct Vtable {
    pub type_id: TypeId,
    pub name: NameFn,
    pub composite: Option<Composite>,
    pub drop: DropFn,
    pub debug: Option<DebugFn>,
}

impl PartialEq for Vtable {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id
    }
}

impl Vtable {
    pub const fn drop_only<T>() -> Self
    where
        T: 'static,
    {
        Vtable {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>,
            composite: None,
            drop: drop_slot::<T>,
            debug: None,
        }
    }
}

pub trait HasVtable: 'static {
    const VTABLE: Vtable;
}

impl<T> HasVtable for T
where
    T: 'static,
{
    const VTABLE: Vtable = Vtable::drop_only::<T>();
}

/// # Safety
/// `p` is the header of a live `Box<Slot<T>>` that is not used again.
pub unsafe fn drop_slot<T>(p: NonNull<Header>) {
    drop(unsafe { Box::from_raw(p.cast::<Slot<T>>().as_ptr()) });
}
