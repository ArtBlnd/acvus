//! The vtable a `Large` value carries in its allocation header: how to
//! drop and print the payload behind the pointer. The vtable, the header and
//! the slot are the interpreter's `repr`'s, which pairs each vtable's drop
//! and print with the allocation they read.
//!
//! A vtable is a constant of the type it describes (RFC-0048 rule 2), so the
//! address a header holds is a promoted constant, and a promoted constant
//! may have more than one address across codegen units.

pub use crate::repr::{Header, SlotVtable, Vtable};

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

/// `type_name` is not a `const fn` on 1.97.1, so a vtable that is itself a
/// constant carries the call that produces the name rather than the name.
pub type NameFn = fn() -> &'static str;

pub trait HasVtable: 'static + Sized {
    const VTABLE: SlotVtable<Self>;
}

impl<T> HasVtable for T
where
    T: 'static,
{
    const VTABLE: SlotVtable<T> = SlotVtable::drop_only();
}
