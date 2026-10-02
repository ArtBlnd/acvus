//! The extern side of acvus: the contract a runtime implements, and the way
//! an extern fn reaches a runtime (`docs/extern-note.md`).

#![deny(unsafe_code)]

mod crossing;
mod kind;
mod repr;
mod runtime;

pub use acvus_extern_macro::Structural;
pub use crossing::Crossing;
pub use kind::{
    EnumKind, FnKind, ObjectKind, OpaqueKind, OptionKind, RefKind, ResultKind, ScalarKind, StringKind, TupleKind,
};
pub use repr::{Boxed, StoredOpaque, StoredRef, Words};
pub use runtime::{Arguments, ExternFn, Registry, Runtime, Stores};
