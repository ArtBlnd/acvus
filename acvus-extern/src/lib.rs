//! The extern side of acvus: the contract a runtime implements, and the way
//! an extern fn reaches a runtime (`docs/extern-note.md`).

#![deny(unsafe_code)]

mod crossing;
mod kind;
mod runtime;

pub use acvus_extern_macro::Structural;
pub use crossing::Crossing;
pub use kind::{EnumKind, FnKind, ObjectKind, OptionKind, ResultKind, ScalarKind, StringKind, TupleKind};
pub use runtime::{Arguments, ExternFn, Registry, Runtime};
