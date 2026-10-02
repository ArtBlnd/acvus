//! The extern side of acvus: the contract a runtime implements, and the way
//! an extern fn reaches a runtime (`docs/extern-note.md`).

#![deny(unsafe_code)]

mod kind;
mod runtime;

pub use kind::{FnKind, OptionKind, ResultKind, ScalarKind, StringKind, TupleKind};
pub use runtime::{Arguments, ExternFn, Registry, Runtime};
