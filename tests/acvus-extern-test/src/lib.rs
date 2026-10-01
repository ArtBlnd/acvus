//! `WordRuntime`, a runtime for the crossing's tests (runtime contract §3),
//! trusting its own typed reads as every runtime does. It stands in for the
//! interpreter
//! until the interpreter is ported, and keeps the interpreter's representation
//! on master (`acvus-interpreter/src/value.rs`, `acvus-extern/src/repr.rs`): a
//! value is a kind and a 64-bit word; a scalar lies in the word, and anything
//! larger is an allocation the word points at. It checks nothing, as the
//! interpreter checks nothing. `Frame` is the run that calls through it.
//!
//! `unsafe` is written only in `boundary`, as in every crate of the workspace.

#![deny(unsafe_code)]

mod boundary;
mod frame;
mod runtime;

pub use boundary::{Cursor, Held, Number, Pair, Tally, Word};
pub use frame::{Frame, Pass};
pub use runtime::WordRuntime;
