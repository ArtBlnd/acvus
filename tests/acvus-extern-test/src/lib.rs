//! `WordRuntime`, a trusting runtime for the crossing's tests (runtime
//! contract §3): the contract's only kind of implementation, run as it is and
//! under `acvus_extern::adapter::Adapter`. It stands in for the interpreter
//! until the interpreter is ported, and keeps the interpreter's representation
//! on master (`acvus-interpreter/src/value.rs`, `acvus-extern/src/repr.rs`): a
//! value is a kind and a 64-bit word; a scalar lies in the word, and anything
//! larger is an allocation the word points at. It checks nothing, as the
//! interpreter checks nothing; what it answers to `Validate` it reads from the
//! kind, for the adapter alone. `Frame` is the run that calls through it.
//!
//! `shapes` holds shapes whose views are written by hand as `shape!` writes
//! them, standing in for it until it is ported.
//!
//! `unsafe` is written only in `boundary`, as in every crate of the workspace.

#![deny(unsafe_code)]

mod boundary;
mod frame;
mod runtime;
pub mod shapes;

pub use boundary::{Held, Pair, PartsMut, Tally, Word};
pub use frame::{Frame, Pass};
pub use runtime::WordRuntime;
