//! `WordRuntime`'s values and its operations.

use acvus_extern::runtime::Runtime;

/// A value: its kind and its word, as the interpreter's `Value` on master.
#[derive(Debug, Clone, Copy)]
pub struct Word {}

/// The runtime: its interner, and nothing else; every value's storage is its
/// own allocation.
pub struct WordRuntime {}

// Runtime, Validate, Instances and CallsBack for WordRuntime: the fill.
#[allow(dead_code)]
fn implements_the_contract<R: Runtime>() {}
