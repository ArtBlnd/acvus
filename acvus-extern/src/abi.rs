//! The context persistence ABI: an extension type says how it dumps to bytes
//! and restores from them, so a context that holds it can persist across runs.

/// An extension type that persists in a context across runs (the context
/// model). The compiler and the interpreter never serialize an extension
/// payload themselves; only the type's author knows a sound dump and restore.
/// A `Regex` dumps its pattern and recompiles it on restore; a value with no
/// sound on-disk form implements neither, and cannot live in a persisted
/// context.
///
/// `V1` names this contract's version. No byte format is promised stable, and a
/// later contract is a new trait rather than an edit to this one.
///
/// # Safety
/// Implementing this trait promises that a `restore` of what `dump` produced
/// yields a value whose internal invariants hold, so that every later operation
/// on the restored value is sound. A dump and restore that do not round-trip
/// those invariants make the value's own code unsound; the compiler cannot
/// check the round-trip, so the implementer owns it.
pub unsafe trait AcvusContextAbiUnsafeV1: Sized {
    fn dump(&self) -> Vec<u8>;
    fn restore(bytes: &[u8]) -> Result<Self, ContextRestoreError>;
}

/// Why a `restore` could not read its bytes. Which dump produced them is the
/// caller's promise (see the trait's safety contract); these are the failures a
/// caller that kept that promise still meets, from bytes damaged in transit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextRestoreError {
    Truncated,
    Malformed { reason: String },
}
