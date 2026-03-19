//! Validation passes for MIR modules.
//!
//! Re-exports from `acvus_mir::validate` for convenience when using the pass layer.

pub use acvus_mir::validate::{ValidationError, ValidationErrorKind, validate};
