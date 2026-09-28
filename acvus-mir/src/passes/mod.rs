//! passes (passes-structure): three dimensions, two frameworks, one
//! pipeline. Built beside the old `analysis` and `optimize` (D46), which stay
//! as a reference for behaviour and are never mixed in.
//!
//! - `domain`: values and the language's value operations. It knows no CFG.
//! - `analysis`: where values flow. Two engines (sparse, dense) and the
//!   manager that keeps and invalidates their facts.
//! - `optimize`: rewriting through `ir::cfg::Rewriter` only, reading facts.
//! - `pipeline`: the one entry; forms, costs, fixpoints and their bounds.

pub mod analysis;
pub mod domain;
pub mod optimize;
pub mod pipeline;
