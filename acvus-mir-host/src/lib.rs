//! acvus-mir-extern — Declaring the world outside the pure solver.
//!
//! Rust `extern` functions and types are described here as type-level
//! specifications, so the MIR solver can read them. No execution/hosting —
//! that lives elsewhere.
//!
//! # Modules
//! - [`ity`]: Type identity bridge (Rust type ↔ MIR Ty), incl. Copy/Drop meta.

pub mod ity;

// ── Re-exports: public API ─────────────────────────────────────────

pub use ity::{Callable, Hosted, ITy, Inferrable, Monomorphize, Typeck};

// ── Re-exports: MIR types used in generated code ───────────────────

pub use acvus_mir::graph::types::{FnKind, Function};
pub use acvus_mir::ty::UserDefinedDecl;
pub use acvus_mir::ty::{Param, Ty};
pub use acvus_mir::ty::{Infer, InferTy, ParamTerm, Solver, lift_ty};
pub use acvus_mir::ty::{Poly, PolyBuilder, PolyParam, PolyTy, lift_to_poly};
pub use acvus_utils::{Interner, QualifiedRef};

// ── Re-export: proc macro ──────────────────────────────────────────

pub use acvus_mir_host_macro::{extern_fn, ExternType};
