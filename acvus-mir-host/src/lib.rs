//! acvus-mir-host — The contract for hosting MIR execution.
//!
//! # Modules
//! - [`ity`]: Type identity bridge (Rust type ↔ MIR Ty).
//! - [`scope`]: Handler's interface to interpreter state.
//! - [`registrar`]: Registration + autoref specialization.
//! - [`error`]: Host boundary errors.

pub mod error;
pub mod ity;
pub mod registrar;
pub mod scope;
pub mod testing;

// ── Re-exports: public API ─────────────────────────────────────────

pub use error::{ExternFnFuture, HostError};
pub use ity::{Callable, Eff, EffectParam, Hosted, ITy, Inferrable, Monomorphize, Typeck};
pub use registrar::{
    AutoregClone, AutoregCopy, AutoregMove, ExternFnDef, Registrar, TypeMarker,
};
pub use scope::{CallArgs, Scope};

// ── Re-exports: MIR types used in generated code ───────────────────

pub use acvus_mir::graph::types::{FnKind, Function};
pub use acvus_mir::ty::UserDefinedDecl;
pub use acvus_mir::ty::{Effect, EffectSet, EffectTarget, Hint, Param, Ty};
pub use acvus_mir::ty::{Infer, InferEffect, InferTy, ParamTerm, Solver, lift_effect, lift_ty};
pub use acvus_mir::ty::{Poly, PolyBuilder, PolyEffect, PolyParam, PolyTy, lift_to_poly, lift_effect_to_poly};
pub use acvus_utils::{Interner, QualifiedRef};

// ── Re-export: proc macro ──────────────────────────────────────────

pub use acvus_mir_host_macro::{extern_fn, ExternType};
