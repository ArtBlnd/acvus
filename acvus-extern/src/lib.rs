//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to the compiler and the interpreter.
//!
//! See RFC-0009.

mod effect;
mod func;
mod len;
mod registry;
mod ty_arg;

pub use effect::{Eff, EffectArg, EffectVar, Idempotent, Opaque, Pure};
pub use func::{Fn0, Fn1, Fn2};
pub use len::{Arr, Len, LenArg, LenVar};
pub use registry::{
    AsyncHandler, ExternFn, ExternFnDecl, ExternItems, ExternRegistry, ExternTypeDecl,
    Registered, SyncHandler,
};
pub use ty_arg::{PolyVars, TyArg, TyVar, Typeck};

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry};

pub use acvus_interpreter::{
    ExternHandler, ExternTypeName, ExternValue, FnValue, FromValue, FromValues, IntoValue,
    PayloadMismatch, RuntimeError, Value, ValueKind,
};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, LenTerm, ParamTerm, Poly, PolyBuilder, PolyTy, TyTerm,
    TypeRegistry, UserDefinedDecl,
};
pub use acvus_utils::{Interner, QualifiedRef};
pub use rustc_hash::FxHashMap;

#[doc(hidden)]
pub mod __private {
    pub use acvus_interpreter::{into_async_extern_handler, into_sync_extern_handler};
}
