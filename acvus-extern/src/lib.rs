//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0009.

mod convert;
mod effect;
mod error;
mod extern_value;
mod func;
mod handler;
mod identity;
mod len;
mod registry;
mod runtime;
mod ty_arg;

pub use convert::{FromValue, FromValues, IntoValue};
pub use effect::{Eff, EffectArg, EffectVar, Idempotent, Opaque, Pure};
pub use error::ExternError;
pub use extern_value::{ExternTypeName, ExternValue, PayloadMismatch};
pub use func::{Fn0, Fn1, Fn2};
pub use handler::{
    ExternEntry, ExternHandler, MonoHandler, MonoInstance, into_async_extern_handler,
    into_sync_extern_handler,
};
pub use identity::{IdentityArg, IdentityVar, Idn};
pub use len::{Arr, Len, LenArg, LenVar};
pub use registry::{
    AsyncHandler, ExternFn, ExternFnDecl, ExternItems, ExternRegistry, ExternTypeDecl, Registered,
    SyncHandler,
};
pub use runtime::{Runtime, TypesOnly};
pub use ty_arg::{Monomorphize, PolyVars, TyArg, TyVar, Typeck, VarCounts};

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, LenTerm, ParamTerm, Poly, PolyBuilder, PolyTy, Ty,
    TyTerm, TyVarBound, TypeRegistry, UserDefinedDecl, try_freeze_poly,
};
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::BoxFuture;
pub use rustc_hash::FxHashMap;
