//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0009.

extern crate self as acvus_extern;

pub mod core;
pub mod repr;
mod effect;
mod error;
mod func;
mod handler;
mod identity;
mod len;
mod obj;
mod reference;
mod space;
mod registry;
mod runtime;
mod ty_arg;

pub use effect::{Eff, EffectArg, EffectVar, Idempotent, Opaque, Pure};
pub use error::ExternError;
pub use func::{CallToken, ClosureFn, Fn0, Fn1, Fn2, Fn3};
pub use handler::{ExternEntry, ExternHandler, MonoHandler, MonoInstance};
pub use identity::{IdentityArg, IdentityVar, Idn};
pub use len::{Arr, Len, LenArg, LenVar};
pub use obj::{Cross, Obj, erase_field, materialize_field};
pub use reference::{Carried, Ref, RefMut};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use repr::{AsCross, AsIs, Crossing, HasRepr};
pub use registry::{
    CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers, HasInstance,
    Manifest, Registry, SharedSignature, SignatureDecl,
};
pub use runtime::{Runtime, TypesOnly};
pub use ty_arg::{Monomorphize, PolyVars, TyArg, TyVar, Typeck, VarCounts};

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry, extern_signature};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, LenTerm, Mutability, ParamTerm, Poly, PolyBuilder,
    PolyTy, Ty, TyTerm, TyVarBound, TypeRegistry, UserDefinedDecl, lift_to_poly, try_freeze_poly,
};
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::BoxFuture;
pub use rustc_hash::FxHashMap;
