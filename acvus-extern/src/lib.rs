//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0009.

extern crate self as acvus_extern;

pub mod core;
mod effect;
mod error;
mod func;
mod handler;
mod identity;
mod len;
mod obj;
mod reference;
mod registry;
pub mod repr;
mod runtime;
mod space;
mod ty_arg;
mod vec;

pub use effect::{Eff, EffectArg, EffectVar, Idempotent, Opaque, Pure};
pub use error::ExternError;
pub use func::{CallToken, ClosureFn, Fn0, Fn1, Fn2, Fn3};
pub use handler::{ExternEntry, ExternHandler, MonoHandler, MonoInstance};
pub use identity::{IdentityArg, IdentityVar, Idn};
pub use len::{Arr, Len, LenArg, LenVar};
pub use obj::{
    Cross, Obj, Variant, erase_field, materialize_field, materialize_payload, take_payload,
};
pub use reference::{Carried, Ref, RefMut};
pub use registry::{
    CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers, HasInstance,
    Manifest, Registry, SharedSignature, SignatureDecl,
};
pub use repr::{AsCross, AsIs, Crossing, HasRepr};
pub use runtime::{Runtime, TypesOnly};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use ty_arg::{Monomorphize, PolyVars, TyArg, TyVar, Typeck, VarCounts};
pub use vec::vec_ty;

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry, extern_signature};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, LenTerm, Mutability, ParamTerm, Poly, PolyBuilder,
    PolyTy, Ty, TyTerm, TyVarBound, TypeRegistry, UserDefinedDecl, lift_to_poly, try_freeze_poly,
};
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::BoxFuture;
pub use rustc_hash::FxHashMap;
