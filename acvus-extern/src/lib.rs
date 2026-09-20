//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0009.

extern crate self as acvus_extern;

pub mod core;
pub mod derive;
mod effect;
mod erased;
mod func;
mod handler;
mod identity;
mod len;
mod obj;
mod owned;
mod projection;
mod reference;
mod registry;
mod runtime;
mod slice;
mod space;
mod str;
mod ty_arg;
mod vec;

pub use derive::transparent::Transparent;
pub use effect::{Eff, EffectArg, EffectVar, Idempotent, Opaque, Pure};
pub use erased::Erased;
pub use func::{CallToken, ClosureFn, Fn0, Fn1, Fn2, Fn3};
pub use handler::{
    Arg, ArgAt, ArgRun, AsyncAtSite, AsyncCall, AsyncFactory, AsyncGlue, AtSite, Borrowable,
    BorrowableSpecialized, ByRef, ByRefMut, ByValue, DirectOp, ExternHandler, Glue, Handler,
    HandlerFactory, InRegisters, InWindow, Instance, Instances, IntoRun, Parameters, REGISTER_FORM,
    Ret, Sited, SitesNoParameterReads, Specialized, TakenForm, Uniform, Unsited, Val, ValuesOnly,
    Width, async_glue0, async_glue1, async_glue2, async_glue3, async_glue4, async_glue5,
    async_glue6, async_glue7, async_glue8, glue0, glue1, glue2, glue3, glue4, glue5, glue6, glue7,
    glue8,
};
pub use identity::{IdentityArg, IdentityVar, Idn};
pub use len::{Arr, Len, LenArg, LenVar};
pub use obj::{
    Cross, FieldAt, Form, FormKind, FromValue, Inline, Obj, ObjectShape, One, OneValue, Pair, Run,
    Stored, TransparentOver, Variant, expect_type, materialize_checked,
};
pub use owned::{Owned, Release, lend_run};
pub use projection::{
    Borrowed, BorrowedWhole, ByProjection, Fields, FieldsMut, ObjectAt, Project, Projected,
    VariantAt, object_fields_at, object_in, object_in_mut, object_of, object_of_mut, payload_at,
    variant_in, variant_in_mut, variant_of, variant_of_mut, variant_tags_at,
};
pub use reference::{Ref, RefMut};
pub use registry::{
    CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers, HasInstance,
    Manifest, MemberType, Registry, SharedSignature, SignatureDecl, family_casts,
};
pub use runtime::{Runtime, TypesOnly};
pub use slice::{Elements, Slice, SliceMut, Words};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use str::{ByStr, RetStr, StrView};
pub use ty_arg::{Monomorphize, Never, PolyVars, SlotRepr, Spec, TyArg, TyVar, Typeck, VarCounts};
pub use vec::vec_ty;

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry, extern_signature};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectTerm, IdentityTerm, LenTerm, Mutability, ObjectTy, ParamTerm, Poly,
    PolyBuilder, PolyTy, Repr, Task, Ty, TyTerm, TyVarBound, TypeArg, TypeRegistry,
    UserDefinedDecl, lift_to_poly, try_freeze_poly,
};
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::BoxFuture;
pub use rustc_hash::FxHashMap;
