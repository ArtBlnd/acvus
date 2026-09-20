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
mod instance;
mod len;
mod loan;
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
pub use effect::{Idempotent, Opaque, Pure};
pub use erased::Erased;
pub use func::{ArgTypes, Args, CallArgs, CallToken, Closure, ClosureFn};
pub use handler::{
    Arg, ArgAt, ArgAtBound, ArgRun, AsyncAtSite, AsyncCall, AsyncFactory, AsyncGlue, AtBound,
    AtBounds, AtEntry, AtSite, Borrowable, BorrowableSpecialized, ByBound, ByRef, ByValue,
    DeclaredInstance, DirectOp, ExternHandler, Glue, Handler, HandlerFactory, InRegisters,
    InWindow, InstanceEntries, Instances, IntoRun, NoEntry, NoInstances, Parameters, REGISTER_FORM,
    Ret, SiteAtBound, Sited, SitedAtEntry, SitesAtEntry, SitesNoParameterReads, Specialized,
    TakenForm, Uniform, Unsited, Val, ValueParameters, ValuesOnly, Width, async_glue, glue,
    glue_at_entry,
};
pub use instance::{
    Bound, Bounds, Carrier, Entry, EntryFn, EntryNode, Held, HeldMut, InstanceOf, InstanceOfAsync,
    NodeArena, Signature, one_value,
};
pub use len::Arr;
pub use loan::{Loan, Mut, Shared};
pub use obj::{
    Cross, FieldAt, Form, FormKind, FromValue, Inline, Obj, ObjectShape, One, OneValue, Pair, Run,
    Stored, TransparentOver, Variant, expect_type, materialize_checked,
};
pub use owned::{Owned, Release, lend_run};
pub use projection::{
    Borrowed, BorrowedWhole, ByProjection, Fields, Lent, Nested, ObjectAt, Project, Projected,
    Reach, VariantAt, object, object_fields_at, payload_at, variant, variant_tags_at,
};
pub use reference::Ref;
pub use registry::{
    BoundAt, CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers,
    InstanceAt, InstanceTable, Manifest, MemberType, Registry, Requirement, SharedSignature,
    SignatureDecl, family_casts,
};
pub use runtime::{Runtime, TypesOnly};
pub use slice::{Elements, Slice, Words};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use str::{ByStr, RetStr, StrView};
pub use ty_arg::{
    Kind, Monomorphize, Never, Nth, PolyVars, SlotRepr, Spec, Term, TyArg, Var, kind,
};
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
