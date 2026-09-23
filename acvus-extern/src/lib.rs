//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0023.

extern crate self as acvus_extern;

pub mod core;
mod ctx;
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

pub use ctx::Ctx;
pub use derive::transparent::Transparent;
pub use effect::{Idempotent, Opaque, Pure, Suspends};
pub use erased::Erased;
pub use func::{ArgTypes, Args, CallArgs, Closure, ClosureFn, Passed, PassedByValue};
pub use handler::{
    Arg, ArgAt, ArgRun, AsyncAtSite, AsyncCall, AsyncFactory, AsyncGlue, AtInstance, AtSite,
    Borrowable, BorrowableSpecialized, ByRef, ByValue, CallForms, CallSite, DeclaredInstance,
    DirectOp, ExternHandler, Glue, Handler, HandlerFactory, InRegisters, InWindow, InstanceEntries,
    Instances, IntoRun, LentBack, Lends, NoInstance, NoInstances, Parameters, REGISTER_FORM, Required,
    RequiredInstance, Ret, RetLent, Sited, SitesNoParameterReads, Specialized, Uniform, Unsited,
    Val, ValueParameters, ValuesOnly, Width, async_glue, async_glue_at_instance, glue,
    glue_at_instance,
};
pub use instance::{
    CalledAt, Instance, InstanceEntry, InstanceRun, Later, Now, Receiver, RequirementOf, RestRun,
    RestoreByValue, RestoreExclusive, RestoreShared, Signature,
};
pub use len::Arr;
pub use loan::{Loan, Mut, Shared};
pub use obj::{
    Cross, FieldAt, Form, FormKind, FromValue, InPlaceElement, Inline, Nothing, Obj, ObjectShape,
    One, OneRegister, OneValue, OptionOf, Pair, RetForms, Returned, Run, Stored,
    SurvivesSuspension, TransparentOver, Variant, erased_description, is_erased_from,
};
pub use owned::{Owned, Release, lend_run};
pub use projection::{
    Borrowed, BorrowedWhole, ByProjection, Fields, Lent, Nested, ObjectAt, Project, Projected,
    Reach, VariantAt, object, object_fields_at, payload_at, variant, variant_tags_at,
};
pub use reference::Ref;
pub use registry::{
    Coercion, CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers,
    InstanceAt, InstanceTable, Manifest, MemberType, Registry, Requirement, SharedSignature,
    SignatureDecl, family_casts,
};
pub use runtime::{Runtime, TypesOnly};
pub use slice::{BySlice, Slice, Words};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use str::{ByStr, RetStr, StrView};
pub use ty_arg::{
    Chosen, ChosenNth, Kind, Monomorphize, Never, Nth, PolyVars, SlotRepr, Spec, Term, TyArg, Var,
    held_effect, kind,
};
pub use vec::vec_ty;

pub use acvus_extern_macro::{ExternType, TyArg, extern_fn, extern_registry, extern_signature};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::ty::{
    CastRule, Effect, EffectArg, EffectTerm, EffectVarBound, IdentityTerm, LenTerm, Mutability, ObjectTy, ParamTerm, Poly,
    HeldTy, Home, PolyBuilder, PolyTy, Repr, RequirementSig, Task, Ty, TyTerm, TyVarBound, TypeArg, TypeRegistry,
    UserDefinedDecl, lift_to_poly, try_freeze_poly,
};
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::{BoxFuture, Either};
pub use rustc_hash::FxHashMap;
