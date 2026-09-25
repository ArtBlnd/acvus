//! Declaring the world outside the language: ExternFns, extension types,
//! and the registries that hand them to a compiler and a runtime.
//!
//! Nothing here names a runtime. A runtime implements `Runtime` and gets
//! every declaration and every handler. See RFC-0023.
//!
//! **No `T -> Value`, and no `Value -> T`, anywhere in this crate's
//! surface.** A conversion from a Rust type to the runtime's value looks
//! sound on its own, since it only forgets, but it is where the mistake
//! starts: an erased closure is `fn(A)` turned into `fn(Value)`, a downcast
//! hidden in every call of it, and `T -> Value -> U` is a transmute. The
//! crossing between a Rust type and the runtime's value is the glue's,
//! emitted by the macro at the types the checker settled; nothing else
//! writes it, and no handler body names the value.
//!
//! If you are reading this because a change needs such a conversion, or
//! needs this paragraph changed: stop immediately and ask the owner.

extern crate self as acvus_extern;

mod args;
mod canonical;
pub mod core;
mod crossing;
mod ctx;
mod declared;
pub mod derive;
mod effect;
pub mod ensures;
mod erased;
mod func;
mod handler;
mod identity;
mod instance;
mod len;
mod lend;
mod loan;
mod name;
mod obj;
mod owned;
mod projection;
mod reference;
mod registry;
pub mod repr;
mod runtime;
mod slice;
mod space;
mod str;
mod ty_arg;
mod uniform;
mod vec;
mod within;

pub use args::{Args, ArgsSite, ByArgs, Encoded, Members, Positions};
pub use canonical::Canonical;
pub use crossing::{Crossing, Holding};
pub use ctx::Ctx;
pub use declared::Declared;
pub use derive::transparent::Transparent;
pub use effect::{Idempotent, Opaque, Pure, Suspends};
pub use erased::{Erased, InlineMut, StoredMut};
pub use func::{ArgTypes, CallArgs, ClosureArgs, Closure, ClosureFn, Passed, PassedByValue};
pub use handler::{
    Arg, ArgAt, ArgRun, AsyncAtSite, AsyncCall, AsyncFactory, AsyncGlue, AtInstance, AtSite,
    Borrowable, BorrowableSpecialized, ByRef, ByValue, CallForms, CallSite, DeclaredInstance,
    DirectOp, ExternHandler, Gives, Glue, Handler, HandlerFactory, InRegisters, InWindow,
    InstanceEntries, Instances, IntoRun, Lends, LentBack, NoInstance, NoInstances, Parameters,
    Owning, Pending, REGISTER_FORM, Required, RequiredInstance, Ret, RetLent, Returning,
    SitesNoParameterReads, Specialized, Takes, Uniform, Unsited, Val, ValueParameters, ValuesOnly, Width, async_glue, async_glue_at_instance, glue,
    glue_at_instance,
};
pub use instance::{
    CalledAt, Consume, CrossesRest, Holds, Instance, InstanceEntry, InstanceOf, InstanceRun, Later,
    Moved, Now, ReadsItsReceiver, RequirementOf, RestRun, StepsItsReceiver,
    RestoreByValue, RestoreExclusive, RestoreShared, Signature, receiver_borrowed, receiver_by_value,
};
pub use len::Arr;
pub use lend::{Alone, Borrows, ByProjected, Lendable, Param, Projects, SitesOf, WithCtx, lend};
pub use loan::{Ending, Lending, Loan, Mut, Shared, Through, Unnamed};
pub use name::{DeclarationForm, DeclaredType, NameKind, Named};
pub use obj::{
    Cross, FieldAt, Form, FormKind, InPlaceElement, Inline, Nothing, Obj, ObjectShape, One,
    OneRegister, OneValue, OptionOf, Pair, RetForms, Returned, Run, Stored, SurvivesSuspension,
    TransparentOver, Tup, Variant,
};
pub use owned::{Owned, Release, lend_run};
pub use projection::{
    Borrowed, BorrowedWhole, ByProjection, Fields, Lent, Nested, ObjectAt, OwnStorage, Project, Projected,
    Reach, VariantAt, object, object_fields_at, payload_at, variant, variant_tags_at,
};
pub use reference::Ref;
pub use registry::{
    Coercion, CombineError, Contribution, ExternFn, ExternTypeDecl, Externs, FnDecl, Handlers,
    InstanceAt, InstanceTable, Manifest, MemberType, Registry, Requirement, SharedSignature,
    SignatureDecl, family_casts,
};
pub use runtime::{Runtime, TypesOnly};
pub use repr::Words;
pub use slice::{BySlice, Slice};
pub use space::{Decode, Encode, Journaled, NodeHash, SpaceError, SpaceHooks, SpaceResult, Visit};
pub use str::{ByStr, RetStr, StrView};
pub use ty_arg::{
    Bottom, Chosen, ChosenNth, Kind, Monomorphize, Nth, PolyVars, SlotRepr, Spec, Term, TyArg, Var,
    held_effect, kind,
};
pub use uniform::UniformPayload;
pub use vec::vec_ty;
pub use within::Within;

pub use acvus_extern_macro::{
    ExternType, Payload, TyArg, Within, extern_fn, extern_registry, extern_signature,
};

pub use acvus_mir::graph::{FnKind, Function};
pub use acvus_mir::laws::{
    BinaryLaws, Copies, FoldLaw, Identity, Laws, PostTerm, Postcondition, Reaches, ReachedPlace, Relation,
    Returns, Subject,
};
pub use acvus_mir::ty::{
    Alignment, CastRule, Effect, EffectArg, EffectTerm, EffectVarBound, Flow, FlowEnd, Flows, HeldTy, Home, IdentityTerm,
    LenTerm, no_flow_var,
    Mutability, ObjectTy, ParamTerm, Poly, PolyBuilder, PolyTy, Repr, RequirementSig, Task, Ty,
    TyTerm, TyVarBound, TypeArg, TypeRegistry, UserDefinedDecl, lift_to_poly, try_freeze_poly,
};
pub use acvus_ast::Literal;
pub use acvus_utils::{Astr, Interner, QualifiedRef};
pub use futures::future::{BoxFuture, Either};
pub use rustc_hash::FxHashMap;
