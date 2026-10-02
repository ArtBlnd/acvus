//! The extern side of acvus: the contract a runtime implements, and the way
//! an extern fn reaches a runtime (`docs/extern-note.md`).

#![deny(unsafe_code, unsafe_op_in_unsafe_fn)]

mod crossing;
mod declared;
mod kind;
mod marker;
mod repr;
mod runtime;
mod uniform;

pub use acvus_extern_macro::{Opaque, Structural};
pub use crossing::Crossing;
pub use declared::{
    Declared, DeclaredEffect, DeclaredIdentity, Declaring, EffectAt, IdentityAt, OpaqueDecl, Ty, Var,
};
pub use kind::{EnumKind, FnKind, ObjectKind, OpaqueKind, OptionKind, RefKind, ResultKind, ScalarKind, StringKind, TupleKind};
pub use marker::{EffectVar, IdentityVar, Never, SomeEffect, SomeIdentity, TypeVar};
pub use repr::{Boxed, StoredOpaque, StoredRef, Words};
pub use runtime::{Arguments, ExternFn, Registry, Runtime, Stores};
pub use uniform::{Lend, Uniform};
