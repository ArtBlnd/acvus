//! The extern side of the runtime contract: how a value crosses between a run
//! and an extern fn, whatever runtime runs it.
//!
//! `unsafe` is written only under `boundary`; every other module forbids it.
//! Each fact the boundary's `unsafe` relies on is established there once and
//! consumed everywhere else through a type.

#![deny(unsafe_code)]

#[macro_use]
mod lists;

mod boundary;
pub mod callback;
pub mod executor;
pub mod decl;
mod crossing;
pub mod done;
pub mod register;
pub mod runtime;

pub use boundary::instance;
#[doc(hidden)]
pub use boundary::seal;

pub use acvus_extern_macro::{Opaque, extern_fn, generic_fn};

pub use boundary::entry::{Argument, Entry, Impl, PairedMode, RequirementMode, Settled, SiteShape, mode};
pub use boundary::loaned::{Mut, Ref};
pub use boundary::signed::{GenericParam, LayoutFact, SignedS1, SignedS2, SignedS3, Structural, StructuralFacts};
pub use boundary::stored::LayoutWitness;
pub use boundary::uniform::{Index, Primitive, Release, Releasing, Uniform};
