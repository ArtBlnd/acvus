//! The extern side of the runtime contract: how a value crosses between a run
//! and an extern fn, whatever runtime runs it.
//!
//! `unsafe` is written only under `boundary`; every other module forbids it.
//! Each fact the boundary's `unsafe` relies on is established there once and
//! consumed everywhere else through a type.

#![deny(unsafe_code)]

mod boundary;
pub mod adapter;
pub mod callback;
pub mod context;
mod crossing;
pub mod done;
pub mod guard;
pub mod impls;
pub mod register;
pub mod runtime;

#[doc(hidden)]
pub use boundary::seal;

pub use boundary::entry::{Args, Entry, Settled, SiteShape, mode};
pub use boundary::instance::{Instance, InstanceEntry, InstanceOf};
pub use boundary::loaned::{Loaned, LoanedMut};
pub use boundary::signed::{GenericParam, LayoutFact, SignedS1, SignedS2, SignedS3, Structural, StructuralFacts};
pub use boundary::stored::LayoutWitness;
pub use boundary::uniform::{Primitive, Release, Releasing, Uniform};
pub use crossing::Parted;
