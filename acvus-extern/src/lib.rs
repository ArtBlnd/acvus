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
pub mod done;
pub mod guard;
pub mod impls;
pub mod runtime;
pub mod shape;

pub use boundary::entry::{Args, Entry, Settled, SiteShape};
pub use boundary::instance::{Instance, InstanceEntry, InstanceOf};
pub use boundary::loaned::{Loaned, LoanedMut};
pub use boundary::signed::{SignedS1, SignedS2, SignedS3, Structural, StructuralFacts};
pub use boundary::stored::LayoutWitness;
pub use boundary::uniform::{Release, Releasing, Uniform};
