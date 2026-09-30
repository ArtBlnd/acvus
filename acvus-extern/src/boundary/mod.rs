//! The only modules that write `unsafe`. Each establishes one fact the rest of
//! the crate consumes through a type:
//!
//! - `entry`: a value is read, and a result made, only at the site's settled
//!   type, by the one door of a call (G1, G9, G10).
//! - `uniform`: a type variable's value is released once, by its owner (G7).
//! - `stored`: a type and its canonical form are one layout (G5).
//! - `loaned`: a retained loan is opened only within a call, at the type it
//!   was lent at (K12).
//! - `instance`: an impl value reaches only the receiver it was chosen for
//!   (G6, K14).
//! - `signed`: the facts no type can carry, signed by a runtime or a derive.
//! - `seal`: the crossing traits whose methods choose a typed read's type are
//!   implemented only here and by `shape!` (safety §1).
//! - `adapter`: the adapter's signatures of S1-S3, X's carried over (K15).

#![allow(unsafe_code)]

pub(crate) mod adapter;
pub(crate) mod entry;
pub(crate) mod instance;
pub(crate) mod loaned;
pub mod seal;
pub(crate) mod signed;
pub(crate) mod stored;
pub(crate) mod uniform;
