//! The only modules that write `unsafe`. Each establishes one fact the rest of
//! the crate consumes through a type:
//!
//! - `entry`: a value is read, and a result made, only at the site's settled
//!   type, by the one door of a call (G1, G9, G10).
//! - `uniform`: a type variable's value is released once, by its owner (G7).
//! - `stored`: a type and its canonical form are one layout (G5).
//! - `loaned`: a retained loan is opened only within a call, at the type it
//!   was lent at (K12).
//! - `signed`: the facts no type can carry, signed by a runtime or a derive.
//! - `instance`: a requirement's handle is the site's impl value seen in
//!   place, lent for the call (impls §3).
//! - `seal`: the seals, one `unsafe` trait per meaning, implemented only
//!   here and by the macros (safety §1).

#![allow(unsafe_code)]

pub(crate) mod entry;
pub mod instance;
pub(crate) mod loaned;
pub mod seal;
pub(crate) mod signed;
pub(crate) mod stored;
#[cfg(test)]
pub(crate) mod stub;
pub(crate) mod uniform;
