//! Identity positions in extension-type declarations.
//!
//! `IdentityArg` names the K-th identity variable of a declaration;
//! `IdentityVar` is the bound of a generic parameter that is one, and
//! `()` fills it at runtime. A type with an identity parameter is a
//! distinct source per value and moves rather than copies.

use acvus_mir::ty::{IdentityTerm, Poly};

use crate::ty_arg::PolyVars;

pub trait IdentityArg: Send + Sync + 'static {
    fn poly_identity(vars: &PolyVars) -> IdentityTerm<Poly>;
}

pub trait IdentityVar: Send + Sync + 'static {}

impl<I: IdentityArg> IdentityVar for I {}
impl IdentityVar for () {}

/// The K-th identity variable of a declaration. Uninhabited.
pub enum Idn<const K: usize> {}

impl<const K: usize> IdentityArg for Idn<K> {
    fn poly_identity(vars: &PolyVars) -> IdentityTerm<Poly> {
        vars.identities[K]
    }
}
