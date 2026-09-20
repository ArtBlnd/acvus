//! Identity positions in extension-type declarations.
//!
//! An identity variable is named by `Nth<kind::Identity, K>` while the
//! declaration's type is built. A type with an identity parameter is a
//! distinct source per value and moves rather than copies.

use crate::ty_arg::{Var, kind};

/// The runtime carries no identity: an identity variable is settled before
/// the handler runs, so the runtime fills it with nothing.
impl Var<kind::Identity> for () {}
