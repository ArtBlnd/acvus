//! Every term to its concrete type, once (compilation); the lambdas' claimed
//! flows into their function types.

use super::ctxt::Ctxt;
use super::output::Typeck;
use crate::ty::{Concrete, Types};

pub(super) fn finish(cx: Ctxt<'_>, types: &mut Types<Concrete>) -> Typeck {
    todo!("typeck::writeback::finish")
}
