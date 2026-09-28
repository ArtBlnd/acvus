//! Patterns (patterns §1): a pattern naming a variant grows the enum where
//! transparent, and is refused where it is not (patterns §2); a `!`
//! scrutinee reaches no arm (patterns §4). Exhaustiveness is validate's.

use acvus_ast::{Pattern, Slot};

use super::ctxt::Ctxt;
use crate::infer::Term;

/// Binds `pattern`'s names against a scrutinee of term `scrutinee`.
pub(super) fn check_pattern<S: Slot>(cx: &mut Ctxt<'_>, pattern: &Pattern<S>, scrutinee: Term) {
    todo!("typeck::pat::check_pattern")
}
