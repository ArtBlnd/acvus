//! The context a page op (RFC-0025) names, for passes that order calls
//! against a run's own page ops by their summary (RFC-0017).

use crate::graph::QualifiedRef;
use crate::ir::InstKind;

pub(crate) fn context_read(kind: &InstKind) -> Option<QualifiedRef> {
    match kind {
        InstKind::Fetch { context, .. } => Some(*context),
        _ => None,
    }
}

pub(crate) fn context_written(kind: &InstKind) -> Option<QualifiedRef> {
    match kind {
        InstKind::Commit { context, .. } => Some(*context),
        _ => None,
    }
}
