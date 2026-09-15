//! Which context a `Take` or `Assign` reaches whole, for passes that order
//! calls against a run's own context accesses (RFC-0017).

use crate::graph::QualifiedRef;
use crate::ir::{InstKind, RefTarget};

/// The context a whole `Take` reads.
pub(crate) fn context_read(kind: &InstKind) -> Option<QualifiedRef> {
    match kind {
        InstKind::Take {
            target: RefTarget::Context(qref),
            path,
            ..
        } if path.is_empty() => Some(*qref),
        _ => None,
    }
}

/// The context a whole `Assign` writes.
pub(crate) fn context_written(kind: &InstKind) -> Option<QualifiedRef> {
    match kind {
        InstKind::Assign {
            target: RefTarget::Context(qref),
            path,
            ..
        } if path.is_empty() => Some(*qref),
        _ => None,
    }
}
