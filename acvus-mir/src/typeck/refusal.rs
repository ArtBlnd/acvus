//! Every refusal typeck makes, in the program's terms (refusals): what was
//! written, where, and the demand that failed. A part never determined is
//! named by the value that left it open, never drawn as a type.

use acvus_ast::Span;
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use crate::infer::Mismatch;

#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub span: Span,
}

/// Grouped by the dimension that refuses (typeck-structure section 2).
#[derive(Debug, Clone, PartialEq)]
pub enum RefusalKind {
    // A: a form the language does not admit.
    LambdaRecursion,
    InputAssigned { input: Astr },
    UnnamedVariant { tag: Astr },
    UnknownName { name: Astr },
    UnknownContext { context: QualifiedRef },
    /// A once function where many calls are asked (functions §3).
    CalledOnceWhereMany,
    /// A context's or a run's value whose type holds a loan (ownership §5).
    HoldsLoans,
    /// `*r = v` through a shared reference (operators §6).
    StoreThroughShared,
    // B: two types that do not relate.
    Mismatch(Mismatch),
    NoConversion,
    // C: growth into what may not grow.
    GrowsOpaque,
    // Decisions.
    NoDeclaration { name: Astr },
    SeveralDeclarations { name: Astr, left: Box<[QualifiedRef]> },
    /// A construct whose type never became known (types §6), named by the
    /// value that left it open.
    NeverKnown { open: Astr },
    /// A bounded variable no use answered (types §5).
    BoundUnanswered,
}

/// The one way a refusal is made (S4).
pub(super) fn push(cx: &mut Ctxt<'_>, kind: RefusalKind, span: Span) {
    cx.out.refusals.push(Refusal { kind, span });
}
