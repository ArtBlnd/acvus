//! Every refusal typeck makes, in the program's terms (refusals): what was
//! written, where, and the demand that failed. A part never determined is
//! named by the value that left it open, never drawn as a type.

use acvus_ast::Span;
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use crate::infer::Mismatch;
use crate::ty::CastTy;

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
    /// `fn` in a script, which has none (programs §2).
    FnDeclInScript,
    /// `_` or `..` where a value is read: they are a pattern's.
    PatternInValue,
    /// A variant written with other than one payload.
    VariantArity { tag: Astr },
    /// `as` to a name that is not a number, `bool` or `char`
    /// (operators §5).
    UnknownCastTarget { name: Astr },
    // B: two types that do not relate.
    Mismatch(Mismatch),
    NoConversion,
    /// `for` over a value that is no reference to a container, array or
    /// range (control-flow).
    NotIterable,
    /// `?` on a value that is neither a `Result` nor an `Option`
    /// (control-flow).
    NotTryable,
    /// `*r` where `r` is no reference (operators §6).
    NotAReference,
    /// `as` from a type Rust does not cast to `to` (operators §5); `from`
    /// is `None` where the source is no number, `bool` or `char`.
    CastNotAdmitted { from: Option<CastTy>, to: CastTy },
    /// `break` or `continue` with no loop around it before a lambda's
    /// boundary (control-flow).
    OutsideLoop,
    // C: growth into what may not grow.
    GrowsOpaque,
    // Decisions.
    NoDeclaration { name: Astr },
    SeveralDeclarations { name: Astr, left: Box<[QualifiedRef]> },
    /// A construct whose type never became known (types §6), named by the
    /// value that left it open.
    NeverKnown { open: Astr },
    /// A construct at the refusal's span whose type never became known,
    /// where no name the program wrote stands for the value left open.
    Undecided,
    /// A bounded variable no use answered (types §5).
    BoundUnanswered,
}

/// The one way a refusal is made (S4).
pub(super) fn push(cx: &mut Ctxt<'_>, kind: RefusalKind, span: Span) {
    cx.out.refusals.push(Refusal { kind, span });
}

/// B's refusal in the dimension that refuses: growth into an opaque term is
/// C's (types §1), no conversion at all is its own, and every other is B's
/// relation failing.
pub(super) fn mismatch(cx: &mut Ctxt<'_>, mismatch: Mismatch, span: Span) {
    let kind = match mismatch {
        Mismatch::OpaqueLacksField { .. } | Mismatch::OpaqueLacksVariant { .. } => RefusalKind::GrowsOpaque,
        Mismatch::Conversions { candidates } if candidates.is_empty() => RefusalKind::NoConversion,
        other => RefusalKind::Mismatch(other),
    };
    push(cx, kind, span);
}
