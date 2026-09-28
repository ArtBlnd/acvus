//! Decisions (types §6): what waits for a closed type. One set, one
//! fixpoint (S2, S3).
//!
//! `solve` fires every decision whose terms are closed, each once, until
//! none fires; at a stall it applies the defaults of types §5 all at once
//! and resumes; a decision still open after that is refused, naming the
//! value that left it open. Firings are at most the decisions, finite since
//! requirements end (extern-fn §6); default rounds are at most the
//! variables (compilation).

use acvus_ast::{AstId, Span};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::Ctxt;
use crate::infer::{OpenSources, RequiredAt, Term};
use crate::ty::Mutability;

/// A signature of `core` a form calls (operators §2, §3, §7). `&&`, `||`,
/// `!`, `^`, `&`, `|`, `<<` and `>>` are the language's own operations, not
/// calls (operators §2): `op` types them without a reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum CoreSig {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Eq,
    Cmp,
    AsSlice(Mutability),
    AsStr,
    Display,
    Clone,
}

/// What a reach decides among.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Callee {
    /// Every declaration a bare name may reach.
    Name(Astr),
    /// A qualified name: one declaration.
    Declaration(QualifiedRef),
    /// A method: the declarations of that name, the receiver first.
    Method(Astr),
    Core(CoreSig),
    /// An instance a reached declaration requires.
    Required(RequiredAt),
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Decision {
    /// Which declaration a call reaches, and at which instance (types §6).
    Reach { site: AstId, span: Span, callee: Callee, args: Box<[Term]>, result: Term },
    /// What a `for` iterates by (control-flow).
    Iterate { site: AstId, span: Span, head: Term, item: Term },
    /// What a `?` unwraps (control-flow).
    Try { site: AstId, span: Span, operand: Term, value: Term, exit: Term },
    /// A value entering a slot (types §3, §7).
    Enter { site: AstId, span: Span, value: Term, slot: Term },
}

/// The pending decisions, and what they may still change.
#[derive(Default)]
pub(super) struct Decisions {
    pending: Vec<Decision>,
}

impl Decisions {
    pub fn push(&mut self, decision: Decision) {
        self.pending.push(decision);
    }
}

impl OpenSources for Decisions {
    /// Whether a pending decision outside the cycles through `term` may
    /// still enter into it or bind it (types §6).
    fn may_change(&self, term: Term) -> bool {
        todo!("typeck::Decisions::may_change")
    }
}

/// The fixpoint of the module doc.
pub(super) fn solve(cx: &mut Ctxt<'_>) {
    todo!("typeck::decision::solve")
}
