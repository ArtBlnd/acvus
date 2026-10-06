//! `!`, the type of no value (types §4-6): the one rule typeck keeps for it,
//! and the operations every other part of typeck applies it through.
//!
//! - An expression that gives no value decides nothing of its type: its
//!   type is what its uses decide, and `!` where nothing does (types §4,
//!   §5). Such a form is a `return`, a block whose last statement is `break`
//!   or `continue` (`diverged`), and a call of a declaration whose result is
//!   `!` (`call_result`).
//! - `!` says nothing of a type (`says_nothing`): a decision that reads it
//!   decides nothing from it and refuses nothing for it (types §6,
//!   `settle`), and a reach matches it with every candidate (`reach`).
//! - So a term is `!` only by the default (types §5, `default`): the program
//!   writes no type, and a declaration writes `!` only as a signature's whole
//!   result (`decl::verify`), which a call's type does not take.

use acvus_ast::Span;

use super::ctxt::Ctxt;
use super::decision::{Decision, Reading, closed_for, read_head};
use super::output::StructureNode;
use crate::decl::{Base, Head};
use crate::infer::Term;
use crate::ty::{TyKind, TyVarId};

/// The type of a form that gives no value: a fresh term its uses decide.
pub(super) fn diverged(cx: &mut Ctxt<'_>) -> Term {
    cx.fresh()
}

/// A call's type, `result`, takes its declaration's result, `declared`; a
/// call of a declaration whose result is `!` gives no value, so its type is
/// its uses'. The callee's type keeps the `!`.
pub(super) fn call_result(cx: &mut Ctxt<'_>, result: Term, declared: Term, span: Span) {
    if matches!(cx.infer.types().kind(cx.infer.shallow(declared)), TyKind::Never) {
        return;
    }
    cx.eq(result, declared, span);
}

/// Whether a closed head is `!`, which says nothing of a type.
pub(super) fn says_nothing(head: Head) -> bool {
    head == Head::Base(Base::Never)
}

/// types §5: `var`, which nothing constrains, is `!`. The only `!` typeck
/// makes.
pub(super) fn default(cx: &mut Ctxt<'_>, var: TyVarId) {
    let var = cx.node(TyKind::Var(var));
    let never = cx.node(TyKind::Never);
    cx.infer
        .eq(var, never)
        .expect("an unbound variable of no bound takes `!`: it is no reference, so no reference target refuses it");
}

/// `decision` fired where the one head it reads (`read_head`) is closed at
/// `!`: what it leaves, deciding nothing from the `!` and refusing nothing
/// for it. Handed back where it reads no `!`, or where another part of the
/// rule takes the `!` it reads.
pub(super) fn settle(cx: &mut Ctxt<'_>, decision: Decision, reading: Reading) -> Result<(), Decision> {
    let reads_never = read_head(&decision)
        .and_then(|term| closed_for(cx, term, reading))
        .is_some_and(|closed| says_nothing(closed.head));
    if !reads_never {
        return Err(decision);
    }
    match decision {
        // What each makes is left to its uses.
        Decision::Field { .. }
        | Decision::Index { .. }
        | Decision::Deref { .. }
        | Decision::Iterate { .. }
        | Decision::Try { .. }
        | Decision::Cast { .. }
        | Decision::Lend { .. }
        | Decision::BlockEnd { .. }
        | Decision::Operand { .. } => Ok(()),
        // The pattern's site is typed, as every site is; the names it binds
        // are left to their uses (patterns §4).
        Decision::Destructure { shape, scrutinee, .. } => {
            super::pat::record_matched(cx, &shape, scrutinee);
            Ok(())
        }
        // No part is compared, as no value is (operators §4).
        Decision::Compare { node, .. } => {
            cx.out.structures.insert(node, StructureNode::Parts(Box::default()));
            Ok(())
        }
        // A value entering a slot folds `!` there (`Infer::enter`), a
        // receiver's adjustment reads it as a reach does, and an instance's
        // decision is settled as it fires within the instance.
        Decision::Enter { .. } | Decision::Adjust(_) | Decision::In { .. } => Err(decision),
        // These read no one head.
        Decision::Reach { .. } | Decision::Instantiate { .. } | Decision::Settle { .. } | Decision::StartMark { .. } => {
            Err(decision)
        }
    }
}
