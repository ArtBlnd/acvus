//! Statements, and a body of statements and a tail.

use acvus_ast::{Expr, Slot, Stmt};

use super::ctxt::Ctxt;
use crate::infer::Term;

/// Statements then a tail: the tail's term, or `()` without one.
pub(super) fn check_body<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) -> Term {
    todo!("typeck::stmt::check_body")
}

pub(super) fn check_stmts<S: Slot>(cx: &mut Ctxt<'_>, stmts: &[Stmt<S>]) {
    todo!("typeck::stmt::check_stmts")
}

/// `Store`, `DerefStore`, `LetBind`, `LetUninit`, `Assign`, `While`,
/// `WhileLet`, `For`, `Break`, `Continue`, `Anyorder`, `Append`, `Expr`.
/// A `FnDecl` is not a script's (programs §2): it is refused as a form.
pub(super) fn check_stmt<S: Slot>(cx: &mut Ctxt<'_>, stmt: &Stmt<S>) {
    todo!("typeck::stmt::check_stmt")
}
