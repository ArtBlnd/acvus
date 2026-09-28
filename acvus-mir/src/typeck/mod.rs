//! A, the expression rule, and the decisions (rfc_compiler: types §6,
//! refusals). The checker walks a body once, states what each form demands
//! in B's words, and leaves to decisions what waits for a closed type.
//! typeck decides only what a type decides; exhaustiveness, initialization,
//! moves and borrows are validate's, and drops are drop insertion's.
//!
//! Invariants of the structure:
//! - S1. `Ctxt` holds four things (infer, scope, decisions, output); no rule
//!   adds state beside them.
//! - S2. Anything decided later is a `Decision`; no rule keeps a list to
//!   drain after solving.
//! - S3. A decision reads only closed terms (types §6).
//! - S4. A refusal is made only through `refusal`.
//! - S5. Every pass states its cost; the fixpoint fires each decision once.

mod call;
mod cast;
mod closure;
mod context;
mod control;
mod ctxt;
mod decision;
mod effect;
mod expr;
mod method;
mod op;
mod output;
mod pat;
mod place;
mod reach;
mod refusal;
mod scope;
mod stmt;
mod template;
mod variant;
mod writeback;

use acvus_ast::{Script, Slot, Template};

use crate::decl::Declarations;
use crate::ty::{Concrete, Types};

pub use output::{AsCast, Capture, Captured, Chosen, Closure, Conversion, Impl, Iteration, Name, TryExit, Typeck};
pub use refusal::{Refusal, RefusalKind};

use ctxt::Ctxt;

/// A script (programs §2): statements and a tail.
pub fn check_script<S: Slot>(decls: &Declarations, types: &mut Types<Concrete>, script: &Script<S>) -> Typeck {
    let mut cx = Ctxt::new(decls);
    let tail = stmt::check_body(&mut cx, &script.stmts, script.tail.as_deref());
    context::check_run_result(&mut cx, tail, script.span);
    decision::solve(&mut cx);
    writeback::finish(cx, types)
}

/// A template (programs §1): text and `{{ }}`, each a call of
/// `core::display`.
pub fn check_template<S: Slot>(decls: &Declarations, types: &mut Types<Concrete>, template: &Template<S>) -> Typeck {
    let mut cx = Ctxt::new(decls);
    stmt::check_stmts(&mut cx, &template.body);
    decision::solve(&mut cx);
    writeback::finish(cx, types)
}

// A module (programs §3) is to-be: its entry gives its declarations'
// schemes to `Declarations`, where a script reaches them as an extern's.
