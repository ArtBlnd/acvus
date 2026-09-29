//! A, the expression rule, and the decisions (rfc_compiler: types §6,
//! refusals). The checker walks a body once, states what each form demands
//! in B's words, and leaves to decisions what waits for a closed type.
//! typeck decides only what a type decides; exhaustiveness, initialization,
//! moves and borrows are check's, and drops are drop insertion's.
//!
//! Invariants of the structure:
//! - S1. `Ctxt` holds four things (infer, scope, decisions, output); no rule
//!   adds state beside them.
//! - S2. Anything decided later is a `Decision`; no rule keeps a list to
//!   drain after solving, but for the uses of captured names, which the
//!   capture walk reads once the fixpoint ends, as Rust's upvar analysis
//!   follows its type check (D85, `closure::settle_captures`).
//! - S3. A decision reads only closed terms (types §6).
//! - S4. A refusal is made only through `refusal`.
//! - S5. Every pass states its cost; the fixpoint fires each decision once.
//! - S7. Every decision has an owner, the innermost walk it was made in: a
//!   lambda's body, a value a `let` or an assignment gives a binding, or
//!   the run; a decision made while one fires has that one's. A generalized
//!   binding's requirements are the decisions of its values' walks still
//!   open over what it generalizes (G, `generalize`).
//! - S6. A transparent structured type is taken apart only by `structure`,
//!   S's one rule (types §1, §3): a value coerced to a slot written
//!   elsewhere, a lifted impl, and a comparison each record a node per part
//!   and decide each part at a node of its own.

mod call;
mod cast;
mod closure;
mod context;
mod control;
mod ctxt;
mod decision;
mod effect;
mod expr;
mod generalize;
mod interface;
mod method;
mod op;
mod output;
mod pat;
mod place;
mod reach;
mod refusal;
mod scope;
mod stmt;
mod structure;
mod template;
#[cfg(test)]
mod tests;
mod variant;
mod writeback;

use acvus_ast::{Expr, Script, Slot, Template};
use acvus_utils::{Astr, Interner};

use crate::decl::Declarations;
use crate::ty::{Concrete, Ty, TyKind, Types};

pub use generalize::InstanceId;
pub use interface::{Interface, Interfaces};
pub use output::{InstanceDecided, Adjustment, AsCast, Capture, CaptureOf, Captured, Chosen, Closure, Coercion, Conversion, Impl, Iteration, Name, Own, StructureTree, TryExit, Typeck, UnitCall};
pub use refusal::{Exceeding, Refusal, RefusalKind, Unimplemented};

/// A unit's inputs as a check takes them (graph): open, each a fresh type
/// its uses decide, the unit generalized at the end into its interface; or
/// fixed at an instantiation's types, in the table the check writes back
/// into, each input entering its type.
pub enum Inputs<'a> {
    Open,
    Fixed(&'a [(Astr, Ty)]),
}

use ctxt::Ctxt;
use scope::Run;

/// A script (programs §2): statements and a tail.
/// `interfaces` are the units its calls may reach.
pub fn check_script<S: Slot>(
    interner: &Interner,
    decls: &Declarations,
    interfaces: &Interfaces,
    inputs: Inputs<'_>,
    types: &mut Types<Concrete>,
    script: &Script<S>,
) -> Typeck {
    let mut cx = Ctxt::new(interner, decls, interfaces, Run::Script);
    let open = enter_inputs(&mut cx, types, inputs);
    let tail = script.tail.as_deref();
    let value = stmt::check_body(&mut cx, &script.stmts, tail);
    let site = tail.map_or(script.id, Expr::id);
    let span = tail.map_or(script.span, Expr::span);
    let result = context::check_run_result(&mut cx, site, value, span);
    generalize::generalize_all(&mut cx);
    generalize_run(open);
    decision::solve(&mut cx);
    closure::settle_captures(&mut cx);
    writeback::finish(cx, types, result)
}

/// A template (programs §1): text and `{{ }}`, each a call of
/// `core::display`.
/// `interfaces` are the units its calls may reach.
pub fn check_template<S: Slot>(
    interner: &Interner,
    decls: &Declarations,
    interfaces: &Interfaces,
    inputs: Inputs<'_>,
    types: &mut Types<Concrete>,
    template: &Template<S>,
) -> Typeck {
    let mut cx = Ctxt::new(interner, decls, interfaces, Run::Template);
    let open = enter_inputs(&mut cx, types, inputs);
    stmt::check_stmts(&mut cx, &template.body);
    generalize::generalize_all(&mut cx);
    generalize_run(open);
    decision::solve(&mut cx);
    closure::settle_captures(&mut cx);
    let built = cx.node(TyKind::String);
    writeback::finish(cx, types, built)
}

/// Each input declared at a type without variables, at the concrete type
/// it is in `types`: what the tests fix a run's inputs at.
#[cfg(test)]
pub(crate) fn fixed_inputs(decls: &Declarations, types: &mut Types<Concrete>, declared: &[(Astr, crate::ty::TyId<crate::ty::Poly>)]) -> Vec<(Astr, Ty)> {
    let mut infer = crate::infer::Infer::new();
    let mut memo = crate::infer::FreezeMemo::default();
    declared
        .iter()
        .map(|(name, ty)| {
            let term = infer.instantiate_type(decls, *ty);
            let concrete = infer.freeze(types, &mut memo, term).expect("an input's declared type holds no variable");
            (*name, concrete)
        })
        .collect()
}

/// Whether the run's inputs are open (programs §4): each `$name` read is a
/// fresh term its uses decide. Fixed, each given input enters its type,
/// written elsewhere, and a `$name` not given is unknown.
fn enter_inputs(cx: &mut Ctxt<'_>, types: &Types<Concrete>, inputs: Inputs<'_>) -> bool {
    match inputs {
        Inputs::Open => true,
        Inputs::Fixed(given) => {
            let terms = given.iter().map(|(name, ty)| (*name, cx.infer.instantiate_concrete(types, *ty))).collect();
            cx.scope.fix_inputs(terms);
            false
        }
    }
}

/// The run with its inputs open, generalized into its interface.
///
/// # Panics
/// Always where `open`: how the run's open decisions become its scheme is
/// not settled, so the open check stops here.
fn generalize_run(open: bool) {
    if open {
        todo!(
            "generalizing a run into its interface: a decision still open over a generalized variable has a \
             `decl::Requirement` form only where it is a reach of one generic function, and whether a default \
             applies before generalizing is not settled"
        );
    }
}

// A module (programs §3) is to-be: its entry gives its declarations'
// schemes to `Declarations`, where a script reaches them as an extern's.
