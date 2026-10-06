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
mod never;
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
pub use output::{InstanceDecided, Adjustment, AsCast, Capture, CaptureOf, Captured, Chosen, Closure, Coercion, Conversion, Impl, Iteration, Name, Own, RunInput, StructureTree, TryExit, Typeck, UnitCall};
pub use refusal::{Exceeding, Frozen, FrozenMismatch, Refusal, RefusalKind, Unimplemented};

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
    let result = context::check_body_result(&mut cx, site, value, span);
    let unit = open.then(|| unit_scheme(&mut cx, script.id, result));
    generalize::generalize_all(&mut cx);
    export_unit(&mut cx, unit);
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
    let built = cx.node(TyKind::String);
    let unit = open.then(|| unit_scheme(&mut cx, template.id, built));
    generalize::generalize_all(&mut cx);
    export_unit(&mut cx, unit);
    decision::solve(&mut cx);
    closure::settle_captures(&mut cx);
    writeback::finish(cx, types, built)
}

/// Each input declared at a type without variables, at the concrete type
/// it is in `types`: what the tests fix a run's inputs at.
#[cfg(test)]
pub(crate) fn fixed_inputs(decls: &Declarations, types: &mut Types<Concrete>, declared: &[(Astr, crate::ty::TyId<crate::ty::Poly>)]) -> Vec<(Astr, Ty)> {
    declared.iter().map(|(name, ty)| (*name, declared_concrete(decls, types, *ty))).collect()
}

/// A type the host declared, with no variable (`decl::verify` refuses an
/// entry's type holding one), at the concrete type it is in `types`.
pub(crate) fn declared_concrete(decls: &Declarations, types: &mut Types<Concrete>, declared: crate::ty::TyId<crate::ty::Poly>) -> Ty {
    let mut infer = crate::infer::Infer::new();
    let term = infer.instantiate_type(decls, declared);
    infer
        .freeze(types, &mut crate::infer::FreezeMemo::default(), term)
        .expect("a declared type holds no variable")
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

/// The order of a unit's parameters: its inputs by the text of their names,
/// so a call that passes them by name passes them in one order wherever it
/// is written (programs §1-§2).
fn parameter_order<T>(cx: &Ctxt<'_>, inputs: &mut [(Astr, T)]) {
    inputs.sort_by(|(a, _), (b, _)| cx.interner.resolve(*a).cmp(cx.interner.resolve(*b)));
}

/// The check, its inputs open, as one scheme (`generalize::register_unit`):
/// a function from its inputs, in the order of its parameters
/// (`parameter_order`), to `result`, with the check's effect, called any
/// number of times and capturing nothing (programs §1-§2, functions §3).
fn unit_scheme(cx: &mut Ctxt<'_>, root: acvus_ast::AstId, result: crate::infer::Term) -> (generalize::SchemeId, Box<[Astr]>) {
    let mut inputs = cx.scope.inputs();
    parameter_order(cx, &mut inputs);
    let effect = effect::body_effect(cx);
    let flows = crate::ty::FlowTerm::Var(cx.infer.fresh_flow());
    let values = cx.infer.fresh_values();
    let term = cx.node(TyKind::Fn {
        params: inputs.iter().map(|(_, term)| *term).collect(),
        ret: result,
        effect,
        flows,
        calls: crate::ty::CallsTerm::Known(crate::ty::Calls::Many),
        holds: crate::ty::HoldsTerm::Known(crate::ty::Holds::Nothing),
        values,
    });
    let names = inputs.into_iter().map(|(name, _)| name).collect();
    (generalize::register_unit(cx, root, term), names)
}

/// The unit's scheme, once settled, exported into its interface: settled
/// is final, since every decision still reading what it generalizes is one
/// of its requirements, so what solving decides after is the check's alone.
fn export_unit(cx: &mut Ctxt<'_>, unit: Option<(generalize::SchemeId, Box<[Astr]>)>) {
    let Some((scheme, params)) = unit else {
        return;
    };
    let (store, scheme) = generalize::export(cx, scheme);
    cx.out.interface = Some(Interface { store, params, scheme });
}

// A module (programs §3) is to-be: its entry gives its declarations'
// schemes to `Declarations`, where a script reaches them as an extern's.
