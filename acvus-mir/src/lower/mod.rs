//! lower: an admitted program and what typeck
//! decided become one `ir::cfg::Module`. Lower decides only the shape of the
//! IR: blocks, values, storage, calls. It reads every type and every decision
//! from `Typeck` and decides none of them again. The order token is the
//! Builder's; no rule here names a token.
//!
//! Three axes: control (every form one region with one entry, and one exit
//! where a run gets past it and none where no run does (`ctxt::Lowered`);
//! `Diamond` for `if`, `&&`, `||`; `Switch` with its join for `match`;
//! `For`/`While` for loops), value (an expression a run gets past is a typed
//! value in the current block; a value-producing form hands it on as its
//! join's parameter), and effect (carried by the Builder).

mod call;
mod closure;
mod coerce;
mod context;
mod control;
mod ctxt;
mod decided;
mod expr;
mod part_tree;
mod pat;
mod place;
mod stmt;
mod template;
#[cfg(test)]
mod tests;

pub mod required;

use acvus_ast::{Script, Slot, Template};

use crate::decl::Declarations;
use crate::ir::cfg::{BodyId, Module};
use crate::ty::{Concrete, Ty, Types};
use crate::typeck::Typeck;

use context::{enter_inputs, input_params};
use ctxt::{Entry, Lcx, ordered};

/// Lowers an admitted script. A refused program is never lowered
/// (refusals): the entry composes typeck, and lowers only when typeck
/// refused nothing.
///
/// # Panics
/// `typeck` holds a refusal: that is the entry's defect.
pub fn lower_script<S: Slot>(decls: &Declarations, typeck: &Typeck, types: Types<Concrete>, script: &Script<S>) -> Module {
    let mut module = Module::new(types);
    let entry = lower_script_into(decls, typeck, &mut module, script);
    module.set_entry(entry);
    module
}

/// `lower_script`'s body, added to `module`, whose type table `typeck` wrote
/// back into: one body of a module holding several, one per instantiation
/// of its source (graph).
pub fn lower_script_into<S: Slot>(decls: &Declarations, typeck: &Typeck, module: &mut Module, script: &Script<S>) -> BodyId {
    assert_admitted(typeck);
    let ret = run_result(typeck);
    let closures = decided::closure_types(typeck, module.types_mut());
    let builder = module.build_body(ordered(&typeck.body_effect), ret);
    let cx = Lcx::new(decls, decided::Decided::run(typeck, &closures), builder, 0);
    let entry = Entry { captures: Vec::new(), params: input_params(&typeck.inputs), contexts: run_contexts(typeck) };
    let content = |cx: &mut Lcx<'_, '_>, _: &[_], params: &[_]| {
        enter_inputs(cx, &typeck.inputs, params);
        stmt::lower_body(cx, &script.stmts, script.tail.as_deref(), script.span).and_then(|value| match &script.tail {
            Some(tail) => call::convert(cx, tail.id(), value, tail.span()),
            None => Ok(value),
        })
    };
    cx.make_body(entry, content, script.span)
}

/// Lowers an admitted template: one body returning the `String` it builds,
/// text appended as literals and `{{ x }}` through `core::display`.
///
/// # Panics
/// As `lower_script`.
pub fn lower_template<S: Slot>(
    decls: &Declarations,
    typeck: &Typeck,
    types: Types<Concrete>,
    template: &Template<S>,
) -> Module {
    let mut module = Module::new(types);
    let entry = lower_template_into(decls, typeck, &mut module, template);
    module.set_entry(entry);
    module
}

/// `lower_template`'s body, added to `module`, as `lower_script_into`.
pub fn lower_template_into<S: Slot>(decls: &Declarations, typeck: &Typeck, module: &mut Module, template: &Template<S>) -> BodyId {
    assert_admitted(typeck);
    let ret = run_result(typeck);
    let closures = decided::closure_types(typeck, module.types_mut());
    let builder = module.build_body(ordered(&typeck.body_effect), ret);
    let cx = Lcx::new(decls, decided::Decided::run(typeck, &closures), builder, 0);
    let entry = Entry { captures: Vec::new(), params: input_params(&typeck.inputs), contexts: run_contexts(typeck) };
    let content = |cx: &mut Lcx<'_, '_>, _: &[_], params: &[_]| {
        enter_inputs(cx, &typeck.inputs, params);
        template::open_output(cx, template.span);
        stmt::lower_stmts(cx, &template.body)?;
        template::take_output(cx, template.span)
    };
    cx.make_body(entry, content, template.span)
}

/// The contexts the run's own body names, outside its lambdas
/// (`Typeck::contexts`).
fn run_contexts(typeck: &Typeck) -> &[(acvus_utils::QualifiedRef, Ty)] {
    typeck.contexts.get(&None).map_or(&[], |named| named)
}

fn assert_admitted(typeck: &Typeck) {
    assert!(
        typeck.refusals.is_empty(),
        "a refused program is never lowered (refusals), and this one was refused: {:?}",
        typeck.refusals
    );
}

/// The run's result type, as typeck recorded it.
///
/// # Panics
/// typeck recorded none: it records one for every admitted program.
fn run_result(typeck: &Typeck) -> Ty {
    typeck.result.expect("typeck records the run's result type of every admitted program")
}
