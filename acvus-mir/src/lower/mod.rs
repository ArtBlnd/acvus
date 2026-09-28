//! lower (lower-structure.md, D57): an admitted program and what typeck
//! decided become one `ir::cfg::Module`. Lower decides only the shape of the
//! IR: blocks, values, storage, calls. It reads every type and every decision
//! from `Typeck` and decides none of them again. The order token is the
//! Builder's; no rule here names a token.
//!
//! Three axes: control (every form one region with one entry and one exit;
//! `Diamond` for `if`, `&&`, `||`; `Switch` with its join for `match`;
//! `For`/`While` for loops), value (an expression is a typed value in the
//! current block; a value-producing form hands it on as its join's
//! parameter), and effect (carried by the Builder).

mod call;
mod closure;
mod context;
mod control;
mod ctxt;
mod expr;
mod pat;
mod place;
mod stmt;
mod template;
#[cfg(test)]
mod tests;

pub mod required;

use acvus_ast::{Script, Slot, Template};

use crate::decl::Declarations;
use crate::ir::cfg::{Module, Terminator};
use crate::ty::{Concrete, Ty, Types};
use crate::typeck::Typeck;

use context::{Namer, enter_body, enter_inputs};
use control::return_value;
use ctxt::{Lcx, ordered};

/// Lowers an admitted script. A refused program is never lowered
/// (refusals): the entry composes typeck, and lowers only when typeck
/// refused nothing.
///
/// # Panics
/// `typeck` holds a refusal: that is the entry's defect.
pub fn lower_script<S: Slot>(decls: &Declarations, typeck: &Typeck, types: Types<Concrete>, script: &Script<S>) -> Module {
    assert_admitted(typeck);
    let named = {
        let mut namer = Namer::new(typeck);
        namer.stmts(&script.stmts);
        if let Some(tail) = &script.tail {
            namer.expr(tail);
        }
        namer.finish()
    };
    let ret = run_result(typeck);
    let mut module = Module::new(types);
    let builder = module.build_body(ordered(&typeck.body_effect), ret);
    let mut cx = Lcx::new(decls, typeck, builder, 0);
    enter_body(&mut cx, &named);
    enter_inputs(&mut cx, &named);
    let value = stmt::lower_body(&mut cx, &script.stmts, script.tail.as_deref(), script.span);
    let value = match &script.tail {
        Some(tail) => call::convert(&mut cx, tail.id(), value, tail.span()),
        None => value,
    };
    return_value(&mut cx, value, script.span);
    cx.builder.terminate(Terminator::Diverge, script.span);
    let entry = cx.builder.seal().unwrap_or_else(|open| panic!("a lowered script's body is sealed whole: {open:?}"));
    module.set_entry(entry);
    module
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
    assert_admitted(typeck);
    let named = {
        let mut namer = Namer::new(typeck);
        namer.stmts(&template.body);
        namer.finish()
    };
    let ret = run_result(typeck);
    let mut module = Module::new(types);
    let builder = module.build_body(ordered(&typeck.body_effect), ret);
    let mut cx = Lcx::new(decls, typeck, builder, 0);
    enter_body(&mut cx, &named);
    enter_inputs(&mut cx, &named);
    template::open_output(&mut cx, template.span);
    stmt::lower_stmts(&mut cx, &template.body);
    let value = template::take_output(&mut cx, template.span);
    return_value(&mut cx, value, template.span);
    cx.builder.terminate(Terminator::Diverge, template.span);
    let entry = cx.builder.seal().unwrap_or_else(|open| panic!("a lowered template's body is sealed whole: {open:?}"));
    module.set_entry(entry);
    module
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
