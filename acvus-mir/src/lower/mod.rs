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

pub mod required;

use acvus_ast::{Script, Slot, Template};

use crate::decl::Declarations;
use crate::ir::cfg::Module;
use crate::ty::{Concrete, Types};
use crate::typeck::Typeck;

/// Lowers an admitted script. A refused program is never lowered
/// (refusals): the entry composes typeck, and lowers only when typeck
/// refused nothing.
///
/// # Panics
/// `typeck` holds a refusal: that is the entry's defect.
pub fn lower_script<S: Slot>(decls: &Declarations, typeck: &Typeck, types: Types<Concrete>, script: &Script<S>) -> Module {
    todo!("lower::lower_script")
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
    todo!("lower::lower_template")
}
