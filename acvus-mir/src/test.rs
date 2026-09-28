//! Test helpers for compiling source -> MIR.
//!
//! The front end (typeck, lower, graph infer/lower) is being rebuilt; these
//! wrappers keep their signatures so the tests that call them still name a
//! real function, and will run again once lower exists.

use acvus_utils::Interner;

use crate::ir::MirModule;
use crate::ty::Ty;

/// Compile a template source string through the full graph pipeline.
pub fn compile_template(
    _interner: &Interner,
    _source: &str,
    _ctx: &[(&str, Ty)],
) -> Result<MirModule, String> {
    todo!("compiles source through typeck and lower, which are being rebuilt")
}

/// Compile a script source string through the full graph pipeline.
pub fn compile_script(
    _interner: &Interner,
    _source: &str,
    _ctx: &[(&str, Ty)],
) -> Result<MirModule, String> {
    todo!("compiles source through typeck and lower, which are being rebuilt")
}

/// Compile a template and return the printed IR. Panics with full error on failure.
pub fn compile_and_dump(interner: &Interner, source: &str, ctx: &[(&str, Ty)]) -> String {
    let module =
        compile_template(interner, source, ctx).unwrap_or_else(|e| panic!("compile failed:\n{e}"));
    crate::printer::dump(interner, &module)
}
