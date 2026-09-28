use super::decision::Decisions;
use super::output::Output;
use super::scope::Scope;
use crate::decl::Declarations;
use crate::infer::Infer;

/// The shared state of one checked body (S1): the declarations it reads,
/// and four things it builds. No rule holds state of its own.
pub(super) struct Ctxt<'d> {
    pub decls: &'d Declarations,
    pub infer: Infer,
    pub scope: Scope,
    pub decisions: Decisions,
    pub out: Output,
}

impl<'d> Ctxt<'d> {
    pub fn new(decls: &'d Declarations) -> Self {
        Self {
            decls,
            infer: Infer::new(),
            scope: Scope::new(),
            decisions: Decisions::default(),
            out: Output::default(),
        }
    }
}
