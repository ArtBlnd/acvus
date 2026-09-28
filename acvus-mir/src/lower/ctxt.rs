//! What lowering one body holds: the Builder, what typeck decided, the slot of
//! each binding, the targets of the loops it is inside, and the home of each
//! context it names. A lambda's body is lowered with its own `Lcx` over a
//! nested Builder.

use rustc_hash::FxHashMap;

use acvus_ast::AstId;
use acvus_utils::QualifiedRef;

use crate::decl::Declarations;
use crate::ir::cfg::{BlockId, Builder};
use crate::ir::ValueId;
use crate::typeck::Typeck;

/// Where `break` and `continue` go: the innermost loop's exit and header
/// (control-flow).
#[derive(Debug, Clone, Copy)]
pub(super) struct LoopTargets {
    pub exit: BlockId,
    pub header: BlockId,
}

/// Where a context lives in this body (context §3): a slot loaded at entry
/// and committed around calls and at return, or none for a volatile one,
/// which is loaded and committed at each access.
#[derive(Debug, Clone, Copy)]
pub(super) enum ContextHome {
    Slot(ValueId),
    Volatile,
}

pub(super) struct Lcx<'a, 'm> {
    pub decls: &'a Declarations,
    pub typeck: &'a Typeck,
    pub builder: Builder<'m>,
    pub bindings: FxHashMap<AstId, ValueId>,
    pub loops: Vec<LoopTargets>,
    pub contexts: FxHashMap<QualifiedRef, ContextHome>,
}
