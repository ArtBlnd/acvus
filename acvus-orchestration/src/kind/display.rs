use acvus_mir::ty::Ty;

use crate::compile::CompiledScript;

/// Compiled iterable display configuration.
#[derive(Debug, Clone)]
pub struct CompiledDisplay {
    pub iterator: CompiledScript,
    pub template: CompiledScript,
    pub item_ty: Ty,
}

/// Compiled static display configuration.
#[derive(Debug, Clone)]
pub struct CompiledDisplayStatic {
    pub template: CompiledScript,
}
