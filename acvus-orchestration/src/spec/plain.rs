

use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashSet;

use crate::compile::CompiledBlock;
use crate::error::OrchError;

/// Plain node spec — renders a single template, no model call.
#[derive(Debug, Clone)]
pub struct PlainSpec {
    pub source: String,
}

impl PlainSpec {
    pub fn output_ty(&self) -> Ty {
        Ty::String
    }
}

/// Compiled plain node.
#[derive(Debug, Clone)]
pub struct CompiledPlain {
    pub block: CompiledBlock,
}

/// Compile a plain node spec.
pub fn compile_plain(
    interner: &Interner,
    spec: &PlainSpec,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledPlain, FxHashSet<Astr>), Vec<OrchError>> {
    let block =
        crate::compile::compile_template(interner, &spec.source, 0, registry)
            .map_err(|e| vec![e])?;
    let keys = block.context_keys.clone();
    Ok((CompiledPlain { block }, keys))
}
