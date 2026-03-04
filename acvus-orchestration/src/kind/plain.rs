use std::collections::{HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;

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
    spec: &PlainSpec,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledPlain, HashSet<String>), Vec<OrchError>> {
    let block = crate::compile::compile_template(&spec.source, 0, context_types, registry)
        .map_err(|e| vec![e])?;
    let keys = block.context_keys.clone();
    Ok((CompiledPlain { block }, keys))
}
