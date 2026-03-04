use std::collections::HashSet;

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;

use crate::compile::{CompiledScript, compile_script};
use crate::error::OrchError;

/// Expr node spec — evaluates a script expression and stores the result.
#[derive(Debug, Clone)]
pub struct ExprSpec {
    pub source: String,
    pub output_ty: Ty,
}

/// Compiled expr node.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    pub script: CompiledScript,
}

/// Compile an expr node spec.
pub fn compile_expr(
    spec: &ExprSpec,
    context_types: &std::collections::HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledExpr, HashSet<String>), Vec<OrchError>> {
    let (script, _ty) =
        compile_script(&spec.source, context_types, registry).map_err(|e| vec![e])?;
    let keys = script.context_keys.clone();
    Ok((CompiledExpr { script }, keys))
}
