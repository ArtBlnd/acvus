use std::collections::{HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};

use crate::compile::{CompiledScript, compile_script_with_hint};
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
    interner: &Interner,
    spec: &ExprSpec,
    context_types: &HashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledExpr, HashSet<Astr>), Vec<OrchError>> {
    let hint = match &spec.output_ty {
        Ty::Infer => None,
        ty => Some(ty),
    };
    let (script, _ty) =
        compile_script_with_hint(interner, &spec.source, context_types, registry, hint).map_err(|e| vec![e])?;
    let keys = script.context_keys.clone();
    Ok((CompiledExpr { script }, keys))
}
