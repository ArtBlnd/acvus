

use acvus_mir::context_registry::ContextTypeRegistry;
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashSet;

use crate::compile::{CompiledScript, compile_script_with_hint};
use crate::error::OrchError;

/// Expr node spec — evaluates a script expression and stores the result.
#[derive(Debug, Clone)]
pub struct ExpressionSpec {
    pub source: String,
    /// Explicit output type hint. `None` = infer from the expression.
    pub output_ty: Option<Ty>,
}

/// Compiled expr node.
#[derive(Debug, Clone)]
pub struct CompiledExpression {
    pub script: CompiledScript,
}

/// Compile an expr node spec.
pub fn compile_expression(
    interner: &Interner,
    spec: &ExpressionSpec,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledExpression, FxHashSet<Astr>), Vec<OrchError>> {
    let hint = spec.output_ty.as_ref();
    let (script, _ty) =
        compile_script_with_hint(interner, &spec.source, registry, hint)
            .map_err(|e| vec![e])?;
    let keys = script.context_keys.clone();
    Ok((CompiledExpression { script }, keys))
}
