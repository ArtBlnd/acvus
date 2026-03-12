

use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::compile::{CompiledScript, compile_script_with_hint};
use crate::error::OrchError;

/// Expr node spec — evaluates a script expression and stores the result.
#[derive(Debug, Clone)]
pub struct ExprSpec {
    pub source: String,
    pub output_ty: Ty,
    /// Optional initial state. When Some, `@self` is available in the node body.
    /// Only Expr nodes support `initial_value` (stateful accumulation via `@self`).
    pub initial_value: Option<String>,
}

/// Compiled expr node.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    pub script: CompiledScript,
    /// Compiled initial_value script. When Some, `@self` is available in the node body.
    pub initial_value: Option<CompiledScript>,
}

/// Compile an expr node spec.
pub fn compile_expr(
    interner: &Interner,
    spec: &ExprSpec,
    context_types: &FxHashMap<Astr, Ty>,
) -> Result<(CompiledExpr, FxHashSet<Astr>), Vec<OrchError>> {
    let hint = match &spec.output_ty {
        Ty::Infer => None,
        ty => Some(ty),
    };
    let (script, _ty) =
        compile_script_with_hint(interner, &spec.source, context_types, hint)
            .map_err(|e| vec![e])?;
    let keys = script.context_keys.clone();
    Ok((CompiledExpr { script, initial_value: None }, keys))
}
