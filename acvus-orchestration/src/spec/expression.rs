use acvus_mir::ty::Ty;
/// Expr node spec — evaluates a script expression and stores the result.
#[derive(Debug, Clone)]
pub struct ExpressionSpec {
    pub source: String,
    /// Explicit output type hint. `None` = infer from the expression.
    pub output_ty: Option<Ty>,
}
