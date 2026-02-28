use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};
use crate::span::Span;
use crate::token::Token;

/// The result of parsing tag content within `{{ }}`.
#[derive(Debug, Clone, PartialEq)]
pub enum TagContent {
    /// A standalone expression: `{{ expr }}`.
    Expr(Expr),
    /// A binding: `{{ lhs = rhs }}`.
    Binding { lhs: Expr, rhs: Expr, span: Span },
    /// A continuation arm: `{{ pattern = }}`.
    ContinuationArm { pattern: Expr, span: Span },
    /// An iteration: `{{ lhs in rhs }}`.
    Iteration { lhs: Expr, rhs: Expr, span: Span },
}

/// Intermediate type for parsing list elements before splitting into head/rest/tail.
pub enum ListElem {
    Expr(Expr),
    Rest(Span),
}

/// Convert a flat `Vec<ListElem>` into a type-safe `Expr::List`.
/// Errors if multiple `..` are present.
/// Returns LALRPOP-compatible error type.
pub fn build_list(
    items: Vec<ListElem>,
    span: Span,
) -> Result<Expr, lalrpop_util::ParseError<usize, Token, ParseError>> {
    let mut head = Vec::new();
    let mut rest = None;
    let mut tail = Vec::new();
    for item in items {
        match item {
            ListElem::Expr(e) => {
                if rest.is_some() {
                    tail.push(e);
                } else {
                    head.push(e);
                }
            }
            ListElem::Rest(s) => {
                if rest.is_some() {
                    return Err(lalrpop_util::ParseError::User {
                        error: ParseError::new(
                            ParseErrorKind::InvalidPattern("multiple `..` in list".into()),
                            span,
                        ),
                    });
                }
                rest = Some(s);
            }
        }
    }
    Ok(Expr::List { head, rest, tail, span })
}

/// Convert a parsed expression (LHS of `->`) into a Lambda node.
/// The params_expr is semantically validated: single Ident → 1 param,
/// Group → multiple params, Paren(Ident) → 1 param.
pub fn expr_to_lambda(params_expr: Expr, body: Expr, span: Span) -> Expr {
    let params = match params_expr {
        Expr::Ident { name, span, .. } => vec![LambdaParam { name, span }],
        Expr::Tuple { elements, .. } => elements
            .into_iter()
            .map(|elem| match elem {
                TupleElem::Expr(Expr::Ident { name, span, .. }) => LambdaParam { name, span },
                TupleElem::Expr(other) => LambdaParam {
                    name: format!("<invalid:{:?}>", other.span()),
                    span: other.span(),
                },
                TupleElem::Wildcard(span) => LambdaParam {
                    name: "<invalid:wildcard>".into(),
                    span,
                },
            })
            .collect(),
        Expr::Paren { inner, .. } => match *inner {
            Expr::Ident { name, span, .. } => vec![LambdaParam { name, span }],
            other => vec![LambdaParam {
                name: format!("<invalid:{:?}>", other.span()),
                span: other.span(),
            }],
        },
        other => vec![LambdaParam {
            name: format!("<invalid:{:?}>", other.span()),
            span: other.span(),
        }],
    };
    Expr::Lambda {
        params,
        body: Box::new(body),
        span,
    }
}
