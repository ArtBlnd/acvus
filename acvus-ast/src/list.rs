use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};
use crate::span::Span;
use crate::token::Token;

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
    Ok(Expr::List {
        id: AstId::alloc(),
        head,
        rest,
        tail,
        span,
    })
}

