use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};
use crate::parser::{GrammarError, Recover, refused};
use crate::span::Span;

/// Intermediate type for parsing list elements before splitting into head/rest/tail.
pub enum ListElem<S> {
    Expr(Expr<S>),
    Rest(Span),
}

/// Convert a flat `Vec<ListElem>` into a type-safe `Expr::List`.
/// Returns LALRPOP-compatible error type.
pub(crate) fn build_list<S>(
    errors: &mut Vec<ParseError>,
    items: Vec<ListElem<S>>,
    span: Span,
) -> Result<Expr<S>, GrammarError>
where
    S: Recover,
{
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
                    return refused::<S>(
                        errors,
                        ParseError::new(
                            ParseErrorKind::InvalidPattern("multiple `..` in list".into()),
                            span,
                        ),
                        span,
                    )
                    .map(Expr::Error);
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
