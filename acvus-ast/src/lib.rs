pub mod ast;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod tag_content;
pub mod token;

#[allow(
    clippy::all,
    unused_parens,
    unused_imports,
    dead_code,
    unused_variables
)]
mod grammar {
    use lalrpop_util::lalrpop_mod;
    lalrpop_mod!(pub grammar, "/grammar.rs");
    pub use grammar::*;
}

pub use ast::*;
pub use error::ParseError;
pub use parser::{parse_script, parse_template};
pub use span::{Span, Spanned};

/// Parse a template source string into an AST.
pub fn parse(source: &str) -> Result<Template, ParseError> {
    parse_template(source)
}
