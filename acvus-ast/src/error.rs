use std::fmt;

use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}

impl ParseError {
    pub fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at {}..{}",
            self.kind, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    // Scanner errors
    UnclosedTag,
    UnclosedComment,
    UnclosedString,

    // Tokenizer errors
    UnexpectedCharacter(char),
    InvalidNumber(String),

    // Grammar errors
    UnexpectedToken(String),
    UnexpectedEof,

    // Tree builder errors
    UnmatchedCloseBlock,
    UnmatchedCatchAll,
    UnclosedBlock,
    ExpectedCloseBlock,

    // Pattern conversion errors
    InvalidPattern(String),
}

impl fmt::Display for ParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseErrorKind::UnclosedTag => write!(f, "unclosed tag, expected `}}}}`"),
            ParseErrorKind::UnclosedComment => {
                write!(f, "unclosed comment, expected `--}}}}`")
            }
            ParseErrorKind::UnclosedString => write!(f, "unclosed string literal"),
            ParseErrorKind::UnexpectedCharacter(c) => write!(f, "unexpected character '{c}'"),
            ParseErrorKind::InvalidNumber(s) => write!(f, "invalid number '{s}'"),
            ParseErrorKind::UnexpectedToken(s) => write!(f, "unexpected token: {s}"),
            ParseErrorKind::UnexpectedEof => write!(f, "unexpected end of input"),
            ParseErrorKind::UnmatchedCloseBlock => {
                write!(f, "`{{{{/}}}}` without matching open block")
            }
            ParseErrorKind::UnmatchedCatchAll => {
                write!(f, "`{{{{_}}}}` without matching open block")
            }
            ParseErrorKind::UnclosedBlock => write!(f, "block not closed, expected `{{{{/}}}}`"),
            ParseErrorKind::ExpectedCloseBlock => write!(f, "expected `{{{{/}}}}`"),
            ParseErrorKind::InvalidPattern(s) => write!(f, "invalid pattern: {s}"),
        }
    }
}
