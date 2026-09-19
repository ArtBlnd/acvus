use std::fmt;

use crate::span::Span;
use crate::token::Token;

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
        write!(f, "{} at {}..{}", self.kind, self.span.start, self.span.end)
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
    /// A `'…'`, `b'…'` or `b"…"` whose text is not what that literal
    /// holds (RFC-0058).
    BadLiteral(crate::literal::LiteralErrorKind),

    // Grammar errors
    UnexpectedToken {
        found: Found,
        expected: Expected,
    },
    ExtraToken {
        found: Found,
    },
    InvalidToken,
    UnexpectedEof,

    // Tree builder errors
    UnmatchedCloseBlock,
    UnmatchedCatchAll,
    UnclosedBlock,
    ExpectedCloseBlock,

    // Pattern conversion errors
    InvalidPattern(String),
    RefutablePattern,

    // Script errors
    InvalidAssignTarget,
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
            ParseErrorKind::BadLiteral(kind) => write!(f, "{kind}"),
            ParseErrorKind::UnexpectedToken { found, expected } => {
                write!(f, "expected {expected}, found {found}")
            }
            ParseErrorKind::ExtraToken { found } => {
                write!(f, "found {found} after the end of the input")
            }
            ParseErrorKind::InvalidToken => write!(f, "not the start of any token"),
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
            ParseErrorKind::RefutablePattern => write!(
                f,
                "refutable pattern not allowed in `in` binding; use `=` for pattern matching"
            ),
            ParseErrorKind::InvalidAssignTarget => write!(
                f,
                "not an assignment target: the left of `=` is a place -- a name, \
                 an `@context` or a `$parameter` under any path of `.field` and \
                 `[index]` steps -- or `*reference`"
            ),
        }
    }
}

/// The token a diagnostic reports: the text the user wrote where the token
/// carries it, and the token's class where the text is a lexer placeholder.
#[derive(Debug, Clone, PartialEq)]
pub struct Found(String);

impl Found {
    pub fn of(token: &Token) -> Self {
        match Terminal::of_token(token).class() {
            Class::Named(name) => Self(name.to_string()),
            Class::Written(_) | Class::Literal(_) => Self(format!("`{token}`")),
        }
    }
}

impl fmt::Display for Found {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the grammar admitted at a span: the nonterminal whose own opening
/// terminals the set covers, and the terminals beyond it.
#[derive(Debug, Clone, PartialEq)]
pub struct Expected {
    covers: Option<Nonterminal>,
    terminals: Vec<Terminal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nonterminal {
    Statement,
    Expression,
    Pattern,
    Literal,
}

impl Nonterminal {
    fn spoken(self) -> &'static str {
        match self {
            Nonterminal::Statement => "a statement",
            Nonterminal::Expression => "an expression",
            Nonterminal::Pattern => "a pattern",
            Nonterminal::Literal => "a literal",
        }
    }
}

impl Expected {
    /// lalrpop reports its expected set as the terminal names
    /// `grammar.lalrpop` gives them, quotes included.
    pub fn of_grammar_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut set: Vec<Terminal> = names
            .into_iter()
            .map(|name| {
                let unquoted = name.trim_matches('"');
                Terminal::of_grammar_name(unquoted).unwrap_or_else(|| {
                    panic!(
                        "the grammar lists terminal {name}, which `Terminal` does not have; \
                         `terminals_match_the_grammar` holds the two in step"
                    )
                })
            })
            .collect();
        set.sort_unstable();
        set.dedup();
        Self::of_set(set)
    }

    fn of_set(set: Vec<Terminal>) -> Self {
        let Some(coverage) = Coverage::of(&set) else {
            return Self {
                covers: None,
                terminals: set,
            };
        };
        Self {
            terminals: set
                .into_iter()
                .filter(|t| !coverage.opening.contains(t))
                .collect(),
            covers: Some(coverage.nonterminal),
        }
    }
}

/// A nonterminal a set covers, with the terminals that nonterminal's own
/// openings account for.
struct Coverage {
    nonterminal: Nonterminal,
    opening: Vec<Terminal>,
}

impl Coverage {
    fn of(set: &[Terminal]) -> Option<Self> {
        let covers = |part: &[Terminal]| part.iter().all(|t| set.contains(t));
        if covers(OPERAND_START) {
            if covers(STATEMENT_KEYWORDS) {
                return Some(Self {
                    nonterminal: Nonterminal::Statement,
                    opening: [OPERAND_START, STATEMENT_KEYWORDS, LAMBDA_START].concat(),
                });
            }
            if set.contains(&Terminal::Underscore) {
                return Some(Self {
                    nonterminal: Nonterminal::Pattern,
                    opening: [OPERAND_START, &[Terminal::Underscore], LAMBDA_START].concat(),
                });
            }
            return Some(Self {
                nonterminal: Nonterminal::Expression,
                opening: [OPERAND_START, LAMBDA_START].concat(),
            });
        }
        match set == LITERALS {
            true => Some(Self {
                nonterminal: Nonterminal::Literal,
                opening: LITERALS.to_vec(),
            }),
            false => None,
        }
    }
}

impl fmt::Display for Expected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = self
            .covers
            .map(|n| n.spoken().to_string())
            .into_iter()
            .collect();
        for spoken in self.terminals.iter().map(|t| t.class().spoken()) {
            if !parts.iter().any(|seen| seen == &spoken) {
                parts.push(spoken);
            }
        }
        let (last, rest) = parts
            .split_last()
            .expect("a lalrpop expected set is nonempty");
        match rest.is_empty() {
            true => f.write_str(last),
            false => write!(f, "{} or {last}", rest.join(", ")),
        }
    }
}

enum Class {
    Written(&'static str),
    Named(&'static str),
    Literal(&'static str),
}

impl Class {
    fn spoken(&self) -> String {
        match self {
            Class::Written(text) => format!("`{text}`"),
            Class::Named(name) | Class::Literal(name) => name.to_string(),
        }
    }
}

/// A terminal of the grammar, by the name `grammar.lalrpop`'s
/// `extern { enum Token { … } }` block gives it.
///
/// That block is the contract between the tokenizer and the parser, and its
/// names are the strings lalrpop puts in an expected set. `of_token` is the
/// compiler's half of the contract: a new `Token` variant does not compile
/// until it has a `Terminal`. The test `terminals_match_the_grammar` is the
/// other half, against the grammar file itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Terminal {
    Int,
    IntOf,
    Char,
    Byte,
    Bytes,
    Float,
    Str,
    Ident,
    ParamRef,
    ContextRef,
    True,
    False,
    Underscore,
    Some,
    None,
    Ok,
    Err,
    Let,
    If,
    Else,
    While,
    For,
    In,
    Break,
    Continue,
    Anyorder,
    Match,
    Mut,
    As,
    Amp,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Bang,
    Question,
    AndAnd,
    OrOr,
    EqEq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    Assign,
    Arrow,
    FatArrow,
    DotDot,
    Dot,
    Pipe,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    DoubleColon,
    Semicolon,
    FmtStart,
    FmtMid,
    FmtEnd,
}

/// Every terminal an operand can begin with.
const OPERAND_START: &[Terminal] = &[
    Terminal::Int,
    Terminal::IntOf,
    Terminal::Char,
    Terminal::Byte,
    Terminal::Bytes,
    Terminal::Float,
    Terminal::Str,
    Terminal::Ident,
    Terminal::ParamRef,
    Terminal::ContextRef,
    Terminal::True,
    Terminal::False,
    Terminal::Some,
    Terminal::None,
    Terminal::Ok,
    Terminal::Err,
    Terminal::If,
    Terminal::Match,
    Terminal::Amp,
    Terminal::Minus,
    Terminal::Star,
    Terminal::Bang,
    Terminal::LParen,
    Terminal::LBracket,
    Terminal::LBrace,
    Terminal::FmtStart,
];

const LAMBDA_START: &[Terminal] = &[Terminal::Pipe];

const STATEMENT_KEYWORDS: &[Terminal] = &[
    Terminal::Let,
    Terminal::While,
    Terminal::For,
    Terminal::Break,
    Terminal::Continue,
    Terminal::Anyorder,
];

const LITERALS: [Terminal; 7] = [
    Terminal::Int,
    Terminal::IntOf,
    Terminal::Char,
    Terminal::Byte,
    Terminal::Bytes,
    Terminal::Float,
    Terminal::Str,
];

impl Terminal {
    pub const ALL: [Terminal; 64] = [
        Terminal::Int,
        Terminal::IntOf,
        Terminal::Char,
        Terminal::Byte,
        Terminal::Bytes,
        Terminal::Float,
        Terminal::Str,
        Terminal::Ident,
        Terminal::ParamRef,
        Terminal::ContextRef,
        Terminal::True,
        Terminal::False,
        Terminal::Underscore,
        Terminal::Some,
        Terminal::None,
        Terminal::Ok,
        Terminal::Err,
        Terminal::Let,
        Terminal::If,
        Terminal::Else,
        Terminal::While,
        Terminal::For,
        Terminal::In,
        Terminal::Break,
        Terminal::Continue,
        Terminal::Anyorder,
        Terminal::Match,
        Terminal::Mut,
        Terminal::As,
        Terminal::Amp,
        Terminal::Plus,
        Terminal::Minus,
        Terminal::Star,
        Terminal::Slash,
        Terminal::Percent,
        Terminal::Bang,
        Terminal::Question,
        Terminal::AndAnd,
        Terminal::OrOr,
        Terminal::EqEq,
        Terminal::Neq,
        Terminal::Lt,
        Terminal::Gt,
        Terminal::Lte,
        Terminal::Gte,
        Terminal::Assign,
        Terminal::Arrow,
        Terminal::FatArrow,
        Terminal::DotDot,
        Terminal::Dot,
        Terminal::Pipe,
        Terminal::LParen,
        Terminal::RParen,
        Terminal::LBracket,
        Terminal::RBracket,
        Terminal::LBrace,
        Terminal::RBrace,
        Terminal::Comma,
        Terminal::Colon,
        Terminal::DoubleColon,
        Terminal::Semicolon,
        Terminal::FmtStart,
        Terminal::FmtMid,
        Terminal::FmtEnd,
    ];

    pub fn of_token(token: &Token) -> Self {
        match token {
            Token::IntLit(_) => Terminal::Int,
            Token::IntLitOf(_) => Terminal::IntOf,
            Token::CharLit(_) => Terminal::Char,
            Token::ByteLit(_) => Terminal::Byte,
            Token::ByteStrLit(_) => Terminal::Bytes,
            Token::FloatLit(_) => Terminal::Float,
            Token::StringLit(_) => Terminal::Str,
            Token::Ident(_) => Terminal::Ident,
            Token::ParamRef(_) => Terminal::ParamRef,
            Token::ContextRef(_) => Terminal::ContextRef,
            Token::True => Terminal::True,
            Token::False => Terminal::False,
            Token::Underscore => Terminal::Underscore,
            Token::Some => Terminal::Some,
            Token::None => Terminal::None,
            Token::Ok => Terminal::Ok,
            Token::Err => Terminal::Err,
            Token::Let => Terminal::Let,
            Token::If => Terminal::If,
            Token::Else => Terminal::Else,
            Token::While => Terminal::While,
            Token::For => Terminal::For,
            Token::In => Terminal::In,
            Token::Break => Terminal::Break,
            Token::Continue => Terminal::Continue,
            Token::Anyorder => Terminal::Anyorder,
            Token::Match => Terminal::Match,
            Token::Mut => Terminal::Mut,
            Token::As => Terminal::As,
            Token::Amp => Terminal::Amp,
            Token::Plus => Terminal::Plus,
            Token::Minus => Terminal::Minus,
            Token::Star => Terminal::Star,
            Token::Slash => Terminal::Slash,
            Token::Percent => Terminal::Percent,
            Token::Bang => Terminal::Bang,
            Token::Question => Terminal::Question,
            Token::AndAnd => Terminal::AndAnd,
            Token::OrOr => Terminal::OrOr,
            Token::Eq => Terminal::EqEq,
            Token::Neq => Terminal::Neq,
            Token::Lt => Terminal::Lt,
            Token::Gt => Terminal::Gt,
            Token::Lte => Terminal::Lte,
            Token::Gte => Terminal::Gte,
            Token::Assign => Terminal::Assign,
            Token::Arrow => Terminal::Arrow,
            Token::FatArrow => Terminal::FatArrow,
            Token::DotDot => Terminal::DotDot,
            Token::Dot => Terminal::Dot,
            Token::Pipe => Terminal::Pipe,
            Token::LParen => Terminal::LParen,
            Token::RParen => Terminal::RParen,
            Token::LBracket => Terminal::LBracket,
            Token::RBracket => Terminal::RBracket,
            Token::LBrace => Terminal::LBrace,
            Token::RBrace => Terminal::RBrace,
            Token::Comma => Terminal::Comma,
            Token::Colon => Terminal::Colon,
            Token::DoubleColon => Terminal::DoubleColon,
            Token::Semicolon => Terminal::Semicolon,
            Token::FmtStringStart(_) => Terminal::FmtStart,
            Token::FmtStringMid(_) => Terminal::FmtMid,
            Token::FmtStringEnd(_) => Terminal::FmtEnd,
        }
    }

    fn of_grammar_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.grammar_name() == name)
    }

    pub fn grammar_name(self) -> &'static str {
        match self {
            Terminal::Int => "int",
            Terminal::IntOf => "int_of",
            Terminal::Char => "char",
            Terminal::Byte => "byte",
            Terminal::Bytes => "bytes",
            Terminal::Float => "float",
            Terminal::Str => "string",
            Terminal::Ident => "ident",
            Terminal::ParamRef => "$ref",
            Terminal::ContextRef => "@ref",
            Terminal::True => "true",
            Terminal::False => "false",
            Terminal::Underscore => "_",
            Terminal::Some => "Some",
            Terminal::None => "None",
            Terminal::Ok => "Ok",
            Terminal::Err => "Err",
            Terminal::Let => "let",
            Terminal::If => "if",
            Terminal::Else => "else",
            Terminal::While => "while",
            Terminal::For => "for",
            Terminal::In => "in",
            Terminal::Break => "break",
            Terminal::Continue => "continue",
            Terminal::Anyorder => "anyorder",
            Terminal::Match => "match",
            Terminal::Mut => "mut",
            Terminal::As => "as",
            Terminal::Amp => "&",
            Terminal::Plus => "+",
            Terminal::Minus => "-",
            Terminal::Star => "*",
            Terminal::Slash => "/",
            Terminal::Percent => "%",
            Terminal::Bang => "!",
            Terminal::Question => "?",
            Terminal::AndAnd => "&&",
            Terminal::OrOr => "||",
            Terminal::EqEq => "==",
            Terminal::Neq => "!=",
            Terminal::Lt => "<",
            Terminal::Gt => ">",
            Terminal::Lte => "<=",
            Terminal::Gte => ">=",
            Terminal::Assign => "=",
            Terminal::Arrow => "->",
            Terminal::FatArrow => "=>",
            Terminal::DotDot => "..",
            Terminal::Dot => ".",
            Terminal::Pipe => "|",
            Terminal::LParen => "(",
            Terminal::RParen => ")",
            Terminal::LBracket => "[",
            Terminal::RBracket => "]",
            Terminal::LBrace => "{",
            Terminal::RBrace => "}",
            Terminal::Comma => ",",
            Terminal::Colon => ":",
            Terminal::DoubleColon => "::",
            Terminal::Semicolon => ";",
            Terminal::FmtStart => "fmt_start",
            Terminal::FmtMid => "fmt_mid",
            Terminal::FmtEnd => "fmt_end",
        }
    }

    fn class(self) -> Class {
        match self {
            Terminal::Int | Terminal::IntOf | Terminal::Float => Class::Literal("a number"),
            Terminal::Char => Class::Literal("a character"),
            Terminal::Byte => Class::Literal("a byte"),
            Terminal::Bytes => Class::Literal("a byte string"),
            Terminal::Str => Class::Literal("a string"),
            Terminal::Ident => Class::Named("a name"),
            Terminal::ParamRef => Class::Named("a `$parameter`"),
            Terminal::ContextRef => Class::Named("an `@context`"),
            Terminal::FmtStart | Terminal::FmtMid | Terminal::FmtEnd => {
                Class::Named("a format string")
            }
            other => Class::Written(other.grammar_name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    /// The names `grammar.lalrpop`'s `extern { enum Token { … } }` block
    /// declares, which are the strings lalrpop reports as its expected set.
    fn grammar_terminal_names() -> Vec<String> {
        let grammar = include_str!("grammar.lalrpop");
        let block = grammar
            .split_once("enum Token {")
            .expect("the extern block declares the token enum")
            .1
            .split_once("\n    }")
            .expect("the token enum is closed")
            .0;
        block
            .lines()
            .filter_map(|line| line.trim().strip_prefix('"'))
            .map(|rest| {
                rest.split_once('"')
                    .expect("a terminal name is quoted")
                    .0
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn terminals_match_the_grammar() {
        let mut from_grammar = grammar_terminal_names();
        let mut from_table: Vec<String> = Terminal::ALL
            .iter()
            .map(|t| t.grammar_name().to_string())
            .collect();
        assert_eq!(from_grammar.len(), from_table.len());
        from_grammar.sort();
        from_table.sort();
        assert_eq!(from_grammar, from_table);
    }

    #[test]
    fn every_token_has_a_terminal_and_every_terminal_a_class() {
        let interner = Interner::new();
        let name = interner.intern("x");
        let tokens = [
            Token::IntLit(1),
            Token::IntLitOf(crate::literal::SuffixedInt {
                value: 1,
                width: crate::literal::IntWidth::I64,
            }),
            Token::CharLit("c".into()),
            Token::ByteLit("b".into()),
            Token::ByteStrLit("bs".into()),
            Token::FloatLit(1.0),
            Token::StringLit("s".into()),
            Token::Ident(name),
            Token::ParamRef(name),
            Token::ContextRef(name),
            Token::True,
            Token::False,
            Token::Underscore,
            Token::Some,
            Token::None,
            Token::Ok,
            Token::Err,
            Token::Let,
            Token::If,
            Token::Else,
            Token::While,
            Token::For,
            Token::In,
            Token::Break,
            Token::Continue,
            Token::Anyorder,
            Token::Match,
            Token::Mut,
            Token::As,
            Token::Amp,
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Percent,
            Token::Bang,
            Token::Question,
            Token::AndAnd,
            Token::OrOr,
            Token::Eq,
            Token::Neq,
            Token::Lt,
            Token::Gt,
            Token::Lte,
            Token::Gte,
            Token::Assign,
            Token::Arrow,
            Token::FatArrow,
            Token::DotDot,
            Token::Dot,
            Token::Pipe,
            Token::LParen,
            Token::RParen,
            Token::LBracket,
            Token::RBracket,
            Token::LBrace,
            Token::RBrace,
            Token::Comma,
            Token::Colon,
            Token::DoubleColon,
            Token::Semicolon,
            Token::FmtStringStart("f".into()),
            Token::FmtStringMid("f".into()),
            Token::FmtStringEnd("f".into()),
        ];
        let mut reached: Vec<Terminal> = tokens.iter().map(Terminal::of_token).collect();
        reached.sort_unstable();
        reached.dedup();
        assert_eq!(reached, Terminal::ALL.to_vec());
        for t in Terminal::ALL {
            assert!(!t.class().spoken().is_empty(), "{t:?}");
            assert_eq!(Terminal::of_grammar_name(t.grammar_name()), Some(t));
        }
    }

    /// Each set is one lalrpop reported for the source in the name.
    #[test]
    fn a_set_stands_for_the_nonterminal_it_covers() {
        let spoken = |set: &str| -> String {
            Expected::of_grammar_names(set.split(' ').filter(|n| !n.is_empty())).to_string()
        };
        let operand = r#""int" "int_of" "char" "byte" "bytes" "float" "string" "ident" "$ref" "@ref" "true" "false" "Some" "None" "Ok" "Err" "if" "match" "&" "-" "*" "!" "(" "[" "{" "fmt_start""#;
        // `let x = 1 +;`
        assert_eq!(spoken(operand), "an expression");
        // `let x = ;`
        assert_eq!(spoken(&format!(r#"{operand} "|""#)), "an expression");
        // `let x = 1; }`
        assert_eq!(
            spoken(&format!(
                r#"{operand} "|" "let" "while" "for" "break" "continue" "anyorder""#
            )),
            "a statement"
        );
        // `match 1 { => 1 }`
        assert_eq!(
            spoken(&format!(r#"{operand} "_" "|" "}}""#)),
            "a pattern or `}`"
        );
        // `let x = [1,;`
        assert_eq!(
            spoken(&format!(r#"{operand} ".." "|" "]""#)),
            "an expression, `..` or `]`"
        );
        // `{ a: 1, ;`
        assert_eq!(
            spoken(r#""ident" "$ref" "@ref" "}""#),
            "a name, a `$parameter`, an `@context` or `}`"
        );
        // `let 1 = 2;`
        assert_eq!(spoken(r#""ident""#), "a name");
        // `let x = (1;`
        assert_eq!(spoken(r#"")" ",""#), "`)` or `,`");
        assert_eq!(
            spoken(r#""int" "int_of" "char" "byte" "bytes" "float" "string""#),
            "a literal"
        );
    }

    #[test]
    fn a_found_token_is_its_text_or_its_class() {
        let interner = Interner::new();
        assert_eq!(Found::of(&Token::Semicolon).to_string(), "`;`");
        assert_eq!(Found::of(&Token::IntLit(2)).to_string(), "`2`");
        assert_eq!(
            Found::of(&Token::Ident(interner.intern("x"))).to_string(),
            "a name"
        );
    }
}
