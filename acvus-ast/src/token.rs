use acvus_utils::{Astr, Interner};
use logos::Logos;
use std::fmt;

fn parse_string_literal(lex: &mut logos::Lexer<'_, Token>) -> Option<String> {
    let slice = lex.slice();
    let inner = &slice[1..slice.len() - 1];
    let mut result = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'n' => result.push('\n'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '"' => result.push('"'),
                c => {
                    result.push('\\');
                    result.push(c);
                }
            }
        } else {
            result.push(c);
        }
    }
    Some(result)
}

/// Tokens produced by the expression tokenizer, driven by logos.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(extras = Interner)]
pub enum Token {
    // ── Keywords (exact match, higher priority than ident regex) ──
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("in", priority = 3)]
    In,
    #[token("_", priority = 3)]
    Underscore,
    #[token("Some", priority = 3)]
    Some,
    #[token("None", priority = 3)]
    None,

    // ── Identifiers ──
    #[regex(r"[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(lex.slice()), priority = 2)]
    Ident(Astr),

    // ── Variable reference: $name ──
    #[regex(r"\$[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(&lex.slice()[1..]))]
    VarRef(Astr),

    // ── Context reference: @name ──
    #[regex(r"@[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(&lex.slice()[1..]))]
    ContextRef(Astr),

    // ── Literals ──
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntLit(i64),
    #[regex(r#""([^"\\]|\\.)*""#, parse_string_literal)]
    StringLit(String),

    // ── Two-char operators ──
    #[token("::")]
    DoubleColon,
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("==")]
    Eq,
    #[token("!=")]
    Neq,
    #[token("<=")]
    Lte,
    #[token(">=")]
    Gte,
    #[token("->")]
    Arrow,
    #[token("..=")]
    DotDotEq,
    #[token("=..")]
    EqDotDot,
    #[token("..")]
    DotDot,

    // ── Single-char operators ──
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("!")]
    Bang,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("=")]
    Assign,
    #[token(".")]
    Dot,
    #[token("|")]
    Pipe,

    // ── Delimiters ──
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,

    // ── Format string segments (emitted by ExprTokenizer, not by logos) ──
    FmtStringStart(String),
    FmtStringMid(String),
    FmtStringEnd(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::IntLit(n) => write!(f, "{n}"),
            Token::FloatLit(n) => write!(f, "{n}"),
            Token::StringLit(s) => write!(f, "\"{s}\""),
            Token::Ident(_) => write!(f, "<ident>"),
            Token::VarRef(_) => write!(f, "$<ref>"),
            Token::ContextRef(_) => write!(f, "@<ref>"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::In => write!(f, "in"),
            Token::Underscore => write!(f, "_"),
            Token::Some => write!(f, "Some"),
            Token::None => write!(f, "None"),
            Token::DoubleColon => write!(f, "::"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Bang => write!(f, "!"),
            Token::AndAnd => write!(f, "&&"),
            Token::OrOr => write!(f, "||"),
            Token::Eq => write!(f, "=="),
            Token::Neq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Lte => write!(f, "<="),
            Token::Gte => write!(f, ">="),
            Token::Assign => write!(f, "="),
            Token::Arrow => write!(f, "->"),
            Token::DotDotEq => write!(f, "..="),
            Token::EqDotDot => write!(f, "=.."),
            Token::DotDot => write!(f, ".."),
            Token::Dot => write!(f, "."),
            Token::Pipe => write!(f, "|"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::FmtStringStart(s) => write!(f, "fmt_start({s:?})"),
            Token::FmtStringMid(s) => write!(f, "fmt_mid({s:?})"),
            Token::FmtStringEnd(s) => write!(f, "fmt_end({s:?})"),
        }
    }
}
