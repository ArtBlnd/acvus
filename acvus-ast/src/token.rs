use acvus_utils::{Astr, Interner};
use logos::Logos;
use std::fmt;

use crate::literal::{IntWidth, SuffixedInt};

/// The text a quoted literal encloses: the slice without its `open`-byte
/// prefix and its one-byte closing quote.
fn inner_text(lex: &mut logos::Lexer<'_, Token>, open: usize) -> String {
    let slice = lex.slice();
    slice[open..slice.len() - 1].to_owned()
}

/// `10u64`: the digits and the width the suffix names.
fn suffixed_int(lex: &mut logos::Lexer<'_, Token>) -> Option<SuffixedInt> {
    let slice = lex.slice();
    let at = slice.find(|c: char| !c.is_ascii_digit())?;
    let width = IntWidth::of_name(&slice[at..])?;
    Some(SuffixedInt {
        value: slice[..at].parse().ok()?,
        width,
    })
}

/// Tokens produced by the expression tokenizer, driven by logos.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\r]+")]
#[logos(extras = Interner)]
pub enum Token {
    // -- Keywords (exact match, higher priority than ident regex) --
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("_", priority = 3)]
    Underscore,
    #[token("Some", priority = 3)]
    Some,
    #[token("None", priority = 3)]
    None,
    #[token("Ok", priority = 3)]
    Ok,
    #[token("Err", priority = 3)]
    Err,

    // -- Script mode keywords --
    #[token("let", priority = 3)]
    Let,
    #[token("if", priority = 3)]
    If,
    #[token("else", priority = 3)]
    Else,
    #[token("while", priority = 3)]
    While,
    #[token("for", priority = 3)]
    For,
    #[token("in", priority = 3)]
    In,
    #[token("break", priority = 3)]
    Break,
    #[token("continue", priority = 3)]
    Continue,
    #[token("anyorder", priority = 3)]
    Anyorder,
    #[token("match", priority = 3)]
    Match,
    #[token("mut", priority = 3)]
    Mut,
    #[token("as", priority = 3)]
    As,

    // -- Identifiers --
    #[regex(r"[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(lex.slice()), priority = 2)]
    Ident(Astr),

    // -- Extern parameter: $name --
    #[regex(r"\$[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(&lex.slice()[1..]))]
    ParamRef(Astr),

    // -- Context reference: @name --
    #[regex(r"@[\p{L}_][\p{L}\p{N}_]*", |lex| lex.extras.intern(&lex.slice()[1..]))]
    ContextRef(Astr),

    // -- Literals --
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    FloatLit(f64),
    #[regex(r"[0-9]+(i8|i16|i32|i64|u8|u16|u32|u64)", suffixed_int, priority = 4)]
    IntLitOf(SuffixedInt),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i128>().ok())]
    IntLit(i128),
    /// The text between the quotes of `"…"`, undecoded, as `'…'` and
    /// `b"…"` are: the grammar decodes it against the one escape table, so
    /// an escape the table does not name is a parse error carrying the
    /// literal's span. Undecoded text is also what the format-string
    /// scanner needs, because a `{{ }}` tag inside it is then at the
    /// offsets the source has.
    #[regex(r#""([^"\\]|\\.)*""#, |lex| inner_text(lex, 1))]
    StringLit(String),
    /// The text between the quotes of `'…'`, undecoded: the grammar
    /// decodes it, so a bad escape is a parse error carrying the
    /// literal's span rather than an unexpected character. The content
    /// admits no bare `'` and no newline, so a literal ends at the first
    /// quote that is not escaped and `'a' == 'b'` is three tokens.
    #[regex(r"'([^'\\\n]|\\[^\n])*'", |lex| inner_text(lex, 1))]
    CharLit(String),
    /// The text between the quotes of `b'…'`, undecoded.
    #[regex(r"b'([^'\\\n]|\\[^\n])*'", |lex| inner_text(lex, 2))]
    ByteLit(String),
    /// The text between the quotes of `b"…"`, undecoded.
    #[regex(r#"b"([^"\\]|\\.)*""#, |lex| inner_text(lex, 2))]
    ByteStrLit(String),

    // -- Two-char operators --
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
    #[token("=>")]
    FatArrow,
    #[token("..")]
    DotDot,

    // -- Single-char operators --
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
    #[token("?")]
    Question,
    #[token("&")]
    Amp,
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

    // -- Delimiters --
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

    // -- Format string segments (emitted by ExprTokenizer, not by logos) --
    FmtStringStart(String),
    FmtStringMid(String),
    FmtStringEnd(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::IntLit(n) => write!(f, "{n}"),
            Token::IntLitOf(lit) => write!(f, "{}{}", lit.value, lit.width.name()),
            Token::CharLit(s) => write!(f, "'{s}'"),
            Token::ByteLit(s) => write!(f, "b'{s}'"),
            Token::ByteStrLit(s) => write!(f, "b\"{s}\""),
            Token::FloatLit(n) => write!(f, "{n}"),
            Token::StringLit(s) => write!(f, "\"{s}\""),
            Token::Ident(_) => write!(f, "<ident>"),
            Token::ParamRef(_) => write!(f, "$<param>"),
            Token::ContextRef(_) => write!(f, "@<ref>"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Underscore => write!(f, "_"),
            Token::Some => write!(f, "Some"),
            Token::None => write!(f, "None"),
            Token::Ok => write!(f, "Ok"),
            Token::Err => write!(f, "Err"),
            Token::Let => write!(f, "let"),
            Token::If => write!(f, "if"),
            Token::Else => write!(f, "else"),
            Token::While => write!(f, "while"),
            Token::For => write!(f, "for"),
            Token::In => write!(f, "in"),
            Token::Break => write!(f, "break"),
            Token::Continue => write!(f, "continue"),
            Token::Anyorder => write!(f, "anyorder"),
            Token::Match => write!(f, "match"),
            Token::Mut => write!(f, "mut"),
            Token::As => write!(f, "as"),
            Token::Amp => write!(f, "&"),
            Token::DoubleColon => write!(f, "::"),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Percent => write!(f, "%"),
            Token::Bang => write!(f, "!"),
            Token::Question => write!(f, "?"),
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
            Token::FatArrow => write!(f, "=>"),
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
