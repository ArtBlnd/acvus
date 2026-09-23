use std::collections::VecDeque;

use acvus_utils::Interner;
use logos::Logos;

use crate::error::{ParseError, ParseErrorKind};
use crate::literal::SuffixedInt;
use crate::span::Span;
use crate::token::Token;

// -- Phase 1: Template line scanner -------------------------------------

/// One piece of a template text line: bytes to append as written, or a
/// `{{ }}` tag whose expression is appended in their place.
#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Text { value: String, span: Span },
    Tag {
        content: String,
        span: Span,
        inner_span: Span,
    },
}

/// One line of a template (RFC-0071).
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    /// A line whose first non-blank character is `%`: one statement of the
    /// script grammar, without its `;` and without the braces that would
    /// open or close a block. `content` is what followed the `%`.
    Stmt { content: String, span: Span },
    /// Every other line: text, appended as written.
    Text { pieces: Vec<Piece>, span: Span },
}

/// Classify a template source into lines.
///
/// A line whose first non-blank character is `%` is a statement line. A
/// line beginning with `%%` is a text line holding one `%`. A text line
/// ending in `\` is appended without its newline; every other text line
/// carries its newline.
pub fn scan_template(source: &str) -> Result<Vec<Line>, ParseError> {
    let mut lines = Vec::new();
    let mut at = 0;

    while at < source.len() {
        let newline = source[at..].find('\n').map(|i| at + i);
        let content_end = newline.unwrap_or(source.len());
        let line = &source[at..content_end];
        let indent = line
            .bytes()
            .position(|b| b != b' ' && b != b'\t')
            .unwrap_or(line.len());

        if !line.starts_with("%%") && line.as_bytes().get(indent) == Some(&b'%') {
            let start = at + indent + 1;
            lines.push(Line::Stmt {
                content: source[start..content_end].to_string(),
                span: Span::new(start, content_end),
            });
        } else {
            let continued = line.ends_with('\\');
            let escaped_percent = line.starts_with("%%");
            let start = at + if escaped_percent { 2 } else { 0 };
            let end = content_end - usize::from(continued);
            let mut pieces = split_tags(source, start, end)?;
            let edges = LineEdges {
                opening: if escaped_percent { "%" } else { "" },
                closing: match newline.is_some() && !continued {
                    true => "\n",
                    false => "",
                },
            };
            edges.wrap(&mut pieces, Span::new(start, end));
            lines.push(Line::Text {
                pieces,
                span: Span::new(at, content_end),
            });
        }

        at = newline.map_or(source.len(), |nl| nl + 1);
    }

    Ok(lines)
}

/// The text a line owns outside its own bytes: the `%` a leading `%%`
/// stands for, and the newline the line carries where it did not end in
/// `\` (RFC-0071 rule 2).
struct LineEdges {
    opening: &'static str,
    closing: &'static str,
}

impl LineEdges {
    fn wrap(&self, pieces: &mut Vec<Piece>, span: Span) {
        if !self.opening.is_empty() {
            match pieces.first_mut() {
                Some(Piece::Text { value, .. }) => value.insert_str(0, self.opening),
                _ => pieces.insert(
                    0,
                    Piece::Text {
                        value: self.opening.to_string(),
                        span,
                    },
                ),
            }
        }
        if !self.closing.is_empty() {
            match pieces.last_mut() {
                Some(Piece::Text { value, .. }) => value.push_str(self.closing),
                _ => pieces.push(Piece::Text {
                    value: self.closing.to_string(),
                    span,
                }),
            }
        }
    }
}

/// Split `source[start..end]` into text runs and `{{ }}` tags.
fn split_tags(source: &str, start: usize, end: usize) -> Result<Vec<Piece>, ParseError> {
    let bytes = source.as_bytes();
    let mut pieces = Vec::new();
    let mut text_from = start;
    let mut at = start;

    while at < end {
        if !(bytes[at] == b'{' && at + 1 < end && bytes[at + 1] == b'{') {
            at += 1;
            continue;
        }
        let Some(close) = close_of_tag(bytes, at + 2, end) else {
            return Err(ParseError::new(
                ParseErrorKind::UnclosedTag,
                Span::new(at, end),
            ));
        };
        if text_from < at {
            pieces.push(Piece::Text {
                value: source[text_from..at].to_string(),
                span: Span::new(text_from, at),
            });
        }
        let inner = &source[at + 2..close];
        let leading = inner.len() - inner.trim_start().len();
        let trailing = inner.len() - inner.trim_end().len();
        pieces.push(Piece::Tag {
            content: inner.trim().to_string(),
            span: Span::new(at, close + 2),
            inner_span: Span::new(at + 2 + leading, close - trailing),
        });
        at = close + 2;
        text_from = at;
    }

    if text_from < end {
        pieces.push(Piece::Text {
            value: source[text_from..end].to_string(),
            span: Span::new(text_from, end),
        });
    }
    Ok(pieces)
}

/// The offset of the `}}` that closes a tag whose content starts at `from`,
/// with the content's literals and its nested braces stepped over, so a
/// `}}` inside a string literal or an object literal's `}` is not the
/// tag's end (RFC-0071 rule 3). `None` where no `}}` closes it before
/// `limit`.
fn close_of_tag(bytes: &[u8], from: usize, limit: usize) -> Option<usize> {
    let mut at = from;
    let mut depth = 0u32;
    while at < limit {
        if let Some(after) = char_literal_end(bytes, at) {
            at = after;
            continue;
        }
        match bytes[at] {
            b'"' => at = string_literal_end(bytes, at, limit)?,
            b'{' => {
                depth += 1;
                at += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                at += 1;
            }
            b'}' if at + 1 < limit && bytes[at + 1] == b'}' => return Some(at),
            _ => at += 1,
        }
    }
    None
}

/// One past the closing quote of the `"…"` starting at `at`.
fn string_literal_end(bytes: &[u8], at: usize, limit: usize) -> Option<usize> {
    let mut i = at + 1;
    while i < limit {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

/// The longest text a `'…'` literal can be: `'\u{10FFFF}'`.
const CHAR_LITERAL_MAX: usize = 12;

/// One past the closing quote of the `'…'` starting at `pos`, so the tag
/// scanner steps over a `"` or a `}}` a character literal holds. `None`
/// where `pos` is not a quote or no closing quote follows within the
/// longest a literal can be, which leaves an apostrophe in a tag exactly
/// where it was.
fn char_literal_end(bytes: &[u8], pos: usize) -> Option<usize> {
    if bytes[pos] != b'\'' {
        return None;
    }
    let mut at = pos + 1;
    while at < bytes.len() && at <= pos + CHAR_LITERAL_MAX {
        match bytes[at] {
            b'\\' => at += 2,
            b'\n' => return None,
            b'\'' => return Some(at + 1),
            _ => at += 1,
        }
    }
    None
}

// -- Phase 2: Expression Tokenizer (logos-backed) -----------------------

/// What logos produced for one stretch of the input.
type Lexed = (Result<Token, ()>, std::ops::Range<usize>);

/// Whether a token can be the last token of an expression, which is what
/// decides a following `-`: after one, the minus is the subtraction
/// operator, and everywhere else it is the sign of the literal it touches
/// (`Signed`).
fn ends_a_value(token: &Token) -> bool {
    match token {
        Token::Ident(_)
        | Token::ParamRef(_)
        | Token::ContextRef(_)
        | Token::IntLit(_)
        | Token::IntLitOf(_)
        | Token::FloatLit(_)
        | Token::StringLit(_)
        | Token::CharLit(_)
        | Token::ByteLit(_)
        | Token::ByteStrLit(_)
        | Token::FmtStringEnd(_)
        | Token::True
        | Token::False
        | Token::None
        | Token::Underscore
        | Token::Question
        | Token::RParen
        | Token::RBracket
        | Token::RBrace => true,
        Token::Some
        | Token::Ok
        | Token::Err
        | Token::Let
        | Token::If
        | Token::Else
        | Token::While
        | Token::For
        | Token::In
        | Token::Break
        | Token::Continue
        | Token::Return
        | Token::Anyorder
        | Token::Match
        | Token::Mut
        | Token::As
        | Token::FmtStringStart(_)
        | Token::FmtStringMid(_)
        | Token::DoubleColon
        | Token::AndAnd
        | Token::OrOr
        | Token::Eq
        | Token::Neq
        | Token::Lte
        | Token::Gte
        | Token::Arrow
        | Token::FatArrow
        | Token::DotDot
        | Token::Plus
        | Token::Minus
        | Token::Star
        | Token::Slash
        | Token::Percent
        | Token::Bang
        | Token::Amp
        | Token::Lt
        | Token::Gt
        | Token::Assign
        | Token::Dot
        | Token::Pipe
        | Token::LParen
        | Token::LBracket
        | Token::LBrace
        | Token::Comma
        | Token::Colon
        | Token::Semicolon => false,
    }
}

/// logos' tokens, with a `-` folded into the integer literal it touches
/// (RFC-0058): `-128i8` is the one `i8` literal whose value is the width's
/// minimum, and the range check then sees the signed value.
///
/// The rule is the token pair's own: the minus ends where the digits begin,
/// and the token before it does not end a value.
///
/// The grammar cannot carry the rule. A production for `"-" "int"` beside
/// `"-" UnaryExpr` is 25 local ambiguities, because after `- 1` with `+`
/// ahead both a negative literal and a negation of a literal parse, and
/// lalrpop has no location to compare before it must choose.
struct Signed<'input> {
    inner: logos::SpannedIter<'input, Token>,
    input: &'input str,
    base_offset: usize,
    held: Option<Lexed>,
    after_value: bool,
}

impl<'input> Signed<'input> {
    fn new(input: &'input str, base_offset: usize, interner: &Interner) -> Self {
        Self {
            inner: Token::lexer_with_extras(input, interner.clone()).spanned(),
            input,
            base_offset,
            held: None,
            after_value: false,
        }
    }

    /// The minus and the integer literal it touches as one literal token,
    /// or the minus alone with what followed it held for the next call.
    fn signed(&mut self, minus: std::ops::Range<usize>) -> Lexed {
        let Some((lexed, span)) = self.inner.next() else {
            return (Ok(Token::Minus), minus);
        };
        let folded = match lexed {
            Ok(Token::IntLit(value)) if span.start == minus.end => Token::IntLit(-value),
            Ok(Token::IntLitOf(lit)) if span.start == minus.end => Token::IntLitOf(SuffixedInt {
                value: -lit.value,
                width: lit.width,
            }),
            other => {
                self.held = Some((other, span));
                return (Ok(Token::Minus), minus);
            }
        };
        (Ok(folded), minus.start..span.end)
    }
}

impl Iterator for Signed<'_> {
    type Item = Result<(usize, Token, usize), ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let (lexed, span) = self.held.take().or_else(|| self.inner.next())?;
        let (lexed, span) = match lexed {
            Ok(Token::Minus) if !self.after_value => self.signed(span),
            lexed => (lexed, span),
        };

        self.after_value = matches!(&lexed, Ok(token) if ends_a_value(token));
        let start = self.base_offset + span.start;
        let end = self.base_offset + span.end;
        Some(match lexed {
            Ok(token) => Ok((start, token, end)),
            Err(()) => {
                let c = self.input[span.start..]
                    .chars()
                    .next()
                    .expect("lexer error token must contain at least one character");
                Err(ParseError::new(
                    ParseErrorKind::UnexpectedCharacter(c),
                    Span::new(start, end),
                ))
            }
        })
    }
}

/// Tokenizer for expression content within `{{ }}` tags.
/// Produces `(start, Token, end)` triples for LALRPOP.
///
/// When a `StringLit` containing `{{` is encountered, it is expanded into
/// `FmtStringStart`, inner expression tokens, optional `FmtStringMid` segments,
/// and a final `FmtStringEnd`.
pub struct ExprTokenizer<'input> {
    tokens: Signed<'input>,
    pending: VecDeque<Result<(usize, Token, usize), ParseError>>,
    interner: Interner,
}

impl<'input> ExprTokenizer<'input> {
    pub fn new(input: &'input str, base_offset: usize, interner: &Interner) -> Self {
        Self {
            tokens: Signed::new(input, base_offset, interner),
            pending: VecDeque::new(),
            interner: interner.clone(),
        }
    }
}

impl Iterator for ExprTokenizer<'_> {
    type Item = Result<(usize, Token, usize), ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.pending.pop_front() {
            return Some(item);
        }

        match self.tokens.next()? {
            Ok((start, Token::StringLit(text), end)) if text.contains("{{") => {
                self.pending = expand_format_string(&text, start, end, &self.interner);
                self.pending.pop_front()
            }
            item => Some(item),
        }
    }
}

/// One `{{ … }}` of a format string: the expression's text and where in the
/// literal's content it began.
struct Interpolation {
    at: usize,
    source: String,
}

/// Expand a format string (e.g. `hello {{ name }}`) into LALRPOP tokens.
///
/// The input `content` is the undecoded string body (from `StringLit`), so a
/// segment's offsets are the source's and the grammar decodes each segment
/// against the one escape table.
/// `base_start`/`base_end` are absolute offsets of the original `StringLit` token.
///
/// The tokens are `FmtStringStart(text0) <expr0> FmtStringMid(text1) <expr1>
/// … FmtStringEnd(textn)`. A `{{` the literal never closes is not an
/// interpolation: it is the two characters, which is how `{{ "{{" }}`
/// writes a literal `{{` (RFC-0071 rule 3). A literal holding no
/// interpolation at all comes back as the `StringLit` it was.
fn expand_format_string(
    content: &str,
    base_start: usize,
    base_end: usize,
    interner: &Interner,
) -> VecDeque<Result<(usize, Token, usize), ParseError>> {
    let mut texts: Vec<String> = Vec::new();
    let mut exprs: Vec<Interpolation> = Vec::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut text_start = 0;

    while pos < len {
        if !(bytes[pos] == b'{' && pos + 1 < len && bytes[pos + 1] == b'{') {
            pos += 1;
            continue;
        }
        match close_of_tag(bytes, pos + 2, len) {
            Some(close) => {
                texts.push(content[text_start..pos].to_string());
                exprs.push(Interpolation {
                    at: pos + 2,
                    source: content[pos + 2..close].to_string(),
                });
                pos = close + 2;
                text_start = pos;
            }
            None => pos += 2,
        }
    }

    if exprs.is_empty() {
        return VecDeque::from([Ok((
            base_start,
            Token::StringLit(content.to_string()),
            base_end,
        ))]);
    }

    texts.push(content[text_start..].to_string());

    let quote_offset = 1;
    let last_text_idx = texts.len() - 1;
    let mut out = VecDeque::new();
    let mut cursor = base_start + quote_offset;

    for (i, text) in texts.into_iter().enumerate() {
        let tok_end = cursor + text.len();
        let tok = match i {
            0 => Token::FmtStringStart(text),
            n if n == last_text_idx => Token::FmtStringEnd(text),
            _ => Token::FmtStringMid(text),
        };
        out.push_back(Ok((cursor, tok, tok_end)));
        cursor = tok_end;

        if let Some(expr) = exprs.get(i) {
            let expr_at = base_start + quote_offset + expr.at;
            out.extend(Signed::new(&expr.source, expr_at, interner));
            cursor = expr_at + expr.source.len() + 2;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::IntWidth;

    // -- Line scanner --

    fn text_of(line: &Line) -> String {
        let Line::Text { pieces, .. } = line else {
            panic!("{line:?}");
        };
        pieces
            .iter()
            .map(|p| match p {
                Piece::Text { value, .. } => value.clone(),
                Piece::Tag { content, .. } => format!("{{{{{content}}}}}"),
            })
            .collect()
    }

    #[test]
    fn a_line_without_a_percent_is_text_with_its_newline() {
        let lines = scan_template("hello world\nsecond\n").unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(text_of(&lines[0]), "hello world\n");
        assert_eq!(text_of(&lines[1]), "second\n");
    }

    #[test]
    fn the_last_line_without_a_newline_carries_none() {
        let lines = scan_template("hello").unwrap();
        assert_eq!(text_of(&lines[0]), "hello");
    }

    #[test]
    fn a_first_non_blank_percent_is_a_statement_line() {
        let lines = scan_template("    % let x = 1\n").unwrap();
        assert_eq!(
            lines,
            vec![Line::Stmt {
                content: " let x = 1".to_string(),
                span: Span::new(5, 15),
            }]
        );
    }

    #[test]
    fn a_double_percent_is_a_text_line_holding_one() {
        let lines = scan_template("%% not a statement\n").unwrap();
        assert_eq!(text_of(&lines[0]), "% not a statement\n");
    }

    #[test]
    fn a_trailing_backslash_drops_the_newline() {
        let lines = scan_template("a\\\nb\n").unwrap();
        assert_eq!(text_of(&lines[0]), "a");
        assert_eq!(text_of(&lines[1]), "b\n");
    }

    #[test]
    fn a_blank_line_is_its_newline() {
        let lines = scan_template("a\n\nb").unwrap();
        assert_eq!(text_of(&lines[1]), "\n");
    }

    #[test]
    fn a_tag_splits_the_line() {
        let lines = scan_template("hello {{ name }} world").unwrap();
        let Line::Text { pieces, .. } = &lines[0] else {
            panic!();
        };
        assert_eq!(pieces.len(), 3);
        assert!(matches!(&pieces[0], Piece::Text { value, .. } if value == "hello "));
        assert!(matches!(&pieces[1], Piece::Tag { content, .. } if content == "name"));
        assert!(matches!(&pieces[2], Piece::Text { value, .. } if value == " world"));
    }

    /// The tag's content is tokenized, so a string literal inside it may
    /// hold `{{` or `}}` and an object literal's `}` is not the end.
    #[test]
    fn a_literal_inside_a_tag_does_not_close_it() {
        assert_eq!(text_of(&scan_template(r#"{{ "a}}b" }}"#).unwrap()[0]), r#"{{"a}}b"}}"#);
        assert_eq!(text_of(&scan_template(r#"{{ "{{" }}"#).unwrap()[0]), r#"{{"{{"}}"#);
        assert_eq!(
            text_of(&scan_template("{{ f({ g: 1, }) }}").unwrap()[0]),
            "{{f({ g: 1, })}}"
        );
    }

    #[test]
    fn a_tag_the_line_does_not_close_is_refused() {
        assert_eq!(
            scan_template("{{ hello").unwrap_err().kind,
            ParseErrorKind::UnclosedTag
        );
    }

    // -- Tokenizer Tests --

    #[test]
    fn tokenize_ident() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("name", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::Ident(s) if interner.resolve(*s) == "name"));
    }

    #[test]
    fn tokenize_var_ref() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("$global", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::ParamRef(s) if interner.resolve(*s) == "global"));
    }

    #[test]
    fn tokenize_context_ref() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("@users", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::ContextRef(s) if interner.resolve(*s) == "users"));
    }

    #[test]
    fn tokenize_underscore() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("_", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::Underscore));
    }

    #[test]
    fn tokenize_keywords() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("true false", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(&tokens[0].1, Token::True));
        assert!(matches!(&tokens[1].1, Token::False));
    }

    #[test]
    fn tokenize_numbers() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("42 3.14", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(&tokens[0].1, Token::IntLit(42)));
        assert!(matches!(&tokens[1].1, Token::FloatLit(f) if (*f - 3.14).abs() < f64::EPSILON));
    }

    /// The token carries the text between the quotes as the source spells
    /// it; the grammar decodes it (`literal::decode_str`).
    #[test]
    fn tokenize_string() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new(r#""hello \"world\"""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::StringLit(s) if s == r#"hello \"world\""#));
    }

    fn tokens_of(interner: &Interner, source: &str) -> Vec<Token> {
        ExprTokenizer::new(source, 0, interner)
            .map(|item| item.expect(source).1)
            .collect()
    }

    /// The signed value is what `acvus-mir`'s range check reads, so
    /// `-128i8` arrives there as `-128` and fits (RFC-0058).
    #[test]
    fn a_minus_on_a_literal_is_the_literals_sign() {
        let i = Interner::new();
        assert_eq!(tokens_of(&i, "-1"), vec![Token::IntLit(-1)]);
        assert_eq!(
            tokens_of(&i, "-128i8"),
            vec![Token::IntLitOf(SuffixedInt {
                value: -128,
                width: IntWidth::I8,
            })]
        );
        assert_eq!(
            tokens_of(&i, "f(-1)"),
            vec![
                Token::Ident(i.intern("f")),
                Token::LParen,
                Token::IntLit(-1),
                Token::RParen,
            ]
        );
        assert_eq!(
            tokens_of(&i, "2 * -1"),
            vec![Token::IntLit(2), Token::Star, Token::IntLit(-1)]
        );
        assert_eq!(
            tokens_of(&i, "[1, -1]"),
            vec![
                Token::LBracket,
                Token::IntLit(1),
                Token::Comma,
                Token::IntLit(-1),
                Token::RBracket,
            ]
        );
    }

    #[test]
    fn a_minus_after_a_value_or_a_space_is_subtraction() {
        let i = Interner::new();
        assert_eq!(
            tokens_of(&i, "- 1"),
            vec![Token::Minus, Token::IntLit(1)],
            "a space between them is two tokens"
        );
        assert_eq!(
            tokens_of(&i, "1 - 1"),
            vec![Token::IntLit(1), Token::Minus, Token::IntLit(1)]
        );
        assert_eq!(
            tokens_of(&i, "1-1"),
            vec![Token::IntLit(1), Token::Minus, Token::IntLit(1)]
        );
        assert_eq!(
            tokens_of(&i, "(1) -1"),
            vec![
                Token::LParen,
                Token::IntLit(1),
                Token::RParen,
                Token::Minus,
                Token::IntLit(1),
            ]
        );
        assert_eq!(
            tokens_of(&i, "-x"),
            vec![Token::Minus, Token::Ident(i.intern("x"))],
            "a name is not a literal"
        );
        assert_eq!(
            tokens_of(&i, "-1.5"),
            vec![Token::Minus, Token::FloatLit(1.5)],
            "a float literal has no suffix and no fold"
        );
    }

    /// A `{{ }}` tag inside a format string is tokenized by the same pass,
    /// so the fold holds there too.
    #[test]
    fn a_format_string_folds_the_sign_as_well() {
        let i = Interner::new();
        assert_eq!(
            tokens_of(&i, r#""at {{ -1 }}""#),
            vec![
                Token::FmtStringStart("at ".into()),
                Token::IntLit(-1),
                Token::FmtStringEnd(String::new()),
            ]
        );
    }

    #[test]
    fn tokenize_two_char_operators() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("== != <= >= -> ..", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 6);
        assert!(matches!(&tokens[0].1, Token::Eq));
        assert!(matches!(&tokens[1].1, Token::Neq));
        assert!(matches!(&tokens[2].1, Token::Lte));
        assert!(matches!(&tokens[3].1, Token::Gte));
        assert!(matches!(&tokens[4].1, Token::Arrow));
        assert!(matches!(&tokens[5].1, Token::DotDot));
    }

    #[test]
    fn tokenize_complex_expr() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("list | filter(|x| -> x != 0)", 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::Ident(s) if interner.resolve(*s) == "list"));
        assert!(matches!(types[1], Token::Pipe));
        assert!(matches!(types[2], Token::Ident(s) if interner.resolve(*s) == "filter"));
        assert!(matches!(types[3], Token::LParen));
        assert!(matches!(types[4], Token::Pipe));
        assert!(matches!(types[5], Token::Ident(s) if interner.resolve(*s) == "x"));
        assert!(matches!(types[6], Token::Pipe));
        assert!(matches!(types[7], Token::Arrow));
        assert!(matches!(types[8], Token::Ident(s) if interner.resolve(*s) == "x"));
        assert!(matches!(types[9], Token::Neq));
        assert!(matches!(types[10], Token::IntLit(0)));
        assert!(matches!(types[11], Token::RParen));
    }

    #[test]
    fn tokenize_absolute_offsets() {
        let interner = Interner::new();
        let tokens: Vec<_> = ExprTokenizer::new("ab", 10, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens[0].0, 10); // start
        assert_eq!(tokens[0].2, 12); // end
    }

    // -- Format String Tokenizer Tests --

    #[test]
    fn tokenize_fmt_simple() {
        let interner = Interner::new();
        // "hello {{ name }}!" -> FmtStringStart("hello "), Ident("name"), FmtStringEnd("!")
        let tokens: Vec<_> = ExprTokenizer::new(r#""hello {{ name }}!""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::FmtStringStart(s) if s == "hello "));
        assert!(matches!(types[1], Token::Ident(s) if interner.resolve(*s) == "name"));
        assert!(matches!(types[2], Token::FmtStringEnd(s) if s == "!"));
    }

    #[test]
    fn tokenize_fmt_multiple_interpolations() {
        let interner = Interner::new();
        // "{{ a }}, {{ b }}" -> FmtStringStart(""), Ident(a), FmtStringMid(", "), Ident(b), FmtStringEnd("")
        let tokens: Vec<_> = ExprTokenizer::new(r#""{{ a }}, {{ b }}""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::FmtStringStart(s) if s.is_empty()));
        assert!(matches!(types[1], Token::Ident(s) if interner.resolve(*s) == "a"));
        assert!(matches!(types[2], Token::FmtStringMid(s) if s == ", "));
        assert!(matches!(types[3], Token::Ident(s) if interner.resolve(*s) == "b"));
        assert!(matches!(types[4], Token::FmtStringEnd(s) if s.is_empty()));
    }

    #[test]
    fn tokenize_fmt_expr_with_pipe() {
        let interner = Interner::new();
        // "age: {{ age | to_string }}" -> FmtStringStart("age: "), Ident(age), Pipe, Ident(to_string), FmtStringEnd("")
        let tokens: Vec<_> = ExprTokenizer::new(r#""age: {{ age | to_string }}""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::FmtStringStart(s) if s == "age: "));
        assert!(matches!(types[1], Token::Ident(s) if interner.resolve(*s) == "age"));
        assert!(matches!(types[2], Token::Pipe));
        assert!(matches!(types[3], Token::Ident(s) if interner.resolve(*s) == "to_string"));
        assert!(matches!(types[4], Token::FmtStringEnd(s) if s.is_empty()));
    }

    #[test]
    fn tokenize_fmt_no_interpolation_passthrough() {
        let interner = Interner::new();
        // "hello world" without {{ }} -> plain StringLit
        let tokens: Vec<_> = ExprTokenizer::new(r#""hello world""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::StringLit(s) if s == "hello world"));
    }

    #[test]
    fn tokenize_fmt_expr_with_add() {
        let interner = Interner::new();
        // "result: {{ a + b }}" -> FmtStringStart("result: "), Ident(a), Plus, Ident(b), FmtStringEnd("")
        let tokens: Vec<_> = ExprTokenizer::new(r#""result: {{ a + b }}""#, 0, &interner)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::FmtStringStart(s) if s == "result: "));
        assert!(matches!(types[1], Token::Ident(s) if interner.resolve(*s) == "a"));
        assert!(matches!(types[2], Token::Plus));
        assert!(matches!(types[3], Token::Ident(s) if interner.resolve(*s) == "b"));
        assert!(matches!(types[4], Token::FmtStringEnd(s) if s.is_empty()));
    }
}
