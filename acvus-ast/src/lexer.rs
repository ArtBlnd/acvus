use std::collections::VecDeque;

use acvus_utils::Interner;
use logos::Logos;

use crate::ast::IndentModifier;
use crate::error::{ParseError, ParseErrorKind};
use crate::literal::SuffixedInt;
use crate::span::Span;
use crate::token::Token;

// -- Phase 1: Template Scanner ------------------------------------------

/// A segment produced by the template scanner.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// Literal text outside `{{ }}`.
    Text { value: String, span: Span },
    /// A comment `{{-- ... --}}`.
    Comment { value: String, span: Span },
    /// A close block `{{/}}`, optionally with indent modifier `{{/+2}}` or `{{/-2}}`.
    CloseBlock {
        span: Span,
        indent: Option<IndentModifier>,
    },
    /// A catch-all `{{_}}`.
    CatchAll { span: Span },
    /// An expression tag `{{ ... }}` (content is the inner text, trimmed).
    ExprTag {
        content: String,
        span: Span,
        inner_span: Span,
    },
}

/// Scan a template source string into segments.
pub fn scan_template(source: &str) -> Result<Vec<Segment>, ParseError> {
    let mut segments = Vec::new();
    // (trim_left, trim_right) per segment; only meaningful for non-Text segments
    let mut trims: Vec<(bool, bool)> = Vec::new();
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        // Detect opening delimiter: `{-{` (trim) or `{{` (normal)
        let open = detect_open(bytes, pos);
        if let Some((trim_left, skip)) = open {
            let tag_start = pos;
            pos += skip;

            // Check for comment: `--` immediately after open delimiter
            if pos + 1 < len && bytes[pos] == b'-' && bytes[pos + 1] == b'-' {
                pos += 2; // skip `--`
                let comment_start = pos;
                loop {
                    // Try close: `--}-}` or `--}}`
                    if pos + 1 < len
                        && bytes[pos] == b'-'
                        && bytes[pos + 1] == b'-'
                        && let Some((trim_right, close_skip)) = detect_close(bytes, pos + 2)
                    {
                        let comment_end = pos;
                        pos += 2 + close_skip; // skip `--` + close delimiter
                        let value = source[comment_start..comment_end].to_string();
                        segments.push(Segment::Comment {
                            value,
                            span: Span::new(tag_start, pos),
                        });
                        trims.push((trim_left, trim_right));
                        break;
                    }
                    if pos >= len {
                        return Err(ParseError::new(
                            ParseErrorKind::UnclosedComment,
                            Span::new(tag_start, len),
                        ));
                    }
                    pos += 1;
                }
            } else {
                // Regular tag
                let inner_start = pos;
                let inner_end;
                let trim_right;

                // Scan for close delimiter, respecting string literals
                loop {
                    if pos >= len {
                        return Err(ParseError::new(
                            ParseErrorKind::UnclosedTag,
                            Span::new(tag_start, len),
                        ));
                    }
                    if bytes[pos] == b'"' {
                        pos += 1;
                        loop {
                            if pos >= len {
                                return Err(ParseError::new(
                                    ParseErrorKind::UnclosedString,
                                    Span::new(tag_start, len),
                                ));
                            }
                            if bytes[pos] == b'\\' {
                                pos += 2;
                                continue;
                            }
                            if bytes[pos] == b'"' {
                                pos += 1;
                                break;
                            }
                            pos += 1;
                        }
                    } else if let Some(after) = char_literal_end(bytes, pos) {
                        pos = after;
                    } else if let Some((tr, close_skip)) = detect_close(bytes, pos) {
                        inner_end = pos;
                        trim_right = tr;
                        pos += close_skip;
                        break;
                    } else {
                        pos += 1;
                    }
                }

                let inner = &source[inner_start..inner_end];
                let trimmed = inner.trim();
                let tag_span = Span::new(tag_start, pos);

                if let Some(indent) = parse_close_block(trimmed) {
                    segments.push(Segment::CloseBlock {
                        span: tag_span,
                        indent: indent.1,
                    });
                } else if trimmed == "_" {
                    segments.push(Segment::CatchAll { span: tag_span });
                } else {
                    let leading_ws = inner.len() - inner.trim_start().len();
                    let trailing_ws = inner.len() - inner.trim_end().len();
                    let trim_start = inner_start + leading_ws;
                    let trim_end = inner_end - trailing_ws;
                    segments.push(Segment::ExprTag {
                        content: trimmed.to_string(),
                        span: tag_span,
                        inner_span: Span::new(trim_start, trim_end),
                    });
                }
                trims.push((trim_left, trim_right));
            }
        } else {
            // Literal text
            let start = pos;
            while pos < len {
                if detect_open(bytes, pos).is_some() {
                    break;
                }
                pos += 1;
            }
            segments.push(Segment::Text {
                value: source[start..pos].to_string(),
                span: Span::new(start, pos),
            });
            trims.push((false, false));
        }
    }

    apply_whitespace_trimming(&mut segments, &trims);
    Ok(segments)
}

/// Detect an opening delimiter at `pos`.
/// Returns `Some((trim_left, bytes_to_skip))` or `None`.
fn detect_open(bytes: &[u8], pos: usize) -> Option<(bool, usize)> {
    let len = bytes.len();
    if pos + 2 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'-' && bytes[pos + 2] == b'{' {
        Some((true, 3))
    } else if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
        Some((false, 2))
    } else {
        None
    }
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

/// Detect a closing delimiter at `pos`.
/// Returns `Some((trim_right, bytes_to_skip))` or `None`.
fn detect_close(bytes: &[u8], pos: usize) -> Option<(bool, usize)> {
    let len = bytes.len();
    if pos + 2 < len && bytes[pos] == b'}' && bytes[pos + 1] == b'-' && bytes[pos + 2] == b'}' {
        Some((true, 3))
    } else if pos + 1 < len && bytes[pos] == b'}' && bytes[pos + 1] == b'}' {
        Some((false, 2))
    } else {
        None
    }
}

/// Trim trailing whitespace from `s` up to and including the first `\n` encountered
/// (scanning backwards). Stops at non-whitespace or after consuming `\n`.
fn trim_trailing(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        match bytes[i - 1] {
            b' ' | b'\t' => i -= 1,
            b'\n' => {
                i -= 1;
                break;
            }
            _ => break,
        }
    }
    &s[..i]
}

/// Trim leading whitespace from `s` up to and including the first `\n` encountered
/// (scanning forwards). Stops at non-whitespace or after consuming `\n`.
fn trim_leading(s: &str) -> &str {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        match bytes[i] {
            b' ' | b'\t' => i += 1,
            b'\n' => {
                i += 1;
                break;
            }
            _ => break,
        }
    }
    &s[i..]
}

/// Apply whitespace trimming to segments based on trim flags.
fn apply_whitespace_trimming(segments: &mut Vec<Segment>, trims: &[(bool, bool)]) {
    let len = segments.len();
    for i in 0..len {
        let (trim_left, trim_right) = trims[i];

        if trim_left
            && i > 0
            && let Segment::Text { value, .. } = &mut segments[i - 1]
        {
            *value = trim_trailing(value).to_string();
        }
        if trim_right
            && i + 1 < len
            && let Segment::Text { value, .. } = &mut segments[i + 1]
        {
            *value = trim_leading(value).to_string();
        }
    }

    // Remove empty Text segments
    segments.retain(|seg| !matches!(seg, Segment::Text { value, .. } if value.is_empty()));
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

/// Expand a format string (e.g. `hello {{ name }}!`) into LALRPOP tokens.
///
/// The input `content` is the undecoded string body (from `StringLit`), so a
/// segment's offsets are the source's and the grammar decodes each segment
/// against the one escape table.
/// `base_start`/`base_end` are absolute offsets of the original `StringLit` token.
///
/// ## Algorithm
///
/// 1. Split `content` by `{{ }}` pairs into alternating text/expr segments.
///    Result is always `[text, expr, text, expr, ..., text]` (starts and ends with text).
///
/// 2. Emit tokens:
///    `FmtStringStart(text0)  <expr0 tokens>  FmtStringMid(text1)  <expr1 tokens>  ...  FmtStringEnd(textn)`
///
/// ## `}}` matching
///
/// - Brace depth tracked: `{` increments, `}` at depth>0 decrements.
/// - `}}` at depth==0 closes the interpolation.
/// - Quoted strings (`"..."`) inside expressions are skipped (with `\"` escape handling).
fn expand_format_string(
    content: &str,
    base_start: usize,
    base_end: usize,
    interner: &Interner,
) -> VecDeque<Result<(usize, Token, usize), ParseError>> {
    let err_span = Span::new(base_start, base_end);

    // -- Phase 1: split into [text, expr, text, expr, ..., text] --

    let mut texts: Vec<String> = Vec::new();
    let mut exprs: Vec<String> = Vec::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut text_start = 0;

    while pos < len {
        if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            texts.push(content[text_start..pos].to_string());

            let expr_start = pos + 2;
            let mut scan = expr_start;
            let mut depth = 0u32;

            loop {
                if scan >= len {
                    return VecDeque::from([Err(ParseError::new(
                        ParseErrorKind::UnclosedTag,
                        err_span,
                    ))]);
                }
                if let Some(after) = char_literal_end(bytes, scan) {
                    scan = after;
                    continue;
                }
                match bytes[scan] {
                    b'"' => {
                        scan += 1;
                        while scan < len {
                            if bytes[scan] == b'\\' {
                                scan += 2;
                                continue;
                            }
                            if bytes[scan] == b'"' {
                                scan += 1;
                                break;
                            }
                            scan += 1;
                        }
                    }
                    b'{' => {
                        depth += 1;
                        scan += 1;
                    }
                    b'}' if depth > 0 => {
                        depth -= 1;
                        scan += 1;
                    }
                    b'}' if scan + 1 < len && bytes[scan + 1] == b'}' => {
                        exprs.push(content[expr_start..scan].to_string());
                        pos = scan + 2;
                        text_start = pos;
                        break;
                    }
                    _ => scan += 1,
                }
            }
        } else {
            pos += 1;
        }
    }

    texts.push(content[text_start..].to_string());
    // Invariant: texts.len() == exprs.len() + 1

    // -- Phase 2: emit tokens --
    // Pattern: Start(text0) <expr0> Mid(text1) <expr1> ... End(textn)

    let mut out = VecDeque::new();
    let last_text_idx = texts.len() - 1;

    // Compute byte offsets for each text/expr segment within the source.
    // base_start points to the opening `"` of the string literal.
    // Content positions: `"` (1 byte) + content bytes.
    // Text/expr boundaries were tracked in phase 1 via `pos`.
    //
    // Rebuild positions: walk the content structure to assign correct offsets.
    // The content layout is: text0 {{ expr0 }} text1 {{ expr1 }} ... textn
    let quote_offset = 1; // opening `"`
    let mut cursor = base_start + quote_offset;

    for (i, text) in texts.into_iter().enumerate() {
        let text_len = text.len();
        let tok_start = cursor;
        let tok_end = cursor + text_len;
        cursor = tok_end;

        let tok = match i {
            0 => Token::FmtStringStart(text),
            n if n == last_text_idx => Token::FmtStringEnd(text),
            _ => Token::FmtStringMid(text),
        };
        out.push_back(Ok((tok_start, tok, tok_end)));

        // After each text except the last, emit the corresponding expression tokens
        if let Some(expr_str) = exprs.get(i) {
            cursor += 2; // skip `{{`
            out.extend(Signed::new(expr_str, cursor, interner));
            cursor += expr_str.len() + 2; // expr content + `}}`
        }
    }

    out
}

/// Try to parse a trimmed tag content as a close block.
/// Returns `Some(((), indent))` if it matches `/` optionally followed by `+N` or `-N`.
/// Returns `None` if it's not a close block pattern.
fn parse_close_block(trimmed: &str) -> Option<((), Option<IndentModifier>)> {
    if !trimmed.starts_with('/') {
        return None;
    }
    let rest = trimmed[1..].trim();
    if rest.is_empty() {
        return Some(((), None));
    }
    let (sign, digits) = if let Some(d) = rest.strip_prefix('+') {
        ('+', d.trim())
    } else if let Some(d) = rest.strip_prefix('-') {
        ('-', d.trim())
    } else {
        return None;
    };
    if digits.is_empty() {
        return None;
    }
    let n: u32 = digits.parse().ok()?;
    let modifier = match sign {
        '+' => IndentModifier::Increase(n),
        '-' => IndentModifier::Decrease(n),
        _ => unreachable!(),
    };
    Some(((), Some(modifier)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::IntWidth;

    // -- Scanner Tests --

    #[test]
    fn scan_literal_text() {
        let segs = scan_template("hello world").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello world"));
    }

    #[test]
    fn scan_inline_expr() {
        let segs = scan_template("{{ name }}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::ExprTag { content, .. } if content == "name"));
    }

    #[test]
    fn scan_comment() {
        let segs = scan_template("{{-- a comment --}}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::Comment { value, .. } if value == " a comment "));
    }

    #[test]
    fn scan_close_block() {
        let segs = scan_template("{{/}}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::CloseBlock { indent: None, .. }));
    }

    #[test]
    fn scan_close_block_indent_increase() {
        let segs = scan_template("{{/+2}}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(
            &segs[0],
            Segment::CloseBlock {
                indent: Some(IndentModifier::Increase(2)),
                ..
            }
        ));
    }

    #[test]
    fn scan_close_block_indent_decrease() {
        let segs = scan_template("{{/-3}}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(
            &segs[0],
            Segment::CloseBlock {
                indent: Some(IndentModifier::Decrease(3)),
                ..
            }
        ));
    }

    #[test]
    fn scan_close_block_indent_with_spaces() {
        let segs = scan_template("{{ / + 2 }}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(
            &segs[0],
            Segment::CloseBlock {
                indent: Some(IndentModifier::Increase(2)),
                ..
            }
        ));
    }

    #[test]
    fn scan_catch_all() {
        let segs = scan_template("{{_}}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::CatchAll { .. }));
    }

    #[test]
    fn scan_catch_all_with_spaces() {
        let segs = scan_template("{{ _ }}").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::CatchAll { .. }));
    }

    #[test]
    fn scan_mixed() {
        let segs = scan_template("hello {{ name }} world").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello "));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "name"));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == " world"));
    }

    #[test]
    fn scan_string_with_braces() {
        let segs = scan_template(r#"{{ "a}}b" }}"#).unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::ExprTag { content, .. } if content == r#""a}}b""#));
    }

    #[test]
    fn scan_unclosed_tag() {
        let result = scan_template("{{ hello");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            ParseErrorKind::UnclosedTag
        ));
    }

    #[test]
    fn scan_unclosed_comment() {
        let result = scan_template("{{-- hello");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            ParseErrorKind::UnclosedComment
        ));
    }

    // -- Trim Helper Tests --

    #[test]
    fn trim_trailing_newline() {
        assert_eq!(trim_trailing("hello  \n  "), "hello  ");
    }

    #[test]
    fn trim_trailing_no_newline() {
        assert_eq!(trim_trailing("hello  "), "hello");
    }

    #[test]
    fn trim_trailing_only_spaces() {
        assert_eq!(trim_trailing("   "), "");
    }

    #[test]
    fn trim_trailing_ends_with_text() {
        assert_eq!(trim_trailing("hello"), "hello");
    }

    #[test]
    fn trim_trailing_tabs() {
        assert_eq!(trim_trailing("hello\n\t\t"), "hello");
    }

    #[test]
    fn trim_leading_newline() {
        assert_eq!(trim_leading("  \n  world"), "  world");
    }

    #[test]
    fn trim_leading_no_newline() {
        assert_eq!(trim_leading("  world"), "world");
    }

    #[test]
    fn trim_leading_only_spaces() {
        assert_eq!(trim_leading("   "), "");
    }

    #[test]
    fn trim_leading_starts_with_text() {
        assert_eq!(trim_leading("world"), "world");
    }

    #[test]
    fn trim_leading_tabs() {
        assert_eq!(trim_leading("\t\t\nhello"), "hello");
    }

    // -- Whitespace Trimming Scanner Tests --

    #[test]
    fn scan_trim_left() {
        let segs = scan_template("hello  \n  {-{ x }}").unwrap();
        assert_eq!(segs.len(), 2);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello  "));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "x"));
    }

    #[test]
    fn scan_trim_right() {
        let segs = scan_template("{{ x }-}  \n  world").unwrap();
        assert_eq!(segs.len(), 2);
        assert!(matches!(&segs[0], Segment::ExprTag { content, .. } if content == "x"));
        assert!(matches!(&segs[1], Segment::Text { value, .. } if value == "  world"));
    }

    #[test]
    fn scan_trim_both() {
        let segs = scan_template("hello  \n  {-{ x }-}  \n  world").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello  "));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "x"));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == "  world"));
    }

    #[test]
    fn scan_trim_removes_empty_text() {
        // "\n{-{ x }-}\n" - newlines are consumed, leaving empty texts that get removed
        let segs = scan_template("\n{-{ x }-}\n").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::ExprTag { content, .. } if content == "x"));
    }

    #[test]
    fn scan_trim_partial() {
        // "  \n  " trims to "  " (only up to \n)
        let segs = scan_template("  \n  {-{ x }-}  \n  ").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "  "));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "x"));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == "  "));
    }

    #[test]
    fn scan_trim_comment() {
        let segs = scan_template("hello  \n  {-{-- comment --}-}  \n  world").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello  "));
        assert!(matches!(&segs[1], Segment::Comment { .. }));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == "  world"));
    }

    #[test]
    fn scan_trim_close_block() {
        let segs = scan_template("\n{-{/}-}\n").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::CloseBlock { indent: None, .. }));
    }

    #[test]
    fn scan_trim_catch_all() {
        let segs = scan_template("\n{-{ _ }-}\n").unwrap();
        assert_eq!(segs.len(), 1);
        assert!(matches!(&segs[0], Segment::CatchAll { .. }));
    }

    #[test]
    fn scan_no_trim_unchanged() {
        // Normal delimiters should not trim
        let segs = scan_template("hello  \n  {{ x }}  \n  world").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello  \n  "));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "x"));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == "  \n  world"));
    }

    #[test]
    fn scan_trim_no_whitespace_to_trim() {
        let segs = scan_template("hello{-{ x }-}world").unwrap();
        assert_eq!(segs.len(), 3);
        assert!(matches!(&segs[0], Segment::Text { value, .. } if value == "hello"));
        assert!(matches!(&segs[1], Segment::ExprTag { content, .. } if content == "x"));
        assert!(matches!(&segs[2], Segment::Text { value, .. } if value == "world"));
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
