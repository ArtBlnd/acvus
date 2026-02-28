use logos::Logos;

use crate::error::{ParseError, ParseErrorKind};
use crate::span::Span;
use crate::token::Token;

// ── Phase 1: Template Scanner ──────────────────────────────────────────

/// A segment produced by the template scanner.
#[derive(Debug, Clone, PartialEq)]
pub enum Segment {
    /// Literal text outside `{{ }}`.
    Text { value: String, span: Span },
    /// A comment `{{-- ... --}}`.
    Comment { value: String, span: Span },
    /// A close block `{{/}}`.
    CloseBlock { span: Span },
    /// A catch-all `{{_}}`.
    CatchAll { span: Span },
    /// An expression tag `{{ ... }}` (content is the inner text, trimmed).
    ExprTag { content: String, span: Span, inner_span: Span },
}

/// Scan a template source string into segments.
pub fn scan_template(source: &str) -> Result<Vec<Segment>, ParseError> {
    let mut segments = Vec::new();
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            // Check for comment `{{--`
            if pos + 3 < len && bytes[pos + 2] == b'-' && bytes[pos + 3] == b'-' {
                let start = pos;
                pos += 4; // skip `{{--`
                let comment_start = pos;
                loop {
                    if pos + 3 < len
                        && bytes[pos] == b'-'
                        && bytes[pos + 1] == b'-'
                        && bytes[pos + 2] == b'}'
                        && bytes[pos + 3] == b'}'
                    {
                        let comment_end = pos;
                        pos += 4; // skip `--}}`
                        let value = source[comment_start..comment_end].to_string();
                        segments.push(Segment::Comment {
                            value,
                            span: Span::new(start, pos),
                        });
                        break;
                    }
                    if pos >= len {
                        return Err(ParseError::new(
                            ParseErrorKind::UnclosedComment,
                            Span::new(start, len),
                        ));
                    }
                    pos += 1;
                }
            } else {
                // Regular tag `{{ ... }}`
                let tag_start = pos;
                pos += 2; // skip `{{`
                let inner_start = pos;
                let inner_end;

                // Scan for `}}`, respecting string literals
                loop {
                    if pos >= len {
                        return Err(ParseError::new(
                            ParseErrorKind::UnclosedTag,
                            Span::new(tag_start, len),
                        ));
                    }
                    if bytes[pos] == b'"' {
                        // Skip string literal
                        pos += 1;
                        loop {
                            if pos >= len {
                                return Err(ParseError::new(
                                    ParseErrorKind::UnclosedString,
                                    Span::new(tag_start, len),
                                ));
                            }
                            if bytes[pos] == b'\\' {
                                pos += 2; // skip escape
                                continue;
                            }
                            if bytes[pos] == b'"' {
                                pos += 1;
                                break;
                            }
                            pos += 1;
                        }
                    } else if pos + 1 < len && bytes[pos] == b'}' && bytes[pos + 1] == b'}' {
                        inner_end = pos;
                        pos += 2; // skip `}}`
                        break;
                    } else {
                        pos += 1;
                    }
                }

                let inner = &source[inner_start..inner_end];
                let trimmed = inner.trim();
                let tag_span = Span::new(tag_start, pos);

                if trimmed == "/" {
                    segments.push(Segment::CloseBlock { span: tag_span });
                } else if trimmed == "_" {
                    segments.push(Segment::CatchAll { span: tag_span });
                } else {
                    // Compute inner_span as the span of the trimmed content
                    // relative to the source
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
            }
        } else {
            // Literal text
            let start = pos;
            while pos < len {
                if pos + 1 < len && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
                    break;
                }
                pos += 1;
            }
            segments.push(Segment::Text {
                value: source[start..pos].to_string(),
                span: Span::new(start, pos),
            });
        }
    }

    Ok(segments)
}

// ── Phase 2: Expression Tokenizer (logos-backed) ───────────────────────

/// Tokenizer for expression content within `{{ }}` tags.
/// Wraps a logos lexer and produces `(start, Token, end)` triples for LALRPOP.
pub struct ExprTokenizer<'input> {
    lexer: logos::Lexer<'input, Token>,
    base_offset: usize,
}

impl<'input> ExprTokenizer<'input> {
    pub fn new(input: &'input str, base_offset: usize) -> Self {
        Self {
            lexer: Token::lexer(input),
            base_offset,
        }
    }
}

impl<'input> Iterator for ExprTokenizer<'input> {
    type Item = Result<(usize, Token, usize), ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.lexer.next()?;
        let span = self.lexer.span();
        let start = self.base_offset + span.start;
        let end = self.base_offset + span.end;
        match result {
            Ok(token) => Some(Ok((start, token, end))),
            Err(()) => {
                let c = self.lexer.slice().chars().next().unwrap_or('?');
                Some(Err(ParseError::new(
                    ParseErrorKind::UnexpectedCharacter(c),
                    Span::new(start, end),
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scanner Tests ──

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
        assert!(matches!(&segs[0], Segment::CloseBlock { .. }));
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
        assert!(
            matches!(&segs[0], Segment::ExprTag { content, .. } if content == r#""a}}b""#)
        );
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

    // ── Tokenizer Tests ──

    #[test]
    fn tokenize_ident() {
        let tokens: Vec<_> = ExprTokenizer::new("name", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::Ident(s) if s == "name"));
    }

    #[test]
    fn tokenize_storage_ref() {
        let tokens: Vec<_> = ExprTokenizer::new("$global", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::StorageRef(s) if s == "global"));
    }

    #[test]
    fn tokenize_underscore() {
        let tokens: Vec<_> = ExprTokenizer::new("_", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::Underscore));
    }

    #[test]
    fn tokenize_keywords() {
        let tokens: Vec<_> = ExprTokenizer::new("true false", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(&tokens[0].1, Token::True));
        assert!(matches!(&tokens[1].1, Token::False));
    }

    #[test]
    fn tokenize_numbers() {
        let tokens: Vec<_> = ExprTokenizer::new("42 3.14", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(&tokens[0].1, Token::IntLit(42)));
        assert!(matches!(&tokens[1].1, Token::FloatLit(f) if (*f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn tokenize_string() {
        let tokens: Vec<_> = ExprTokenizer::new(r#""hello \"world\"""#, 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 1);
        assert!(matches!(&tokens[0].1, Token::StringLit(s) if s == r#"hello "world""#));
    }

    #[test]
    fn tokenize_two_char_operators() {
        let tokens: Vec<_> = ExprTokenizer::new("== != <= >= -> ..", 0)
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
    fn tokenize_range_operators() {
        let tokens: Vec<_> = ExprTokenizer::new("..= =..", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(matches!(&tokens[0].1, Token::DotDotEq));
        assert!(matches!(&tokens[1].1, Token::EqDotDot));
    }

    #[test]
    fn tokenize_complex_expr() {
        let tokens: Vec<_> = ExprTokenizer::new("list | filter(x -> x != 0)", 0)
            .collect::<Result<_, _>>()
            .unwrap();
        let types: Vec<_> = tokens.iter().map(|t| &t.1).collect();
        assert!(matches!(types[0], Token::Ident(s) if s == "list"));
        assert!(matches!(types[1], Token::Pipe));
        assert!(matches!(types[2], Token::Ident(s) if s == "filter"));
        assert!(matches!(types[3], Token::LParen));
        assert!(matches!(types[4], Token::Ident(s) if s == "x"));
        assert!(matches!(types[5], Token::Arrow));
        assert!(matches!(types[6], Token::Ident(s) if s == "x"));
        assert!(matches!(types[7], Token::Neq));
        assert!(matches!(types[8], Token::IntLit(0)));
        assert!(matches!(types[9], Token::RParen));
    }

    #[test]
    fn tokenize_absolute_offsets() {
        let tokens: Vec<_> = ExprTokenizer::new("ab", 10)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tokens[0].0, 10); // start
        assert_eq!(tokens[0].2, 12); // end
    }
}
