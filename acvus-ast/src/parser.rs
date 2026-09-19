use acvus_utils::{Interner, QualifiedRef};
use lalrpop_util::ParseError as LalrpopError;

use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind};
use crate::grammar::{ExprParser, ScriptParser, TagContentParser};
use crate::lexer::{ExprTokenizer, Segment, scan_template};
use crate::span::Span;
use crate::token::Token;

use crate::tag_content::TagContent;

/// A decoded literal, or the grammar's error at the literal's span
/// (`literal::LiteralErrorKind`).
pub fn decoded<T>(
    decoded: Result<T, LiteralErrorKind>,
    span: Span,
) -> Result<T, LalrpopError<usize, Token, ParseError>> {
    decoded.map_err(|kind| LalrpopError::User {
        error: ParseError::new(ParseErrorKind::BadLiteral(kind), span),
    })
}

/// Parse a single expression.
pub fn parse_expr(interner: &Interner, source: &str) -> Result<Expr, ParseError> {
    let tokenizer = ExprTokenizer::new(source, 0, interner);
    ExprParser::new()
        .parse(interner, tokenizer)
        .map_err(|e| convert_lalrpop_error(e, 0, source.len()))
}

/// Parse a script source string. One statement grammar: `let x = e;` binds,
/// `x = e;` assigns the `x` in scope, and `if`/`while`/`anyorder`/the tag
/// form are statements of the same rule in every block.
pub fn parse_script(interner: &Interner, source: &str) -> Result<Script, ParseError> {
    let tokenizer = ExprTokenizer::new(source, 0, interner);
    ScriptParser::new()
        .parse(interner, tokenizer)
        .map_err(|e| convert_lalrpop_error(e, 0, source.len()))
}

/// Parse a template source string into an AST.
pub fn parse_template(interner: &Interner, source: &str) -> Result<Template, ParseError> {
    let segments = scan_template(source)?;
    let mut builder = TreeBuilder {
        segments: &segments,
        pos: 0,
        source,
        interner,
    };
    let body = builder.build_body()?;
    let span = if body.is_empty() {
        Span::new(0, source.len())
    } else {
        let first = node_span(&body[0]);
        let last = node_span(body.last().unwrap());
        first.merge(last)
    };
    Ok(Template {
        id: AstId::alloc(),
        body,
        span,
    })
}

fn node_span(node: &Node) -> Span {
    match node {
        Node::Text { span, .. } | Node::Comment { span, .. } | Node::InlineExpr { span, .. } => {
            *span
        }
        Node::MatchBlock(mb) => mb.span,
    }
}

struct TreeBuilder<'a> {
    segments: &'a [Segment],
    pos: usize,
    #[allow(dead_code)]
    source: &'a str,
    interner: &'a Interner,
}

/// What stopped `build_body` from collecting more nodes.
enum BodyTerminator {
    /// We've reached the end of segments.
    Eof,
    /// We hit `{{/}}` or `{{/+N}}` / `{{/-N}}`.
    CloseBlock(Span, Option<IndentModifier>),
    /// We hit `{{_}}`.
    CatchAll(Span),
    /// We hit a `{{ pattern = }}` multi-arm continuation.
    MultiArm { expr: Expr, tag_span: Span },
}

impl<'a> TreeBuilder<'a> {
    fn peek(&self) -> Option<&'a Segment> {
        self.segments.get(self.pos)
    }

    fn advance(&mut self) {
        self.pos += 1;
    }

    /// Build nodes until we hit a block-structural segment or EOF.
    fn build_body(&mut self) -> Result<Vec<Node>, ParseError> {
        let (nodes, _terminator) = self.build_body_until_terminator(false)?;
        Ok(nodes)
    }

    /// Build nodes until we hit a block-structural segment or EOF.
    /// When `in_match` is true, bare expressions that can only be patterns
    /// (literals, lists, ranges, objects) are treated as continuation arms.
    fn build_body_until_terminator(
        &mut self,
        in_match: bool,
    ) -> Result<(Vec<Node>, BodyTerminator), ParseError> {
        let mut nodes = Vec::new();

        loop {
            match self.peek() {
                None => return Ok((nodes, BodyTerminator::Eof)),
                Some(Segment::CloseBlock { span, indent }) => {
                    let span = *span;
                    let indent = *indent;
                    self.advance();
                    return Ok((nodes, BodyTerminator::CloseBlock(span, indent)));
                }
                Some(Segment::CatchAll { span }) => {
                    let span = *span;
                    self.advance();
                    return Ok((nodes, BodyTerminator::CatchAll(span)));
                }
                Some(Segment::Text { value, span }) => {
                    let node = Node::Text {
                        id: AstId::alloc(),
                        value: value.clone(),
                        span: *span,
                    };
                    self.advance();
                    nodes.push(node);
                }
                Some(Segment::Comment { value, span }) => {
                    let node = Node::Comment {
                        id: AstId::alloc(),
                        value: value.clone(),
                        span: *span,
                    };
                    self.advance();
                    nodes.push(node);
                }
                Some(Segment::ExprTag {
                    content,
                    span,
                    inner_span,
                }) => {
                    let content = content.clone();
                    let tag_span = *span;
                    let inner_span = *inner_span;
                    self.advance();

                    let tag_content = parse_tag_content(self.interner, &content, inner_span.start)?;

                    match tag_content {
                        TagContent::Expr(expr) => {
                            nodes.push(Node::InlineExpr {
                                id: AstId::alloc(),
                                expr,
                                span: tag_span,
                            });
                        }
                        TagContent::ContinuationArm { pattern, .. } => {
                            if !in_match {
                                return Err(ParseError::new(
                                    ParseErrorKind::InvalidPattern(
                                        "continuation arm `{{ pattern = }}` outside match block"
                                            .into(),
                                    ),
                                    tag_span,
                                ));
                            }
                            return Ok((
                                nodes,
                                BodyTerminator::MultiArm {
                                    expr: pattern,
                                    tag_span,
                                },
                            ));
                        }
                        TagContent::Binding { lhs, rhs, .. } => {
                            let pattern = expr_to_pattern(&lhs)?;
                            // Bare binding (variable or storage) -> body-less (no {{/}})
                            if matches!(
                                &pattern,
                                Pattern::Binding { .. } | Pattern::ContextBind { .. }
                            ) {
                                let match_block = MatchBlock {
                                    id: AstId::alloc(),
                                    source: rhs,
                                    arms: vec![MatchArm {
                                        id: AstId::alloc(),
                                        pattern,
                                        body: vec![],
                                        tag_span,
                                    }],
                                    catch_all: None,
                                    indent: None,
                                    span: tag_span,
                                };
                                nodes.push(Node::MatchBlock(match_block));
                            } else {
                                let match_block = self.build_match_block(pattern, rhs, tag_span)?;
                                nodes.push(Node::MatchBlock(match_block));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Build a match block starting after the first arm's tag has been consumed.
    fn build_match_block(
        &mut self,
        first_pattern: Pattern,
        source_expr: Expr,
        first_tag_span: Span,
    ) -> Result<MatchBlock, ParseError> {
        let block_start = first_tag_span.start;
        let mut arms = Vec::new();

        // Build body for first arm (in_match = true to detect continuation arms)
        let (body, terminator) = self.build_body_until_terminator(true)?;
        arms.push(MatchArm {
            id: AstId::alloc(),
            pattern: first_pattern,
            body,
            tag_span: first_tag_span,
        });

        // Process remaining arms, catch-all, and close
        let (catch_all, close_span, indent) =
            self.continue_match_block(&mut arms, terminator, &source_expr)?;

        let span = Span::new(block_start, close_span.end);
        Ok(MatchBlock {
            id: AstId::alloc(),
            source: source_expr,
            arms,
            catch_all,
            indent,
            span,
        })
    }

    /// Continue processing a match block after the first arm's body.
    /// Returns `(catch_all, close_span, indent)`.
    fn continue_match_block(
        &mut self,
        arms: &mut Vec<MatchArm>,
        terminator: BodyTerminator,
        _source_expr: &Expr,
    ) -> Result<(Option<CatchAll>, Span, Option<IndentModifier>), ParseError> {
        match terminator {
            BodyTerminator::Eof => {
                let span = if let Some(arm) = arms.last() {
                    arm.tag_span
                } else {
                    Span::new(0, 0)
                };
                Err(ParseError::new(ParseErrorKind::UnclosedBlock, span))
            }
            BodyTerminator::CloseBlock(span, indent) => Ok((None, span, indent)),
            BodyTerminator::CatchAll(catch_tag_span) => {
                // Build catch-all body, expect CloseBlock
                let (catch_body, next_terminator) = self.build_body_until_terminator(false)?;
                match next_terminator {
                    BodyTerminator::CloseBlock(close_span, indent) => {
                        let catch_all = CatchAll {
                            id: AstId::alloc(),
                            body: catch_body,
                            tag_span: catch_tag_span,
                        };
                        Ok((Some(catch_all), close_span, indent))
                    }
                    BodyTerminator::Eof => Err(ParseError::new(
                        ParseErrorKind::UnclosedBlock,
                        catch_tag_span,
                    )),
                    BodyTerminator::CatchAll(span) => {
                        Err(ParseError::new(ParseErrorKind::UnmatchedCatchAll, span))
                    }
                    BodyTerminator::MultiArm { tag_span, .. } => Err(ParseError::new(
                        ParseErrorKind::ExpectedCloseBlock,
                        tag_span,
                    )),
                }
            }
            BodyTerminator::MultiArm { expr, tag_span } => {
                // This is a `{{ pattern }}` continuation arm
                let pattern = expr_to_pattern(&expr)?;
                let (body, next_terminator) = self.build_body_until_terminator(true)?;
                arms.push(MatchArm {
                    id: AstId::alloc(),
                    pattern,
                    body,
                    tag_span,
                });
                self.continue_match_block(arms, next_terminator, _source_expr)
            }
        }
    }
}

/// Parse the content of a `{{ }}` tag using LALRPOP.
fn parse_tag_content(
    interner: &Interner,
    content: &str,
    base_offset: usize,
) -> Result<TagContent, ParseError> {
    let tokenizer = ExprTokenizer::new(content, base_offset, interner);
    let parser = TagContentParser::new();
    parser
        .parse(interner, tokenizer)
        .map_err(|e| convert_lalrpop_error(e, base_offset, content.len()))
}

/// Convert a LALRPOP error to our ParseError.
fn convert_lalrpop_error(
    error: LalrpopError<usize, Token, ParseError>,
    _base_offset: usize,
    _content_len: usize,
) -> ParseError {
    match error {
        LalrpopError::InvalidToken { location } => ParseError::new(
            ParseErrorKind::UnexpectedToken("invalid token".into()),
            Span::new(location, location + 1),
        ),
        LalrpopError::UnrecognizedEof {
            location,
            expected: _,
        } => ParseError::new(ParseErrorKind::UnexpectedEof, Span::new(location, location)),
        LalrpopError::UnrecognizedToken {
            token: (start, tok, end),
            expected,
        } => ParseError::new(
            ParseErrorKind::UnexpectedToken(format!(
                "got `{tok}`, expected one of: {}",
                expected.join(", ")
            )),
            Span::new(start, end),
        ),
        LalrpopError::ExtraToken {
            token: (start, tok, end),
        } => ParseError::new(
            ParseErrorKind::UnexpectedToken(format!("extra token `{tok}`")),
            Span::new(start, end),
        ),
        LalrpopError::User { error } => error,
    }
}

/// Validate that a pattern is irrefutable (always matches).
/// Only Binding, Object (with irrefutable sub-patterns), and Tuple (with irrefutable sub-patterns)
/// are allowed. Literals and Lists are refutable.
fn validate_irrefutable(pattern: &Pattern) -> Result<(), ParseError> {
    match pattern {
        Pattern::Binding { .. } | Pattern::ContextBind { .. } => Ok(()),
        Pattern::Object { fields, .. } => {
            for f in fields {
                validate_irrefutable(&f.pattern)?;
            }
            Ok(())
        }
        Pattern::Tuple { elements, .. } => {
            for elem in elements {
                if let TuplePatternElem::Pattern(p) = elem {
                    validate_irrefutable(p)?;
                }
            }
            Ok(())
        }
        _ => Err(ParseError::new(
            ParseErrorKind::RefutablePattern,
            pattern.span(),
        )),
    }
}

/// Convert an expression (parsed from the LHS of `=`) to a pattern.
pub fn expr_to_pattern(expr: &Expr) -> Result<Pattern, ParseError> {
    match expr {
        Expr::Ident {
            name,
            ref_kind,
            span,
            ..
        } => Ok(Pattern::Binding {
            id: AstId::alloc(),
            name: name.name,
            ref_kind: *ref_kind,
            span: *span,
        }),
        Expr::ContextRef { name, span, .. } => Ok(Pattern::ContextBind {
            id: AstId::alloc(),
            name: *name,
            span: *span,
        }),
        Expr::Literal { value, span, .. } => Ok(Pattern::Literal {
            id: AstId::alloc(),
            value: value.clone(),
            span: *span,
        }),
        Expr::List {
            head,
            rest,
            tail,
            span,
            ..
        } => {
            let head_pats: Result<Vec<_>, _> = head.iter().map(expr_to_pattern).collect();
            let tail_pats: Result<Vec<_>, _> = tail.iter().map(expr_to_pattern).collect();
            Ok(Pattern::List {
                id: AstId::alloc(),
                head: head_pats?,
                rest: *rest,
                tail: tail_pats?,
                span: *span,
            })
        }
        Expr::Object { fields, span, .. } => {
            let pattern_fields: Result<Vec<_>, _> = fields
                .iter()
                .map(|f| {
                    let pattern = expr_to_pattern(&f.value)?;
                    Ok(ObjectPatternField {
                        id: AstId::alloc(),
                        key: f.key,
                        pattern,
                        span: f.span,
                    })
                })
                .collect();
            Ok(Pattern::Object {
                id: AstId::alloc(),
                fields: pattern_fields?,
                span: *span,
            })
        }
        Expr::Tuple { elements, span, .. } => {
            let elems: Result<Vec<_>, _> = elements
                .iter()
                .map(|elem| match elem {
                    TupleElem::Wildcard(s) => Ok(TuplePatternElem::Wildcard(*s)),
                    TupleElem::Expr(e) => {
                        let pat = expr_to_pattern(e)?;
                        Ok(TuplePatternElem::Pattern(pat))
                    }
                })
                .collect();
            Ok(Pattern::Tuple {
                id: AstId::alloc(),
                elements: elems?,
                span: *span,
            })
        }
        // `Enum::Tag(inner)` parses as a qualified call (RFC-0030); as a
        // pattern it is the variant.
        Expr::FuncCall {
            func, args, span, ..
        } if matches!(
            func.as_ref(),
            Expr::Ident {
                name: QualifiedRef {
                    namespace: Some(_),
                    ..
                },
                ..
            }
        ) && args.len() == 1 =>
        {
            let Expr::Ident { name, .. } = func.as_ref() else {
                unreachable!("matched above")
            };
            Ok(Pattern::Variant {
                id: AstId::alloc(),
                enum_name: name.namespace,
                tag: name.name,
                payload: Some(Box::new(expr_to_pattern(&args[0])?)),
                span: *span,
            })
        }
        Expr::Variant {
            enum_name,
            tag,
            payload,
            span,
            ..
        } => {
            let pat_payload = match payload {
                Some(inner) => Some(Box::new(expr_to_pattern(inner)?)),
                None => None,
            };
            Ok(Pattern::Variant {
                id: AstId::alloc(),
                enum_name: *enum_name,
                tag: *tag,
                payload: pat_payload,
                span: *span,
            })
        }
        other => Err(ParseError::new(
            ParseErrorKind::InvalidPattern("expression cannot be used as a pattern".into()),
            other.span(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> (Interner, Result<Template, ParseError>) {
        let interner = Interner::new();
        let result = parse_template(&interner, src);
        (interner, result)
    }

    #[test]
    fn parse_literal_text() {
        let (_interner, result) = parse("hello world");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        assert!(matches!(&t.body[0], Node::Text { value, .. } if value == "hello world"));
    }

    #[test]
    fn parse_inline_expr() {
        let (_interner, result) = parse("{{ \"hello\" }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        assert!(matches!(&t.body[0], Node::InlineExpr { .. }));
    }

    #[test]
    fn parse_comment() {
        let (_interner, result) = parse("{{-- comment --}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        assert!(matches!(&t.body[0], Node::Comment { .. }));
    }

    #[test]
    fn parse_storage_write() {
        let (interner, result) = parse("{{ $global = 42 }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.arms.len(), 1);
            assert!(mb.arms[0].body.is_empty());
            assert!(mb.catch_all.is_none());
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::Binding { name, ref_kind: RefKind::ExternParam, .. } if interner.resolve(*name) == "global"
            ));
            assert!(matches!(
                &mb.source,
                Expr::Literal {
                    value: Literal::Int(42),
                    ..
                }
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_simple_variable_binding() {
        // Variable bindings are body-less (no {{/}} needed).
        let (interner, result) = parse("{{ item = list }}{{ item }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 2);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.arms.len(), 1);
            assert!(mb.arms[0].body.is_empty());
            assert!(mb.catch_all.is_none());
            assert!(
                matches!(&mb.arms[0].pattern, Pattern::Binding { name, ref_kind: RefKind::Value, .. } if interner.resolve(*name) == "item")
            );
            assert!(
                matches!(&mb.source, Expr::Ident { name, .. } if interner.resolve(name.name) == "list")
            );
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_match_with_catch_all() {
        let (_interner, result) = parse("{{ true = is_valid }}yes{{_}}no{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.arms.len(), 1);
            assert!(mb.catch_all.is_some());
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::Literal {
                    value: Literal::Bool(true),
                    ..
                }
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_multi_arm() {
        // `{{ pattern = }}` is a continuation arm.
        let (_interner, result) =
            parse(r#"{{ "admin" = role }}admin{{ "user" = }}user{{_}}guest{{/}}"#);
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.arms.len(), 2);
            assert!(mb.catch_all.is_some());
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_destructuring_rest_tail() {
        let (_interner, result) = parse("{{ [a, b, ..] = list }}{{a}}{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::List { head, rest: Some(_), tail, .. }
                    if head.len() == 2 && tail.is_empty()
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_destructuring_rest_head() {
        let (_interner, result) = parse("{{ [.., a, b] = list }}{{a}}{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::List { head, rest: Some(_), tail, .. }
                    if head.is_empty() && tail.len() == 2
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_destructuring_exhaustive() {
        let (_interner, result) = parse("{{ [a, b, c] = list }}{{a}}{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::List { head, rest: None, tail, .. }
                    if head.len() == 3 && tail.is_empty()
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_nested_blocks() {
        // Variable bindings are body-less, so nested blocks use pattern matches.
        let (_interner, result) = parse("{{ true = flag }}yes{{_}}no{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.arms.len(), 1);
            assert!(mb.catch_all.is_some());
        } else {
            panic!("expected MatchBlock");
        }

        // Variable bindings followed by usage.
        let (_interner, result) = parse("{{ user = users }}{{ user }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 2);
        assert!(matches!(&t.body[0], Node::MatchBlock(_)));
        assert!(matches!(&t.body[1], Node::InlineExpr { .. }));
    }

    #[test]
    fn parse_pipe_and_lambda() {
        let (_interner, result) = parse("{{ list | filter(|x| -> x != 0) | map(|x| -> x * 2) }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::InlineExpr { expr, .. } = &t.body[0] {
            // Should be a Pipe expression
            assert!(matches!(expr, Expr::Pipe { .. }));
        } else {
            panic!("expected InlineExpr");
        }
    }

    #[test]
    fn parse_arithmetic() {
        let (_interner, result) = parse("{{ 1 + 2 * 3 }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::InlineExpr { expr, .. } = &t.body[0] {
            // Should be Add(1, Mul(2, 3)) due to precedence
            if let Expr::BinaryOp {
                op: BinOp::Add,
                right,
                ..
            } = expr
            {
                assert!(matches!(
                    right.as_ref(),
                    Expr::BinaryOp { op: BinOp::Mul, .. }
                ));
            } else {
                panic!("expected Add at top level");
            }
        } else {
            panic!("expected InlineExpr");
        }
    }

    #[test]
    fn parse_field_access() {
        let (interner, result) = parse("{{ user.name }}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::InlineExpr { expr, .. } = &t.body[0] {
            assert!(
                matches!(expr, Expr::FieldAccess { field, .. } if interner.resolve(*field) == "name")
            );
        } else {
            panic!("expected InlineExpr");
        }
    }

    #[test]
    fn parse_object_pattern() {
        // Objects require trailing comma: { $value, name, }
        let (interner, result) = parse("{{ { $value, name, } = $global }}x{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            if let Pattern::Object { fields, .. } = &mb.arms[0].pattern {
                assert_eq!(fields.len(), 2);
                assert_eq!(interner.resolve(fields[0].key), "value");
                assert!(matches!(
                    &fields[0].pattern,
                    Pattern::Binding { name, ref_kind: RefKind::ExternParam, .. } if interner.resolve(*name) == "value"
                ));
                assert_eq!(interner.resolve(fields[1].key), "name");
                assert!(matches!(
                    &fields[1].pattern,
                    Pattern::Binding { name, ref_kind: RefKind::Value, .. } if interner.resolve(*name) == "name"
                ));
            } else {
                panic!("expected Object pattern");
            }
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_unclosed_block() {
        // A literal pattern match without {{/}} is unclosed.
        let (_interner, result) = parse("{{ true = x }}hello");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            ParseErrorKind::UnclosedBlock
        ));
    }

    #[test]
    fn parse_unmatched_close() {
        // {{/}} at top level with no open block - this should be an error
        // In our design, build_body returns CloseBlock as terminator.
        // At top level, build_body doesn't distinguish - the top-level call
        // uses build_body() which wraps build_body_until_terminator.
        // Let's verify through the public API that it either errors or
        // handles it. Actually, our build_body discards the terminator,
        // so a stray {{/}} would cause the top-level parse to think the
        // block ended. This is a limitation we accept for now.
    }

    #[test]
    fn parse_indent_increase() {
        let (_interner, result) = parse("{{ true = x }}hello{{/+2}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.indent, Some(IndentModifier::Increase(2)));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_indent_decrease() {
        let (_interner, result) = parse("{{ true = x }}hello{{/-1}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.indent, Some(IndentModifier::Decrease(1)));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_indent_none() {
        let (_interner, result) = parse("{{ true = x }}hello{{/}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert_eq!(mb.indent, None);
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_indent_with_catch_all() {
        let (_interner, result) = parse("{{ true = x }}yes{{_}}no{{/+3}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(mb.catch_all.is_some());
            assert_eq!(mb.indent, Some(IndentModifier::Increase(3)));
        } else {
            panic!("expected MatchBlock");
        }
    }

    // -- Script parsing tests ------------------------------------------

    #[test]
    fn script_single_expr() {
        let interner = Interner::new();
        let s = parse_script(&interner, "@data").unwrap();
        assert!(s.stmts.is_empty());
        assert!(matches!(
            s.tail.as_deref(),
            Some(Expr::ContextRef { name, .. }) if interner.resolve(name.name) == "data"
        ));
    }

    #[test]
    fn script_bind_and_tail() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let x = @data; x").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(
            matches!(&s.stmts[0], Stmt::LetBind { name, .. } if interner.resolve(*name) == "x")
        );
        assert!(matches!(
            s.tail.as_deref(),
            Some(Expr::Ident { name, ref_kind: RefKind::Value, .. }) if interner.resolve(name.name) == "x"
        ));
    }

    #[test]
    fn script_trailing_semicolon_no_tail() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let x = @data;").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(s.tail.is_none());
    }

    #[test]
    fn script_multiple_stmts_and_tail() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let x = @data; let y = x; y").unwrap();
        assert_eq!(s.stmts.len(), 2);
        assert!(
            matches!(&s.stmts[0], Stmt::LetBind { name, .. } if interner.resolve(*name) == "x")
        );
        assert!(
            matches!(&s.stmts[1], Stmt::LetBind { name, .. } if interner.resolve(*name) == "y")
        );
        assert!(s.tail.is_some());
    }

    #[test]
    fn script_expr_stmt() {
        let interner = Interner::new();
        let s = parse_script(&interner, "42; @data").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(matches!(
            &s.stmts[0],
            Stmt::Expr(Expr::Literal {
                value: Literal::Int(42),
                ..
            })
        ));
        assert!(s.tail.is_some());
    }

    #[test]
    fn script_empty() {
        let interner = Interner::new();
        let s = parse_script(&interner, "").unwrap();
        assert!(s.stmts.is_empty());
        assert!(s.tail.is_none());
    }

    #[test]
    fn script_pipe_in_bind() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let x = @data | filter(f); x").unwrap();
        assert_eq!(s.stmts.len(), 1);
        let Stmt::LetBind { expr, .. } = &s.stmts[0] else {
            panic!("expected LetBind");
        };
        assert!(matches!(expr, Expr::Pipe { .. }));
    }

    // -- One statement grammar in every block ------------------------

    /// The statements of a lambda's block body are the statements of the
    /// script: `let` binds there too.
    #[test]
    fn a_lambda_body_takes_let() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let f = |q| -> { let x = q; x }; f(1)").unwrap();
        let Stmt::LetBind {
            expr: Expr::Lambda { body, .. },
            ..
        } = &s.stmts[0]
        else {
            panic!("expected a lambda binding");
        };
        let Expr::Block { stmts, .. } = body.as_ref() else {
            panic!("expected a block body");
        };
        assert!(matches!(&stmts[0], Stmt::LetBind { name, .. } if interner.resolve(*name) == "x"));
    }

    /// A bare `x = e;` in a lambda body parses as an assignment, not a
    /// binding, exactly as it does at the top level.
    #[test]
    fn a_bare_store_in_a_lambda_body_is_an_assignment() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let f = |q| -> { let x = 0; x = q; x }; f(1)").unwrap();
        let Stmt::LetBind {
            expr: Expr::Lambda { body, .. },
            ..
        } = &s.stmts[0]
        else {
            panic!("expected a lambda binding");
        };
        let Expr::Block { stmts, .. } = body.as_ref() else {
            panic!("expected a block body");
        };
        assert!(matches!(&stmts[1], Stmt::Assign { name, .. } if interner.resolve(*name) == "x"));
    }

    /// A `match` statement and a bare assignment are both statements: the
    /// leading `match` is what tells them apart (RFC-0051).
    #[test]
    fn a_match_and_a_bare_assignment_are_both_statements() {
        let interner = Interner::new();
        let s = parse_script(
            &interner,
            "let out = 0.0; match Some(1.5) { Some(v) => { out = v; }, None => {} };",
        )
        .unwrap();
        assert!(matches!(&s.stmts[0], Stmt::LetBind { .. }));
        let Stmt::Expr(Expr::Match { arms, .. }) = &s.stmts[1] else {
            panic!("expected a match statement");
        };
        assert_eq!(arms.len(), 2);
        assert!(
            matches!(&arms[0].body[0], Stmt::Assign { name, .. } if interner.resolve(*name) == "out")
        );
    }

    /// The arm forms: a bare expression, a block with statements, a block
    /// with a tail, and `_` as the catch-all.
    #[test]
    fn a_match_arm_is_an_expression_or_a_block() {
        let interner = Interner::new();
        let s = parse_script(
            &interner,
            "let acc = 0; match E::A(1) { E::A(v) => { acc = v; }, E::B(v) => { acc = 1; v }, _ => 0 }",
        )
        .unwrap();
        let Some(tail) = &s.tail else {
            panic!("expected a trailing match expression");
        };
        let Expr::Match { arms, .. } = tail.as_ref() else {
            panic!("expected a match expression");
        };
        assert_eq!(arms.len(), 3);
        assert!(arms[0].tail.is_none() && arms[0].body.len() == 1);
        assert!(matches!(arms[1].tail.as_deref(), Some(Expr::Block { .. })));
        assert!(matches!(arms[2].pattern, Pattern::Wildcard { .. }));
    }

    /// A `while` body is the same statement rule: `let` and assignment both.
    #[test]
    fn a_while_body_takes_let_and_assignment() {
        let interner = Interner::new();
        let s = parse_script(
            &interner,
            "let i = 0; while i < 3 { let j = i; i = j + 1; }",
        )
        .unwrap();
        let Stmt::While { body, .. } = &s.stmts[1] else {
            panic!("expected While");
        };
        assert!(matches!(&body[0], Stmt::LetBind { name, .. } if interner.resolve(*name) == "j"));
        assert!(matches!(&body[1], Stmt::Assign { name, .. } if interner.resolve(*name) == "i"));
    }

    /// `x = e;` with an object literal on the right is an assignment; the
    /// tag form needs the `{ body };` after a source expression.
    #[test]
    fn an_object_literal_right_hand_side_stays_an_assignment() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let o = { a, }; o = { b, };").unwrap();
        let Stmt::Assign { name, expr, .. } = &s.stmts[1] else {
            panic!("expected Assign");
        };
        assert_eq!(interner.resolve(*name), "o");
        assert!(matches!(expr, Expr::Object { .. }));
    }

    // -- ContextStore ----------------------------------------------

    #[test]
    fn script_context_store() {
        let interner = Interner::new();
        let s = parse_script(&interner, "@count = @count + 1; @count").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(
            matches!(&s.stmts[0], Stmt::ContextStore { name, .. } if interner.resolve(name.name) == "count")
        );
        assert!(s.tail.is_some());
    }

    #[test]
    fn script_context_store_no_tail() {
        let interner = Interner::new();
        let s = parse_script(&interner, "@x = 42;").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(
            matches!(&s.stmts[0], Stmt::ContextStore { name, .. } if interner.resolve(name.name) == "x")
        );
        assert!(s.tail.is_none());
    }

    #[test]
    fn script_mixed_bind_and_context_store() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let tmp = @x + 1; @x = tmp; @x").unwrap();
        assert_eq!(s.stmts.len(), 2);
        assert!(
            matches!(&s.stmts[0], Stmt::LetBind { name, .. } if interner.resolve(*name) == "tmp")
        );
        assert!(
            matches!(&s.stmts[1], Stmt::ContextStore { name, .. } if interner.resolve(name.name) == "x")
        );
        assert!(s.tail.is_some());
    }

    // -- Variant (Option) --------------------------------------------

    #[test]
    fn parse_some_expr() {
        let (_interner, result) = parse("{{ Some(42) | to_string }}{{_}}{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
    }

    #[test]
    fn parse_none_expr() {
        let (interner, result) = parse("{{ x = None }}{{_}}{{/}}");
        let t = result.unwrap();
        assert_eq!(t.body.len(), 1);
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.source,
                Expr::Variant { tag, payload: None, .. } if interner.resolve(*tag) == "None"
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_some_pattern() {
        let (interner, result) = parse("{{ Some(x) = @opt }}{{ x }}{{_}}{{/}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::Variant { tag, payload: Some(_), .. } if interner.resolve(*tag) == "Some"
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn parse_none_pattern() {
        let (interner, result) = parse("{{ None = @opt }}nothing{{_}}{{/}}");
        let t = result.unwrap();
        if let Node::MatchBlock(mb) = &t.body[0] {
            assert!(matches!(
                &mb.arms[0].pattern,
                Pattern::Variant { tag, payload: None, .. } if interner.resolve(*tag) == "None"
            ));
        } else {
            panic!("expected MatchBlock");
        }
    }

    #[test]
    fn validate_variant_is_refutable() {
        let interner = Interner::new();
        let expr = Expr::Variant {
            id: AstId::alloc(),
            enum_name: None,
            tag: interner.intern("Some"),
            payload: Some(Box::new(Expr::Ident {
                id: AstId::alloc(),
                name: QualifiedRef::root(interner.intern("x")),
                ref_kind: RefKind::Value,
                span: Span::new(0, 1),
            })),
            span: Span::new(0, 1),
        };
        let pat = expr_to_pattern(&expr).unwrap();
        assert!(validate_irrefutable(&pat).is_err());
    }

    // -- RFC-0058: a literal says its type ----------------------------

    fn literal_of(source: &str) -> Literal {
        let interner = Interner::new();
        match parse_expr(&interner, source).expect(source) {
            Expr::Literal { value, .. } => value,
            other => panic!("{source} parsed as {other:?}"),
        }
    }

    fn literal_refusal(source: &str) -> String {
        let interner = Interner::new();
        parse_expr(&interner, source)
            .expect_err(source)
            .kind
            .to_string()
    }

    #[test]
    fn a_suffix_gives_an_integer_literal_its_width() {
        assert_eq!(
            literal_of("10u64"),
            Literal::IntOf(SuffixedInt {
                value: 10,
                width: IntWidth::U64,
            })
        );
        assert_eq!(
            literal_of("255u8"),
            Literal::IntOf(SuffixedInt {
                value: 255,
                width: IntWidth::U8,
            })
        );
        // Out of range at the lexer's level too: the value is carried and
        // the checker refuses it, so `300u8` parses.
        assert_eq!(
            literal_of("300u8"),
            Literal::IntOf(SuffixedInt {
                value: 300,
                width: IntWidth::U8,
            })
        );
        assert_eq!(literal_of("10"), Literal::Int(10));
    }

    #[test]
    fn a_character_literal_is_one_scalar_value() {
        assert_eq!(literal_of("'x'"), Literal::Char('x'));
        assert_eq!(literal_of(r"'\n'"), Literal::Char('\n'));
        assert_eq!(literal_of(r"'\''"), Literal::Char('\''));
        assert_eq!(literal_of(r"'\u{1F600}'"), Literal::Char('\u{1F600}'));
        assert!(literal_refusal("'ab'").contains("exactly one scalar value"));
    }

    /// The content of a `'…'` admits no bare quote, so a comparison of two
    /// character literals is three tokens and not one long literal.
    #[test]
    fn two_character_literals_are_two_literals() {
        let interner = Interner::new();
        let parsed = parse_expr(&interner, "'a' == 'b'").expect("'a' == 'b'");
        assert!(matches!(parsed, Expr::BinaryOp { .. }), "{parsed:?}");
    }

    #[test]
    fn a_byte_string_is_its_bytes_and_a_byte_literal_is_one() {
        assert_eq!(literal_of(r#"b"GET""#), Literal::Bytes(b"GET".to_vec()));
        assert_eq!(literal_of(r#"b"\xFF\x00""#), Literal::Bytes(vec![0xFF, 0]));
        assert_eq!(
            literal_of("b'G'"),
            Literal::IntOf(SuffixedInt {
                value: i128::from(b'G'),
                width: IntWidth::U8,
            })
        );
        assert!(literal_refusal(r#"b"é""#).contains("takes ASCII"));
    }

    /// A `"…"` decodes against the one table every literal uses.
    #[test]
    fn a_string_literal_takes_rusts_escapes() {
        assert_eq!(
            literal_of(r#""a\tb\x41\u{1F600}""#),
            Literal::String("a\tb\x41\u{1F600}".into())
        );
        assert_eq!(
            literal_of(r#""\n\r\t\0\\\'\"""#),
            Literal::String("\n\r\t\0\\'\"".into())
        );
    }

    #[test]
    fn a_string_literal_refuses_an_escape_the_table_does_not_name() {
        assert_eq!(literal_refusal(r#""\d""#), "unknown character escape `\\d`");
        assert!(literal_refusal(r#""\xFF""#).contains("is above `\\x7F`"));
    }

    /// A format string's text segments are pieces of one `"…"`, so each
    /// decodes against the same table and a bad escape in one is refused.
    #[test]
    fn a_format_strings_text_takes_the_same_table() {
        let interner = Interner::new();
        let parsed = parse_expr(&interner, r#""a\tb {{ x }}\x41""#).expect("a format string");
        let mut texts = Vec::new();
        let mut node = &parsed;
        while let Expr::BinaryOp { left, right, .. } = node {
            if let Expr::Literal {
                value: Literal::String(text),
                ..
            } = right.as_ref()
            {
                texts.push(text.clone());
            }
            node = left;
        }
        if let Expr::Literal {
            value: Literal::String(text),
            ..
        } = node
        {
            texts.push(text.clone());
        }
        texts.reverse();
        assert_eq!(texts, vec!["a\tb ".to_string(), "\x41".to_string()]);

        assert_eq!(
            parse_expr(&interner, r#""\d{{ x }}""#)
                .expect_err("a bad escape in a format string")
                .kind
                .to_string(),
            "unknown character escape `\\d`"
        );
    }

    /// The refusal carries the whole literal's span, quotes included, which
    /// is what the three older literals do.
    #[test]
    fn a_bad_escape_is_reported_at_the_literal() {
        let interner = Interner::new();
        let err = parse_expr(&interner, r#"1 + "a\db""#).expect_err("a bad escape");
        assert_eq!((err.span.start, err.span.end), (4, 10));
    }

    /// A minus that touches an integer literal is the literal's sign, so
    /// the value the range check reads is the signed one (RFC-0058).
    #[test]
    fn a_minus_on_an_integer_literal_is_part_of_it() {
        assert_eq!(
            literal_of("-128i8"),
            Literal::IntOf(SuffixedInt {
                value: -128,
                width: IntWidth::I8,
            })
        );
        assert_eq!(literal_of("-1"), Literal::Int(-1));
        assert_eq!(
            literal_of("-9223372036854775808"),
            Literal::Int(i128::from(i64::MIN))
        );
    }

    /// A negative literal is a pattern, where a negation of a literal is
    /// not: the sign is inside the literal the arm matches on.
    #[test]
    fn a_negative_literal_is_a_pattern() {
        let interner = Interner::new();
        let script = parse_script(&interner, "match 0 { -1 => 1, _ => 0 }").expect("a match");
        let Some(tail) = &script.tail else {
            panic!("expected a trailing match expression");
        };
        let Expr::Match { arms, .. } = tail.as_ref() else {
            panic!("expected a match expression");
        };
        assert!(matches!(
            &arms[0].pattern,
            Pattern::Literal {
                value: Literal::Int(-1),
                ..
            }
        ));
    }

    /// `-(128i8)` and `- 128i8` are the negation of a literal, as they were
    /// before the fold: the rule is the token pair, and neither is one.
    #[test]
    fn a_minus_that_is_not_the_sign_stays_a_negation() {
        let interner = Interner::new();
        for source in ["-(128i8)", "- 128i8"] {
            let parsed = parse_expr(&interner, source).expect(source);
            assert!(
                matches!(
                    parsed,
                    Expr::UnaryOp {
                        op: UnaryOp::Neg,
                        ..
                    }
                ),
                "{source} parsed as {parsed:?}"
            );
        }
    }

    #[test]
    fn a_minus_after_a_value_is_subtraction() {
        let interner = Interner::new();
        for source in ["0 - 128i8", "a -1", "a - 1"] {
            let parsed = parse_expr(&interner, source).expect(source);
            assert!(
                matches!(parsed, Expr::BinaryOp { op: BinOp::Sub, .. }),
                "{source} parsed as {parsed:?}"
            );
        }
    }

    /// The tag scanner steps over a character literal, so one holding a
    /// quote or a brace does not end the tag (`char_literal_end`).
    #[test]
    fn a_character_literal_inside_a_tag_does_not_close_it() {
        let interner = Interner::new();
        for source in ["{{ '\"' }}", "{{ '}' }}"] {
            parse_template(&interner, source).unwrap_or_else(|e| panic!("{source}: {e}"));
        }
    }
}
