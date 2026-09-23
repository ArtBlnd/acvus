use std::convert::Infallible;

use acvus_utils::{Interner, QualifiedRef};
use lalrpop_util::{ErrorRecovery, ParseError as LalrpopError};

use crate::ast::*;
use crate::error::{BlockStatement, Expected, Found, ParseError, ParseErrorKind};
use crate::grammar::{
    ArmLineParser, BindLineParser, ExprParser, ForLineParser, ScriptParser, TemplateStmtParser,
};
use crate::lexer::{ExprTokenizer, Line, Piece, TagEnd, scan_template};
use crate::span::Span;
use crate::token::Token;

pub(crate) type GrammarError = LalrpopError<usize, Token, ParseError>;
type Recovery = ErrorRecovery<usize, Token, ParseError>;

/// A parse that met errors: the tree it recovered and every error it met,
/// in source order.
#[derive(Debug, Clone, PartialEq)]
pub struct Recovered<T> {
    pub tree: T,
    pub errors: Vec<ParseError>,
}

impl<T> Recovered<T> {
    fn new(tree: T, mut errors: Vec<ParseError>) -> Self {
        errors.sort_by_key(|error| error.span.start);
        Self { tree, errors }
    }
}

pub(crate) trait Recover: Slot {
    type Stop: Into<ParseError>;

    fn recovered(
        errors: &mut Vec<ParseError>,
        reported: Reported,
        span: Span,
    ) -> Result<Self, Self::Stop>;
    fn resumed(errors: &mut Vec<ParseError>, reported: Reported) -> Result<(), Self::Stop>;
    fn widened(&self, span: Span) -> Self;
}

impl Recover for Clean {
    type Stop = ParseError;

    fn recovered(_: &mut Vec<ParseError>, reported: Reported, _: Span) -> Result<Self, ParseError> {
        Err(reported.first)
    }

    fn resumed(_: &mut Vec<ParseError>, reported: Reported) -> Result<(), ParseError> {
        Err(reported.first)
    }

    fn widened(&self, _: Span) -> Self {
        match *self {}
    }
}

impl Recover for ErrorNode {
    type Stop = Infallible;

    fn recovered(
        errors: &mut Vec<ParseError>,
        reported: Reported,
        span: Span,
    ) -> Result<Self, Infallible> {
        reported.record(errors);
        Ok(ErrorNode {
            id: AstId::alloc(),
            span,
        })
    }

    fn resumed(errors: &mut Vec<ParseError>, reported: Reported) -> Result<(), Infallible> {
        reported.record(errors);
        Ok(())
    }

    fn widened(&self, span: Span) -> Self {
        ErrorNode { id: self.id, span }
    }
}

impl From<Infallible> for ParseError {
    fn from(never: Infallible) -> Self {
        match never {}
    }
}

pub(crate) struct Reported {
    first: ParseError,
    stepped_over: Vec<ParseError>,
}

impl Reported {
    fn one(error: ParseError) -> Self {
        Self {
            first: error,
            stepped_over: Vec::new(),
        }
    }

    fn of(recovery: Recovery) -> Self {
        let first = convert_lalrpop_error(recovery.error);
        let stepped_over = recovery
            .dropped_tokens
            .into_iter()
            .filter_map(|(start, token, end)| match token {
                Token::Unreadable(c) if start != first.span.start => Some(ParseError::new(
                    ParseErrorKind::UnexpectedCharacter(c),
                    Span::new(start, end),
                )),
                _ => None,
            })
            .collect();
        Self {
            first,
            stepped_over,
        }
    }

    fn record(self, errors: &mut Vec<ParseError>) {
        errors.push(self.first);
        errors.extend(self.stepped_over);
    }
}

fn stopped<S>(stop: S::Stop) -> GrammarError
where
    S: Recover,
{
    LalrpopError::User { error: stop.into() }
}

pub(crate) fn recovered<S>(
    errors: &mut Vec<ParseError>,
    recovery: Recovery,
    span: Span,
) -> Result<S, GrammarError>
where
    S: Recover,
{
    S::recovered(errors, Reported::of(recovery), span).map_err(stopped::<S>)
}

pub(crate) fn resumed<S>(
    errors: &mut Vec<ParseError>,
    recovery: Recovery,
) -> Result<(), GrammarError>
where
    S: Recover,
{
    S::resumed(errors, Reported::of(recovery)).map_err(stopped::<S>)
}

pub(crate) fn refused<S>(
    errors: &mut Vec<ParseError>,
    error: ParseError,
    span: Span,
) -> Result<S, GrammarError>
where
    S: Recover,
{
    S::recovered(errors, Reported::one(error), span).map_err(stopped::<S>)
}

fn reported<S>(errors: &mut Vec<ParseError>, error: ParseError) -> Result<(), GrammarError>
where
    S: Recover,
{
    S::resumed(errors, Reported::one(error)).map_err(stopped::<S>)
}

/// `x in head`: what the `for` of a `% for` line is followed by.
pub struct ForLine<S> {
    pub binder: Binder,
    pub head: ForHead<S>,
}

/// `pattern = source`: what the `let` of a `% if let` or `% while let`
/// line is followed by.
pub struct BindLine<S> {
    pub pattern: Pattern<S>,
    pub source: Expr<S>,
}

pub(crate) fn literal_of<S, T, F>(
    errors: &mut Vec<ParseError>,
    decoded: Result<T, LiteralErrorKind>,
    span: Span,
    value: F,
) -> Result<Expr<S>, GrammarError>
where
    S: Recover,
    F: FnOnce(T) -> Literal,
{
    match decoded {
        Ok(decoded) => Ok(Expr::Literal {
            id: AstId::alloc(),
            value: value(decoded),
            span,
        }),
        Err(kind) => refused::<S>(
            errors,
            ParseError::new(ParseErrorKind::BadLiteral(kind), span),
            span,
        )
        .map(Expr::Error),
    }
}

/// Parse a single expression.
pub fn parse_expr(interner: &Interner, source: &str) -> Result<Expr, ParseError> {
    let tokenizer = ExprTokenizer::new(source, 0, interner);
    ExprParser::new()
        .parse(interner, &mut Vec::new(), tokenizer)
        .map_err(convert_lalrpop_error)
}

/// Parse a script source string. One statement grammar: `let x = e;` binds,
/// `x = e;` assigns the `x` in scope, and `if`/`while`/`anyorder` are
/// statements of the same rule in every block.
pub fn parse_script(
    interner: &Interner,
    source: &str,
) -> Result<Script, Recovered<Script<ErrorNode>>> {
    let tokenizer = ExprTokenizer::new(source, 0, interner);
    match ScriptParser::new().parse(interner, &mut Vec::new(), tokenizer) {
        Ok(script) => Ok(script),
        Err(_) => Err(recover_script(interner, source)),
    }
}

fn recover_script(interner: &Interner, source: &str) -> Recovered<Script<ErrorNode>> {
    let mut errors = Vec::new();
    let tokenizer = ExprTokenizer::new(source, 0, interner);
    let whole = Span::new(0, source.len());
    let tree = match ScriptParser::new().parse(interner, &mut errors, tokenizer) {
        Ok(script) => script,
        Err(error) => {
            let Ok(node) = ErrorNode::recovered(
                &mut errors,
                Reported::one(convert_lalrpop_error(error)),
                whole,
            );
            Script {
                id: AstId::alloc(),
                stmts: vec![Stmt::Error(node)],
                tail: None,
                span: whole,
            }
        }
    };
    Recovered::new(tree, errors)
}

/// Parse a template source string into an AST (RFC-0071).
pub fn parse_template(
    interner: &Interner,
    source: &str,
) -> Result<Template, Recovered<Template<ErrorNode>>> {
    let lines = scan_template(source);
    let span = Span::new(0, source.len());
    match Builder::<Clean>::new(interner, &mut Vec::new()).template(&lines, span) {
        Ok(template) => Ok(template),
        Err(_) => {
            let mut errors = Vec::new();
            let Ok(template) =
                Builder::<ErrorNode>::new(interner, &mut errors).template(&lines, span);
            Err(Recovered::new(template, errors))
        }
    }
}

/// A block a `%` line opened and `% end` has yet to close.
enum Open<S> {
    If(IfChain<S>),
    Match(MatchChain<S>),
    For {
        id: AstId,
        callee_id: AstId,
        binder: Binder,
        head: ForHead<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    While {
        cond: Expr<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    WhileLet {
        pattern: Pattern<S>,
        source: Expr<S>,
        body: Vec<Stmt<S>>,
        span: Span,
    },
    Anyorder {
        body: Vec<Stmt<S>>,
        span: Span,
    },
}

/// The `% if` / `% else if` / `% else` chain of one block.
struct IfChain<S> {
    first: IfArm<S>,
    else_ifs: Vec<IfArm<S>>,
    otherwise: Option<ElseArm<S>>,
    span: Span,
}

struct IfArm<S> {
    head: IfHead<S>,
    body: Vec<Stmt<S>>,
    span: Span,
}

struct ElseArm<S> {
    body: Vec<Stmt<S>>,
    span: Span,
}

enum IfHead<S> {
    Cond(Expr<S>),
    Bind {
        pattern: Pattern<S>,
        source: Expr<S>,
    },
}

/// The `% match` scrutinee and the `% pattern =>` arms opened under it.
struct MatchChain<S> {
    scrutinee: Expr<S>,
    arms: Vec<MatchExprArm<S>>,
    span: Span,
}

/// A `%` line's head, once the leading keyword has been read.
enum Head<S> {
    Open(Open<S>),
    Else(ElseArm<S>),
    ElseIf(IfArm<S>),
    Arm { pattern: Pattern<S>, span: Span },
    End(Span),
    Plain(Stmt<S>),
    Comment,
}

struct Builder<'a, 'e, S> {
    interner: &'a Interner,
    errors: &'e mut Vec<ParseError>,
    open: Vec<Open<S>>,
    body: Vec<Stmt<S>>,
}

impl<'a, 'e, S> Builder<'a, 'e, S>
where
    S: Recover,
{
    fn new(interner: &'a Interner, errors: &'e mut Vec<ParseError>) -> Self {
        Self {
            interner,
            errors,
            open: Vec::new(),
            body: Vec::new(),
        }
    }

    fn template(mut self, lines: &[Line], span: Span) -> Result<Template<S>, S::Stop> {
        for line in lines {
            self.line(line)?;
        }
        self.finish(span)
    }

    fn refused(&mut self, error: ParseError, span: Span) -> Result<S, S::Stop> {
        S::recovered(self.errors, Reported::one(error), span)
    }

    fn line(&mut self, line: &Line) -> Result<(), S::Stop> {
        match line {
            Line::Text { pieces, .. } => {
                for piece in pieces {
                    let stmt = self.append_of(piece)?;
                    self.push(stmt)?;
                }
                Ok(())
            }
            Line::Stmt { content, span } => {
                let head = match self.head(content, *span) {
                    Ok(head) => head,
                    Err(error) => Head::Plain(Stmt::Error(self.refused(error, *span)?)),
                };
                match head {
                    Head::Comment => Ok(()),
                    Head::Plain(stmt) => self.push(stmt),
                    Head::Open(open) => {
                        self.open.push(open);
                        Ok(())
                    }
                    Head::End(span) => self.close(span),
                    Head::ElseIf(arm) => self.else_if(arm, *span),
                    Head::Else(arm) => self.otherwise(arm, *span),
                    Head::Arm { pattern, span } => self.arm(pattern, span),
                }
            }
        }
    }

    /// One piece of a text line, as the append it is (RFC-0071 rule 2).
    fn append_of(&mut self, piece: &Piece) -> Result<Stmt<S>, S::Stop> {
        match piece {
            Piece::Text { value, span } => Ok(append(
                Expr::Literal {
                    id: AstId::alloc(),
                    value: Literal::String(value.clone()),
                    span: *span,
                },
                *span,
            )),
            Piece::Tag {
                content,
                span,
                inner_span,
                end,
            } => {
                match end {
                    TagEnd::Closed => {}
                    TagEnd::LineEnd => S::resumed(
                        self.errors,
                        Reported::one(ParseError::new(ParseErrorKind::UnclosedTag, *span)),
                    )?,
                }
                let parsed = Rest {
                    text: content,
                    at: inner_span.start,
                }
                .parse(
                    self.interner,
                    self.errors,
                    |interner, errors, tokenizer| {
                        ExprParser::new().parse(interner, errors, tokenizer)
                    },
                );
                let expr = match parsed {
                    Ok(expr) => expr,
                    Err(error) => Expr::Error(self.refused(error, *inner_span)?),
                };
                Ok(append(expr, *span))
            }
        }
    }

    /// Read a `%` line: the leading keyword names the rule the rest of the
    /// line is parsed by, so the grammar has no line form to disambiguate.
    fn head(&mut self, content: &str, span: Span) -> Result<Head<S>, ParseError> {
        let tokens: Vec<(usize, Token, usize)> =
            ExprTokenizer::new(content, span.start, self.interner).collect();
        let Some((_, first, first_end)) = tokens.first().cloned() else {
            return Ok(Head::Comment);
        };
        let rest = |at: usize| Rest {
            text: &content[at - span.start..],
            at,
        };

        if let Token::Ident(name) = first
            && tokens.len() == 1
            && self.interner.resolve(name) == "end"
        {
            return Ok(Head::End(span));
        }
        if let Some((arrow_start, _, _)) = tokens.last().filter(|t| t.1 == Token::FatArrow) {
            let head = Rest {
                text: &content[..arrow_start - span.start],
                at: span.start,
            };
            let pattern =
                head.parse(self.interner, self.errors, |interner, errors, tokenizer| {
                    ArmLineParser::new().parse(interner, errors, tokenizer)
                })?;
            return Ok(Head::Arm { pattern, span });
        }

        match first {
            Token::Else => self.else_head(&tokens, rest(first_end), span),
            Token::If => Ok(Head::Open(Open::If(IfChain {
                first: self.if_arm(&tokens, rest(first_end), span)?,
                else_ifs: Vec::new(),
                otherwise: None,
                span,
            }))),
            Token::For => {
                let line = rest(first_end).parse(
                    self.interner,
                    self.errors,
                    |interner, errors, tokenizer| {
                        ForLineParser::new().parse(interner, errors, tokenizer)
                    },
                )?;
                Ok(Head::Open(Open::For {
                    id: AstId::alloc(),
                    callee_id: AstId::alloc(),
                    binder: line.binder,
                    head: line.head,
                    body: Vec::new(),
                    span,
                }))
            }
            Token::While => match tokens.get(1).map(|t| &t.1) {
                Some(Token::Let) => {
                    let bind = rest(tokens[1].2).parse(
                        self.interner,
                        self.errors,
                        |interner, errors, tokenizer| {
                            BindLineParser::new().parse(interner, errors, tokenizer)
                        },
                    )?;
                    Ok(Head::Open(Open::WhileLet {
                        pattern: bind.pattern,
                        source: bind.source,
                        body: Vec::new(),
                        span,
                    }))
                }
                _ => Ok(Head::Open(Open::While {
                    cond: rest(first_end).parse(
                        self.interner,
                        self.errors,
                        |interner, errors, tokenizer| {
                            ExprParser::new().parse(interner, errors, tokenizer)
                        },
                    )?,
                    body: Vec::new(),
                    span,
                })),
            },
            Token::Match => Ok(Head::Open(Open::Match(MatchChain {
                scrutinee: rest(first_end).parse(
                    self.interner,
                    self.errors,
                    |interner, errors, tokenizer| {
                        ExprParser::new().parse(interner, errors, tokenizer)
                    },
                )?,
                arms: Vec::new(),
                span,
            }))),
            Token::Anyorder if tokens.len() == 1 => Ok(Head::Open(Open::Anyorder {
                body: Vec::new(),
                span,
            })),
            _ => Ok(Head::Plain(
                Rest {
                    text: content,
                    at: span.start,
                }
                .parse(
                    self.interner,
                    self.errors,
                    |interner, errors, tokenizer| {
                        TemplateStmtParser::new().parse(interner, errors, tokenizer)
                    },
                )?,
            )),
        }
    }

    fn if_arm(
        &mut self,
        tokens: &[(usize, Token, usize)],
        rest: Rest<'_>,
        span: Span,
    ) -> Result<IfArm<S>, ParseError> {
        let head = match tokens.get(1).map(|t| &t.1) {
            Some(Token::Let) => {
                let bind = Rest {
                    text: &rest.text[tokens[1].2 - rest.at..],
                    at: tokens[1].2,
                }
                .parse(
                    self.interner,
                    self.errors,
                    |interner, errors, tokenizer| {
                        BindLineParser::new().parse(interner, errors, tokenizer)
                    },
                )?;
                IfHead::Bind {
                    pattern: bind.pattern,
                    source: bind.source,
                }
            }
            _ => IfHead::Cond(rest.parse(
                self.interner,
                self.errors,
                |interner, errors, tokenizer| ExprParser::new().parse(interner, errors, tokenizer),
            )?),
        };
        Ok(IfArm {
            head,
            body: Vec::new(),
            span,
        })
    }

    fn else_head(
        &mut self,
        tokens: &[(usize, Token, usize)],
        rest: Rest<'_>,
        span: Span,
    ) -> Result<Head<S>, ParseError> {
        match tokens.get(1).map(|t| &t.1) {
            None => Ok(Head::Else(ElseArm {
                body: Vec::new(),
                span,
            })),
            Some(Token::If) => Ok(Head::ElseIf(self.if_arm(
                &tokens[1..],
                Rest {
                    text: &rest.text[tokens[1].2 - rest.at..],
                    at: tokens[1].2,
                },
                span,
            )?)),
            Some(_) => Err(ParseError::new(ParseErrorKind::ElseOutsideIf, span)),
        }
    }

    /// The body the next statement joins: the innermost open block's, or
    /// the template's own.
    fn body_mut(&mut self) -> &mut Vec<Stmt<S>> {
        for open in self.open.iter_mut().rev() {
            match open {
                Open::If(chain) => {
                    return match &mut chain.otherwise {
                        Some(arm) => &mut arm.body,
                        None => &mut chain.else_ifs.last_mut().unwrap_or(&mut chain.first).body,
                    };
                }
                Open::Match(chain) => match chain.arms.last_mut() {
                    Some(arm) => return &mut arm.body,
                    None => continue,
                },
                Open::For { body, .. }
                | Open::While { body, .. }
                | Open::WhileLet { body, .. }
                | Open::Anyorder { body, .. } => return body,
            }
        }
        &mut self.body
    }

    fn push(&mut self, stmt: Stmt<S>) -> Result<(), S::Stop> {
        let stmt = match self.open.last() {
            Some(Open::Match(chain)) if chain.arms.is_empty() => {
                let span = stmt_span(&stmt);
                Stmt::Error(self.refused(
                    ParseError::new(ParseErrorKind::MatchBodyBeforeArm, span),
                    span,
                )?)
            }
            Some(_) | None => stmt,
        };
        self.body_mut().push(stmt);
        Ok(())
    }

    fn else_if(&mut self, arm: IfArm<S>, span: Span) -> Result<(), S::Stop> {
        let refusal = match self.if_chain_mut() {
            None => ParseErrorKind::ElseOutsideIf,
            Some(chain) if chain.otherwise.is_some() => ParseErrorKind::ElseAfterElse,
            Some(chain) => {
                chain.else_ifs.push(arm);
                return Ok(());
            }
        };
        let node = self.refused(ParseError::new(refusal, span), span)?;
        self.push(Stmt::Error(node))
    }

    fn otherwise(&mut self, arm: ElseArm<S>, span: Span) -> Result<(), S::Stop> {
        let refusal = match self.if_chain_mut() {
            None => ParseErrorKind::ElseOutsideIf,
            Some(chain) if chain.otherwise.is_some() => ParseErrorKind::ElseAfterElse,
            Some(chain) => {
                chain.otherwise = Some(arm);
                return Ok(());
            }
        };
        let node = self.refused(ParseError::new(refusal, span), span)?;
        self.push(Stmt::Error(node))
    }

    fn if_chain_mut(&mut self) -> Option<&mut IfChain<S>> {
        match self.open.last_mut() {
            Some(Open::If(chain)) => Some(chain),
            _ => None,
        }
    }

    fn arm(&mut self, pattern: Pattern<S>, span: Span) -> Result<(), S::Stop> {
        let Some(Open::Match(chain)) = self.open.last_mut() else {
            let node =
                self.refused(ParseError::new(ParseErrorKind::ArmOutsideMatch, span), span)?;
            return self.push(Stmt::Error(node));
        };
        chain.arms.push(MatchExprArm {
            id: AstId::alloc(),
            pattern,
            body: Vec::new(),
            tail: None,
            span,
        });
        Ok(())
    }

    fn close(&mut self, span: Span) -> Result<(), S::Stop> {
        let Some(open) = self.open.pop() else {
            let node = self.refused(ParseError::new(ParseErrorKind::UnmatchedEnd, span), span)?;
            return self.push(Stmt::Error(node));
        };
        let stmt = closed(open, span);
        self.push(stmt)
    }

    fn finish(mut self, span: Span) -> Result<Template<S>, S::Stop> {
        let end = Span::new(span.end, span.end);
        while let Some(open) = self.open.pop() {
            S::resumed(
                self.errors,
                Reported::one(ParseError::new(
                    ParseErrorKind::UnclosedBlock,
                    open_span(&open),
                )),
            )?;
            let stmt = closed(open, end);
            self.push(stmt)?;
        }
        Ok(Template {
            id: AstId::alloc(),
            body: std::mem::take(&mut self.body),
            span,
        })
    }
}

/// The text of a `%` line that follows the keyword already read, and where
/// it begins in the source.
struct Rest<'a> {
    text: &'a str,
    at: usize,
}

impl Rest<'_> {
    fn parse<T, F>(
        &self,
        interner: &Interner,
        errors: &mut Vec<ParseError>,
        parse: F,
    ) -> Result<T, ParseError>
    where
        F: FnOnce(&Interner, &mut Vec<ParseError>, ExprTokenizer<'_>) -> Result<T, GrammarError>,
    {
        parse(
            interner,
            errors,
            ExprTokenizer::new(self.text, self.at, interner),
        )
        .map_err(convert_lalrpop_error)
    }
}

fn append<S>(expr: Expr<S>, span: Span) -> Stmt<S> {
    Stmt::Append {
        id: AstId::alloc(),
        expr,
        span,
    }
}

fn open_span<S>(open: &Open<S>) -> Span {
    match open {
        Open::If(chain) => chain.span,
        Open::Match(chain) => chain.span,
        Open::For { span, .. }
        | Open::While { span, .. }
        | Open::WhileLet { span, .. }
        | Open::Anyorder { span, .. } => *span,
    }
}

fn stmt_span<S>(stmt: &Stmt<S>) -> Span
where
    S: Slot,
{
    match stmt {
        Stmt::Store { span, .. }
        | Stmt::DerefStore { span, .. }
        | Stmt::LetBind { span, .. }
        | Stmt::LetUninit { span, .. }
        | Stmt::Assign { span, .. }
        | Stmt::While { span, .. }
        | Stmt::For { span, .. }
        | Stmt::Break { span, .. }
        | Stmt::Continue { span, .. }
        | Stmt::WhileLet { span, .. }
        | Stmt::Anyorder { span, .. }
        | Stmt::Append { span, .. } => *span,
        Stmt::Expr(expr) => expr.span(),
        Stmt::Error(node) => node.node().span,
    }
}

/// The statement a `% end` closes the block into.
fn closed<S>(open: Open<S>, end: Span) -> Stmt<S> {
    match open {
        Open::If(chain) => Stmt::Expr(if_expr_of(chain, end)),
        Open::Match(chain) => Stmt::Expr(Expr::Match {
            id: AstId::alloc(),
            scrutinee: Box::new(chain.scrutinee),
            arms: chain.arms,
            span: chain.span.merge(end),
        }),
        Open::For {
            id,
            callee_id,
            binder,
            head,
            body,
            span,
        } => Stmt::For {
            id,
            callee_id,
            binder,
            head,
            body,
            span: span.merge(end),
        },
        Open::While { cond, body, span } => Stmt::While {
            id: AstId::alloc(),
            cond,
            body,
            span: span.merge(end),
        },
        Open::WhileLet {
            pattern,
            source,
            body,
            span,
        } => Stmt::WhileLet {
            id: AstId::alloc(),
            pattern,
            source,
            body,
            span: span.merge(end),
        },
        Open::Anyorder { body, span } => Stmt::Anyorder {
            id: AstId::alloc(),
            body,
            span: span.merge(end),
        },
    }
}

fn if_expr_of<S>(chain: IfChain<S>, end: Span) -> Expr<S> {
    let mut branch = chain.otherwise.map(|arm| {
        Box::new(ElseBranch::Else {
            body: arm.body,
            tail: None,
            span: arm.span.merge(end),
        })
    });
    for arm in chain.else_ifs.into_iter().rev() {
        branch = Some(Box::new(ElseBranch::ElseIf(if_arm_expr(arm, branch, end))));
    }
    if_arm_expr(chain.first, branch, end)
}

fn if_arm_expr<S>(arm: IfArm<S>, branch: Option<Box<ElseBranch<S>>>, end: Span) -> Expr<S> {
    let span = arm.span.merge(end);
    match arm.head {
        IfHead::Cond(cond) => Expr::If {
            id: AstId::alloc(),
            cond: Box::new(cond),
            then_body: arm.body,
            then_tail: None,
            else_branch: branch,
            span,
        },
        IfHead::Bind { pattern, source } => Expr::IfLet {
            id: AstId::alloc(),
            pattern,
            source: Box::new(source),
            then_body: arm.body,
            then_tail: None,
            else_branch: branch,
            span,
        },
    }
}

/// Convert a LALRPOP error to our ParseError. The `expected` set arrives as
/// the grammar's terminal names, which `Expected` maps to the nonterminal
/// they stand for.
fn convert_lalrpop_error(error: GrammarError) -> ParseError {
    match error {
        LalrpopError::InvalidToken { location } => ParseError::new(
            ParseErrorKind::InvalidToken,
            Span::new(location, location + 1),
        ),
        LalrpopError::UnrecognizedEof {
            location,
            expected: _,
        } => ParseError::new(ParseErrorKind::UnexpectedEof, Span::new(location, location)),
        LalrpopError::UnrecognizedToken {
            token: (start, Token::Unreadable(c), end),
            expected: _,
        }
        | LalrpopError::ExtraToken {
            token: (start, Token::Unreadable(c), end),
        } => ParseError::new(
            ParseErrorKind::UnexpectedCharacter(c),
            Span::new(start, end),
        ),
        // An empty expected set is the grammar saying nothing may follow,
        // which is what `ExtraToken` reports. A template's `%` line reaches
        // it wherever a line holds a complete statement and more text.
        LalrpopError::UnrecognizedToken {
            token: (start, tok, end),
            expected,
        } if expected.is_empty() => ParseError::new(
            ParseErrorKind::ExtraToken {
                found: Found::of(&tok),
            },
            Span::new(start, end),
        ),
        LalrpopError::UnrecognizedToken {
            token: (start, tok, end),
            expected,
        } => ParseError::new(
            ParseErrorKind::UnexpectedToken {
                found: Found::of(&tok),
                expected: Expected::of_grammar_names(expected.iter().map(String::as_str)),
            },
            Span::new(start, end),
        ),
        LalrpopError::ExtraToken {
            token: (start, tok, end),
        } => ParseError::new(
            ParseErrorKind::ExtraToken {
                found: Found::of(&tok),
            },
            Span::new(start, end),
        ),
        LalrpopError::User { error } => error,
    }
}

/// `place = value;`. A bare name stays a statement of its own because
/// `acvus-mir`'s checker refuses assigning a name that is not bound, or one
/// the enclosing lambda captured, against the bindings in scope.
pub(crate) fn build_assign<S>(
    errors: &mut Vec<ParseError>,
    lhs: Expr<S>,
    rhs: Expr<S>,
    span: Span,
) -> Result<Stmt<S>, GrammarError>
where
    S: Recover,
{
    match lhs {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } => Ok(Stmt::Assign {
            id: AstId::alloc(),
            name: name.name,
            expr: rhs,
            span,
        }),
        Expr::UnaryOp {
            op: UnaryOp::Deref,
            operand,
            ..
        } => Ok(Stmt::DerefStore {
            id: AstId::alloc(),
            target: operand,
            expr: rhs,
            span,
        }),
        Expr::Error(target) => Ok(Stmt::Error(target.widened(span))),
        lhs => match Place::of(lhs) {
            Some(place) => Ok(Stmt::Store {
                id: AstId::alloc(),
                place,
                expr: rhs,
                span,
            }),
            None => refused::<S>(
                errors,
                ParseError::new(ParseErrorKind::InvalidAssignTarget, span),
                span,
            )
            .map(Stmt::Error),
        },
    }
}

/// The end of a block statement: the `}` closing its block, with no `;`.
pub(crate) fn block_closed<S>(
    errors: &mut Vec<ParseError>,
    block: BlockStatement,
    semicolon: Option<Span>,
) -> Result<(), GrammarError>
where
    S: Recover,
{
    match semicolon {
        None => Ok(()),
        Some(span) => reported::<S>(
            errors,
            ParseError::new(ParseErrorKind::SemicolonAfterBlock(block), span),
        ),
    }
}

/// Statements whose last `if` or `match` a following statement or tail
/// made a statement of its own.
pub(crate) fn stated<S>((mut stmts, last): (Vec<Stmt<S>>, Expr<S>)) -> Vec<Stmt<S>> {
    stmts.push(Stmt::Expr(last));
    stmts
}

/// `ns::f(args)` and `Enum::Tag(payload)` leave this function as one shape,
/// a call of a qualified name: which of the two a `QualifiedRef` names is
/// decided in `acvus-mir`'s checker, against the names in scope (RFC-0030).
pub fn build_call<S>(func: Expr<S>, args: Vec<Expr<S>>, span: Span) -> Expr<S> {
    let func = match func {
        Expr::Variant {
            enum_name: Some(namespace),
            tag,
            payload: None,
            ..
        } => Expr::Ident {
            id: AstId::alloc(),
            name: QualifiedRef::qualified(namespace, tag),
            ref_kind: RefKind::Value,
            span,
        },
        other => other,
    };
    Expr::FuncCall {
        id: AstId::alloc(),
        func: Box::new(func),
        args,
        span,
    }
}

/// Convert an expression (parsed from the LHS of `=`) to a pattern.
pub(crate) fn pattern_of<S>(
    errors: &mut Vec<ParseError>,
    expr: &Expr<S>,
) -> Result<Pattern<S>, GrammarError>
where
    S: Recover,
{
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
        } => Ok(Pattern::List {
            id: AstId::alloc(),
            head: patterns_of(errors, head)?,
            rest: *rest,
            tail: patterns_of(errors, tail)?,
            span: *span,
        }),
        Expr::Object { fields, span, .. } => {
            let mut pattern_fields = Vec::with_capacity(fields.len());
            for field in fields {
                pattern_fields.push(ObjectPatternField {
                    id: AstId::alloc(),
                    key: field.key,
                    pattern: pattern_of(errors, &field.value)?,
                    span: field.span,
                });
            }
            Ok(Pattern::Object {
                id: AstId::alloc(),
                fields: pattern_fields,
                span: *span,
            })
        }
        Expr::Tuple { elements, span, .. } => {
            let mut elems = Vec::with_capacity(elements.len());
            for elem in elements {
                elems.push(match elem {
                    TupleElem::Wildcard(s) => TuplePatternElem::Wildcard(*s),
                    TupleElem::Expr(e) => TuplePatternElem::Pattern(pattern_of(errors, e)?),
                });
            }
            Ok(Pattern::Tuple {
                id: AstId::alloc(),
                elements: elems,
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
                payload: Some(Box::new(pattern_of(errors, &args[0])?)),
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
                Some(inner) => Some(Box::new(pattern_of(errors, inner)?)),
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
        Expr::Error(node) => Ok(Pattern::Error(node.clone())),
        other => refused::<S>(
            errors,
            ParseError::new(
                ParseErrorKind::InvalidPattern("expression cannot be used as a pattern".into()),
                other.span(),
            ),
            other.span(),
        )
        .map(Pattern::Error),
    }
}

fn patterns_of<S>(
    errors: &mut Vec<ParseError>,
    exprs: &[Expr<S>],
) -> Result<Vec<Pattern<S>>, GrammarError>
where
    S: Recover,
{
    exprs.iter().map(|expr| pattern_of(errors, expr)).collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    fn template(src: &str) -> Template {
        let interner = Interner::new();
        parse_template(&interner, src).unwrap_or_else(|e| panic!("{src}: {:?}", e.errors))
    }

    fn first_error<T>(recovered: Recovered<T>) -> ParseError {
        recovered
            .errors
            .into_iter()
            .next()
            .expect("a recovered parse reports its errors")
    }

    fn refusal(src: &str) -> ParseErrorKind {
        let interner = Interner::new();
        first_error(parse_template(&interner, src).expect_err(src)).kind
    }

    fn appended<S>(stmt: &Stmt<S>) -> &Expr<S>
    where
        S: std::fmt::Debug,
    {
        let Stmt::Append { expr, .. } = stmt else {
            panic!("{stmt:?}");
        };
        expr
    }

    fn text_of(stmt: &Stmt) -> &str {
        let Expr::Literal {
            value: Literal::String(text),
            ..
        } = appended(stmt)
        else {
            panic!("{stmt:?}");
        };
        text
    }

    /// A line that does not begin with `%` is text, its newline included
    /// (RFC-0071 rule 2).
    #[test]
    fn a_text_line_is_one_append_with_its_newline() {
        let t = template("hello world\n");
        assert_eq!(t.body.len(), 1);
        assert_eq!(text_of(&t.body[0]), "hello world\n");
    }

    #[test]
    fn a_tag_is_an_append_of_its_expression() {
        let t = template("a {{ name }} b");
        assert_eq!(t.body.len(), 3);
        assert_eq!(text_of(&t.body[0]), "a ");
        assert!(matches!(appended(&t.body[1]), Expr::Ident { .. }));
        assert_eq!(text_of(&t.body[2]), " b");
    }

    #[test]
    fn a_percent_line_is_a_statement() {
        let t = template("% let x = 1\n{{ x }}");
        assert!(matches!(&t.body[0], Stmt::LetBind { .. }));
        assert!(matches!(&t.body[1], Stmt::Append { .. }));
    }

    #[test]
    fn a_percent_may_be_indented() {
        let t = template("    % let x = 1\n");
        assert_eq!(t.body.len(), 1);
        assert!(matches!(&t.body[0], Stmt::LetBind { .. }));
    }

    #[test]
    fn a_double_percent_line_is_text_holding_one() {
        assert_eq!(text_of(&template("%% literal\n").body[0]), "% literal\n");
    }

    #[test]
    fn a_trailing_backslash_drops_the_newline() {
        let t = template("a\\\nb\n");
        assert_eq!(text_of(&t.body[0]), "a");
        assert_eq!(text_of(&t.body[1]), "b\n");
    }

    #[test]
    fn a_comment_line_leaves_nothing() {
        assert!(template("% // what follows\nx").body.len() == 1);
    }

    #[test]
    fn a_for_block_closes_at_end() {
        let t = template("% for m in &$xs\n- {{ &m }}\n% end\n");
        let Stmt::For { body, .. } = &t.body[0] else {
            panic!("{:?}", t.body[0]);
        };
        assert_eq!(body.len(), 3);
        assert_eq!(text_of(&body[0]), "- ");
        assert_eq!(text_of(&body[2]), "\n");
    }

    #[test]
    fn an_if_chain_is_one_expression_statement() {
        let t = template("% if $a\nA\n% else if $b\nB\n% else\nC\n% end\n");
        let Stmt::Expr(Expr::If {
            then_body,
            else_branch,
            ..
        }) = &t.body[0]
        else {
            panic!("{:?}", t.body[0]);
        };
        assert_eq!(text_of(&then_body[0]), "A\n");
        let Some(branch) = else_branch else {
            panic!("expected an else branch");
        };
        let ElseBranch::ElseIf(Expr::If {
            then_body,
            else_branch: Some(tail),
            ..
        }) = branch.as_ref()
        else {
            panic!("{branch:?}");
        };
        assert_eq!(text_of(&then_body[0]), "B\n");
        let ElseBranch::Else { body, .. } = tail.as_ref() else {
            panic!("{tail:?}");
        };
        assert_eq!(text_of(&body[0]), "C\n");
    }

    #[test]
    fn an_if_let_block_binds() {
        let t = template("% if let Some(v) = $o\n{{ v }}\n% end\n");
        assert!(matches!(&t.body[0], Stmt::Expr(Expr::IfLet { .. })));
    }

    #[test]
    fn a_match_takes_one_arm_per_line() {
        let t = template("% match $m\n% \"a\" =>\nA\n% _ =>\nB\n% end\n");
        let Stmt::Expr(Expr::Match { arms, .. }) = &t.body[0] else {
            panic!("{:?}", t.body[0]);
        };
        assert_eq!(arms.len(), 2);
        assert!(matches!(
            &arms[0].pattern,
            Pattern::Literal {
                value: Literal::String(_),
                ..
            }
        ));
        assert!(matches!(&arms[1].pattern, Pattern::Wildcard { .. }));
        assert_eq!(text_of(&arms[1].body[0]), "B\n");
    }

    #[test]
    fn a_while_and_an_anyorder_are_blocks() {
        assert!(matches!(
            &template("% while $c\nx\n% end\n").body[0],
            Stmt::While { .. }
        ));
        assert!(matches!(
            &template("% while let Some(v) = f()\n{{ v }}\n% end\n").body[0],
            Stmt::WhileLet { .. }
        ));
        assert!(matches!(
            &template("% anyorder\nx\n% end\n").body[0],
            Stmt::Anyorder { .. }
        ));
    }

    /// An `if` and a `match` are operands of the expression grammar, so
    /// inline branching needs no template form (RFC-0071).
    #[test]
    fn an_inline_if_is_an_expression() {
        let t = template(r#"{{ if $c { "a" } else { "b" } }}"#);
        assert!(matches!(appended(&t.body[0]), Expr::If { .. }));
    }

    /// The tag's content is tokenized, so a string literal inside it may
    /// hold the delimiters (RFC-0071 rule 3).
    #[test]
    fn a_tag_writes_a_literal_brace_pair() {
        let t = template(r#"{{ "{{" }}"#);
        let Expr::Literal {
            value: Literal::String(text),
            ..
        } = appended(&t.body[0])
        else {
            panic!("{:?}", t.body[0]);
        };
        assert_eq!(text, "{{");
    }

    #[test]
    fn each_block_structure_fault_is_its_own_refusal() {
        assert_eq!(refusal("% end\n"), ParseErrorKind::UnmatchedEnd);
        assert_eq!(
            refusal("% for x in &$v\na\n"),
            ParseErrorKind::UnclosedBlock
        );
        assert_eq!(refusal("% else\n"), ParseErrorKind::ElseOutsideIf);
        assert_eq!(
            refusal("% if $c\na\n% else\nb\n% else\nc\n% end\n"),
            ParseErrorKind::ElseAfterElse
        );
        assert_eq!(refusal("% 1 =>\n% end\n"), ParseErrorKind::ArmOutsideMatch);
        assert_eq!(
            refusal("% match $m\ntext\n% _ =>\na\n% end\n"),
            ParseErrorKind::MatchBodyBeforeArm
        );
    }

    /// A `%` line the statement grammar does not admit is a diagnostic at
    /// the line, not text. The message is total: a grammar state that
    /// admits nothing more reports what stands after the statement.
    #[test]
    fn a_percent_line_that_does_not_parse_is_refused() {
        let interner = Interner::new();
        let err = first_error(
            parse_template(&interner, "ok\n% let = \nmore\n").expect_err("a malformed line"),
        );
        assert_eq!(err.span, Span::new(9, 10));
        assert_eq!(err.kind.to_string(), "expected a name, found `=`");

        let err = first_error(
            parse_template(&interner, "% if $c { 1 } else { 2 }\n")
                .expect_err("an inline `if` on a `%` line"),
        );
        assert_eq!(err.kind.to_string(), "found `{` after the end of the input");
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
            matches!(&s.stmts[0], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "x")
        );
        assert!(matches!(
            s.tail.as_deref(),
            Some(Expr::Ident { name, ref_kind: RefKind::Value, .. }) if interner.resolve(name.name) == "x"
        ));
    }

    #[test]
    fn a_semicolon_after_a_block_statement_is_reported_at_the_semicolon() {
        let interner = Interner::new();
        for (src, block) in [
            ("for x in xs { } ; 1", "a `for` block"),
            ("while c { } ; 1", "a `while` block"),
            ("while let Some(x) = o { } ; 1", "a `while` block"),
            ("anyorder { } ; 1", "an `anyorder` block"),
        ] {
            let error = first_error(parse_script(&interner, src).unwrap_err());
            let at = src.find(';').unwrap();
            assert_eq!(error.span, Span::new(at, at + 1), "{src}");
            assert_eq!(
                error.kind.to_string(),
                format!("`;` is not allowed after {block}"),
                "{src}"
            );
        }
        assert!(parse_script(&interner, "for x in xs { } while c { } anyorder { } 1").is_ok());
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
            matches!(&s.stmts[0], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "x")
        );
        assert!(
            matches!(&s.stmts[1], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "y")
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

    // -- `//` to end of line -------------------------------------------

    #[test]
    fn a_line_comment_and_a_trailing_comment_are_whitespace() {
        let interner = Interner::new();
        let source = "// what this script does\nlet x = @data; // the binding\nx\n";
        let s = parse_script(&interner, source).unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(
            matches!(&s.stmts[0], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "x")
        );
        let tail = s.tail.as_deref().expect("the tail is the last line");
        let at = source.rfind('x').expect("the tail's own byte");
        assert_eq!(tail.span(), Span::new(at, at + 1));
    }

    #[test]
    fn a_comment_ends_at_the_newline() {
        let interner = Interner::new();
        let s = parse_script(&interner, "1 + // and the rest of this line\n2").unwrap();
        assert!(s.stmts.is_empty());
        assert!(matches!(
            s.tail.as_deref(),
            Some(Expr::BinaryOp { op: BinOp::Add, .. })
        ));
    }

    /// The string lexer owns a literal's extent, so a `//` between its
    /// quotes is two characters of text and not the start of a comment.
    #[test]
    fn a_double_slash_inside_a_string_is_text() {
        assert_eq!(
            literal_of(r#""https://acvus.example/a""#),
            Literal::String("https://acvus.example/a".to_string())
        );
    }

    /// A tag's extent is one line, so a comment inside one ends at the
    /// tag's `}}`.
    #[test]
    fn a_comment_inside_a_tag_is_whitespace() {
        let interner = Interner::new();
        let t = parse_template(&interner, "{{ user // the name it carries }}").unwrap();
        assert_eq!(t.body.len(), 1);
        let Stmt::Append { expr, .. } = &t.body[0] else {
            panic!("{:?}", t.body[0]);
        };
        assert!(matches!(
            expr,
            Expr::Ident { name, .. } if interner.resolve(name.name) == "user"
        ));
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
        assert!(
            matches!(&stmts[0], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "x")
        );
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

    /// An object literal's trailing comma is required -- a decision, not a
    /// hole: the comma is what tells `{ g, }`, the object, from `{ g }`, the
    /// block whose tail is `g`.
    #[test]
    fn an_object_literal_requires_its_trailing_comma() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let o = { g: 1, };").unwrap();
        let Stmt::LetBind { expr, .. } = &s.stmts[0] else {
            panic!("expected a let");
        };
        assert!(matches!(expr, Expr::Object { .. }));
        assert!(parse_script(&interner, "let o = { g: 1 };").is_err());
        // A list, a tuple and a call each admit one and require none.
        for src in [
            "let v = [1, 2];",
            "let v = [1, 2, ];",
            "let t = (1, 2);",
            "let t = (1, 2, );",
            "let x = f(1, 2);",
            "let x = f(1, 2, );",
        ] {
            parse_script(&interner, src).unwrap();
        }
    }

    /// A `match` and an `if` stand wherever an expression stands: an
    /// operator's side, a call argument, a parenthesized expression, a list
    /// element, an object field value, a method receiver, a scrutinee.
    #[test]
    fn a_match_and_an_if_are_operands() {
        let interner = Interner::new();
        let one = |src: &str| -> Expr {
            let s = parse_script(&interner, src).unwrap();
            let Stmt::LetBind { expr, .. } = &s.stmts[0] else {
                panic!("expected a let for {src}");
            };
            expr.clone()
        };
        let m = "match 1 { 1 => 2, _ => 3, }";
        let i = "if c { 1 } else { 2 }";
        for head in [m, i] {
            assert!(matches!(
                one(&format!("let x = 10 + {head};")),
                Expr::BinaryOp { right, .. } if matches!(*right, Expr::Match { .. } | Expr::If { .. })
            ));
            assert!(matches!(
                one(&format!("let x = f({head});")),
                Expr::FuncCall { args, .. } if matches!(args[0], Expr::Match { .. } | Expr::If { .. })
            ));
            assert!(matches!(
                one(&format!("let x = ({head}) + 10;")),
                Expr::BinaryOp { left, .. } if matches!(*left, Expr::Paren { .. })
            ));
            assert!(matches!(
                one(&format!("let x = [{head}, 1];")),
                Expr::List { head: elems, .. } if matches!(elems[0], Expr::Match { .. } | Expr::If { .. })
            ));
            assert!(matches!(
                one(&format!("let x = {{ f: {head}, }};")),
                Expr::Object { fields, .. } if matches!(fields[0].value, Expr::Match { .. } | Expr::If { .. })
            ));
            assert!(matches!(
                one(&format!("let x = {head}.to_string();")),
                Expr::MethodCall { .. }
            ));
            assert!(matches!(
                one(&format!("let x = 1 | {head};")),
                Expr::Pipe { .. }
            ));
            assert!(matches!(
                one(&format!("let x = match {head} {{ _ => 0, }};")),
                Expr::Match { .. }
            ));
        }
        // A statement that begins with one ends in `;` or at its `}`.
        let s = parse_script(&interner, "if c { f(); }; match 1 { _ => { f(); }, };").unwrap();
        assert!(matches!(&s.stmts[0], Stmt::Expr(Expr::If { .. })));
        assert!(matches!(&s.stmts[1], Stmt::Expr(Expr::Match { .. })));
    }

    /// An `if` or a `match` that begins a statement ends at its `}`:
    /// followed by more of the body it is a statement, and at the end of the
    /// body it is the tail. A block reads on.
    #[test]
    fn a_block_ended_expression_is_a_statement_without_a_semicolon() {
        let interner = Interner::new();
        for src in [
            "if c { f(); } x",
            "if c { f(); } else { g(); } x",
            "if let Some(y) = o { f(); } x",
            "match n { 1 => { f(); }, _ => {} } x",
        ] {
            let s = parse_script(&interner, src).unwrap();
            assert_eq!(s.stmts.len(), 1, "{src}");
            assert!(
                matches!(
                    &s.stmts[0],
                    Stmt::Expr(Expr::If { .. } | Expr::IfLet { .. } | Expr::Match { .. })
                ),
                "{src}"
            );
            assert!(
                matches!(s.tail.as_deref(), Some(Expr::Ident { .. })),
                "{src}"
            );
        }

        // Last in a body, it is the tail, and a `;` makes it a statement.
        let s = parse_script(&interner, "f(); if c { 1 } else { 2 }").unwrap();
        assert_eq!(s.stmts.len(), 1);
        assert!(matches!(s.tail.as_deref(), Some(Expr::If { .. })));
        let s = parse_script(&interner, "f(); if c { 1 } else { 2 };").unwrap();
        assert_eq!(s.stmts.len(), 2);
        assert!(s.tail.is_none());

        // In a loop body, which has no tail, it is a statement.
        let s = parse_script(
            &interner,
            "for x in &mut v { if x.a == 1 { x.a = 2; } } v.len()",
        )
        .unwrap();
        let Stmt::For { body, .. } = &s.stmts[0] else {
            panic!("expected a for");
        };
        assert!(matches!(body.as_slice(), [Stmt::Expr(Expr::If { .. })]));

        // No operator continues it: what follows begins the next statement.
        let s = parse_script(&interner, "if c { f(); } -x").unwrap();
        assert!(matches!(
            s.tail.as_deref(),
            Some(Expr::UnaryOp {
                op: UnaryOp::Neg,
                ..
            })
        ));
        let s = parse_script(&interner, "if c { f(); } *r = 1;").unwrap();
        assert!(matches!(&s.stmts[1], Stmt::DerefStore { .. }));
        let src = "if c { 1 } else { 2 } + 1";
        let error = first_error(parse_script(&interner, src).unwrap_err());
        let at = src.find('+').unwrap();
        assert_eq!(error.span, Span::new(at, at + 1));
        assert!(matches!(error.kind, ParseErrorKind::UnexpectedToken { .. }));
        // A block that begins a statement is an operand.
        let s = parse_script(&interner, "{ [5, 6] }[1]").unwrap();
        assert!(s.stmts.is_empty());
        assert!(matches!(s.tail.as_deref(), Some(Expr::Index { .. })));
        // Anywhere else an `if` is an operand.
        assert!(matches!(
            parse_script(&interner, "(if c { 1 } else { 2 }) + 1")
                .unwrap()
                .tail
                .as_deref(),
            Some(Expr::BinaryOp { .. })
        ));
    }

    /// Every postfix follows a qualified call, `?` included: the qualified
    /// name is a primary, not a level above the postfixes (RFC-0038).
    #[test]
    fn a_postfix_follows_a_qualified_call() {
        let interner = Interner::new();
        let s = parse_script(&interner, "let n = i64::from_str(s)?; n").unwrap();
        let Stmt::LetBind { expr, .. } = &s.stmts[0] else {
            panic!("expected a let");
        };
        let Expr::Try { inner, .. } = expr else {
            panic!("expected `?` over the call, got {expr:?}");
        };
        let Expr::FuncCall { func, .. } = inner.as_ref() else {
            panic!("expected a call under `?`");
        };
        let Expr::Ident { name, .. } = func.as_ref() else {
            panic!("expected a qualified name as the callee");
        };
        assert_eq!(name.namespace.map(|ns| interner.resolve(ns)), Some("i64"));
        assert_eq!(interner.resolve(name.name), "from_str");

        for src in [
            "let n = (i64::from_str(s))?; n",
            "let n = E::f(s).g; n",
            "let n = E::f(s)[0]; n",
            "let n = E::f(s).g(1); n",
            "let n = E::Tag; n",
        ] {
            parse_script(&interner, src).unwrap();
        }
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
        assert!(
            matches!(&body[0], Stmt::LetBind { binder, .. } if interner.resolve(binder.name) == "j")
        );
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

    // -- Store -------------------------------------------------------

    /// The left of `=` is a place: a bare name is an `Assign`, a `*r` a
    /// `DerefStore`, and every path of field and index steps from a name, a
    /// `$parameter` or an `@context` is a `Store` of that place.
    #[test]
    fn an_assignment_target_is_a_place() {
        fn shape(place: &Place, interner: &Interner) -> String {
            match place {
                Place::Field { object, field, .. } => {
                    format!("{}.{}", shape(object, interner), interner.resolve(*field))
                }
                Place::Base(PlaceBase::Root {
                    root: Root::Local(name),
                    ..
                }) => interner.resolve(*name).to_string(),
                Place::Base(PlaceBase::Root {
                    root: Root::ExternParam(name),
                    ..
                }) => format!("${}", interner.resolve(*name)),
                Place::Base(PlaceBase::Root {
                    root: Root::Context(name),
                    ..
                }) => format!("@{}", interner.resolve(*name)),
                Place::Base(PlaceBase::Element { container, .. }) => {
                    format!("{}[]", container_shape(container.expr(), interner))
                }
            }
        }
        fn container_shape(expr: &Expr, interner: &Interner) -> String {
            match expr {
                Expr::Ident {
                    name,
                    ref_kind: RefKind::ExternParam,
                    ..
                } => format!("${}", interner.resolve(name.name)),
                Expr::Ident { name, .. } => interner.resolve(name.name).to_string(),
                Expr::ContextRef { name, .. } => format!("@{}", interner.resolve(name.name)),
                Expr::FieldAccess { object, field, .. } => {
                    format!(
                        "{}.{}",
                        container_shape(object, interner),
                        interner.resolve(*field)
                    )
                }
                Expr::Index { object, .. } => format!("{}[]", container_shape(object, interner)),
                other => format!("{other:?}"),
            }
        }
        let interner = Interner::new();
        let store = |src: &str| -> String {
            let s = parse_script(&interner, src).unwrap();
            let Stmt::Store { place, .. } = &s.stmts[0] else {
                panic!("expected a Store for {src}");
            };
            shape(place, &interner)
        };
        assert_eq!(store("@count = @count + 1; @count"), "@count");
        assert_eq!(store("@a.x.y = 0;"), "@a.x.y");
        assert_eq!(store("v[0] = 5;"), "v[]");
        assert_eq!(store("v[0].f = 5;"), "v[].f");
        assert_eq!(store("o.g[0] = 7;"), "o.g[]");
        assert_eq!(store("o.f[i].g = 9;"), "o.f[].g");
        assert_eq!(store("$p.f = 1;"), "$p.f");

        let s = parse_script(&interner, "x = 1; *r = 2;").unwrap();
        assert!(matches!(&s.stmts[0], Stmt::Assign { name, .. } if interner.resolve(*name) == "x"));
        assert!(matches!(&s.stmts[1], Stmt::DerefStore { .. }));

        for src in [
            "f(x) = 1;",
            "1 = 2;",
            "(o).f = 1;",
            "E::A = 1;",
            "f(x)[0] = 1;",
        ] {
            assert_eq!(
                first_error(parse_script(&interner, src).unwrap_err()).kind,
                ParseErrorKind::InvalidAssignTarget,
                "{src}"
            );
        }
    }

    // -- Variant (Option) --------------------------------------------

    /// `Some`, `None` and a payload pattern read the same inside a
    /// template as in a script: the tag's content is the one expression
    /// grammar, and an arm line's head is the one pattern grammar.
    #[test]
    fn the_option_forms_read_the_same_in_a_template() {
        let t = template("{{ Some(42) | to_string }}");
        assert!(matches!(
            appended(&t.body[0]),
            Expr::Pipe { .. } | Expr::FuncCall { .. }
        ));

        let t = template("% match $o\n% Some(x) =>\n{{ x }}\n% None =>\nnothing\n% end\n");
        let Stmt::Expr(Expr::Match { arms, .. }) = &t.body[0] else {
            panic!("{:?}", t.body[0]);
        };
        assert!(matches!(
            &arms[0].pattern,
            Pattern::Variant {
                payload: Some(_),
                ..
            }
        ));
        assert!(matches!(
            &arms[1].pattern,
            Pattern::Variant { payload: None, .. }
        ));
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
            parse_template(&interner, source)
                .unwrap_or_else(|e| panic!("{source}: {:?}", e.errors));
        }
    }

    #[test]
    fn a_binder_spans_its_name_alone() {
        let interner = Interner::new();
        let source = "let a = 1; let b; for c in 0..2 { } let f = |d| -> d; a";
        let script = parse_script(&interner, source).expect(source);
        let [
            Stmt::LetBind { binder: a, .. },
            Stmt::LetUninit { binder: b, .. },
            Stmt::For { binder: c, .. },
            Stmt::LetBind {
                binder: f,
                expr: Expr::Lambda { params, .. },
                ..
            },
        ] = script.stmts.as_slice()
        else {
            panic!("unexpected statements: {:?}", script.stmts);
        };
        let [d] = params.as_slice() else {
            panic!("one parameter: {params:?}");
        };
        for (binder, name) in [(a, "a"), (b, "b"), (c, "c"), (f, "f"), (d, "d")] {
            assert_eq!(interner.resolve(binder.name), name);
            assert_eq!(&source[binder.span.start..binder.span.end], name);
        }

        let source = "% let t = 1\n% let u\n% for v in 0..2\n% end\n";
        let template = parse_template(&interner, source).expect(source);
        let [
            Stmt::LetBind { binder: t, .. },
            Stmt::LetUninit { binder: u, .. },
            Stmt::For { binder: v, .. },
        ] = template.body.as_slice()
        else {
            panic!("unexpected statements: {:?}", template.body);
        };
        for (binder, name) in [(t, "t"), (u, "u"), (v, "v")] {
            assert_eq!(interner.resolve(binder.name), name);
            assert_eq!(&source[binder.span.start..binder.span.end], name);
        }
    }

    mod recovery {
        use super::*;

        fn recovered_script(source: &str) -> Recovered<Script<ErrorNode>> {
            let interner = Interner::new();
            parse_script(&interner, source).expect_err(source)
        }

        fn recovered_template(source: &str) -> Recovered<Template<ErrorNode>> {
            let interner = Interner::new();
            parse_template(&interner, source).expect_err(source)
        }

        fn error_span(stmt: &Stmt<ErrorNode>) -> Span {
            let Stmt::Error(node) = stmt else {
                panic!("expected an error statement, got {stmt:?}");
            };
            node.span
        }

        fn error_spans(recovered: &Recovered<impl std::fmt::Debug>) -> Vec<Span> {
            recovered.errors.iter().map(|error| error.span).collect()
        }

        #[test]
        fn every_broken_statement_is_reported_and_the_rest_parses() {
            let source = "let a = 1;\nlet = 2;\nlet b = a;\nfoo(;\nb";
            let recovered = recovered_script(source);
            assert_eq!(
                error_spans(&recovered),
                vec![Span::new(15, 16), Span::new(35, 36)]
            );
            assert_eq!(
                recovered.errors[0].kind.to_string(),
                "expected a name, found `=`"
            );
            let [a, first, b, second] = recovered.tree.stmts.as_slice() else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert!(matches!(a, Stmt::LetBind { .. }));
            assert_eq!(
                &source[error_span(first).start..error_span(first).end],
                "let = 2"
            );
            assert!(matches!(b, Stmt::LetBind { .. }));
            assert_eq!(
                &source[error_span(second).start..error_span(second).end],
                "foo("
            );
            assert!(matches!(
                recovered.tree.tail.as_deref(),
                Some(Expr::Ident { .. })
            ));
        }

        #[test]
        fn a_let_whose_value_ends_at_the_end_of_the_source_keeps_its_binding() {
            let source = "let y = x.";
            let recovered = recovered_script(source);
            assert_eq!(recovered.errors.len(), 1);
            assert_eq!(recovered.errors[0].kind, ParseErrorKind::UnexpectedEof);
            let [
                Stmt::LetBind {
                    expr: Expr::Error(value),
                    span,
                    ..
                },
            ] = recovered.tree.stmts.as_slice()
            else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert_eq!(&source[value.span.start..value.span.end], "x.");
            assert_eq!(*span, Span::new(0, source.len()));
            assert!(recovered.tree.tail.is_none());
        }

        #[test]
        fn a_broken_let_value_leaves_the_next_line_parsed() {
            let source = "let y = x.\nlet z = 1;";
            let recovered = recovered_script(source);
            assert_eq!(error_spans(&recovered), vec![Span::new(11, 14)]);
            let [
                Stmt::LetBind {
                    expr: Expr::Error(value),
                    ..
                },
                Stmt::LetBind {
                    expr: Expr::Literal { .. },
                    ..
                },
            ] = recovered.tree.stmts.as_slice()
            else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert_eq!(&source[value.span.start..value.span.end], "x.");

            let source = "let y = x.;\nfoo(y);";
            let recovered = recovered_script(source);
            assert_eq!(error_spans(&recovered), vec![Span::new(10, 11)]);
            let [
                Stmt::LetBind {
                    expr: Expr::Error(value),
                    span,
                    ..
                },
                Stmt::Expr(Expr::FuncCall { .. }),
            ] = recovered.tree.stmts.as_slice()
            else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert_eq!(&source[value.span.start..value.span.end], "x.");
            assert_eq!(&source[span.start..span.end], "let y = x.;");
        }

        #[test]
        fn a_trailing_dot_reads_the_next_line() {
            let interner = Interner::new();
            let script = parse_script(&interner, "let y = x.\nfoo(y);").expect("a method call");
            assert!(matches!(
                &script.stmts[0],
                Stmt::LetBind {
                    expr: Expr::MethodCall { .. },
                    ..
                }
            ));
        }

        #[test]
        fn a_call_the_source_ends_inside_is_the_error_tail() {
            let source = "f(x.";
            let recovered = recovered_script(source);
            assert_eq!(recovered.errors.len(), 1);
            assert!(recovered.tree.stmts.is_empty());
            let Some(Expr::Error(tail)) = recovered.tree.tail.as_deref() else {
                panic!("{:?}", recovered.tree.tail);
            };
            assert_eq!(tail.span, Span::new(0, source.len()));
        }

        #[test]
        fn an_argument_that_does_not_parse_leaves_the_call() {
            let source = "f(x., y)";
            let recovered = recovered_script(source);
            assert_eq!(error_spans(&recovered), vec![Span::new(4, 5)]);
            let Some(Expr::FuncCall { args, .. }) = recovered.tree.tail.as_deref() else {
                panic!("{:?}", recovered.tree.tail);
            };
            let [Expr::Error(first), Expr::Ident { .. }] = args.as_slice() else {
                panic!("{args:?}");
            };
            assert_eq!(&source[first.span.start..first.span.end], "x.");
        }

        #[test]
        fn a_missing_semicolon_is_reported_and_the_statement_kept() {
            let source = "let a = 1\nlet b = a;";
            let recovered = recovered_script(source);
            assert_eq!(error_spans(&recovered), vec![Span::new(10, 13)]);
            let [Stmt::LetBind { span, .. }, Stmt::LetBind { .. }] =
                recovered.tree.stmts.as_slice()
            else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert_eq!(&source[span.start..span.end], "let a = 1");
        }

        #[test]
        fn an_unreadable_character_is_reported_and_stepped_over() {
            let source = "let a = #;\nlet b = 2;";
            let recovered = recovered_script(source);
            assert_eq!(
                recovered.errors,
                vec![ParseError::new(
                    ParseErrorKind::UnexpectedCharacter('#'),
                    Span::new(8, 9)
                )]
            );
            let [
                Stmt::LetBind {
                    expr: Expr::Error(value),
                    ..
                },
                Stmt::LetBind { .. },
            ] = recovered.tree.stmts.as_slice()
            else {
                panic!("{:?}", recovered.tree.stmts);
            };
            assert_eq!(value.span, Span::new(8, 9));

            let recovered = recovered_script("let a = # ^;");
            assert_eq!(
                recovered
                    .errors
                    .iter()
                    .map(|error| error.kind.clone())
                    .collect::<Vec<_>>(),
                vec![
                    ParseErrorKind::UnexpectedCharacter('#'),
                    ParseErrorKind::UnexpectedCharacter('^'),
                ]
            );
        }

        #[test]
        fn a_percent_line_that_does_not_parse_is_one_error_statement() {
            let source = "a\n% let = 1\nb {{ x }}\n";
            let recovered = recovered_template(source);
            assert_eq!(error_spans(&recovered), vec![Span::new(8, 9)]);
            assert_eq!(error_span(&recovered.tree.body[1]), Span::new(3, 11));
            assert!(matches!(
                appended(&recovered.tree.body[3]),
                Expr::Ident { .. }
            ));
        }

        #[test]
        fn a_tag_that_does_not_parse_is_one_error_expression() {
            let source = "a {{ x + }} b\n";
            let recovered = recovered_template(source);
            assert_eq!(recovered.errors.len(), 1);
            let Stmt::Append {
                expr: Expr::Error(node),
                span,
                ..
            } = &recovered.tree.body[1]
            else {
                panic!("{:?}", recovered.tree.body);
            };
            assert_eq!(&source[node.span.start..node.span.end], "x +");
            assert_eq!(&source[span.start..span.end], "{{ x + }}");
        }

        #[test]
        fn a_tag_no_brace_pair_closes_runs_to_its_line_end() {
            let source = "a {{ name\nb\n";
            let recovered = recovered_template(source);
            assert_eq!(
                recovered.errors,
                vec![ParseError::new(
                    ParseErrorKind::UnclosedTag,
                    Span::new(2, 9)
                )]
            );
            assert!(matches!(
                appended(&recovered.tree.body[1]),
                Expr::Ident { .. }
            ));
            let Stmt::Append {
                expr:
                    Expr::Literal {
                        value: Literal::String(text),
                        ..
                    },
                ..
            } = recovered.tree.body.last().expect("the second line")
            else {
                panic!("{:?}", recovered.tree.body);
            };
            assert_eq!(text, "b\n");
        }

        #[test]
        fn a_block_no_end_closes_is_closed_at_the_end_of_the_source() {
            let source = "% for x in $xs\n{{ x }}\n";
            let recovered = recovered_template(source);
            assert_eq!(
                recovered.errors,
                vec![ParseError::new(
                    ParseErrorKind::UnclosedBlock,
                    Span::new(1, 14)
                )]
            );
            let [Stmt::For { body, span, .. }] = recovered.tree.body.as_slice() else {
                panic!("{:?}", recovered.tree.body);
            };
            assert_eq!(body.len(), 2);
            assert_eq!(*span, Span::new(1, source.len()));
        }

        #[test]
        fn errors_are_in_source_order() {
            let recovered = recovered_template("% for x in $xs\n% let = 1\n");
            let spans = error_spans(&recovered);
            assert_eq!(spans.len(), 2);
            assert_eq!(spans[0], Span::new(1, 14));
            assert!(spans.windows(2).all(|pair| pair[0].start <= pair[1].start));
        }

        #[test]
        fn a_misplaced_template_line_is_an_error_statement() {
            for (source, kind) in [
                ("% end\n", ParseErrorKind::UnmatchedEnd),
                ("% else\n", ParseErrorKind::ElseOutsideIf),
                ("% 1 =>\n", ParseErrorKind::ArmOutsideMatch),
            ] {
                let recovered = recovered_template(source);
                assert_eq!(recovered.errors.len(), 1, "{source}");
                assert_eq!(recovered.errors[0].kind, kind, "{source}");
                assert!(
                    matches!(recovered.tree.body.as_slice(), [Stmt::Error(_)]),
                    "{source}"
                );
            }
        }
    }
}
