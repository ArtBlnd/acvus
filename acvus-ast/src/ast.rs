use crate::span::Span;

/// A parsed template.
#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    pub body: Vec<Node>,
    pub span: Span,
}

/// A node in the template body.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Literal text outside `{{ }}`.
    Text {
        value: String,
        span: Span,
    },
    /// A comment `{{-- ... --}}`.
    Comment {
        value: String,
        span: Span,
    },
    /// An inline expression `{{ expr }}` with no binding.
    InlineExpr {
        expr: Expr,
        span: Span,
    },
    /// A match block `{{ pattern = expr }} ... {{/}}`.
    /// Variable writes (`{{ $name = expr }}`) are also represented as a
    /// MatchBlock with a single arm whose pattern is
    /// `Pattern::Binding { ref_kind: Variable, .. }` and an empty body.
    MatchBlock(MatchBlock),
    /// An iteration block `{{ pattern in expr }} ... {{/}}`.
    IterBlock(IterBlock),
}

/// A match block with one or more arms and optional catch-all.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchBlock {
    pub source: Expr,
    pub arms: Vec<MatchArm>,
    pub catch_all: Option<CatchAll>,
    pub indent: Option<IndentModifier>,
    pub span: Span,
}

/// A single arm in a match block.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Vec<Node>,
    pub tag_span: Span,
}

/// An iteration block with a single irrefutable pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct IterBlock {
    pub pattern: Pattern,
    pub source: Expr,
    pub body: Vec<Node>,
    pub catch_all: Option<CatchAll>,
    pub indent: Option<IndentModifier>,
    pub span: Span,
}

/// The catch-all `{{_}}` arm.
#[derive(Debug, Clone, PartialEq)]
pub struct CatchAll {
    pub body: Vec<Node>,
    pub tag_span: Span,
}

/// An expression in the template language.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A reference: `name`, `$name`, or `@name`.
    Ident {
        name: String,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal value.
    Literal {
        value: Literal,
        span: Span,
    },
    /// A binary operation: `a + b`.
    BinaryOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
        span: Span,
    },
    /// A unary operation: `-x`, `!x`.
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// Field access: `a.b`.
    FieldAccess {
        object: Box<Expr>,
        field: String,
        span: Span,
    },
    /// Function call: `f(args)`.
    FuncCall {
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// Pipe: `expr | func`.
    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// Lambda: `x -> expr` or `(x, y) -> expr`.
    Lambda {
        params: Vec<LambdaParam>,
        body: Box<Expr>,
        span: Span,
    },
    /// Parenthesized expression: `(expr)`.
    Paren {
        inner: Box<Expr>,
        span: Span,
    },
    /// A list: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// `rest` is `Some` if `..` is present. `head` is before `..`, `tail` is after.
    /// If no `..`, all elements are in `head` and `tail` is empty.
    List {
        head: Vec<Expr>,
        rest: Option<Span>,
        tail: Vec<Expr>,
        span: Span,
    },
    /// A group used for lambda parameter lists: `(a, b)`.
    /// This is a temporary node that only appears as the LHS of `->`.
    Group {
        elements: Vec<Expr>,
        span: Span,
    },
    /// An object literal: `{ field1, $field2, field3 }`.
    Object {
        fields: Vec<ObjectExprField>,
        span: Span,
    },
    /// A range: `0..10`, `0..=10`, `0=..10`.
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        kind: RangeKind,
        span: Span,
    },
    /// A tuple: `(a, b, c)` — 0 or 2+ elements.
    /// Elements can be expressions or wildcards `_`.
    Tuple {
        elements: Vec<TupleElem>,
        span: Span,
    },
}

/// An element in a tuple expression: either a real expression or a wildcard `_`.
#[derive(Debug, Clone, PartialEq)]
pub enum TupleElem {
    Expr(Expr),
    Wildcard(Span),
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Ident { span, .. }
            | Expr::Literal { span, .. }
            | Expr::BinaryOp { span, .. }
            | Expr::UnaryOp { span, .. }
            | Expr::FieldAccess { span, .. }
            | Expr::FuncCall { span, .. }
            | Expr::Pipe { span, .. }
            | Expr::Lambda { span, .. }
            | Expr::Paren { span, .. }
            | Expr::List { span, .. }
            | Expr::Group { span, .. }
            | Expr::Object { span, .. }
            | Expr::Range { span, .. }
            | Expr::Tuple { span, .. } => *span,
        }
    }
}

/// A lambda parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaParam {
    pub name: String,
    pub span: Span,
}

/// A field in an object expression.
/// Shorthand `{ name }` → key="name", value=Ident("name", Value).
/// Shorthand `{ $name }` → key="name", value=Ident("name", Variable).
/// Shorthand `{ @name }` → key="name", value=Ident("name", Context).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectExprField {
    pub key: String,
    pub value: Expr,
    pub span: Span,
}

/// A pattern used on the LHS of `=` in a match block.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// A binding that captures a value: `item`, `$name`, or `@name`.
    Binding {
        name: String,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal pattern that filters: `true`, `"admin"`, `42`.
    Literal {
        value: Literal,
        span: Span,
    },
    /// A list pattern: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// Same structure as `Expr::List`: `head` before `..`, `tail` after.
    List {
        head: Vec<Pattern>,
        rest: Option<Span>,
        tail: Vec<Pattern>,
        span: Span,
    },
    /// An object pattern: `{ name, $value, status: "active" }`.
    Object {
        fields: Vec<ObjectPatternField>,
        span: Span,
    },
    /// A range pattern: `0..10`, `0..=10`, `0=..10`.
    Range {
        start: Box<Pattern>,
        end: Box<Pattern>,
        kind: RangeKind,
        span: Span,
    },
    /// A tuple pattern: `(a, b, c)`.
    Tuple {
        elements: Vec<TuplePatternElem>,
        span: Span,
    },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Binding { span, .. }
            | Pattern::Literal { span, .. }
            | Pattern::List { span, .. }
            | Pattern::Object { span, .. }
            | Pattern::Range { span, .. }
            | Pattern::Tuple { span, .. } => *span,
        }
    }
}

/// An element in a tuple pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum TuplePatternElem {
    /// A sub-pattern.
    Pattern(Pattern),
    /// A wildcard `_` that ignores the element.
    Wildcard(Span),
}

/// A field in an object pattern: `{ key: pattern }` or shorthand `{ name }` / `{ $name }`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectPatternField {
    pub key: String,
    pub pattern: Pattern,
    pub span: Span,
}

/// An indent modifier on a close block `{{/+N}}` or `{{/-N}}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndentModifier {
    Increase(u32),
    Decrease(u32),
}

/// The kind of range: `..`, `..=`, or `=..`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeKind {
    /// `0..10` — exclusive end: [start, end)
    Exclusive,
    /// `0..=10` — inclusive end: [start, end]
    InclusiveEnd,
    /// `0=..10` — exclusive start: (start, end]
    ExclusiveStart,
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    And,
    Or,
    Xor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Mod,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

/// The kind of reference for an identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    /// A bare name: `x`.
    Value,
    /// A context reference: `@x` (read-only, externally injected).
    Context,
    /// A variable reference: `$x` (mutable local).
    Variable,
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Byte(u8),
    List(Vec<Literal>),
}
