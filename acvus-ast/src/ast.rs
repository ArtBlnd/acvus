use acvus_utils::Astr;

use crate::span::Span;

/// A parsed script (standalone expressions with semicolons).
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

/// A statement in a script.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Bind { name: Astr, expr: Expr, span: Span },
    ContextStore { name: Astr, expr: Expr, span: Span },
    Expr(Expr),
}

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
    Text { value: String, span: Span },
    /// A comment `{{-- ... --}}`.
    Comment { value: String, span: Span },
    /// An inline expression `{{ expr }}` with no binding.
    InlineExpr { expr: Expr, span: Span },
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
        name: Astr,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal value.
    Literal { value: Literal, span: Span },
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
        field: Astr,
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
    Paren { inner: Box<Expr>, span: Span },
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
    Group { elements: Vec<Expr>, span: Span },
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
    /// A block expression: `{ stmt; stmt; expr }`.
    Block {
        stmts: Vec<Stmt>,
        tail: Box<Expr>,
        span: Span,
    },
    /// A variant constructor: `Some(expr)`, `None`, or `Color::Red`.
    Variant {
        enum_name: Option<Astr>,
        tag: Astr,
        payload: Option<Box<Expr>>,
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
            | Expr::Tuple { span, .. }
            | Expr::Variant { span, .. }
            | Expr::Block { span, .. } => *span,
        }
    }
}

/// A lambda parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaParam {
    pub name: Astr,
    pub span: Span,
}

/// A field in an object expression.
/// Shorthand `{ name }` → key="name", value=Ident("name", Value).
/// Shorthand `{ $name }` → key="name", value=Ident("name", Variable).
/// Shorthand `{ @name }` → key="name", value=Ident("name", Context).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectExprField {
    pub key: Astr,
    pub value: Expr,
    pub span: Span,
}

/// A pattern used on the LHS of `=` in a match block.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// A binding that captures a value: `item`, `$name`, or `@name`.
    Binding {
        name: Astr,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal pattern that filters: `true`, `"admin"`, `42`.
    Literal { value: Literal, span: Span },
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
    /// A variant pattern: `Some(inner)`, `None`, or `Color::Red`.
    Variant {
        enum_name: Option<Astr>,
        tag: Astr,
        payload: Option<Box<Pattern>>,
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
            | Pattern::Tuple { span, .. }
            | Pattern::Variant { span, .. } => *span,
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
    pub key: Astr,
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

// ── AST walk: context reference extraction ──────────────────────────

/// Extract all `@name` context references from a Script AST.
pub fn extract_script_context_refs(script: &Script) -> rustc_hash::FxHashSet<Astr> {
    let mut refs = rustc_hash::FxHashSet::default();
    for stmt in &script.stmts {
        match stmt {
            Stmt::Bind { expr, .. } => walk_expr(expr, &mut refs),
            Stmt::ContextStore { name, expr, .. } => {
                refs.insert(*name);
                walk_expr(expr, &mut refs);
            }
            Stmt::Expr(expr) => walk_expr(expr, &mut refs),
        }
    }
    if let Some(tail) = &script.tail {
        walk_expr(tail, &mut refs);
    }
    refs
}

/// Extract all `@name` context references from a Template AST.
pub fn extract_template_context_refs(template: &Template) -> rustc_hash::FxHashSet<Astr> {
    let mut refs = rustc_hash::FxHashSet::default();
    walk_nodes(&template.body, &mut refs);
    refs
}

fn walk_nodes(nodes: &[Node], refs: &mut rustc_hash::FxHashSet<Astr>) {
    for node in nodes {
        match node {
            Node::Text { .. } | Node::Comment { .. } => {}
            Node::InlineExpr { expr, .. } => walk_expr(expr, refs),
            Node::MatchBlock(mb) => {
                walk_expr(&mb.source, refs);
                for arm in &mb.arms {
                    walk_pattern(&arm.pattern, refs);
                    walk_nodes(&arm.body, refs);
                }
                if let Some(ca) = &mb.catch_all {
                    walk_nodes(&ca.body, refs);
                }
            }
            Node::IterBlock(ib) => {
                walk_expr(&ib.source, refs);
                walk_nodes(&ib.body, refs);
                if let Some(ca) = &ib.catch_all {
                    walk_nodes(&ca.body, refs);
                }
            }
        }
    }
}

fn walk_pattern(pattern: &Pattern, refs: &mut rustc_hash::FxHashSet<Astr>) {
    match pattern {
        Pattern::Binding { name, ref_kind, .. } => {
            match ref_kind {
                RefKind::Context => { refs.insert(*name); }
                RefKind::Variable | RefKind::Value => {}
            }
        }
        Pattern::Literal { .. } => {}
        Pattern::List { head, tail, .. } => {
            for p in head { walk_pattern(p, refs); }
            for p in tail { walk_pattern(p, refs); }
        }
        Pattern::Range { start, end, .. } => {
            walk_pattern(start, refs);
            walk_pattern(end, refs);
        }
        Pattern::Object { fields, .. } => {
            for f in fields {
                walk_pattern(&f.pattern, refs);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for e in elements {
                match e {
                    TuplePatternElem::Pattern(p) => walk_pattern(p, refs),
                    TuplePatternElem::Wildcard(_) => {}
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(p) = payload { walk_pattern(p, refs); }
        }
    }
}

fn walk_expr(expr: &Expr, refs: &mut rustc_hash::FxHashSet<Astr>) {
    match expr {
        Expr::Ident { name, ref_kind: RefKind::Context, .. } => {
            refs.insert(*name);
        }
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::Variant { .. } => {}
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. }
        | Expr::Range { start: left, end: right, .. } => {
            walk_expr(left, refs);
            walk_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. } | Expr::Paren { inner: operand, .. } => {
            walk_expr(operand, refs);
        }
        Expr::FieldAccess { object, .. } => walk_expr(object, refs),
        Expr::FuncCall { func, args, .. } => {
            walk_expr(func, refs);
            for arg in args { walk_expr(arg, refs); }
        }
        Expr::Lambda { body, .. } => walk_expr(body, refs),
        Expr::List { head, tail, .. } => {
            for e in head { walk_expr(e, refs); }
            for e in tail { walk_expr(e, refs); }
        }
        Expr::Group { elements, .. } => {
            for e in elements { walk_expr(e, refs); }
        }
        Expr::Object { fields, .. } => {
            for f in fields { walk_expr(&f.value, refs); }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(expr) = e { walk_expr(expr, refs); }
            }
        }
        Expr::Block { stmts, tail, .. } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Bind { expr, .. } => walk_expr(expr, refs),
                    Stmt::ContextStore { name, expr, .. } => {
                        refs.insert(*name);
                        walk_expr(expr, refs);
                    }
                    Stmt::Expr(expr) => walk_expr(expr, refs),
                }
            }
            walk_expr(tail, refs);
        }
    }
}
