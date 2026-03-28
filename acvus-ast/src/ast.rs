use acvus_utils::{Astr, QualifiedRef};

use crate::span::Span;

acvus_utils::declare_local_id!(pub AstId);

impl AstId {
    pub fn alloc() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// A parsed script (standalone expressions with semicolons).
#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub id: AstId,
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

/// A statement in a script.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Bind {
        id: AstId,
        name: Astr,
        expr: Expr,
        span: Span,
    },
    ContextStore {
        id: AstId,
        name: QualifiedRef,
        expr: Expr,
        span: Span,
    },
    Expr(Expr),
    /// Match-bind (if-let): `pattern = source { body };`
    MatchBind {
        id: AstId,
        pattern: Pattern,
        source: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    /// Iteration (for): `pattern : source { body };`
    Iterate {
        id: AstId,
        pattern: Pattern,
        source: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
}

/// A parsed template.
#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    pub id: AstId,
    pub body: Vec<Node>,
    pub span: Span,
}

/// A node in the template body.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Literal text outside `{{ }}`.
    Text { id: AstId, value: String, span: Span },
    /// A comment `{{-- ... --}}`.
    Comment { id: AstId, value: String, span: Span },
    /// An inline expression `{{ expr }}` with no binding.
    InlineExpr { id: AstId, expr: Expr, span: Span },
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
    pub id: AstId,
    pub source: Expr,
    pub arms: Vec<MatchArm>,
    pub catch_all: Option<CatchAll>,
    pub indent: Option<IndentModifier>,
    pub span: Span,
}

/// A single arm in a match block.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub id: AstId,
    pub pattern: Pattern,
    pub body: Vec<Node>,
    pub tag_span: Span,
}

/// An iteration block with a single irrefutable pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct IterBlock {
    pub id: AstId,
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
    pub id: AstId,
    pub body: Vec<Node>,
    pub tag_span: Span,
}

/// An expression in the template language.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A reference: `name`, `$name`, or `@name`.
    Ident {
        id: AstId,
        name: Astr,
        ref_kind: RefKind,
        span: Span,
    },
    /// A literal value.
    Literal { id: AstId, value: Literal, span: Span },
    /// A binary operation: `a + b`.
    BinaryOp {
        id: AstId,
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
        span: Span,
    },
    /// A unary operation: `-x`, `!x`.
    UnaryOp {
        id: AstId,
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },
    /// Field access: `a.b`.
    FieldAccess {
        id: AstId,
        object: Box<Expr>,
        field: Astr,
        span: Span,
    },
    /// Function call: `f(args)`.
    FuncCall {
        id: AstId,
        func: Box<Expr>,
        args: Vec<Expr>,
        span: Span,
    },
    /// Pipe: `expr | func`.
    Pipe {
        id: AstId,
        left: Box<Expr>,
        right: Box<Expr>,
        span: Span,
    },
    /// Lambda: `|x| -> expr` or `|x, y| -> expr`.
    Lambda {
        id: AstId,
        params: Vec<LambdaParam>,
        body: Box<Expr>,
        span: Span,
    },
    /// Parenthesized expression: `(expr)`.
    Paren { id: AstId, inner: Box<Expr>, span: Span },
    /// A list: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// `rest` is `Some` if `..` is present. `head` is before `..`, `tail` is after.
    /// If no `..`, all elements are in `head` and `tail` is empty.
    List {
        id: AstId,
        head: Vec<Expr>,
        rest: Option<Span>,
        tail: Vec<Expr>,
        span: Span,
    },
    /// A group used for lambda parameter lists: `(a, b)`.
    /// This is a temporary node that only appears as the LHS of `->`.
    Group { id: AstId, elements: Vec<Expr>, span: Span },
    /// An object literal: `{ field1, $field2, field3 }`.
    Object {
        id: AstId,
        fields: Vec<ObjectExprField>,
        span: Span,
    },
    /// A range: `0..10`, `0..=10`, `0=..10`.
    Range {
        id: AstId,
        start: Box<Expr>,
        end: Box<Expr>,
        kind: RangeKind,
        span: Span,
    },
    /// A tuple: `(a, b, c)` — 0 or 2+ elements.
    /// Elements can be expressions or wildcards `_`.
    Tuple {
        id: AstId,
        elements: Vec<TupleElem>,
        span: Span,
    },
    /// A block expression: `{ stmt; stmt; expr }`.
    Block {
        id: AstId,
        stmts: Vec<Stmt>,
        tail: Box<Expr>,
        span: Span,
    },
    /// A context reference: `@name`.
    ContextRef { id: AstId, name: QualifiedRef, span: Span },
    /// A variant constructor: `Some(expr)`, `None`, or `Color::Red`.
    Variant {
        id: AstId,
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
    pub fn id(&self) -> AstId {
        match self {
            Expr::Ident { id, .. }
            | Expr::Literal { id, .. }
            | Expr::BinaryOp { id, .. }
            | Expr::UnaryOp { id, .. }
            | Expr::FieldAccess { id, .. }
            | Expr::FuncCall { id, .. }
            | Expr::Pipe { id, .. }
            | Expr::Lambda { id, .. }
            | Expr::Paren { id, .. }
            | Expr::List { id, .. }
            | Expr::Group { id, .. }
            | Expr::Object { id, .. }
            | Expr::Range { id, .. }
            | Expr::Tuple { id, .. }
            | Expr::ContextRef { id, .. }
            | Expr::Variant { id, .. }
            | Expr::Block { id, .. } => *id,
        }
    }

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
            | Expr::ContextRef { span, .. }
            | Expr::Variant { span, .. }
            | Expr::Block { span, .. } => *span,
        }
    }
}

/// A lambda parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaParam {
    pub id: AstId,
    pub name: Astr,
    pub span: Span,
}

/// A field in an object expression.
/// Shorthand `{ name }` → key="name", value=Ident("name", Value).
/// Shorthand `{ $name }` → key="name", value=Ident("name", Variable).
/// Shorthand `{ @name }` → key="name", value=Ident("name", Context).
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectExprField {
    pub id: AstId,
    pub key: Astr,
    pub value: Expr,
    pub span: Span,
}

/// A pattern used on the LHS of `=` in a match block.
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// A binding that captures a value: `item` or `$name`.
    Binding {
        id: AstId,
        name: Astr,
        ref_kind: RefKind,
        span: Span,
    },
    /// A context binding: `@name`.
    ContextBind { id: AstId, name: QualifiedRef, span: Span },
    /// A literal pattern that filters: `true`, `"admin"`, `42`.
    Literal { id: AstId, value: Literal, span: Span },
    /// A list pattern: `[a, b, c]`, `[a, b, ..]`, `[.., a, b]`, `[a, .., b]`.
    /// Same structure as `Expr::List`: `head` before `..`, `tail` after.
    List {
        id: AstId,
        head: Vec<Pattern>,
        rest: Option<Span>,
        tail: Vec<Pattern>,
        span: Span,
    },
    /// An object pattern: `{ name, $value, status: "active" }`.
    Object {
        id: AstId,
        fields: Vec<ObjectPatternField>,
        span: Span,
    },
    /// A range pattern: `0..10`, `0..=10`, `0=..10`.
    Range {
        id: AstId,
        start: Box<Pattern>,
        end: Box<Pattern>,
        kind: RangeKind,
        span: Span,
    },
    /// A tuple pattern: `(a, b, c)`.
    Tuple {
        id: AstId,
        elements: Vec<TuplePatternElem>,
        span: Span,
    },
    /// A variant pattern: `Some(inner)`, `None`, or `Color::Red`.
    Variant {
        id: AstId,
        enum_name: Option<Astr>,
        tag: Astr,
        payload: Option<Box<Pattern>>,
        span: Span,
    },
}

impl Pattern {
    pub fn id(&self) -> AstId {
        match self {
            Pattern::Binding { id, .. }
            | Pattern::ContextBind { id, .. }
            | Pattern::Literal { id, .. }
            | Pattern::List { id, .. }
            | Pattern::Object { id, .. }
            | Pattern::Range { id, .. }
            | Pattern::Tuple { id, .. }
            | Pattern::Variant { id, .. } => *id,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Pattern::Binding { span, .. }
            | Pattern::ContextBind { span, .. }
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
    pub id: AstId,
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
    /// An extern parameter: `$x` (immutable, externally injected).
    ExternParam,
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
pub fn extract_script_context_refs(script: &Script) -> rustc_hash::FxHashSet<QualifiedRef> {
    let mut refs = rustc_hash::FxHashSet::default();
    walk_stmts(&script.stmts, &mut refs);
    if let Some(tail) = &script.tail {
        walk_expr(tail, &mut refs);
    }
    refs
}

fn walk_stmts(stmts: &[Stmt], refs: &mut rustc_hash::FxHashSet<QualifiedRef>) {
    for stmt in stmts {
        match stmt {
            Stmt::Bind { expr, .. } => walk_expr(expr, refs),
            Stmt::ContextStore { name, expr, .. } => {
                refs.insert(*name);
                walk_expr(expr, refs);
            }
            Stmt::Expr(expr) => walk_expr(expr, refs),
            Stmt::MatchBind {
                pattern,
                source,
                body,
                ..
            } => {
                walk_pattern(pattern, refs);
                walk_expr(source, refs);
                walk_stmts(body, refs);
            }
            Stmt::Iterate {
                pattern,
                source,
                body,
                ..
            } => {
                walk_pattern(pattern, refs);
                walk_expr(source, refs);
                walk_stmts(body, refs);
            }
        }
    }
}

/// Extract all `@name` context references from a Template AST.
pub fn extract_template_context_refs(template: &Template) -> rustc_hash::FxHashSet<QualifiedRef> {
    let mut refs = rustc_hash::FxHashSet::default();
    walk_nodes(&template.body, &mut refs);
    refs
}

fn walk_nodes(nodes: &[Node], refs: &mut rustc_hash::FxHashSet<QualifiedRef>) {
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

fn walk_pattern(pattern: &Pattern, refs: &mut rustc_hash::FxHashSet<QualifiedRef>) {
    match pattern {
        Pattern::ContextBind { name, .. } => {
            refs.insert(*name);
        }
        Pattern::Binding { .. } | Pattern::Literal { .. } => {}
        Pattern::List { head, tail, .. } => {
            for p in head {
                walk_pattern(p, refs);
            }
            for p in tail {
                walk_pattern(p, refs);
            }
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
            if let Some(p) = payload {
                walk_pattern(p, refs);
            }
        }
    }
}

fn walk_expr(expr: &Expr, refs: &mut rustc_hash::FxHashSet<QualifiedRef>) {
    match expr {
        Expr::ContextRef { name, .. } => {
            refs.insert(*name);
        }
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::Variant { .. } => {}
        Expr::BinaryOp { left, right, .. }
        | Expr::Pipe { left, right, .. }
        | Expr::Range {
            start: left,
            end: right,
            ..
        } => {
            walk_expr(left, refs);
            walk_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. } | Expr::Paren { inner: operand, .. } => {
            walk_expr(operand, refs);
        }
        Expr::FieldAccess { object, .. } => walk_expr(object, refs),
        Expr::FuncCall { func, args, .. } => {
            walk_expr(func, refs);
            for arg in args {
                walk_expr(arg, refs);
            }
        }
        Expr::Lambda { body, .. } => walk_expr(body, refs),
        Expr::List { head, tail, .. } => {
            for e in head {
                walk_expr(e, refs);
            }
            for e in tail {
                walk_expr(e, refs);
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements {
                walk_expr(e, refs);
            }
        }
        Expr::Object { fields, .. } => {
            for f in fields {
                walk_expr(&f.value, refs);
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(expr) = e {
                    walk_expr(expr, refs);
                }
            }
        }
        Expr::Block { stmts, tail, .. } => {
            walk_stmts(stmts, refs);
            walk_expr(tail, refs);
        }
    }
}
