//! AST-level placeholder substitution.
//!
//! Replaces dummy ident nodes (`__acvus_ph_<name>__`) with provided Expr values.
//! Supports splice placeholders (`__acvus_splice_<name>__`) that expand a `Vec<Expr>`
//! into sequence contexts (list elements, function args, pipe chains, binary op chains, etc.).

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::ast::*;
use crate::span::Span;

/// A substitution value: either a single expression or a splice (multiple expressions).
#[derive(Debug, Clone)]
pub enum SubstValue {
    /// Replace placeholder with a single expression.
    Single(Expr),
    /// Splice multiple expressions into a sequence context.
    Splice(Vec<Expr>),
}

/// Substitute placeholder idents in a Script AST.
pub fn substitute_script(script: Script, subs: &FxHashMap<Astr, SubstValue>) -> Script {
    Script {
        id: AstId::alloc(),
        stmts: script
            .stmts
            .into_iter()
            .map(|s| sub_stmt(s, subs))
            .collect(),
        tail: script.tail.map(|e| Box::new(sub_expr(*e, subs))),
        span: script.span,
    }
}

/// Substitute placeholder idents in a Template AST.
pub fn substitute_template(template: Template, subs: &FxHashMap<Astr, SubstValue>) -> Template {
    Template {
        id: AstId::alloc(),
        body: template
            .body
            .into_iter()
            .map(|n| sub_node(n, subs))
            .collect(),
        span: template.span,
    }
}

/// Substitute an expression in a sequence context, potentially returning multiple expressions.
///
/// If the expression is a splice placeholder, returns the splice `Vec<Expr>`.
/// If it's a single placeholder, returns the replacement wrapped in a vec.
/// Otherwise, recursively substitutes and returns a single-element vec.
fn sub_expr_seq(expr: Expr, subs: &FxHashMap<Astr, SubstValue>) -> Vec<Expr> {
    match &expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } => match subs.get(name) {
            Some(SubstValue::Splice(exprs)) => exprs.clone(),
            Some(SubstValue::Single(e)) => vec![e.clone()],
            None => vec![expr],
        },
        _ => vec![sub_expr(expr, subs)],
    }
}

fn sub_expr(expr: Expr, subs: &FxHashMap<Astr, SubstValue>) -> Expr {
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } => match subs.get(&name) {
            Some(SubstValue::Single(replacement)) => replacement.clone(),
            Some(SubstValue::Splice(_)) => panic!(
                "splice placeholder in non-sequence context \
                 (compile-time validation should have caught this)"
            ),
            None => expr,
        },
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } => expr,

        // Binary chains: flatten same-op chain → splice → re-fold (left-associative).
        Expr::BinaryOp {
            left,
            op,
            right,
            span,
            ..
        } => {
            let parts = flatten_binop(*left, *right, op);
            let spliced: Vec<Expr> = parts
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect();
            fold_binop(spliced, op, span)
        }

        Expr::UnaryOp { op, operand, span, .. } => Expr::UnaryOp {
            id: AstId::alloc(),
            op,
            operand: Box::new(sub_expr(*operand, subs)),
            span,
        },
        Expr::FieldAccess {
            object,
            field,
            span,
            ..
        } => Expr::FieldAccess {
            id: AstId::alloc(),
            object: Box::new(sub_expr(*object, subs)),
            field,
            span,
        },

        // Function args: sequence context.
        Expr::FuncCall { func, args, span, .. } => Expr::FuncCall {
            id: AstId::alloc(),
            func: Box::new(sub_expr(*func, subs)),
            args: args
                .into_iter()
                .flat_map(|a| sub_expr_seq(a, subs))
                .collect(),
            span,
        },

        // Pipe chains: flatten → splice → re-fold (left-associative).
        Expr::Pipe { left, right, span, .. } => {
            let stages = flatten_pipe(*left, *right);
            let spliced: Vec<Expr> = stages
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect();
            fold_pipe(spliced, span)
        }

        Expr::Lambda { params, body, span, .. } => Expr::Lambda {
            id: AstId::alloc(),
            params,
            body: Box::new(sub_expr(*body, subs)),
            span,
        },
        Expr::Paren { inner, span, .. } => Expr::Paren {
            id: AstId::alloc(),
            inner: Box::new(sub_expr(*inner, subs)),
            span,
        },

        // List elements: sequence context (both head and tail).
        Expr::List {
            head,
            rest,
            tail,
            span,
            ..
        } => Expr::List {
            id: AstId::alloc(),
            head: head
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect(),
            rest,
            tail: tail
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect(),
            span,
        },

        // Group elements: sequence context.
        Expr::Group { elements, span, .. } => Expr::Group {
            id: AstId::alloc(),
            elements: elements
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect(),
            span,
        },

        Expr::Object { fields, span, .. } => Expr::Object {
            id: AstId::alloc(),
            fields: fields
                .into_iter()
                .map(|f| ObjectExprField {
                    id: AstId::alloc(),
                    key: f.key,
                    value: sub_expr(f.value, subs),
                    span: f.span,
                })
                .collect(),
            span,
        },
        Expr::Range {
            start,
            end,
            kind,
            span,
            ..
        } => Expr::Range {
            id: AstId::alloc(),
            start: Box::new(sub_expr(*start, subs)),
            end: Box::new(sub_expr(*end, subs)),
            kind,
            span,
        },

        // Tuple elements: sequence context.
        Expr::Tuple { elements, span, .. } => Expr::Tuple {
            id: AstId::alloc(),
            elements: elements
                .into_iter()
                .flat_map(|e| match e {
                    TupleElem::Expr(expr) => sub_expr_seq(expr, subs)
                        .into_iter()
                        .map(TupleElem::Expr)
                        .collect::<Vec<_>>(),
                    w @ TupleElem::Wildcard(_) => vec![w],
                })
                .collect(),
            span,
        },

        Expr::Block { stmts, tail, span, .. } => Expr::Block {
            id: AstId::alloc(),
            stmts: stmts.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            tail: Box::new(sub_expr(*tail, subs)),
            span,
        },
        Expr::Variant {
            enum_name,
            tag,
            payload,
            span,
            ..
        } => Expr::Variant {
            id: AstId::alloc(),
            enum_name,
            tag,
            payload: payload.map(|p| Box::new(sub_expr(*p, subs))),
            span,
        },
    }
}

// ── Flatten / fold helpers ───────────────────────────────────────────

/// Flatten a left-associative pipe chain into a sequence of stages.
///
/// `Pipe(Pipe(a, b), c)` → `[a, b, c]`
fn flatten_pipe(left: Expr, right: Expr) -> Vec<Expr> {
    let mut stages = match left {
        Expr::Pipe { left, right, .. } => flatten_pipe(*left, *right),
        other => vec![other],
    };
    stages.push(right);
    stages
}

/// Re-fold a sequence of stages into a left-associative pipe chain.
fn fold_pipe(stages: Vec<Expr>, span: Span) -> Expr {
    assert!(!stages.is_empty(), "splice produced empty pipe chain");
    stages
        .into_iter()
        .reduce(|left, right| Expr::Pipe {
            id: AstId::alloc(),
            left: Box::new(left),
            right: Box::new(right),
            span,
        })
        .unwrap()
}

/// Flatten a left-associative binary op chain (same operator) into a sequence of operands.
///
/// `Add(Add(a, b), c)` → `[a, b, c]`
///
/// Only flattens nodes with the same operator; different-op nodes are preserved as-is.
fn flatten_binop(left: Expr, right: Expr, target_op: BinOp) -> Vec<Expr> {
    let mut parts = match left {
        Expr::BinaryOp {
            left, op, right, ..
        } if op == target_op => flatten_binop(*left, *right, target_op),
        other => vec![other],
    };
    parts.push(right);
    parts
}

/// Re-fold a sequence of operands into a left-associative binary op chain.
fn fold_binop(parts: Vec<Expr>, op: BinOp, span: Span) -> Expr {
    assert!(
        !parts.is_empty(),
        "splice produced empty binary operation chain"
    );
    parts
        .into_iter()
        .reduce(|left, right| Expr::BinaryOp {
            id: AstId::alloc(),
            left: Box::new(left),
            op,
            right: Box::new(right),
            span,
        })
        .unwrap()
}

// ── Statement / node substitution ────────────────────────────────────

fn sub_stmt(stmt: Stmt, subs: &FxHashMap<Astr, SubstValue>) -> Stmt {
    match stmt {
        Stmt::Bind { name, expr, span, .. } => Stmt::Bind {
            id: AstId::alloc(),
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::ContextStore { name, expr, span, .. } => Stmt::ContextStore {
            id: AstId::alloc(),
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::Expr(expr) => Stmt::Expr(sub_expr(expr, subs)),
        Stmt::MatchBind {
            pattern,
            source,
            body,
            span,
            ..
        } => Stmt::MatchBind {
            id: AstId::alloc(),
            pattern,
            source: sub_expr(source, subs),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
        Stmt::Iterate {
            pattern,
            source,
            body,
            span,
            ..
        } => Stmt::Iterate {
            id: AstId::alloc(),
            pattern,
            source: sub_expr(source, subs),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
    }
}

fn sub_node(node: Node, subs: &FxHashMap<Astr, SubstValue>) -> Node {
    match node {
        Node::Text { .. } | Node::Comment { .. } => node,
        Node::InlineExpr { expr, span, .. } => Node::InlineExpr {
            id: AstId::alloc(),
            expr: sub_expr(expr, subs),
            span,
        },
        Node::MatchBlock(mb) => Node::MatchBlock(MatchBlock {
            id: AstId::alloc(),
            source: sub_expr(mb.source, subs),
            arms: mb
                .arms
                .into_iter()
                .map(|arm| MatchArm {
                    id: AstId::alloc(),
                    pattern: arm.pattern,
                    body: arm.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                    tag_span: arm.tag_span,
                })
                .collect(),
            catch_all: mb.catch_all.map(|ca| CatchAll {
                id: AstId::alloc(),
                body: ca.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                tag_span: ca.tag_span,
            }),
            indent: mb.indent,
            span: mb.span,
        }),
        Node::IterBlock(ib) => Node::IterBlock(IterBlock {
            id: AstId::alloc(),
            pattern: ib.pattern,
            source: sub_expr(ib.source, subs),
            body: ib.body.into_iter().map(|n| sub_node(n, subs)).collect(),
            catch_all: ib.catch_all.map(|ca| CatchAll {
                id: AstId::alloc(),
                body: ca.body.into_iter().map(|n| sub_node(n, subs)).collect(),
                tag_span: ca.tag_span,
            }),
            indent: ib.indent,
            span: ib.span,
        }),
    }
}

// ── Compile-time splice position validation ──────────────────────────
//
// Validates that splice placeholder idents only appear in sequence contexts
// (list elements, function args, tuple elements, pipe/binary op chains).
// Called from the proc macro at compile time.

/// Validate splice positions in a Script AST.
/// Returns `(name, span)` for each splice placeholder in an invalid (non-sequence) position.
pub fn validate_splice_positions_script(
    script: &Script,
    splice_names: &[Astr],
) -> Vec<(Astr, Span)> {
    let mut errors = Vec::new();
    for stmt in &script.stmts {
        validate_splice_stmt(stmt, splice_names, &mut errors);
    }
    if let Some(tail) = &script.tail {
        validate_splice_expr(tail, false, splice_names, &mut errors);
    }
    errors
}

/// Validate splice positions in a Template AST.
/// Returns `(name, span)` for each splice placeholder in an invalid (non-sequence) position.
pub fn validate_splice_positions_template(
    template: &Template,
    splice_names: &[Astr],
) -> Vec<(Astr, Span)> {
    let mut errors = Vec::new();
    for node in &template.body {
        validate_splice_node(node, splice_names, &mut errors);
    }
    errors
}

/// Validate an expression. `in_seq` indicates whether this expression is in a
/// sequence context where splice is allowed.
fn validate_splice_expr(
    expr: &Expr,
    in_seq: bool,
    splice_names: &[Astr],
    errors: &mut Vec<(Astr, Span)>,
) {
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            span,
            ..
        } => {
            if splice_names.contains(name) && !in_seq {
                errors.push((*name, *span));
            }
        }
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } => {}

        // Binary op chain: both sides are sequence contexts.
        Expr::BinaryOp { left, right, .. } => {
            validate_splice_expr(left, true, splice_names, errors);
            validate_splice_expr(right, true, splice_names, errors);
        }

        // Pipe chain: both sides are sequence contexts.
        Expr::Pipe { left, right, .. } => {
            validate_splice_expr(left, true, splice_names, errors);
            validate_splice_expr(right, true, splice_names, errors);
        }

        // Non-sequence contexts:
        Expr::UnaryOp { operand, .. } => {
            validate_splice_expr(operand, false, splice_names, errors);
        }
        Expr::FieldAccess { object, .. } => {
            validate_splice_expr(object, false, splice_names, errors);
        }
        Expr::FuncCall { func, args, .. } => {
            validate_splice_expr(func, false, splice_names, errors);
            for arg in args {
                validate_splice_expr(arg, true, splice_names, errors);
            }
        }
        Expr::Lambda { body, .. } => {
            validate_splice_expr(body, false, splice_names, errors);
        }
        Expr::Paren { inner, .. } => {
            validate_splice_expr(inner, false, splice_names, errors);
        }
        Expr::List { head, tail, .. } => {
            for e in head {
                validate_splice_expr(e, true, splice_names, errors);
            }
            for e in tail {
                validate_splice_expr(e, true, splice_names, errors);
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements {
                validate_splice_expr(e, true, splice_names, errors);
            }
        }
        Expr::Object { fields, .. } => {
            for f in fields {
                validate_splice_expr(&f.value, false, splice_names, errors);
            }
        }
        Expr::Range { start, end, .. } => {
            validate_splice_expr(start, false, splice_names, errors);
            validate_splice_expr(end, false, splice_names, errors);
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                match e {
                    TupleElem::Expr(expr) => {
                        validate_splice_expr(expr, true, splice_names, errors);
                    }
                    TupleElem::Wildcard(_) => {}
                }
            }
        }
        Expr::Block { stmts, tail, .. } => {
            for s in stmts {
                validate_splice_stmt(s, splice_names, errors);
            }
            validate_splice_expr(tail, false, splice_names, errors);
        }
        Expr::Variant { payload, .. } => {
            if let Some(p) = payload {
                validate_splice_expr(p, false, splice_names, errors);
            }
        }
    }
}

fn validate_splice_stmt(stmt: &Stmt, splice_names: &[Astr], errors: &mut Vec<(Astr, Span)>) {
    match stmt {
        Stmt::Bind { expr, .. } | Stmt::ContextStore { expr, .. } => {
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Stmt::Expr(expr) => {
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Stmt::MatchBind { source, body, .. } => {
            validate_splice_expr(source, false, splice_names, errors);
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
        }
        Stmt::Iterate { source, body, .. } => {
            validate_splice_expr(source, false, splice_names, errors);
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
        }
    }
}

fn validate_splice_node(node: &Node, splice_names: &[Astr], errors: &mut Vec<(Astr, Span)>) {
    match node {
        Node::Text { .. } | Node::Comment { .. } => {}
        Node::InlineExpr { expr, .. } => {
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Node::MatchBlock(mb) => {
            validate_splice_expr(&mb.source, false, splice_names, errors);
            for arm in &mb.arms {
                for n in &arm.body {
                    validate_splice_node(n, splice_names, errors);
                }
            }
            if let Some(ca) = &mb.catch_all {
                for n in &ca.body {
                    validate_splice_node(n, splice_names, errors);
                }
            }
        }
        Node::IterBlock(ib) => {
            validate_splice_expr(&ib.source, false, splice_names, errors);
            for n in &ib.body {
                validate_splice_node(n, splice_names, errors);
            }
            if let Some(ca) = &ib.catch_all {
                for n in &ca.body {
                    validate_splice_node(n, splice_names, errors);
                }
            }
        }
    }
}
