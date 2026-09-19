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
        } => match subs.get(&name.name) {
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
        } => match subs.get(&name.name) {
            Some(SubstValue::Single(replacement)) => replacement.clone(),
            Some(SubstValue::Splice(_)) => panic!(
                "splice placeholder in non-sequence context \
                 (compile-time validation should have caught this)"
            ),
            None => expr,
        },
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } => expr,

        // Binary chains: flatten same-op chain -> splice -> re-fold (left-associative).
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

        Expr::UnaryOp {
            op, operand, span, ..
        } => Expr::UnaryOp {
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

        Expr::Index {
            object,
            index,
            span,
            ..
        } => Expr::Index {
            id: AstId::alloc(),
            callee_id: AstId::alloc(),
            object: Box::new(sub_expr(*object, subs)),
            index: Box::new(sub_expr(*index, subs)),
            span,
        },

        Expr::MethodCall {
            receiver,
            name,
            args,
            span,
            ..
        } => Expr::MethodCall {
            id: AstId::alloc(),
            callee_id: AstId::alloc(),
            receiver: Box::new(sub_expr(*receiver, subs)),
            name,
            args: args
                .into_iter()
                .flat_map(|a| sub_expr_seq(a, subs))
                .collect(),
            span,
        },

        // Function args: sequence context.
        Expr::FuncCall {
            func, args, span, ..
        } => Expr::FuncCall {
            id: AstId::alloc(),
            func: Box::new(sub_expr(*func, subs)),
            args: args
                .into_iter()
                .flat_map(|a| sub_expr_seq(a, subs))
                .collect(),
            span,
        },

        // Pipe chains: flatten -> splice -> re-fold (left-associative).
        Expr::Pipe {
            left, right, span, ..
        } => {
            let stages = flatten_pipe(*left, *right);
            let spliced: Vec<Expr> = stages
                .into_iter()
                .flat_map(|e| sub_expr_seq(e, subs))
                .collect();
            fold_pipe(spliced, span)
        }

        Expr::Lambda {
            params, body, span, ..
        } => Expr::Lambda {
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
        Expr::Try { inner, span, .. } => Expr::Try {
            id: AstId::alloc(),
            inner: Box::new(sub_expr(*inner, subs)),
            span,
        },
        Expr::Cast {
            expr,
            target,
            target_span,
            span,
            ..
        } => Expr::Cast {
            id: AstId::alloc(),
            expr: Box::new(sub_expr(*expr, subs)),
            target,
            target_span,
            span,
        },
        Expr::Borrow {
            mutable,
            place,
            span,
            ..
        } => Expr::Borrow {
            id: AstId::alloc(),
            mutable,
            place: Box::new(sub_expr(*place, subs)),
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

        Expr::Block {
            stmts, tail, span, ..
        } => Expr::Block {
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
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            span,
            ..
        } => Expr::If {
            id: AstId::alloc(),
            cond: Box::new(sub_expr(*cond, subs)),
            then_body: then_body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            then_tail: then_tail.map(|e| Box::new(sub_expr(*e, subs))),
            else_branch: else_branch.map(|eb| Box::new(sub_else_branch(*eb, subs))),
            span,
        },
        Expr::Match {
            scrutinee,
            arms,
            span,
            ..
        } => Expr::Match {
            id: AstId::alloc(),
            scrutinee: Box::new(sub_expr(*scrutinee, subs)),
            arms: arms
                .into_iter()
                .map(|arm| MatchExprArm {
                    id: AstId::alloc(),
                    pattern: arm.pattern,
                    body: arm.body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
                    tail: arm.tail.map(|e| Box::new(sub_expr(*e, subs))),
                    span: arm.span,
                })
                .collect(),
            span,
        },
        Expr::IfLet {
            pattern,
            source,
            then_body,
            then_tail,
            else_branch,
            span,
            ..
        } => Expr::IfLet {
            id: AstId::alloc(),
            pattern,
            source: Box::new(sub_expr(*source, subs)),
            then_body: then_body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            then_tail: then_tail.map(|e| Box::new(sub_expr(*e, subs))),
            else_branch: else_branch.map(|eb| Box::new(sub_else_branch(*eb, subs))),
            span,
        },
    }
}

fn sub_else_branch(eb: ElseBranch, subs: &FxHashMap<Astr, SubstValue>) -> ElseBranch {
    match eb {
        ElseBranch::ElseIf(expr) => ElseBranch::ElseIf(sub_expr(expr, subs)),
        ElseBranch::Else { body, tail, span } => ElseBranch::Else {
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            tail: tail.map(|e| Box::new(sub_expr(*e, subs))),
            span,
        },
    }
}

// -- Flatten / fold helpers -------------------------------------------

/// Flatten a left-associative pipe chain into a sequence of stages.
///
/// `Pipe(Pipe(a, b), c)` -> `[a, b, c]`
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
/// `Add(Add(a, b), c)` -> `[a, b, c]`
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

// -- Statement / node substitution ------------------------------------

fn sub_stmt(stmt: Stmt, subs: &FxHashMap<Astr, SubstValue>) -> Stmt {
    match stmt {
        Stmt::Store {
            place, expr, span, ..
        } => Stmt::Store {
            id: AstId::alloc(),
            place: Box::new(sub_expr(*place, subs)),
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::DerefStore {
            target, expr, span, ..
        } => Stmt::DerefStore {
            id: AstId::alloc(),
            target: Box::new(sub_expr(*target, subs)),
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::Expr(expr) => Stmt::Expr(sub_expr(expr, subs)),
        // Script mode statements
        Stmt::LetBind {
            name, expr, span, ..
        } => Stmt::LetBind {
            id: AstId::alloc(),
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::LetUninit { name, span, .. } => Stmt::LetUninit {
            id: AstId::alloc(),
            name,
            span,
        },
        Stmt::Assign {
            name, expr, span, ..
        } => Stmt::Assign {
            id: AstId::alloc(),
            name,
            expr: sub_expr(expr, subs),
            span,
        },
        Stmt::While {
            cond, body, span, ..
        } => Stmt::While {
            id: AstId::alloc(),
            cond: sub_expr(cond, subs),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
        Stmt::For {
            binding,
            head,
            body,
            span,
            ..
        } => Stmt::For {
            id: AstId::alloc(),
            callee_id: AstId::alloc(),
            binding,
            head: match head {
                ForHead::Value(e) => ForHead::Value(sub_expr(e, subs)),
                ForHead::Range { lo, hi } => ForHead::Range {
                    lo: sub_expr(lo, subs),
                    hi: sub_expr(hi, subs),
                },
            },
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
        Stmt::Break { span, .. } => Stmt::Break {
            id: AstId::alloc(),
            span,
        },
        Stmt::Continue { span, .. } => Stmt::Continue {
            id: AstId::alloc(),
            span,
        },
        Stmt::Anyorder { body, span, .. } => Stmt::Anyorder {
            id: AstId::alloc(),
            body: body.into_iter().map(|s| sub_stmt(s, subs)).collect(),
            span,
        },
        Stmt::WhileLet {
            pattern,
            source,
            body,
            span,
            ..
        } => Stmt::WhileLet {
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
    }
}

// -- Compile-time splice position validation --------------------------
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
            if splice_names.contains(&name.name) && !in_seq {
                errors.push((name.name, *span));
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
        Expr::Index { object, index, .. } => {
            validate_splice_expr(object, false, splice_names, errors);
            validate_splice_expr(index, false, splice_names, errors);
        }
        Expr::FuncCall { func, args, .. } => {
            validate_splice_expr(func, false, splice_names, errors);
            for arg in args {
                validate_splice_expr(arg, true, splice_names, errors);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            validate_splice_expr(receiver, false, splice_names, errors);
            for arg in args {
                validate_splice_expr(arg, true, splice_names, errors);
            }
        }
        Expr::Lambda { body, .. } => {
            validate_splice_expr(body, false, splice_names, errors);
        }
        Expr::Paren { inner, .. }
        | Expr::Try { inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::Borrow { place: inner, .. } => {
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
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            validate_splice_expr(cond, false, splice_names, errors);
            for s in then_body {
                validate_splice_stmt(s, splice_names, errors);
            }
            if let Some(tail) = then_tail {
                validate_splice_expr(tail, false, splice_names, errors);
            }
            if let Some(eb) = else_branch {
                validate_splice_else_branch(eb, splice_names, errors);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            validate_splice_expr(scrutinee, false, splice_names, errors);
            for arm in arms {
                for s in &arm.body {
                    validate_splice_stmt(s, splice_names, errors);
                }
                if let Some(tail) = &arm.tail {
                    validate_splice_expr(tail, false, splice_names, errors);
                }
            }
        }
        Expr::IfLet {
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            validate_splice_expr(source, false, splice_names, errors);
            for s in then_body {
                validate_splice_stmt(s, splice_names, errors);
            }
            if let Some(tail) = then_tail {
                validate_splice_expr(tail, false, splice_names, errors);
            }
            if let Some(eb) = else_branch {
                validate_splice_else_branch(eb, splice_names, errors);
            }
        }
    }
}

fn validate_splice_else_branch(
    eb: &ElseBranch,
    splice_names: &[Astr],
    errors: &mut Vec<(Astr, Span)>,
) {
    match eb {
        ElseBranch::ElseIf(expr) => validate_splice_expr(expr, false, splice_names, errors),
        ElseBranch::Else { body, tail, .. } => {
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
            if let Some(tail) = tail {
                validate_splice_expr(tail, false, splice_names, errors);
            }
        }
    }
}

fn validate_splice_stmt(stmt: &Stmt, splice_names: &[Astr], errors: &mut Vec<(Astr, Span)>) {
    match stmt {
        Stmt::DerefStore { target, expr, .. } => {
            validate_splice_expr(target, false, splice_names, errors);
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Stmt::Store { place, expr, .. } => {
            validate_splice_expr(place, false, splice_names, errors);
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Stmt::Expr(expr) => {
            validate_splice_expr(expr, false, splice_names, errors);
        }
        // Script mode statements
        Stmt::LetBind { expr, .. } | Stmt::Assign { expr, .. } => {
            validate_splice_expr(expr, false, splice_names, errors);
        }
        Stmt::LetUninit { .. } => {}
        Stmt::WhileLet { source, body, .. } => {
            validate_splice_expr(source, false, splice_names, errors);
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
        }
        Stmt::For { head, body, .. } => {
            match head {
                ForHead::Value(e) => validate_splice_expr(e, false, splice_names, errors),
                ForHead::Range { lo, hi } => {
                    validate_splice_expr(lo, false, splice_names, errors);
                    validate_splice_expr(hi, false, splice_names, errors);
                }
            }
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => {}
        Stmt::While { cond, body, .. } => {
            validate_splice_expr(cond, false, splice_names, errors);
            for s in body {
                validate_splice_stmt(s, splice_names, errors);
            }
        }
        Stmt::Anyorder { body, .. } => {
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
    }
}
