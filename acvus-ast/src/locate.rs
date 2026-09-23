//! The nodes of a parsed body, by the source they cover.

use rustc_hash::FxHashMap;

use crate::ast::*;
use crate::span::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub id: AstId,
    pub span: Span,
    /// The id the checker records this node's callee under, where that is
    /// not the node's own: the `callee_id` of a method call, an index and a
    /// `for`.
    pub callee_id: Option<AstId>,
}

#[derive(Debug, Clone)]
pub struct Nodes {
    preorder: Vec<Node>,
}

impl Nodes {
    pub fn of_script<S>(script: &Script<S>) -> Self
    where
        S: Slot,
    {
        let mut nodes = Self {
            preorder: Vec::new(),
        };
        nodes.push(script.id, script.span, None);
        nodes.stmts(&script.stmts);
        if let Some(tail) = &script.tail {
            nodes.expr(tail);
        }
        nodes
    }

    pub fn of_template<S>(template: &Template<S>) -> Self
    where
        S: Slot,
    {
        let mut nodes = Self {
            preorder: Vec::new(),
        };
        nodes.push(template.id, template.span, None);
        nodes.stmts(&template.body);
        nodes
    }

    /// The nodes whose span holds `offset`, `start <= offset < end`,
    /// innermost first.
    pub fn at(&self, offset: usize) -> Vec<Node> {
        let mut holding: Vec<Node> = self
            .preorder
            .iter()
            .rev()
            .filter(|node| node.span.start <= offset && offset < node.span.end)
            .copied()
            .collect();
        holding.sort_by_key(|node| node.span.end - node.span.start);
        holding
    }

    pub fn spans(&self) -> FxHashMap<AstId, Span> {
        self.preorder
            .iter()
            .map(|node| (node.id, node.span))
            .collect()
    }

    fn push(&mut self, id: AstId, span: Span, callee_id: Option<AstId>) {
        self.preorder.push(Node {
            id,
            span,
            callee_id,
        });
    }

    fn error<S>(&mut self, node: &S)
    where
        S: Slot,
    {
        let ErrorNode { id, span } = node.node();
        self.push(id, span, None);
    }

    fn binder(&mut self, binder: &Binder) {
        self.push(binder.id, binder.span, None);
    }

    fn stmts<S>(&mut self, stmts: &[Stmt<S>])
    where
        S: Slot,
    {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    fn stmt<S>(&mut self, stmt: &Stmt<S>)
    where
        S: Slot,
    {
        match stmt {
            Stmt::Store {
                id,
                place,
                expr,
                span,
            } => {
                self.push(*id, *span, None);
                self.place(place);
                self.expr(expr);
            }
            Stmt::DerefStore {
                id,
                target,
                expr,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(target);
                self.expr(expr);
            }
            Stmt::Expr(expr) => self.expr(expr),
            Stmt::LetBind {
                id,
                binder,
                expr,
                span,
            } => {
                self.push(*id, *span, None);
                self.binder(binder);
                self.expr(expr);
            }
            Stmt::LetUninit { id, binder, span } => {
                self.push(*id, *span, None);
                self.binder(binder);
            }
            Stmt::Assign { id, expr, span, .. } => {
                self.push(*id, *span, None);
                self.expr(expr);
            }
            Stmt::Break { id, span } | Stmt::Continue { id, span } => self.push(*id, *span, None),
            Stmt::While {
                id,
                cond,
                body,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(cond);
                self.stmts(body);
            }
            Stmt::For {
                id,
                callee_id,
                binder,
                head,
                body,
                span,
            } => {
                self.push(*id, *span, Some(*callee_id));
                self.binder(binder);
                match head {
                    ForHead::Value(value) => self.expr(value),
                    ForHead::Range { lo, hi } => {
                        self.expr(lo);
                        self.expr(hi);
                    }
                }
                self.stmts(body);
            }
            Stmt::WhileLet {
                id,
                pattern,
                source,
                body,
                span,
            } => {
                self.push(*id, *span, None);
                self.pattern(pattern);
                self.expr(source);
                self.stmts(body);
            }
            Stmt::Anyorder { id, body, span } => {
                self.push(*id, *span, None);
                self.stmts(body);
            }
            Stmt::Append { id, expr, span } => {
                self.push(*id, *span, None);
                self.expr(expr);
            }
            Stmt::Error(node) => self.error(node),
        }
    }

    fn place<S>(&mut self, place: &Place<S>)
    where
        S: Slot,
    {
        match place {
            Place::Base(PlaceBase::Root { id, span, .. }) => self.push(*id, *span, None),
            Place::Base(PlaceBase::Element {
                id,
                callee_id,
                container,
                index,
                span,
            }) => {
                self.push(*id, *span, Some(*callee_id));
                self.expr(container.expr());
                self.expr(index);
            }
            Place::Field {
                id, object, span, ..
            } => {
                self.push(*id, *span, None);
                self.place(object);
            }
        }
    }

    fn expr<S>(&mut self, expr: &Expr<S>)
    where
        S: Slot,
    {
        match expr {
            Expr::Ident { id, span, .. }
            | Expr::Literal { id, span, .. }
            | Expr::ContextRef { id, span, .. } => self.push(*id, *span, None),
            Expr::BinaryOp {
                id,
                left,
                right,
                span,
                ..
            }
            | Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(left);
                self.expr(right);
            }
            Expr::UnaryOp {
                id,
                operand: inner,
                span,
                ..
            }
            | Expr::FieldAccess {
                id,
                object: inner,
                span,
                ..
            }
            | Expr::Paren { id, inner, span }
            | Expr::Borrow {
                id,
                place: inner,
                span,
                ..
            }
            | Expr::Cast {
                id,
                expr: inner,
                span,
                ..
            }
            | Expr::Try { id, inner, span }
            | Expr::Return {
                id,
                value: inner,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(inner);
            }
            Expr::Index {
                id,
                callee_id,
                object,
                index,
                span,
            } => {
                self.push(*id, *span, Some(*callee_id));
                self.expr(object);
                self.expr(index);
            }
            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(func);
                for arg in args {
                    self.expr(arg);
                }
            }
            Expr::MethodCall {
                id,
                callee_id,
                receiver,
                args,
                span,
                ..
            } => {
                self.push(*id, *span, Some(*callee_id));
                self.expr(receiver);
                for arg in args {
                    self.expr(arg);
                }
            }
            Expr::Lambda {
                id,
                params,
                body,
                span,
            } => {
                self.push(*id, *span, None);
                for param in params {
                    self.binder(param);
                }
                self.expr(body);
            }
            Expr::List {
                id,
                head,
                tail,
                span,
                ..
            } => {
                self.push(*id, *span, None);
                for element in head.iter().chain(tail) {
                    self.expr(element);
                }
            }
            Expr::Group { id, elements, span } => {
                self.push(*id, *span, None);
                for element in elements {
                    self.expr(element);
                }
            }
            Expr::Object { id, fields, span } => {
                self.push(*id, *span, None);
                for field in fields {
                    self.push(field.id, field.span, None);
                    self.expr(&field.value);
                }
            }
            Expr::Tuple { id, elements, span } => {
                self.push(*id, *span, None);
                for element in elements {
                    match element {
                        TupleElem::Expr(element) => self.expr(element),
                        TupleElem::Wildcard(_) => {}
                    }
                }
            }
            Expr::Block {
                id,
                stmts,
                tail,
                span,
            } => {
                self.push(*id, *span, None);
                self.stmts(stmts);
                self.expr(tail);
            }
            Expr::Variant {
                id, payload, span, ..
            } => {
                self.push(*id, *span, None);
                if let Some(payload) = payload {
                    self.expr(payload);
                }
            }
            Expr::If {
                id,
                cond,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(cond);
                self.branch(then_body, then_tail.as_deref(), else_branch.as_deref());
            }
            Expr::IfLet {
                id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                self.push(*id, *span, None);
                self.pattern(pattern);
                self.expr(source);
                self.branch(then_body, then_tail.as_deref(), else_branch.as_deref());
            }
            Expr::Match {
                id,
                scrutinee,
                arms,
                span,
            } => {
                self.push(*id, *span, None);
                self.expr(scrutinee);
                for arm in arms {
                    self.push(arm.id, arm.span, None);
                    self.pattern(&arm.pattern);
                    self.stmts(&arm.body);
                    if let Some(tail) = &arm.tail {
                        self.expr(tail);
                    }
                }
            }
            Expr::Error(node) => self.error(node),
        }
    }

    fn branch<S>(
        &mut self,
        then_body: &[Stmt<S>],
        then_tail: Option<&Expr<S>>,
        else_branch: Option<&ElseBranch<S>>,
    ) where
        S: Slot,
    {
        self.stmts(then_body);
        if let Some(tail) = then_tail {
            self.expr(tail);
        }
        match else_branch {
            Some(ElseBranch::ElseIf(chained)) => self.expr(chained),
            Some(ElseBranch::Else { body, tail, .. }) => {
                self.stmts(body);
                if let Some(tail) = tail {
                    self.expr(tail);
                }
            }
            None => {}
        }
    }

    fn pattern<S>(&mut self, pattern: &Pattern<S>)
    where
        S: Slot,
    {
        match pattern {
            Pattern::Binding { id, span, .. }
            | Pattern::ContextBind { id, span, .. }
            | Pattern::Literal { id, span, .. }
            | Pattern::Wildcard { id, span } => self.push(*id, *span, None),
            Pattern::List {
                id,
                head,
                tail,
                span,
                ..
            } => {
                self.push(*id, *span, None);
                for part in head.iter().chain(tail) {
                    self.pattern(part);
                }
            }
            Pattern::Object { id, fields, span } => {
                self.push(*id, *span, None);
                for field in fields {
                    self.push(field.id, field.span, None);
                    self.pattern(&field.pattern);
                }
            }
            Pattern::Tuple { id, elements, span } => {
                self.push(*id, *span, None);
                for element in elements {
                    match element {
                        TuplePatternElem::Pattern(part) => self.pattern(part),
                        TuplePatternElem::Wildcard(_) => {}
                    }
                }
            }
            Pattern::Variant {
                id, payload, span, ..
            } => {
                self.push(*id, *span, None);
                if let Some(payload) = payload {
                    self.pattern(payload);
                }
            }
            Pattern::Error(node) => self.error(node),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    #[test]
    fn nodes_at_an_offset_are_innermost_first() {
        let interner = Interner::new();
        let source = "let s = 1; s + 2";
        let script = crate::parse_script(&interner, source).expect("the source parses");
        let Some(tail) = &script.tail else {
            panic!("the script has a tail");
        };
        let Expr::BinaryOp { left, .. } = tail.as_ref() else {
            panic!("the tail is a binary operation");
        };
        let at_use = source.rfind('s').expect("the use is in the source");
        let ids: Vec<AstId> = Nodes::of_script(&script)
            .at(at_use)
            .iter()
            .map(|node| node.id)
            .collect();
        assert_eq!(ids, vec![left.id(), tail.id(), script.id]);
    }

    #[test]
    fn an_offset_at_a_span_end_is_outside_it() {
        let interner = Interner::new();
        let source = "1";
        let script = crate::parse_script(&interner, source).expect("the source parses");
        let nodes = Nodes::of_script(&script);
        assert!(nodes.at(source.len()).is_empty());
    }
}
