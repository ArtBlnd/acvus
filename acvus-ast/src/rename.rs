use acvus_utils::{Astr, QualifiedRef};

use crate::ast::*;

pub trait Renames {
    fn context(&mut self, name: Astr) -> Astr;
    fn input(&mut self, name: Astr) -> Astr;
    /// A method call's name is one of these as well, since `recv.f(a)`
    /// resolves `f` as the bare call `f(recv, a)` does.
    fn value(&mut self, name: Astr) -> Astr;
}

pub fn rename_script<S, R>(script: &mut Script<S>, renames: &mut R)
where
    R: Renames,
{
    stmts(&mut script.stmts, renames);
    if let Some(tail) = &mut script.tail {
        expr(tail, renames);
    }
}

pub fn rename_template<S, R>(template: &mut Template<S>, renames: &mut R)
where
    R: Renames,
{
    stmts(&mut template.body, renames);
}

fn stmts<S, R>(stmts: &mut [Stmt<S>], renames: &mut R)
where
    R: Renames,
{
    for stmt in stmts {
        match stmt {
            Stmt::Store { place: p, expr: e, .. } => {
                place(p, renames);
                expr(e, renames);
            }
            Stmt::DerefStore { target, expr: e, .. } => {
                expr(target, renames);
                expr(e, renames);
            }
            Stmt::Expr(e) => expr(e, renames),
            Stmt::LetBind { binder: b, expr: e, .. } => {
                binder(b, renames);
                expr(e, renames);
            }
            Stmt::LetUninit { binder: b, .. } => binder(b, renames),
            Stmt::Assign { name, expr: e, .. } => {
                *name = renames.value(*name);
                expr(e, renames);
            }
            Stmt::While { cond, body, .. } => {
                expr(cond, renames);
                self::stmts(body, renames);
            }
            Stmt::For {
                binder: b,
                head,
                body,
                ..
            } => {
                binder(b, renames);
                match head {
                    ForHead::Value(e) => expr(e, renames),
                    ForHead::Range { lo, hi } => {
                        expr(lo, renames);
                        expr(hi, renames);
                    }
                }
                self::stmts(body, renames);
            }
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Error(_) => {}
            Stmt::WhileLet {
                pattern: p,
                source,
                body,
                ..
            } => {
                pattern(p, renames);
                expr(source, renames);
                self::stmts(body, renames);
            }
            Stmt::Anyorder { body, .. } => self::stmts(body, renames),
            Stmt::Append { expr: e, .. } => expr(e, renames),
        }
    }
}

fn binder<R>(binder: &mut Binder, renames: &mut R)
where
    R: Renames,
{
    binder.name = renames.value(binder.name);
}

fn bare<R>(name: &mut QualifiedRef, rename: impl FnOnce(&mut R, Astr) -> Astr, renames: &mut R) {
    if name.namespace.is_none() {
        name.name = rename(renames, name.name);
    }
}

fn place<S, R>(place: &mut Place<S>, renames: &mut R)
where
    R: Renames,
{
    match place {
        Place::Field { object, .. } => self::place(object, renames),
        Place::Base(PlaceBase::Root { root, .. }) => {
            *root = match *root {
                Root::Local(name) => Root::Local(renames.value(name)),
                Root::ExternParam(name) => Root::ExternParam(renames.input(name)),
                Root::Context(name) => Root::Context(renames.context(name)),
            };
        }
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            expr(container.expr_mut(), renames);
            expr(index, renames);
        }
    }
}

fn pattern<S, R>(pattern: &mut Pattern<S>, renames: &mut R)
where
    R: Renames,
{
    match pattern {
        Pattern::Binding { name, ref_kind, .. } => {
            *name = match ref_kind {
                RefKind::Value => renames.value(*name),
                RefKind::ExternParam => renames.input(*name),
            };
        }
        Pattern::ContextBind { name, .. } => bare(name, R::context, renames),
        Pattern::Literal { .. } | Pattern::Wildcard { .. } | Pattern::Error(_) => {}
        Pattern::List { head, tail, .. } => {
            for p in head.iter_mut().chain(tail) {
                self::pattern(p, renames);
            }
        }
        Pattern::Object { fields, .. } => {
            for field in fields {
                self::pattern(&mut field.pattern, renames);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for element in elements {
                match element {
                    TuplePatternElem::Pattern(p) => self::pattern(p, renames),
                    TuplePatternElem::Wildcard(_) => {}
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(p) = payload {
                self::pattern(p, renames);
            }
        }
    }
}

fn expr<S, R>(e: &mut Expr<S>, renames: &mut R)
where
    R: Renames,
{
    match e {
        Expr::Ident { name, ref_kind, .. } => match ref_kind {
            RefKind::Value => bare(name, R::value, renames),
            RefKind::ExternParam => bare(name, R::input, renames),
        },
        Expr::ContextRef { name, .. } => bare(name, R::context, renames),
        Expr::Literal { .. } | Expr::Error(_) => {}
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. } => {
            expr(left, renames);
            expr(right, renames);
        }
        Expr::UnaryOp { operand: inner, .. }
        | Expr::FieldAccess { object: inner, .. }
        | Expr::Paren { inner, .. }
        | Expr::Borrow { place: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::Try { inner, .. }
        | Expr::Return { value: inner, .. } => expr(inner, renames),
        Expr::Index { object, index, .. } => {
            expr(object, renames);
            expr(index, renames);
        }
        Expr::FuncCall { func, args, .. } => {
            expr(func, renames);
            for arg in args {
                expr(arg, renames);
            }
        }
        Expr::MethodCall {
            receiver,
            name,
            args,
            ..
        } => {
            expr(receiver, renames);
            *name = renames.value(*name);
            for arg in args {
                expr(arg, renames);
            }
        }
        Expr::Lambda { params, body, .. } => {
            for param in params {
                binder(param, renames);
            }
            expr(body, renames);
        }
        Expr::List { head, tail, .. } => {
            for element in head.iter_mut().chain(tail) {
                expr(element, renames);
            }
        }
        Expr::Group { elements, .. } => {
            for element in elements {
                expr(element, renames);
            }
        }
        Expr::Object { fields, .. } => {
            for field in fields {
                expr(&mut field.value, renames);
            }
        }
        Expr::Tuple { elements, .. } => {
            for element in elements {
                match element {
                    TupleElem::Expr(inner) => expr(inner, renames),
                    TupleElem::Wildcard(_) => {}
                }
            }
        }
        Expr::Block { stmts: body, tail, .. } => {
            stmts(body, renames);
            expr(tail, renames);
        }
        Expr::Variant { payload, .. } => {
            if let Some(payload) = payload {
                expr(payload, renames);
            }
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            expr(cond, renames);
            branch(then_body, then_tail, renames);
            if let Some(eb) = else_branch {
                else_(eb, renames);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            expr(scrutinee, renames);
            for arm in arms {
                pattern(&mut arm.pattern, renames);
                branch(&mut arm.body, &mut arm.tail, renames);
            }
        }
        Expr::IfLet {
            pattern: p,
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            pattern(p, renames);
            expr(source, renames);
            branch(then_body, then_tail, renames);
            if let Some(eb) = else_branch {
                else_(eb, renames);
            }
        }
    }
}

fn branch<S, R>(body: &mut [Stmt<S>], tail: &mut Option<Box<Expr<S>>>, renames: &mut R)
where
    R: Renames,
{
    stmts(body, renames);
    if let Some(tail) = tail {
        expr(tail, renames);
    }
}

fn else_<S, R>(eb: &mut ElseBranch<S>, renames: &mut R)
where
    R: Renames,
{
    match eb {
        ElseBranch::ElseIf(e) => expr(e, renames),
        ElseBranch::Else { body, tail, .. } => branch(body, tail, renames),
    }
}
