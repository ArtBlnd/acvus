//! Contexts (context §3): a context is a place outside the body, which a
//! body names by `RefTarget::Context` and reads, takes, lends and assigns
//! as any place. A body states the contexts it names, each with the
//! volatility its declaration states; where the page is
//! seen beyond the program, at a call that may touch a context and at a
//! return, is `passes::analysis::Loans`' to record, not lower's. A
//! volatile context's place operations are effects, on the order chain.
//!
//! The contexts a body names are read off its own statements, not its
//! lambdas'. Inputs (programs §4) are read off the same walk, its lambdas'
//! included: each is a parameter of the entry body, held in a slot, which
//! a lambda that reads it captures.

use acvus_ast::{ElseBranch, Expr, ForHead, ObjectPatternField, Pattern, Place, PlaceBase, RefKind, Root, Slot, Span, Stmt, TuplePatternElem};
use acvus_utils::{Astr, QualifiedRef};

use super::ctxt::{Home, Lcx};
use super::place::{Operand, Place as LPlace, assign, read};
use crate::ir::{RefTarget, ValueId};
use crate::ty::Ty;

/// What a body names itself, each once, in the order it first names it.
#[derive(Default)]
pub(super) struct Named {
    pub contexts: Vec<NamedContext>,
    pub inputs: Vec<NamedInput>,
}

/// A context a body names, and its type.
pub(super) struct NamedContext {
    pub context: QualifiedRef,
    pub ty: Ty,
}

/// An input a body names, its type, and where it first names it.
pub(super) struct NamedInput {
    pub name: Astr,
    pub ty: Ty,
    pub span: Span,
}

/// Reads what the statements and tail of one body name.
pub(super) struct Namer<'t> {
    typeck: &'t crate::typeck::Typeck,
    named: Named,
    /// How many lambdas deep the walk is: a lambda's contexts are its own
    /// body's, and the inputs it reads are the entry's, which it captures.
    lambdas: usize,
}

impl<'t> Namer<'t> {
    pub fn new(typeck: &'t crate::typeck::Typeck) -> Self {
        Namer { typeck, named: Named::default(), lambdas: 0 }
    }

    pub fn finish(self) -> Named {
        self.named
    }

    fn context(&mut self, context: QualifiedRef, ty: Ty) {
        if self.lambdas == 0 && !self.named.contexts.iter().any(|named| named.context == context) {
            self.named.contexts.push(NamedContext { context, ty });
        }
    }

    fn input(&mut self, name: Astr, ty: Ty, span: Span) {
        if !self.named.inputs.iter().any(|named| named.name == name) {
            self.named.inputs.push(NamedInput { name, ty, span });
        }
    }

    fn ty(&self, site: acvus_ast::AstId) -> Ty {
        *self
            .typeck
            .types
            .get(&site)
            .unwrap_or_else(|| panic!("typeck types every site of an admitted program, and not {site:?}"))
    }

    pub fn stmts<S: Slot>(&mut self, stmts: &[Stmt<S>]) {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    fn stmt<S: Slot>(&mut self, stmt: &Stmt<S>) {
        match stmt {
            Stmt::Store { place, expr, .. } => {
                self.expr(expr);
                self.place(place);
            }
            Stmt::DerefStore { target, expr, .. } => {
                self.expr(target);
                self.expr(expr);
            }
            Stmt::Expr(expr) | Stmt::LetBind { expr, .. } | Stmt::Assign { expr, .. } | Stmt::Append { expr, .. } => {
                self.expr(expr)
            }
            Stmt::While { cond, body, .. } => {
                self.expr(cond);
                self.stmts(body);
            }
            Stmt::For { head, body, .. } => {
                match head {
                    ForHead::Value(expr) => self.expr(expr),
                    ForHead::Range { lo, hi } => {
                        self.expr(lo);
                        self.expr(hi);
                    }
                }
                self.stmts(body);
            }
            Stmt::WhileLet { pattern, source, body, .. } => {
                self.expr(source);
                self.pattern(pattern);
                self.stmts(body);
            }
            Stmt::Anyorder { body, .. } => self.stmts(body),
            Stmt::LetUninit { .. }
            | Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::Text { .. }
            | Stmt::FnDecl(_)
            | Stmt::Error(_) => {}
        }
    }

    fn place<S: Slot>(&mut self, place: &Place<S>) {
        match place {
            Place::Base(PlaceBase::Root { id, root: Root::Context(name), .. }) => {
                self.context(QualifiedRef::root(*name), self.ty(*id))
            }
            Place::Base(PlaceBase::Root { .. }) => {}
            Place::Base(PlaceBase::Element { container, index, .. }) => {
                self.expr(container.expr());
                self.expr(index);
            }
            Place::Field { object, .. } => self.place(object),
        }
    }

    pub fn expr<S: Slot>(&mut self, expr: &Expr<S>) {
        acvus_utils::grow(|| self.expr_level(expr))
    }

    fn expr_level<S: Slot>(&mut self, expr: &Expr<S>) {
        match expr {
            Expr::Ident { id, name, ref_kind: RefKind::ExternParam, span } => self.input(name.name, self.ty(*id), *span),
            Expr::ContextRef { id, name, .. } => self.context(*name, self.ty(*id)),
            Expr::Lambda { body, .. } => {
                self.lambdas += 1;
                self.expr(body);
                self.lambdas -= 1;
            }
            Expr::Ident { .. } | Expr::Literal { .. } | Expr::Error(_) => {}
            Expr::BinaryOp { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::Index { object, index, .. } => {
                self.expr(object);
                self.expr(index);
            }
            Expr::Pipe { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::UnaryOp { operand: inner, .. }
            | Expr::FieldAccess { object: inner, .. }
            | Expr::Paren { inner, .. }
            | Expr::Borrow { place: inner, .. }
            | Expr::Cast { expr: inner, .. }
            | Expr::Try { inner, .. } => self.expr(inner),
            Expr::Return { value, .. } => self.expr(value),
            Expr::FuncCall { func, args, .. } => {
                self.expr(func);
                args.exprs().for_each(|arg| self.expr(arg));
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.expr(receiver);
                args.iter().for_each(|arg| self.expr(arg));
            }
            Expr::List { head, tail, .. } => head.iter().chain(tail).for_each(|item| self.expr(item)),
            Expr::Group { elements, .. } => elements.iter().for_each(|element| self.expr(element)),
            Expr::Object { fields, .. } => fields.iter().for_each(|field| self.expr(&field.value)),
            Expr::Tuple { elements, .. } => {
                for element in elements {
                    if let acvus_ast::TupleElem::Expr(element) = element {
                        self.expr(element);
                    }
                }
            }
            Expr::Block { stmts, tail, .. } => {
                self.stmts(stmts);
                self.expr(tail);
            }
            Expr::Variant { payload, .. } => {
                if let Some(payload) = payload {
                    self.expr(payload);
                }
            }
            Expr::If { cond, then_body, then_tail, else_branch, .. } => {
                self.expr(cond);
                self.branch(then_body, then_tail.as_deref());
                self.else_branch(else_branch.as_deref());
            }
            Expr::IfLet { pattern, source, then_body, then_tail, else_branch, .. } => {
                self.expr(source);
                self.pattern(pattern);
                self.branch(then_body, then_tail.as_deref());
                self.else_branch(else_branch.as_deref());
            }
            Expr::Match { scrutinee, arms, .. } => {
                self.expr(scrutinee);
                for arm in arms {
                    self.pattern(&arm.pattern);
                    self.branch(&arm.body, arm.tail.as_deref());
                }
            }
        }
    }

    fn branch<S: Slot>(&mut self, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) {
        self.stmts(stmts);
        if let Some(tail) = tail {
            self.expr(tail);
        }
    }

    fn else_branch<S: Slot>(&mut self, branch: Option<&ElseBranch<S>>) {
        match branch {
            None => {}
            Some(ElseBranch::ElseIf(expr)) => self.expr(expr),
            Some(ElseBranch::Else { body, tail, .. }) => self.branch(body, tail.as_deref()),
        }
    }

    /// A pattern `@x` stores into the context; the context's type is what
    /// the matched value is coerced to there, or the matched value's own.
    fn pattern<S: Slot>(&mut self, pattern: &Pattern<S>) {
        match pattern {
            Pattern::ContextBind { id, name, .. } => {
                let ty = match self.typeck.coercions.get(id) {
                    Some(coercion) => coercion.to(),
                    None => self.ty(*id),
                };
                self.context(*name, ty);
            }
            Pattern::List { head, tail, .. } => head.iter().chain(tail).for_each(|part| self.pattern(part)),
            Pattern::Object { fields, .. } => fields.iter().for_each(|ObjectPatternField { pattern, .. }| self.pattern(pattern)),
            Pattern::Tuple { elements, .. } => {
                for element in elements {
                    if let TuplePatternElem::Pattern(part) = element {
                        self.pattern(part);
                    }
                }
            }
            Pattern::Variant { payload: Some(payload), .. } => self.pattern(payload),
            Pattern::Binding { .. }
            | Pattern::Literal { .. }
            | Pattern::Wildcard { .. }
            | Pattern::Variant { payload: None, .. }
            | Pattern::Error(_) => {}
        }
    }
}

/// The body's entry: it states each context it names, with its type and
/// the volatility its declaration states.
pub(super) fn enter_body(cx: &mut Lcx<'_, '_>, named: &Named) {
    for NamedContext { context, ty } in &named.contexts {
        let volatile = cx
            .decls
            .context_is_volatile(*context)
            .unwrap_or_else(|| panic!("the context {context:?} an admitted program names is declared"));
        cx.builder.context(*context, *ty, volatile);
    }
}

/// The entry body's inputs: each it names is a parameter held in a slot
/// (programs §4). A lambda reaches an input by capturing it.
pub(super) fn enter_inputs(cx: &mut Lcx<'_, '_>, named: &Named) {
    for NamedInput { name, ty, span } in &named.inputs {
        let param = cx.builder.param(*name, *ty, *span);
        let slot = cx.builder.slot(*ty, *span);
        assign(cx, slot_place(slot, *ty), param, *span);
        cx.inputs.insert(*name, Home::Slot(slot));
    }
}

fn slot_place(slot: ValueId, ty: Ty) -> LPlace {
    LPlace { target: RefTarget::Var(slot), path: Vec::new(), ty }
}

/// The whole of `context`, a `ty`, as a place.
pub(super) fn context_place(context: QualifiedRef, ty: Ty) -> LPlace {
    LPlace { target: RefTarget::Context(context), path: Vec::new(), ty }
}

/// `@x` read: a take of the context, which copies it where its type does.
pub(super) fn lower_context_read(cx: &mut Lcx<'_, '_>, context: QualifiedRef, ty: Ty, span: Span) -> ValueId {
    read(cx, Operand::Place(context_place(context, ty)), span)
}

/// A store into `context` of a value a pattern bound (`@x` in a pattern).
pub(super) fn store(cx: &mut Lcx<'_, '_>, context: QualifiedRef, value: ValueId, span: Span) {
    let ty = cx.value_ty(value);
    assign(cx, context_place(context, ty), value, span);
}
