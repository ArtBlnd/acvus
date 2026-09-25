//! A body's flows (RFC-0079 rule 5), inferred by the checker from the body,
//! for each lambda and for the whole body. The inference is flow-insensitive
//! and joins by union; what a write through a reference puts into storage is
//! kept in one set that every read through storage takes, so a write is
//! never lost for not knowing which storage it reached. The borrow check of
//! the lowered body checks this inference: a result or a write holding a
//! loan the flows do not name is refused there (`validate::borrow_check`).
//!
//! Writes are not inferred precisely, and that is a decision: every input
//! may reach what a `&mut` parameter or a mutable capture points at, the
//! reading step 2 of RFC-0079 gave every call. A call's precision is spent
//! on its result.

use std::collections::BTreeSet;

use acvus_ast::{
    AstId, ElseBranch, Expr, ForHead, MatchExprArm, ObjectExprField, Pattern, Place, PlaceBase,
    RefKind, Root, Slot, Stmt, TupleElem, TuplePatternElem, UnaryOp,
};
use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use super::{CallChoice, Resolved, TypeChecker};
use crate::analysis::loans::{positions, reads_through, writes_through};
use crate::solver::{Answer, SettledSignature};
use crate::ty::{Alignment, Flow, FlowEnd, FlowTerm, Flows, InferTy, Source, Ty, TyTerm};

type Origin = BTreeSet<Source>;

/// The body the flows are inferred for: a lambda, or the whole body.
struct Frame {
    lambda: Option<AstId>,
    /// A lambda's parameters, by the binder that holds each.
    params: FxHashMap<AstId, usize>,
    /// The whole body's parameters, by the `$` name that reads each.
    inputs: FxHashMap<Astr, usize>,
}

/// Which lambda binds each binder, and which lambda encloses each lambda.
#[derive(Default)]
struct Scopes {
    owner: FxHashMap<AstId, Option<AstId>>,
    parent: FxHashMap<AstId, Option<AstId>>,
}

impl Scopes {
    /// Whether a binder is bound by `frame` or a body enclosing it: a name
    /// a lambda nested in `frame` reads from outside itself.
    fn outside_of_nested(&self, binder: AstId, frame: Option<AstId>) -> bool {
        let Some(&owner) = self.owner.get(&binder) else {
            return true;
        };
        let mut at = frame;
        loop {
            if at == owner {
                return true;
            }
            match at {
                Some(lambda) => at = self.parent.get(&lambda).copied().flatten(),
                None => return false,
            }
        }
    }
}

#[derive(Default, Clone, PartialEq)]
struct Origins {
    locals: FxHashMap<AstId, Origin>,
    written: Origin,
    result: Origin,
}

struct LambdaSite<'e, S> {
    id: AstId,
    params: Vec<AstId>,
    body: &'e Expr<S>,
}

/// A body to walk: a script's statements and tail, or a template's
/// statements.
pub(super) struct Body<'e, S> {
    pub stmts: &'e [Stmt<S>],
    pub tail: Option<&'e Expr<S>>,
}

fn any(origin: &Origin) -> Origin {
    origin
        .iter()
        .map(|source| Source {
            from: source.from,
            alignment: Alignment::Any,
        })
        .collect()
}

fn one(from: FlowEnd, alignment: Alignment) -> Origin {
    BTreeSet::from([Source { from, alignment }])
}

impl<S> TypeChecker<'_, '_, '_, S>
where
    S: Slot,
{
    /// Every lambda's flows raised to what its body joins, until no lambda's
    /// grows, and then the whole body's flows. A lambda's flows read the
    /// flows of what it calls, which may be another lambda's, so the rounds
    /// climb together; each round only joins, and a function type's flows
    /// are a finite set, so they stop.
    pub(super) fn infer_flows(&mut self, body: Body<'_, S>) -> Flows {
        let mut scopes = Scopes::default();
        let mut lambdas: Vec<LambdaSite<'_, S>> = Vec::new();
        scope_stmts(body.stmts, None, &mut scopes, &mut lambdas);
        if let Some(tail) = body.tail {
            scope_expr(tail, None, &mut scopes, &mut lambdas);
        }
        loop {
            let mut grew = false;
            for site in &lambdas {
                let Some(lambda) = self.lambda_type(site.id) else {
                    continue;
                };
                let flows = self.lambda_flows(site, &scopes, &lambda);
                let raised = self
                    .solver
                    .raise_flows(&lambda.flows, &flows)
                    .expect("a lambda's type carries a flow variable of its own");
                grew |= raised == crate::solver::Raised::Grew;
            }
            if !grew {
                break;
            }
        }
        self.whole_body_flows(&body, &scopes)
    }

    /// `None` for a lambda the checker never reached, which is in a body
    /// it refused.
    fn lambda_type(&self, lambda: AstId) -> Option<LambdaType> {
        let recorded = self.type_map.get(&lambda)?;
        let TyTerm::Fn {
            params,
            ret,
            captures,
            flows,
            ..
        } = self.solver.resolve_ty(recorded)
        else {
            unreachable!("check_lambda records a lambda's type as a function type")
        };
        Some(LambdaType {
            params: params
                .iter()
                .map(|param| self.closed_ty(&param.ty))
                .collect(),
            ret: self.closed_ty(&ret),
            captures: captures
                .iter()
                .map(|capture| self.closed_ty(capture))
                .collect(),
            flows,
        })
    }

    fn lambda_flows(
        &self,
        site: &LambdaSite<'_, S>,
        scopes: &Scopes,
        lambda: &LambdaType,
    ) -> Flows {
        let frame = Frame {
            lambda: Some(site.id),
            params: site
                .params
                .iter()
                .enumerate()
                .map(|(index, binder)| (*binder, index))
                .collect(),
            inputs: FxHashMap::default(),
        };
        let found = self.walk_to_fixpoint(&frame, scopes, |walk| {
            let value = walk.value(site.body);
            walk.state.result.extend(value);
        });
        Signature {
            params: lambda.params.clone(),
            ret: lambda.ret.clone(),
            has_captures: !lambda.captures.is_empty(),
            captures_written: lambda
                .captures
                .iter()
                .any(|capture| capture.as_ref().is_none_or(writes_through)),
        }
        .flows(&found.result)
    }

    fn whole_body_flows(&self, body: &Body<'_, S>, scopes: &Scopes) -> Flows {
        let frame = Frame {
            lambda: None,
            params: FxHashMap::default(),
            inputs: self
                .param_types
                .iter()
                .enumerate()
                .map(|(index, param)| (param.name, index))
                .collect(),
        };
        let found = self.walk_to_fixpoint(&frame, scopes, |walk| {
            walk.stmts(body.stmts);
            if let Some(tail) = body.tail {
                let value = walk.value(tail);
                walk.state.result.extend(value);
            }
        });
        let ret = match body.tail {
            Some(tail) => self
                .return_ty
                .as_ref()
                .or_else(|| self.type_map.get(&tail.id()))
                .and_then(|ty| self.closed_ty(ty)),
            None => Some(Ty::Unit),
        };
        Signature {
            params: self
                .param_types
                .iter()
                .map(|param| self.closed_ty(&param.ty))
                .collect(),
            ret,
            has_captures: false,
            captures_written: false,
        }
        .flows(&found.result)
    }

    fn walk_to_fixpoint(
        &self,
        frame: &Frame,
        scopes: &Scopes,
        mut walk_body: impl FnMut(&mut Walk<'_, '_, '_, '_, '_, '_, S>),
    ) -> Origins {
        let mut state = Origins::default();
        loop {
            let before = state.clone();
            let stored: Origin = state.locals.values().flatten().cloned().collect();
            let mut walk = Walk {
                checker: self,
                frame,
                scopes,
                stored,
                state: &mut state,
            };
            walk_body(&mut walk);
            if state == before {
                return state;
            }
        }
    }

    /// `None` where the type is not closed, which the body's refusal
    /// reports; every reader takes an unknown type as one that may hold,
    /// read through and write through anything.
    fn closed_ty(&self, ty: &InferTy) -> Option<Ty> {
        self.solver.close_ty(&self.solver.resolve_ty(ty)).ok()
    }
}

/// A lambda's function type, as `check_lambda` recorded it.
struct LambdaType {
    params: Vec<Option<Ty>>,
    ret: Option<Ty>,
    captures: Vec<Option<Ty>>,
    flows: FlowTerm<crate::ty::Infer>,
}

/// The ends a body's flows name, as its type has them.
struct Signature {
    params: Vec<Option<Ty>>,
    ret: Option<Ty>,
    has_captures: bool,
    captures_written: bool,
}

impl Signature {
    fn flows(&self, result: &Origin) -> Flows {
        let inputs: Vec<FlowEnd> = (0..self.params.len())
            .map(FlowEnd::Param)
            .chain(self.has_captures.then_some(FlowEnd::Captures))
            .collect();
        let one_shape = |index: usize| match (&self.ret, self.params.get(index)) {
            (Some(ret), Some(Some(param))) => positions(ret) == positions(param),
            _ => false,
        };
        let into_result = result.iter().map(|source| {
            let alignment = match *source {
                Source {
                    from: FlowEnd::Param(index),
                    alignment: Alignment::Aligned,
                } if one_shape(index) => Alignment::Aligned,
                Source { .. } => Alignment::Any,
            };
            Flow {
                to: FlowEnd::Result,
                from: source.from,
                alignment,
            }
        });
        let written: Vec<FlowEnd> = self
            .params
            .iter()
            .enumerate()
            .filter(|(_, ty)| ty.as_ref().is_none_or(writes_through))
            .map(|(index, _)| FlowEnd::Param(index))
            .chain(self.captures_written.then_some(FlowEnd::Captures))
            .collect();
        let into_written = written.iter().flat_map(|to| {
            inputs.iter().map(|from| Flow {
                to: *to,
                from: *from,
                alignment: Alignment::Any,
            })
        });
        Flows::of(into_result.chain(into_written))
    }
}

struct Walk<'w, 'c, 'a, 's, 'src, 'f, S> {
    checker: &'w TypeChecker<'c, 's, 'src, S>,
    frame: &'f Frame,
    scopes: &'f Scopes,
    /// Everything the locals held after the previous walk: what a read
    /// through a reference may find in a local it names.
    stored: Origin,
    state: &'a mut Origins,
}

impl<S> Walk<'_, '_, '_, '_, '_, '_, S>
where
    S: Slot,
{
    fn ty(&self, id: AstId) -> Option<Ty> {
        self.checker
            .type_map
            .get(&id)
            .and_then(|ty| self.checker.closed_ty(ty))
    }

    fn holds(&self, id: AstId) -> bool {
        self.ty(id).is_none_or(|ty| positions(&ty) > 0)
    }

    /// What reading through storage may find beyond the value read: what a
    /// local holds, and what was written through a reference.
    fn through_storage(&self) -> Origin {
        any(&self.stored)
            .union(&any(&self.state.written))
            .cloned()
            .collect()
    }

    fn resolved(&self, id: AstId) -> Option<Resolved> {
        self.checker.resolved.0.get(&id).copied()
    }

    /// A binding the frame's lambda does not bind is one it captured. What
    /// the binding was assigned in the frame is joined, and so is what was
    /// written through a reference, which may have been into it.
    fn read_binder(&self, binder: AstId) -> Origin {
        let bound_here = self.scopes.owner.get(&binder).copied().flatten() == self.frame.lambda;
        let mut origin = match self.frame.params.get(&binder) {
            Some(index) => one(FlowEnd::Param(*index), Alignment::Aligned),
            None if bound_here => Origin::new(),
            None => one(FlowEnd::Captures, Alignment::Any),
        };
        if let Some(assigned) = self.state.locals.get(&binder) {
            origin.extend(assigned.iter().cloned());
        }
        origin.extend(any(&self.state.written));
        origin
    }

    /// A `$` input read in the whole body is its parameter, and read in a
    /// lambda one of the lambda's captures. A name the body declares no
    /// input for was refused where it was read.
    fn read_input(&self, name: Astr) -> Origin {
        let mut origin = match self.frame.lambda {
            Some(_) => one(FlowEnd::Captures, Alignment::Any),
            None => self
                .frame
                .inputs
                .get(&name)
                .map(|index| one(FlowEnd::Param(*index), Alignment::Aligned))
                .unwrap_or_default(),
        };
        origin.extend(any(&self.state.written));
        origin
    }

    /// Every input of the frame, and everything stored: what a value the
    /// walk cannot place may hold.
    fn every_input(&self) -> Origin {
        let params = self
            .frame
            .params
            .values()
            .chain(self.frame.inputs.values())
            .map(|index| FlowEnd::Param(*index));
        let captures = self.frame.lambda.map(|_| FlowEnd::Captures);
        params
            .chain(captures)
            .map(|from| Source {
                from,
                alignment: Alignment::Any,
            })
            .chain(self.through_storage())
            .collect()
    }

    fn stmts(&mut self, stmts: &[Stmt<S>]) {
        for stmt in stmts {
            self.stmt(stmt);
        }
    }

    fn stmt(&mut self, stmt: &Stmt<S>) {
        match stmt {
            Stmt::Store { place, expr, .. } => {
                let value = any(&self.value(expr));
                if let Some(binder) = self.place(place) {
                    self.state
                        .locals
                        .entry(binder)
                        .or_default()
                        .extend(value.clone());
                }
                self.state.written.extend(value);
            }
            Stmt::DerefStore { target, expr, .. } => {
                self.value(target);
                let value = any(&self.value(expr));
                self.state.written.extend(value);
            }
            Stmt::Expr(expr) | Stmt::Append { expr, .. } => {
                self.value(expr);
            }
            Stmt::LetBind { binder, expr, .. } => {
                let value = self.value(expr);
                self.state
                    .locals
                    .entry(binder.id)
                    .or_default()
                    .extend(value);
            }
            Stmt::LetUninit { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
            Stmt::Assign { id, expr, .. } => {
                let value = self.value(expr);
                if let Some(Resolved::Local(binder)) = self.resolved(*id) {
                    self.state.locals.entry(binder).or_default().extend(value);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.value(cond);
                self.stmts(body);
            }
            Stmt::For {
                binder, head, body, ..
            } => {
                let element = match head {
                    ForHead::Value(source) => {
                        let source = any(&self.value(source));
                        source.union(&self.through_storage()).cloned().collect()
                    }
                    ForHead::Range { lo, hi } => {
                        self.value(lo);
                        self.value(hi);
                        Origin::new()
                    }
                };
                self.state
                    .locals
                    .entry(binder.id)
                    .or_default()
                    .extend(element);
                self.stmts(body);
            }
            Stmt::WhileLet {
                pattern,
                source,
                body,
                ..
            } => {
                let bound = self.matched(source);
                self.bind(pattern, &bound);
                self.stmts(body);
            }
            Stmt::Anyorder { body, .. } => self.stmts(body),
            Stmt::Error(_) => {}
        }
    }

    /// The local a store writes into, if its root is one; the index
    /// expressions on the way are walked.
    fn place(&mut self, place: &Place<S>) -> Option<AstId> {
        match place {
            Place::Field { object, .. } => self.place(object),
            Place::Base(PlaceBase::Root { id, root, .. }) => match root {
                Root::Local(_) => match self.resolved(*id)? {
                    Resolved::Local(binder) => Some(binder),
                    Resolved::Function(_) | Resolved::Context(_) | Resolved::Input(_) => None,
                },
                Root::ExternParam(_) | Root::Context(_) => None,
            },
            Place::Base(PlaceBase::Element {
                container, index, ..
            }) => {
                self.value(index);
                self.value(container.expr());
                None
            }
        }
    }

    /// What a pattern over `source` binds from: a binding may be a
    /// reference into the storage the source names.
    fn matched(&mut self, source: &Expr<S>) -> Origin {
        let source = any(&self.place_value(source));
        source.union(&self.through_storage()).cloned().collect()
    }

    /// What a reference to `place` may hold: the loans its base holds, the
    /// one on the storage it names among them, whatever the type of the
    /// part the place reaches. A step through storage takes what storage
    /// holds.
    fn place_value(&mut self, place: &Expr<S>) -> Origin {
        match place {
            Expr::Paren { inner, .. } => self.place_value(inner),
            Expr::FieldAccess { object, .. }
            | Expr::UnaryOp {
                op: UnaryOp::Deref,
                operand: object,
                ..
            } => self
                .place_value(object)
                .union(&self.through_storage())
                .cloned()
                .collect(),
            Expr::Index { object, index, .. } => {
                self.value(index);
                self.place_value(object)
                    .union(&self.through_storage())
                    .cloned()
                    .collect()
            }
            Expr::Ident { .. } => self.value_unfiltered(place),
            other => self.value(other),
        }
    }

    /// A call's receiver, lent from its place where the checker lends it.
    /// A receiver whose passing is settled only as the body freezes is
    /// read as lent, which holds what passing it by value holds.
    fn receiver(&mut self, receiver: &Expr<S>) -> Origin {
        match self.checker.passing.get(&receiver.id()) {
            Some(super::Passing::Value | super::Passing::AsIs) => self.value(receiver),
            Some(super::Passing::Lent(_)) | None => any(&self.place_value(receiver)),
        }
    }

    fn bind(&mut self, pattern: &Pattern<S>, from: &Origin) {
        match pattern {
            Pattern::Binding { id, .. } => {
                self.state
                    .locals
                    .entry(*id)
                    .or_default()
                    .extend(from.clone());
            }
            Pattern::List { head, tail, .. } => {
                for part in head.iter().chain(tail) {
                    self.bind(part, from);
                }
            }
            Pattern::Object { fields, .. } => {
                for field in fields {
                    self.bind(&field.pattern, from);
                }
            }
            Pattern::Tuple { elements, .. } => {
                for element in elements {
                    if let TuplePatternElem::Pattern(part) = element {
                        self.bind(part, from);
                    }
                }
            }
            Pattern::Variant { payload, .. } => {
                if let Some(payload) = payload {
                    self.bind(payload, from);
                }
            }
            Pattern::ContextBind { .. }
            | Pattern::Literal { .. }
            | Pattern::Wildcard { .. }
            | Pattern::Error(_) => {}
        }
    }

    fn arm(&mut self, arm: &MatchExprArm<S>, bound: &Origin) -> Origin {
        self.bind(&arm.pattern, bound);
        self.stmts(&arm.body);
        arm.tail
            .as_ref()
            .map_or_else(Origin::new, |tail| self.value(tail))
    }

    fn branch(&mut self, stmts: &[Stmt<S>], tail: Option<&Expr<S>>) -> Origin {
        self.stmts(stmts);
        tail.map_or_else(Origin::new, |tail| self.value(tail))
    }

    fn else_branch(&mut self, branch: Option<&ElseBranch<S>>) -> Origin {
        match branch {
            None => Origin::new(),
            Some(ElseBranch::ElseIf(expr)) => self.value(expr),
            Some(ElseBranch::Else { body, tail, .. }) => self.branch(body, tail.as_deref()),
        }
    }

    /// What `expr`'s value may hold. Every sub-expression is walked,
    /// whatever the value's type, so a write or a call inside it counts.
    fn value(&mut self, expr: &Expr<S>) -> Origin {
        let origin = self.value_unfiltered(expr);
        match self.holds(expr.id()) {
            true => origin,
            false => Origin::new(),
        }
    }

    fn value_unfiltered(&mut self, expr: &Expr<S>) -> Origin {
        match expr {
            Expr::Ident {
                id, name, ref_kind, ..
            } => match ref_kind {
                RefKind::ExternParam => self.read_input(name.name),
                RefKind::Value => match self.resolved(*id) {
                    Some(Resolved::Local(binder)) => self.read_binder(binder),
                    Some(Resolved::Input(input)) => self.read_input(input),
                    // A declared function or a context holds no loan, and a
                    // name the checker resolved to nothing is a declared
                    // function read as a value or a refused name.
                    Some(Resolved::Function(_) | Resolved::Context(_)) | None => Origin::new(),
                },
            },
            Expr::Literal { .. } | Expr::ContextRef { .. } | Expr::Error(_) => Origin::new(),
            Expr::Paren { inner, .. } | Expr::Cast { expr: inner, .. } => self.value(inner),
            Expr::Borrow { place, .. } => any(&self.place_value(place)),
            Expr::UnaryOp {
                op: UnaryOp::Deref,
                operand,
                ..
            } => {
                let pointer = any(&self.value(operand));
                pointer.union(&self.through_storage()).cloned().collect()
            }
            Expr::UnaryOp { operand, .. } => {
                let operand = any(&self.value(operand));
                operand.union(&self.through_storage()).cloned().collect()
            }
            Expr::BinaryOp { left, right, .. } => {
                let mut both = any(&self.value(left));
                both.extend(any(&self.value(right)));
                both.union(&self.through_storage()).cloned().collect()
            }
            Expr::FieldAccess { object, .. } => {
                let object = any(&self.value(object));
                object.union(&self.through_storage()).cloned().collect()
            }
            Expr::Index { object, index, .. } => {
                let mut both = any(&self.value(object));
                both.extend(any(&self.value(index)));
                both.union(&self.through_storage()).cloned().collect()
            }
            Expr::FuncCall { func, args, .. } => self.func_call(func, None, args),
            Expr::MethodCall {
                callee_id,
                receiver,
                args,
                ..
            } => {
                let callee = self.named_callee(*callee_id);
                let signature = self.callee_signature(&callee);
                let receiver = self.receiver(receiver);
                let args: Vec<Origin> = std::iter::once(receiver)
                    .chain(
                        args.iter()
                            .enumerate()
                            .map(|(index, arg)| self.argument(&signature, 1 + index, arg)),
                    )
                    .collect();
                self.call(callee, args)
            }
            Expr::Pipe { left, right, .. } => match right.as_ref() {
                Expr::FuncCall { func, args, .. } => self.func_call(func, Some(left), args),
                Expr::Ident {
                    ref_kind: RefKind::Value,
                    ..
                } => self.func_call(right, Some(left), &[]),
                other => {
                    let callee = Callee {
                        ty: self.checker.type_map.get(&other.id()).cloned(),
                        captures: self.value(other),
                    };
                    let signature = self.callee_signature(&callee);
                    let left = self.argument(&signature, 0, left);
                    self.call(callee, vec![left])
                }
            },
            Expr::Lambda { body, .. } => self.captured(body),
            Expr::List { head, tail, .. } => {
                let mut all = Origin::new();
                for element in head.iter().chain(tail) {
                    all.extend(any(&self.value(element)));
                }
                all
            }
            Expr::Group { elements, .. } => {
                let mut all = Origin::new();
                for element in elements {
                    all.extend(any(&self.value(element)));
                }
                all
            }
            Expr::Object { fields, .. } => {
                let mut all = Origin::new();
                for ObjectExprField { value, .. } in fields {
                    all.extend(any(&self.value(value)));
                }
                all
            }
            Expr::Tuple { elements, .. } => {
                let mut all = Origin::new();
                for element in elements {
                    if let TupleElem::Expr(element) = element {
                        all.extend(any(&self.value(element)));
                    }
                }
                all
            }
            Expr::Variant { payload, .. } => payload
                .as_ref()
                .map_or_else(Origin::new, |payload| any(&self.value(payload))),
            Expr::Block { stmts, tail, .. } => self.branch(stmts, Some(tail)),
            Expr::Try { inner, .. } => {
                let inner = any(&self.value(inner));
                self.state.result.extend(inner.clone());
                inner
            }
            Expr::Return { value, .. } => {
                let value = self.value(value);
                self.state.result.extend(value);
                Origin::new()
            }
            Expr::If {
                cond,
                then_body,
                then_tail,
                else_branch,
                ..
            } => {
                self.value(cond);
                let mut both = self.branch(then_body, then_tail.as_deref());
                both.extend(self.else_branch(else_branch.as_deref()));
                both
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                let bound = self.matched(scrutinee);
                let mut all = Origin::new();
                for arm in arms {
                    all.extend(self.arm(arm, &bound));
                }
                all
            }
            Expr::IfLet {
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                ..
            } => {
                let bound = self.matched(source);
                self.bind(pattern, &bound);
                let mut both = self.branch(then_body, then_tail.as_deref());
                both.extend(self.else_branch(else_branch.as_deref()));
                both
            }
        }
    }

    fn func_call(&mut self, func: &Expr<S>, piped: Option<&Expr<S>>, args: &[Expr<S>]) -> Origin {
        let callee = match self.checker.calls.get(&func.id()) {
            Some(_) => self.named_callee(func.id()),
            None => Callee {
                ty: self.checker.type_map.get(&func.id()).cloned(),
                captures: self.value(func),
            },
        };
        let signature = self.callee_signature(&callee);
        let args: Vec<Origin> = piped
            .into_iter()
            .chain(args)
            .enumerate()
            .map(|(index, arg)| self.argument(&signature, index, arg))
            .collect();
        self.call(callee, args)
    }

    /// An argument as the callee's parameter at `index` takes it. A value
    /// of no position handed to a parameter that has one was converted on
    /// the way, into a reference to where the value lives (a `String` to a
    /// `&str`), and holds what a reference to its place holds.
    fn argument(&mut self, signature: &CalleeSignature, index: usize, arg: &Expr<S>) -> Origin {
        let value = self.value(arg);
        let param_holds = signature
            .params
            .get(index)
            .is_none_or(|param| param.as_ref().is_none_or(|ty| positions(ty) > 0));
        match param_holds && !self.holds(arg.id()) {
            true => any(&self.place_value(arg)),
            false => value,
        }
    }

    /// A callee named at `id`: a declaration, which captures nothing, or a
    /// local binding holding a lambda, whose captures are what the binding
    /// holds.
    fn named_callee(&self, id: AstId) -> Callee {
        let binding = match self.checker.calls.get(&id) {
            Some(CallChoice::Binding) => true,
            Some(CallChoice::Decided(decision)) => matches!(
                self.checker.solver.answer(*decision),
                Some(Answer::Signature {
                    settled: SettledSignature::Local,
                    ..
                })
            ),
            // A declared callee captures nothing.
            Some(_) | None => false,
        };
        let captures = match binding {
            true => match self.checker.candidate_binder_of.get(&id) {
                Some(binder) => any(&self.read_binder(*binder)),
                None => self.every_input(),
            },
            false => Origin::new(),
        };
        Callee {
            ty: self.checker.type_map.get(&id).cloned(),
            captures,
        }
    }

    /// The flows and parameters of what a call reaches. A callee the solve
    /// left without a function type was refused; the union stands for it.
    fn callee_signature(&self, callee: &Callee) -> CalleeSignature {
        let resolved = callee
            .ty
            .as_ref()
            .map(|ty| self.checker.solver.resolve_ty(ty));
        let function = match resolved {
            Some(TyTerm::Ref(_, inner)) => inner.ty().into_owned(),
            Some(other) => other,
            None => TyTerm::Error(crate::ty::ErrorToken::new()),
        };
        match &function {
            TyTerm::Fn { flows, params, .. } => CalleeSignature {
                flows: self.checker.solver.flows_of(flows),
                params: params
                    .iter()
                    .map(|param| self.checker.closed_ty(&param.ty))
                    .collect(),
                captures_written: self
                    .checker
                    .closed_ty(&function)
                    .is_none_or(|ty| writes_through(&ty)),
            },
            _ => CalleeSignature {
                flows: Flows::Every,
                params: Vec::new(),
                captures_written: true,
            },
        }
    }

    /// What input `end` of a call holds: the argument, or the callee's
    /// captures, and what either reads through storage. A flow naming a
    /// parameter the call has no argument for is a refused call's; every
    /// argument stands for it.
    fn call_input(
        &self,
        callee: &Callee,
        signature: &CalleeSignature,
        args: &[Origin],
        end: FlowEnd,
    ) -> Origin {
        let reads = |index: usize| {
            signature
                .params
                .get(index)
                .and_then(Option::as_ref)
                .is_none_or(reads_through)
        };
        match end {
            FlowEnd::Param(index) => match args.get(index) {
                Some(arg) if reads(index) => self.with_storage(arg.clone()),
                Some(arg) => arg.clone(),
                None => self.with_storage(args.iter().flat_map(any).collect()),
            },
            FlowEnd::Captures => self.with_storage(any(&callee.captures)),
            FlowEnd::Result => unreachable!("a call's result is no input"),
        }
    }

    fn with_storage(&self, origin: Origin) -> Origin {
        origin.union(&self.through_storage()).cloned().collect()
    }

    /// A call's result by its callee's flows, and what the callee may write.
    fn call(&mut self, callee: Callee, args: Vec<Origin>) -> Origin {
        let signature = self.callee_signature(&callee);
        let flows = &signature.flows;
        let param_ty = |index: usize| signature.params.get(index).cloned().flatten();
        let input = |walk: &Self, end: FlowEnd| walk.call_input(&callee, &signature, &args, end);
        let arity = args.len();
        let mut result = Origin::new();
        for Source { from, alignment } in flows.into_end(FlowEnd::Result, arity) {
            for source in input(self, from) {
                let both_aligned =
                    alignment == Alignment::Aligned && source.alignment == Alignment::Aligned;
                result.insert(Source {
                    from: source.from,
                    alignment: match both_aligned {
                        true => Alignment::Aligned,
                        false => Alignment::Any,
                    },
                });
            }
        }
        let written_ends = (0..arity)
            .filter(|index| param_ty(*index).is_none_or(|ty| writes_through(&ty)))
            .map(FlowEnd::Param)
            .chain(signature.captures_written.then_some(FlowEnd::Captures));
        let mut written = Origin::new();
        for to in written_ends {
            for Source { from, .. } in flows.into_end(to, arity) {
                written.extend(any(&input(self, from)));
            }
        }
        self.state.written.extend(written);
        result
    }

    /// A lambda nested in the body is a value holding what it captures:
    /// every name it reads from outside itself, as the body holds it.
    fn captured(&mut self, body: &Expr<S>) -> Origin {
        let mut reads = Vec::new();
        reads_in_expr(body, &mut reads);
        let mut all = Origin::new();
        for read in reads {
            let origin = match read {
                Read::Binder(binder) => {
                    match self.scopes.outside_of_nested(binder, self.frame.lambda) {
                        true => self.read_binder(binder),
                        false => continue,
                    }
                }
                Read::Input(name) => self.read_input(name),
            };
            all.extend(any(&origin));
        }
        match all.is_empty() {
            true => all,
            false => all.union(&self.through_storage()).cloned().collect(),
        }
    }
}

/// A callee as the call reaches it: its type, and what its value captured.
struct Callee {
    ty: Option<InferTy>,
    captures: Origin,
}

struct CalleeSignature {
    flows: Flows,
    /// `None` where a parameter's type is not closed.
    params: Vec<Option<Ty>>,
    captures_written: bool,
}

enum Read {
    Binder(AstId),
    Input(Astr),
}

// -- Scopes ----------------------------------------------------------

fn scope_stmts<'e, S>(
    stmts: &'e [Stmt<S>],
    at: Option<AstId>,
    scopes: &mut Scopes,
    lambdas: &mut Vec<LambdaSite<'e, S>>,
) {
    for stmt in stmts {
        scope_stmt(stmt, at, scopes, lambdas);
    }
}

fn scope_stmt<'e, S>(
    stmt: &'e Stmt<S>,
    at: Option<AstId>,
    scopes: &mut Scopes,
    lambdas: &mut Vec<LambdaSite<'e, S>>,
) {
    match stmt {
        Stmt::Store { place, expr, .. } => {
            scope_place(place, at, scopes, lambdas);
            scope_expr(expr, at, scopes, lambdas);
        }
        Stmt::DerefStore { target, expr, .. } => {
            scope_expr(target, at, scopes, lambdas);
            scope_expr(expr, at, scopes, lambdas);
        }
        Stmt::Expr(expr) | Stmt::Append { expr, .. } | Stmt::Assign { expr, .. } => {
            scope_expr(expr, at, scopes, lambdas)
        }
        Stmt::LetBind { binder, expr, .. } => {
            scopes.owner.insert(binder.id, at);
            scope_expr(expr, at, scopes, lambdas);
        }
        Stmt::LetUninit { binder, .. } => {
            scopes.owner.insert(binder.id, at);
        }
        Stmt::While { cond, body, .. } => {
            scope_expr(cond, at, scopes, lambdas);
            scope_stmts(body, at, scopes, lambdas);
        }
        Stmt::For {
            binder, head, body, ..
        } => {
            scopes.owner.insert(binder.id, at);
            match head {
                ForHead::Value(source) => scope_expr(source, at, scopes, lambdas),
                ForHead::Range { lo, hi } => {
                    scope_expr(lo, at, scopes, lambdas);
                    scope_expr(hi, at, scopes, lambdas);
                }
            }
            scope_stmts(body, at, scopes, lambdas);
        }
        Stmt::WhileLet {
            pattern,
            source,
            body,
            ..
        } => {
            scope_pattern(pattern, at, scopes);
            scope_expr(source, at, scopes, lambdas);
            scope_stmts(body, at, scopes, lambdas);
        }
        Stmt::Anyorder { body, .. } => scope_stmts(body, at, scopes, lambdas),
        Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Error(_) => {}
    }
}

fn scope_place<'e, S>(
    place: &'e Place<S>,
    at: Option<AstId>,
    scopes: &mut Scopes,
    lambdas: &mut Vec<LambdaSite<'e, S>>,
) {
    match place {
        Place::Field { object, .. } => scope_place(object, at, scopes, lambdas),
        Place::Base(PlaceBase::Root { .. }) => {}
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            scope_expr(container.expr(), at, scopes, lambdas);
            scope_expr(index, at, scopes, lambdas);
        }
    }
}

fn scope_pattern<S>(pattern: &Pattern<S>, at: Option<AstId>, scopes: &mut Scopes) {
    match pattern {
        Pattern::Binding { id, .. } => {
            scopes.owner.insert(*id, at);
        }
        Pattern::List { head, tail, .. } => {
            for part in head.iter().chain(tail) {
                scope_pattern(part, at, scopes);
            }
        }
        Pattern::Object { fields, .. } => {
            for field in fields {
                scope_pattern(&field.pattern, at, scopes);
            }
        }
        Pattern::Tuple { elements, .. } => {
            for element in elements {
                if let TuplePatternElem::Pattern(part) = element {
                    scope_pattern(part, at, scopes);
                }
            }
        }
        Pattern::Variant { payload, .. } => {
            if let Some(payload) = payload {
                scope_pattern(payload, at, scopes);
            }
        }
        Pattern::ContextBind { .. }
        | Pattern::Literal { .. }
        | Pattern::Wildcard { .. }
        | Pattern::Error(_) => {}
    }
}

fn scope_else<'e, S>(
    branch: &'e ElseBranch<S>,
    at: Option<AstId>,
    scopes: &mut Scopes,
    lambdas: &mut Vec<LambdaSite<'e, S>>,
) {
    match branch {
        ElseBranch::ElseIf(expr) => scope_expr(expr, at, scopes, lambdas),
        ElseBranch::Else { body, tail, .. } => {
            scope_stmts(body, at, scopes, lambdas);
            if let Some(tail) = tail {
                scope_expr(tail, at, scopes, lambdas);
            }
        }
    }
}

fn scope_expr<'e, S>(
    expr: &'e Expr<S>,
    at: Option<AstId>,
    scopes: &mut Scopes,
    lambdas: &mut Vec<LambdaSite<'e, S>>,
) {
    match expr {
        Expr::Lambda {
            id, params, body, ..
        } => {
            scopes.parent.insert(*id, at);
            for param in params {
                scopes.owner.insert(param.id, Some(*id));
            }
            scope_expr(body, Some(*id), scopes, lambdas);
            lambdas.push(LambdaSite {
                id: *id,
                params: params.iter().map(|param| param.id).collect(),
                body,
            });
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            scope_expr(scrutinee, at, scopes, lambdas);
            for arm in arms {
                scope_pattern(&arm.pattern, at, scopes);
                scope_stmts(&arm.body, at, scopes, lambdas);
                if let Some(tail) = &arm.tail {
                    scope_expr(tail, at, scopes, lambdas);
                }
            }
        }
        Expr::IfLet {
            pattern,
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            scope_pattern(pattern, at, scopes);
            scope_expr(source, at, scopes, lambdas);
            scope_stmts(then_body, at, scopes, lambdas);
            if let Some(tail) = then_tail {
                scope_expr(tail, at, scopes, lambdas);
            }
            if let Some(branch) = else_branch {
                scope_else(branch, at, scopes, lambdas);
            }
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            scope_expr(cond, at, scopes, lambdas);
            scope_stmts(then_body, at, scopes, lambdas);
            if let Some(tail) = then_tail {
                scope_expr(tail, at, scopes, lambdas);
            }
            if let Some(branch) = else_branch {
                scope_else(branch, at, scopes, lambdas);
            }
        }
        Expr::Block { stmts, tail, .. } => {
            scope_stmts(stmts, at, scopes, lambdas);
            scope_expr(tail, at, scopes, lambdas);
        }
        other => {
            for child in children(other) {
                scope_expr(child, at, scopes, lambdas);
            }
        }
    }
}

/// The sub-expressions of an expression that binds nothing and holds no
/// statement.
fn children<S>(expr: &Expr<S>) -> Vec<&Expr<S>> {
    match expr {
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } | Expr::Error(_) => {
            Vec::new()
        }
        Expr::BinaryOp { left, right, .. } => vec![left, right],
        Expr::UnaryOp { operand, .. } => vec![operand],
        Expr::FieldAccess { object, .. } => vec![object],
        Expr::Index { object, index, .. } => vec![object, index],
        Expr::FuncCall { func, args, .. } => std::iter::once(&**func).chain(args).collect(),
        Expr::MethodCall { receiver, args, .. } => {
            std::iter::once(&**receiver).chain(args).collect()
        }
        Expr::Pipe { left, right, .. } => vec![left, right],
        Expr::Paren { inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::Try { inner, .. }
        | Expr::Return { value: inner, .. }
        | Expr::Borrow { place: inner, .. } => vec![inner],
        Expr::List { head, tail, .. } => head.iter().chain(tail).collect(),
        Expr::Group { elements, .. } => elements.iter().collect(),
        Expr::Object { fields, .. } => fields.iter().map(|field| &field.value).collect(),
        Expr::Tuple { elements, .. } => elements
            .iter()
            .filter_map(|element| match element {
                TupleElem::Expr(expr) => Some(expr),
                TupleElem::Wildcard(_) => None,
            })
            .collect(),
        Expr::Variant { payload, .. } => payload.iter().map(|payload| &**payload).collect(),
        Expr::Lambda { body, .. } => vec![body],
        Expr::Block { tail, .. } => vec![tail],
        Expr::If {
            cond, then_tail, ..
        } => std::iter::once(&**cond)
            .chain(then_tail.as_deref())
            .collect(),
        Expr::Match { scrutinee, .. } => vec![scrutinee],
        Expr::IfLet {
            source, then_tail, ..
        } => std::iter::once(&**source)
            .chain(then_tail.as_deref())
            .collect(),
    }
}

// -- Names a lambda reads ----------------------------------------------

fn reads_in_stmts<S>(stmts: &[Stmt<S>], reads: &mut Vec<Read>) {
    for stmt in stmts {
        match stmt {
            Stmt::Store { place, expr, .. } => {
                reads_in_place(place, reads);
                reads_in_expr(expr, reads);
            }
            Stmt::DerefStore { target, expr, .. } => {
                reads_in_expr(target, reads);
                reads_in_expr(expr, reads);
            }
            Stmt::Expr(expr)
            | Stmt::Append { expr, .. }
            | Stmt::Assign { expr, .. }
            | Stmt::LetBind { expr, .. } => reads_in_expr(expr, reads),
            Stmt::While { cond, body, .. } => {
                reads_in_expr(cond, reads);
                reads_in_stmts(body, reads);
            }
            Stmt::For { head, body, .. } => {
                match head {
                    ForHead::Value(source) => reads_in_expr(source, reads),
                    ForHead::Range { lo, hi } => {
                        reads_in_expr(lo, reads);
                        reads_in_expr(hi, reads);
                    }
                }
                reads_in_stmts(body, reads);
            }
            Stmt::WhileLet { source, body, .. } => {
                reads_in_expr(source, reads);
                reads_in_stmts(body, reads);
            }
            Stmt::Anyorder { body, .. } => reads_in_stmts(body, reads),
            Stmt::LetUninit { .. }
            | Stmt::Break { .. }
            | Stmt::Continue { .. }
            | Stmt::Error(_) => {}
        }
    }
}

fn reads_in_place<S>(place: &Place<S>, reads: &mut Vec<Read>) {
    match place {
        Place::Field { object, .. } => reads_in_place(object, reads),
        Place::Base(PlaceBase::Root { root, .. }) => {
            if let Root::ExternParam(name) = root {
                reads.push(Read::Input(*name));
            }
        }
        Place::Base(PlaceBase::Element {
            container, index, ..
        }) => {
            reads_in_expr(container.expr(), reads);
            reads_in_expr(index, reads);
        }
    }
}

/// Every name `expr` reads, a nested lambda's included. A name read by
/// assignment is the assigned binding's, which a lambda may not assign when
/// it captured it, so only reads are gathered.
fn reads_in_expr<S>(expr: &Expr<S>, reads: &mut Vec<Read>) {
    match expr {
        Expr::Ident {
            id, name, ref_kind, ..
        } => reads.push(match ref_kind {
            RefKind::Value => Read::Binder(*id),
            RefKind::ExternParam => Read::Input(name.name),
        }),
        Expr::Block { stmts, tail, .. } => {
            reads_in_stmts(stmts, reads);
            reads_in_expr(tail, reads);
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        }
        | Expr::IfLet {
            source: cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            reads_in_expr(cond, reads);
            reads_in_stmts(then_body, reads);
            if let Some(tail) = then_tail {
                reads_in_expr(tail, reads);
            }
            match else_branch.as_deref() {
                Some(ElseBranch::ElseIf(expr)) => reads_in_expr(expr, reads),
                Some(ElseBranch::Else { body, tail, .. }) => {
                    reads_in_stmts(body, reads);
                    if let Some(tail) = tail {
                        reads_in_expr(tail, reads);
                    }
                }
                None => {}
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            reads_in_expr(scrutinee, reads);
            for arm in arms {
                reads_in_stmts(&arm.body, reads);
                if let Some(tail) = &arm.tail {
                    reads_in_expr(tail, reads);
                }
            }
        }
        other => {
            for child in children(other) {
                reads_in_expr(child, reads);
            }
        }
    }
}
