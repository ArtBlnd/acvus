use acvus_ast::{
    AstId, BinOp, ElseBranch, Expr, IndentModifier, Literal, MatchBlock, Node, ObjectExprField,
    ObjectPatternField, Pattern, RefKind, Script, Span, Stmt, Template, TupleElem,
    TuplePatternElem, UnaryOp,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet};

use crate::graph::QualifiedRef;
use crate::ir::{
    Callee, CastKind, ExternCast, ExternInstance, IndexAccess, IndexMode, Inst, InstKind, Label,
    MirBody, MirModule, OrderEdge, PathSeg, RefTarget, ValOrigin, ValueId,
};
use crate::ty::{Effect, Mutability, Task, Ty, TypeArg};
use crate::typeck::TypeResolution;

pub struct Lowerer<'a> {
    body: MirBody,
    /// Interner for string interning.
    interner: &'a Interner,
    /// Stack of scopes: variable name -> its binding. A binding introduced
    /// while a name is already bound shadows it: a fresh slot, and the outer
    /// binding is untouched and visible again when the scope ends.
    scopes: Vec<FxHashMap<Astr, Local>>,
    /// Frozen type resolution from typeck. Contains type_map, coercion_map, direct_calls.
    resolution: Freeze<TypeResolution>,
    /// Coercion map from type checker (expr AstId -> CastKind).
    coercion_lookup: FxHashMap<AstId, CastKind>,
    /// Closures produced during lowering.
    closures: FxHashMap<Label, MirBody>,
    /// Global closure label counter - shared across nesting levels to prevent
    /// label collisions when nested closures each allocate from a sub-body.
    closure_label_count: u32,
    /// The storage slot holding the current `Order` of the body being lowered
    /// (RFC-0007). `None` for a body whose effect is Pure. The SSA pass
    /// promotes the slot, so branches and loops join orders through phis.
    order_slot: Option<ValueId>,
    /// The innermost `anyorder` block being lowered, if any.
    anyorder: Option<AnyorderScope>,
    context_slots: BTreeMap<QualifiedRef, ValueId>,
    /// The places the calls being lowered have cast for their callees,
    /// innermost last (RFC-0041).
    holds: Vec<Held>,
}

/// A place cast for a call: until the call restores it, it holds `ty`.
struct Held {
    target: RefTarget,
    path: Vec<PathSeg>,
    ty: Ty,
}

/// An `anyorder` block while its body is lowered (RFC-0007): every
/// effectful call inside takes `entry`, and `acc` accumulates a merge of
/// what they yielded. A loop inside stores to `acc` on every iteration, so
/// the SSA pass gives it a loop phi.
#[derive(Clone, Copy)]
struct AnyorderScope {
    entry: ValueId,
    acc: ValueId,
}

/// A place a program names: where it is, and the type of what it holds.
#[derive(Clone)]
struct Place {
    target: RefTarget,
    path: Vec<PathSeg>,
    ty: Ty,
}

/// The arguments of one call, and the places the call leaves to restore.
struct CallArgs {
    values: Vec<ValueId>,
    restores: Vec<PlaceRestore>,
}

/// A place cast into the callee's representation for the call: `back`
/// runs on the value it holds afterwards, and the result is stored back.
struct PlaceRestore {
    span: Span,
    place: Place,
    back: ExternCast,
}

/// A place lent as the call argument at `id`.
struct Lent {
    id: AstId,
    span: Span,
    place: Place,
    mutability: Mutability,
}

struct PlacePart {
    path: Vec<PathSeg>,
    ty: Ty,
    pattern: Pattern,
}

struct RefPart {
    reference: ValueId,
    pattern: Pattern,
}

/// What a pattern is applied to (RFC-0024).
#[derive(Clone)]
enum PatSrc {
    Value(ValueId),
    Place {
        target: RefTarget,
        path: Vec<PathSeg>,
        ty: Ty,
    },
}

fn fields(path: &[Astr]) -> Vec<PathSeg> {
    path.iter().copied().map(PathSeg::Field).collect()
}

/// A context place lent to a call through a temporary local.
/// A local variable binding: its type and the storage slot it owns.
/// What a call's summary says it may touch; a type with no effect says
/// everything.
enum TouchedContexts {
    Every,
    These(BTreeSet<QualifiedRef>),
}

struct Local {
    ty: Ty,
    slot: ValueId,
}

/// A connective whose left operand can decide the result alone (RFC-0020).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShortCircuit {
    And,
    Or,
}

impl ShortCircuit {
    /// The result the left operand decides alone, which is also the value
    /// of the left operand that decides it.
    fn decided(self) -> bool {
        match self {
            ShortCircuit::And => false,
            ShortCircuit::Or => true,
        }
    }
}

/// The connective an operator is, for the two operators that short-circuit.
fn short_circuit_of(op: BinOp) -> Option<ShortCircuit> {
    match op {
        BinOp::And => Some(ShortCircuit::And),
        BinOp::Or => Some(ShortCircuit::Or),
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Mod
        | BinOp::Eq
        | BinOp::Neq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Lte
        | BinOp::Gte
        | BinOp::Xor
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::Shl
        | BinOp::Shr => None,
    }
}

/// A pattern that matches every value of its type: a binding, a context
/// bind, a tuple of such, or an object of such. The type checker has
/// already required every key an object pattern names to exist on the
/// source type, so the pattern cannot fail at run time.
fn pattern_is_irrefutable(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Binding { .. } | Pattern::ContextBind { .. } => true,
        Pattern::Tuple { elements, .. } => elements.iter().all(|e| match e {
            TuplePatternElem::Pattern(p) => pattern_is_irrefutable(p),
            TuplePatternElem::Wildcard(_) => true,
        }),
        Pattern::Object { fields, .. } => fields
            .iter()
            .all(|field| pattern_is_irrefutable(&field.pattern)),
        Pattern::Literal { .. } | Pattern::List { .. } | Pattern::Variant { .. } => false,
    }
}

/// Adjust indentation of a text string according to an `IndentModifier`.
/// All lines (including the first) are affected.
fn adjust_text_indent(text: &str, modifier: &IndentModifier) -> String {
    let mut result = String::with_capacity(text.len());
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            result.push('\n');
        }
        match modifier {
            IndentModifier::Decrease(n) => {
                let n = *n as usize;
                let spaces = line.len() - line.trim_start_matches(' ').len();
                let remove = spaces.min(n);
                result.push_str(&line[remove..]);
            }
            IndentModifier::Increase(n) => {
                let n = *n as usize;
                if !line.is_empty() {
                    for _ in 0..n {
                        result.push(' ');
                    }
                }
                result.push_str(line);
            }
        }
    }
    result
}

/// Recursively apply an indent modifier to all `Node::Text` nodes in a slice.
fn apply_indent_to_nodes(nodes: &[Node], modifier: &IndentModifier) -> Vec<Node> {
    nodes
        .iter()
        .map(|node| match node {
            Node::Text { value, span, .. } => Node::Text {
                id: acvus_ast::AstId::alloc(),
                value: adjust_text_indent(value, modifier),
                span: *span,
            },
            Node::MatchBlock(mb) => Node::MatchBlock(MatchBlock {
                id: acvus_ast::AstId::alloc(),
                arms: mb
                    .arms
                    .iter()
                    .map(|arm| acvus_ast::MatchArm {
                        id: acvus_ast::AstId::alloc(),
                        pattern: arm.pattern.clone(),
                        body: apply_indent_to_nodes(&arm.body, modifier),
                        tag_span: arm.tag_span,
                    })
                    .collect(),
                catch_all: mb.catch_all.as_ref().map(|ca| acvus_ast::CatchAll {
                    id: acvus_ast::AstId::alloc(),
                    body: apply_indent_to_nodes(&ca.body, modifier),
                    tag_span: ca.tag_span,
                }),
                source: mb.source.clone(),
                indent: mb.indent,
                span: mb.span,
            }),
            other => other.clone(),
        })
        .collect()
}

impl<'a> Lowerer<'a> {
    pub fn new(interner: &'a Interner, resolution: Freeze<TypeResolution>) -> Self {
        let coercion_lookup: FxHashMap<AstId, CastKind> =
            resolution.coercion_map.iter().cloned().collect();
        let initial_scope = FxHashMap::default();
        let mut body = MirBody::new();

        // Allocate param_regs for extern params (LLVM-style: params are SSA values).
        // param_regs[i] holds the initial value of extern_params[i].
        // SSA will use these as entry definitions instead of Ref+Load.
        for (name, ty) in &resolution.extern_params {
            let reg = body.val_factory.next();
            body.val_types.insert(reg, ty.clone());
            body.params.push((*name, reg));
        }
        body.task = resolution.effect.task;

        Self {
            body,
            interner,
            scopes: vec![initial_scope],
            resolution,
            coercion_lookup,
            closures: FxHashMap::default(),
            closure_label_count: 0,
            order_slot: None,
            anyorder: None,
            context_slots: BTreeMap::new(),
            holds: Vec::new(),
        }
    }

    pub fn lower_template(mut self, template: &Template) -> MirModule {
        let effect = self.resolution.effect.clone();
        self.enter_body_order(effect, template.span);
        self.enter_contexts(
            acvus_ast::direct_template_context_refs(template),
            template.span,
        );
        let result = self.lower_nodes(&template.body, template.span);
        self.emit_return(template.span, result);
        self.build_module()
    }

    pub fn lower_script(mut self, script: &Script) -> MirModule {
        let effect = self.resolution.effect.clone();
        self.enter_body_order(effect, script.span);
        self.enter_contexts(acvus_ast::direct_script_context_refs(script), script.span);
        for stmt in &script.stmts {
            self.lower_stmt(stmt);
        }
        if let Some(tail) = &script.tail {
            let val = self.lower_expr(tail);
            self.emit_return(script.span, val);
        }
        self.build_module()
    }

    /// Give the body being lowered its entry `Order` when its effect is not
    /// Pure: an order parameter, stored into a fresh slot that every
    /// effectful call reads and advances.
    fn enter_body_order(&mut self, effect: Effect, span: Span) {
        self.anyorder = None;
        if effect.is_pure() {
            self.order_slot = None;
            return;
        }
        let order_param = self.alloc_val();
        self.set_val_type(order_param, Ty::Order);
        self.body.order_param = Some(order_param);
        let slot = self.alloc_val();
        self.set_origin(slot, ValOrigin::Named(self.interner.intern("$order")));
        self.emit_assign(span, RefTarget::Var(slot), vec![], order_param);
        self.order_slot = Some(slot);
    }

    fn enter_contexts(&mut self, named: FxHashSet<QualifiedRef>, span: Span) {
        let named: BTreeSet<QualifiedRef> = named.into_iter().collect();
        for qref in named {
            let ty = self
                .resolution
                .context_types
                .get(&qref)
                .cloned()
                .unwrap_or_else(|| panic!("context {qref:?} is named but was not typed"));
            let slot = self.alloc_val();
            self.set_val_type(slot, ty.clone());
            self.set_origin(slot, ValOrigin::Context(qref.name));
            self.fetch_into(span, qref, slot, ty);
            self.context_slots.insert(qref, slot);
        }
    }

    fn context_slot(&self, qref: QualifiedRef) -> ValueId {
        *self
            .context_slots
            .get(&qref)
            .unwrap_or_else(|| panic!("context {qref:?} has no variable in this body"))
    }

    fn slot_type(&self, slot: ValueId) -> Ty {
        self.body
            .val_types
            .get(&slot)
            .cloned()
            .unwrap_or_else(|| panic!("slot {slot:?} has no type"))
    }

    fn fetch_into(&mut self, span: Span, context: QualifiedRef, slot: ValueId, ty: Ty) {
        let fetched = self.alloc_val();
        self.set_val_type(fetched, ty);
        self.set_origin(fetched, ValOrigin::Context(context.name));
        self.emit_inst(
            span,
            InstKind::Fetch {
                dst: fetched,
                context,
            },
        );
        self.emit_assign(span, RefTarget::Var(slot), vec![], fetched);
    }

    fn commit_from(&mut self, span: Span, context: QualifiedRef, slot: ValueId) {
        let ty = self.slot_type(slot);
        let value = self.emit_take(span, RefTarget::Var(slot), vec![], ty);
        self.emit_inst(span, InstKind::Commit { context, value });
    }

    /// The summary of a call (RFC-0017): the callee's, joined with that of
    /// every function value it is passed.
    fn touched_contexts(&self, callee_ty: &Ty, args: &[ValueId]) -> TouchedContexts {
        let mut touched = BTreeSet::new();
        let mut join = |ty: &Ty| -> bool {
            let Some(effect) = ty.effect() else {
                return false;
            };
            touched.extend(effect.reads.iter().copied());
            touched.extend(effect.writes.iter().copied());
            true
        };
        if !join(callee_ty) {
            return TouchedContexts::Every;
        }
        for arg in args {
            if let Some(ty) = self.body.val_types.get(arg)
                && matches!(ty, Ty::Fn { .. })
                && !join(ty)
            {
                return TouchedContexts::Every;
            }
        }
        TouchedContexts::These(touched)
    }

    fn bracketed_contexts(&self, callee_ty: &Ty, args: &[ValueId]) -> Vec<(QualifiedRef, ValueId)> {
        match self.touched_contexts(callee_ty, args) {
            TouchedContexts::Every => self.context_slots.iter().map(|(q, s)| (*q, *s)).collect(),
            TouchedContexts::These(touched) => self
                .context_slots
                .iter()
                .filter(|(q, _)| touched.contains(q))
                .map(|(q, s)| (*q, *s))
                .collect(),
        }
    }

    /// `inner?` (RFC-0038): test the operand's tag; on `Ok`/`Some` the
    /// payload is the value, on `Err`/`None` the function returns the
    /// operand's failure rebuilt at the function's return type.
    fn lower_try(&mut self, id: AstId, inner: &Expr, span: Span) -> ValueId {
        let src = self.lower_expr(inner);
        let operand_ty = self
            .body
            .val_types
            .get(&src)
            .cloned()
            .unwrap_or(Ty::error());
        let return_ty = self
            .resolution
            .try_returns
            .get(&id)
            .cloned()
            .unwrap_or(Ty::error());
        let (ok_tag, fail_tag, fail_carries) = match &operand_ty {
            Ty::Result(..) => ("Ok", "Err", true),
            _ => ("Some", "None", false),
        };
        let ok_tag = self.interner.intern(ok_tag);
        let fail_tag = self.interner.intern(fail_tag);

        let is_ok = self.alloc_val();
        self.set_val_type(is_ok, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::TestVariant {
                dst: is_ok,
                src,
                tag: ok_tag,
            },
        );
        let ok_label = self.alloc_label();
        let fail_label = self.alloc_label();
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: is_ok,
                then_label: ok_label,
                then_args: vec![],
                else_label: fail_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, fail_label);
        let failure = self.alloc_val();
        self.set_val_type(failure, return_ty.clone());
        let payload = if fail_carries {
            let err = self.alloc_val();
            self.set_val_type(err, self.payload_type(&operand_ty, fail_tag));
            self.emit_inst(span, InstKind::UnwrapVariant { dst: err, src });
            Some(err)
        } else {
            None
        };
        self.emit_inst(
            span,
            InstKind::MakeVariant {
                dst: failure,
                tag: fail_tag,
                payload,
            },
        );
        self.emit_return(span, failure);

        self.emit_label(span, ok_label);
        let dst = self.alloc_expr(id);
        self.emit_inst(span, InstKind::UnwrapVariant { dst, src });
        dst
    }

    /// Leave the body with `value`, yielding the current `Order` last.
    fn emit_return(&mut self, span: Span, value: ValueId) {
        let contexts: Vec<(QualifiedRef, ValueId)> =
            self.context_slots.iter().map(|(q, s)| (*q, *s)).collect();
        for (qref, slot) in contexts {
            self.commit_from(span, qref, slot);
        }
        let order = self.current_order(span);
        self.emit_inst(span, InstKind::Return { value, order });
    }

    /// Read the current `Order` of the body, if it has one.
    fn current_order(&mut self, span: Span) -> Option<ValueId> {
        self.order_slot
            .map(|slot| self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order))
    }

    /// Emit a call. A callee whose effect is not Pure takes the current
    /// `Order` and the call advances it.
    fn emit_call(
        &mut self,
        span: Span,
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: Vec<ValueId>,
    ) {
        let bracketed = self.bracketed_contexts(&callee_ty, &args);
        for (qref, slot) in &bracketed {
            self.commit_from(span, *qref, *slot);
        }
        let effectful = callee_ty.effect().is_some_and(|e| !e.is_pure());
        let order = if effectful {
            let slot = self.order_slot.unwrap_or_else(|| {
                panic!("effectful call lowered inside a body whose effect is Pure")
            });
            let before = match self.anyorder {
                Some(scope) => scope.entry,
                None => self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order),
            };
            let after = self.alloc_val();
            self.set_val_type(after, Ty::Order);
            Some(OrderEdge { before, after })
        } else {
            None
        };
        self.emit_inst(
            span,
            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                order,
            },
        );
        for (qref, slot) in bracketed {
            let ty = self.slot_type(slot);
            self.fetch_into(span, qref, slot, ty);
        }
        let Some(edge) = order else {
            return;
        };
        match self.anyorder {
            Some(scope) => {
                let acc = self.emit_take(span, RefTarget::Var(scope.acc), vec![], Ty::Order);
                let merged = self.alloc_val();
                self.set_val_type(merged, Ty::Order);
                self.emit_inst(
                    span,
                    InstKind::Merge {
                        dst: merged,
                        orders: vec![acc, edge.after],
                    },
                );
                self.emit_assign(span, RefTarget::Var(scope.acc), vec![], merged);
            }
            None => {
                let slot = self.order_slot.expect("an order edge needs the slot");
                self.emit_assign(span, RefTarget::Var(slot), vec![], edge.after);
            }
        }
    }

    /// Lower `anyorder { body }`. The block takes the current `Order` once;
    /// every effectful call inside takes it, and the block yields a merge of
    /// everything they yielded. Inside a block already open, nothing changes.
    fn lower_anyorder(&mut self, body: &[Stmt], span: Span) {
        let Some(slot) = self.order_slot else {
            // A Pure body issues no effect; the block declares nothing.
            self.push_scope();
            for s in body {
                self.lower_stmt(s);
            }
            self.pop_scope();
            return;
        };
        let outer = self.anyorder;
        if outer.is_none() {
            let entry = self.emit_take(span, RefTarget::Var(slot), vec![], Ty::Order);
            let acc = self.alloc_val();
            self.set_origin(acc, ValOrigin::Named(self.interner.intern("$anyorder")));
            self.emit_assign(span, RefTarget::Var(acc), vec![], entry);
            self.anyorder = Some(AnyorderScope { entry, acc });
        }
        self.push_scope();
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();
        if outer.is_none() {
            let scope = self.anyorder.take().expect("the block opened above");
            let exit = self.emit_take(span, RefTarget::Var(scope.acc), vec![], Ty::Order);
            self.emit_assign(span, RefTarget::Var(slot), vec![], exit);
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::ContextStore {
                name,
                path,
                expr,
                span,
                ..
            } => {
                self.lower_context_store(*name, path, expr, *span);
            }
            Stmt::VarFieldStore {
                name,
                path,
                expr,
                span,
                ..
            } => {
                self.lower_var_field_store(*name, path, expr, *span);
            }
            Stmt::DerefStore {
                target, expr, span, ..
            } => {
                self.lower_deref_store(target, expr, *span);
            }
            Stmt::IndexStore {
                place, expr, span, ..
            } => {
                let Expr::Index {
                    id,
                    callee_id,
                    object,
                    index,
                    span: index_span,
                } = place.as_ref()
                else {
                    unreachable!("the parser builds an IndexStore from an index expression")
                };
                let value = self.lower_expr(expr);
                let taken = self.take_slice(
                    object,
                    *callee_id,
                    *id,
                    self.index_access(*id).mutability,
                    *index_span,
                );
                let index = self.lower_expr(index);
                self.emit_inst(
                    *span,
                    InstKind::IndexSet {
                        slice: taken,
                        index,
                        value,
                    },
                );
            }
            Stmt::Expr(expr) => {
                self.lower_expr(expr);
            }
            Stmt::MatchBind {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                self.lower_stmt_match_bind(pattern, source, body, *span);
            }

            // -- Script mode statements ------------------------------
            Stmt::LetBind {
                name, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let ty = self
                    .body
                    .val_types
                    .get(&val)
                    .cloned()
                    .unwrap_or(Ty::error());
                let slot = self.define_var(*name, ty);
                self.emit_assign(*span, RefTarget::Var(slot), vec![], val);
            }
            Stmt::LetUninit { id, name, .. } => {
                // Type from typeck (fresh variable, unified later).
                let ty = self.type_of_id(*id);
                let slot = self.define_var(*name, ty.clone());
                self.set_val_type(slot, ty);
                // No store - init_check tracks this as uninit.
            }
            Stmt::Assign {
                name, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let slot = self
                    .lookup_var_slot(*name)
                    .expect("Assign to undefined variable - should have been caught by typeck");
                self.emit_assign(*span, RefTarget::Var(slot), vec![], val);
            }
            Stmt::While {
                cond, body, span, ..
            } => {
                self.lower_while(cond, body, *span);
            }
            Stmt::Anyorder { body, span, .. } => {
                self.lower_anyorder(body, *span);
            }
            Stmt::WhileLet {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                self.lower_while_let(pattern, source, body, *span);
            }
        }
    }

    /// Lower a match-bind statement: `pattern = source { body; };`
    ///
    /// The tag form and the `if let` expression with no `else` are the same
    /// match run for its effect, and go through `lower_match_bind_arm`.
    fn lower_stmt_match_bind(
        &mut self,
        pattern: &Pattern,
        source: &Expr,
        body: &[Stmt],
        span: Span,
    ) {
        let src = self.pattern_source(source);
        self.lower_match_bind_arm(pattern, src, body, None, span);
    }

    /// One arm matched for its effect: test the pattern, and on a match bind
    /// it and run the body, joining where the failed test goes. An irrefutable
    /// pattern has no test and no branch. No value leaves the join.
    fn lower_match_bind_arm(
        &mut self,
        pattern: &Pattern,
        src: PatSrc,
        body: &[Stmt],
        tail: Option<&Expr>,
        span: Span,
    ) {
        if pattern_is_irrefutable(pattern) {
            self.lower_match_bind_body(pattern, src, body, tail, span);
            return;
        }

        let matched = self.lower_pattern_test(pattern, src.clone(), span);
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: matched,
                then_label: body_label,
                then_args: vec![],
                else_label: end_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, body_label);
        self.lower_match_bind_body(pattern, src, body, tail, span);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: end_label,
                args: vec![],
            },
        );

        self.emit_label(span, end_label);
    }

    /// The arm's bindings and its statements, in a scope of their own. A tail
    /// expression is run for its effect and its value discarded.
    fn lower_match_bind_body(
        &mut self,
        pattern: &Pattern,
        src: PatSrc,
        body: &[Stmt],
        tail: Option<&Expr>,
        span: Span,
    ) {
        self.push_scope();
        self.lower_pattern_bind(pattern, src, span);
        for s in body {
            self.lower_stmt(s);
        }
        if let Some(tail) = tail {
            self.lower_expr(tail);
        }
        self.pop_scope();
    }

    fn lower_while(&mut self, cond: &Expr, body: &[Stmt], span: Span) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );
        self.emit_label(span, loop_label);

        let cond_val = self.lower_expr(cond);
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: cond_val,
                then_label: body_label,
                then_args: vec![],
                else_label: end_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, body_label);
        self.push_scope();
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );

        self.emit_label(span, end_label);
    }

    /// Lower `while let pattern = source { body }`.
    ///
    /// loop_label:
    ///   src = lower(source)
    ///   matched = test(pattern, src)
    ///   JumpIf matched -> body_label, end_label
    /// body_label:
    ///   bind(pattern, src)
    ///   body...
    ///   Jump loop_label
    /// end_label:
    fn lower_while_let(&mut self, pattern: &Pattern, source: &Expr, body: &[Stmt], span: Span) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );
        self.emit_label(span, loop_label);

        let src = self.pattern_source(source);
        let matched = self.lower_pattern_test(pattern, src.clone(), span);
        self.emit_inst(
            span,
            InstKind::JumpIf {
                cond: matched,
                then_label: body_label,
                then_args: vec![],
                else_label: end_label,
                else_args: vec![],
            },
        );

        self.emit_label(span, body_label);
        self.push_scope();
        self.lower_pattern_bind(pattern, src.clone(), span);
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );

        self.emit_label(span, end_label);
    }

    /// Lower `if cond { body; tail } else { ... }` as an expression.
    fn lower_if_expr(
        &mut self,
        id: AstId,
        cond: &Expr,
        then_body: &[Stmt],
        then_tail: &Option<Box<Expr>>,
        else_branch: &Option<Box<ElseBranch>>,
        span: Span,
    ) -> ValueId {
        let result_ty = self.type_of_id(id);
        let then_label = self.alloc_label();
        let merge_label = self.alloc_label();

        let cond_val = self.lower_expr(cond);

        match else_branch {
            Some(eb) => {
                let else_label = self.alloc_label();
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: cond_val,
                        then_label,
                        then_args: vec![],
                        else_label,
                        else_args: vec![],
                    },
                );

                // Then branch.
                self.emit_label(span, then_label);
                self.push_scope();
                for s in then_body {
                    self.lower_stmt(s);
                }
                let then_val = match then_tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![then_val],
                    },
                );

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![else_val],
                    },
                );

                // Merge.
                let result = self.alloc_val();
                self.set_val_type(result, result_ty);
                self.emit_inst(
                    span,
                    InstKind::BlockLabel {
                        label: merge_label,
                        params: vec![result],
                        merge_of: None,
                    },
                );
                result
            }
            None => {
                // No else: then branch stores to a var slot, else path skips.
                // The result is Unit (no value merge needed).
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: cond_val,
                        then_label,
                        then_args: vec![],
                        else_label: merge_label,
                        else_args: vec![],
                    },
                );

                self.emit_label(span, then_label);
                self.push_scope();
                for s in then_body {
                    self.lower_stmt(s);
                }
                if let Some(tail) = then_tail {
                    self.lower_expr(tail); // value discarded
                }
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![],
                    },
                );

                self.emit_label(span, merge_label);
                self.emit_unit(span)
            }
        }
    }

    /// Lower `if let pattern = source { body; tail } else { ... }` as an expression.
    fn lower_if_let_expr(
        &mut self,
        id: AstId,
        pattern: &Pattern,
        source: &Expr,
        then_body: &[Stmt],
        then_tail: &Option<Box<Expr>>,
        else_branch: &Option<Box<ElseBranch>>,
        span: Span,
    ) -> ValueId {
        let result_ty = self.type_of_id(id);
        let src = self.pattern_source(source);

        match else_branch {
            Some(eb) => {
                let matched = self.lower_pattern_test(pattern, src.clone(), span);
                let then_label = self.alloc_label();
                let merge_label = self.alloc_label();
                let else_label = self.alloc_label();
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: matched,
                        then_label,
                        then_args: vec![],
                        else_label,
                        else_args: vec![],
                    },
                );

                // Then branch.
                self.emit_label(span, then_label);
                self.push_scope();
                self.lower_pattern_bind(pattern, src.clone(), span);
                for s in then_body {
                    self.lower_stmt(s);
                }
                let then_val = match then_tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![then_val],
                    },
                );

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(
                    span,
                    InstKind::Jump {
                        label: merge_label,
                        args: vec![else_val],
                    },
                );

                // Merge.
                let result = self.alloc_val();
                self.set_val_type(result, result_ty);
                self.emit_inst(
                    span,
                    InstKind::BlockLabel {
                        label: merge_label,
                        params: vec![result],
                        merge_of: None,
                    },
                );
                result
            }
            None => {
                // The same match the tag form `pattern = source { body };`
                // writes: one arm, run for its effect.
                self.lower_match_bind_arm(pattern, src, then_body, then_tail.as_deref(), span);
                self.emit_unit(span)
            }
        }
    }

    /// Lower an else branch, returning the value it produces.
    fn lower_else_branch(&mut self, eb: &ElseBranch, span: Span, _merge_label: Label) -> ValueId {
        match eb {
            ElseBranch::ElseIf(expr) => self.lower_expr(expr),
            ElseBranch::Else { body, tail, .. } => {
                self.push_scope();
                for s in body {
                    self.lower_stmt(s);
                }
                let val = match tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                val
            }
        }
    }

    fn build_module(self) -> MirModule {
        MirModule {
            main: self.body,
            closures: self.closures,
        }
    }

    /// The storage a place expression names. Type checking admitted only a
    /// local, a context, or a field path of one.
    fn place(&mut self, place: &Expr) -> Place {
        let ty = self.type_of_id(place.id());
        let mut path: Vec<PathSeg> = Vec::new();
        let mut root = place;
        loop {
            match root {
                Expr::FieldAccess { object, field, .. } => {
                    path.push(PathSeg::Field(*field));
                    root = object;
                }
                Expr::Paren { inner, .. } => root = inner,
                _ => break,
            }
        }
        path.reverse();
        let target = self.storage_through(root).unwrap_or_else(|| {
            panic!("not a place: {root:?}; type checking admits only places here")
        });
        // A place that is a reference names what the reference names
        // (RFC-0029).
        let ty = match ty {
            Ty::Ref(_, inner) if path.is_empty() && matches!(target, RefTarget::Through(_)) => {
                inner.ty
            }
            ty => ty,
        };
        Place { target, path, ty }
    }

    /// An operator operand borrowed for the expression (RFC-0020).
    fn lend_operand(&mut self, operand: &Expr) -> ValueId {
        let ty = self.type_of_id(operand.id());
        if matches!(ty, Ty::Ref(..)) {
            return self.lower_expr(operand);
        }
        let mut path: Vec<PathSeg> = Vec::new();
        let mut root = operand;
        loop {
            match root {
                Expr::FieldAccess { object, field, .. } => {
                    path.push(PathSeg::Field(*field));
                    root = object;
                }
                Expr::Paren { inner, .. } => root = inner,
                _ => break,
            }
        }
        path.reverse();
        let span = operand.span();
        if let Some(target) = self.storage_through(root) {
            return self.emit_ref(span, target, path, Mutability::Shared, ty);
        }
        let value = self.lower_expr(operand);
        let owned = self.alloc_val();
        self.set_val_type(owned, ty.clone());
        self.emit_assign(span, RefTarget::Var(owned), vec![], value);
        self.emit_ref(span, RefTarget::Var(owned), vec![], Mutability::Shared, ty)
    }

    /// `a[i]` (RFC-0047): the container's slice, then the element. The
    /// mode is the checker's, from `index_modes`; nothing here decides how
    /// the element comes back.
    fn lower_index(
        &mut self,
        id: AstId,
        callee_id: AstId,
        object: &Expr,
        index: &Expr,
        span: Span,
    ) -> ValueId {
        let access = self.index_access(id);
        self.lower_index_as(access, id, callee_id, object, index, span)
    }

    fn index_access(&self, id: AstId) -> IndexAccess {
        self.resolution
            .index_access
            .get(&id)
            .copied()
            .unwrap_or_else(|| panic!("type checking settles every index expression"))
    }

    /// The element as a reference into the slice, whatever its type: what
    /// a place below an `a[i]` names — `&a[i]`, `a[i][j]`, `a[i].f`, and a
    /// method receiver.
    fn lower_index_as(
        &mut self,
        access: IndexAccess,
        id: AstId,
        callee_id: AstId,
        object: &Expr,
        index: &Expr,
        span: Span,
    ) -> ValueId {
        let IndexAccess { mutability, mode } = access;
        let slice = self.take_slice(object, callee_id, id, mutability, span);
        let index = self.lower_expr(index);
        let element = self.type_of_id(id);
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            match mode {
                IndexMode::Copy => element,
                IndexMode::Ref => Ty::Ref(mutability, Box::new(TypeArg::uniform(element))),
            },
        );
        self.emit_inst(
            span,
            InstKind::Index {
                dst,
                slice,
                index,
                mode,
            },
        );
        dst
    }

    /// The whole run of a container's elements, at the instance the checker
    /// settled on the index expression's callee id.
    fn take_slice(
        &mut self,
        object: &Expr,
        callee_id: AstId,
        id: AstId,
        mutability: Mutability,
        span: Span,
    ) -> ValueId {
        let Some(Callee::Extern { id: qref, instance }) =
            self.resolution.direct_calls.get(&callee_id).copied()
        else {
            panic!("type checking settles an `as_slice` instance on every index expression")
        };
        // A container that is already a reference is passed as it is
        // (RFC-0030): reborrowing it would put a `Ref` of its own in every
        // iteration, which the `AsSlice` could then never rise above.
        let container = match crate::typeck::is_place(object)
            && !matches!(self.type_of_id(object.id()), Ty::Ref(..))
        {
            true => {
                let mut restores = Vec::new();
                let lent = Lent {
                    id: object.id(),
                    span: object.span(),
                    place: self.place(object),
                    mutability,
                };
                let container = self.lend_place(lent, &mut restores);
                debug_assert!(
                    restores.is_empty(),
                    "a container lent for a slice crosses no boundary, so it is not cast back"
                );
                container
            }
            false => self.lower_expr(object),
        };
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            Ty::Ref(
                mutability,
                Box::new(TypeArg::uniform(Ty::Slice(Box::new(self.type_of_id(id))))),
            ),
        );
        self.emit_inst(
            span,
            InstKind::AsSlice {
                dst,
                container,
                mutability,
                instance: ExternInstance { id: qref, instance },
            },
        );
        dst
    }

    /// The storage a root expression names, through the reference it holds
    /// when it holds one.
    /// The place a root names: its storage, or, when the root is a
    /// reference (a storage holding one, or any expression of `&T`), the
    /// storage through that reference (RFC-0024).
    fn storage_through(&mut self, root: &Expr) -> Option<RefTarget> {
        // `a[i]` names the element the `Index` leaves a reference into
        // (RFC-0047 §3): the place is that reference, as `*r`'s is.
        if let Expr::Index {
            id,
            callee_id,
            object,
            index,
            span,
        } = root
        {
            let access = IndexAccess {
                mode: IndexMode::Ref,
                ..self.index_access(*id)
            };
            let reference = self.lower_index_as(access, *id, *callee_id, object, index, *span);
            return Some(RefTarget::Through(reference));
        }
        let ty = self.type_of_id(root.id());
        match (self.storage_of(root), matches!(ty, Ty::Ref(..))) {
            (Some(target), false) => Some(target),
            (Some(target), true) => {
                let reference = self.emit_take(root.span(), target, vec![], ty);
                Some(RefTarget::Through(reference))
            }
            (None, true) => Some(RefTarget::Through(self.lower_expr(root))),
            (None, false) => None,
        }
    }

    /// The storage a root expression names, if it is a local, a parameter,
    /// or a context.
    fn storage_of(&mut self, root: &Expr) -> Option<RefTarget> {
        match root {
            Expr::ContextRef { name, .. } => Some(RefTarget::Var(self.context_slot(*name))),
            Expr::Ident {
                name,
                ref_kind: RefKind::ExternParam,
                ..
            } => match self.try_param_slot(name.name) {
                Some(param_reg) => Some(RefTarget::Param(param_reg)),
                None if self.is_defined(name.name) => {
                    Some(RefTarget::Var(self.var_slot(name.name)))
                }
                None => None,
            },
            Expr::Ident {
                name,
                ref_kind: RefKind::Value,
                ..
            } if self.is_defined(name.name) => Some(RefTarget::Var(self.var_slot(name.name))),
            _ => None,
        }
    }

    /// Move `value` into a storage.
    fn emit_assign(&mut self, span: Span, target: RefTarget, path: Vec<PathSeg>, value: ValueId) {
        self.emit_inst(
            span,
            InstKind::Assign {
                target,
                path,
                value,
            },
        );
    }

    /// Move the value of a storage out into a fresh value of `ty`.
    fn emit_take(&mut self, span: Span, target: RefTarget, path: Vec<PathSeg>, ty: Ty) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, ty);
        self.emit_inst(span, InstKind::Take { dst, target, path });
        dst
    }

    /// A register holding a `&T` read at `T` where `T` is a word: the word
    /// copy of RFC-0018, which is what an operator's operand and a closure's
    /// capture both take. A register of any other type is returned as it is.
    fn read_word_through(&mut self, span: Span, taken: ValueId) -> ValueId {
        let Some(Ty::Ref(_, arg)) = self.body.val_types.get(&taken).cloned() else {
            return taken;
        };
        if !arg.ty.is_primitive() {
            return taken;
        }
        self.emit_take(span, RefTarget::Through(taken), vec![], arg.ty)
    }

    /// A reference to a storage: a fresh `&T` / `&mut T` value.
    fn emit_ref(
        &mut self,
        span: Span,
        target: RefTarget,
        path: Vec<PathSeg>,
        mutability: Mutability,
        inner_ty: Ty,
    ) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(
            dst,
            Ty::Ref(mutability, Box::new(TypeArg::uniform(inner_ty))),
        );
        self.set_origin(dst, ValOrigin::RefField(target.clone(), path.clone()));
        self.emit_inst(
            span,
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            },
        );
        dst
    }

    fn alloc_val(&mut self) -> ValueId {
        self.body.val_factory.next()
    }

    fn alloc_label(&mut self) -> Label {
        let l = Label(self.body.label_count);
        self.body.label_count += 1;
        l
    }

    /// Allocate a closure label from the global counter (not body-local).
    /// Prevents label collisions when nested closures each run in a sub-body.
    fn alloc_closure_label(&mut self) -> Label {
        let l = Label(self.closure_label_count);
        self.closure_label_count += 1;
        l
    }

    fn emit(&mut self, inst: Inst) {
        self.body.insts.push(inst);
    }

    fn emit_inst(&mut self, span: Span, kind: InstKind) {
        self.emit(Inst { span, kind });
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop().expect("a scope to pop");
    }

    /// Introduce a binding in the current scope with a fresh slot. A name
    /// already bound, here or outside, is shadowed.
    fn define_var(&mut self, name: Astr, ty: Ty) -> ValueId {
        let slot = self.body.val_factory.next();
        self.set_origin(slot, ValOrigin::Named(name));
        self.set_val_type(slot, ty.clone());
        let scope = self.scopes.last_mut().expect("a scope to define in");
        scope.insert(name, Local { ty, slot });
        slot
    }

    fn is_defined(&self, name: Astr) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(&name))
    }

    fn var_type(&self, name: Astr) -> Ty {
        self.lookup_local(name)
            .map(|l| l.ty.clone())
            .unwrap_or_else(Ty::error)
    }

    /// The innermost binding of `name`.
    fn lookup_local(&self, name: Astr) -> Option<&Local> {
        self.scopes.iter().rev().find_map(|scope| scope.get(&name))
    }

    fn lookup_var_slot(&self, name: Astr) -> Option<ValueId> {
        self.lookup_local(name).map(|l| l.slot)
    }

    /// The slot of the innermost binding of `name`; the name must be bound.
    fn var_slot(&mut self, name: Astr) -> ValueId {
        self.lookup_var_slot(name)
            .unwrap_or_else(|| panic!("variable {:?} used before it is bound", name))
    }

    /// Look up the param_reg for an extern parameter by name.
    /// Returns None if not found (e.g., inside a lambda where the param was captured).
    fn try_param_slot(&self, name: Astr) -> Option<ValueId> {
        for (pname, preg) in &self.body.params {
            if *pname == name {
                return Some(*preg);
            }
        }
        None
    }

    /// Look up the param_reg for an extern parameter by name.
    /// Panics if the param is not found (should be caught by typeck).
    fn param_slot(&self, name: Astr) -> ValueId {
        self.try_param_slot(name)
            .unwrap_or_else(|| panic!("param_slot: extern param {:?} not found", name))
    }

    fn set_val_type(&mut self, val: ValueId, ty: Ty) {
        self.body.val_types.insert(val, ty);
    }

    fn tuple_elem_type(&self, tuple_val: ValueId, index: usize) -> Ty {
        if let Some(Ty::Tuple(elems)) = self.body.val_types.get(&tuple_val) {
            elems.get(index).cloned().unwrap_or(Ty::error())
        } else {
            Ty::error()
        }
    }

    fn array_elem_type(&self, array_val: ValueId) -> Ty {
        if let Some(Ty::Array(elem, _)) = self.body.val_types.get(&array_val) {
            elem.as_ref().clone()
        } else {
            Ty::error()
        }
    }

    fn array_len(&self, array_val: ValueId) -> Option<usize> {
        match self.body.val_types.get(&array_val) {
            Some(Ty::Array(_, len)) => Some(len.get()),
            _ => None,
        }
    }

    fn object_field_type(&self, object_val: ValueId, key: Astr) -> Ty {
        if let Some(Ty::Object(fields)) = self.body.val_types.get(&object_val) {
            fields.get(&key).cloned().unwrap_or(Ty::error())
        } else {
            Ty::error()
        }
    }

    fn variant_inner_type(&self, variant_val: ValueId, tag: Astr) -> Ty {
        match self.body.val_types.get(&variant_val) {
            Some(ty) => self.payload_type(ty, tag),
            None => Ty::error(),
        }
    }

    fn set_origin(&mut self, val: ValueId, origin: ValOrigin) {
        self.body.debug.set(val, origin);
    }

    fn emit_label(&mut self, span: Span, label: Label) {
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label,
                params: vec![],
                merge_of: None,
            },
        );
    }

    /// Allocate a new value with type inferred from AST id.
    fn alloc_typed(&mut self, id: AstId) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, self.type_of_id(id));
        dst
    }

    /// Allocate a new value with type inferred from AST id and origin set to Expr.
    fn alloc_expr(&mut self, id: AstId) -> ValueId {
        let dst = self.alloc_typed(id);
        self.set_origin(dst, ValOrigin::Expr);
        dst
    }

    fn emit_array_index(
        &mut self,
        span: Span,
        array: ValueId,
        index: usize,
        elem_ty: Ty,
    ) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, elem_ty);
        self.emit_inst(span, InstKind::ArrayIndex { dst, array, index });
        dst
    }

    /// `left && right`, with `right` lowered only on the path that reaches
    /// it (RFC-0020).
    fn emit_and<Right>(&mut self, span: Span, left: ValueId, right: Right) -> ValueId
    where
        Right: FnOnce(&mut Self) -> ValueId,
    {
        let dst = self.alloc_val();
        self.lower_short_circuit(span, dst, left, ShortCircuit::And, right)
    }

    /// RFC-0020: `a && b` is `if a { b } else { false }` and `a || b` is
    /// `if a { true } else { b }`. `right` is lowered inside the block the
    /// other path enters, so the right operand's effects happen only where
    /// it is evaluated.
    fn lower_short_circuit<Right>(
        &mut self,
        span: Span,
        dst: ValueId,
        left: ValueId,
        connective: ShortCircuit,
        right: Right,
    ) -> ValueId
    where
        Right: FnOnce(&mut Self) -> ValueId,
    {
        let right_label = self.alloc_label();
        let decided_label = self.alloc_label();
        let merge_label = self.alloc_label();

        let jump_if = match connective {
            ShortCircuit::And => InstKind::JumpIf {
                cond: left,
                then_label: right_label,
                then_args: vec![],
                else_label: decided_label,
                else_args: vec![],
            },
            ShortCircuit::Or => InstKind::JumpIf {
                cond: left,
                then_label: decided_label,
                then_args: vec![],
                else_label: right_label,
                else_args: vec![],
            },
        };
        self.emit_inst(span, jump_if);

        self.emit_label(span, right_label);
        let right_val = right(self);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: merge_label,
                args: vec![right_val],
            },
        );

        self.emit_label(span, decided_label);
        let decided_val = self.emit_const_bool(span, connective.decided());
        self.emit_inst(
            span,
            InstKind::Jump {
                label: merge_label,
                args: vec![decided_val],
            },
        );

        self.set_val_type(dst, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: merge_label,
                params: vec![dst],
                merge_of: None,
            },
        );
        dst
    }

    /// Emit short-circuit merge: `ok_val` flows into the success path,
    /// `false` into the fail path, and a block param merges them.
    fn emit_fail_merge(&mut self, span: Span, ok_val: ValueId, fail_label: Label) -> ValueId {
        let result_label = self.alloc_label();
        let result_param = self.alloc_val();
        self.set_val_type(result_param, Ty::Bool);

        self.emit_inst(
            span,
            InstKind::Jump {
                label: result_label,
                args: vec![ok_val],
            },
        );

        // Fail path.
        self.emit_label(span, fail_label);
        let false_val = self.emit_const_bool(span, false);
        self.emit_inst(
            span,
            InstKind::Jump {
                label: result_label,
                args: vec![false_val],
            },
        );

        // Merge.
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: result_label,
                params: vec![result_param],
                merge_of: None,
            },
        );
        result_param
    }

    fn emit_unit(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Unit);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::Unit,
            },
        );
        dst
    }

    fn emit_const_bool(&mut self, span: Span, value: bool) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::Bool(value),
            },
        );
        dst
    }

    fn type_of_id(&self, id: AstId) -> Ty {
        self.resolution
            .type_map
            .get(&id)
            .cloned()
            .unwrap_or(Ty::error())
    }

    // --- Node lowering ---

    /// Lower a sequence of template nodes into a single concatenated String value.
    fn lower_nodes(&mut self, nodes: &[Node], span: Span) -> ValueId {
        let parts: Vec<ValueId> = nodes
            .iter()
            .map(|node| self.lower_node(node, span))
            .collect();
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(span, InstKind::StringConcat { dst, parts });
        dst
    }

    /// Lower a single template node, returning a String-typed ValueId.
    fn lower_node(&mut self, node: &Node, parent_span: Span) -> ValueId {
        match node {
            Node::Text { value, span, .. } => {
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::String);
                self.set_origin(dst, ValOrigin::Expr);
                self.emit_inst(
                    *span,
                    InstKind::Const {
                        dst,
                        value: Literal::String(value.clone()),
                    },
                );
                dst
            }
            Node::Comment { .. } => self.emit_empty_string(parent_span),
            Node::InlineExpr { expr, .. } => self.lower_expr(expr),
            Node::MatchBlock(mb) => self.lower_match_block(mb),
        }
    }

    fn emit_empty_string(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(
            span,
            InstKind::Const {
                dst,
                value: Literal::String(String::new()),
            },
        );
        dst
    }

    /// If the coercion map indicates this span needs a cast, emit a Cast
    /// instruction and return the new ValueId. Otherwise return `val` as-is.
    fn maybe_cast(&mut self, id: AstId, span: Span, val: ValueId) -> ValueId {
        match self.coercion_lookup.get(&id).cloned() {
            Some(CastKind::Extern {
                fn_ref,
                instance,
                callee_ty,
            }) => self.emit_extern_cast(
                span,
                &ExternCast {
                    fn_ref,
                    instance,
                    callee_ty,
                },
                val,
            ),
            Some(CastKind::ThroughRef { .. }) => {
                unreachable!("a cast through a reference is lowered where the argument is lent")
            }
            None => val,
        }
    }

    /// A call of a cast function on `val`: pure, one argument, no context.
    fn emit_extern_cast(&mut self, span: Span, cast: &ExternCast, val: ValueId) -> ValueId {
        let Ty::Fn { ret, .. } = &cast.callee_ty else {
            panic!("a cast's callee_ty is not Fn: {:?}", cast.callee_ty)
        };
        let cast_dst = self.alloc_val();
        self.set_val_type(cast_dst, (**ret).clone());
        self.emit_call(
            span,
            cast_dst,
            Callee::Extern {
                id: cast.fn_ref,
                instance: cast.instance,
            },
            cast.callee_ty.clone(),
            vec![val],
        );
        cast_dst
    }

    // --- Expression lowering ---

    /// Lower an expression to a value.
    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        if matches!(self.type_of_id(expr.id()), Ty::Never) {
            self.emit_diverge(expr.span());
        }
        self.maybe_cast(expr.id(), expr.span(), val)
    }

    /// The expression just lowered was typed `!`: nothing after it runs.
    /// The block ends here, and what the source wrote after it lands in a
    /// block no jump reaches (RFC-0038).
    fn emit_diverge(&mut self, span: Span) {
        self.emit_inst(span, InstKind::Diverge);
        let unreachable = self.alloc_label();
        self.emit_label(span, unreachable);
    }

    fn lower_expr_inner(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Borrow {
                place,
                mutable,
                span,
                ..
            } => {
                let mutability = if *mutable {
                    Mutability::Mut
                } else {
                    Mutability::Shared
                };
                let place = self.place(place);
                self.emit_ref(*span, place.target, place.path, mutability, place.ty)
            }
            Expr::Literal { id, value, span } => {
                let dst = self.alloc_expr(*id);
                self.emit_inst(
                    *span,
                    InstKind::Const {
                        dst,
                        value: value.clone(),
                    },
                );
                dst
            }

            Expr::ContextRef {
                id,
                name: qref,
                span,
            } => {
                let inner_ty = self.type_of_id(*id);
                let slot = self.context_slot(*qref);
                let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], inner_ty);
                self.set_origin(dst, ValOrigin::Context(qref.name));
                dst
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => match ref_kind {
                RefKind::ExternParam => {
                    let ty = self.type_of_id(*id);
                    // Try current body's params first. If not found (e.g., inside a
                    // lambda that captured this param as a regular variable), fall
                    // through to local variable lookup.
                    if let Some(param_reg) = self.try_param_slot(name.name) {
                        let dst = self.emit_take(*span, RefTarget::Param(param_reg), vec![], ty);
                        self.set_origin(dst, ValOrigin::ExternParam(name.name));
                        dst
                    } else if self.is_defined(name.name) {
                        let slot = self.var_slot(name.name);
                        let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                        self.set_origin(dst, ValOrigin::Named(name.name));
                        dst
                    } else {
                        // Not found - emit poison (should be caught by typeck).
                        let dst = self.alloc_val();
                        self.set_val_type(dst, ty);
                        self.emit_inst(*span, InstKind::Poison { dst });
                        dst
                    }
                }
                RefKind::Value => {
                    if !self.is_defined(name.name) {
                        // Undefined variable - emit Unit (dead code).
                        // TODO: this should be an error, not silent Unit.
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Unit);
                        return dst;
                    }
                    let ty = self.var_type(name.name);
                    let slot = self.var_slot(name.name);
                    let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                    self.set_origin(dst, ValOrigin::Named(name.name));
                    dst
                }
            },

            Expr::BinaryOp {
                id,
                left,
                op,
                right,
                span,
            } => {
                if let Some(connective) = short_circuit_of(*op) {
                    let l = self.lower_expr(left);
                    let l = self.read_word_through(*span, l);
                    let dst = self.alloc_expr(*id);
                    return self.lower_short_circuit(*span, dst, l, connective, |s| {
                        let r = s.lower_expr(right);
                        s.read_word_through(*span, r)
                    });
                }
                let left_ty = self.type_of_id(left.id());
                let on_string = matches!(&left_ty, Ty::String)
                    || matches!(&left_ty, Ty::Ref(_, inner) if matches!(inner.ty, Ty::String));
                if on_string && matches!(op, BinOp::Eq | BinOp::Neq | BinOp::Add) {
                    let l = self.lend_operand(left);
                    let r = self.lend_operand(right);
                    let dst = self.alloc_expr(*id);
                    match op {
                        BinOp::Add => {
                            self.emit_inst(
                                *span,
                                InstKind::StringConcat {
                                    dst,
                                    parts: vec![l, r],
                                },
                            );
                        }
                        BinOp::Eq => {
                            self.emit_inst(*span, InstKind::StringEq { dst, a: l, b: r });
                        }
                        _ => {
                            let eq = self.alloc_val();
                            self.set_val_type(eq, Ty::Bool);
                            self.emit_inst(
                                *span,
                                InstKind::StringEq {
                                    dst: eq,
                                    a: l,
                                    b: r,
                                },
                            );
                            self.emit_inst(
                                *span,
                                InstKind::UnaryOp {
                                    dst,
                                    op: UnaryOp::Not,
                                    operand: eq,
                                },
                            );
                        }
                    }
                    return dst;
                }
                if let Some(call) = self.resolution.operator_calls.get(id).cloned() {
                    let (callee, fn_ty) = (call.callee, call.ty);
                    let l = self.lend_operand(left);
                    let r = self.lend_operand(right);
                    let dst = self.alloc_expr(*id);
                    let (call_dst, negate) = match op {
                        BinOp::Neq => {
                            let v = self.alloc_val();
                            self.set_val_type(v, Ty::Bool);
                            (v, true)
                        }
                        _ => (dst, false),
                    };
                    self.emit_call(*span, call_dst, callee, fn_ty, vec![l, r]);
                    if negate {
                        self.emit_inst(
                            *span,
                            InstKind::UnaryOp {
                                dst,
                                op: UnaryOp::Not,
                                operand: call_dst,
                            },
                        );
                    }
                    return dst;
                }
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                let l = self.read_word_through(*span, l);
                let r = self.read_word_through(*span, r);
                let dst = self.alloc_expr(*id);
                self.emit_inst(
                    *span,
                    InstKind::BinOp {
                        dst,
                        op: *op,
                        left: l,
                        right: r,
                    },
                );
                dst
            }

            Expr::UnaryOp {
                id,
                op,
                operand,
                span,
            } => {
                let o = self.lower_expr(operand);
                let dst = self.alloc_expr(*id);
                let kind = match op {
                    UnaryOp::Deref => InstKind::Take {
                        dst,
                        target: RefTarget::Through(o),
                        path: vec![],
                    },
                    UnaryOp::Neg | UnaryOp::Not => InstKind::UnaryOp {
                        dst,
                        op: *op,
                        operand: o,
                    },
                };
                self.emit_inst(*span, kind);
                dst
            }

            Expr::Index {
                id,
                callee_id,
                object,
                index,
                span,
            } => self.lower_index(*id, *callee_id, object, index, *span),

            Expr::FieldAccess {
                id,
                object,
                field,
                span,
            } => {
                /// Walk a FieldAccess chain, collecting field names.
                /// Returns (root_expr, accumulated_path).
                fn collect_field_chain(expr: &Expr) -> (&Expr, Vec<Astr>) {
                    match expr {
                        Expr::FieldAccess { object, field, .. } => {
                            let (root, mut path) = collect_field_chain(object);
                            path.push(*field);
                            (root, path)
                        }
                        Expr::Paren { inner, .. } => collect_field_chain(inner),
                        other => (other, vec![]),
                    }
                }

                let field_ty = self.type_of_id(*id);
                let (root, mut path) = collect_field_chain(object);
                path.push(*field);
                let path = fields(&path);

                if let Some(target) = self.storage_through(root) {
                    let dst = self.emit_take(*span, target.clone(), path.clone(), field_ty);
                    self.set_origin(dst, ValOrigin::RefField(target, path));
                    dst
                } else {
                    let obj = self.lower_expr(object);
                    let dst = self.alloc_val();
                    self.set_val_type(dst, field_ty);
                    self.set_origin(dst, ValOrigin::Field(obj, *field));
                    self.emit_inst(
                        *span,
                        InstKind::FieldGet {
                            dst,
                            object: obj,
                            field: *field,
                            rest: vec![],
                        },
                    );
                    dst
                }
            }

            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => self.lower_func_call(func, args, None, *id, *span),

            Expr::MethodCall {
                id,
                callee_id,
                receiver,
                name,
                args,
                span,
                ..
            } => self.lower_method_call(*callee_id, receiver, *name, args, *id, *span),

            Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
                // Desugar: `a | f(b, c)` -> `f(a, b, c)`, `a | f` -> `f(a)`
                match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.lower_func_call(func, args, Some(left), *id, *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    } => self.lower_func_call(right, &[], Some(left), *id, *span),
                    _ => {
                        // Fallback: evaluate both sides, call as indirect.
                        let l = self.lower_expr(left);
                        let r = self.lower_expr(right);
                        let fn_ty = self
                            .body
                            .val_types
                            .get(&r)
                            .expect("indirect callee must have val_type")
                            .clone();
                        let dst = self.alloc_typed(expr.id());
                        self.emit_call(*span, dst, Callee::Indirect(r), fn_ty, vec![l]);
                        dst
                    }
                }
            }

            Expr::Lambda {
                id,
                params,
                body,
                span,
            } => {
                let captured: Vec<Astr> = self
                    .resolution
                    .lambda_captures
                    .get(id)
                    .expect("the checker records a capture list for every lambda it checks")
                    .clone();

                // A closure owns what it captures (RFC-0018): each capture is
                // taken from the storage the name binds, at that storage's own
                // type, and not at the `&T` the name reads as inside an
                // enclosing closure body.
                let capture_regs: Vec<ValueId> = captured
                    .iter()
                    .map(|name| {
                        let taken = if let Some(param_reg) = self.try_param_slot(*name) {
                            let ty = self.slot_type(param_reg);
                            let dst =
                                self.emit_take(*span, RefTarget::Param(param_reg), vec![], ty);
                            self.set_origin(dst, ValOrigin::ExternParam(*name));
                            dst
                        } else {
                            let slot = self.var_slot(*name);
                            let ty = self.slot_type(slot);
                            let dst = self.emit_take(*span, RefTarget::Var(slot), vec![], ty);
                            self.set_origin(dst, ValOrigin::Named(*name));
                            dst
                        };
                        self.read_word_through(*span, taken)
                    })
                    .collect();
                // Create closure body.
                let closure_label = self.alloc_closure_label();

                // Build the closure body MIR in a sub-lowerer.
                let mut sub_body = MirBody::new();

                // Captures become the first registers. The closure owns the
                // `T` handed to `MakeClosure` and the body sees a `&T`
                // (RFC-0018), which is what the runtime binds the register
                // to; a register already holding a reference is reborrowed,
                // since no `&&T` exists (RFC-0029).
                let mut closure_capture_regs = Vec::new();
                let mut capture_tys = Vec::new();
                for capture_reg in capture_regs.iter() {
                    let reg = sub_body.val_factory.next();
                    closure_capture_regs.push(reg);
                    let given = self
                        .body
                        .val_types
                        .get(capture_reg)
                        .expect("a capture register is typed where it is taken")
                        .clone();
                    let cap_ty = match given {
                        reference @ Ty::Ref(..) => reference,
                        owned => Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(owned))),
                    };
                    sub_body.val_types.insert(reg, cap_ty.clone());
                    capture_tys.push(cap_ty);
                }

                // Params follow captures.
                let mut closure_param_regs = Vec::new();
                for p in params.iter() {
                    let reg = sub_body.val_factory.next();
                    closure_param_regs.push(reg);
                    let ty = self.type_of_id(p.id);
                    sub_body.val_types.insert(reg, ty);
                }

                // We need to lower the body in context of the sub-body.
                // Swap state.
                let saved_body = std::mem::replace(&mut self.body, sub_body);
                let saved_scopes = std::mem::replace(&mut self.scopes, vec![FxHashMap::default()]);
                let saved_order_slot = self.order_slot;
                let saved_anyorder = self.anyorder;
                let saved_context_slots = std::mem::take(&mut self.context_slots);
                let saved_holds = std::mem::take(&mut self.holds);
                let lambda_effect = self.type_of_id(*id).effect().unwrap_or(Effect::OPAQUE);
                self.enter_body_order(lambda_effect, *span);
                self.enter_contexts(acvus_ast::direct_expr_context_refs(body), *span);

                // Captures and params are the closure body's first bindings.
                for ((name, capture_reg), cap_ty) in captured
                    .iter()
                    .zip(closure_capture_regs.iter())
                    .zip(capture_tys)
                {
                    let slot = self.define_var(*name, cap_ty);
                    self.emit_assign(*span, RefTarget::Var(slot), vec![], *capture_reg);
                }
                for (p, param_reg) in params.iter().zip(closure_param_regs.iter()) {
                    let ty = self.type_of_id(p.id);
                    let slot = self.define_var(p.name, ty);
                    self.emit_assign(p.span, RefTarget::Var(slot), vec![], *param_reg);
                }

                // lower_expr calls maybe_cast(body.span(), val) which will
                // pick up any lambda return coercion registered by the typechecker.
                let result_reg = self.lower_expr(body);
                // Capture the actual return type (may differ from type_map if Cast was inserted).
                let actual_ret_ty = self
                    .body
                    .val_types
                    .get(&result_reg)
                    .cloned()
                    .unwrap_or(Ty::error());
                self.emit_return(*span, result_reg);

                let mut closure_body_mir = std::mem::replace(&mut self.body, saved_body);
                self.scopes = saved_scopes;
                self.order_slot = saved_order_slot;
                self.anyorder = saved_anyorder;
                self.context_slots = saved_context_slots;
                self.holds = saved_holds;

                closure_body_mir.captures =
                    captured.iter().copied().zip(closure_capture_regs).collect();
                closure_body_mir.params = params
                    .iter()
                    .map(|p| p.name)
                    .zip(closure_param_regs)
                    .collect();
                closure_body_mir.task = self
                    .type_of_id(*id)
                    .effect()
                    .map_or(Task::Sync, |effect| effect.task);
                self.closures.insert(closure_label, closure_body_mir);

                // Allocate dst with Fn type. If a return-site Cast was inserted,
                // update the Fn's ret to match the actual (cast) return type.
                let dst = self.alloc_val();
                let mut fn_ty = self.type_of_id(*id);
                if let Ty::Fn { ref mut ret, .. } = fn_ty {
                    **ret = actual_ret_ty;
                }
                self.set_val_type(dst, fn_ty);
                self.emit_inst(
                    *span,
                    InstKind::MakeClosure {
                        dst,
                        body: closure_label,
                        captures: capture_regs,
                    },
                );
                dst
            }

            Expr::Paren { inner, .. } => self.lower_expr(inner),

            Expr::Try { id, inner, span } => self.lower_try(*id, inner, *span),

            Expr::List {
                id,
                head,
                rest: _,
                tail,
                span,
            } => {
                let elements: Vec<ValueId> = head
                    .iter()
                    .chain(tail.iter())
                    .map(|e| self.lower_expr(e))
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(*span, InstKind::MakeArray { dst, elements });
                dst
            }

            Expr::Object { id, fields, span } => {
                let field_regs = fields
                    .iter()
                    .map(|ObjectExprField { key, value, .. }| {
                        let r = self.lower_expr(value);
                        (*key, r)
                    })
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeObject {
                        dst,
                        fields: field_regs,
                    },
                );
                dst
            }

            Expr::Tuple { id, elements, span } => {
                let elem_vals: Vec<ValueId> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.lower_expr(e),
                        TupleElem::Wildcard(s) => {
                            let dst = self.alloc_val();
                            self.set_val_type(dst, Ty::Unit);
                            self.emit_inst(
                                *s,
                                InstKind::Const {
                                    dst,
                                    value: Literal::Bool(false),
                                },
                            );
                            dst
                        }
                    })
                    .collect();
                let dst = self.alloc_typed(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeTuple {
                        dst,
                        elements: elem_vals,
                    },
                );
                dst
            }

            Expr::Group { elements, span, .. } => {
                // Should not appear outside lambda params. Lower last element.
                let Some(last) = elements.last() else {
                    let dst = self.alloc_val();
                    self.set_val_type(dst, Ty::Unit);
                    self.emit_inst(
                        *span,
                        InstKind::Const {
                            dst,
                            value: Literal::Bool(false),
                        },
                    );
                    return dst;
                };
                self.lower_expr(last)
            }

            Expr::Variant {
                id,
                tag,
                payload,
                span,
                ..
            } => {
                let payload_val = payload.as_ref().map(|e| self.lower_expr(e));
                let dst = self.alloc_expr(*id);
                self.emit_inst(
                    *span,
                    InstKind::MakeVariant {
                        dst,
                        tag: *tag,
                        payload: payload_val,
                    },
                );
                dst
            }

            Expr::Block { stmts, tail, .. } => {
                self.push_scope();
                for stmt in stmts {
                    self.lower_stmt(stmt);
                }
                let val = self.lower_expr(tail);
                self.pop_scope();
                val
            }

            Expr::If {
                id,
                cond,
                then_body,
                then_tail,
                else_branch,
                span,
            } => self.lower_if_expr(*id, cond, then_body, then_tail, else_branch, *span),

            Expr::IfLet {
                id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                span,
            } => self.lower_if_let_expr(
                *id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                *span,
            ),
        }
    }

    fn lower_context_store(
        &mut self,
        qref: QualifiedRef,
        path: &[Astr],
        value_expr: &Expr,
        span: Span,
    ) -> ValueId {
        let val = self.lower_expr(value_expr);
        let slot = self.context_slot(qref);
        self.emit_assign(span, RefTarget::Var(slot), fields(path), val);
        val
    }

    fn lower_var_field_store(
        &mut self,
        name: Astr,
        path: &[Astr],
        value_expr: &Expr,
        span: Span,
    ) -> ValueId {
        let val = self.lower_expr(value_expr);
        let slot = self.var_slot(name);
        let var_ty = self.var_type(name);
        let target = if matches!(var_ty, Ty::Ref(..)) {
            let reference = self.emit_take(span, RefTarget::Var(slot), vec![], var_ty);
            RefTarget::Through(reference)
        } else {
            RefTarget::Var(slot)
        };
        self.emit_assign(span, target, fields(path), val);
        val
    }

    fn lower_deref_store(&mut self, target: &Expr, value_expr: &Expr, span: Span) {
        let val = self.lower_expr(value_expr);
        let reference = self.lower_expr(target);
        self.emit_assign(span, RefTarget::Through(reference), vec![], val);
    }

    fn lower_intrinsic_call(
        &mut self,
        intrinsic: crate::typeck::Intrinsic,
        args: &[Expr],
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        match intrinsic {
            crate::typeck::Intrinsic::StringClone => {
                let src = self.lower_expr(&args[0]);
                let dst = self.alloc_expr(call_id);
                self.emit_inst(call_span, InstKind::StringClone { dst, src });
                dst
            }
        }
    }

    /// The arguments of a call: a `&place` argument is lent, any other is
    /// a value.
    fn lower_call_args<'e, I>(&mut self, args: I) -> CallArgs
    where
        I: Iterator<Item = &'e Expr>,
    {
        let mut call = CallArgs {
            values: Vec::new(),
            restores: Vec::new(),
        };
        for a in args {
            let value = match a {
                Expr::Borrow {
                    id,
                    place,
                    mutable,
                    span,
                } => {
                    let mutability = if *mutable {
                        Mutability::Mut
                    } else {
                        Mutability::Shared
                    };
                    let lent = Lent {
                        id: *id,
                        span: *span,
                        place: self.place(place),
                        mutability,
                    };
                    self.lend_place(lent, &mut call.restores)
                }
                other => self.lower_expr(other),
            };
            call.values.push(value);
        }
        call
    }

    /// The reference a lent argument passes. A conversion the checker
    /// answered through the reference casts the place's value into the
    /// callee's representation first, holds the place at that type, and
    /// leaves it to restore after the call; a lend of a held place passes
    /// a reference at the held type; a conversion of the reference itself
    /// casts the reference.
    fn lend_place(&mut self, lent: Lent, restores: &mut Vec<PlaceRestore>) -> ValueId {
        let Lent {
            id,
            span,
            place,
            mutability,
        } = lent;
        if let Some(held) = self.held(&place) {
            return self.emit_ref(span, place.target, place.path, mutability, held);
        }
        match self.coercion_lookup.get(&id).cloned() {
            Some(CastKind::ThroughRef { cast, back, .. }) => {
                let held = self.cast_place(span, &place, &cast);
                self.holds.push(Held {
                    target: place.target,
                    path: place.path.clone(),
                    ty: held.clone(),
                });
                restores.push(PlaceRestore {
                    span,
                    place: Place {
                        ty: held.clone(),
                        ..place.clone()
                    },
                    back,
                });
                self.emit_ref(span, place.target, place.path, mutability, held)
            }
            Some(CastKind::Extern {
                fn_ref,
                instance,
                callee_ty,
            }) => {
                let reference = self.emit_ref(span, place.target, place.path, mutability, place.ty);
                self.emit_extern_cast(
                    span,
                    &ExternCast {
                        fn_ref,
                        instance,
                        callee_ty,
                    },
                    reference,
                )
            }
            None => self.emit_ref(span, place.target, place.path, mutability, place.ty),
        }
    }

    fn held(&self, place: &Place) -> Option<Ty> {
        self.holds
            .iter()
            .rev()
            .find(|held| held.target == place.target && held.path == place.path)
            .map(|held| held.ty.clone())
    }

    /// `place = cast(place)`: the type the place holds afterwards.
    fn cast_place(&mut self, span: Span, place: &Place, cast: &ExternCast) -> Ty {
        let value = self.emit_take(span, place.target, place.path.clone(), place.ty.clone());
        let converted = self.emit_extern_cast(span, cast, value);
        self.emit_assign(span, place.target, place.path.clone(), converted);
        self.slot_type(converted)
    }

    /// A call with its arguments; every place cast for the callee is
    /// released and restored after it.
    fn emit_call_with(
        &mut self,
        span: Span,
        dst: ValueId,
        callee: Callee,
        callee_ty: Ty,
        args: CallArgs,
    ) {
        self.emit_call(span, dst, callee, callee_ty, args.values);
        let Some(kept) = self.holds.len().checked_sub(args.restores.len()) else {
            panic!("a call restores more places than it holds")
        };
        let released = self.holds.split_off(kept);
        for (restore, held) in args.restores.iter().zip(&released) {
            assert!(
                held.target == restore.place.target && held.path == restore.place.path,
                "a call's holds are released in the order its restores were recorded"
            );
            self.cast_place(restore.span, &restore.place, &restore.back);
        }
    }

    /// A call lends the closure (RFC-0018). A slot already holding a
    /// `&Fn` — a capture — is reborrowed, since no `&&Fn` exists
    /// (RFC-0029).
    fn lent_closure(&mut self, name: Astr, span: Span) -> (ValueId, Ty) {
        let slot = self.var_slot(name);
        let (reg, ty) = match self.var_type(name) {
            Ty::Ref(_, lent) => {
                let reference = Ty::Ref(Mutability::Shared, lent.clone());
                let reg = self.emit_take(span, RefTarget::Var(slot), vec![], reference);
                (reg, lent.ty.clone())
            }
            owned => {
                let reg = self.emit_ref(
                    span,
                    RefTarget::Var(slot),
                    vec![],
                    Mutability::Shared,
                    owned.clone(),
                );
                (reg, owned)
            }
        };
        self.set_origin(reg, ValOrigin::Named(name));
        (reg, ty)
    }

    /// `recv.f(args)`: `f(recv', args)`, the receiver lent when `f`'s first
    /// parameter is a reference (RFC-0030).
    fn lower_method_call(
        &mut self,
        callee_id: AstId,
        receiver: &Expr,
        name: Astr,
        args: &[Expr],
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        let callee_ty = self.type_of_id(callee_id);
        let lent = match &callee_ty {
            Ty::Fn { params, .. } => match params.first().map(|p| &p.ty) {
                Some(Ty::Ref(mutability, _)) => Some(*mutability),
                _ => None,
            },
            _ => None,
        };
        let mut restores = Vec::new();
        let first = match lent {
            // A receiver that is already a reference value is passed as it
            // is (RFC-0030).
            Some(_) if !crate::typeck::is_place(receiver) => self.lower_expr(receiver),
            Some(mutability) => {
                let lent = Lent {
                    id: receiver.id(),
                    span: receiver.span(),
                    place: self.place(receiver),
                    mutability,
                };
                self.lend_place(lent, &mut restores)
            }
            None => self.lower_expr(receiver),
        };
        if let Some(intrinsic) = self.resolution.intrinsic_calls.get(&callee_id).copied() {
            return match intrinsic {
                crate::typeck::Intrinsic::StringClone => {
                    let dst = self.alloc_expr(call_id);
                    self.emit_inst(call_span, InstKind::StringClone { dst, src: first });
                    dst
                }
            };
        }
        let mut call = self.lower_call_args(args.iter());
        call.values.insert(0, first);
        call.restores.splice(0..0, restores);
        let dst = self.alloc_typed(call_id);
        match self.resolution.direct_calls.get(&callee_id).copied() {
            Some(callee) => {
                self.set_origin(dst, ValOrigin::Call(callee.id().name));
                self.emit_call_with(call_span, dst, callee, callee_ty, call);
            }
            None if self.is_defined(name) => {
                let (closure_reg, closure_ty) = self.lent_closure(name, call_span);
                self.emit_call_with(
                    call_span,
                    dst,
                    Callee::Indirect(closure_reg),
                    closure_ty,
                    call,
                );
            }
            None => {
                self.emit_inst(call_span, InstKind::Poison { dst });
            }
        }
        dst
    }

    fn lower_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Box<Expr>>,
        call_id: AstId,
        call_span: Span,
    ) -> ValueId {
        if let Some(intrinsic) = self.resolution.intrinsic_calls.get(&func.id()).copied() {
            return self.lower_intrinsic_call(intrinsic, args, call_id, call_span);
        }
        if self.resolution.structural_variant_calls.contains(&call_id)
            && let [payload] = args
        {
            let Expr::Ident { name, .. } = func else {
                unreachable!("a structural variant call is a qualified name");
            };
            let payload_val = self.lower_expr(payload);
            let dst = self.alloc_expr(call_id);
            self.emit_inst(
                call_span,
                InstKind::MakeVariant {
                    dst,
                    tag: name.name,
                    payload: Some(payload_val),
                },
            );
            return dst;
        }
        let call = self.lower_call_args(pipe_left.map(|left| &**left).into_iter().chain(args));
        let dst = self.alloc_typed(call_id);

        // Named function call (Ident).
        if let Expr::Ident {
            name,
            ref_kind,
            span: ident_span,
            ..
        } = func
        {
            match ref_kind {
                // fn_name(args) - named call
                RefKind::Value => {
                    self.set_origin(dst, ValOrigin::Call(name.name));

                    if let Some(callee) = self.resolution.direct_calls.get(&func.id()).copied() {
                        let callee_ty = self.type_of_id(func.id());
                        self.emit_call_with(call_span, dst, callee, callee_ty, call);
                        return dst;
                    }

                    if self.is_defined(name.name) {
                        let (closure_reg, closure_ty) = self.lent_closure(name.name, *ident_span);
                        self.emit_call_with(
                            call_span,
                            dst,
                            Callee::Indirect(closure_reg),
                            closure_ty,
                            call,
                        );
                        return dst;
                    }

                    // Typechecker already reported UndefinedFunction; emit poison.
                    self.emit_inst(call_span, InstKind::Poison { dst });
                    return dst;
                }
                _ => {}
            }
        }

        // Expression call (e.g., (|x| -> x)(42), or complex pipe)
        self.set_origin(dst, ValOrigin::Call(self.interner.intern("<closure>")));
        let func_reg = self.lower_expr(func);
        let fn_ty = self
            .body
            .val_types
            .get(&func_reg)
            .expect("indirect callee must have val_type")
            .clone();
        self.emit_call_with(call_span, dst, Callee::Indirect(func_reg), fn_ty, call);
        dst
    }

    // --- Match block lowering ---

    fn lower_match_block(&mut self, mb: &MatchBlock) -> ValueId {
        // Body-less context bind shorthand.
        if mb.arms.len() == 1
            && mb.arms[0].body.is_empty()
            && let Pattern::ContextBind {
                name: qref,
                span: pat_span,
                ..
            } = &mb.arms[0].pattern
        {
            let src = self.lower_expr(&mb.source);
            let slot = self.context_slot(*qref);
            self.emit_assign(*pat_span, RefTarget::Var(slot), vec![], src);
            return self.emit_empty_string(mb.span);
        }

        // Body-less binding shorthand (variable write or value binding).
        if mb.arms.len() == 1
            && mb.arms[0].body.is_empty()
            && let Pattern::Binding {
                name,
                ref_kind,
                span: pat_span,
                ..
            } = &mb.arms[0].pattern
        {
            let src = self.lower_expr(&mb.source);
            match ref_kind {
                RefKind::ExternParam => {
                    // Typeck already reported ExternParamAssign.
                    let dst = self.alloc_val();
                    self.emit_inst(*pat_span, InstKind::Poison { dst });
                    return dst;
                }
                RefKind::Value => {
                    let ty = self
                        .body
                        .val_types
                        .get(&src)
                        .cloned()
                        .unwrap_or(Ty::error());
                    let slot = self.define_var(*name, ty);
                    self.emit_assign(*pat_span, RefTarget::Var(slot), vec![], src);
                }
            }
            return self.emit_empty_string(mb.span);
        }

        // Pre-compute indent-adjusted arm bodies and catch-all body.
        let adjusted_arm_bodies: Option<Vec<Vec<Node>>> = mb.indent.as_ref().map(|modifier| {
            mb.arms
                .iter()
                .map(|arm| apply_indent_to_nodes(&arm.body, modifier))
                .collect()
        });
        let adjusted_catch_all_body: Option<Vec<Node>> = mb.indent.as_ref().and_then(|modifier| {
            mb.catch_all
                .as_ref()
                .map(|ca| apply_indent_to_nodes(&ca.body, modifier))
        });

        let source_reg = self.pattern_source(&mb.source);

        // Match is single-value pattern matching (no iteration).
        // Try each arm against the source value; first match wins.
        let end_label = self.alloc_label();
        let catch_all_label = self.alloc_label();

        let arm_labels: Vec<Label> = mb.arms.iter().map(|_| self.alloc_label()).collect();

        for (i, arm) in mb.arms.iter().enumerate() {
            let arm_label = arm_labels[i];
            let next_label = arm_labels.get(i + 1).copied().unwrap_or(catch_all_label);

            self.emit_label(arm.tag_span, arm_label);
            self.push_scope();

            // Test pattern against source value.
            let matched = self.lower_pattern_test(&arm.pattern, source_reg.clone(), arm.tag_span);

            // If pattern didn't match, try next arm (or catch-all).
            let arm_body_label = self.alloc_label();
            self.emit_inst(
                arm.tag_span,
                InstKind::JumpIf {
                    cond: matched,
                    then_label: arm_body_label,
                    then_args: vec![],
                    else_label: next_label,
                    else_args: vec![],
                },
            );
            self.emit_label(arm.tag_span, arm_body_label);

            // Bind pattern variables.
            self.lower_pattern_bind(&arm.pattern, source_reg.clone(), arm.tag_span);

            // Lower arm body (use indent-adjusted body if available).
            let body = adjusted_arm_bodies
                .as_ref()
                .map(|bodies| bodies[i].as_slice())
                .unwrap_or(&arm.body);
            let arm_result = self.lower_nodes(body, arm.tag_span);

            self.pop_scope();
            // After body, jump to end with the arm's concat result.
            self.emit_inst(
                arm.tag_span,
                InstKind::Jump {
                    label: end_label,
                    args: vec![arm_result],
                },
            );
        }

        // Catch-all block.
        self.emit_label(mb.span, catch_all_label);
        let catch_all_result = if let Some(catch_all) = &mb.catch_all {
            self.push_scope();
            let body = adjusted_catch_all_body
                .as_deref()
                .unwrap_or(&catch_all.body);
            let result = self.lower_nodes(body, mb.span);
            self.pop_scope();
            result
        } else {
            self.emit_empty_string(mb.span)
        };
        self.emit_inst(
            mb.span,
            InstKind::Jump {
                label: end_label,
                args: vec![catch_all_result],
            },
        );

        // Merge point: PHI receives the string result from whichever arm/catch-all matched.
        let merge_result = self.alloc_val();
        self.set_val_type(merge_result, Ty::String);
        self.emit_inst(
            mb.span,
            InstKind::BlockLabel {
                label: end_label,
                params: vec![merge_result],
                merge_of: Some(arm_labels[0]),
            },
        );
        merge_result
    }

    // --- Pattern test lowering ---

    /// Emit instructions that test whether `src_reg` matches `pattern`.
    /// Returns a register holding a Bool (true = match).
    fn reference_inner(&self, reg: ValueId) -> Option<Ty> {
        match self.body.val_types.get(&reg) {
            Some(Ty::Ref(_, inner)) => Some(inner.ty.clone()),
            _ => None,
        }
    }

    fn emit_part_ref(&mut self, span: Span, reference: ValueId, seg: PathSeg, ty: Ty) -> ValueId {
        self.emit_ref(
            span,
            RefTarget::Through(reference),
            vec![seg],
            Mutability::Shared,
            ty,
        )
    }

    fn payload_type(&self, inner: &Ty, tag: Astr) -> Ty {
        match inner {
            Ty::Option(payload) => payload.as_ref().clone(),
            Ty::Result(ok, err) => {
                if self.interner.resolve(tag) == "Ok" {
                    ok.as_ref().clone()
                } else {
                    err.as_ref().clone()
                }
            }
            Ty::Enum { variants, .. } => variants
                .get(&tag)
                .cloned()
                .flatten()
                .map(|t| *t)
                .unwrap_or(Ty::Unit),
            _ => Ty::error(),
        }
    }

    /// RFC-0024.
    fn pattern_parts_through(
        &mut self,
        pattern: &Pattern,
        reference: ValueId,
        inner: &Ty,
        span: Span,
    ) -> Vec<RefPart> {
        self.pattern_parts_place(pattern, &[], inner)
            .into_iter()
            .map(|part| {
                let seg = part
                    .path
                    .into_iter()
                    .next()
                    .expect("a part is one segment under the reference");
                RefPart {
                    reference: self.emit_part_ref(span, reference, seg, part.ty),
                    pattern: part.pattern,
                }
            })
            .collect()
    }

    fn lower_pattern_test_through(
        &mut self,
        pattern: &Pattern,
        reference: ValueId,
        inner: &Ty,
        span: Span,
    ) -> ValueId {
        match pattern {
            Pattern::ContextBind { .. } | Pattern::Binding { .. } => {
                self.emit_const_bool(span, true)
            }
            Pattern::Literal { value, .. } => {
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestLiteral {
                        dst,
                        src: reference,
                        value: value.clone(),
                    },
                );
                dst
            }
            Pattern::List { .. } | Pattern::Object { .. } | Pattern::Tuple { .. } => {
                let mut all_ok = self.emit_const_bool(span, true);
                for part in self.pattern_parts_through(pattern, reference, inner, span) {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        s.lower_pattern_test_value(&part.pattern, part.reference, span)
                    });
                }
                all_ok
            }
            Pattern::Variant { tag, payload, .. } => {
                let tag_ok = self.alloc_val();
                self.set_val_type(tag_ok, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestVariant {
                        dst: tag_ok,
                        src: reference,
                        tag: *tag,
                    },
                );
                let Some(inner_pat) = payload else {
                    return tag_ok;
                };
                if pattern_is_irrefutable(inner_pat) {
                    return tag_ok;
                }
                let check_inner_label = self.alloc_label();
                let fail_label = self.alloc_label();
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: tag_ok,
                        then_label: check_inner_label,
                        then_args: vec![],
                        else_label: fail_label,
                        else_args: vec![],
                    },
                );
                self.emit_label(span, check_inner_label);
                let part = self
                    .pattern_parts_through(pattern, reference, inner, span)
                    .into_iter()
                    .next()
                    .expect("a payload pattern names one part");
                let inner_ok = self.lower_pattern_test_value(&part.pattern, part.reference, span);
                self.emit_fail_merge(span, inner_ok, fail_label)
            }
        }
    }

    fn lower_pattern_bind_through(
        &mut self,
        pattern: &Pattern,
        reference: ValueId,
        inner: &Ty,
        span: Span,
    ) {
        match pattern {
            Pattern::Binding {
                name,
                ref_kind: RefKind::Value,
                ..
            } => {
                self.set_origin(reference, ValOrigin::Named(*name));
                let ty = Ty::Ref(
                    Mutability::Shared,
                    Box::new(TypeArg::uniform(inner.clone())),
                );
                let slot = self.define_var(*name, ty);
                self.emit_assign(span, RefTarget::Var(slot), vec![], reference);
            }
            Pattern::Binding {
                ref_kind: RefKind::ExternParam,
                ..
            }
            | Pattern::ContextBind { .. } => {
                let dst = self.alloc_val();
                self.emit_inst(span, InstKind::Poison { dst });
            }
            Pattern::Literal { .. } => {}
            Pattern::List { .. }
            | Pattern::Object { .. }
            | Pattern::Tuple { .. }
            | Pattern::Variant { .. } => {
                for part in self.pattern_parts_through(pattern, reference, inner, span) {
                    self.lower_pattern_bind_value(&part.pattern, part.reference, span);
                }
            }
        }
    }

    /// RFC-0024.
    fn pattern_source(&mut self, source: &Expr) -> PatSrc {
        let mut path: Vec<PathSeg> = Vec::new();
        let mut root = source;
        loop {
            match root {
                Expr::FieldAccess { object, field, .. } => {
                    path.push(PathSeg::Field(*field));
                    root = object;
                }
                Expr::Paren { inner, .. } => root = inner,
                _ => break,
            }
        }
        path.reverse();
        let root_ty = self.type_of_id(root.id());
        if matches!(root_ty, Ty::Ref(..)) {
            return PatSrc::Value(self.lower_expr(source));
        }
        match self.storage_of(root) {
            Some(target) => PatSrc::Place {
                target,
                path,
                ty: self.type_of_id(source.id()),
            },
            None => self.spill_pattern_source(source),
        }
    }

    /// A pattern's source that is not a place is stored in a slot of its
    /// own before matching: the test reads it and the bind takes from it,
    /// and a temporary can be taken only once.
    fn spill_pattern_source(&mut self, source: &Expr) -> PatSrc {
        let value = self.lower_expr(source);
        let ty = self.type_of_id(source.id());
        let slot = self.body.val_factory.next();
        self.set_origin(slot, ValOrigin::Named(self.interner.intern("$source")));
        self.set_val_type(slot, ty.clone());
        self.emit_assign(source.span(), RefTarget::Var(slot), vec![], value);
        PatSrc::Place {
            target: RefTarget::Var(slot),
            path: vec![],
            ty,
        }
    }

    fn lower_pattern_test(&mut self, pattern: &Pattern, src: PatSrc, span: Span) -> ValueId {
        match src {
            PatSrc::Value(reg) => self.lower_pattern_test_value(pattern, reg, span),
            PatSrc::Place { target, path, ty } => {
                self.lower_pattern_test_place(pattern, &target, &path, &ty, span)
            }
        }
    }

    fn lower_pattern_bind(&mut self, pattern: &Pattern, src: PatSrc, span: Span) {
        match src {
            PatSrc::Value(reg) => self.lower_pattern_bind_value(pattern, reg, span),
            PatSrc::Place { target, path, ty } => {
                self.lower_pattern_bind_place(pattern, &target, &path, &ty, span)
            }
        }
    }

    fn pattern_parts_place(&self, pattern: &Pattern, path: &[PathSeg], ty: &Ty) -> Vec<PlacePart> {
        let part = |seg: PathSeg, ty: Ty, pattern: &Pattern| {
            let mut p = path.to_vec();
            p.push(seg);
            PlacePart {
                path: p,
                ty,
                pattern: pattern.clone(),
            }
        };
        match pattern {
            Pattern::List { head, tail, .. } => {
                let (elem, len) = match ty {
                    Ty::Array(elem, len) => (elem.as_ref().clone(), Some(len.get())),
                    _ => (Ty::error(), None),
                };
                let head_parts = head
                    .iter()
                    .enumerate()
                    .map(|(i, p)| part(PathSeg::Index(i), elem.clone(), p));
                let tail_parts = len.into_iter().flat_map(|len| {
                    tail.iter()
                        .enumerate()
                        .map(move |(i, p)| (len - tail.len() + i, p))
                        .collect::<Vec<_>>()
                });
                head_parts
                    .chain(tail_parts.map(|(i, p)| part(PathSeg::Index(i), elem.clone(), p)))
                    .collect()
            }
            Pattern::Object { fields: pats, .. } => pats
                .iter()
                .map(|ObjectPatternField { key, pattern, .. }| {
                    let fty = match ty {
                        Ty::Object(field_tys) => field_tys.get(key).cloned().unwrap_or(Ty::error()),
                        _ => Ty::error(),
                    };
                    part(PathSeg::Field(*key), fty, pattern)
                })
                .collect(),
            Pattern::Tuple { elements, .. } => elements
                .iter()
                .enumerate()
                .filter_map(|(i, elem)| match elem {
                    TuplePatternElem::Pattern(pat) => Some((i, pat)),
                    _ => None,
                })
                .map(|(i, pat)| {
                    let ety = match ty {
                        Ty::Tuple(elem_tys) => elem_tys.get(i).cloned().unwrap_or(Ty::error()),
                        _ => Ty::error(),
                    };
                    part(PathSeg::Index(i), ety, pat)
                })
                .collect(),
            Pattern::Variant { tag, payload, .. } => match payload {
                Some(inner) => vec![part(PathSeg::Payload, self.payload_type(ty, *tag), inner)],
                None => Vec::new(),
            },
            Pattern::Binding { .. } | Pattern::ContextBind { .. } | Pattern::Literal { .. } => {
                Vec::new()
            }
        }
    }

    fn lower_pattern_test_place(
        &mut self,
        pattern: &Pattern,
        target: &RefTarget,
        path: &[PathSeg],
        ty: &Ty,
        span: Span,
    ) -> ValueId {
        match pattern {
            Pattern::ContextBind { .. } | Pattern::Binding { .. } => {
                self.emit_const_bool(span, true)
            }
            Pattern::Literal { value, .. } => {
                let read = self.emit_take(span, target.clone(), path.to_vec(), ty.clone());
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestLiteral {
                        dst,
                        src: read,
                        value: value.clone(),
                    },
                );
                dst
            }
            Pattern::List { .. } | Pattern::Object { .. } | Pattern::Tuple { .. } => {
                let mut all_ok = self.emit_const_bool(span, true);
                for part in self.pattern_parts_place(pattern, path, ty) {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        s.lower_pattern_test_place(
                            &part.pattern,
                            target,
                            &part.path,
                            &part.ty,
                            span,
                        )
                    });
                }
                all_ok
            }
            Pattern::Variant { tag, payload, .. } => {
                let reference = self.emit_ref(
                    span,
                    target.clone(),
                    path.to_vec(),
                    Mutability::Shared,
                    ty.clone(),
                );
                let tag_ok = self.alloc_val();
                self.set_val_type(tag_ok, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestVariant {
                        dst: tag_ok,
                        src: reference,
                        tag: *tag,
                    },
                );
                let Some(inner_pat) = payload else {
                    return tag_ok;
                };
                if pattern_is_irrefutable(inner_pat) {
                    return tag_ok;
                }
                let check_inner_label = self.alloc_label();
                let fail_label = self.alloc_label();
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: tag_ok,
                        then_label: check_inner_label,
                        then_args: vec![],
                        else_label: fail_label,
                        else_args: vec![],
                    },
                );
                self.emit_label(span, check_inner_label);
                let part = self
                    .pattern_parts_place(pattern, path, ty)
                    .into_iter()
                    .next()
                    .expect("a payload pattern names one part");
                let inner_ok = self.lower_pattern_test_place(
                    &part.pattern,
                    target,
                    &part.path,
                    &part.ty,
                    span,
                );
                self.emit_fail_merge(span, inner_ok, fail_label)
            }
        }
    }

    fn lower_pattern_bind_place(
        &mut self,
        pattern: &Pattern,
        target: &RefTarget,
        path: &[PathSeg],
        ty: &Ty,
        span: Span,
    ) {
        match pattern {
            Pattern::Binding {
                name,
                ref_kind: RefKind::Value,
                ..
            } => {
                let read = self.emit_take(span, target.clone(), path.to_vec(), ty.clone());
                self.set_origin(read, ValOrigin::Named(*name));
                let slot = self.define_var(*name, ty.clone());
                self.emit_assign(span, RefTarget::Var(slot), vec![], read);
            }
            Pattern::Binding {
                ref_kind: RefKind::ExternParam,
                ..
            } => {
                let dst = self.alloc_val();
                self.emit_inst(span, InstKind::Poison { dst });
            }
            Pattern::ContextBind { name: qref, .. } => {
                let read = self.emit_take(span, target.clone(), path.to_vec(), ty.clone());
                let slot = self.context_slot(*qref);
                self.emit_assign(span, RefTarget::Var(slot), vec![], read);
            }
            Pattern::Literal { .. } => {}
            Pattern::List { .. }
            | Pattern::Object { .. }
            | Pattern::Tuple { .. }
            | Pattern::Variant { .. } => {
                for part in self.pattern_parts_place(pattern, path, ty) {
                    self.lower_pattern_bind_place(
                        &part.pattern,
                        target,
                        &part.path,
                        &part.ty,
                        span,
                    );
                }
            }
        }
    }

    fn lower_pattern_test_value(
        &mut self,
        pattern: &Pattern,
        src_reg: ValueId,
        span: Span,
    ) -> ValueId {
        if let Some(inner) = self.reference_inner(src_reg) {
            return self.lower_pattern_test_through(pattern, src_reg, &inner, span);
        }
        match pattern {
            // Context bind is always irrefutable.
            Pattern::ContextBind { .. } => self.emit_const_bool(span, true),

            Pattern::Binding { ref_kind, .. } => {
                if *ref_kind == RefKind::ExternParam {
                    // Typeck already reported ExternParamAssign.
                    let dst = self.alloc_val();
                    self.emit_inst(span, InstKind::Poison { dst });
                }
                self.emit_const_bool(span, true)
            }

            Pattern::Literal { value, .. } => {
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestLiteral {
                        dst,
                        src: src_reg,
                        value: value.clone(),
                    },
                );
                dst
            }

            Pattern::List { head, tail, .. } => {
                let Some(len) = self.array_len(src_reg) else {
                    let dst = self.alloc_val();
                    self.emit_inst(span, InstKind::Poison { dst });
                    return dst;
                };
                let elem_ty = self.array_elem_type(src_reg);
                let mut all_ok = self.emit_const_bool(span, true);
                for (i, p) in head.iter().enumerate() {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        let elem = s.emit_array_index(span, src_reg, i, elem_ty.clone());
                        s.lower_pattern_test_value(p, elem, span)
                    });
                }
                for (i, p) in tail.iter().enumerate() {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        let index = len - tail.len() + i;
                        let elem = s.emit_array_index(span, src_reg, index, elem_ty.clone());
                        s.lower_pattern_test_value(p, elem, span)
                    });
                }
                all_ok
            }

            Pattern::Object { fields, .. } => {
                // Test that all keys exist.
                let mut all_ok = self.emit_const_bool(span, true);

                for ObjectPatternField { key, pattern, .. } in fields {
                    all_ok = self.emit_and(span, all_ok, |s| {
                        let key_ok = s.alloc_val();
                        s.set_val_type(key_ok, Ty::Bool);
                        s.emit_inst(
                            span,
                            InstKind::TestObjectKey {
                                dst: key_ok,
                                src: src_reg,
                                key: *key,
                            },
                        );

                        let field_val = s.alloc_val();
                        s.set_val_type(field_val, s.object_field_type(src_reg, *key));
                        s.emit_inst(
                            span,
                            InstKind::ObjectGet {
                                dst: field_val,
                                object: src_reg,
                                key: *key,
                            },
                        );

                        s.lower_pattern_test_value(pattern, field_val, span)
                    });
                }

                all_ok
            }

            Pattern::Tuple { elements, .. } => {
                // Tuple length is guaranteed by the type system - always matches.
                // Test each sub-pattern element.
                let mut all_ok = self.emit_const_bool(span, true);

                for (i, elem) in elements.iter().enumerate() {
                    let TuplePatternElem::Pattern(pat) = elem else {
                        continue;
                    };
                    all_ok = self.emit_and(span, all_ok, |s| {
                        let field_val = s.alloc_val();
                        s.set_val_type(field_val, s.tuple_elem_type(src_reg, i));
                        s.emit_inst(
                            span,
                            InstKind::TupleIndex {
                                dst: field_val,
                                tuple: src_reg,
                                index: i,
                            },
                        );
                        s.lower_pattern_test_value(pat, field_val, span)
                    });
                }

                all_ok
            }

            Pattern::Variant { tag, payload, .. } => {
                // Test if the variant tag matches.
                let tag_ok = self.alloc_val();
                self.set_val_type(tag_ok, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestVariant {
                        dst: tag_ok,
                        src: src_reg,
                        tag: *tag,
                    },
                );

                let Some(inner_pat) = payload else {
                    // No payload (e.g. None) - tag test is the final result.
                    return tag_ok;
                };
                // An inner pattern that cannot fail adds no test, and unwrapping
                // here would move the payload out before the bind path does.
                if pattern_is_irrefutable(inner_pat) {
                    return tag_ok;
                }

                // Has payload - short-circuit: if tag fails, skip inner test.
                let check_inner_label = self.alloc_label();
                let fail_label = self.alloc_label();

                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: tag_ok,
                        then_label: check_inner_label,
                        then_args: vec![],
                        else_label: fail_label,
                        else_args: vec![],
                    },
                );

                // Success path: unwrap and test inner pattern.
                self.emit_label(span, check_inner_label);
                let inner_val = self.alloc_val();
                self.set_val_type(inner_val, self.variant_inner_type(src_reg, *tag));
                self.emit_inst(
                    span,
                    InstKind::UnwrapVariant {
                        dst: inner_val,
                        src: src_reg,
                    },
                );
                let inner_ok = self.lower_pattern_test_value(inner_pat, inner_val, span);

                self.emit_fail_merge(span, inner_ok, fail_label)
            }
        }
    }

    fn extract_range_bounds(&self, start: &Pattern, end: &Pattern) -> (i64, i64) {
        let extract = |p: &Pattern| match p {
            Pattern::Literal {
                value: Literal::Int(n),
                ..
            } => *n as i64,
            _ => 0,
        };
        (extract(start), extract(end))
    }

    /// Emit instructions that bind pattern variables from a matched value.
    fn lower_pattern_bind_value(&mut self, pattern: &Pattern, src_reg: ValueId, span: Span) {
        if let Some(inner) = self.reference_inner(src_reg) {
            self.lower_pattern_bind_through(pattern, src_reg, &inner, span);
            return;
        }
        match pattern {
            Pattern::ContextBind { name: qref, .. } => {
                let slot = self.context_slot(*qref);
                self.emit_assign(span, RefTarget::Var(slot), vec![], src_reg);
            }
            Pattern::Binding {
                name,
                ref_kind: RefKind::Value,
                ..
            } => {
                self.set_origin(src_reg, ValOrigin::Named(*name));
                let ty = self
                    .body
                    .val_types
                    .get(&src_reg)
                    .cloned()
                    .unwrap_or(Ty::error());
                let slot = self.define_var(*name, ty);
                self.emit_assign(span, RefTarget::Var(slot), vec![], src_reg);
            }
            Pattern::Binding {
                name: _,
                ref_kind: RefKind::ExternParam,
                ..
            } => {
                // Typeck already reported ExternParamAssign.
                let dst = self.alloc_val();
                self.emit_inst(span, InstKind::Poison { dst });
            }
            Pattern::Literal { .. } => {}

            Pattern::List { head, tail, .. } => {
                let Some(len) = self.array_len(src_reg) else {
                    return;
                };
                let elem_ty = self.array_elem_type(src_reg);
                for (i, p) in head.iter().enumerate() {
                    let elem = self.emit_array_index(span, src_reg, i, elem_ty.clone());
                    self.lower_pattern_bind_value(p, elem, span);
                }
                for (i, p) in tail.iter().enumerate() {
                    let elem =
                        self.emit_array_index(span, src_reg, len - tail.len() + i, elem_ty.clone());
                    self.lower_pattern_bind_value(p, elem, span);
                }
            }

            Pattern::Object { fields, .. } => {
                for ObjectPatternField { key, pattern, .. } in fields {
                    let field_val = self.alloc_val();
                    self.set_val_type(field_val, self.object_field_type(src_reg, *key));
                    self.emit_inst(
                        span,
                        InstKind::ObjectGet {
                            dst: field_val,
                            object: src_reg,
                            key: *key,
                        },
                    );
                    self.lower_pattern_bind_value(pattern, field_val, span);
                }
            }

            Pattern::Tuple { elements, .. } => {
                for (i, elem) in elements.iter().enumerate() {
                    let TuplePatternElem::Pattern(pat) = elem else {
                        continue;
                    };
                    let field_val = self.alloc_val();
                    self.set_val_type(field_val, self.tuple_elem_type(src_reg, i));
                    self.emit_inst(
                        span,
                        InstKind::TupleIndex {
                            dst: field_val,
                            tuple: src_reg,
                            index: i,
                        },
                    );
                    self.lower_pattern_bind_value(pat, field_val, span);
                }
            }

            Pattern::Variant { tag, payload, .. } => {
                let Some(inner_pat) = payload else {
                    return;
                };
                let inner_val = self.alloc_val();
                self.set_val_type(inner_val, self.variant_inner_type(src_reg, *tag));
                self.emit_inst(
                    span,
                    InstKind::UnwrapVariant {
                        dst: inner_val,
                        src: src_reg,
                    },
                );
                self.lower_pattern_bind_value(inner_pat, inner_val, span);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower(interner: &Interner, source: &str) -> MirModule {
        lower_with(interner, source, &FxHashMap::default())
    }

    fn lower_with(interner: &Interner, source: &str, context: &FxHashMap<Astr, Ty>) -> MirModule {
        let ctx: Vec<(&str, Ty)> = context
            .iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        let module = crate::test::compile_template(interner, source, &ctx).expect("compile failed");
        module
    }

    #[test]
    fn lower_text_node() {
        let interner = Interner::new();
        let module = lower(&interner, "hello world");
        // Template: empty_str const + text const + concat + return
        let has_text = module.main.insts.iter().any(|i| {
            matches!(
                &i.kind,
                InstKind::Const { value: Literal::String(s), .. } if s == "hello world"
            )
        });
        let has_return = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::Return { .. }));
        assert!(has_text);
        assert!(has_return);
    }

    #[test]
    fn lower_string_emit() {
        let interner = Interner::new();
        let module = lower(&interner, r#"{{ "hello" }}"#);
        // InlineExpr emits Const only (Yield removed, pending Iterator<String> redesign)
        assert!(module.main.insts.len() >= 1);
        assert!(matches!(&module.main.insts[0].kind, InstKind::Const { .. }));
    }

    #[test]
    fn extern_param_write_rejected() {
        let interner = Interner::new();
        let result = crate::test::compile_template(&interner, "{{ $count = 42 }}", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn lower_match_block() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        // Use a non-binding pattern to trigger full match block (no iteration).
        let module = lower_with(&interner, r#"{{ true = @n == 1 }}matched{{/}}"#, &context);
        // Should have pattern test and conditional jump.
        let has_jump_if = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::JumpIf { .. }));
        assert!(has_jump_if);
    }

    #[test]
    #[ignore = "requires Phase 2: builtin -> graph Function migration"]
    fn lower_builtin_call() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let module = lower_with(&interner, r#"{{ @n | to_string }}"#, &context);
        let has_call = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::FunctionCall { .. }));
        assert!(has_call);
    }

    #[test]
    fn adjust_text_indent_decrease() {
        let text = "first\n    second\n      third";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result.as_str(), "first\n  second\n    third");
    }

    #[test]
    fn adjust_text_indent_decrease_clamp() {
        let text = "first\n second\n  third";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(4));
        assert_eq!(result.as_str(), "first\nsecond\nthird");
    }

    #[test]
    fn adjust_text_indent_increase() {
        let text = "first\nsecond\n  third";
        let result = adjust_text_indent(text, &IndentModifier::Increase(3));
        assert_eq!(result.as_str(), "   first\n   second\n     third");
    }

    #[test]
    fn adjust_text_indent_first_line_also_adjusted() {
        let text = "  first\n  second";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result.as_str(), "first\nsecond");
    }

    #[test]
    fn adjust_text_indent_no_newline() {
        let text = "  hello";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result.as_str(), "hello");
    }

    #[test]
    fn lower_match_block_indent_decrease() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let source = "{{ true = @n == 1 }}\n    matched\n    here{{/-2}}";
        let module = lower_with(&interner, source, &context);
        let texts: Vec<&str> = module
            .main
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const {
                    value: Literal::String(s),
                    ..
                } => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("\n  matched\n  here")));
    }

    #[test]
    fn lower_match_block_indent_increase() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::I64)]);
        let source = "{{ true = @n == 1 }}\nmatched{{/+4}}";
        let module = lower_with(&interner, source, &context);
        let texts: Vec<&str> = module
            .main
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::Const {
                    value: Literal::String(s),
                    ..
                } => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("\n    matched")));
    }
}
