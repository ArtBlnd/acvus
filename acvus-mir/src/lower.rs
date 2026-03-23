use std::sync::Arc;

use acvus_ast::{
    BinOp, Expr, IndentModifier, IterBlock, Literal, MatchBlock, Node, ObjectExprField,
    ObjectPatternField, Pattern, RefKind, Script, Span, Stmt, Template, TupleElem,
    TuplePatternElem,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{ContextId, FunctionId};
use crate::hints::{Hint, HintTable};
use crate::ir::{
    Callee, CastKind, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId,
};
use crate::ty::{Effect, Ty};
use crate::typeck::{CoercionMap, TypeMap};

pub struct Lowerer<'a> {
    body: MirBody,
    /// Interner for string interning.
    interner: &'a Interner,
    /// Stack of scopes: variable name → register.
    scopes: Vec<FxHashMap<Astr, ValueId>>,
    /// Type map from type checker.
    type_map: TypeMap,
    /// Coercion map from type checker (expr_span → CastKind).
    coercion_lookup: FxHashMap<Span, CastKind>,
    /// Closures produced during lowering.
    closures: FxHashMap<Label, Arc<MirBody>>,
    /// Global closure label counter — shared across nesting levels to prevent
    /// label collisions when nested closures each allocate from a sub-body.
    closure_label_count: u32,
    /// Hint table.
    hints: HintTable,
    /// Context name → (ContextId, Ty).
    context_ids: Freeze<FxHashMap<Astr, (ContextId, Ty)>>,
    /// Function name → (FunctionId, Ty).
    function_ids: Freeze<FxHashMap<Astr, (FunctionId, Ty)>>,
    /// ValueIds that are projections (not yet materialized values).
    /// FieldAccess on a projection produces another projection.
    /// Use `ensure_loaded` to materialize.
    projections: FxHashSet<ValueId>,
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
            Node::Text { value, span } => Node::Text {
                value: adjust_text_indent(value, modifier),
                span: *span,
            },
            Node::MatchBlock(mb) => Node::MatchBlock(MatchBlock {
                arms: mb
                    .arms
                    .iter()
                    .map(|arm| acvus_ast::MatchArm {
                        pattern: arm.pattern.clone(),
                        body: apply_indent_to_nodes(&arm.body, modifier),
                        tag_span: arm.tag_span,
                    })
                    .collect(),
                catch_all: mb.catch_all.as_ref().map(|ca| acvus_ast::CatchAll {
                    body: apply_indent_to_nodes(&ca.body, modifier),
                    tag_span: ca.tag_span,
                }),
                source: mb.source.clone(),
                indent: mb.indent,
                span: mb.span,
            }),
            Node::IterBlock(ib) => Node::IterBlock(acvus_ast::IterBlock {
                pattern: ib.pattern.clone(),
                source: ib.source.clone(),
                body: apply_indent_to_nodes(&ib.body, modifier),
                catch_all: ib.catch_all.as_ref().map(|ca| acvus_ast::CatchAll {
                    body: apply_indent_to_nodes(&ca.body, modifier),
                    tag_span: ca.tag_span,
                }),
                indent: ib.indent,
                span: ib.span,
            }),
            other => other.clone(),
        })
        .collect()
}

impl<'a> Lowerer<'a> {
    pub fn new(
        interner: &'a Interner,
        type_map: TypeMap,
        coercion_map: CoercionMap,
        context_ids: Freeze<FxHashMap<Astr, (ContextId, Ty)>>,
        function_ids: Freeze<FxHashMap<Astr, (FunctionId, Ty)>>,
    ) -> Self {
        let coercion_lookup: FxHashMap<Span, CastKind> = coercion_map.into_iter().collect();
        // Pre-inject function names into the initial scope.
        let mut initial_scope = FxHashMap::default();
        // (function values will be resolved at call sites, not as scope values)
        Self {
            body: MirBody::new(),
            interner,
            scopes: vec![initial_scope],
            type_map,
            coercion_lookup,
            closures: FxHashMap::default(),
            closure_label_count: 0,
            hints: HintTable::new(),
            context_ids,
            function_ids,
            projections: FxHashSet::default(),
        }
    }

    pub fn lower_template(mut self, template: &Template) -> (MirModule, HintTable) {
        let result = self.lower_nodes(&template.body, template.span);
        self.emit_inst(template.span, InstKind::Return(result));
        self.build_module()
    }

    pub fn lower_script(mut self, script: &Script) -> (MirModule, HintTable) {
        for stmt in &script.stmts {
            match stmt {
                Stmt::Bind { name, expr, .. } => {
                    let val = self.lower_expr(expr);
                    self.scopes.last_mut().unwrap().insert(*name, val);
                }
                Stmt::ContextStore { name, expr, span } => {
                    let val = self.lower_expr(expr);
                    let ctx_id = self.context_id(*name);
                    let ctx_ty = self.context_ty(*name);
                    let proj = self.alloc_typed(*span);
                    self.emit_inst(
                        *span,
                        InstKind::ContextProject {
                            dst: proj,
                            id: ctx_id,
                            ty: ctx_ty,
                        },
                    );
                    self.emit_inst(
                        *span,
                        InstKind::ContextStore {
                            dst: proj,
                            value: val,
                        },
                    );
                }
                Stmt::Expr(expr) => {
                    self.lower_expr(expr);
                }
            }
        }
        if let Some(tail) = &script.tail {
            let val = self.lower_expr(tail);
            self.emit_inst(script.span, InstKind::Return(val));
        }
        self.build_module()
    }

    fn build_module(self) -> (MirModule, HintTable) {
        let module = MirModule {
            main: self.body,
            closures: self.closures,
        };
        (module, self.hints)
    }

    fn context_id(&self, name: Astr) -> ContextId {
        self.context_ids
            .get(&name)
            .map(|(id, _)| *id)
            .unwrap_or_else(|| panic!("unknown context @{}", self.interner.resolve(name)))
    }

    fn context_ty(&self, name: Astr) -> Ty {
        self.context_ids
            .get(&name)
            .map(|(_, ty)| ty.clone())
            .unwrap_or_else(|| panic!("unknown context @{}", self.interner.resolve(name)))
    }

    fn function_id(&self, name: Astr) -> FunctionId {
        self.function_ids
            .get(&name)
            .map(|(id, _)| *id)
            .unwrap_or_else(|| panic!("unknown function {}", self.interner.resolve(name)))
    }


    /// If `val` is a projection, materialize it into a value via ContextLoad.
    /// If it's already a value, return as-is.
    fn ensure_loaded(&mut self, span: Span, val: ValueId) -> ValueId {
        if self.projections.remove(&val) {
            let dst = self.alloc_typed(span);
            self.emit_inst(span, InstKind::ContextLoad { dst, src: val });
            dst
        } else {
            val
        }
    }

    /// Mark a ValueId as a projection.
    fn mark_projection(&mut self, val: ValueId) {
        self.projections.insert(val);
    }

    fn is_projection(&self, val: ValueId) -> bool {
        self.projections.contains(&val)
    }

    /// If val is a projection, emit ContextLoad to materialize it.
    /// Otherwise return val unchanged.
    fn materialize(&mut self, val: ValueId, span: Span) -> ValueId {
        if self.is_projection(val) {
            let dst = self.alloc_val();
            let ty = self.body.val_types.get(&val).cloned().unwrap_or(Ty::Unit);
            self.set_val_type(dst, ty);
            self.emit_inst(span, InstKind::ContextLoad { dst, src: val });
            dst
        } else {
            val
        }
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
        self.scopes.pop();
    }

    fn define_var(&mut self, name: Astr, reg: ValueId) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, reg);
        }
    }

    fn lookup_var(&self, name: Astr) -> Option<ValueId> {
        for scope in self.scopes.iter().rev() {
            if let Some(&reg) = scope.get(&name) {
                return Some(reg);
            }
        }
        None
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

    fn list_elem_type(&self, list_val: ValueId) -> Ty {
        if let Some(Ty::List(elem) | Ty::Deque(elem, _)) = self.body.val_types.get(&list_val) {
            elem.as_ref().clone()
        } else {
            Ty::error()
        }
    }

    fn object_field_type(&self, object_val: ValueId, key: Astr) -> Ty {
        if let Some(Ty::Object(fields)) = self.body.val_types.get(&object_val) {
            fields.get(&key).cloned().unwrap_or(Ty::error())
        } else {
            Ty::error()
        }
    }

    fn variant_inner_type(&self, variant_val: ValueId) -> Ty {
        if let Some(Ty::Option(inner)) = self.body.val_types.get(&variant_val) {
            inner.as_ref().clone()
        } else {
            Ty::error()
        }
    }

    fn iterable_elem_type(&self, src_val: ValueId) -> Ty {
        match self.body.val_types.get(&src_val) {
            Some(Ty::List(elem) | Ty::Deque(elem, _)) => elem.as_ref().clone(),
            Some(Ty::Range) => Ty::Int,
            _ => Ty::error(),
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

    /// Allocate a new value with type inferred from span.
    fn alloc_typed(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, self.type_of_span(span));
        dst
    }

    /// Allocate a new value with type inferred from span and origin set to Expr.
    fn alloc_expr(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_typed(span);
        self.set_origin(dst, ValOrigin::Expr);
        dst
    }

    fn emit_list_index(&mut self, span: Span, list: ValueId, index: i32, elem_ty: Ty) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, elem_ty);
        self.emit_inst(span, InstKind::ListIndex { dst, list, index });
        dst
    }

    fn emit_and(&mut self, span: Span, left: ValueId, right: ValueId) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Bool);
        self.emit_inst(
            span,
            InstKind::BinOp {
                dst,
                op: acvus_ast::BinOp::And,
                left,
                right,
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

    fn type_of_span(&self, span: Span) -> Ty {
        self.type_map.get(&span).cloned().unwrap_or(Ty::error())
    }

    // --- Node lowering ---

    /// Lower a sequence of template nodes into a single concatenated String value.
    fn lower_nodes(&mut self, nodes: &[Node], span: Span) -> ValueId {
        let mut acc = self.emit_empty_string(span);
        for node in nodes {
            let val = self.lower_node(node, span);
            acc = self.emit_concat(span, acc, val);
        }
        acc
    }

    /// Lower a single template node, returning a String-typed ValueId.
    fn lower_node(&mut self, node: &Node, parent_span: Span) -> ValueId {
        match node {
            Node::Text { value, span } => {
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
            Node::IterBlock(ib) => self.lower_iter_block(ib),
        }
    }

    fn emit_empty_string(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(span, InstKind::Const { dst, value: Literal::String(String::new()) });
        dst
    }

    fn emit_concat(&mut self, span: Span, left: ValueId, right: ValueId) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(span, InstKind::BinOp { dst, op: BinOp::Add, left, right });
        dst
    }

    /// If the coercion map indicates this span needs a cast, emit a Cast
    /// instruction and return the new ValueId. Otherwise return `val` as-is.
    fn maybe_cast(&mut self, span: Span, val: ValueId) -> ValueId {
        let val = self.materialize(val, span);
        if let Some(&kind) = self.coercion_lookup.get(&span) {
            let src_ty = self
                .body
                .val_types
                .get(&val)
                .cloned()
                .unwrap_or(Ty::error());
            let dst_ty = kind.result_ty(&src_ty);
            let cast_dst = self.alloc_val();
            self.set_val_type(cast_dst, dst_ty);
            self.emit_inst(
                span,
                InstKind::Cast {
                    dst: cast_dst,
                    src: val,
                    kind,
                },
            );
            cast_dst
        } else {
            val
        }
    }

    // --- Expression lowering ---

    /// Lower an expression to a **value** (not a projection).
    /// If the result is a projection, it is materialized via ContextLoad.
    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        let val = self.maybe_cast(expr.span(), val);
        self.ensure_loaded(expr.span(), val)
    }

    /// Lower an expression, preserving projections (no automatic ContextLoad).
    /// Used by FieldAccess and ContextStore where projection must be maintained.
    fn lower_expr_projectable(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        self.maybe_cast(expr.span(), val)
    }

    fn lower_expr_inner(&mut self, expr: &Expr) -> ValueId {
        match expr {
            Expr::Literal { value, span } => {
                let dst = self.alloc_expr(*span);
                self.emit_inst(
                    *span,
                    InstKind::Const {
                        dst,
                        value: value.clone(),
                    },
                );
                dst
            }

            Expr::Ident {
                name,
                ref_kind,
                span,
            } => match ref_kind {
                RefKind::Context => {
                    let ctx_id = self.context_id(*name);
                    let ctx_ty = self.context_ty(*name);
                    let dst = self.alloc_typed(*span);
                    self.set_origin(dst, ValOrigin::Context(*name));
                    self.emit_inst(
                        *span,
                        InstKind::ContextProject {
                            dst,
                            id: ctx_id,
                            ty: ctx_ty,
                        },
                    );
                    self.mark_projection(dst);
                    dst
                }
                RefKind::Variable => {
                    if let Some(reg) = self.lookup_var(*name) {
                        return reg;
                    }
                    let dst = self.alloc_typed(*span);
                    self.set_origin(dst, ValOrigin::Variable(*name));
                    self.emit_inst(*span, InstKind::VarLoad { dst, name: *name });
                    dst
                }
                RefKind::Value => {
                    if let Some(reg) = self.lookup_var(*name) {
                        return reg;
                    }
                    let dst = self.alloc_val();
                    self.set_val_type(dst, Ty::Unit);
                    dst
                }
            },

            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                let dst = self.alloc_expr(*span);
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

            Expr::UnaryOp { op, operand, span } => {
                let o = self.lower_expr(operand);
                let dst = self.alloc_expr(*span);
                self.emit_inst(
                    *span,
                    InstKind::UnaryOp {
                        dst,
                        op: *op,
                        operand: o,
                    },
                );
                dst
            }

            Expr::FieldAccess {
                object,
                field,
                span,
            } => {
                let obj = self.lower_expr_projectable(object);
                let is_proj = self.is_projection(obj);
                let dst = self.alloc_val();
                self.set_val_type(dst, self.type_of_span(*span));
                self.set_origin(dst, ValOrigin::Field(obj, *field));
                self.emit_inst(
                    *span,
                    InstKind::FieldGet {
                        dst,
                        object: obj,
                        field: *field,
                    },
                );
                // Projection propagates through field access.
                if is_proj {
                    self.mark_projection(dst);
                }
                dst
            }

            Expr::FuncCall { func, args, span } => self.lower_func_call(func, args, None, *span),

            Expr::Pipe { left, right, span } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`, `a | f` → `f(a)`
                match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.lower_func_call(func, args, Some(left), *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    } => self.lower_func_call(right, &[], Some(left), *span),
                    _ => {
                        // Fallback: evaluate both sides, call as indirect.
                        let l = self.lower_expr(left);
                        let r = self.lower_expr(right);
                        let dst = self.alloc_typed(*span);
                        self.emit_inst(
                            *span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Indirect(r),
                                args: vec![l],
                            },
                        );
                        dst
                    }
                }
            }

            Expr::Lambda { params, body, span } => {
                // Capture analysis: find free variables in body.
                let param_names: FxHashSet<Astr> = params.iter().map(|p| p.name).collect();
                let free_vars = self.free_vars_in_expr(body, &param_names);

                let capture_regs: Vec<ValueId> = free_vars
                    .iter()
                    .filter_map(|name| self.lookup_var(*name))
                    .collect();
                // Create closure body.
                let closure_label = self.alloc_closure_label();

                // Build the closure body MIR in a sub-lowerer.
                let mut sub_body = MirBody::new();
                let mut sub_scopes = vec![FxHashMap::default()];

                // Captures become the first registers.
                let mut closure_capture_regs = Vec::new();
                for (i, name) in free_vars.iter().enumerate() {
                    let reg = sub_body.val_factory.next();
                    closure_capture_regs.push(reg);
                    sub_scopes[0].insert(*name, reg);
                    if let Some(outer_reg) = capture_regs.get(i)
                        && let Some(ty) = self.body.val_types.get(outer_reg)
                    {
                        sub_body.val_types.insert(reg, ty.clone());
                    }
                }

                // Params follow captures.
                let mut closure_param_regs = Vec::new();
                for p in params.iter() {
                    let reg = sub_body.val_factory.next();
                    closure_param_regs.push(reg);
                    sub_scopes[0].insert(p.name, reg);
                    let ty = self.type_of_span(p.span);
                    sub_body.val_types.insert(reg, ty);
                }

                // We need to lower the body in context of the sub-body.
                // Swap state.
                let saved_body = std::mem::replace(&mut self.body, sub_body);
                let saved_scopes = std::mem::replace(&mut self.scopes, sub_scopes);

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
                self.emit_inst(*span, InstKind::Return(result_reg));

                let mut closure_body_mir = std::mem::replace(&mut self.body, saved_body);
                self.scopes = saved_scopes;

                closure_body_mir.capture_regs = closure_capture_regs;
                closure_body_mir.param_regs = closure_param_regs;
                self.closures.insert(closure_label, Arc::new(closure_body_mir));

                // Allocate dst with Fn type. If a return-site Cast was inserted,
                // update the Fn's ret to match the actual (cast) return type.
                let dst = self.alloc_val();
                let mut fn_ty = self.type_of_span(*span);
                if let Ty::Fn { ref mut ret, .. } = fn_ty {
                    *ret = Box::new(actual_ret_ty);
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

            Expr::List {
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
                let dst = self.alloc_typed(*span);
                self.emit_inst(*span, InstKind::MakeDeque { dst, elements });
                dst
            }

            Expr::Object { fields, span } => {
                let field_regs = fields
                    .iter()
                    .map(|ObjectExprField { key, value, .. }| {
                        let r = self.lower_expr(value);
                        (*key, r)
                    })
                    .collect();
                let dst = self.alloc_typed(*span);
                self.emit_inst(
                    *span,
                    InstKind::MakeObject {
                        dst,
                        fields: field_regs,
                    },
                );
                dst
            }

            Expr::Range {
                start,
                end,
                kind,
                span,
            } => {
                let s = self.lower_expr(start);
                let e = self.lower_expr(end);
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Range);
                self.emit_inst(
                    *span,
                    InstKind::MakeRange {
                        dst,
                        start: s,
                        end: e,
                        kind: *kind,
                    },
                );
                dst
            }

            Expr::Tuple { elements, span } => {
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
                let dst = self.alloc_typed(*span);
                self.emit_inst(
                    *span,
                    InstKind::MakeTuple {
                        dst,
                        elements: elem_vals,
                    },
                );
                dst
            }

            Expr::Group { elements, span } => {
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
                tag, payload, span, ..
            } => {
                let payload_val = payload.as_ref().map(|e| self.lower_expr(e));
                let dst = self.alloc_expr(*span);
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
                    match stmt {
                        Stmt::Bind { name, expr, .. } => {
                            let val = self.lower_expr(expr);
                            self.scopes.last_mut().unwrap().insert(*name, val);
                        }
                        Stmt::ContextStore { name, expr, span } => {
                            self.lower_context_store(*name, expr, *span);
                        }
                        Stmt::Expr(expr) => {
                            self.lower_expr(expr);
                        }
                    }
                }
                let val = self.lower_expr(tail);
                self.pop_scope();
                val
            }
        }
    }

    fn lower_context_store(
        &mut self,
        name: Astr,
        value_expr: &Expr,
        span: Span,
    ) -> ValueId {
        let val = self.lower_expr(value_expr);
        let ctx_id = self.context_id(name);
        let ctx_ty = self.context_ty(name);
        let proj = self.alloc_typed(span);
        self.emit_inst(
            span,
            InstKind::ContextProject {
                dst: proj,
                id: ctx_id,
                ty: ctx_ty,
            },
        );
        self.emit_inst(
            span,
            InstKind::ContextStore {
                dst: proj,
                value: val,
            },
        );
        val
    }

    fn lower_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Box<Expr>>,
        call_span: Span,
    ) -> ValueId {
        let mut arg_regs: Vec<ValueId> =
            Vec::with_capacity(args.len() + pipe_left.is_some() as usize);
        if let Some(left) = pipe_left {
            let val = self.lower_expr(left);
            arg_regs.push(self.materialize(val, call_span));
        }
        for a in args {
            let val = self.lower_expr(a);
            arg_regs.push(self.materialize(val, call_span));
        }
        let dst = self.alloc_typed(call_span);

        // Named function call (Ident).
        if let Expr::Ident { name, ref_kind, .. } = func {
            match ref_kind {
                // @fn_name(args) — context-based function (legacy, will be migrated)
                RefKind::Context => {
                    let fn_id = self.function_ids.get(name).map(|(id, _)| *id);
                    if let Some(fn_id) = fn_id {
                        self.set_origin(dst, ValOrigin::Call(*name));
                        self.emit_inst(
                            call_span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Direct(fn_id),
                                args: arg_regs,
                            },
                        );
                        return dst;
                    }
                }
                // fn_name(args) — named call
                RefKind::Value => {
                    self.set_origin(dst, ValOrigin::Call(*name));

                    // 1. Graph function (Direct call)
                    let fn_id = self.function_ids.get(name).map(|(id, _)| *id);
                    if let Some(fn_id) = fn_id {
                        self.emit_inst(
                            call_span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Direct(fn_id),
                                args: arg_regs,
                            },
                        );
                        return dst;
                    }

                    // 2. Local variable (closure/lambda — Indirect call)
                    if let Some(closure_reg) = self.lookup_var(*name) {
                        self.emit_inst(
                            call_span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Indirect(closure_reg),
                                args: arg_regs,
                            },
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

        // Expression call (e.g., (x -> x)(42), or complex pipe)
        self.set_origin(dst, ValOrigin::Call(self.interner.intern("<closure>")));
        let func_reg = self.lower_expr(func);
        self.emit_inst(
            call_span,
            InstKind::FunctionCall {
                dst,
                callee: Callee::Indirect(func_reg),
                args: arg_regs,
            },
        );
        dst
    }

    // --- Match block lowering ---

    fn lower_match_block(&mut self, mb: &MatchBlock) -> ValueId {
        // Body-less binding shorthand (variable write or value binding).
        if mb.arms.len() == 1
            && mb.arms[0].body.is_empty()
            && let Pattern::Binding {
                name,
                ref_kind,
                span: pat_span,
            } = &mb.arms[0].pattern
        {
            let src = self.lower_expr(&mb.source);
            match ref_kind {
                RefKind::Variable => {
                    self.emit_inst(*pat_span, InstKind::VarStore { name: *name, src });
                }
                RefKind::Context => {
                    let ctx_id = self.context_id(*name);
                    let ctx_ty = self.context_ty(*name);
                    let proj = self.alloc_typed(*pat_span);
                    self.emit_inst(
                        *pat_span,
                        InstKind::ContextProject {
                            dst: proj,
                            id: ctx_id,
                            ty: ctx_ty,
                        },
                    );
                    self.emit_inst(
                        *pat_span,
                        InstKind::ContextStore {
                            dst: proj,
                            value: src,
                        },
                    );
                }
                RefKind::Value => {
                    self.define_var(*name, src);
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

        let source_reg = self.lower_expr(&mb.source);

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
            let matched = self.lower_pattern_test(&arm.pattern, source_reg, arm.tag_span);

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
            self.lower_pattern_bind(&arm.pattern, source_reg, arm.tag_span);

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

    // --- Iter block lowering ---

    fn lower_iter_block(&mut self, ib: &IterBlock) -> ValueId {
        // Pre-compute indent-adjusted body and catch-all body.
        let adjusted_body: Option<Vec<Node>> = ib
            .indent
            .as_ref()
            .map(|modifier| apply_indent_to_nodes(&ib.body, modifier));
        let adjusted_catch_all_body: Option<Vec<Node>> = ib.indent.as_ref().and_then(|modifier| {
            ib.catch_all
                .as_ref()
                .map(|ca| apply_indent_to_nodes(&ca.body, modifier))
        });

        let source_raw = self.lower_expr(&ib.source);
        let source_reg = self.materialize(source_raw, ib.span);

        // Determine element type and CastKind from the source type.
        let elem_ty = self.iterable_elem_type(source_reg);
        let iter_ty = Ty::Iterator(Box::new(elem_ty.clone()), Effect::pure());
        let cast_kind = match self.body.val_types.get(&source_reg) {
            Some(Ty::Deque(..)) => CastKind::DequeToIterator,
            Some(Ty::List(_)) => CastKind::ListToIterator,
            Some(Ty::Range) => CastKind::RangeToIterator,
            _ => CastKind::DequeToIterator, // fallback — type checker should prevent this
        };

        // Cast source → Iterator.
        let cast_dst = self.alloc_val();
        self.set_val_type(cast_dst, iter_ty.clone());
        self.emit_inst(
            ib.span,
            InstKind::Cast {
                dst: cast_dst,
                src: source_reg,
                kind: cast_kind,
            },
        );

        let loop_label = self.alloc_label();
        let catch_all_label = self.alloc_label();
        let end_label = self.alloc_label();

        // Initial accumulator: empty string.
        let init_acc = self.emit_empty_string(ib.span);

        // Jump to loop with initial iterator + accumulator.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: loop_label,
                args: vec![cast_dst, init_acc],
            },
        );

        // Loop header — receives iterator + accumulator as block params.
        let iter_param = self.alloc_val();
        self.set_val_type(iter_param, iter_ty.clone());
        let acc_param = self.alloc_val();
        self.set_val_type(acc_param, Ty::String);
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: loop_label,
                params: vec![iter_param, acc_param],
                merge_of: None,
            },
        );

        // Pull one element — jumps to catch_all if exhausted.
        let value_reg = self.alloc_val();
        self.set_val_type(value_reg, elem_ty);
        let rest_iter = self.alloc_val();
        self.set_val_type(rest_iter, iter_ty);
        self.emit_inst(
            ib.span,
            InstKind::IterStep {
                dst: value_reg,
                iter_src: iter_param,
                iter_dst: rest_iter,
                done: catch_all_label,
                done_args: vec![acc_param],
            },
        );

        // Bind pattern (irrefutable — no test needed).
        self.push_scope();
        self.lower_pattern_bind(&ib.pattern, value_reg, ib.span);

        // Lower body — concat all body nodes into a single string.
        let body = adjusted_body.as_deref().unwrap_or(&ib.body);
        let body_result = self.lower_nodes(body, ib.span);

        // Accumulate: new_acc = acc_param + body_result.
        let new_acc = self.emit_concat(ib.span, acc_param, body_result);

        self.pop_scope();

        // Jump back to loop with rest iterator + new accumulator.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: loop_label,
                args: vec![rest_iter, new_acc],
            },
        );

        // Catch-all block — receives final accumulator.
        let catch_all_acc = self.alloc_val();
        self.set_val_type(catch_all_acc, Ty::String);
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: catch_all_label,
                params: vec![catch_all_acc],
                merge_of: None,
            },
        );
        let final_acc = if let Some(catch_all) = &ib.catch_all {
            self.push_scope();
            let ca_body = adjusted_catch_all_body
                .as_deref()
                .unwrap_or(&catch_all.body);
            let ca_result = self.lower_nodes(ca_body, ib.span);
            self.pop_scope();
            self.emit_concat(ib.span, catch_all_acc, ca_result)
        } else {
            catch_all_acc
        };

        // Jump to end with final result.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: end_label,
                args: vec![final_acc],
            },
        );

        // End block — receives the final string.
        let end_result = self.alloc_val();
        self.set_val_type(end_result, Ty::String);
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: end_label,
                params: vec![end_result],
                merge_of: None,
            },
        );
        end_result
    }

    // --- Pattern test lowering ---

    /// Emit instructions that test whether `src_reg` matches `pattern`.
    /// Returns a register holding a Bool (true = match).
    fn lower_pattern_test(&mut self, pattern: &Pattern, src_reg: ValueId, span: Span) -> ValueId {
        match pattern {
            Pattern::Binding { name, ref_kind, .. } => {
                if *ref_kind == RefKind::Variable {
                    self.emit_inst(
                        span,
                        InstKind::VarStore {
                            name: *name,
                            src: src_reg,
                        },
                    );
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

            Pattern::List {
                head, rest, tail, ..
            } => {
                let min_len = head.len() + tail.len();
                let exact = rest.is_none();

                let len_ok = self.alloc_val();
                self.set_val_type(len_ok, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestListLen {
                        dst: len_ok,
                        src: src_reg,
                        min_len,
                        exact,
                    },
                );

                // If length doesn't match, short-circuit.
                let check_elems_label = self.alloc_label();
                let fail_label = self.alloc_label();

                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: len_ok,
                        then_label: check_elems_label,
                        then_args: vec![],
                        else_label: fail_label,
                        else_args: vec![],
                    },
                );
                self.emit_label(span, check_elems_label);

                // Check each head element.
                let mut all_ok = len_ok;
                let elem_ty = self.list_elem_type(src_reg);
                for (i, p) in head.iter().enumerate() {
                    let elem = self.emit_list_index(span, src_reg, i as i32, elem_ty.clone());
                    let value_id = self.lower_pattern_test(p, elem, span);
                    all_ok = self.emit_and(span, all_ok, value_id);
                }

                // Check each tail element (indexed from end).
                for (i, p) in tail.iter().enumerate() {
                    let elem = self.emit_list_index(
                        span,
                        src_reg,
                        -((tail.len() - i) as i32),
                        elem_ty.clone(),
                    );
                    let value_id = self.lower_pattern_test(p, elem, span);
                    all_ok = self.emit_and(span, all_ok, value_id);
                }

                self.emit_fail_merge(span, all_ok, fail_label)
            }

            Pattern::Object { fields, .. } => {
                // Test that all keys exist.
                let mut all_ok = self.emit_const_bool(span, true);

                for ObjectPatternField { key, pattern, .. } in fields {
                    let key_ok = self.alloc_val();
                    self.set_val_type(key_ok, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::TestObjectKey {
                            dst: key_ok,
                            src: src_reg,
                            key: *key,
                        },
                    );

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

                    let sub_ok = self.lower_pattern_test(pattern, field_val, span);
                    all_ok = self.emit_and(span, all_ok, sub_ok);
                }

                all_ok
            }

            Pattern::Range {
                start, end, kind, ..
            } => {
                let (start_val, end_val) = self.extract_range_bounds(start, end);
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::TestRange {
                        dst,
                        src: src_reg,
                        start: start_val,
                        end: end_val,
                        kind: *kind,
                    },
                );
                dst
            }

            Pattern::Tuple { elements, .. } => {
                // Tuple length is guaranteed by the type system — always matches.
                // Test each sub-pattern element.
                let mut all_ok = self.emit_const_bool(span, true);

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
                    let value_id = self.lower_pattern_test(pat, field_val, span);
                    all_ok = self.emit_and(span, all_ok, value_id);
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
                    // No payload (e.g. None) — tag test is the final result.
                    return tag_ok;
                };

                // Has payload — short-circuit: if tag fails, skip inner test.
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
                self.set_val_type(inner_val, self.variant_inner_type(src_reg));
                self.emit_inst(
                    span,
                    InstKind::UnwrapVariant {
                        dst: inner_val,
                        src: src_reg,
                    },
                );
                let inner_ok = self.lower_pattern_test(inner_pat, inner_val, span);

                self.emit_fail_merge(span, inner_ok, fail_label)
            }
        }
    }

    fn extract_range_bounds(&self, start: &Pattern, end: &Pattern) -> (i64, i64) {
        let extract = |p: &Pattern| match p {
            Pattern::Literal {
                value: Literal::Int(n),
                ..
            } => *n,
            _ => 0,
        };
        (extract(start), extract(end))
    }

    /// Emit instructions that bind pattern variables from a matched value.
    fn lower_pattern_bind(&mut self, pattern: &Pattern, src_reg: ValueId, span: Span) {
        match pattern {
            Pattern::Binding {
                name,
                ref_kind: RefKind::Value,
                ..
            } => {
                self.set_origin(src_reg, ValOrigin::Named(*name));
                self.define_var(*name, src_reg);
            }
            Pattern::Binding { .. } | Pattern::Literal { .. } => {}

            Pattern::List {
                head,
                rest: _,
                tail,
                ..
            } => {
                let elem_ty = self.list_elem_type(src_reg);
                for (i, p) in head.iter().enumerate() {
                    let elem = self.emit_list_index(span, src_reg, i as i32, elem_ty.clone());
                    self.lower_pattern_bind(p, elem, span);
                }
                for (i, p) in tail.iter().enumerate() {
                    let elem = self.emit_list_index(
                        span,
                        src_reg,
                        -((tail.len() - i) as i32),
                        elem_ty.clone(),
                    );
                    self.lower_pattern_bind(p, elem, span);
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
                    self.lower_pattern_bind(pattern, field_val, span);
                }
            }

            Pattern::Range { .. } => {}

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
                    self.lower_pattern_bind(pat, field_val, span);
                }
            }

            Pattern::Variant { payload, .. } => {
                let Some(inner_pat) = payload else {
                    return;
                };
                let inner_val = self.alloc_val();
                self.set_val_type(inner_val, self.variant_inner_type(src_reg));
                self.emit_inst(
                    span,
                    InstKind::UnwrapVariant {
                        dst: inner_val,
                        src: src_reg,
                    },
                );
                self.lower_pattern_bind(inner_pat, inner_val, span);
            }
        }
    }

    // --- Free variable analysis ---

    fn free_vars_in_expr(&self, expr: &Expr, bound: &FxHashSet<Astr>) -> Vec<Astr> {
        let mut free = Vec::new();
        let mut seen = FxHashSet::default();
        self.collect_free_vars(expr, bound, &mut free, &mut seen);
        free
    }

    fn collect_free_vars(
        &self,
        expr: &Expr,
        bound: &FxHashSet<Astr>,
        free: &mut Vec<Astr>,
        seen: &mut FxHashSet<Astr>,
    ) {
        match expr {
            Expr::Ident {
                name,
                ref_kind: RefKind::Value,
                ..
            } => {
                if bound.contains(name) || seen.contains(name) {
                    return;
                }
                if self.lookup_var(*name).is_some() {
                    seen.insert(*name);
                    free.push(*name);
                }
            }
            // Variable/context refs resolve at runtime — no capture needed.
            Expr::Ident {
                ref_kind: RefKind::Variable | RefKind::Context,
                ..
            } => {}
            Expr::BinaryOp { left, right, .. } => {
                self.collect_free_vars(left, bound, free, seen);
                self.collect_free_vars(right, bound, free, seen);
            }
            Expr::UnaryOp { operand, .. } => {
                self.collect_free_vars(operand, bound, free, seen);
            }
            Expr::FieldAccess { object, .. } => {
                self.collect_free_vars(object, bound, free, seen);
            }
            Expr::FuncCall { func, args, .. } => {
                self.collect_free_vars(func, bound, free, seen);
                for a in args {
                    self.collect_free_vars(a, bound, free, seen);
                }
            }
            Expr::Pipe { left, right, .. } => {
                self.collect_free_vars(left, bound, free, seen);
                self.collect_free_vars(right, bound, free, seen);
            }
            Expr::Lambda { params, body, .. } => {
                let mut inner_bound = bound.clone();
                for p in params {
                    inner_bound.insert(p.name);
                }
                self.collect_free_vars(body, &inner_bound, free, seen);
            }
            Expr::Paren { inner, .. } => {
                self.collect_free_vars(inner, bound, free, seen);
            }
            Expr::List { head, tail, .. } => {
                for e in head.iter().chain(tail.iter()) {
                    self.collect_free_vars(e, bound, free, seen);
                }
            }
            Expr::Object { fields, .. } => {
                for f in fields {
                    self.collect_free_vars(&f.value, bound, free, seen);
                }
            }
            Expr::Range { start, end, .. } => {
                self.collect_free_vars(start, bound, free, seen);
                self.collect_free_vars(end, bound, free, seen);
            }
            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    let TupleElem::Expr(e) = elem else { continue };
                    self.collect_free_vars(e, bound, free, seen);
                }
            }
            Expr::Group { elements, .. } => {
                for e in elements {
                    self.collect_free_vars(e, bound, free, seen);
                }
            }
            Expr::Literal { .. } => {}
            Expr::Variant {
                payload: Some(inner),
                ..
            } => {
                self.collect_free_vars(inner, bound, free, seen);
            }
            Expr::Variant { payload: None, .. } => {}
            Expr::Block { stmts, tail, .. } => {
                let mut inner_bound = bound.clone();
                for stmt in stmts {
                    match stmt {
                        Stmt::Bind { name, expr, .. } => {
                            self.collect_free_vars(expr, &inner_bound, free, seen);
                            inner_bound.insert(*name);
                        }
                        Stmt::ContextStore { expr, .. } => {
                            self.collect_free_vars(expr, &inner_bound, free, seen);
                        }
                        Stmt::Expr(e) => {
                            self.collect_free_vars(e, &inner_bound, free, seen);
                        }
                    }
                }
                self.collect_free_vars(tail, &inner_bound, free, seen);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph::ContextId, ty::TySubst, typeck::TypeChecker};

    fn lower(interner: &Interner, source: &str) -> MirModule {
        lower_with(interner, source, &FxHashMap::default())
    }

    fn lower_with(interner: &Interner, source: &str, context: &FxHashMap<Astr, Ty>) -> MirModule {
        let ctx: Vec<(&str, Ty)> = context
            .iter()
            .map(|(name, ty)| (interner.resolve(*name), ty.clone()))
            .collect();
        let (module, _) =
            crate::test::compile_template(interner, source, &ctx).expect("compile failed");
        module
    }

    #[test]
    fn lower_text_node() {
        let interner = Interner::new();
        let module = lower(&interner, "hello world");
        // Template: empty_str const + text const + concat + return
        let has_text = module.main.insts.iter().any(|i| matches!(
            &i.kind,
            InstKind::Const { value: Literal::String(s), .. } if s == "hello world"
        ));
        let has_return = module.main.insts.iter().any(|i| matches!(&i.kind, InstKind::Return(_)));
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
    fn lower_var_write() {
        let interner = Interner::new();
        let module = lower(&interner, "{{ $count = 42 }}");
        let has_store = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::VarStore { name, .. } if interner.resolve(*name) == "count"));
        assert!(has_store);
    }

    #[test]
    fn lower_match_block() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("name"), Ty::String)]);
        // Use a non-binding pattern to trigger full match block (no iteration).
        let module = lower_with(
            &interner,
            r#"{{ true = @name == "test" }}matched{{/}}"#,
            &context,
        );
        // Should have pattern test and conditional jump.
        let has_jump_if = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::JumpIf { .. }));
        assert!(has_jump_if);
    }

    #[test]
    #[ignore = "requires Phase 2: builtin → graph Function migration"]
    fn lower_builtin_call() {
        let interner = Interner::new();
        let context = FxHashMap::from_iter([(interner.intern("n"), Ty::Int)]);
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
        let context = FxHashMap::from_iter([(interner.intern("name"), Ty::String)]);
        let source = "{{ true = @name == \"test\" }}\n    matched\n    here{{/-2}}";
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
        let context = FxHashMap::from_iter([(interner.intern("name"), Ty::String)]);
        let source = "{{ true = @name == \"test\" }}\nmatched{{/+4}}";
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
