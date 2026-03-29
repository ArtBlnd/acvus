use acvus_ast::{
    AstId, BinOp, Expr, IndentModifier, IterBlock, Literal, MatchBlock, Node, ObjectExprField,
    ObjectPatternField, Pattern, RefKind, Script, Span, Stmt, Template, TupleElem,
    TuplePatternElem,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::QualifiedRef;
use crate::hints::HintTable;
use crate::ir::{Callee, CastKind, Inst, InstKind, Label, MirBody, MirModule, ValOrigin, ValueId};
use crate::ty::{Ty};
use crate::typeck::{CoercionMap, TypeMap};

pub struct Lowerer<'a> {
    body: MirBody,
    /// Interner for string interning.
    interner: &'a Interner,
    /// Stack of scopes: variable name → type (for validity, capture, and VarLoad typing).
    scopes: Vec<FxHashMap<Astr, Ty>>,
    /// Type map from type checker.
    type_map: TypeMap,
    /// Coercion map from type checker (expr AstId → CastKind).
    coercion_lookup: FxHashMap<AstId, CastKind>,
    /// Closures produced during lowering.
    closures: FxHashMap<Label, MirBody>,
    /// Global closure label counter — shared across nesting levels to prevent
    /// label collisions when nested closures each allocate from a sub-body.
    closure_label_count: u32,
    /// Hint table.
    hints: HintTable,
    /// Context QualifiedRef → Ty.
    context_ids: Freeze<FxHashMap<QualifiedRef, Ty>>,
    /// Function name → (QualifiedRef, Ty).
    function_ids: Freeze<FxHashMap<Astr, (QualifiedRef, Ty)>>,
    /// ValueIds that are projections (not yet materialized values).
    /// FieldAccess on a projection produces another projection.
    /// Use `ensure_loaded` to materialize.
    /// Value: whether the projection is volatile.
    projections: FxHashMap<ValueId, bool>,
    /// Contexts whose loads/stores must not be elided by SSA (externally observable).
    volatile_contexts: FxHashSet<QualifiedRef>,
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
            Node::IterBlock(ib) => Node::IterBlock(acvus_ast::IterBlock {
                id: acvus_ast::AstId::alloc(),
                pattern: ib.pattern.clone(),
                source: ib.source.clone(),
                body: apply_indent_to_nodes(&ib.body, modifier),
                catch_all: ib.catch_all.as_ref().map(|ca| acvus_ast::CatchAll {
                    id: acvus_ast::AstId::alloc(),
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
        context_ids: Freeze<FxHashMap<QualifiedRef, Ty>>,
        function_ids: Freeze<FxHashMap<Astr, (QualifiedRef, Ty)>>,
        volatile_contexts: FxHashSet<QualifiedRef>,
    ) -> Self {
        let coercion_lookup: FxHashMap<AstId, CastKind> = coercion_map.into_iter().collect();
        // Pre-inject function names into the initial scope.
        let initial_scope = FxHashMap::default();
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
            projections: FxHashMap::default(),
            volatile_contexts,
        }
    }

    /// Emit ContextProject + ContextLoad for all known contexts at entry.
    /// This is the alloca equivalent — SSA pass will promote these to PHI form.
    /// Order is deterministic (sorted by context name).
    fn emit_entry_context_loads(&mut self, span: Span) {
        let mut entries: Vec<_> = self
            .context_ids
            .iter()
            .map(|(&qref, ty)| (qref, ty.clone()))
            .collect();
        entries.sort_by_key(|(qref, _)| self.interner.resolve(qref.name).to_string());

        for (qref, ty) in entries {
            let volatile = self.is_volatile_ctx(&qref);
            let proj = self.alloc_val();
            self.set_val_type(proj, ty.clone());
            self.set_origin(proj, ValOrigin::Context(qref.name));
            self.emit_inst(
                span,
                InstKind::ContextProject {
                    dst: proj,
                    ctx: qref,
                    volatile,
                },
            );
            self.mark_projection(proj, volatile);
            let val = self.alloc_val();
            self.set_val_type(val, ty);
            self.emit_inst(
                span,
                InstKind::ContextLoad {
                    dst: val,
                    src: proj,
                    volatile,
                },
            );
        }
    }

    pub fn lower_template(mut self, template: &Template) -> (MirModule, HintTable) {
        self.emit_entry_context_loads(template.span);
        let result = self.lower_nodes(&template.body, template.span);
        self.emit_inst(template.span, InstKind::Return(result));
        self.build_module()
    }

    pub fn lower_script(mut self, script: &Script) -> (MirModule, HintTable) {
        self.emit_entry_context_loads(script.span);
        for stmt in &script.stmts {
            self.lower_stmt(stmt);
        }
        if let Some(tail) = &script.tail {
            let val = self.lower_expr(tail);
            self.emit_inst(script.span, InstKind::Return(val));
        }
        self.build_module()
    }

    fn lower_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Bind { name, expr, span, .. } => {
                let val = self.lower_expr(expr);
                let ty = self
                    .body
                    .val_types
                    .get(&val)
                    .cloned()
                    .unwrap_or(Ty::error());
                self.emit_inst(
                    *span,
                    InstKind::VarStore {
                        name: *name,
                        src: val,
                    },
                );
                self.define_var(*name, ty);
            }
            Stmt::ContextStore { name, expr, span, .. } => {
                self.lower_context_store(*name, expr, *span); // name is QualifiedRef
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
            Stmt::Iterate {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                self.lower_stmt_iterate(pattern, source, body, *span);
            }
        }
    }

    /// Lower a match-bind statement: `pattern = source { body; };`
    ///
    /// Single arm, no catch-all, no value produced.
    /// If the pattern is irrefutable (binding), skip the test.
    fn lower_stmt_match_bind(
        &mut self,
        pattern: &Pattern,
        source: &Expr,
        body: &[Stmt],
        span: Span,
    ) {
        let source_reg = self.lower_expr(source);

        let is_irrefutable = matches!(
            pattern,
            Pattern::Binding { .. } | Pattern::ContextBind { .. }
        );

        if is_irrefutable {
            // No branching needed — just bind and execute body.
            self.push_scope();
            self.lower_pattern_bind(pattern, source_reg, span);
            for s in body {
                self.lower_stmt(s);
            }
            self.pop_scope();
        } else {
            // Refutable: test → branch → bind + body → merge.
            let body_label = self.alloc_label();
            let end_label = self.alloc_label();

            let matched = self.lower_pattern_test(pattern, source_reg, span);
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
            self.lower_pattern_bind(pattern, source_reg, span);
            for s in body {
                self.lower_stmt(s);
            }
            self.pop_scope();
            self.emit_inst(
                span,
                InstKind::Jump {
                    label: end_label,
                    args: vec![],
                },
            );

            self.emit_label(span, end_label);
        }
    }

    /// Lower an iterate statement: `pattern in source { body; };`
    ///
    /// No string accumulation — just executes body for each element.
    fn lower_stmt_iterate(&mut self, pattern: &Pattern, source: &Expr, body: &[Stmt], span: Span) {
        let source_raw = self.lower_expr(source);
        let source_reg = self.materialize(source_raw, span);

        let elem_ty = self.iterable_elem_type(source_reg);

        // Convert source to List if needed (Deque→List, Range→List).
        let list_reg = match self.body.val_types.get(&source_reg) {
            Some(Ty::List(_)) => source_reg,
            Some(Ty::Deque(..)) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(elem_ty.clone())));
                self.emit_inst(span, InstKind::Cast { dst: cast_dst, src: source_reg, kind: CastKind::DequeToList });
                cast_dst
            }
            Some(Ty::Range) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(Ty::Int)));
                self.emit_inst(span, InstKind::Cast { dst: cast_dst, src: source_reg, kind: CastKind::RangeToList });
                cast_dst
            }
            _ => source_reg,
        };

        // Initial index = 0.
        let zero = self.alloc_val();
        self.set_val_type(zero, Ty::Int);
        self.emit_inst(span, InstKind::Const { dst: zero, value: Literal::Int(0) });

        let loop_label = self.alloc_label();
        let end_label = self.alloc_label();

        // Jump to loop with initial index.
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![zero],
            },
        );

        // Loop header — receives index as block param.
        let index_param = self.alloc_val();
        self.set_val_type(index_param, Ty::Int);
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: loop_label,
                params: vec![index_param],
                merge_of: None,
            },
        );

        // ListStep — if index >= len, jump to end; else dst = list[index].
        let value_reg = self.alloc_val();
        self.set_val_type(value_reg, elem_ty);
        let next_index = self.alloc_val();
        self.set_val_type(next_index, Ty::Int);
        self.emit_inst(
            span,
            InstKind::ListStep {
                dst: value_reg,
                list: list_reg,
                index_src: index_param,
                index_dst: next_index,
                done: end_label,
                done_args: vec![],
            },
        );

        // Body label after ListStep (ListStep is a terminator in CFG).
        let body_label = self.alloc_label();
        self.emit_label(span, body_label);

        // Bind pattern + execute body.
        self.push_scope();
        self.lower_pattern_bind(pattern, value_reg, span);
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();

        // Jump back to loop with next index.
        self.emit_inst(
            span,
            InstKind::Jump {
                label: loop_label,
                args: vec![next_index],
            },
        );

        // End block.
        self.emit_inst(
            span,
            InstKind::BlockLabel {
                label: end_label,
                params: vec![],
                merge_of: None,
            },
        );
    }

    fn build_module(self) -> (MirModule, HintTable) {
        let module = MirModule {
            main: self.body,
            closures: self.closures,
        };
        (module, self.hints)
    }

    /// If `val` is a projection, materialize it into a value via ContextLoad.
    /// If it's already a value, return as-is.
    fn ensure_loaded(&mut self, span: Span, val: ValueId) -> ValueId {
        if let Some(volatile) = self.projections.remove(&val) {
            let dst = self.alloc_val();
            let ty = self.body.val_types.get(&val).cloned().unwrap_or(Ty::error());
            self.set_val_type(dst, ty);
            self.emit_inst(span, InstKind::ContextLoad { dst, src: val, volatile });
            dst
        } else {
            val
        }
    }

    /// Mark a ValueId as a projection with its volatile flag.
    fn mark_projection(&mut self, val: ValueId, volatile: bool) {
        self.projections.insert(val, volatile);
    }

    fn is_projection(&self, val: ValueId) -> bool {
        self.projections.contains_key(&val)
    }

    fn is_volatile_ctx(&self, ctx: &QualifiedRef) -> bool {
        self.volatile_contexts.contains(ctx)
    }

    /// If val is a projection, emit ContextLoad to materialize it.
    /// Otherwise return val unchanged.
    fn materialize(&mut self, val: ValueId, span: Span) -> ValueId {
        if let Some(&volatile) = self.projections.get(&val) {
            let dst = self.alloc_val();
            let ty = self.body.val_types.get(&val).cloned().unwrap_or(Ty::Unit);
            self.set_val_type(dst, ty);
            self.emit_inst(span, InstKind::ContextLoad { dst, src: val, volatile });
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
        let inner = self.scopes.pop().unwrap();
        if let Some(outer) = self.scopes.last_mut() {
            // Hoisting: variables that exist in outer scope keep their (possibly updated) type.
            // Variables defined only in inner scope do NOT propagate out.
            for (name, ty) in &inner {
                if outer.contains_key(name) {
                    // Already in outer — update type (inner may have re-bound with different value).
                    outer.insert(*name, ty.clone());
                }
                // else: inner-only definition — does not escape.
            }
        }
    }

    fn define_var(&mut self, name: Astr, ty: Ty) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn is_defined(&self, name: Astr) -> bool {
        self.scopes
            .iter()
            .rev()
            .any(|scope| scope.contains_key(&name))
    }

    fn var_type(&self, name: Astr) -> Ty {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(&name) {
                return ty.clone();
            }
        }
        Ty::error()
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

    fn type_of_id(&self, id: AstId) -> Ty {
        self.type_map.get(&id).cloned().unwrap_or(Ty::error())
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
            Node::IterBlock(ib) => self.lower_iter_block(ib),
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

    fn emit_concat(&mut self, span: Span, left: ValueId, right: ValueId) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::String);
        self.emit_inst(
            span,
            InstKind::BinOp {
                dst,
                op: BinOp::Add,
                left,
                right,
            },
        );
        dst
    }

    /// If the coercion map indicates this span needs a cast, emit a Cast
    /// instruction and return the new ValueId. Otherwise return `val` as-is.
    fn maybe_cast(&mut self, id: AstId, span: Span, val: ValueId) -> ValueId {
        let val = self.materialize(val, span);
        if let Some(&kind) = self.coercion_lookup.get(&id) {
            match kind {
                CastKind::Extern(fn_ref) => {
                    // ExternCast → lower as FunctionCall (pure, 1 arg, no context).
                    // Determine result type from the cast function's return type.
                    let ret_ty = self
                        .function_ids
                        .get(&fn_ref.name)
                        .and_then(|(_, ty)| match ty {
                            Ty::Fn { ret, .. } => Some(ret.as_ref().clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| Ty::error());
                    let cast_dst = self.alloc_val();
                    self.set_val_type(cast_dst, ret_ty);
                    self.emit_inst(
                        span,
                        InstKind::FunctionCall {
                            dst: cast_dst,
                            callee: Callee::Direct(fn_ref),
                            args: vec![val],
                            context_uses: vec![],
                            context_defs: vec![],
                        },
                    );
                    cast_dst
                }
                _ => {
                    // Native cast — inline conversion.
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
                }
            }
        } else {
            val
        }
    }

    // --- Expression lowering ---

    /// Lower an expression to a **value** (not a projection).
    /// If the result is a projection, it is materialized via ContextLoad.
    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        let val = self.maybe_cast(expr.id(), expr.span(), val);
        self.ensure_loaded(expr.span(), val)
    }

    /// Lower an expression, preserving projections (no automatic ContextLoad).
    /// Used by FieldAccess and ContextStore where projection must be maintained.
    fn lower_expr_projectable(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        self.maybe_cast(expr.id(), expr.span(), val)
    }

    fn lower_expr_inner(&mut self, expr: &Expr) -> ValueId {
        match expr {
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

            Expr::ContextRef { id, name: qref, span } => {
                let volatile = self.is_volatile_ctx(qref);
                let dst = self.alloc_typed(*id);
                self.set_origin(dst, ValOrigin::Context(qref.name));
                self.emit_inst(*span, InstKind::ContextProject { dst, ctx: *qref, volatile });
                self.mark_projection(dst, volatile);
                dst
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => match ref_kind {
                RefKind::ExternParam => {
                    let dst = self.alloc_typed(*id);
                    self.set_origin(dst, ValOrigin::ExternParam(*name));
                    self.emit_inst(*span, InstKind::ParamLoad { dst, name: *name });
                    dst
                }
                RefKind::Value => {
                    if !self.is_defined(*name) {
                        // Undefined variable — emit Unit (dead code).
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Unit);
                        return dst;
                    }
                    let dst = self.alloc_val();
                    let ty = self.var_type(*name);
                    self.set_val_type(dst, ty);
                    self.set_origin(dst, ValOrigin::Named(*name));
                    self.emit_inst(*span, InstKind::VarLoad { dst, name: *name });
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
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
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

            Expr::UnaryOp { id, op, operand, span } => {
                let o = self.lower_expr(operand);
                let dst = self.alloc_expr(*id);
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
                id,
                object,
                field,
                span,
            } => {
                let obj = self.lower_expr_projectable(object);
                let is_proj = self.is_projection(obj);
                let dst = self.alloc_val();
                self.set_val_type(dst, self.type_of_id(*id));
                self.set_origin(dst, ValOrigin::Field(obj, *field));
                self.emit_inst(
                    *span,
                    InstKind::FieldGet {
                        dst,
                        object: obj,
                        field: *field,
                    },
                );
                // Projection propagates through field access (including volatile flag).
                if let Some(&volatile) = self.projections.get(&obj) {
                    self.mark_projection(dst, volatile);
                }
                dst
            }

            Expr::FuncCall { id, func, args, span } => self.lower_func_call(func, args, None, *id, *span),

            Expr::Pipe { id, left, right, span } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`, `a | f` → `f(a)`
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
                        let dst = self.alloc_typed(expr.id());
                        self.emit_inst(
                            *span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Indirect(r),
                                args: vec![l],
                                context_uses: vec![],
                                context_defs: vec![],
                            },
                        );
                        dst
                    }
                }
            }

            Expr::Lambda { id, params, body, span } => {
                // Capture analysis: find free variables in body.
                let param_names: FxHashSet<Astr> = params.iter().map(|p| p.name).collect();
                let free_vars = self.free_vars_in_expr(body, &param_names);

                // Emit VarLoad for each captured variable (snapshot current value).
                let capture_regs: Vec<ValueId> = free_vars
                    .iter()
                    .map(|(name, var_id, var_span)| {
                        let dst = self.alloc_typed(*var_id);
                        self.set_origin(dst, ValOrigin::Named(*name));
                        self.emit_inst(*var_span, InstKind::VarLoad { dst, name: *name });
                        dst
                    })
                    .collect();
                // Create closure body.
                let closure_label = self.alloc_closure_label();

                // Build the closure body MIR in a sub-lowerer.
                let mut sub_body = MirBody::new();
                let mut sub_scopes: Vec<FxHashMap<Astr, Ty>> = vec![FxHashMap::default()];

                // Captures become the first registers.
                let mut closure_capture_regs = Vec::new();
                for (i, (name, _, _)) in free_vars.iter().enumerate() {
                    let reg = sub_body.val_factory.next();
                    closure_capture_regs.push(reg);
                    let cap_ty = capture_regs
                        .get(i)
                        .and_then(|r| self.body.val_types.get(r))
                        .cloned()
                        .unwrap_or(Ty::error());
                    sub_scopes[0].insert(*name, cap_ty.clone());
                    sub_body.val_types.insert(reg, cap_ty);
                }

                // Params follow captures.
                let mut closure_param_regs = Vec::new();
                for p in params.iter() {
                    let reg = sub_body.val_factory.next();
                    closure_param_regs.push(reg);
                    let ty = self.type_of_id(p.id);
                    sub_scopes[0].insert(p.name, ty.clone());
                    sub_body.val_types.insert(reg, ty);
                }

                // We need to lower the body in context of the sub-body.
                // Swap state.
                let saved_body = std::mem::replace(&mut self.body, sub_body);
                let saved_scopes = std::mem::replace(&mut self.scopes, sub_scopes);
                let saved_projections = std::mem::take(&mut self.projections);

                // Emit VarStore for captures so VarLoad in body can find them.
                for ((name, _, _), capture_reg) in free_vars.iter().zip(closure_capture_regs.iter()) {
                    self.emit_inst(
                        *span,
                        InstKind::VarStore {
                            name: *name,
                            src: *capture_reg,
                        },
                    );
                }
                // Emit VarStore for params.
                for (p, param_reg) in params.iter().zip(closure_param_regs.iter()) {
                    self.emit_inst(
                        p.span,
                        InstKind::VarStore {
                            name: p.name,
                            src: *param_reg,
                        },
                    );
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
                self.emit_inst(*span, InstKind::Return(result_reg));

                let mut closure_body_mir = std::mem::replace(&mut self.body, saved_body);
                self.scopes = saved_scopes;
                self.projections = saved_projections;

                closure_body_mir.capture_regs = closure_capture_regs;
                closure_body_mir.param_regs = closure_param_regs;
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
                self.emit_inst(*span, InstKind::MakeDeque { dst, elements });
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

            Expr::Range {
                start,
                end,
                kind,
                span,
                ..
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
                id, tag, payload, span, ..
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
        }
    }

    fn lower_context_store(
        &mut self,
        qref: QualifiedRef,
        value_expr: &Expr,
        span: Span,
    ) -> ValueId {
        let val = self.lower_expr(value_expr);
        let volatile = self.is_volatile_ctx(&qref);
        let ctx_id = qref;
        let proj = self.alloc_val();
        let val_ty = self.body.val_types.get(&val).cloned().unwrap_or(Ty::error());
        self.set_val_type(proj, val_ty);
        self.emit_inst(
            span,
            InstKind::ContextProject {
                dst: proj,
                ctx: ctx_id,
                volatile,
            },
        );
        self.emit_inst(
            span,
            InstKind::ContextStore {
                dst: proj,
                value: val,
                volatile,
            },
        );
        val
    }

    fn lower_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Box<Expr>>,
        call_id: AstId,
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
        let dst = self.alloc_typed(call_id);

        // @fn_name(args) — context-based function call.
        if let Expr::ContextRef { name: qref, .. } = func {
            let fn_id = self.function_ids.get(&qref.name).map(|(id, _)| *id);
            if let Some(fn_id) = fn_id {
                self.set_origin(dst, ValOrigin::Call(qref.name));
                self.emit_inst(
                    call_span,
                    InstKind::FunctionCall {
                        dst,
                        callee: Callee::Direct(fn_id),
                        args: arg_regs,
                        context_uses: vec![],
                        context_defs: vec![],
                    },
                );
                return dst;
            }
        }

        // Named function call (Ident).
        if let Expr::Ident {
            name,
            ref_kind,
            span: ident_span,
            ..
        } = func
        {
            match ref_kind {
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
                                context_uses: vec![],
                                context_defs: vec![],
                            },
                        );
                        return dst;
                    }

                    // 2. Local variable (closure/lambda — Indirect call)
                    if self.is_defined(*name) {
                        let closure_reg = self.alloc_val();
                        let closure_ty = self.var_type(*name);
                        self.set_val_type(closure_reg, closure_ty);
                        self.set_origin(closure_reg, ValOrigin::Named(*name));
                        self.emit_inst(
                            *ident_span,
                            InstKind::VarLoad {
                                dst: closure_reg,
                                name: *name,
                            },
                        );
                        self.emit_inst(
                            call_span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Indirect(closure_reg),
                                args: arg_regs,
                                context_uses: vec![],
                                context_defs: vec![],
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

        // Expression call (e.g., (|x| -> x)(42), or complex pipe)
        self.set_origin(dst, ValOrigin::Call(self.interner.intern("<closure>")));
        let func_reg = self.lower_expr(func);
        self.emit_inst(
            call_span,
            InstKind::FunctionCall {
                dst,
                callee: Callee::Indirect(func_reg),
                args: arg_regs,
                context_uses: vec![],
                context_defs: vec![],
            },
        );
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
            let volatile = self.is_volatile_ctx(qref);
            let proj = self.alloc_val();
            let src_ty = self.body.val_types.get(&src).cloned().unwrap_or(Ty::error());
            self.set_val_type(proj, src_ty);
            self.emit_inst(
                *pat_span,
                InstKind::ContextProject {
                    dst: proj,
                    ctx: *qref,
                    volatile,
                },
            );
            self.emit_inst(
                *pat_span,
                InstKind::ContextStore {
                    dst: proj,
                    value: src,
                    volatile,
                },
            );
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
                    self.emit_inst(*pat_span, InstKind::VarStore { name: *name, src });
                    self.define_var(*name, ty);
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

        // Determine element type and convert source to List if needed.
        let elem_ty = self.iterable_elem_type(source_reg);
        let list_reg = match self.body.val_types.get(&source_reg) {
            Some(Ty::List(_)) => source_reg,
            Some(Ty::Deque(..)) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(elem_ty.clone())));
                self.emit_inst(ib.span, InstKind::Cast { dst: cast_dst, src: source_reg, kind: CastKind::DequeToList });
                cast_dst
            }
            Some(Ty::Range) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(Ty::Int)));
                self.emit_inst(ib.span, InstKind::Cast { dst: cast_dst, src: source_reg, kind: CastKind::RangeToList });
                cast_dst
            }
            _ => source_reg,
        };

        // Initial index = 0.
        let zero = self.alloc_val();
        self.set_val_type(zero, Ty::Int);
        self.emit_inst(ib.span, InstKind::Const { dst: zero, value: Literal::Int(0) });

        let loop_label = self.alloc_label();
        let catch_all_label = self.alloc_label();
        let end_label = self.alloc_label();

        // Initial accumulator: empty string.
        let init_acc = self.emit_empty_string(ib.span);

        // Jump to loop with initial index + accumulator.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: loop_label,
                args: vec![zero, init_acc],
            },
        );

        // Loop header — receives index + accumulator as block params.
        let index_param = self.alloc_val();
        self.set_val_type(index_param, Ty::Int);
        let acc_param = self.alloc_val();
        self.set_val_type(acc_param, Ty::String);
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: loop_label,
                params: vec![index_param, acc_param],
                merge_of: None,
            },
        );

        // ListStep — if index >= len, jump to catch_all; else dst = list[index].
        let value_reg = self.alloc_val();
        self.set_val_type(value_reg, elem_ty);
        let next_index = self.alloc_val();
        self.set_val_type(next_index, Ty::Int);
        self.emit_inst(
            ib.span,
            InstKind::ListStep {
                dst: value_reg,
                list: list_reg,
                index_src: index_param,
                index_dst: next_index,
                done: catch_all_label,
                done_args: vec![acc_param],
            },
        );

        // Body label after ListStep (ListStep is a terminator in CFG).
        let body_label = self.alloc_label();
        self.emit_label(ib.span, body_label);

        // Bind pattern (irrefutable — no test needed).
        self.push_scope();
        self.lower_pattern_bind(&ib.pattern, value_reg, ib.span);

        // Lower body — concat all body nodes into a single string.
        let body = adjusted_body.as_deref().unwrap_or(&ib.body);
        let body_result = self.lower_nodes(body, ib.span);

        // Accumulate: new_acc = acc_param + body_result.
        let new_acc = self.emit_concat(ib.span, acc_param, body_result);

        self.pop_scope();

        // Jump back to loop with next index + new accumulator.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: loop_label,
                args: vec![next_index, new_acc],
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
            Pattern::ContextBind { name: qref, .. } => {
                let volatile = self.is_volatile_ctx(qref);
                let proj = self.alloc_val();
                let src_ty = self.body.val_types.get(&src_reg).cloned().unwrap_or(Ty::error());
                self.set_val_type(proj, src_ty);
                self.emit_inst(
                    span,
                    InstKind::ContextProject {
                        dst: proj,
                        ctx: *qref,
                        volatile,
                    },
                );
                self.emit_inst(
                    span,
                    InstKind::ContextStore {
                        dst: proj,
                        value: src_reg,
                        volatile,
                    },
                );
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
                self.emit_inst(
                    span,
                    InstKind::VarStore {
                        name: *name,
                        src: src_reg,
                    },
                );
                self.define_var(*name, ty);
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

    fn free_vars_in_expr(&self, expr: &Expr, bound: &FxHashSet<Astr>) -> Vec<(Astr, AstId, Span)> {
        let mut free = Vec::new();
        let mut seen = FxHashSet::default();
        self.collect_free_vars(expr, bound, &mut free, &mut seen);
        free
    }

    fn collect_free_vars_stmts(
        &self,
        stmts: &[Stmt],
        bound: &mut FxHashSet<Astr>,
        free: &mut Vec<(Astr, AstId, Span)>,
        seen: &mut FxHashSet<Astr>,
    ) {
        for stmt in stmts {
            match stmt {
                Stmt::Bind { name, expr, .. } => {
                    self.collect_free_vars(expr, bound, free, seen);
                    bound.insert(*name);
                }
                Stmt::ContextStore { expr, .. } | Stmt::Expr(expr) => {
                    self.collect_free_vars(expr, bound, free, seen);
                }
                Stmt::MatchBind { source, body, .. } | Stmt::Iterate { source, body, .. } => {
                    self.collect_free_vars(source, bound, free, seen);
                    let mut inner = bound.clone();
                    self.collect_free_vars_stmts(body, &mut inner, free, seen);
                }
            }
        }
    }

    fn collect_free_vars(
        &self,
        expr: &Expr,
        bound: &FxHashSet<Astr>,
        free: &mut Vec<(Astr, AstId, Span)>,
        seen: &mut FxHashSet<Astr>,
    ) {
        match expr {
            Expr::Ident {
                id,
                name,
                ref_kind: RefKind::Value,
                span,
            } => {
                if bound.contains(name) || seen.contains(name) {
                    return;
                }
                if self.is_defined(*name) {
                    seen.insert(*name);
                    free.push((*name, *id, *span));
                }
            }
            // ExternParam/context refs resolve at runtime — no capture needed.
            Expr::Ident {
                ref_kind: RefKind::ExternParam,
                ..
            }
            | Expr::ContextRef { .. } => {}
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
                self.collect_free_vars_stmts(stmts, &mut inner_bound, free, seen);
                self.collect_free_vars(tail, &inner_bound, free, seen);
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
        let (module, _) =
            crate::test::compile_template(interner, source, &ctx).expect("compile failed");
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
            .any(|i| matches!(&i.kind, InstKind::Return(_)));
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
