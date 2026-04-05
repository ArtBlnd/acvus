use acvus_ast::{
    AstId, BinOp, ElseBranch, Expr, IndentModifier, IterBlock, Literal, MatchBlock, Node,
    ObjectExprField, ObjectPatternField, Pattern, RefKind, Script, Span, Stmt, Template,
    TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{ContextPolicy, QualifiedRef};
use crate::ir::{
    Callee, CastKind, Inst, InstKind, Label, MirBody, MirModule, RefTarget, ValOrigin, ValueId,
};
use crate::graph::FnMetadata;
use crate::ty::Ty;
use crate::typeck::{CoercionMap, TypeMap};

pub struct Lowerer<'a> {
    body: MirBody,
    /// Interner for string interning.
    interner: &'a Interner,
    /// Stack of scopes: variable name → type (for validity, capture, and Ref+Load typing).
    scopes: Vec<FxHashMap<Astr, Ty>>,
    /// Variable name → storage slot ValueId.
    /// Each variable gets a unique slot (like LLVM's alloca).
    var_slots: FxHashMap<Astr, ValueId>,
    /// Type map from type checker.
    type_map: TypeMap,
    /// Coercion map from type checker (expr AstId → CastKind).
    coercion_lookup: FxHashMap<AstId, CastKind>,
    /// Closures produced during lowering.
    closures: FxHashMap<Label, MirBody>,
    /// Global closure label counter — shared across nesting levels to prevent
    /// label collisions when nested closures each allocate from a sub-body.
    closure_label_count: u32,
    /// Context QualifiedRef → Ty.
    context_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
    /// Function QualifiedRef → FnMetadata. From InferResult.fn_metadata.
    fn_metadata: Freeze<FxHashMap<QualifiedRef, FnMetadata>>,
    /// External constraints on contexts (volatile, read_only, etc.).
    policies: FxHashMap<QualifiedRef, ContextPolicy>,
    /// Context projection alias stack: @x → (@a, [x]) means @x is an alias for @a.x.
    /// Pushed/popped around match-bind bodies for destructure projection.
    context_aliases: Vec<FxHashMap<QualifiedRef, (QualifiedRef, Vec<Astr>)>>,
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
        context_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
        fn_metadata: Freeze<FxHashMap<QualifiedRef, FnMetadata>>,
        policies: FxHashMap<QualifiedRef, ContextPolicy>,
        extern_params: Vec<(Astr, Ty)>,
    ) -> Self {
        let coercion_lookup: FxHashMap<AstId, CastKind> = coercion_map.into_iter().collect();
        let initial_scope = FxHashMap::default();
        let mut body = MirBody::new();

        // Allocate param_regs for extern params (LLVM-style: params are SSA values).
        // param_regs[i] holds the initial value of extern_params[i].
        // SSA will use these as entry definitions instead of Ref+Load.
        for (name, ty) in &extern_params {
            let reg = body.val_factory.next();
            body.val_types.insert(reg, ty.clone());
            body.params.push((*name, reg));
        }

        Self {
            body,
            interner,
            scopes: vec![initial_scope],
            var_slots: FxHashMap::default(),
            type_map,
            coercion_lookup,
            closures: FxHashMap::default(),
            closure_label_count: 0,
            context_types,
            fn_metadata,
            policies,
            context_aliases: vec![],
        }
    }

    /// Emit Ref + Load for all known contexts at entry.
    /// This is the alloca equivalent — SSA pass will promote these to PHI form.
    /// Order is deterministic (sorted by context name).
    fn emit_entry_context_loads(&mut self, span: Span) {
        let mut entries: Vec<_> = self
            .context_types
            .iter()
            .map(|(&qref, ty)| (qref, ty.clone()))
            .collect();
        entries.sort_by_key(|(qref, _)| self.interner.resolve(qref.name).to_string());

        for (qref, ty) in entries {
            let val = self.emit_ref_load(span, RefTarget::Context(qref), vec![], ty);
            self.set_origin(val, ValOrigin::Context(qref.name));
        }
    }

    pub fn lower_template(mut self, template: &Template) -> MirModule {
        self.emit_entry_context_loads(template.span);
        let result = self.lower_nodes(&template.body, template.span);
        self.emit_inst(template.span, InstKind::Return(result));
        self.build_module()
    }

    pub fn lower_script(mut self, script: &Script) -> MirModule {
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
            Stmt::Bind {
                name, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let ty = self
                    .body
                    .val_types
                    .get(&val)
                    .cloned()
                    .unwrap_or(Ty::error());
                let slot = self.var_slot(*name);
                self.set_origin(slot, ValOrigin::Named(*name));
                self.emit_ref_store(*span, RefTarget::Var(slot), vec![], val);
                self.define_var(*name, ty);
            }
            Stmt::ContextStore {
                name, path, expr, span, ..
            } => {
                // Resolve alias: if @x → @a.x, then @x.y = v becomes @a.x.y = v
                if let Some((real_ctx, alias_path)) = self.resolve_context_alias(name) {
                    let mut full_path = alias_path;
                    full_path.extend_from_slice(path);
                    self.lower_context_store(real_ctx, &full_path, expr, *span);
                } else {
                    self.lower_context_store(*name, path, expr, *span);
                }
            }
            Stmt::VarFieldStore {
                name, path, expr, span, ..
            } => {
                self.lower_var_field_store(*name, path, expr, *span);
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

            // ── Script mode statements ──────────────────────────────

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
                let slot = self.var_slot(*name);
                self.set_origin(slot, ValOrigin::Named(*name));
                self.emit_ref_store(*span, RefTarget::Var(slot), vec![], val);
                self.define_var(*name, ty);
            }
            Stmt::LetUninit { id, name, span, .. } => {
                let slot = self.var_slot(*name);
                self.set_origin(slot, ValOrigin::Named(*name));
                // Type from typeck (fresh variable, unified later).
                let ty = self.type_of_id(*id);
                self.set_val_type(slot, ty.clone());
                self.define_var(*name, ty);
                // No store — init_check tracks this as uninit.
            }
            Stmt::Assign {
                name, expr, span, ..
            } => {
                let val = self.lower_expr(expr);
                let slot = self.lookup_var_slot(*name)
                    .expect("Assign to undefined variable — should have been caught by typeck");
                self.emit_ref_store(*span, RefTarget::Var(slot), vec![], val);
            }
            Stmt::For {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                self.lower_stmt_iterate(pattern, source, body, *span);
            }
            Stmt::While {
                cond, body, span, ..
            } => {
                self.lower_while(cond, body, *span);
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
    /// Single arm, no catch-all, no value produced.
    /// If the pattern is irrefutable (binding), skip the test.
    fn lower_stmt_match_bind(
        &mut self,
        pattern: &Pattern,
        source: &Expr,
        body: &[Stmt],
        span: Span,
    ) {
        // Projection destructure: { @x, @y, } = @a { body }
        // When source is ContextRef and pattern is Object with ContextBind fields,
        // register aliases instead of copying values.
        if let (Expr::ContextRef { name: source_ctx, .. }, Pattern::Object { fields, .. }) =
            (source, pattern)
        {
            self.push_scope();
            self.push_context_alias_scope();
            for field in fields {
                match &field.pattern {
                    Pattern::ContextBind { name: alias_ctx, .. } => {
                        self.register_context_alias(*alias_ctx, *source_ctx, vec![field.key]);
                    }
                    _ => {
                        // Non-projection sub-pattern: copy via field load.
                        let source_reg = self.lower_expr(source);
                        let source_val = self.materialize(source_reg, span);
                        let field_val = self.alloc_val();
                        self.set_val_type(
                            field_val,
                            self.object_field_type(source_val, field.key),
                        );
                        self.emit_inst(
                            span,
                            InstKind::ObjectGet {
                                dst: field_val,
                                object: source_val,
                                key: field.key,
                            },
                        );
                        self.lower_pattern_bind(&field.pattern, field_val, span);
                    }
                }
            }
            for s in body {
                self.lower_stmt(s);
            }
            self.pop_context_alias_scope();
            self.pop_scope();
            return;
        }

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
                self.emit_inst(
                    span,
                    InstKind::Cast {
                        dst: cast_dst,
                        src: source_reg,
                        kind: CastKind::DequeToList,
                    },
                );
                cast_dst
            }
            Some(Ty::Range) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(Ty::Int)));
                self.emit_inst(
                    span,
                    InstKind::Cast {
                        dst: cast_dst,
                        src: source_reg,
                        kind: CastKind::RangeToList,
                    },
                );
                cast_dst
            }
            _ => source_reg,
        };

        // Initial index = 0.
        let zero = self.alloc_val();
        self.set_val_type(zero, Ty::Int);
        self.emit_inst(
            span,
            InstKind::Const {
                dst: zero,
                value: Literal::Int(0),
            },
        );

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

    /// Lower `while cond { body }`.
    ///
    /// loop_label:
    ///   cond = lower(cond)
    ///   JumpIf cond → body_label, end_label
    /// body_label:
    ///   body...
    ///   Jump loop_label
    /// end_label:
    fn lower_while(&mut self, cond: &Expr, body: &[Stmt], span: Span) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(span, InstKind::Jump { label: loop_label, args: vec![] });
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
        self.emit_inst(span, InstKind::Jump { label: loop_label, args: vec![] });

        self.emit_label(span, end_label);
    }

    /// Lower `while let pattern = source { body }`.
    ///
    /// loop_label:
    ///   src = lower(source)
    ///   matched = test(pattern, src)
    ///   JumpIf matched → body_label, end_label
    /// body_label:
    ///   bind(pattern, src)
    ///   body...
    ///   Jump loop_label
    /// end_label:
    fn lower_while_let(
        &mut self,
        pattern: &Pattern,
        source: &Expr,
        body: &[Stmt],
        span: Span,
    ) {
        let loop_label = self.alloc_label();
        let body_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_inst(span, InstKind::Jump { label: loop_label, args: vec![] });
        self.emit_label(span, loop_label);

        let src = self.lower_expr(source);
        let matched = self.lower_pattern_test(pattern, src, span);
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
        self.lower_pattern_bind(pattern, src, span);
        for s in body {
            self.lower_stmt(s);
        }
        self.pop_scope();
        self.emit_inst(span, InstKind::Jump { label: loop_label, args: vec![] });

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
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![then_val] });

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![else_val] });

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
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![] });

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
        let src = self.lower_expr(source);
        let matched = self.lower_pattern_test(pattern, src, span);

        let then_label = self.alloc_label();
        let merge_label = self.alloc_label();

        match else_branch {
            Some(eb) => {
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
                self.lower_pattern_bind(pattern, src, span);
                for s in then_body {
                    self.lower_stmt(s);
                }
                let then_val = match then_tail {
                    Some(tail) => self.lower_expr(tail),
                    None => self.emit_unit(span),
                };
                self.pop_scope();
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![then_val] });

                // Else branch.
                self.emit_label(span, else_label);
                let else_val = self.lower_else_branch(eb, span, merge_label);
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![else_val] });

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
                self.emit_inst(
                    span,
                    InstKind::JumpIf {
                        cond: matched,
                        then_label,
                        then_args: vec![],
                        else_label: merge_label,
                        else_args: vec![],
                    },
                );

                self.emit_label(span, then_label);
                self.push_scope();
                self.lower_pattern_bind(pattern, src, span);
                for s in then_body {
                    self.lower_stmt(s);
                }
                if let Some(tail) = then_tail {
                    self.lower_expr(tail);
                }
                self.pop_scope();
                self.emit_inst(span, InstKind::Jump { label: merge_label, args: vec![] });

                self.emit_label(span, merge_label);
                self.emit_unit(span)
            }
        }
    }

    /// Lower an else branch, returning the value it produces.
    fn lower_else_branch(
        &mut self,
        eb: &ElseBranch,
        span: Span,
        _merge_label: Label,
    ) -> ValueId {
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

    /// If `val` is a Ref<T>, emit Load to materialize it into T.
    /// If it's already a scalar value, return as-is.
    fn ensure_loaded(&mut self, span: Span, val: ValueId) -> ValueId {
        if let Some(Ty::Ref(inner, volatile)) = self.body.val_types.get(&val).cloned() {
            let dst = self.alloc_val();
            self.set_val_type(dst, *inner);
            self.emit_inst(
                span,
                InstKind::Load {
                    dst,
                    src: val,
                    volatile,
                },
            );
            dst
        } else {
            val
        }
    }

    fn context_policy(&self, ctx: &QualifiedRef) -> ContextPolicy {
        // Resolve alias first: if @x → @a.x, use @a's policy.
        if let Some((real_ctx, _)) = self.resolve_context_alias(ctx) {
            return self.policies.get(&real_ctx).copied().unwrap_or_default();
        }
        self.policies.get(ctx).copied().unwrap_or_default()
    }

    /// Resolve a context alias. Returns (real_context, path) if aliased.
    fn resolve_context_alias(&self, ctx: &QualifiedRef) -> Option<(QualifiedRef, Vec<Astr>)> {
        for scope in self.context_aliases.iter().rev() {
            if let Some((real_ctx, path)) = scope.get(ctx) {
                return Some((*real_ctx, path.clone()));
            }
        }
        None
    }

    fn push_context_alias_scope(&mut self) {
        self.context_aliases.push(FxHashMap::default());
    }

    fn pop_context_alias_scope(&mut self) {
        self.context_aliases.pop();
    }

    fn register_context_alias(&mut self, alias: QualifiedRef, target: QualifiedRef, path: Vec<Astr>) {
        if let Some(scope) = self.context_aliases.last_mut() {
            scope.insert(alias, (target, path));
        }
    }

    /// Determine volatile flag from a RefTarget.
    fn ref_volatile(&self, target: &RefTarget) -> bool {
        match target {
            RefTarget::Context(qref) => self.context_policy(qref).volatile,
            RefTarget::Var(_) | RefTarget::Param(_) => false,
        }
    }

    /// Emit Ref + Store: write `value` to the given storage target.
    fn emit_ref_store(
        &mut self,
        span: Span,
        target: RefTarget,
        path: Vec<Astr>,
        value: ValueId,
    ) {
        let val_ty = self
            .body
            .val_types
            .get(&value)
            .cloned()
            .unwrap_or(Ty::error());
        let volatile = self.ref_volatile(&target);
        let ref_dst = self.alloc_val();
        self.set_val_type(ref_dst, Ty::Ref(Box::new(val_ty), volatile));
        // Set origin on Ref dst for printer context name resolution.
        match &target {
            RefTarget::Context(qref) => self.set_origin(ref_dst, ValOrigin::Context(qref.name)),
            RefTarget::Var(_) | RefTarget::Param(_) => {
                // Origin is already set on the slot by the caller before calling emit_ref_store.
                // The Ref dst inherits from the target, but the slot's debug name is
                // set separately via set_origin on the slot ValueId.
            }
        }
        self.emit_inst(
            span,
            InstKind::Ref {
                dst: ref_dst,
                target,
                path,
            },
        );
        self.emit_inst(
            span,
            InstKind::Store {
                dst: ref_dst,
                value,
                volatile,
            },
        );
    }

    /// Emit Ref + Load: read a value from the given storage target.
    fn emit_ref_load(
        &mut self,
        span: Span,
        target: RefTarget,
        path: Vec<Astr>,
        result_ty: Ty,
    ) -> ValueId {
        let volatile = self.ref_volatile(&target);
        let ref_dst = self.alloc_val();
        self.set_val_type(ref_dst, Ty::Ref(Box::new(result_ty.clone()), volatile));
        // Set origin on Ref dst for printer context name resolution.
        match &target {
            RefTarget::Context(qref) => self.set_origin(ref_dst, ValOrigin::Context(qref.name)),
            RefTarget::Var(_) | RefTarget::Param(_) => {
                // Origin set separately on the slot ValueId.
            }
        }
        self.emit_inst(
            span,
            InstKind::Ref {
                dst: ref_dst,
                target,
                path,
            },
        );
        let dst = self.alloc_val();
        self.set_val_type(dst, result_ty);
        self.emit_inst(
            span,
            InstKind::Load {
                dst,
                src: ref_dst,
                volatile,
            },
        );
        dst
    }

    /// If val is a Ref<T>, emit Load to materialize it.
    /// Otherwise return val unchanged.
    fn materialize(&mut self, val: ValueId, span: Span) -> ValueId {
        if let Some(Ty::Ref(inner, volatile)) = self.body.val_types.get(&val).cloned() {
            let dst = self.alloc_val();
            self.set_val_type(dst, *inner);
            self.emit_inst(
                span,
                InstKind::Load {
                    dst,
                    src: val,
                    volatile,
                },
            );
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

    /// Get or create a storage slot ValueId for the given variable name.
    fn lookup_var_slot(&self, name: Astr) -> Option<ValueId> {
        self.var_slots.get(&name).copied()
    }

    fn var_slot(&mut self, name: Astr) -> ValueId {
        if let Some(&slot) = self.var_slots.get(&name) {
            slot
        } else {
            let slot = self.body.val_factory.next();
            self.var_slots.insert(name, slot);
            slot
        }
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

    fn emit_unit(&mut self, span: Span) -> ValueId {
        let dst = self.alloc_val();
        self.set_val_type(dst, Ty::Unit);
        self.emit_inst(span, InstKind::Const { dst, value: Literal::Unit });
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
                        .fn_metadata
                        .get(&fn_ref)
                        .and_then(|m| match &m.ty {
                            Ty::Fn { ret, .. } => Some(ret.as_ref().clone()),
                            _ => None,
                        })
                        .unwrap_or_else(Ty::error);
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
    /// If the result is a Ref<T>, it is materialized via Load.
    fn lower_expr(&mut self, expr: &Expr) -> ValueId {
        let val = self.lower_expr_inner(expr);
        let val = self.maybe_cast(expr.id(), expr.span(), val);
        self.ensure_loaded(expr.span(), val)
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

            Expr::ContextRef {
                id,
                name: qref,
                span,
            } => {
                // Resolve alias: @x → @a.x becomes Ref { target: Context(a), path: [x] }
                let (real_ctx, path) = if let Some((real, path)) = self.resolve_context_alias(qref) {
                    (real, path)
                } else {
                    (*qref, vec![])
                };
                let policy = self.context_policy(&real_ctx);
                let inner_ty = self.type_of_id(*id);
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Ref(Box::new(inner_ty), policy.volatile));
                if path.is_empty() {
                    self.set_origin(dst, ValOrigin::Context(real_ctx.name));
                } else {
                    self.set_origin(dst, ValOrigin::RefField(RefTarget::Context(real_ctx), path.clone()));
                }
                self.emit_inst(
                    *span,
                    InstKind::Ref {
                        dst,
                        target: RefTarget::Context(real_ctx),
                        path,
                    },
                );
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
                        let dst = self.emit_ref_load(*span, RefTarget::Param(param_reg), vec![], ty);
                        self.set_origin(dst, ValOrigin::ExternParam(name.name));
                        dst
                    } else if self.is_defined(name.name) {
                        let slot = self.var_slot(name.name);
                        let dst = self.emit_ref_load(*span, RefTarget::Var(slot), vec![], ty);
                        self.set_origin(dst, ValOrigin::Named(name.name));
                        dst
                    } else {
                        // Not found — emit poison (should be caught by typeck).
                        let dst = self.alloc_val();
                        self.set_val_type(dst, ty);
                        self.emit_inst(*span, InstKind::Poison { dst });
                        dst
                    }
                }
                RefKind::Value => {
                    if !self.is_defined(name.name) {
                        // Undefined variable — emit Unit (dead code).
                        // TODO: this should be an error, not silent Unit.
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Unit);
                        return dst;
                    }
                    let ty = self.var_type(name.name);
                    let slot = self.var_slot(name.name);
                    let dst = self.emit_ref_load(*span, RefTarget::Var(slot), vec![], ty);
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

            Expr::UnaryOp {
                id,
                op,
                operand,
                span,
            } => {
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
                /// Walk a FieldAccess chain, collecting field names.
                /// Returns (root_expr, accumulated_path).
                fn collect_field_chain(expr: &Expr) -> (&Expr, Vec<Astr>) {
                    match expr {
                        Expr::FieldAccess { object, field, .. } => {
                            let (root, mut path) = collect_field_chain(object);
                            path.push(*field);
                            (root, path)
                        }
                        other => (other, vec![]),
                    }
                }

                let field_ty = self.type_of_id(*id);
                let (root, mut path) = collect_field_chain(object);
                path.push(*field);

                match root {
                    Expr::ContextRef { name: qref, .. } => {
                        let policy = self.context_policy(qref);
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Ref(Box::new(field_ty), policy.volatile));
                        self.set_origin(
                            dst,
                            ValOrigin::RefField(RefTarget::Context(*qref), path.clone()),
                        );
                        self.emit_inst(
                            *span,
                            InstKind::Ref {
                                dst,
                                target: RefTarget::Context(*qref),
                                path,
                            },
                        );
                        dst
                    }
                    Expr::Ident {
                        name,
                        ref_kind: RefKind::Value,
                        ..
                    } if self.is_defined(name.name) => {
                        let slot = self.var_slot(name.name);
                        let dst = self.alloc_val();
                        self.set_val_type(dst, Ty::Ref(Box::new(field_ty), false));
                        self.set_origin(
                            dst,
                            ValOrigin::RefField(RefTarget::Var(slot), path.clone()),
                        );
                        self.emit_inst(
                            *span,
                            InstKind::Ref {
                                dst,
                                target: RefTarget::Var(slot),
                                path,
                            },
                        );
                        dst
                    }
                    Expr::Ident {
                        name,
                        ref_kind: RefKind::ExternParam,
                        ..
                    } => {
                        if let Some(param_reg) = self.try_param_slot(name.name) {
                            let dst = self.alloc_val();
                            self.set_val_type(dst, Ty::Ref(Box::new(field_ty), false));
                            self.set_origin(
                                dst,
                                ValOrigin::RefField(RefTarget::Param(param_reg), path.clone()),
                            );
                            self.emit_inst(
                                *span,
                                InstKind::Ref {
                                    dst,
                                    target: RefTarget::Param(param_reg),
                                    path,
                                },
                            );
                            dst
                        } else if self.is_defined(name.name) {
                            // Captured param — treat as local variable.
                            let slot = self.var_slot(name.name);
                            let dst = self.alloc_val();
                            self.set_val_type(dst, Ty::Ref(Box::new(field_ty), false));
                            self.set_origin(
                                dst,
                                ValOrigin::RefField(RefTarget::Var(slot), path.clone()),
                            );
                            self.emit_inst(
                                *span,
                                InstKind::Ref {
                                    dst,
                                    target: RefTarget::Var(slot),
                                    path,
                                },
                            );
                            dst
                        } else {
                            // Fallback: lower as scalar FieldGet.
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
                    // Otherwise: lower object to scalar, then FieldGet.
                    _ => {
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
            }

            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => self.lower_func_call(func, args, None, *id, *span),

            Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
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

            Expr::Lambda {
                id,
                params,
                body,
                span,
            } => {
                // Capture analysis: find free variables in body.
                let param_names: FxHashSet<Astr> = params.iter().map(|p| p.name).collect();
                let free_vars = self.free_vars_in_expr(body, &param_names);

                // Emit Ref+Load for each captured variable (snapshot current value).
                // If the free var is an extern param in the current scope, use Param ref;
                // otherwise use Var ref.
                let capture_regs: Vec<ValueId> = free_vars
                    .iter()
                    .map(|(name, var_id, var_span)| {
                        let ty = self.type_of_id(*var_id);
                        if let Some(param_reg) = self.try_param_slot(*name) {
                            let dst = self.emit_ref_load(*var_span, RefTarget::Param(param_reg), vec![], ty);
                            self.set_origin(dst, ValOrigin::ExternParam(*name));
                            dst
                        } else {
                            let slot = self.var_slot(*name);
                            let dst = self.emit_ref_load(*var_span, RefTarget::Var(slot), vec![], ty);
                            self.set_origin(dst, ValOrigin::Named(*name));
                            dst
                        }
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
                let saved_var_slots = std::mem::replace(&mut self.var_slots, FxHashMap::default());

                // Emit Ref+Store for captures so Ref+Load in body can find them.
                for ((name, _, _), capture_reg) in free_vars.iter().zip(closure_capture_regs.iter())
                {
                    let slot = self.var_slot(*name);
                    self.set_origin(slot, ValOrigin::Named(*name));
                    self.emit_ref_store(*span, RefTarget::Var(slot), vec![], *capture_reg);
                }
                // Emit Ref+Store for params.
                for (p, param_reg) in params.iter().zip(closure_param_regs.iter()) {
                    let slot = self.var_slot(p.name);
                    self.set_origin(slot, ValOrigin::Named(p.name));
                    self.emit_ref_store(p.span, RefTarget::Var(slot), vec![], *param_reg);
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
                self.var_slots = saved_var_slots;

                closure_body_mir.captures = free_vars.iter().map(|(name, _, _)| *name).zip(closure_capture_regs).collect();
                closure_body_mir.params = params.iter().map(|p| p.name).zip(closure_param_regs).collect();
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
            } => self.lower_if_let_expr(*id, pattern, source, then_body, then_tail, else_branch, *span),
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
        self.emit_ref_store(span, RefTarget::Context(qref), path.to_vec(), val);
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
        self.emit_ref_store(span, RefTarget::Var(slot), path.to_vec(), val);
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
            let fn_meta = self.fn_metadata.get(qref);
            if fn_meta.is_some() {
                let fn_id = *qref;
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
                    self.set_origin(dst, ValOrigin::Call(name.name));

                    // 1. Graph function (Direct call)
                    if self.fn_metadata.contains_key(name) {
                        self.emit_inst(
                            call_span,
                            InstKind::FunctionCall {
                                dst,
                                callee: Callee::Direct(*name),
                                args: arg_regs,
                                context_uses: vec![],
                                context_defs: vec![],
                            },
                        );
                        return dst;
                    }

                    // 2. Local variable (closure/lambda — Indirect call)
                    if self.is_defined(name.name) {
                        let closure_ty = self.var_type(name.name);
                        let slot = self.var_slot(name.name);
                        let closure_reg = self.emit_ref_load(
                            *ident_span,
                            RefTarget::Var(slot),
                            vec![],
                            closure_ty,
                        );
                        self.set_origin(closure_reg, ValOrigin::Named(name.name));
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
            self.emit_ref_store(*pat_span, RefTarget::Context(*qref), vec![], src);
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
                    let slot = self.var_slot(*name);
                    self.set_origin(slot, ValOrigin::Named(*name));
                    self.emit_ref_store(*pat_span, RefTarget::Var(slot), vec![], src);
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
                self.emit_inst(
                    ib.span,
                    InstKind::Cast {
                        dst: cast_dst,
                        src: source_reg,
                        kind: CastKind::DequeToList,
                    },
                );
                cast_dst
            }
            Some(Ty::Range) => {
                let cast_dst = self.alloc_val();
                self.set_val_type(cast_dst, Ty::List(Box::new(Ty::Int)));
                self.emit_inst(
                    ib.span,
                    InstKind::Cast {
                        dst: cast_dst,
                        src: source_reg,
                        kind: CastKind::RangeToList,
                    },
                );
                cast_dst
            }
            _ => source_reg,
        };

        // Initial index = 0.
        let zero = self.alloc_val();
        self.set_val_type(zero, Ty::Int);
        self.emit_inst(
            ib.span,
            InstKind::Const {
                dst: zero,
                value: Literal::Int(0),
            },
        );

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
                self.emit_ref_store(span, RefTarget::Context(*qref), vec![], src_reg);
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
                let slot = self.var_slot(*name);
                self.set_origin(slot, ValOrigin::Named(*name));
                self.emit_ref_store(span, RefTarget::Var(slot), vec![], src_reg);
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
                Stmt::ContextStore { expr, .. } | Stmt::VarFieldStore { expr, .. } | Stmt::Expr(expr) => {
                    self.collect_free_vars(expr, bound, free, seen);
                }
                Stmt::MatchBind { source, body, .. }
                | Stmt::Iterate { source, body, .. }
                | Stmt::For { source, body, .. }
                | Stmt::WhileLet { source, body, .. } => {
                    self.collect_free_vars(source, bound, free, seen);
                    let mut inner = bound.clone();
                    self.collect_free_vars_stmts(body, &mut inner, free, seen);
                }
                Stmt::LetBind { name, expr, .. } => {
                    self.collect_free_vars(expr, bound, free, seen);
                    bound.insert(*name);
                }
                Stmt::Assign { expr, .. } => {
                    self.collect_free_vars(expr, bound, free, seen);
                }
                Stmt::LetUninit { name, .. } => {
                    bound.insert(*name);
                }
                Stmt::While { cond, body, .. } => {
                    self.collect_free_vars(cond, bound, free, seen);
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
                if bound.contains(&name.name) || seen.contains(&name.name) {
                    return;
                }
                if self.is_defined(name.name) {
                    seen.insert(name.name);
                    free.push((name.name, *id, *span));
                }
            }
            // ExternParam: must be captured since the closure body may not have
            // access to the parent function's param_regs (ValueId-based).
            Expr::Ident {
                id,
                name,
                ref_kind: RefKind::ExternParam,
                span,
            } => {
                if !bound.contains(&name.name) && !seen.contains(&name.name) {
                    seen.insert(name.name);
                    free.push((name.name, *id, *span));
                }
            }
            // Context refs resolve globally — no capture needed.
            Expr::ContextRef { .. } => {}
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
            Expr::If {
                cond,
                then_body,
                then_tail,
                else_branch,
                ..
            } => {
                self.collect_free_vars(cond, bound, free, seen);
                let mut inner = bound.clone();
                self.collect_free_vars_stmts(then_body, &mut inner, free, seen);
                if let Some(tail) = then_tail {
                    self.collect_free_vars(tail, &inner, free, seen);
                }
                if let Some(eb) = else_branch {
                    self.collect_free_vars_else_branch(eb, bound, free, seen);
                }
            }
            Expr::IfLet {
                source,
                then_body,
                then_tail,
                else_branch,
                ..
            } => {
                self.collect_free_vars(source, bound, free, seen);
                let mut inner = bound.clone();
                self.collect_free_vars_stmts(then_body, &mut inner, free, seen);
                if let Some(tail) = then_tail {
                    self.collect_free_vars(tail, &inner, free, seen);
                }
                if let Some(eb) = else_branch {
                    self.collect_free_vars_else_branch(eb, bound, free, seen);
                }
            }
        }
    }

    fn collect_free_vars_else_branch(
        &self,
        eb: &ElseBranch,
        bound: &FxHashSet<Astr>,
        free: &mut Vec<(Astr, AstId, Span)>,
        seen: &mut FxHashSet<Astr>,
    ) {
        match eb {
            ElseBranch::ElseIf(expr) => self.collect_free_vars(expr, bound, free, seen),
            ElseBranch::Else { body, tail, .. } => {
                let mut inner = bound.clone();
                self.collect_free_vars_stmts(body, &mut inner, free, seen);
                if let Some(tail) = tail {
                    self.collect_free_vars(tail, &inner, free, seen);
                }
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
        let module =
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
