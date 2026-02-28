use std::collections::{HashMap, HashSet};

use acvus_ast::{
    Expr, IndentModifier, IterBlock, Literal, MatchBlock, Node, ObjectExprField,
    ObjectPatternField, Pattern, Span, Template, TupleElem, TuplePatternElem,
};

use crate::builtins::builtins;
use crate::hints::{Hint, HintTable};
use crate::ir::{ClosureBody, Inst, InstKind, Label, MirBody, MirModule, Val, ValOrigin};
use crate::ty::Ty;
use crate::typeck::TypeMap;

pub struct Lowerer {
    body: MirBody,
    /// Stack of scopes: variable name → register.
    scopes: Vec<HashMap<String, Val>>,
    /// Type map from type checker.
    type_map: TypeMap,
    /// Storage variable names (to distinguish storage from local $-vars).
    storage_names: HashSet<String>,
    /// Closures produced during lowering.
    closures: HashMap<Label, ClosureBody>,
    /// Hint table.
    hints: HintTable,
    /// Set of builtin function names (sync calls).
    builtin_names: HashSet<String>,
    /// Interned text constants pool.
    texts: Vec<String>,
}

/// Adjust indentation of a text string according to an `IndentModifier`.
/// Only lines after the first `\n` are affected; the first line is kept as-is.
fn adjust_text_indent(text: &str, modifier: &IndentModifier) -> String {
    let mut result = String::with_capacity(text.len());
    let mut first = true;
    for line in text.split('\n') {
        if !first {
            result.push('\n');
        }
        if first {
            result.push_str(line);
            first = false;
            continue;
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
                for _ in 0..n {
                    result.push(' ');
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

impl Lowerer {
    pub fn new(type_map: TypeMap, storage_names: HashSet<String>) -> Self {
        let builtin_names: HashSet<String> =
            builtins().iter().map(|b| b.name.to_string()).collect();
        Self {
            body: MirBody::new(),
            scopes: vec![HashMap::new()],
            type_map,
            storage_names,
            closures: HashMap::new(),
            hints: HintTable::new(),
            builtin_names,
            texts: Vec::new(),
        }
    }

    pub fn lower_template(mut self, template: &Template) -> (MirModule, HintTable) {
        self.lower_nodes(&template.body);
        let module = MirModule {
            main: self.body,
            closures: self.closures,
            texts: self.texts,
        };
        (module, self.hints)
    }

    fn alloc_val(&mut self) -> Val {
        let r = Val(self.body.val_count);
        self.body.val_count += 1;
        r
    }

    fn alloc_label(&mut self) -> Label {
        let l = Label(self.body.label_count);
        self.body.label_count += 1;
        l
    }

    fn intern_text(&mut self, text: &str) -> usize {
        if let Some(idx) = self.texts.iter().position(|t| t == text) {
            idx
        } else {
            let idx = self.texts.len();
            self.texts.push(text.to_string());
            idx
        }
    }

    fn emit(&mut self, inst: Inst) {
        self.body.insts.push(inst);
    }

    fn emit_inst(&mut self, span: Span, kind: InstKind) {
        self.emit(Inst { span, kind });
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Copy variables defined in the current (top) scope to the parent scope.
    /// This hoists body-less variable bindings out of match arm scopes.
    fn hoist_bodyless_bindings(&mut self) {
        let len = self.scopes.len();
        if len >= 2 {
            let top = self.scopes[len - 1].clone();
            for (name, val) in top {
                self.scopes[len - 2].insert(name, val);
            }
        }
    }

    fn define_var(&mut self, name: &str, reg: Val) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), reg);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<Val> {
        for scope in self.scopes.iter().rev() {
            if let Some(&reg) = scope.get(name) {
                return Some(reg);
            }
        }
        None
    }

    fn set_val_type(&mut self, val: Val, ty: Ty) {
        self.body.val_types.insert(val, ty);
    }

    fn tuple_elem_type(&self, tuple_val: Val, index: usize) -> Ty {
        if let Some(Ty::Tuple(elems)) = self.body.val_types.get(&tuple_val) {
            elems.get(index).cloned().unwrap_or(Ty::Unit)
        } else {
            Ty::Unit
        }
    }

    fn list_elem_type(&self, list_val: Val) -> Ty {
        if let Some(Ty::List(elem)) = self.body.val_types.get(&list_val) {
            elem.as_ref().clone()
        } else {
            Ty::Unit
        }
    }

    fn object_field_type(&self, object_val: Val, key: &str) -> Ty {
        if let Some(Ty::Object(fields)) = self.body.val_types.get(&object_val) {
            fields.get(key).cloned().unwrap_or(Ty::Unit)
        } else {
            Ty::Unit
        }
    }

    fn iterable_elem_type(&self, src_val: Val) -> Ty {
        match self.body.val_types.get(&src_val) {
            Some(Ty::List(elem)) => elem.as_ref().clone(),
            Some(Ty::Range) => Ty::Int,
            _ => Ty::Unit,
        }
    }

    fn set_origin(&mut self, val: Val, origin: ValOrigin) {
        self.body.debug.set(val, origin);
    }

    fn type_of_span(&self, span: Span) -> Ty {
        self.type_map.get(&span).cloned().unwrap_or(Ty::Unit)
    }

    // --- Node lowering ---

    fn lower_nodes(&mut self, nodes: &[Node]) {
        for node in nodes {
            self.lower_node(node);
        }
    }

    fn lower_node(&mut self, node: &Node) {
        match node {
            Node::Text { value, span } => {
                let idx = self.intern_text(value);
                self.emit_inst(*span, InstKind::EmitText(idx));
            }
            Node::Comment { .. } => {
                // Comments produce no instructions.
            }
            Node::InlineExpr { expr, span } => {
                let reg = self.lower_expr(expr);
                self.emit_inst(*span, InstKind::EmitValue(reg));
            }
            Node::MatchBlock(mb) => {
                self.lower_match_block(mb);
            }
            Node::IterBlock(ib) => {
                self.lower_iter_block(ib);
            }
        }
    }

    // --- Expression lowering ---

    fn lower_expr(&mut self, expr: &Expr) -> Val {
        match expr {
            Expr::Literal { value, span } => {
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.set_origin(dst, ValOrigin::Expr);
                self.emit_inst(*span, InstKind::Const { dst, value: value.clone() });
                dst
            }

            Expr::Ident {
                name,
                is_storage_ref,
                span,
            } => {
                if *is_storage_ref {
                    // $name: check local scope first, then storage.
                    if let Some(reg) = self.lookup_var(name) {
                        reg
                    } else {
                        // Storage load.
                        let dst = self.alloc_val();
                        let ty = self.type_of_span(*span);
                        self.set_val_type(dst, ty);
                        self.set_origin(dst, ValOrigin::Storage(name.clone()));
                        self.emit_inst(
                            *span,
                            InstKind::StorageLoad {
                                dst,
                                name: name.clone(),
                            },
                        );
                        dst
                    }
                } else if let Some(reg) = self.lookup_var(name) {
                    reg
                } else {
                    // Unknown variable — emit a placeholder load.
                    // Type checker should have caught this already.
                    let dst = self.alloc_val();
                    self.set_val_type(dst, Ty::Unit);
                    dst
                }
            }

            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let l = self.lower_expr(left);
                let r = self.lower_expr(right);
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.set_origin(dst, ValOrigin::Expr);
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
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.set_origin(dst, ValOrigin::Expr);
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
                let obj = self.lower_expr(object);
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.set_origin(dst, ValOrigin::Field(obj, field.clone()));
                self.emit_inst(
                    *span,
                    InstKind::FieldGet {
                        dst,
                        object: obj,
                        field: field.clone(),
                    },
                );
                dst
            }

            Expr::FuncCall { func, args, span } => self.lower_func_call(func, args, *span),

            Expr::Pipe { left, right, span } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`, `a | f` → `f(a)`
                match right.as_ref() {
                    Expr::FuncCall {
                        func,
                        args,
                        span: _,
                    } => {
                        let mut new_args = vec![left.as_ref().clone()];
                        new_args.extend(args.iter().cloned());
                        self.lower_func_call(func, &new_args, *span)
                    }
                    Expr::Ident {
                        name,
                        is_storage_ref: false,
                        span: id_span,
                    } => {
                        let func_expr = Expr::Ident {
                            name: name.clone(),
                            is_storage_ref: false,
                            span: *id_span,
                        };
                        let new_args = vec![left.as_ref().clone()];
                        self.lower_func_call(&func_expr, &new_args, *span)
                    }
                    _ => {
                        // Fallback: evaluate both sides, call closure.
                        let l = self.lower_expr(left);
                        let r = self.lower_expr(right);
                        let dst = self.alloc_val();
                        self.set_val_type(dst, self.type_of_span(*span));
                        self.emit_inst(
                            *span,
                            InstKind::CallClosure {
                                dst,
                                closure: r,
                                args: vec![l],
                            },
                        );
                        dst
                    }
                }
            }

            Expr::Lambda {
                params,
                body,
                span,
            } => {
                // Capture analysis: find free variables in body.
                let param_names: HashSet<String> =
                    params.iter().map(|p| p.name.clone()).collect();
                let free_vars = self.free_vars_in_expr(body, &param_names);

                let capture_regs: Vec<Val> = free_vars
                    .iter()
                    .filter_map(|name| self.lookup_var(name))
                    .collect();
                let capture_names: Vec<String> = free_vars.clone();

                // Create closure body.
                let closure_label = self.alloc_label();

                // Build the closure body MIR in a sub-lowerer.
                let mut sub_body = MirBody::new();
                let mut sub_scopes: Vec<HashMap<String, Val>> = vec![HashMap::new()];

                // Captures become the first registers.
                for (i, name) in capture_names.iter().enumerate() {
                    let reg = Val(i as u32);
                    sub_body.val_count = sub_body.val_count.max(reg.0 + 1);
                    sub_scopes[0].insert(name.clone(), reg);
                    // Copy capture type from outer body.
                    if let Some(outer_reg) = capture_regs.get(i) {
                        if let Some(ty) = self.body.val_types.get(outer_reg) {
                            sub_body.val_types.insert(reg, ty.clone());
                        }
                    }
                }

                // Params follow captures.
                let param_start = capture_names.len() as u32;
                let param_name_list: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                for (i, p) in params.iter().enumerate() {
                    let reg = Val(param_start + i as u32);
                    sub_body.val_count = sub_body.val_count.max(reg.0 + 1);
                    sub_scopes[0].insert(p.name.clone(), reg);
                    // Set param type from typeck.
                    let ty = self.type_of_span(p.span);
                    sub_body.val_types.insert(reg, ty);
                }

                // We need to lower the body in context of the sub-body.
                // Swap state.
                let saved_body = std::mem::replace(&mut self.body, sub_body);
                let saved_scopes = std::mem::replace(&mut self.scopes, sub_scopes);

                let result_reg = self.lower_expr(body);
                self.emit_inst(*span, InstKind::Return(result_reg));

                let closure_body_mir = std::mem::replace(&mut self.body, saved_body);
                self.scopes = saved_scopes;

                self.closures.insert(
                    closure_label,
                    ClosureBody {
                        capture_names,
                        param_names: param_name_list,
                        body: closure_body_mir,
                    },
                );

                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
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
                let elements: Vec<Val> = head
                    .iter()
                    .chain(tail.iter())
                    .map(|e| self.lower_expr(e))
                    .collect();
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.emit_inst(*span, InstKind::MakeList { dst, elements });
                dst
            }

            Expr::Object { fields, span } => {
                let field_regs: Vec<(String, Val)> = fields
                    .iter()
                    .map(|ObjectExprField { key, value, .. }| {
                        let r = self.lower_expr(value);
                        (key.clone(), r)
                    })
                    .collect();
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
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
                let elem_vals: Vec<Val> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.lower_expr(e),
                        TupleElem::Wildcard(s) => {
                            // Wildcard in expression context: produce a Unit placeholder.
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
                let dst = self.alloc_val();
                let ty = self.type_of_span(*span);
                self.set_val_type(dst, ty);
                self.emit_inst(*span, InstKind::MakeTuple { dst, elements: elem_vals });
                dst
            }

            Expr::Group { elements, span } => {
                // Should not appear outside lambda params. Lower last element.
                if let Some(last) = elements.last() {
                    self.lower_expr(last)
                } else {
                    let dst = self.alloc_val();
                    self.set_val_type(dst, Ty::Unit);
                    self.emit_inst(
                        *span,
                        InstKind::Const {
                            dst,
                            value: Literal::Bool(false),
                        },
                    );
                    dst
                }
            }
        }
    }

    fn lower_func_call(&mut self, func: &Expr, args: &[Expr], call_span: Span) -> Val {
        let func_name = match func {
            Expr::Ident {
                name,
                is_storage_ref: false,
                ..
            } => Some(name.clone()),
            _ => None,
        };

        let arg_regs: Vec<Val> = args.iter().map(|a| self.lower_expr(a)).collect();
        let dst = self.alloc_val();
        let ty = self.type_of_span(call_span);
        self.set_val_type(dst, ty);
        self.set_origin(
            dst,
            ValOrigin::Call(func_name.clone().unwrap_or_else(|| "<closure>".into())),
        );

        if let Some(ref name) = func_name {
            if self.builtin_names.contains(name.as_str()) {
                // Builtin: synchronous Call.
                self.emit_inst(
                    call_span,
                    InstKind::Call {
                        dst,
                        func: name.clone(),
                        args: arg_regs,
                    },
                );
                let idx = self.body.insts.len() - 1;
                self.hints.add(idx, Hint::Pure);
                return dst;
            }

            // Check if it's a local variable (closure call).
            if let Some(closure_reg) = self.lookup_var(name) {
                self.emit_inst(
                    call_span,
                    InstKind::CallClosure {
                        dst,
                        closure: closure_reg,
                        args: arg_regs,
                    },
                );
                return dst;
            }

            // External function: async call + await.
            let future_reg = self.alloc_val();
            self.set_val_type(future_reg, Ty::Unit); // placeholder
            self.emit_inst(
                call_span,
                InstKind::AsyncCall {
                    dst: future_reg,
                    func: name.clone(),
                    args: arg_regs,
                },
            );
            let idx = self.body.insts.len() - 1;
            self.hints.add(idx, Hint::Effectful);

            self.emit_inst(
                call_span,
                InstKind::Await {
                    dst,
                    src: future_reg,
                },
            );
        } else {
            // Expression call (closure).
            let func_reg = self.lower_expr(func);
            self.emit_inst(
                call_span,
                InstKind::CallClosure {
                    dst,
                    closure: func_reg,
                    args: arg_regs,
                },
            );
        }

        dst
    }

    // --- Match block lowering ---

    fn lower_match_block(&mut self, mb: &MatchBlock) {
        // Check for body-less binding shorthand (storage write or variable binding).
        if mb.arms.len() == 1 && mb.arms[0].body.is_empty() {
            match &mb.arms[0].pattern {
                Pattern::Binding {
                    name,
                    is_storage_ref,
                    span: pat_span,
                } => {
                    let src = self.lower_expr(&mb.source);
                    if *is_storage_ref {
                        // Storage write — register name dynamically if needed.
                        self.storage_names.insert(name.clone());
                        self.emit_inst(
                            *pat_span,
                            InstKind::StorageStore {
                                name: name.clone(),
                                src,
                            },
                        );
                    } else {
                        // Local variable binding.
                        self.define_var(name, src);
                    }
                    return;
                }
                _ => {}
            }
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

        let mut arm_labels: Vec<Label> = Vec::new();
        for _ in &mb.arms {
            arm_labels.push(self.alloc_label());
        }

        for (i, arm) in mb.arms.iter().enumerate() {
            let arm_label = arm_labels[i];
            let next_label = if i + 1 < arm_labels.len() {
                arm_labels[i + 1]
            } else {
                catch_all_label
            };

            self.emit_inst(arm.tag_span, InstKind::BlockLabel { label: arm_label, params: vec![] });
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
            self.emit_inst(arm.tag_span, InstKind::BlockLabel { label: arm_body_label, params: vec![] });

            // Bind pattern variables.
            self.lower_pattern_bind(&arm.pattern, source_reg, arm.tag_span);

            // Lower arm body (use indent-adjusted body if available).
            let body = adjusted_arm_bodies
                .as_ref()
                .map(|bodies| bodies[i].as_slice())
                .unwrap_or(&arm.body);
            self.lower_nodes(body);

            // Hoist body-less variable bindings to the outer scope.
            self.hoist_bodyless_bindings();
            self.pop_scope();
            // After body, jump to end (no loop).
            self.emit_inst(arm.tag_span, InstKind::Jump { label: end_label, args: vec![] });
        }

        // Catch-all block.
        self.emit_inst(mb.span, InstKind::BlockLabel { label: catch_all_label, params: vec![] });
        if let Some(catch_all) = &mb.catch_all {
            self.push_scope();
            let body = adjusted_catch_all_body
                .as_deref()
                .unwrap_or(&catch_all.body);
            self.lower_nodes(body);
            self.hoist_bodyless_bindings();
            self.pop_scope();
        }

        self.emit_inst(mb.span, InstKind::BlockLabel { label: end_label, params: vec![] });
    }

    // --- Iter block lowering ---

    fn lower_iter_block(&mut self, ib: &IterBlock) {
        // Pre-compute indent-adjusted body and catch-all body.
        let adjusted_body: Option<Vec<Node>> = ib
            .indent
            .as_ref()
            .map(|modifier| apply_indent_to_nodes(&ib.body, modifier));
        let adjusted_catch_all_body: Option<Vec<Node>> =
            ib.indent.as_ref().and_then(|modifier| {
                ib.catch_all
                    .as_ref()
                    .map(|ca| apply_indent_to_nodes(&ca.body, modifier))
            });

        let source_reg = self.lower_expr(&ib.source);

        // Initialize iterator.
        let iter_reg = self.alloc_val();
        self.set_val_type(iter_reg, Ty::Unit);
        self.emit_inst(
            ib.span,
            InstKind::IterInit {
                dst: iter_reg,
                src: source_reg,
            },
        );

        let loop_label = self.alloc_label();
        let catch_all_label = self.alloc_label();
        let end_label = self.alloc_label();

        // Loop header.
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: loop_label,
                params: vec![],
            },
        );

        let value_reg = self.alloc_val();
        let done_reg = self.alloc_val();
        self.set_val_type(value_reg, self.iterable_elem_type(source_reg));
        self.set_val_type(done_reg, Ty::Bool);
        self.emit_inst(
            ib.span,
            InstKind::IterNext {
                dst_value: value_reg,
                dst_done: done_reg,
                iter: iter_reg,
            },
        );

        // If done, jump to catch-all.
        let body_label = self.alloc_label();
        self.emit_inst(
            ib.span,
            InstKind::JumpIf {
                cond: done_reg,
                then_label: catch_all_label,
                then_args: vec![],
                else_label: body_label,
                else_args: vec![],
            },
        );
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: body_label,
                params: vec![],
            },
        );

        // Bind pattern (irrefutable — no test needed).
        self.push_scope();
        self.lower_pattern_bind(&ib.pattern, value_reg, ib.span);

        // Lower body.
        let body = adjusted_body.as_deref().unwrap_or(&ib.body);
        self.lower_nodes(body);

        self.hoist_bodyless_bindings();
        self.pop_scope();

        // Jump back to loop.
        self.emit_inst(
            ib.span,
            InstKind::Jump {
                label: loop_label,
                args: vec![],
            },
        );

        // Catch-all block.
        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: catch_all_label,
                params: vec![],
            },
        );
        if let Some(catch_all) = &ib.catch_all {
            self.push_scope();
            let ca_body = adjusted_catch_all_body
                .as_deref()
                .unwrap_or(&catch_all.body);
            self.lower_nodes(ca_body);
            self.hoist_bodyless_bindings();
            self.pop_scope();
        }

        self.emit_inst(
            ib.span,
            InstKind::BlockLabel {
                label: end_label,
                params: vec![],
            },
        );
    }

    // --- Pattern test lowering ---

    /// Emit instructions that test whether `src_reg` matches `pattern`.
    /// Returns a register holding a Bool (true = match).
    fn lower_pattern_test(&mut self, pattern: &Pattern, src_reg: Val, span: Span) -> Val {
        match pattern {
            Pattern::Binding {
                is_storage_ref: false,
                ..
            } => {
                // Variable binding always matches.
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::Const {
                        dst,
                        value: Literal::Bool(true),
                    },
                );
                dst
            }

            Pattern::Binding {
                name,
                is_storage_ref: true,
                ..
            } => {
                let dst = self.alloc_val();
                self.set_val_type(dst, Ty::Bool);
                // Storage write in pattern position — always matches, store value.
                self.storage_names.insert(name.clone());
                self.emit_inst(
                    span,
                    InstKind::StorageStore {
                        name: name.clone(),
                        src: src_reg,
                    },
                );
                // Always matches.
                self.emit_inst(
                    span,
                    InstKind::Const {
                        dst,
                        value: Literal::Bool(true),
                    },
                );
                dst
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
                head,
                rest,
                tail,
                ..
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
                let result_label = self.alloc_label();

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

                self.emit_inst(span, InstKind::BlockLabel { label: check_elems_label, params: vec![] });

                // Check each head element.
                let mut all_ok = len_ok;
                let elem_ty = self.list_elem_type(src_reg);
                for (i, p) in head.iter().enumerate() {
                    let elem = self.alloc_val();
                    self.set_val_type(elem, elem_ty.clone());
                    self.emit_inst(
                        span,
                        InstKind::ListIndex {
                            dst: elem,
                            list: src_reg,
                            index: i as i32,
                        },
                    );
                    let elem_ok = self.lower_pattern_test(p, elem, span);
                    // AND with accumulated result.
                    let combined = self.alloc_val();
                    self.set_val_type(combined, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::BinOp {
                            dst: combined,
                            op: acvus_ast::BinOp::And,
                            left: all_ok,
                            right: elem_ok,
                        },
                    );
                    all_ok = combined;
                }

                // Check each tail element (indexed from end).
                for (i, p) in tail.iter().enumerate() {
                    let elem = self.alloc_val();
                    self.set_val_type(elem, elem_ty.clone());
                    self.emit_inst(
                        span,
                        InstKind::ListIndex {
                            dst: elem,
                            list: src_reg,
                            index: -((tail.len() - i) as i32),
                        },
                    );
                    let elem_ok = self.lower_pattern_test(p, elem, span);
                    let combined = self.alloc_val();
                    self.set_val_type(combined, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::BinOp {
                            dst: combined,
                            op: acvus_ast::BinOp::And,
                            left: all_ok,
                            right: elem_ok,
                        },
                    );
                    all_ok = combined;
                }

                // Merge via block parameter.
                let result_param = self.alloc_val();
                self.set_val_type(result_param, Ty::Bool);

                self.emit_inst(span, InstKind::Jump { label: result_label, args: vec![all_ok] });

                // Fail path.
                self.emit_inst(span, InstKind::BlockLabel { label: fail_label, params: vec![] });
                let false_reg = self.alloc_val();
                self.set_val_type(false_reg, Ty::Bool);
                self.emit_inst(
                    span,
                    InstKind::Const {
                        dst: false_reg,
                        value: Literal::Bool(false),
                    },
                );
                self.emit_inst(span, InstKind::Jump { label: result_label, args: vec![false_reg] });

                // Result: block parameter receives value from both paths.
                self.emit_inst(span, InstKind::BlockLabel { label: result_label, params: vec![result_param] });
                result_param
            }

            Pattern::Object { fields, .. } => {
                // Test that all keys exist.
                let mut all_ok = {
                    let r = self.alloc_val();
                    self.set_val_type(r, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::Const {
                            dst: r,
                            value: Literal::Bool(true),
                        },
                    );
                    r
                };

                for ObjectPatternField { key, pattern, .. } in fields {
                    let key_ok = self.alloc_val();
                    self.set_val_type(key_ok, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::TestObjectKey {
                            dst: key_ok,
                            src: src_reg,
                            key: key.clone(),
                        },
                    );

                    // Get the field value.
                    let field_val = self.alloc_val();
                    self.set_val_type(field_val, self.object_field_type(src_reg, key));
                    self.emit_inst(
                        span,
                        InstKind::ObjectGet {
                            dst: field_val,
                            object: src_reg,
                            key: key.clone(),
                        },
                    );

                    // Test the sub-pattern.
                    let sub_ok = self.lower_pattern_test(pattern, field_val, span);

                    // Combine.
                    let combined = self.alloc_val();
                    self.set_val_type(combined, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::BinOp {
                            dst: combined,
                            op: acvus_ast::BinOp::And,
                            left: all_ok,
                            right: sub_ok,
                        },
                    );
                    all_ok = combined;
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
                let mut all_ok = {
                    let r = self.alloc_val();
                    self.set_val_type(r, Ty::Bool);
                    self.emit_inst(
                        span,
                        InstKind::Const {
                            dst: r,
                            value: Literal::Bool(true),
                        },
                    );
                    r
                };

                for (i, elem) in elements.iter().enumerate() {
                    match elem {
                        TuplePatternElem::Wildcard(_) => {
                            // Skip wildcards — no test needed.
                        }
                        TuplePatternElem::Pattern(pat) => {
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
                            let sub_ok = self.lower_pattern_test(pat, field_val, span);
                            let combined = self.alloc_val();
                            self.set_val_type(combined, Ty::Bool);
                            self.emit_inst(
                                span,
                                InstKind::BinOp {
                                    dst: combined,
                                    op: acvus_ast::BinOp::And,
                                    left: all_ok,
                                    right: sub_ok,
                                },
                            );
                            all_ok = combined;
                        }
                    }
                }

                all_ok
            }
        }
    }

    /// Extract integer bounds from range pattern.
    fn extract_range_bounds(&self, start: &Pattern, end: &Pattern) -> (i64, i64) {
        let s = match start {
            Pattern::Literal {
                value: Literal::Int(n),
                ..
            } => *n,
            _ => 0,
        };
        let e = match end {
            Pattern::Literal {
                value: Literal::Int(n),
                ..
            } => *n,
            _ => 0,
        };
        (s, e)
    }

    /// Emit instructions that bind pattern variables from a matched value.
    fn lower_pattern_bind(&mut self, pattern: &Pattern, src_reg: Val, span: Span) {
        match pattern {
            Pattern::Binding {
                name,
                is_storage_ref,
                ..
            } => {
                if *is_storage_ref {
                    // Storage path — value already stored via pattern_test or body-less binding.
                    // No local define; lookup will fall through to storage_load.
                } else {
                    self.set_origin(src_reg, ValOrigin::Named(name.clone()));
                    self.define_var(name, src_reg);
                }
            }

            Pattern::Literal { .. } => {
                // No bindings.
            }

            Pattern::List {
                head,
                rest,
                tail,
                ..
            } => {
                let elem_ty = self.list_elem_type(src_reg);
                for (i, p) in head.iter().enumerate() {
                    let elem = self.alloc_val();
                    self.set_val_type(elem, elem_ty.clone());
                    self.emit_inst(
                        span,
                        InstKind::ListIndex {
                            dst: elem,
                            list: src_reg,
                            index: i as i32,
                        },
                    );
                    self.lower_pattern_bind(p, elem, span);
                }

                // If `..` is present and there's a rest binding, extract the slice.
                if rest.is_some() {
                    // The rest is implicitly discarded (no variable captures it).
                    // Elements in tail are indexed from end.
                }

                for (i, p) in tail.iter().enumerate() {
                    let elem = self.alloc_val();
                    self.set_val_type(elem, elem_ty.clone());
                    self.emit_inst(
                        span,
                        InstKind::ListIndex {
                            dst: elem,
                            list: src_reg,
                            index: -((tail.len() - i) as i32),
                        },
                    );
                    self.lower_pattern_bind(p, elem, span);
                }
            }

            Pattern::Object { fields, .. } => {
                for ObjectPatternField { key, pattern, .. } in fields {
                    let field_val = self.alloc_val();
                    self.set_val_type(field_val, self.object_field_type(src_reg, key));
                    self.emit_inst(
                        span,
                        InstKind::ObjectGet {
                            dst: field_val,
                            object: src_reg,
                            key: key.clone(),
                        },
                    );
                    self.lower_pattern_bind(pattern, field_val, span);
                }
            }

            Pattern::Range { .. } => {
                // No bindings in range patterns.
            }

            Pattern::Tuple { elements, .. } => {
                for (i, elem) in elements.iter().enumerate() {
                    match elem {
                        TuplePatternElem::Wildcard(_) => {
                            // Skip wildcards — no binding.
                        }
                        TuplePatternElem::Pattern(pat) => {
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
                }
            }
        }
    }

    // --- Free variable analysis ---

    fn free_vars_in_expr(&self, expr: &Expr, bound: &HashSet<String>) -> Vec<String> {
        let mut free = Vec::new();
        let mut seen = HashSet::new();
        self.collect_free_vars(expr, bound, &mut free, &mut seen);
        free
    }

    fn collect_free_vars(
        &self,
        expr: &Expr,
        bound: &HashSet<String>,
        free: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) {
        match expr {
            Expr::Ident {
                name,
                is_storage_ref: false,
                ..
            } => {
                if !bound.contains(name) && !seen.contains(name) {
                    if self.lookup_var(name).is_some() {
                        seen.insert(name.clone());
                        free.push(name.clone());
                    }
                }
            }
            Expr::Ident {
                is_storage_ref: true,
                ..
            } => {
                // Storage refs resolve via StorageLoad at runtime — no capture needed.
            }
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
                    inner_bound.insert(p.name.clone());
                }
                self.collect_free_vars(body, &inner_bound, free, seen);
            }
            Expr::Paren { inner, .. } => {
                self.collect_free_vars(inner, bound, free, seen);
            }
            Expr::List {
                head, tail, ..
            } => {
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
                    if let TupleElem::Expr(e) = elem {
                        self.collect_free_vars(e, bound, free, seen);
                    }
                }
            }
            Expr::Group { elements, .. } => {
                for e in elements {
                    self.collect_free_vars(e, bound, free, seen);
                }
            }
            Expr::Literal { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typeck::TypeChecker;
    use std::collections::HashMap;

    fn lower(source: &str) -> MirModule {
        lower_with(source, HashMap::new(), HashMap::new())
    }

    fn lower_with(
        source: &str,
        storage: HashMap<String, Ty>,
        externs: HashMap<String, (Vec<Ty>, Ty)>,
    ) -> MirModule {
        let template = acvus_ast::parse(source).expect("parse failed");
        let storage_names: HashSet<String> = storage.keys().cloned().collect();
        let checker = TypeChecker::new(storage, externs);
        let type_map = checker.check_template(&template).expect("type check failed");
        let lowerer = Lowerer::new(type_map, storage_names);
        let (module, _hints) = lowerer.lower_template(&template);
        module
    }

    #[test]
    fn lower_text_node() {
        let module = lower("hello world");
        assert_eq!(module.main.insts.len(), 1);
        assert!(matches!(
            &module.main.insts[0].kind,
            InstKind::EmitText(0)
        ));
        assert_eq!(module.texts[0], "hello world");
    }

    #[test]
    fn lower_string_emit() {
        let module = lower(r#"{{ "hello" }}"#);
        assert!(module.main.insts.len() >= 2);
        assert!(matches!(&module.main.insts[0].kind, InstKind::Const { .. }));
        assert!(matches!(
            &module.main.insts[1].kind,
            InstKind::EmitValue(_)
        ));
    }

    #[test]
    fn lower_storage_write() {
        let storage = HashMap::from([("count".into(), Ty::Int)]);
        let module = lower_with("{{ $count = 42 }}", storage, HashMap::new());
        let has_store = module.main.insts.iter().any(|i| {
            matches!(&i.kind, InstKind::StorageStore { name, .. } if name == "count")
        });
        assert!(has_store);
    }

    #[test]
    fn lower_match_block() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        // Use a non-binding pattern to trigger full match block (no iteration).
        let module = lower_with(
            r#"{{ true = $name == "test" }}matched{{/}}"#,
            storage,
            HashMap::new(),
        );
        // Match block should NOT have IterInit/IterNext — it's single-value matching.
        let has_iter_init = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::IterInit { .. }));
        assert!(!has_iter_init);
        // Should have pattern test and conditional jump.
        let has_jump_if = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::JumpIf { .. }));
        assert!(has_jump_if);
    }

    #[test]
    fn lower_extern_call() {
        let externs =
            HashMap::from([("fetch_user".into(), (vec![Ty::Int], Ty::String))]);
        let module = lower_with(
            r#"{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}"#,
            HashMap::new(),
            externs,
        );
        let has_async_call = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::AsyncCall { .. }));
        assert!(has_async_call);
    }

    #[test]
    fn lower_builtin_call() {
        let storage = HashMap::from([("n".into(), Ty::Int)]);
        let module = lower_with(
            r#"{{ $n | to_string }}"#,
            storage,
            HashMap::new(),
        );
        let has_call = module
            .main
            .insts
            .iter()
            .any(|i| matches!(&i.kind, InstKind::Call { func, .. } if func == "to_string"));
        assert!(has_call);
    }

    #[test]
    fn adjust_text_indent_decrease() {
        let text = "first\n    second\n      third";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result, "first\n  second\n    third");
    }

    #[test]
    fn adjust_text_indent_decrease_clamp() {
        let text = "first\n second\n  third";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(4));
        assert_eq!(result, "first\nsecond\nthird");
    }

    #[test]
    fn adjust_text_indent_increase() {
        let text = "first\nsecond\n  third";
        let result = adjust_text_indent(text, &IndentModifier::Increase(3));
        assert_eq!(result, "first\n   second\n     third");
    }

    #[test]
    fn adjust_text_indent_first_line_untouched() {
        let text = "  first\n  second";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result, "  first\nsecond");
    }

    #[test]
    fn adjust_text_indent_no_newline() {
        let text = "  hello";
        let result = adjust_text_indent(text, &IndentModifier::Decrease(2));
        assert_eq!(result, "  hello");
    }

    #[test]
    fn lower_match_block_indent_decrease() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let source = "{{ true = $name == \"test\" }}\n    matched\n    here{{/-2}}";
        let module = lower_with(source, storage, HashMap::new());
        // Find EmitText instructions and verify the text was adjusted.
        let texts: Vec<&str> = module
            .main
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::EmitText(idx) => Some(module.texts[*idx].as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("\n  matched\n  here")));
    }

    #[test]
    fn lower_match_block_indent_increase() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let source = "{{ true = $name == \"test\" }}\nmatched{{/+4}}";
        let module = lower_with(source, storage, HashMap::new());
        let texts: Vec<&str> = module
            .main
            .insts
            .iter()
            .filter_map(|i| match &i.kind {
                InstKind::EmitText(idx) => Some(module.texts[*idx].as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.iter().any(|t| t.contains("\n    matched")));
    }
}
