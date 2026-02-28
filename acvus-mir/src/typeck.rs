use std::collections::{BTreeMap, HashMap};

use acvus_ast::{
    BinOp, Expr, IterBlock, Literal, MatchBlock, Node, ObjectExprField, ObjectPatternField,
    Pattern, Span, Template, TupleElem, TuplePatternElem,
};

use crate::builtins::builtins;
use crate::error::{MirError, MirErrorKind};
use crate::ty::{Ty, TySubst};

/// Maps each AST Span to its inferred type.
pub type TypeMap = HashMap<Span, Ty>;

pub struct TypeChecker {
    /// Stack of scopes: each scope maps variable names to types.
    scopes: Vec<HashMap<String, Ty>>,
    /// Storage variable types (provided externally).
    storage_types: HashMap<String, Ty>,
    /// External function signatures: (param_types, return_type).
    extern_fns: HashMap<String, (Vec<Ty>, Ty)>,
    /// Unification state.
    subst: TySubst,
    /// Accumulated type map.
    type_map: TypeMap,
    /// Accumulated errors.
    errors: Vec<MirError>,
}

impl TypeChecker {
    pub fn new(
        storage_types: HashMap<String, Ty>,
        extern_fns: HashMap<String, (Vec<Ty>, Ty)>,
    ) -> Self {
        Self {
            scopes: vec![HashMap::new()],
            storage_types,
            extern_fns,
            subst: TySubst::new(),
            type_map: TypeMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn check_template(mut self, template: &Template) -> Result<TypeMap, Vec<MirError>> {
        self.check_nodes(&template.body);
        if self.errors.is_empty() {
            // Resolve all types in the type map.
            let resolved: TypeMap = self
                .type_map
                .iter()
                .map(|(span, ty)| (*span, self.subst.resolve(ty)))
                .collect();
            Ok(resolved)
        } else {
            Err(self.errors)
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &str, ty: Ty) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup_var(&self, name: &str) -> Option<Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    fn error(&mut self, kind: MirErrorKind, span: Span) {
        self.errors.push(MirError { kind, span });
    }

    fn record(&mut self, span: Span, ty: Ty) {
        self.type_map.insert(span, ty);
    }

    fn check_nodes(&mut self, nodes: &[Node]) {
        for node in nodes {
            self.check_node(node);
        }
    }

    fn check_node(&mut self, node: &Node) {
        match node {
            Node::Text { .. } | Node::Comment { .. } => {}
            Node::InlineExpr { expr, span } => {
                let ty = self.check_expr(expr);
                let resolved = self.subst.resolve(&ty);
                match &resolved {
                    Ty::String => {}
                    Ty::Var(_) => {
                        // Try to unify with String.
                        if self.subst.unify(&ty, &Ty::String).is_err() {
                            self.error(MirErrorKind::EmitNotString { actual: resolved }, *span);
                        }
                    }
                    _ => {
                        self.error(MirErrorKind::EmitNotString { actual: resolved }, *span);
                    }
                }
            }
            Node::MatchBlock(mb) => {
                self.check_match_block(mb);
            }
            Node::IterBlock(ib) => {
                self.check_iter_block(ib);
            }
        }
    }

    fn check_match_block(&mut self, mb: &MatchBlock) {
        // Body-less variable binding: define in current scope (no push/pop).
        if self.is_bodyless_var_binding(mb) {
            let source_ty = self.check_expr(&mb.source);
            let resolved_source = self.subst.resolve(&source_ty);
            self.check_pattern(&mb.arms[0].pattern, &resolved_source, mb.arms[0].tag_span);
            return;
        }

        let source_ty = self.check_expr(&mb.source);
        let resolved_source = self.subst.resolve(&source_ty);

        for arm in &mb.arms {
            let match_ty = self.pattern_match_type(&arm.pattern, &resolved_source);

            self.push_scope();
            self.check_pattern(&arm.pattern, &match_ty, arm.tag_span);
            self.check_nodes(&arm.body);
            // Hoist body-less variable bindings to the outer scope.
            self.hoist_bodyless_bindings();
            self.pop_scope();
        }

        if let Some(catch_all) = &mb.catch_all {
            self.push_scope();
            self.check_nodes(&catch_all.body);
            self.hoist_bodyless_bindings();
            self.pop_scope();
        }
    }

    fn check_iter_block(&mut self, ib: &IterBlock) {
        let source_ty = self.check_expr(&ib.source);
        let resolved = self.subst.resolve(&source_ty);

        let elem_ty = match &resolved {
            Ty::List(inner) => inner.as_ref().clone(),
            Ty::Range => Ty::Int,
            _ => {
                self.error(
                    MirErrorKind::SourceNotIterable { actual: resolved },
                    ib.span,
                );
                return;
            }
        };

        self.push_scope();
        self.check_pattern(&ib.pattern, &elem_ty, ib.span);
        self.check_nodes(&ib.body);
        self.hoist_bodyless_bindings();
        self.pop_scope();

        if let Some(catch_all) = &ib.catch_all {
            self.push_scope();
            self.check_nodes(&catch_all.body);
            self.hoist_bodyless_bindings();
            self.pop_scope();
        }
    }

    /// Check if a match block is a body-less variable binding.
    fn is_bodyless_var_binding(&self, mb: &MatchBlock) -> bool {
        if mb.arms.len() != 1 || !mb.arms[0].body.is_empty() {
            return false;
        }
        match &mb.arms[0].pattern {
            Pattern::Binding { is_storage_ref: false, .. } => true,
            Pattern::Binding { is_storage_ref: true, .. } => {
                // $name = expr is body-less both for storage write and local var definition.
                true
            }
            _ => false,
        }
    }

    /// Copy variables defined in the current (top) scope to the parent scope.
    /// This hoists body-less variable bindings out of match arm scopes.
    fn hoist_bodyless_bindings(&mut self) {
        let len = self.scopes.len();
        if len >= 2 {
            let top = self.scopes[len - 1].clone();
            for (name, ty) in top {
                self.scopes[len - 2].insert(name, ty);
            }
        }
    }

    /// Determine what type a pattern matches against given the source type.
    /// List patterns match the source directly (destructuring).
    /// Other patterns match against the iterated element type.
    fn pattern_match_type(&self, pattern: &Pattern, source_ty: &Ty) -> Ty {
        match pattern {
            Pattern::List { .. } | Pattern::Tuple { .. } => {
                // List/Tuple patterns destructure the source as a whole.
                source_ty.clone()
            }
            _ => {
                // Other patterns match iterated elements.
                match source_ty {
                    Ty::List(inner) => inner.as_ref().clone(),
                    Ty::Range => Ty::Int,
                    _ => source_ty.clone(),
                }
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Ty {
        let ty = match expr {
            Expr::Literal { value, span } => {
                let ty = match value {
                    Literal::Int(_) => Ty::Int,
                    Literal::Float(_) => Ty::Float,
                    Literal::String(_) => Ty::String,
                    Literal::Bool(_) => Ty::Bool,
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::Ident {
                name,
                is_storage_ref,
                span,
            } => {
                // $name: check storage first, then local scope.
                // bare name: check local scope only.
                let ty = if *is_storage_ref {
                    if let Some(ty) = self.storage_types.get(name) {
                        ty.clone()
                    } else if let Some(ty) = self.lookup_var(name) {
                        ty
                    } else {
                        self.error(MirErrorKind::UndefinedVariable(format!("${name}")), *span);
                        Ty::Unit
                    }
                } else if let Some(ty) = self.lookup_var(name) {
                    ty
                } else {
                    self.error(MirErrorKind::UndefinedVariable(name.clone()), *span);
                    Ty::Unit
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.check_expr(left);
                let rt = self.check_expr(right);
                let lt = self.subst.resolve(&lt);
                let rt = self.subst.resolve(&rt);

                let ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        // Unify first so type variables resolve before checking.
                        let unified = self.subst.unify(&lt, &rt).is_ok();
                        let rl = self.subst.resolve(&lt);
                        let rr = self.subst.resolve(&rt);
                        if unified {
                            match (&rl, &rr) {
                                (Ty::Int, Ty::Int) => Ty::Int,
                                (Ty::Float, Ty::Float) => Ty::Float,
                                (Ty::String, Ty::String) if *op == BinOp::Add => Ty::String,
                                _ => {
                                    self.error(
                                        MirErrorKind::TypeMismatchBinOp {
                                            op: op_str(*op),
                                            left: rl,
                                            right: rr,
                                        },
                                        *span,
                                    );
                                    Ty::Unit
                                }
                            }
                        } else {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: op_str(*op),
                                    left: rl,
                                    right: rr,
                                },
                                *span,
                            );
                            Ty::Unit
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.subst.unify(&lt, &rt).is_err() {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: op_str(*op),
                                    left: lt,
                                    right: rt,
                                },
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        if !matches!(&lt, Ty::Bool) {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: op_str(*op),
                                    left: lt.clone(),
                                    right: rt.clone(),
                                },
                                *span,
                            );
                        }
                        if !matches!(&rt, Ty::Bool) {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: op_str(*op),
                                    left: lt,
                                    right: rt,
                                },
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        // Unify first so type variables resolve before checking.
                        let unified = self.subst.unify(&lt, &rt).is_ok();
                        let rl = self.subst.resolve(&lt);
                        let rr = self.subst.resolve(&rt);
                        if !unified || !matches!((&rl, &rr), (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float)) {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: op_str(*op),
                                    left: rl,
                                    right: rr,
                                },
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::UnaryOp { op, operand, span } => {
                let ot = self.check_expr(operand);
                let ot = self.subst.resolve(&ot);
                let ty = match op {
                    acvus_ast::UnaryOp::Neg => match &ot {
                        Ty::Int => Ty::Int,
                        Ty::Float => Ty::Float,
                        Ty::Var(_) => {
                            // Unresolved type variable — try Int first, Float is also valid
                            // but we can't know yet. Keep as-is (the var will resolve later
                            // via surrounding context). Return a fresh var that unifies with
                            // the operand.
                            ot.clone()
                        }
                        _ => {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: "-",
                                    left: ot,
                                    right: Ty::Unit,
                                },
                                *span,
                            );
                            Ty::Unit
                        }
                    },
                    acvus_ast::UnaryOp::Not => {
                        if matches!(&ot, Ty::Var(_)) {
                            // Unresolved — unify with Bool.
                            let _ = self.subst.unify(&ot, &Ty::Bool);
                        } else if !matches!(&ot, Ty::Bool) {
                            self.error(
                                MirErrorKind::TypeMismatchBinOp {
                                    op: "!",
                                    left: ot,
                                    right: Ty::Unit,
                                },
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::FieldAccess {
                object,
                field,
                span,
            } => {
                let ot_raw = self.check_expr(object);
                let ot = self.subst.resolve(&ot_raw);
                let ty = match &ot {
                    Ty::Object(fields) => {
                        if let Some(ft) = fields.get(field) {
                            ft.clone()
                        } else if let Some(leaf_var) = self.subst.find_leaf_var(&ot_raw) {
                            // Object came from a Var (partial projection constraint).
                            // Extend the Object with the new field.
                            let fresh = self.subst.fresh_var();
                            let mut new_fields = fields.clone();
                            new_fields.insert(field.clone(), fresh.clone());
                            self.subst.rebind(leaf_var, Ty::Object(new_fields));
                            fresh
                        } else {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: ot.clone(),
                                    field: field.clone(),
                                },
                                *span,
                            );
                            Ty::Unit
                        }
                    }
                    Ty::Var(_) => {
                        // Unresolved Var — create partial Object constraint.
                        let fresh = self.subst.fresh_var();
                        let partial_obj =
                            Ty::Object(BTreeMap::from([(field.clone(), fresh.clone())]));
                        if self.subst.unify(&ot_raw, &partial_obj).is_err() {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: ot,
                                    field: field.clone(),
                                },
                                *span,
                            );
                        }
                        fresh
                    }
                    _ => {
                        self.error(
                            MirErrorKind::UndefinedField {
                                object_ty: ot,
                                field: field.clone(),
                            },
                            *span,
                        );
                        Ty::Unit
                    }
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::FuncCall { func, args, span } => {
                let ty = self.check_func_call(func, args, *span);
                self.record(*span, ty.clone());
                ty
            }

            Expr::Pipe { left, right, span } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`
                // `a | f` → `f(a)`
                let ty = match right.as_ref() {
                    Expr::FuncCall {
                        func,
                        args,
                        span: _,
                    } => {
                        let mut new_args = vec![left.as_ref().clone()];
                        new_args.extend(args.iter().cloned());
                        self.check_func_call(func, &new_args, *span)
                    }
                    Expr::Ident {
                        name: _,
                        is_storage_ref: false,
                        span: _,
                    } => {
                        let func_expr = right.as_ref().clone();
                        let new_args = vec![left.as_ref().clone()];
                        self.check_func_call(&func_expr, &new_args, *span)
                    }
                    _ => {
                        // Right side of pipe must be a function or function call.
                        let _lt = self.check_expr(left);
                        let _rt = self.check_expr(right);
                        self.error(
                            MirErrorKind::UndefinedFunction("<pipe rhs>".into()),
                            *span,
                        );
                        Ty::Unit
                    }
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::Lambda {
                params,
                body,
                span,
            } => {
                self.push_scope();
                let mut param_types = Vec::new();
                for p in params {
                    let pt = self.subst.fresh_var();
                    self.define_var(&p.name, pt.clone());
                    self.record(p.span, pt.clone());
                    param_types.push(pt);
                }
                let ret = self.check_expr(body);
                self.pop_scope();
                let ty = Ty::Fn {
                    params: param_types,
                    ret: Box::new(ret),
                };
                self.record(*span, ty.clone());
                ty
            }

            Expr::Paren { inner, span } => {
                let ty = self.check_expr(inner);
                self.record(*span, ty.clone());
                ty
            }

            Expr::List {
                head,
                rest,
                tail,
                span,
            } => {
                let all_elems: Vec<_> = head.iter().chain(tail.iter()).collect();
                if all_elems.is_empty() && rest.is_none() {
                    // Empty list `[]` — ambiguous without context.
                    self.error(MirErrorKind::AmbiguousEmptyList, *span);
                    let ty = Ty::List(Box::new(Ty::Unit));
                    self.record(*span, ty.clone());
                    return ty;
                }

                let elem_ty = if let Some(first) = all_elems.first() {
                    self.check_expr(first)
                } else {
                    // Only `..` with no elements: fresh var.
                    self.subst.fresh_var()
                };

                for elem in all_elems.iter().skip(1) {
                    let et = self.check_expr(elem);
                    if self.subst.unify(&elem_ty, &et).is_err() {
                        let resolved_expected = self.subst.resolve(&elem_ty);
                        let resolved_got = self.subst.resolve(&et);
                        self.error(
                            MirErrorKind::HeterogeneousList {
                                expected: resolved_expected,
                                got: resolved_got,
                            },
                            *span,
                        );
                    }
                }

                let ty = Ty::List(Box::new(self.subst.resolve(&elem_ty)));
                self.record(*span, ty.clone());
                ty
            }

            Expr::Object { fields, span } => {
                let mut field_types = BTreeMap::new();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(value);
                    field_types.insert(key.clone(), ft);
                }
                let ty = Ty::Object(field_types);
                self.record(*span, ty.clone());
                ty
            }

            Expr::Range {
                start,
                end,
                kind: _,
                span,
            } => {
                let st = self.check_expr(start);
                let et = self.check_expr(end);
                let st = self.subst.resolve(&st);
                let et = self.subst.resolve(&et);
                if !matches!(&st, Ty::Int) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: st }, *span);
                }
                if !matches!(&et, Ty::Int) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: et }, *span);
                }
                let ty = Ty::Range;
                self.record(*span, ty.clone());
                ty
            }

            Expr::Tuple { elements, span } => {
                let elem_types: Vec<Ty> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.check_expr(e),
                        TupleElem::Wildcard(_) => self.subst.fresh_var(),
                    })
                    .collect();
                let ty = Ty::Tuple(elem_types);
                self.record(*span, ty.clone());
                ty
            }

            Expr::Group { elements, span } => {
                // Group is only valid as lambda param list (handled by parser).
                let ty = if let Some(last) = elements.last() {
                    for e in &elements[..elements.len() - 1] {
                        self.check_expr(e);
                    }
                    self.check_expr(last)
                } else {
                    Ty::Unit
                };
                self.record(*span, ty.clone());
                ty
            }
        };
        ty
    }

    fn check_func_call(&mut self, func: &Expr, args: &[Expr], call_span: Span) -> Ty {
        // Try to resolve as a named function (builtin or extern).
        let func_name = match func {
            Expr::Ident {
                name,
                is_storage_ref: false,
                ..
            } => Some(name.as_str()),
            _ => None,
        };

        if let Some(name) = func_name {
            // Check builtins first.
            for b in builtins() {
                if b.name == name {
                    let (param_tys, ret_ty) = (b.signature)(&mut self.subst);
                    let arg_types: Vec<Ty> = args.iter().map(|a| self.check_expr(a)).collect();

                    if arg_types.len() != param_tys.len() {
                        self.error(
                            MirErrorKind::ArityMismatch {
                                func: name.to_string(),
                                expected: param_tys.len(),
                                got: arg_types.len(),
                            },
                            call_span,
                        );
                        return Ty::Unit;
                    }

                    for (at, pt) in arg_types.iter().zip(param_tys.iter()) {
                        if self.subst.unify(at, pt).is_err() {
                            let ra = self.subst.resolve(at);
                            let rp = self.subst.resolve(pt);
                            self.error(
                                MirErrorKind::UnificationFailure {
                                    expected: rp,
                                    got: ra,
                                },
                                call_span,
                            );
                        }
                    }

                    return self.subst.resolve(&ret_ty);
                }
            }

            // Check extern functions.
            if let Some((param_tys, ret_ty)) = self.extern_fns.get(name).cloned() {
                let arg_types: Vec<Ty> = args.iter().map(|a| self.check_expr(a)).collect();

                if arg_types.len() != param_tys.len() {
                    self.error(
                        MirErrorKind::ArityMismatch {
                            func: name.to_string(),
                            expected: param_tys.len(),
                            got: arg_types.len(),
                        },
                        call_span,
                    );
                    return Ty::Unit;
                }

                for (at, pt) in arg_types.iter().zip(param_tys.iter()) {
                    if self.subst.unify(at, pt).is_err() {
                        let ra = self.subst.resolve(at);
                        let rp = self.subst.resolve(pt);
                        self.error(
                            MirErrorKind::UnificationFailure {
                                expected: rp,
                                got: ra,
                            },
                            call_span,
                        );
                    }
                }

                return self.subst.resolve(&ret_ty);
            }

            // Check if it's a local variable that's a function type.
            if let Some(var_ty) = self.lookup_var(name) {
                let resolved = self.subst.resolve(&var_ty);
                return self.check_callable(&resolved, args, call_span);
            }

            self.error(
                MirErrorKind::UndefinedFunction(name.to_string()),
                call_span,
            );
            return Ty::Unit;
        }

        // Not a simple name — evaluate the function expression.
        let ft = self.check_expr(func);
        let resolved = self.subst.resolve(&ft);
        self.check_callable(&resolved, args, call_span)
    }

    fn check_callable(&mut self, func_ty: &Ty, args: &[Expr], call_span: Span) -> Ty {
        match func_ty {
            Ty::Fn { params, ret } => {
                let arg_types: Vec<Ty> = args.iter().map(|a| self.check_expr(a)).collect();
                if arg_types.len() != params.len() {
                    self.error(
                        MirErrorKind::ArityMismatch {
                            func: "<closure>".to_string(),
                            expected: params.len(),
                            got: arg_types.len(),
                        },
                        call_span,
                    );
                    return Ty::Unit;
                }
                for (at, pt) in arg_types.iter().zip(params.iter()) {
                    if self.subst.unify(at, pt).is_err() {
                        let ra = self.subst.resolve(at);
                        let rp = self.subst.resolve(pt);
                        self.error(
                            MirErrorKind::UnificationFailure {
                                expected: rp,
                                got: ra,
                            },
                            call_span,
                        );
                    }
                }
                self.subst.resolve(ret)
            }
            Ty::Var(_) => {
                // Unknown callable — create fresh return type and unify.
                let arg_types: Vec<Ty> = args.iter().map(|a| self.check_expr(a)).collect();
                let ret = self.subst.fresh_var();
                let fn_ty = Ty::Fn {
                    params: arg_types,
                    ret: Box::new(ret.clone()),
                };
                if self.subst.unify(func_ty, &fn_ty).is_err() {
                    self.error(
                        MirErrorKind::UndefinedFunction("<expr>".into()),
                        call_span,
                    );
                    return Ty::Unit;
                }
                self.subst.resolve(&ret)
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction("<not callable>".into()),
                    call_span,
                );
                Ty::Unit
            }
        }
    }

    fn check_pattern(&mut self, pattern: &Pattern, source_ty: &Ty, span: Span) {
        let source_resolved = self.subst.resolve(source_ty);
        match pattern {
            Pattern::Binding {
                name,
                is_storage_ref,
                span: _,
            } => {
                if *is_storage_ref {
                    if let Some(storage_ty) = self.storage_types.get(name).cloned() {
                        // Storage write: type must match.
                        if self.subst.unify(&source_resolved, &storage_ty).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: storage_ty,
                                    source_ty: source_resolved,
                                },
                                span,
                            );
                        }
                    } else {
                        // $name not in storage → local variable definition.
                        self.define_var(name, source_resolved);
                    }
                } else {
                    // Bare name capture: bind to source type.
                    self.define_var(name, source_resolved);
                }
            }

            Pattern::Literal { value, .. } => {
                let pat_ty = match value {
                    Literal::Int(_) => Ty::Int,
                    Literal::Float(_) => Ty::Float,
                    Literal::String(_) => Ty::String,
                    Literal::Bool(_) => Ty::Bool,
                };
                if self.subst.unify(&source_resolved, &pat_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: pat_ty,
                            source_ty: source_resolved,
                        },
                        span,
                    );
                }
            }

            Pattern::List {
                head,
                rest,
                tail,
                ..
            } => {
                // Source must be List<T>.
                let elem_ty = self.subst.fresh_var();
                let list_ty = Ty::List(Box::new(elem_ty.clone()));
                if self.subst.unify(&source_resolved, &list_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: list_ty,
                            source_ty: source_resolved,
                        },
                        span,
                    );
                    return;
                }
                let resolved_elem = self.subst.resolve(&elem_ty);
                for p in head.iter().chain(tail.iter()) {
                    self.check_pattern(p, &resolved_elem, span);
                }
                let _ = rest;
            }

            Pattern::Object { fields, .. } => {
                // Source must be Object. Open matching: pattern can have subset of fields.
                match &source_resolved {
                    Ty::Object(obj_fields) => {
                        for ObjectPatternField { key, pattern, .. } in fields {
                            if let Some(field_ty) = obj_fields.get(key) {
                                self.check_pattern(pattern, field_ty, span);
                            } else {
                                self.error(
                                    MirErrorKind::UndefinedField {
                                        object_ty: source_resolved.clone(),
                                        field: key.clone(),
                                    },
                                    span,
                                );
                            }
                        }
                    }
                    _ => {
                        let pat_ty = Ty::Object(BTreeMap::new());
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: pat_ty,
                                source_ty: source_resolved,
                            },
                            span,
                        );
                    }
                }
            }

            Pattern::Range {
                start,
                end,
                kind: _,
                ..
            } => {
                // Range pattern matches Int source.
                if self.subst.unify(&source_resolved, &Ty::Int).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: Ty::Int,
                            source_ty: source_resolved,
                        },
                        span,
                    );
                }
                // Range bounds must be literal Ints (validated at pattern level).
                self.check_pattern_is_int(start, span);
                self.check_pattern_is_int(end, span);
            }

            Pattern::Tuple { elements, .. } => {
                // Source must be Tuple with matching arity.
                let elem_vars: Vec<Ty> = elements
                    .iter()
                    .map(|_| self.subst.fresh_var())
                    .collect();
                let tuple_ty = Ty::Tuple(elem_vars.clone());
                if self.subst.unify(&source_resolved, &tuple_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: tuple_ty,
                            source_ty: source_resolved,
                        },
                        span,
                    );
                    return;
                }
                for (i, elem) in elements.iter().enumerate() {
                    let resolved_elem = self.subst.resolve(&elem_vars[i]);
                    match elem {
                        TuplePatternElem::Pattern(pat) => {
                            self.check_pattern(pat, &resolved_elem, span);
                        }
                        TuplePatternElem::Wildcard(_) => {
                            // Wildcard: no binding, skip.
                        }
                    }
                }
            }
        }
    }

    fn check_pattern_is_int(&mut self, pat: &Pattern, span: Span) {
        match pat {
            Pattern::Literal {
                value: Literal::Int(_),
                ..
            } => {}
            Pattern::Literal { value, .. } => {
                let ty = match value {
                    Literal::Float(_) => Ty::Float,
                    Literal::String(_) => Ty::String,
                    Literal::Bool(_) => Ty::Bool,
                    Literal::Int(_) => unreachable!(),
                };
                self.error(MirErrorKind::RangeBoundsNotInt { actual: ty }, span);
            }
            _ => {
                self.error(
                    MirErrorKind::RangeBoundsNotInt { actual: Ty::Unit },
                    span,
                );
            }
        }
    }
}

fn op_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Eq => "==",
        BinOp::Neq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Lte => "<=",
        BinOp::Gte => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::parse;

    fn check(source: &str) -> Result<TypeMap, Vec<MirError>> {
        let template = parse(source).expect("parse failed");
        let checker = TypeChecker::new(HashMap::new(), HashMap::new());
        checker.check_template(&template)
    }

    fn check_with_storage(
        source: &str,
        storage: HashMap<String, Ty>,
    ) -> Result<TypeMap, Vec<MirError>> {
        let template = parse(source).expect("parse failed");
        let checker = TypeChecker::new(storage, HashMap::new());
        checker.check_template(&template)
    }

    fn check_with_extern(
        source: &str,
        extern_fns: HashMap<String, (Vec<Ty>, Ty)>,
    ) -> Result<TypeMap, Vec<MirError>> {
        let template = parse(source).expect("parse failed");
        let checker = TypeChecker::new(HashMap::new(), extern_fns);
        checker.check_template(&template)
    }

    #[test]
    fn literal_string_emit() {
        assert!(check("{{ \"hello\" }}").is_ok());
    }

    #[test]
    fn literal_int_emit_fails() {
        assert!(check("{{ 42 }}").is_err());
    }

    #[test]
    fn arithmetic_int() {
        // Int arithmetic result is Int, not String — emit should fail.
        assert!(check("{{ 1 + 2 }}").is_err());
    }

    #[test]
    fn arithmetic_mixed_fails() {
        // We need the result to be used somewhere. Let's use a match to avoid emit errors.
        // Actually, let's test directly: Int + Float is a type error.
        let src = r#"{{ x = 1 + 2.0 }}{{_}}{{/}}"#;
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn range_bounds_int() {
        let src = "{{ x = 0..10 }}{{_}}{{/}}";
        assert!(check(src).is_ok());
    }

    #[test]
    fn range_bounds_float_fails() {
        let src = "{{ x = 1.0..2.0 }}{{_}}{{/}}";
        assert!(check(src).is_err());
    }

    #[test]
    fn catch_all_optional() {
        // Catch-all is optional — match blocks without {{_}} should type-check fine.
        let src = "{{ x = 42 }}hello{{/}}";
        let result = check(src);
        assert!(result.is_ok());
    }

    #[test]
    fn storage_read() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let src = "{{ $name }}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn undefined_variable() {
        let src = "{{ x = unknown }}{{_}}{{/}}";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn storage_write() {
        let storage = HashMap::from([("count".into(), Ty::Int)]);
        let src = "{{ $count = 42 }}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn extern_fn_call() {
        let externs =
            HashMap::from([("fetch_user".into(), (vec![Ty::Int], Ty::String))]);
        let src = "{{ x = fetch_user(1) }}{{ x }}{{_}}{{/}}";
        assert!(check_with_extern(src, externs).is_ok());
    }

    #[test]
    fn builtin_to_string() {
        let storage = HashMap::from([("count".into(), Ty::Int)]);
        let src = "{{ $count | to_string }}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn field_access() {
        let storage = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([
                ("name".into(), Ty::String),
                ("age".into(), Ty::Int),
            ])),
        )]);
        let src = "{{ $user.name }}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn field_access_undefined() {
        let storage = HashMap::from([(
            "user".into(),
            Ty::Object(BTreeMap::from([("name".into(), Ty::String)])),
        )]);
        let src = "{{ $user.unknown }}";
        let result = check_with_storage(src, storage);
        assert!(result.is_err());
    }

    #[test]
    fn pattern_binding_captures_type() {
        let storage = HashMap::from([("name".into(), Ty::String)]);
        let src = "{{ x = $name }}{{ x }}{{_}}{{/}}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn list_pattern_matching() {
        let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ [a, b, ..] = $items }}{{ a | to_string }}{{_}}{{/}}";
        assert!(check_with_storage(src, storage).is_ok());
    }

    #[test]
    fn lambda_type_check() {
        let storage = HashMap::from([("items".into(), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ x = $items | filter(x -> x != 0) }}{{ x | to_string }}{{_}}{{/}}";
        // filter returns List<Int>, but emitting List<Int> directly requires to_string.
        // Actually x here is each iteration item, which is List<Int>. Let's adjust.
        // Actually "x = ..." creates a match block iterating over the result.
        // The result of filter is List<Int>, each element is Int.
        let result = check_with_storage(src, storage);
        assert!(result.is_ok());
    }
}
