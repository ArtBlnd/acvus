use acvus_ast::{
    BinOp, Expr, IterBlock, Literal, MatchBlock, Node, ObjectExprField, ObjectPatternField,
    Pattern, RefKind, Span, Template, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::builtins::{BuiltinId, registry};
use crate::error::{MirError, MirErrorKind};
use crate::ty::{Ty, TySubst};
use crate::variant::VariantPayload;

/// Maps each AST Span to its inferred type.
pub type TypeMap = FxHashMap<Span, Ty>;

/// Maps each builtin-call Span to the resolved BuiltinId.
/// Produced by the type checker, consumed by the lowerer.
pub type BuiltinMap = FxHashMap<Span, BuiltinId>;

pub struct TypeChecker<'a> {
    /// Interner for string interning.
    interner: &'a Interner,
    /// Context variable types (`@name`, externally declared).
    context_types: &'a FxHashMap<Astr, Ty>,
    /// Stack of scopes: each scope maps variable names to types.
    scopes: Vec<FxHashMap<Astr, Ty>>,
    /// Variable types (`$name`, inferred at first assignment).
    variable_types: FxHashMap<Astr, Ty>,
    /// Unification state.
    subst: TySubst,
    /// Cached fresh Vars for `Ty::Infer` context entries.
    infer_vars: FxHashMap<Astr, Ty>,
    /// Accumulated type map.
    type_map: TypeMap,
    /// Accumulated builtin resolutions (call_span → BuiltinId).
    builtin_map: BuiltinMap,
    /// Accumulated errors.
    errors: Vec<MirError>,
    /// Analysis mode: unknown contexts get fresh Vars instead of errors.
    analysis_mode: bool,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        interner: &'a Interner,
        context_types: &'a FxHashMap<Astr, Ty>,
    ) -> Self {
        Self {
            interner,
            scopes: vec![FxHashMap::default()],
            context_types,
            variable_types: FxHashMap::default(),
            subst: TySubst::new(),
            infer_vars: FxHashMap::default(),
            type_map: TypeMap::default(),
            builtin_map: BuiltinMap::default(),
            errors: Vec::new(),
            analysis_mode: false,
        }
    }

    /// Enable analysis mode: unknown `@context` refs produce fresh type
    /// variables instead of errors, allowing partial type inference.
    pub fn with_analysis_mode(mut self) -> Self {
        self.analysis_mode = true;
        self
    }

    pub fn check_template(
        mut self,
        template: &Template,
    ) -> Result<(TypeMap, BuiltinMap), Vec<MirError>> {
        self.check_nodes(&template.body);
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        // Resolve all types in the type map.
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        Ok((resolved, self.builtin_map))
    }

    pub fn check_script(
        self,
        script: &acvus_ast::Script,
    ) -> Result<(TypeMap, BuiltinMap, Ty), Vec<MirError>> {
        self.check_script_with_hint(script, None)
    }

    pub fn check_script_with_hint(
        mut self,
        script: &acvus_ast::Script,
        expected_tail: Option<&Ty>,
    ) -> Result<(TypeMap, BuiltinMap, Ty), Vec<MirError>> {
        for stmt in &script.stmts {
            match stmt {
                acvus_ast::Stmt::Bind { name, expr, span } => {
                    let ty = self.check_expr(false,expr);
                    self.define_var(*name, ty.clone());
                    self.record(*span, ty);
                }
                acvus_ast::Stmt::Expr(expr) => {
                    self.check_expr(false,expr);
                }
            }
        }
        let tail_ty = match &script.tail {
            Some(expr) => self.check_expr(false,expr),
            None => Ty::Unit,
        };
        // Unify tail with expected type hint (if provided) to resolve ambiguous literals
        if let Some(expected) = expected_tail
            && self.subst.unify(&tail_ty, expected).is_err()
        {
            let span = script
                .tail
                .as_ref()
                .map(|e| e.span())
                .unwrap_or(acvus_ast::Span { start: 0, end: 0 });
            self.error(
                MirErrorKind::UnificationFailure {
                    expected: self.subst.resolve(expected),
                    got: self.subst.resolve(&tail_ty),
                },
                span,
            );
        }
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved_tail = self.subst.resolve(&tail_ty);
        // Check for unresolved type variables (e.g. `[]` without hint).
        // In analysis mode, unresolved vars are expected and allowed.
        if !self.analysis_mode && contains_var(&resolved_tail) {
            let span = script
                .tail
                .as_ref()
                .map(|e| e.span())
                .unwrap_or(acvus_ast::Span { start: 0, end: 0 });
            self.error(MirErrorKind::AmbiguousType, span);
            return Err(self.errors);
        }
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        Ok((resolved, self.builtin_map, resolved_tail))
    }

    /// Like `check_template`, but always returns a (partial) TypeMap and collects errors separately.
    /// Used by analysis-mode compilation so that context-key discovery works even when
    /// the template contains type errors.
    pub fn check_template_partial(
        mut self,
        template: &Template,
    ) -> (TypeMap, BuiltinMap, Vec<MirError>) {
        self.check_nodes(&template.body);
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        (resolved, self.builtin_map, self.errors)
    }

    /// Like `check_script_with_hint`, but always returns a (partial) TypeMap and collects errors separately.
    pub fn check_script_with_hint_partial(
        mut self,
        script: &acvus_ast::Script,
        expected_tail: Option<&Ty>,
    ) -> (TypeMap, BuiltinMap, Ty, Vec<MirError>) {
        for stmt in &script.stmts {
            match stmt {
                acvus_ast::Stmt::Bind { name, expr, span } => {
                    let ty = self.check_expr(false,expr);
                    self.define_var(*name, ty.clone());
                    self.record(*span, ty);
                }
                acvus_ast::Stmt::Expr(expr) => {
                    self.check_expr(false,expr);
                }
            }
        }
        let tail_ty = match &script.tail {
            Some(expr) => self.check_expr(false,expr),
            None => Ty::Unit,
        };
        if let Some(expected) = expected_tail
            && self.subst.unify(&tail_ty, expected).is_err()
        {
            // Record the error but don't fail — we still want the partial results.
            let span = script
                .tail
                .as_ref()
                .map(|e| e.span())
                .unwrap_or(Span { start: 0, end: 0 });
            self.error(
                MirErrorKind::UnificationFailure {
                    expected: self.subst.resolve(expected),
                    got: self.subst.resolve(&tail_ty),
                },
                span,
            );
        }
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        let resolved_tail = self.subst.resolve(&tail_ty);
        (resolved, self.builtin_map, resolved_tail, self.errors)
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: Astr, ty: Ty) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_var(&self, name: Astr) -> Option<Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(&name) {
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

    /// Record the type at a span and return it.
    fn record_ret(&mut self, span: Span, ty: Ty) -> Ty {
        self.record(span, ty.clone());
        ty
    }

    /// Resolve a context variable (`@name`) to its type.
    fn resolve_context_type(&mut self, name: Astr, span: Span) -> Ty {
        let use_infer_var = match self.context_types.get(&name) {
            Some(Ty::Infer) => true,
            Some(ty) => return ty.clone(),
            None => self.analysis_mode,
        };
        if use_infer_var {
            return self
                .infer_vars
                .entry(name)
                .or_insert_with(|| self.subst.fresh_var())
                .clone();
        }
        self.error(
            MirErrorKind::UndefinedContext(self.interner.resolve(name).to_string()),
            span,
        );
        Ty::Error
    }

    fn binop_error(&mut self, op: &'static str, left: Ty, right: Ty, span: Span) {
        self.error(MirErrorKind::TypeMismatchBinOp { op, left, right }, span);
    }

    /// Check arity and unify argument types against parameter types.
    /// Returns `true` if arity matched, `false` (with error emitted) if not.
    fn check_args(
        &mut self,
        func: &str,
        arg_types: &[Ty],
        param_tys: &[Ty],
        call_span: Span,
    ) -> bool {
        if arg_types.len() != param_tys.len() {
            self.error(
                MirErrorKind::ArityMismatch {
                    func: func.to_string(),
                    expected: param_tys.len(),
                    got: arg_types.len(),
                },
                call_span,
            );
            return false;
        }
        for (at, pt) in arg_types.iter().zip(param_tys.iter()) {
            if self.subst.unify(at, pt).is_err() {
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: self.subst.resolve(pt),
                        got: self.subst.resolve(at),
                    },
                    call_span,
                );
            }
        }
        true
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
                let ty = self.check_expr(false,expr);
                let resolved = self.subst.resolve(&ty);
                match &resolved {
                    Ty::String | Ty::Error => {}
                    Ty::Var(_) => {
                        if self.subst.unify(&ty, &Ty::String).is_err() {
                            self.error(MirErrorKind::EmitNotString { actual: resolved }, *span);
                        }
                    }
                    _ => self.error(MirErrorKind::EmitNotString { actual: resolved }, *span),
                }
            }
            Node::MatchBlock(mb) => self.check_match_block(mb),
            Node::IterBlock(ib) => self.check_iter_block(ib),
        }
    }

    fn check_match_block(&mut self, mb: &MatchBlock) {
        // Body-less variable binding: define in current scope (no push/pop).
        if self.is_bodyless_var_binding(mb) {
            let source_ty = self.check_expr(false,&mb.source);
            if matches!(&mb.arms[0].pattern, Pattern::Variant { .. }) {
                self.check_pattern(&mb.arms[0].pattern, &source_ty, mb.arms[0].tag_span);
            } else {
                let resolved_source = self.subst.resolve(&source_ty);
                self.check_pattern(&mb.arms[0].pattern, &resolved_source, mb.arms[0].tag_span);
            }
            return;
        }

        let source_ty = self.check_expr(false,&mb.source);
        let resolved_source = self.subst.resolve(&source_ty);

        for arm in &mb.arms {
            let match_ty = self.pattern_match_type(&arm.pattern, &resolved_source);
            // For patterns that destructure the source as a whole and may
            // contain nested variants (Variant, Tuple, List), pass the
            // unresolved source so unify can trace the Var chain and rebind
            // the merged type. This ensures variant sets from all arms are
            // accumulated into the same type variable.
            // Object patterns are NOT included because they go through the
            // iteration path (pattern_match_type extracts element types).
            let pattern_source = match &arm.pattern {
                Pattern::Variant { .. }
                | Pattern::Tuple { .. }
                | Pattern::List { .. } => source_ty.clone(),
                _ => match_ty,
            };

            self.push_scope();
            self.check_pattern(&arm.pattern, &pattern_source, arm.tag_span);
            self.check_nodes(&arm.body);
            self.pop_scope();
        }

        if let Some(catch_all) = &mb.catch_all {
            self.push_scope();
            self.check_nodes(&catch_all.body);
            self.pop_scope();
        }
    }

    fn check_iter_block(&mut self, ib: &IterBlock) {
        let source_ty = self.check_expr(false,&ib.source);
        let resolved = self.subst.resolve(&source_ty);

        let elem_ty = match &resolved {
            Ty::List(inner) => inner.as_ref().clone(),
            Ty::Range => Ty::Int,
            Ty::Error => Ty::Error,
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
        self.pop_scope();

        if let Some(catch_all) = &ib.catch_all {
            self.push_scope();
            self.check_nodes(&catch_all.body);
            self.pop_scope();
        }
    }

    /// Check if a match block is a body-less variable binding.
    fn is_bodyless_var_binding(&self, mb: &MatchBlock) -> bool {
        mb.arms.len() == 1
            && mb.arms[0].body.is_empty()
            && matches!(&mb.arms[0].pattern, Pattern::Binding { .. })
    }


    /// Determine what type a pattern matches against given the source type.
    /// List patterns match the source directly (destructuring).
    /// Other patterns match against the iterated element type.
    fn pattern_match_type(&self, pattern: &Pattern, source_ty: &Ty) -> Ty {
        match pattern {
            Pattern::List { .. } | Pattern::Tuple { .. } | Pattern::Variant { .. } => {
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

    fn check_expr(&mut self, allow_non_pure: bool, expr: &Expr) -> Ty {
        match expr {
            Expr::Literal { value, span } => {
                let ty = match value {
                    Literal::Int(_) => Ty::Int,
                    Literal::Float(_) => Ty::Float,
                    Literal::String(_) => Ty::String,
                    Literal::Bool(_) => Ty::Bool,
                    Literal::Byte(_) => Ty::Byte,
                    Literal::List(elems) => {
                        if elems.is_empty() {
                            let elem = self.subst.fresh_var();
                            Ty::List(Box::new(elem))
                        } else {
                            let first_ty = self.literal_ty(&elems[0]);
                            for elem in &elems[1..] {
                                let elem_ty = self.literal_ty(elem);
                                if self.subst.unify(&first_ty, &elem_ty).is_err() {
                                    self.error(
                                        MirErrorKind::HeterogeneousList {
                                            expected: self.subst.resolve(&first_ty),
                                            got: self.subst.resolve(&elem_ty),
                                        },
                                        *span,
                                    );
                                }
                            }
                            Ty::List(Box::new(self.subst.resolve(&first_ty)))
                        }
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::Ident {
                name,
                ref_kind,
                span,
            } => {
                let ty = match ref_kind {
                    RefKind::Context => {
                        let ty = self.resolve_context_type(*name, *span);
                        if !allow_non_pure && !ty.is_pure() && !ty.is_error() {
                            self.error(
                                MirErrorKind::NonPureContextLoad {
                                    name: self.interner.resolve(*name).to_string(),
                                    ty: ty.clone(),
                                },
                                *span,
                            );
                            Ty::Error
                        } else {
                            ty
                        }
                    }
                    RefKind::Variable => match self.variable_types.get(name) {
                        Some(ty) => ty.clone(),
                        None => {
                            self.error(
                                MirErrorKind::UndefinedVariable(format!(
                                    "${}",
                                    self.interner.resolve(*name)
                                )),
                                *span,
                            );
                            Ty::Error
                        }
                    },
                    RefKind::Value => match self.lookup_var(*name) {
                        Some(ty) => ty,
                        None => {
                            self.error(
                                MirErrorKind::UndefinedVariable(
                                    self.interner.resolve(*name).to_string(),
                                ),
                                *span,
                            );
                            Ty::Error
                        }
                    },
                };
                self.record_ret(*span, ty)
            }

            Expr::BinaryOp {
                left,
                op,
                right,
                span,
            } => {
                let lt = self.check_expr(false,left);
                let rt = self.check_expr(false,right);
                let lt = self.subst.resolve(&lt);
                let rt = self.subst.resolve(&rt);

                // Early guard: if either operand is Error, suppress cascading errors.
                if lt.is_error() || rt.is_error() {
                    let ty = match op {
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Mod
                        | BinOp::Xor
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::Shl
                        | BinOp::Shr => Ty::Error,
                        _ => Ty::Bool,
                    };
                    return self.record_ret(*span, ty);
                }

                let ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if self.subst.unify(&lt, &rt).is_err() {
                            self.binop_error(
                                op_str(*op),
                                self.subst.resolve(&lt),
                                self.subst.resolve(&rt),
                                *span,
                            );
                            return self.record_ret(*span, Ty::Error);
                        }
                        let rl = self.subst.resolve(&lt);
                        match &rl {
                            Ty::Int | Ty::Float | Ty::Var(_) => rl,
                            Ty::String if *op == BinOp::Add => Ty::String,
                            _ => {
                                self.binop_error(op_str(*op), rl, self.subst.resolve(&rt), *span);
                                Ty::Error
                            }
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.subst.unify(&lt, &rt).is_err() {
                            self.binop_error(op_str(*op), lt, rt, *span);
                        }
                        Ty::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        let lok = self.subst.unify(&lt, &Ty::Bool).is_ok();
                        let rok = self.subst.unify(&rt, &Ty::Bool).is_ok();
                        if !lok || !rok {
                            self.binop_error(
                                op_str(*op),
                                self.subst.resolve(&lt),
                                self.subst.resolve(&rt),
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                    BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                        let lok = self.subst.unify(&lt, &Ty::Int).is_ok();
                        let rok = self.subst.unify(&rt, &Ty::Int).is_ok();
                        if !lok || !rok {
                            self.binop_error(
                                op_str(*op),
                                self.subst.resolve(&lt),
                                self.subst.resolve(&rt),
                                *span,
                            );
                        }
                        Ty::Int
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        let ok = self.subst.unify(&lt, &rt).is_ok()
                            && matches!(self.subst.resolve(&lt), Ty::Int | Ty::Float | Ty::Var(_));
                        if !ok {
                            self.binop_error(
                                op_str(*op),
                                self.subst.resolve(&lt),
                                self.subst.resolve(&rt),
                                *span,
                            );
                        }
                        Ty::Bool
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::UnaryOp { op, operand, span } => {
                let ot = self.check_expr(false,operand);
                let ot = self.subst.resolve(&ot);

                // Early guard: if operand is Error, suppress cascading errors.
                if ot.is_error() {
                    let ty = match op {
                        acvus_ast::UnaryOp::Neg => Ty::Error,
                        acvus_ast::UnaryOp::Not => Ty::Bool,
                    };
                    return self.record_ret(*span, ty);
                }

                let ty = match op {
                    acvus_ast::UnaryOp::Neg => match &ot {
                        Ty::Int => Ty::Int,
                        Ty::Float => Ty::Float,
                        Ty::Var(_) => ot.clone(),
                        _ => {
                            self.binop_error("-", ot, Ty::Error, *span);
                            Ty::Error
                        }
                    },
                    acvus_ast::UnaryOp::Not => {
                        match &ot {
                            Ty::Bool => {}
                            Ty::Var(_) => {
                                let _ = self.subst.unify(&ot, &Ty::Bool);
                            }
                            _ => self.binop_error("!", ot, Ty::Error, *span),
                        }
                        Ty::Bool
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::FieldAccess {
                object,
                field,
                span,
            } => {
                let ot_raw = self.check_expr(false,object);
                let ot = self.subst.resolve(&ot_raw);
                let field_key = *field;
                let field_str = || self.interner.resolve(*field).to_string();
                let ty = match &ot {
                    Ty::Error => Ty::Error,
                    Ty::Object(fields) if fields.contains_key(&field_key) => {
                        fields[&field_key].clone()
                    }
                    Ty::Object(fields) => {
                        let Some(leaf_var) = self.subst.find_leaf_var(&ot_raw) else {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: ot.clone(),
                                    field: field_str(),
                                },
                                *span,
                            );
                            return Ty::Error;
                        };
                        let fresh = self.subst.fresh_var();
                        let mut new_fields = fields.clone();
                        new_fields.insert(field_key, fresh.clone());
                        self.subst.rebind(leaf_var, Ty::Object(new_fields));
                        fresh
                    }
                    Ty::Var(_) => {
                        let fresh = self.subst.fresh_var();
                        let partial_obj =
                            Ty::Object(FxHashMap::from_iter([(field_key, fresh.clone())]));
                        if self.subst.unify(&ot_raw, &partial_obj).is_err() {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: ot,
                                    field: field_str(),
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
                                field: field_str(),
                            },
                            *span,
                        );
                        Ty::Error
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::FuncCall { func, args, span } => {
                let ty = self.check_func_call(func, args, None, *span);
                self.record_ret(*span, ty)
            }

            Expr::Pipe { left, right, span } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`
                // `a | f` → `f(a)`
                let pipe_left = Some(left.as_ref());
                let ty = match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.check_func_call(func, args, pipe_left, *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value | RefKind::Context,
                        ..
                    } => self.check_func_call(right, &[], pipe_left, *span),
                    _ => {
                        let lt = self.check_expr(false,left);
                        let rt = self.check_expr(false,right);
                        self.check_callable(&rt, &[], &Some(lt), *span)
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::Lambda { params, body, span } => {
                self.push_scope();
                let mut param_types = Vec::new();
                for p in params {
                    let pt = self.subst.fresh_var();
                    self.define_var(p.name, pt.clone());
                    self.record(p.span, pt.clone());
                    param_types.push(pt);
                }
                let ret = self.check_expr(false,body);
                self.pop_scope();
                let ty = Ty::Fn {
                    params: param_types,
                    ret: Box::new(ret),
                    is_extern: false,
                };
                self.record_ret(*span, ty)
            }

            Expr::Paren { inner, span } => {
                let ty = self.check_expr(false,inner);
                self.record_ret(*span, ty)
            }

            Expr::List {
                head,
                rest,
                tail,
                span,
            } => {
                let all_elems: Vec<_> = head.iter().chain(tail.iter()).collect();
                if all_elems.is_empty() && rest.is_none() {
                    // Empty list `[]` — element type unknown, use fresh var.
                    // If no hint resolves it, we report the error after resolve.
                    let elem = self.subst.fresh_var();
                    let ty = Ty::List(Box::new(elem));
                    return self.record_ret(*span, ty);
                }

                let elem_ty = match all_elems.first() {
                    Some(first) => self.check_expr(false,first),
                    None => self.subst.fresh_var(), // Only `..` with no elements: fresh var.
                };

                for elem in all_elems.iter().skip(1) {
                    let et = self.check_expr(false,elem);
                    if self.subst.unify(&elem_ty, &et).is_err() {
                        self.error(
                            MirErrorKind::HeterogeneousList {
                                expected: self.subst.resolve(&elem_ty),
                                got: self.subst.resolve(&et),
                            },
                            *span,
                        );
                    }
                }

                let ty = Ty::List(Box::new(self.subst.resolve(&elem_ty)));
                self.record_ret(*span, ty)
            }

            Expr::Object { fields, span } => {
                let mut field_types = FxHashMap::default();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(false,value);
                    field_types.insert(*key, ft);
                }
                let ty = Ty::Object(field_types);
                self.record_ret(*span, ty)
            }

            Expr::Range {
                start,
                end,
                kind: _,
                span,
            } => {
                let st = self.check_expr(false,start);
                let et = self.check_expr(false,end);
                let st = self.subst.resolve(&st);
                let et = self.subst.resolve(&et);
                if !matches!(&st, Ty::Int | Ty::Error) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: st }, *span);
                }
                if !matches!(&et, Ty::Int | Ty::Error) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: et }, *span);
                }
                self.record_ret(*span, Ty::Range)
            }

            Expr::Tuple { elements, span } => {
                let elem_types: Vec<Ty> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.check_expr(false,e),
                        TupleElem::Wildcard(_) => self.subst.fresh_var(),
                    })
                    .collect();
                let ty = Ty::Tuple(elem_types);
                self.record_ret(*span, ty)
            }

            Expr::Group { elements, span } => {
                // Group is only valid as lambda param list (handled by parser).
                let Some(last) = elements.last() else {
                    self.record(*span, Ty::Unit);
                    return Ty::Unit;
                };
                for e in &elements[..elements.len() - 1] {
                    self.check_expr(false,e);
                }
                let ty = self.check_expr(false,last);
                self.record_ret(*span, ty)
            }

            Expr::Variant {
                enum_name: ast_enum_name,
                tag,
                payload,
                span,
            } => {
                // Try builtin (Option) first.
                if let Some((_enum_name, type_params, variant_payload)) =
                    self.resolve_builtin_variant(ast_enum_name, *tag)
                {
                    match &variant_payload {
                        VariantPayload::TypeParam(idx) => {
                            let Some(inner_expr) = payload else {
                                self.error(
                                    MirErrorKind::UnificationFailure {
                                        expected: Ty::Error,
                                        got: Ty::Unit,
                                    },
                                    *span,
                                );
                                return Ty::Error;
                            };
                            let inner_ty = self.check_expr(false,inner_expr);
                            if self.subst.unify(&type_params[*idx], &inner_ty).is_err() {
                                self.error(
                                    MirErrorKind::UnificationFailure {
                                        expected: self.subst.resolve(&type_params[*idx]),
                                        got: self.subst.resolve(&inner_ty),
                                    },
                                    *span,
                                );
                            }
                        }
                        VariantPayload::None => {}
                    }
                    // Builtin Option → Ty::Option
                    let inner = self.subst.resolve(&type_params[0]);
                    let ty = Ty::Option(Box::new(inner));
                    return self.record_ret(*span, ty);
                }

                // Structural enum: requires qualified name (A::B).
                let Some(enum_name) = ast_enum_name else {
                    self.error(
                        MirErrorKind::UndefinedFunction(format!(
                            "unknown variant: {}",
                            self.interner.resolve(*tag)
                        )),
                        *span,
                    );
                    return Ty::Error;
                };

                let payload_ty = match payload {
                    Some(expr) => {
                        let ty = self.check_expr(false,expr);
                        Some(Box::new(ty))
                    }
                    None => None,
                };

                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty);
                let ty = Ty::Enum {
                    name: *enum_name,
                    variants,
                };
                self.record_ret(*span, ty)
            }

            Expr::Block { stmts, tail, span } => {
                self.push_scope();
                for stmt in stmts {
                    match stmt {
                        acvus_ast::Stmt::Bind { name, expr, span } => {
                            let ty = self.check_expr(false,expr);
                            self.define_var(*name, ty.clone());
                            self.record(*span, ty);
                        }
                        acvus_ast::Stmt::Expr(expr) => {
                            self.check_expr(false,expr);
                        }
                    }
                }
                let ty = self.check_expr(false,tail);
                self.pop_scope();
                self.record_ret(*span, ty)
            }
        }
    }

    fn check_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Expr>,
        call_span: Span,
    ) -> Ty {
        // Collect argument types, prepending pipe_left if present.
        let pipe_ty = pipe_left.map(|e| self.check_expr(false,e));

        // Try to resolve as a named function (builtin or extern).
        let Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } = func
        else {
            // Not a simple name — evaluate the function expression.
            // allow_non_pure: function call position, non-pure types (extern fn) are OK.
            let ft = self.check_expr(true, func);
            let resolved = self.subst.resolve(&ft);
            return self.check_callable(&resolved, args, &pipe_ty, call_span);
        };

        // Check builtins first.
        let name_str = self.interner.resolve(*name);
        let candidates = registry().candidates(name_str);
        if !candidates.is_empty() {
            // Evaluate argument types eagerly (needed for overload resolution).
            let arg_types: Vec<Ty> = pipe_ty
                .iter()
                .cloned()
                .chain(args.iter().map(|a| self.check_expr(false, a)))
                .collect();

            if candidates.len() == 1 {
                // Fast path: single candidate, no overload resolution needed.
                let entry = registry().get(candidates[0]);
                let (param_tys, ret_ty) = (entry.signature)(&mut self.subst);
                if !self.check_args(name_str, &arg_types, &param_tys, call_span) {
                    return Ty::Error;
                }
                if let Some(check) = entry.constraint {
                    let resolved_args: Vec<Ty> =
                        arg_types.iter().map(|t| self.subst.resolve(t)).collect();
                    if let Some(msg) = check(&resolved_args, self.interner) {
                        self.error(MirErrorKind::BuiltinConstraint(msg), call_span);
                    }
                }
                self.builtin_map.insert(call_span, candidates[0]);
                return self.subst.resolve(&ret_ty);
            }

            // Multiple candidates: try each with snapshot/rollback.
            for &cand_id in candidates {
                let snap = self.subst.snapshot();
                let entry = registry().get(cand_id);
                let (param_tys, ret_ty) = (entry.signature)(&mut self.subst);

                if arg_types.len() != param_tys.len() {
                    self.subst.rollback(snap);
                    continue;
                }

                let mut ok = true;
                for (arg, param) in arg_types.iter().zip(param_tys.iter()) {
                    if self.subst.unify(arg, param).is_err() {
                        ok = false;
                        break;
                    }
                }

                if ok {
                    // Check constraint if present.
                    if let Some(check) = entry.constraint {
                        let resolved_args: Vec<Ty> =
                            arg_types.iter().map(|t| self.subst.resolve(t)).collect();
                        if let Some(msg) = check(&resolved_args, self.interner) {
                            self.error(MirErrorKind::BuiltinConstraint(msg), call_span);
                        }
                    }
                    self.builtin_map.insert(call_span, cand_id);
                    return self.subst.resolve(&ret_ty);
                }

                self.subst.rollback(snap);
            }

            // No candidate matched.
            let resolved_args: Vec<Ty> =
                arg_types.iter().map(|t| self.subst.resolve(t)).collect();
            self.error(
                MirErrorKind::NoMatchingOverload {
                    name: name_str.to_string(),
                    arg_tys: resolved_args,
                },
                call_span,
            );
            return Ty::Error;
        }

        // Check local variable with function type.
        if let Some(var_ty) = self.lookup_var(*name) {
            let resolved = self.subst.resolve(&var_ty);
            return self.check_callable(&resolved, args, &pipe_ty, call_span);
        }

        self.error(
            MirErrorKind::UndefinedFunction(self.interner.resolve(*name).to_string()),
            call_span,
        );
        return Ty::Error;
    }

    fn check_callable(
        &mut self,
        func_ty: &Ty,
        args: &[Expr],
        pipe_ty: &Option<Ty>,
        call_span: Span,
    ) -> Ty {
        // Early exit for non-callable types.
        match func_ty {
            Ty::Fn { .. } | Ty::Var(_) => {}
            Ty::Error => {
                for a in args {
                    self.check_expr(false,a);
                }
                return Ty::Error;
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction("<not callable>".to_string()),
                    call_span,
                );
                return Ty::Error;
            }
        }

        let arg_types: Vec<Ty> = pipe_ty
            .iter()
            .cloned()
            .chain(args.iter().map(|a| self.check_expr(false,a)))
            .collect();

        match func_ty {
            Ty::Fn { params, ret, .. } => {
                if !self.check_args("<closure>", &arg_types, params, call_span) {
                    return Ty::Error;
                }
                self.subst.resolve(ret)
            }
            Ty::Var(_) => {
                let ret = self.subst.fresh_var();
                let fn_ty = Ty::Fn {
                    params: arg_types,
                    ret: Box::new(ret.clone()),
                    is_extern: false,
                };
                if self.subst.unify(func_ty, &fn_ty).is_err() {
                    self.error(
                        MirErrorKind::UndefinedFunction("<expr>".to_string()),
                        call_span,
                    );
                    return Ty::Error;
                }
                self.subst.resolve(&ret)
            }
            _ => unreachable!(),
        }
    }

    fn check_pattern(&mut self, pattern: &Pattern, source_ty: &Ty, span: Span) {
        let source_resolved = self.subst.resolve(source_ty);
        match pattern {
            Pattern::Binding {
                name,
                ref_kind,
                span: _,
            } => match ref_kind {
                RefKind::Context => {
                    self.error(
                        MirErrorKind::ContextWriteAttempt(self.interner.resolve(*name).to_string()),
                        span,
                    );
                }
                RefKind::Variable => {
                    let Some(existing_ty) = self.variable_types.get(name).cloned() else {
                        self.variable_types.insert(*name, source_resolved);
                        return;
                    };
                    if self.subst.unify(&source_resolved, &existing_ty).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: existing_ty,
                                source_ty: source_resolved,
                            },
                            span,
                        );
                    }
                }
                RefKind::Value => {
                    self.define_var(*name, source_resolved);
                }
            },

            Pattern::Literal { value, .. } => {
                let pat_ty = self.literal_ty(value);
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

            Pattern::List { head, tail, .. } => {
                // Reuse existing element Var when source already resolves to a
                // List. Same rationale as Tuple above.
                let shallow = self.subst.shallow_resolve(source_ty);
                let elem_ty = match shallow {
                    Ty::List(ref inner) => (**inner).clone(),
                    _ => {
                        let var = self.subst.fresh_var();
                        let list_ty = Ty::List(Box::new(var.clone()));
                        if self.subst.unify(source_ty, &list_ty).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: list_ty,
                                    source_ty: source_resolved,
                                },
                                span,
                            );
                            return;
                        }
                        var
                    }
                };
                for p in head.iter().chain(tail.iter()) {
                    self.check_pattern(p, &elem_ty, span);
                }
            }

            Pattern::Object { fields, .. } => {
                // If source is already a concrete Object, match fields directly (open/subset).
                // Otherwise, build an Object from pattern fields and unify to infer the type.
                let obj_fields = if let Ty::Object(obj_fields) = &source_resolved {
                    obj_fields.clone()
                } else {
                    let field_vars: FxHashMap<Astr, Ty> = fields
                        .iter()
                        .map(|f| (f.key, self.subst.fresh_var()))
                        .collect();
                    let obj_ty = Ty::Object(field_vars.clone());
                    if self.subst.unify(source_ty, &obj_ty).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: obj_ty,
                                source_ty: source_resolved,
                            },
                            span,
                        );
                        return;
                    }
                    field_vars
                };
                for ObjectPatternField { key, pattern, .. } in fields {
                    let Some(field_ty) = obj_fields.get(key) else {
                        self.error(
                            MirErrorKind::UndefinedField {
                                object_ty: source_resolved.clone(),
                                field: self.interner.resolve(*key).to_string(),
                            },
                            span,
                        );
                        continue;
                    };
                    let resolved = self.subst.resolve(field_ty);
                    self.check_pattern(pattern, &resolved, span);
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
                // Reuse existing element Vars when source already resolves to a
                // Tuple. This preserves the Var chain so nested Variant patterns
                // can accumulate merged variant sets across match arms via
                // find_leaf_var.
                let shallow = self.subst.shallow_resolve(source_ty);
                let elem_tys = match shallow {
                    Ty::Tuple(ref existing) if existing.len() == elements.len() => {
                        existing.clone()
                    }
                    _ => {
                        let vars: Vec<Ty> =
                            elements.iter().map(|_| self.subst.fresh_var()).collect();
                        let tuple_ty = Ty::Tuple(vars.clone());
                        if self.subst.unify(source_ty, &tuple_ty).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: tuple_ty,
                                    source_ty: source_resolved,
                                },
                                span,
                            );
                            return;
                        }
                        vars
                    }
                };
                for (i, elem) in elements.iter().enumerate() {
                    let TuplePatternElem::Pattern(pat) = elem else {
                        continue; // Wildcard: no binding, skip.
                    };
                    self.check_pattern(pat, &elem_tys[i], span);
                }
            }

            Pattern::Variant {
                enum_name: ast_enum_name,
                tag,
                payload,
                ..
            } => {
                // Try builtin (Option) first.
                if let Some((_enum_name, type_params, variant_payload)) =
                    self.resolve_builtin_variant(ast_enum_name, *tag)
                {
                    let enum_ty = Ty::Option(Box::new(
                        self.subst.resolve(&type_params[0]),
                    ));
                    if self.subst.unify(&source_resolved, &enum_ty).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: enum_ty,
                                source_ty: source_resolved,
                            },
                            span,
                        );
                        return;
                    }

                    if let VariantPayload::TypeParam(idx) = &variant_payload {
                        let resolved_inner = self.subst.resolve(&type_params[*idx]);
                        if let Some(inner_pat) = payload {
                            self.check_pattern(inner_pat, &resolved_inner, span);
                        }
                    }
                    return;
                }

                // Structural enum: requires qualified name.
                let Some(enum_name) = ast_enum_name else {
                    self.error(
                        MirErrorKind::UndefinedFunction(format!(
                            "unknown variant: {}",
                            self.interner.resolve(*tag)
                        )),
                        span,
                    );
                    return;
                };

                // Build Ty::Enum with this single variant.
                let payload_ty = if payload.is_some() {
                    Some(Box::new(self.subst.fresh_var()))
                } else {
                    None
                };
                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty.clone());
                let enum_ty = Ty::Enum {
                    name: *enum_name,
                    variants,
                };
                // Unify against the original (unresolved) source_ty so that
                // find_leaf_var can trace the Var chain and rebind the merged type.
                if self.subst.unify(source_ty, &enum_ty).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: enum_ty,
                            source_ty: source_resolved,
                        },
                        span,
                    );
                    return;
                }

                // Bind payload pattern if present.
                if let Some(inner_pat) = payload {
                    let inner_ty = payload_ty
                        .map(|ty| self.subst.resolve(&ty))
                        .unwrap_or(Ty::Error);
                    self.check_pattern(inner_pat, &inner_ty, span);
                }
            }
        }
    }

    /// Try to resolve a variant tag as a builtin enum (Option).
    /// Returns None if the tag is not a builtin variant.
    fn resolve_builtin_variant(
        &mut self,
        ast_enum_name: &Option<Astr>,
        tag: Astr,
    ) -> Option<(Astr, Vec<Ty>, VariantPayload)> {
        let tag_str = self.interner.resolve(tag);
        let option_name = self.interner.intern("Option");

        // Check qualified name if present.
        if let Some(ename) = ast_enum_name {
            if *ename != option_name {
                return None;
            }
        }

        let payload = match tag_str {
            "Some" => VariantPayload::TypeParam(0),
            "None" => VariantPayload::None,
            _ => return None,
        };

        let type_params = vec![self.subst.fresh_var()];
        Some((option_name, type_params, payload))
    }

    fn literal_ty(&self, lit: &Literal) -> Ty {
        match lit {
            Literal::Int(_) => Ty::Int,
            Literal::Float(_) => Ty::Float,
            Literal::String(_) => Ty::String,
            Literal::Bool(_) => Ty::Bool,
            Literal::Byte(_) => Ty::Byte,
            Literal::List(elems) => match elems.first() {
                Some(first) => Ty::List(Box::new(self.literal_ty(first))),
                None => Ty::List(Box::new(Ty::Error)),
            },
        }
    }

    fn check_pattern_is_int(&mut self, pat: &Pattern, span: Span) {
        match pat {
            Pattern::Literal {
                value: Literal::Int(_),
                ..
            } => {}
            Pattern::Literal { value, .. } => {
                let ty = self.literal_ty(value);
                self.error(MirErrorKind::RangeBoundsNotInt { actual: ty }, span);
            }
            _ => {
                self.error(MirErrorKind::RangeBoundsNotInt { actual: Ty::Error }, span);
            }
        }
    }
}

fn contains_var(ty: &Ty) -> bool {
    match ty {
        Ty::Var(_) => true,
        Ty::List(inner) | Ty::Option(inner) => contains_var(inner),
        Ty::Object(fields) => fields.values().any(contains_var),
        Ty::Tuple(elems) => elems.iter().any(contains_var),
        Ty::Fn { params, ret, .. } => params.iter().any(contains_var) || contains_var(ret),
        Ty::Enum { variants, .. } => variants
            .values()
            .any(|p| p.as_ref().map_or(false, |ty| contains_var(ty))),
        _ => false,
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
        BinOp::Xor => "^",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Mod => "%",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_ast::parse;

    fn check(source: &str) -> Result<TypeMap, Vec<MirError>> {
        let interner = Interner::new();
        check_with_interner(source, &FxHashMap::default(), &interner)
    }

    fn check_with_interner(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> Result<TypeMap, Vec<MirError>> {
        let template = parse(interner, source).expect("parse failed");
        let checker = TypeChecker::new(interner, context);
        checker.check_template(&template).map(|(tm, _bm)| tm)
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
    fn context_read() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "{{ @name }}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn undefined_variable() {
        let src = "{{ x = unknown }}{{_}}{{/}}";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn var_write() {
        let src = "{{ $count = 42 }}";
        assert!(check(src).is_ok());
    }

    #[test]
    fn extern_fn_call() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("fetch_user"),
            Ty::Fn {
                params: vec![Ty::Int],
                ret: Box::new(Ty::String),
                is_extern: true,
            },
        )]);
        let src = "{{ x = @fetch_user(1) }}{{ x }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn builtin_to_string() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("count"), Ty::Int)]);
        let src = "{{ @count | to_string }}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn field_access() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([
                (i.intern("name"), Ty::String),
                (i.intern("age"), Ty::Int),
            ])),
        )]);
        let src = "{{ @user.name }}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn field_access_undefined() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(
            i.intern("user"),
            Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)])),
        )]);
        let src = "{{ @user.unknown }}";
        let result = check_with_interner(src, &context, &i);
        assert!(result.is_err());
    }

    #[test]
    fn pattern_binding_captures_type() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "{{ x = @name }}{{ x }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn list_pattern_matching() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ [a, b, ..] = @items }}{{ a | to_string }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn lambda_type_check() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ x = @items | iter | filter(x -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}";
        let result = check_with_interner(src, &context, &i);
        assert!(result.is_ok());
    }

    // ── Variant (Option) ────────────────────────────────────────────

    #[test]
    fn some_int_is_option_int() {
        let src = "{{ x = Some(42) }}{{_}}{{/}}";
        assert!(check(src).is_ok());
    }

    #[test]
    fn none_is_option() {
        let src = "{{ x = None }}{{_}}{{/}}";
        assert!(check(src).is_ok());
    }

    #[test]
    fn some_pattern_extracts_inner() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::String)))]);
        let src = "{{ Some(x) = @opt }}{{ x }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn none_pattern_matches_option() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::Int)))]);
        let src = "{{ None = @opt }}none{{_}}has value{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn some_unifies_with_option_context() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::Int)))]);
        // match Some(v) against Option<Int> → v : Int
        let src = "{{ Some(v) = @opt }}{{ v | to_string }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_ok());
    }

    #[test]
    fn some_type_mismatch() {
        let i = Interner::new();
        // Some(42) is Option<Int>, cannot match against String
        let context = FxHashMap::from_iter([(i.intern("s"), Ty::String)]);
        let src = "{{ Some(x) = @s }}{{ x }}{{_}}{{/}}";
        assert!(check_with_interner(src, &context, &i).is_err());
    }

    // ── Non-pure context type tests ──

    fn extern_fn_context(interner: &Interner) -> FxHashMap<Astr, Ty> {
        FxHashMap::from_iter([
            (interner.intern("my_fn"), Ty::Fn {
                params: vec![Ty::String],
                ret: Box::new(Ty::String),
                is_extern: true,
            }),
            (interner.intern("name"), Ty::String),
        ])
    }

    #[test]
    fn extern_fn_call_ok() {
        // @my_fn("hello") — calling an extern fn is allowed.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = r#"{{ @my_fn("hello") }}"#;
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn extern_fn_bare_ref_rejected() {
        // f = @my_fn — bare reference to non-pure type is rejected.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = "{{ f = @my_fn }}{{_}}{{/}}";
        let err = check_with_interner(src, &ctx, &i);
        assert!(err.is_err(), "bare reference to extern fn should be rejected");
        let errors = err.unwrap_err();
        assert!(errors.iter().any(|e| matches!(&e.kind, MirErrorKind::NonPureContextLoad { .. })));
    }

    #[test]
    fn extern_fn_pipe_call_ok() {
        // "hello" | @my_fn — pipe into extern fn is a call, should be allowed.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = r#"{{ "hello" | @my_fn }}"#;
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn extern_fn_pipe_with_args_ok() {
        // "hello" | @my_fn — pipe with additional args.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([
            (i.intern("my_fn"), Ty::Fn {
                params: vec![Ty::String, Ty::Int],
                ret: Box::new(Ty::String),
                is_extern: true,
            }),
        ]);
        let src = r#"{{ "hello" | @my_fn(42) }}"#;
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn pure_context_ref_ok() {
        // @name — bare reference to pure type (String) is fine.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = "{{ @name }}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }
}
