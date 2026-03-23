use std::marker::PhantomData;

use acvus_ast::{
    BinOp, Expr, IterBlock, Literal, MatchBlock, Node, ObjectExprField, ObjectPatternField,
    Pattern, RefKind, Span, Template, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::{MirError, MirErrorKind};
use crate::ir::CastKind;
use crate::ty::{Effect, Param, Polarity, Purity, Ty, TySubst, TypeEnv};
use crate::variant::VariantPayload;

/// Maps each AST Span to its inferred type.
pub type TypeMap = FxHashMap<Span, Ty>;

/// Maps expression spans to the coercion needed at that point.
/// Produced by the type checker, consumed by the lowerer.
pub type CoercionMap = Vec<(Span, CastKind)>;

// ── TypeResolution: boundary between TypeChecker and Lowerer ──────────

/// Marker: unresolved type variables may remain (SCC unit, not yet complete).
#[derive(Debug, Clone)]
pub struct Unchecked;

/// Marker: no unresolved type variables (completeness verified).
#[derive(Debug, Clone)]
pub struct Checked;

/// Result of type checking a single script or template.
///
/// `S = Unchecked`: produced by `typecheck_script` / `typecheck_template`.
///   May contain unresolved type variables — other units in the same SCC
///   may resolve them before `check_completeness` is called.
///
/// `S = Checked`: produced by `check_completeness`. All type variables resolved.
///   Only `Checked` resolutions can be lowered to MIR.
#[derive(Debug, Clone)]
pub struct TypeResolution<S = Checked> {
    pub type_map: TypeMap,
    pub coercion_map: CoercionMap,
    pub tail_ty: Ty,
    /// Effect of the function body (Pure or Effectful).
    /// Determined by whether the body reads/writes contexts or calls effectful functions.
    pub body_effect: Effect,
    /// Context names directly read by this function body (shallow — no transitive).
    pub body_reads: FxHashSet<Astr>,
    /// Context names directly written by this function body (shallow — no transitive).
    pub body_writes: FxHashSet<Astr>,
    /// Free parameters inferred during analysis-mode typecheck.
    /// Empty in non-analysis mode.
    pub inferred_params: Vec<(Astr, Ty)>,
    _marker: PhantomData<S>,
}

impl<S> TypeResolution<S> {
    fn new(
        type_map: TypeMap,
        coercion_map: CoercionMap,
        tail_ty: Ty,
        body_effect: Effect,
        body_reads: FxHashSet<Astr>,
        body_writes: FxHashSet<Astr>,
        inferred_params: Vec<(Astr, Ty)>,
    ) -> Self {
        Self {
            type_map,
            coercion_map,
            tail_ty,
            body_effect,
            body_reads,
            body_writes,
            inferred_params,
            _marker: PhantomData,
        }
    }
}

/// Check completeness: verify no unresolved type variables remain,
/// re-resolve the type_map with the final TySubst state, and promote
/// to `Checked`.
///
/// Call this after all units in an SCC have been typechecked, so that
/// cross-unit type variable bindings are reflected.
pub fn check_completeness(
    resolution: TypeResolution<Unchecked>,
    subst: &TySubst,
) -> Result<TypeResolution<Checked>, Vec<MirError>> {
    let tail_ty = subst.resolve(&resolution.tail_ty);
    if contains_var(&tail_ty) {
        return Err(vec![MirError {
            kind: MirErrorKind::AmbiguousType {
                resolved_ty: tail_ty,
            },
            span: Span::ZERO,
        }]);
    }
    // Re-resolve type_map: other units in the SCC may have resolved
    // type variables that were unresolved when this unit was typechecked.
    let type_map = resolution
        .type_map
        .into_iter()
        .map(|(span, ty)| (span, subst.resolve(&ty)))
        .collect();
    Ok(TypeResolution::new(
        type_map,
        resolution.coercion_map,
        tail_ty,
        resolution.body_effect,
        resolution.body_reads,
        resolution.body_writes,
        resolution.inferred_params,
    ))
}

struct LambdaScope {
    depth: usize,
    captures: Vec<Ty>,
    effect: Effect,
}

pub struct TypeChecker<'a, 's> {
    /// Interner for string interning.
    interner: &'a Interner,
    /// Unified type environment: contexts + functions.
    env: &'a TypeEnv,
    /// Stack of scopes: each scope maps variable names to types.
    scopes: Vec<FxHashMap<Astr, Ty>>,
    /// Variable types (`$name`, inferred at first assignment).
    variable_types: FxHashMap<Astr, Ty>,
    /// Unification state (borrowed — may be shared across compilations).
    subst: &'s mut TySubst,
    /// Cached fresh Params for `Ty::Param` context entries.
    infer_vars: FxHashMap<Astr, Ty>,
    /// Accumulated type map.
    type_map: TypeMap,
    /// Accumulated coercion records (span → CastKind).
    coercion_map: CoercionMap,
    /// Accumulated errors.
    errors: Vec<MirError>,
    /// Analysis mode: unknown contexts get fresh Vars instead of errors.
    analysis_mode: bool,
    /// Stack of active lambda scopes. Each entry is (scope_depth, captures, effect).
    /// Nested lambdas push onto this stack; lookups record captures in ALL
    /// enclosing lambdas whose scope depth is exceeded.
    lambda_stack: Vec<LambdaScope>,
    /// Maps lambda expression span → body expression span.
    /// Used by `detect_fn_ret_coercion` to register coercions on the
    /// correct span (body, not lambda) so the lowerer's `maybe_cast`
    /// naturally inserts a Cast at the lambda return site.
    lambda_body_spans: FxHashMap<Span, Span>,
    /// Top-level body effect. Tracks whether the function body performs
    /// effectful operations (context read/write, effectful callee invocation).
    /// Starts as Pure; set to Effectful when any effectful operation is encountered.
    body_effect: Effect,
    /// Context names directly read by this function body (shallow).
    body_reads: FxHashSet<Astr>,
    /// Context names directly written by this function body (shallow).
    body_writes: FxHashSet<Astr>,
    /// In analysis mode: free parameters discovered during typecheck.
    /// These are `RefKind::Value` identifiers that were undefined — treated as
    /// implicit function parameters. Ordered by first occurrence.
    inferred_params: Vec<(Astr, Ty)>,
    /// Declared parameter types from Signature, consumed in order as
    /// free params are discovered during analysis mode.
    declared_param_types: Vec<Ty>,
    /// Next index into declared_param_types.
    next_declared_param: usize,
}

impl<'a, 's> TypeChecker<'a, 's> {
    pub fn new(
        interner: &'a Interner,
        env: &'a TypeEnv,
        subst: &'s mut TySubst,
    ) -> Self {
        Self {
            interner,
            scopes: vec![FxHashMap::default()],
            env,
            variable_types: FxHashMap::default(),
            subst,
            infer_vars: FxHashMap::default(),
            type_map: TypeMap::default(),
            coercion_map: CoercionMap::default(),
            errors: Vec::new(),
            analysis_mode: false,
            lambda_stack: Vec::new(),
            lambda_body_spans: FxHashMap::default(),
            body_effect: Effect::pure(),
            body_reads: FxHashSet::default(),
            body_writes: FxHashSet::default(),
            inferred_params: Vec::new(),
            declared_param_types: Vec::new(),
            next_declared_param: 0,
        }
    }

    /// Pre-bind function parameters as local variables.
    /// Called before typecheck to inject parameter names+types from Signature.
    pub fn with_params(mut self, params: &[Param]) -> Self {
        for param in params {
            self.define_var(param.name, param.ty.clone());
        }
        self
    }

    /// Enable analysis mode: unknown `@context` refs produce fresh type
    /// variables instead of errors, allowing partial type inference.
    pub fn with_analysis_mode(mut self) -> Self {
        self.analysis_mode = true;
        self
    }

    /// Provide declared parameter types from Signature.
    /// In analysis mode, these are consumed in order as free params are discovered.
    pub fn with_declared_param_types(mut self, types: Vec<Ty>) -> Self {
        self.declared_param_types = types;
        self
    }

    /// Type check a template. Consumes self, returns TypeResolution<Unchecked>.
    /// Template tail type is always String (templates emit text).
    pub fn check_template(
        mut self,
        template: &Template,
    ) -> Result<TypeResolution<Unchecked>, Vec<MirError>> {
        self.check_nodes(&template.body);
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        Ok(TypeResolution::new(
            resolved,
            self.coercion_map,
            Ty::String,
            self.body_effect,
            self.body_reads,
            self.body_writes,
            self.inferred_params,
        ))
    }

    /// Type check a script. Consumes self, returns TypeResolution<Unchecked>.
    /// `expected_tail`: if provided, the script's tail expression is unified with this type.
    pub fn check_script(
        mut self,
        script: &acvus_ast::Script,
        expected_tail: Option<&Ty>,
    ) -> Result<TypeResolution<Unchecked>, Vec<MirError>> {
        for stmt in &script.stmts {
            match stmt {
                acvus_ast::Stmt::Bind { name, expr, span } => {
                    let ty = self.check_expr(false, expr);
                    self.define_var(*name, ty.clone());
                    self.record(*span, ty);
                }
                acvus_ast::Stmt::ContextStore { name, expr, span } => {
                    let ty = self.check_expr(false, expr);
                    // Writing external state is effectful.
                    self.body_writes.insert(*name);
                    self.propagate_call_effect(Effect::io());
                    // Unify with the declared context type.
                    let ctx_ty = self
                        .env.contexts
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| self.subst.fresh_param());
                    if self.subst.unify(&ty, &ctx_ty, Polarity::Invariant).is_err() {
                        self.error(
                            MirErrorKind::UnificationFailure {
                                expected: ctx_ty,
                                got: ty.clone(),
                            },
                            *span,
                        );
                    }
                    self.record(*span, ty);
                }
                acvus_ast::Stmt::Expr(expr) => {
                    self.check_expr(false, expr);
                }
            }
        }
        let tail_ty = if let Some(tail) = &script.tail {
            let ty = self.check_expr(false, tail);
            if let Some(expected) = expected_tail {
                if self.unify_covariant(&ty, expected, tail.span()).is_err() {
                    let resolved = self.subst.resolve(&ty);
                    let expected_resolved = self.subst.resolve(expected);
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: expected_resolved,
                            got: resolved,
                        },
                        tail.span(),
                    );
                }
            }
            ty
        } else {
            Ty::Unit
        };
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self
            .type_map
            .iter()
            .map(|(span, ty)| (*span, self.subst.resolve(ty)))
            .collect();
        Ok(TypeResolution::new(
            resolved,
            self.coercion_map,
            tail_ty,
            self.body_effect,
            self.body_reads,
            self.body_writes,
            self.inferred_params,
        ))
    }

    /// Unify `value_ty` with `expected_ty` in covariant position, recording
    /// any coercion needed at `span`. This is the single entry point for all
    /// covariant unification — ensures coercion detection is consistent.
    fn unify_covariant(
        &mut self,
        value_ty: &Ty,
        expected_ty: &Ty,
        span: Span,
    ) -> Result<(), (Ty, Ty)> {
        let result = self.subst.unify(value_ty, expected_ty, Polarity::Covariant);
        if result.is_ok() {
            let resolved_val = self.subst.resolve(value_ty);
            let resolved_exp = self.subst.resolve(expected_ty);
            if let Some(kind) = CastKind::between(&resolved_val, &resolved_exp) {
                self.coercion_map.push((span, kind));
            }
        }
        result
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

    fn lookup_var(&mut self, name: Astr) -> Option<Ty> {
        for (depth, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(ty) = scope.get(&name) {
                // Record as capture in ALL enclosing lambdas whose scope
                // depth is exceeded. This handles transitive captures:
                // if inner lambda captures `base` from outer scope, the
                // outer lambda also needs to capture it.
                for ls in self.lambda_stack.iter_mut() {
                    if depth < ls.depth {
                        ls.captures.push(ty.clone());
                    }
                }
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
        if let Some(ty) = self.env.contexts.get(&name) {
            return ty.clone();
        }
        if self.analysis_mode {
            return self
                .infer_vars
                .entry(name)
                .or_insert_with(|| self.subst.fresh_param())
                .clone();
        }
        self.error(
            MirErrorKind::UndefinedContext(self.interner.resolve(name).to_string()),
            span,
        );
        Ty::error()
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
        arg_spans: &[Span],
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
        for (i, (at, pt)) in arg_types.iter().zip(param_tys.iter()).enumerate() {
            let span = arg_spans.get(i).copied().unwrap_or(call_span);
            if self.unify_covariant(at, pt, span).is_err() {
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: self.subst.resolve(pt),
                        got: self.subst.resolve(at),
                    },
                    call_span,
                );
            }

            // Detect Fn return-type coercion: when arg is a lambda whose
            // return type was coerced (e.g. Deque → Iterator), register the
            // coercion on the lambda body expression's span so the lowerer
            // can insert a Cast at the return site.
            self.detect_fn_ret_coercion(at, pt, span);
        }
        true
    }

    /// If `arg_ty` and `param_ty` are both `Fn` and their resolved return
    /// types require a Cast, look up the lambda body span from the type_map
    /// and register the coercion there.
    fn detect_fn_ret_coercion(&mut self, arg_ty: &Ty, param_ty: &Ty, lambda_span: Span) {
        let resolved_arg = self.subst.resolve(arg_ty);
        let resolved_param = self.subst.resolve(param_ty);
        let (Ty::Fn { ret: arg_ret, .. }, Ty::Fn { ret: param_ret, .. }) =
            (&resolved_arg, &resolved_param)
        else {
            return;
        };
        if let Some(kind) = CastKind::between(arg_ret, param_ret) {
            // Register the coercion on the lambda BODY span (not the lambda
            // expression span). This way the lowerer's `lower_expr(body)` →
            // `maybe_cast(body.span(), val)` naturally picks it up and inserts
            // a Cast before Return.
            if let Some(&body_span) = self.lambda_body_spans.get(&lambda_span) {
                self.coercion_map.push((body_span, kind));
            }
        }
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
                let ty = self.check_expr(false, expr);
                let resolved = self.subst.resolve(&ty);
                match &resolved {
                    Ty::String | Ty::Error(_) => {}
                    Ty::Param { .. } => {
                        if self.unify_covariant(&ty, &Ty::String, *span).is_err() {
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
            let source_ty = self.check_expr(false, &mb.source);
            if matches!(&mb.arms[0].pattern, Pattern::Variant { .. }) {
                self.check_pattern(&mb.arms[0].pattern, &source_ty, mb.arms[0].tag_span);
            } else {
                let resolved_source = self.subst.resolve(&source_ty);
                self.check_pattern(&mb.arms[0].pattern, &resolved_source, mb.arms[0].tag_span);
            }
            return;
        }

        let source_ty = self.check_expr(false, &mb.source);
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
                Pattern::Variant { .. } | Pattern::Tuple { .. } | Pattern::List { .. } => {
                    source_ty.clone()
                }
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
        let source_ty = self.check_expr(false, &ib.source);
        let resolved = self.subst.resolve(&source_ty);

        let elem_ty = match &resolved {
            Ty::List(inner) | Ty::Deque(inner, _) => inner.as_ref().clone(),
            Ty::Range => Ty::Int,
            Ty::Error(_) => Ty::error(),
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
                    Ty::List(inner) | Ty::Deque(inner, _) => inner.as_ref().clone(),
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
                            let elem = self.subst.fresh_param();
                            let origin = self.subst.fresh_concrete_origin();
                            Ty::Deque(Box::new(elem), origin)
                        } else {
                            let first_ty = self.literal_ty(&elems[0]);
                            for elem in &elems[1..] {
                                let elem_ty = self.literal_ty(elem);
                                if self.unify_covariant(&elem_ty, &first_ty, *span).is_err() {
                                    self.error(
                                        MirErrorKind::HeterogeneousList {
                                            expected: self.subst.resolve(&first_ty),
                                            got: self.subst.resolve(&elem_ty),
                                        },
                                        *span,
                                    );
                                }
                            }
                            let origin = self.subst.fresh_concrete_origin();
                            Ty::Deque(Box::new(self.subst.resolve(&first_ty)), origin)
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
                        // Reading external state is effectful.
                        self.body_reads.insert(*name);
                        self.propagate_call_effect(Effect::io());
                        if !allow_non_pure && ty.purity() == Purity::Unpure && !ty.is_error() && !ty.is_param() {
                            self.error(
                                MirErrorKind::NonPureContextLoad {
                                    name: self.interner.resolve(*name).to_string(),
                                    ty: ty.clone(),
                                },
                                *span,
                            );
                            Ty::error()
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
                            Ty::error()
                        }
                    },
                    RefKind::Value => match self.lookup_var(*name) {
                        Some(ty) => ty,
                        None => {
                            if self.analysis_mode {
                                // In analysis mode, undefined values are implicit
                                // function parameters. Use declared type from Signature
                                // if available, otherwise fresh Param.
                                let ty = if self.next_declared_param < self.declared_param_types.len() {
                                    let t = self.declared_param_types[self.next_declared_param].clone();
                                    self.next_declared_param += 1;
                                    t
                                } else {
                                    self.subst.fresh_param()
                                };
                                self.define_var(*name, ty.clone());
                                self.inferred_params.push((*name, ty.clone()));
                                ty
                            } else {
                                self.error(
                                    MirErrorKind::UndefinedVariable(
                                        self.interner.resolve(*name).to_string(),
                                    ),
                                    *span,
                                );
                                Ty::error()
                            }
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
                let lt = self.check_expr(false, left);
                let rt = self.check_expr(false, right);
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
                        | BinOp::Shr => Ty::error(),
                        _ => Ty::Bool,
                    };
                    return self.record_ret(*span, ty);
                }

                let ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if self.subst.unify(&lt, &rt, Polarity::Invariant).is_err() {
                            self.binop_error(
                                op_str(*op),
                                self.subst.resolve(&lt),
                                self.subst.resolve(&rt),
                                *span,
                            );
                            return self.record_ret(*span, Ty::error());
                        }
                        let rl = self.subst.resolve(&lt);
                        match &rl {
                            Ty::Int | Ty::Float | Ty::Param { .. } => rl,
                            Ty::String if *op == BinOp::Add => Ty::String,
                            _ => {
                                self.binop_error(op_str(*op), rl, self.subst.resolve(&rt), *span);
                                Ty::error()
                            }
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.subst.unify(&lt, &rt, Polarity::Invariant).is_err() {
                            self.binop_error(op_str(*op), lt, rt, *span);
                        }
                        Ty::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        let lok = self.unify_covariant(&lt, &Ty::Bool, *span).is_ok();
                        let rok = self.unify_covariant(&rt, &Ty::Bool, *span).is_ok();
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
                        let lok = self.unify_covariant(&lt, &Ty::Int, *span).is_ok();
                        let rok = self.unify_covariant(&rt, &Ty::Int, *span).is_ok();
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
                        let ok = self.subst.unify(&lt, &rt, Polarity::Invariant).is_ok()
                            && matches!(self.subst.resolve(&lt), Ty::Int | Ty::Float | Ty::Param { .. });
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
                let ot = self.check_expr(false, operand);
                let ot = self.subst.resolve(&ot);

                // Early guard: if operand is Error, suppress cascading errors.
                if ot.is_error() {
                    let ty = match op {
                        acvus_ast::UnaryOp::Neg => Ty::error(),
                        acvus_ast::UnaryOp::Not => Ty::Bool,
                    };
                    return self.record_ret(*span, ty);
                }

                let ty = match op {
                    acvus_ast::UnaryOp::Neg => match &ot {
                        Ty::Int => Ty::Int,
                        Ty::Float => Ty::Float,
                        Ty::Param { .. } => ot.clone(),
                        _ => {
                            self.binop_error("-", ot, Ty::error(), *span);
                            Ty::error()
                        }
                    },
                    acvus_ast::UnaryOp::Not => {
                        match &ot {
                            Ty::Bool => {}
                            Ty::Param { .. } => {
                                let _ = self.unify_covariant(&ot, &Ty::Bool, *span);
                            }
                            _ => self.binop_error("!", ot, Ty::error(), *span),
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
                let ot_raw = self.check_expr(false, object);
                let ot = self.subst.resolve(&ot_raw);
                let field_key = *field;
                let field_str = || self.interner.resolve(*field).to_string();
                let ty = match &ot {
                    Ty::Error(_) => Ty::error(),
                    Ty::Object(fields) if fields.contains_key(&field_key) => {
                        fields[&field_key].clone()
                    }
                    Ty::Object(fields) => {
                        let Some(leaf_var) = self.subst.find_leaf_param(&ot_raw) else {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: ot.clone(),
                                    field: field_str(),
                                },
                                *span,
                            );
                            return Ty::error();
                        };
                        let fresh = self.subst.fresh_param();
                        let mut new_fields = fields.clone();
                        new_fields.insert(field_key, fresh.clone());
                        self.subst.rebind(leaf_var, Ty::Object(new_fields));
                        fresh
                    }
                    Ty::Param { .. } => {
                        let fresh = self.subst.fresh_param();
                        let partial_obj =
                            Ty::Object(FxHashMap::from_iter([(field_key, fresh.clone())]));
                        if self
                            .subst
                            .unify(&ot_raw, &partial_obj, Polarity::Invariant)
                            .is_err()
                        {
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
                        Ty::error()
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
                        let lt = self.check_expr(false, left);
                        let rt = self.check_expr(false, right);
                        self.check_callable(&rt, &[], &Some(lt), Some(left.span()), *span)
                    }
                };
                self.record_ret(*span, ty)
            }

            Expr::Lambda { params, body, span } => {
                self.push_scope();
                let mut param_types = Vec::new();
                for p in params {
                    let pt = self.subst.fresh_param();
                    self.define_var(p.name, pt.clone());
                    self.record(p.span, pt.clone());
                    param_types.push(Param::new(p.name, pt));
                }

                // Push lambda scope for capture tracking.
                self.lambda_stack.push(LambdaScope {
                    depth: self.scopes.len() - 1,
                    captures: Vec::new(),
                    effect: Effect::pure(),
                });

                let ret = self.check_expr(false, body);

                // Pop this lambda's scope.
                let ls = self.lambda_stack.pop().unwrap();
                let effect = ls.effect;
                let capture_types: Vec<Ty> = ls
                    .captures
                    .into_iter()
                    .map(|t| self.subst.resolve(&t))
                    .collect();

                // Record body span so detect_fn_ret_coercion can register
                // return-site coercions on the correct expression.
                self.lambda_body_spans.insert(*span, body.span());

                self.pop_scope();
                let ty = Ty::Fn {
                    params: param_types,
                    ret: Box::new(ret),
                    captures: capture_types,
                    effect,
                };
                self.record_ret(*span, ty)
            }

            Expr::Paren { inner, span } => {
                let ty = self.check_expr(false, inner);
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
                    let elem = self.subst.fresh_param();
                    let origin = self.subst.fresh_concrete_origin();
                    let ty = Ty::Deque(Box::new(elem), origin);
                    return self.record_ret(*span, ty);
                }

                let elem_ty = match all_elems.first() {
                    Some(first) => self.check_expr(false, first),
                    None => self.subst.fresh_param(), // Only `..` with no elements: fresh var.
                };

                for elem in all_elems.iter().skip(1) {
                    let et = self.check_expr(false, elem);
                    if self.unify_covariant(&et, &elem_ty, *span).is_err() {
                        self.error(
                            MirErrorKind::HeterogeneousList {
                                expected: self.subst.resolve(&elem_ty),
                                got: self.subst.resolve(&et),
                            },
                            *span,
                        );
                    }
                }

                let origin = self.subst.fresh_concrete_origin();
                let ty = Ty::Deque(Box::new(self.subst.resolve(&elem_ty)), origin);
                self.record_ret(*span, ty)
            }

            Expr::Object { fields, span } => {
                let mut field_types = FxHashMap::default();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(false, value);
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
                let st = self.check_expr(false, start);
                let et = self.check_expr(false, end);
                let st = self.subst.resolve(&st);
                let et = self.subst.resolve(&et);
                if !matches!(&st, Ty::Int | Ty::Error(_)) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: st }, *span);
                }
                if !matches!(&et, Ty::Int | Ty::Error(_)) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: et }, *span);
                }
                self.record_ret(*span, Ty::Range)
            }

            Expr::Tuple { elements, span } => {
                let elem_types: Vec<Ty> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.check_expr(false, e),
                        TupleElem::Wildcard(_) => self.subst.fresh_param(),
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
                    self.check_expr(false, e);
                }
                let ty = self.check_expr(false, last);
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
                                        expected: Ty::error(),
                                        got: Ty::Unit,
                                    },
                                    *span,
                                );
                                return Ty::error();
                            };
                            let inner_ty = self.check_expr(false, inner_expr);
                            if self
                                .unify_covariant(&type_params[*idx], &inner_ty, *span)
                                .is_err()
                            {
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
                    return Ty::error();
                };

                let payload_ty = match payload {
                    Some(expr) => {
                        let ty = self.check_expr(false, expr);
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
                            let ty = self.check_expr(false, expr);
                            self.define_var(*name, ty.clone());
                            self.record(*span, ty);
                        }
                        acvus_ast::Stmt::ContextStore { name, expr, span } => {
                            let ty = self.check_expr(false, expr);
                            self.define_var(*name, ty.clone());
                            self.record(*span, ty);
                        }
                        acvus_ast::Stmt::Expr(expr) => {
                            self.check_expr(false, expr);
                        }
                    }
                }
                let ty = self.check_expr(false, tail);
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
        let pipe_ty = pipe_left.map(|e| self.check_expr(false, e));

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
            let pipe_left_span = pipe_left.map(|e| e.span());
            return self.check_callable(&resolved, args, &pipe_ty, pipe_left_span, call_span);
        };

        // Check named functions (builtins, externs, user-defined).
        let name_str = self.interner.resolve(*name);
        if let Some(fn_sig) = self.env.functions.get(name) {
            let arg_types: Vec<Ty> = pipe_ty
                .iter()
                .cloned()
                .chain(args.iter().map(|a| self.check_expr(false, a)))
                .collect();
            let arg_spans: Vec<Span> = pipe_left
                .iter()
                .map(|e| e.span())
                .chain(args.iter().map(|a| a.span()))
                .collect();

            let fn_ty = self.subst.instantiate(fn_sig);
            match &fn_ty {
                Ty::Fn { params: param_tys, ret, effect, .. } => {
                    self.propagate_call_effect(effect.clone());
                    let tys: Vec<Ty> = param_tys.iter().map(|p| p.ty.clone()).collect();
                    if !self.check_args(name_str, &arg_types, &arg_spans, &tys, call_span) {
                        return Ty::error();
                    }
                    return self.subst.resolve(ret);
                }
                _ => {
                    self.error(MirErrorKind::UndefinedFunction(
                        name_str.to_string(),
                    ), call_span);
                    return Ty::error();
                }
            }
        }

        // Check local variable with function type.
        if let Some(var_ty) = self.lookup_var(*name) {
            let resolved = self.subst.resolve(&var_ty);
            let pipe_left_span = pipe_left.map(|e| e.span());
            return self.check_callable(&resolved, args, &pipe_ty, pipe_left_span, call_span);
        }

        self.error(
            MirErrorKind::UndefinedFunction(self.interner.resolve(*name).to_string()),
            call_span,
        );
        return Ty::error();
    }

    /// Propagate a callee's effect to the enclosing scope.
    /// If inside a lambda, propagates to the lambda scope.
    /// Otherwise, propagates to the top-level body_effect.
    fn propagate_call_effect(&mut self, effect: Effect) {
        let resolved = self.subst.resolve_effect(&effect);
        if resolved.is_effectful() {
            if let Some(ls) = self.lambda_stack.last_mut() {
                ls.effect = Effect::io();
            } else {
                self.body_effect = Effect::io();
            }
        }
    }

    fn check_callable(
        &mut self,
        func_ty: &Ty,
        args: &[Expr],
        pipe_ty: &Option<Ty>,
        pipe_left_span: Option<Span>,
        call_span: Span,
    ) -> Ty {
        // Early exit for non-callable types.
        match func_ty {
            Ty::Fn { .. } | Ty::Param { .. } => {}
            Ty::Error(_) => {
                for a in args {
                    self.check_expr(false, a);
                }
                return Ty::error();
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction("<not callable>".to_string()),
                    call_span,
                );
                return Ty::error();
            }
        }

        let arg_types: Vec<Ty> = pipe_ty
            .iter()
            .cloned()
            .chain(args.iter().map(|a| self.check_expr(false, a)))
            .collect();
        let arg_spans: Vec<Span> = pipe_left_span
            .iter()
            .copied()
            .chain(args.iter().map(|a| a.span()))
            .collect();

        match func_ty {
            Ty::Fn {
                params,
                ret,
                effect,
                ..
            } => {
                // Propagate effect to enclosing scope (lambda or top-level body).
                self.propagate_call_effect(effect.clone());
                let tys: Vec<Ty> = params.iter().map(|p| p.ty.clone()).collect();
                if !self.check_args("<closure>", &arg_types, &arg_spans, &tys, call_span) {
                    return Ty::error();
                }
                self.subst.resolve(ret)
            }
            Ty::Param { .. } => {
                let ret = self.subst.fresh_param();
                let dummy = self.interner.intern("_");
                let fn_ty = Ty::Fn {
                    params: arg_types.into_iter().map(|ty| Param::new(dummy, ty)).collect(),
                    ret: Box::new(ret.clone()),
                    captures: vec![],
                    effect: Effect::pure(),
                };
                if self.unify_covariant(func_ty, &fn_ty, call_span).is_err() {
                    self.error(
                        MirErrorKind::UndefinedFunction("<expr>".to_string()),
                        call_span,
                    );
                    return Ty::error();
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
                    // Context write allowed — mutability will be enforced later.
                    let ctx_ty = self
                        .env.contexts
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| self.subst.fresh_param());
                    if self
                        .subst
                        .unify(&source_resolved, &ctx_ty, Polarity::Invariant)
                        .is_err()
                    {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: ctx_ty,
                                source_ty: source_resolved,
                            },
                            span,
                        );
                    }
                }
                RefKind::Variable => {
                    let Some(existing_ty) = self.variable_types.get(name).cloned() else {
                        self.variable_types.insert(*name, source_resolved);
                        return;
                    };
                    if self
                        .subst
                        .unify(&source_resolved, &existing_ty, Polarity::Invariant)
                        .is_err()
                    {
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
                if self
                    .unify_covariant(&source_resolved, &pat_ty, span)
                    .is_err()
                {
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
                    Ty::List(ref inner) | Ty::Deque(ref inner, _) => (**inner).clone(),
                    _ => {
                        let var = self.subst.fresh_param();
                        let origin = self.subst.fresh_concrete_origin();
                        let list_ty = Ty::Deque(Box::new(var.clone()), origin);
                        if self.unify_covariant(source_ty, &list_ty, span).is_err() {
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
                        .map(|f| (f.key, self.subst.fresh_param()))
                        .collect();
                    let obj_ty = Ty::Object(field_vars.clone());
                    if self.unify_covariant(source_ty, &obj_ty, span).is_err() {
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
                if self
                    .unify_covariant(&source_resolved, &Ty::Int, span)
                    .is_err()
                {
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
                    Ty::Tuple(ref existing) if existing.len() == elements.len() => existing.clone(),
                    _ => {
                        let vars: Vec<Ty> =
                            elements.iter().map(|_| self.subst.fresh_param()).collect();
                        let tuple_ty = Ty::Tuple(vars.clone());
                        if self.unify_covariant(source_ty, &tuple_ty, span).is_err() {
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
                    let enum_ty = Ty::Option(Box::new(self.subst.resolve(&type_params[0])));
                    if self
                        .unify_covariant(&source_resolved, &enum_ty, span)
                        .is_err()
                    {
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
                    Some(Box::new(self.subst.fresh_param()))
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
                if self.unify_covariant(source_ty, &enum_ty, span).is_err() {
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
                        .unwrap_or(Ty::error());
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

        let type_params = vec![self.subst.fresh_param()];
        Some((option_name, type_params, payload))
    }

    fn literal_ty(&mut self, lit: &Literal) -> Ty {
        match lit {
            Literal::Int(_) => Ty::Int,
            Literal::Float(_) => Ty::Float,
            Literal::String(_) => Ty::String,
            Literal::Bool(_) => Ty::Bool,
            Literal::Byte(_) => Ty::Byte,
            Literal::List(elems) => {
                let origin = self.subst.fresh_concrete_origin();
                match elems.first() {
                    Some(first) => Ty::Deque(Box::new(self.literal_ty(first)), origin),
                    None => Ty::Deque(Box::new(Ty::error()), origin),
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
                let ty = self.literal_ty(value);
                self.error(MirErrorKind::RangeBoundsNotInt { actual: ty }, span);
            }
            _ => {
                self.error(
                    MirErrorKind::RangeBoundsNotInt {
                        actual: Ty::error(),
                    },
                    span,
                );
            }
        }
    }
}

pub(crate) fn contains_var(ty: &Ty) -> bool {
    match ty {
        Ty::Param { .. } => true,
        Ty::Error(_) => false,
        Ty::Int | Ty::Float | Ty::String | Ty::Bool | Ty::Unit | Ty::Range | Ty::Byte => false,
        Ty::List(inner) | Ty::Deque(inner, _) | Ty::Option(inner) => contains_var(inner),
        Ty::Iterator(inner, _) | Ty::Sequence(inner, _, _) | Ty::Handle(inner, _) => contains_var(inner),
        Ty::Object(fields) => fields.values().any(contains_var),
        Ty::Tuple(elems) => elems.iter().any(contains_var),
        Ty::Fn { params, ret, .. } => params.iter().any(|p| contains_var(&p.ty)) || contains_var(ret),
        Ty::Enum { variants, .. } => variants
            .values()
            .any(|p| p.as_ref().map_or(false, |ty| contains_var(ty))),
        Ty::Opaque(_) => false,
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

    /// Test helper: create a `Param` with name "_".
    fn p(i: &Interner, ty: Ty) -> Param {
        Param::new(i.intern("_"), ty)
    }

    fn check(source: &str) -> Result<TypeMap, Vec<MirError>> {
        let interner = Interner::new();
        check_with_interner(source, &FxHashMap::default(), &interner)
    }

    fn check_with_interner(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> Result<TypeMap, Vec<MirError>> {
        let template = acvus_ast::parse(interner, source)
            .expect("parse failed");
        let mut subst = TySubst::new();
        let env = crate::ty::TypeEnv {
            contexts: context.clone(),
            functions: crate::builtins::builtin_fn_types(interner),
        };
        let checker = TypeChecker::new(interner, &env, &mut subst);
        let resolution = checker.check_template(&template)?;
        Ok(resolution.type_map)
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
                params: vec![p(&i, Ty::Int)],
                ret: Box::new(Ty::String),
                effect: Effect::pure(),
                captures: vec![],
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
        let src =
            "{{ x = @items | filter(x -> x != 0) | collect }}{{ x | len | to_string }}{{_}}{{/}}";
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
            (
                interner.intern("my_fn"),
                Ty::Fn {
                    params: vec![p(interner, Ty::String)],
                    ret: Box::new(Ty::String),
                    effect: Effect::pure(),
                    captures: vec![],
                },
            ),
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
    fn extern_fn_bare_ref_allowed() {
        // f = @my_fn — Fn is Lazy tier, allowed in non-call position.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = "{{ f = @my_fn }}{{_}}{{/}}";
        let result = check_with_interner(src, &ctx, &i);
        assert!(
            result.is_ok(),
            "bare reference to extern fn should be allowed (Lazy tier): {result:?}"
        );
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
        let ctx = FxHashMap::from_iter([(
            i.intern("my_fn"),
            Ty::Fn {
                params: vec![p(&i, Ty::String), p(&i, Ty::Int)],
                ret: Box::new(Ty::String),
                effect: Effect::pure(),
                captures: vec![],
            },
        )]);
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

    // ── 3-tier purity: Lazy context load tests ──

    #[test]
    fn lazy_list_context_load_ok() {
        // @items : List<Int> — Lazy tier, allowed in non-call position.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ x = @items }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_iterator_context_load_ok() {
        // @it : Iterator<Int> — Lazy tier, allowed in non-call position.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("it"),
            Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
        )]);
        let src = "{{ x = @it }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_sequence_context_load_ok() {
        // @seq : Sequence<Int, O> — Lazy tier, allowed.
        let i = Interner::new();
        let mut subst = TySubst::new();
        let o = subst.fresh_concrete_origin();
        let ctx = FxHashMap::from_iter([(
            i.intern("seq"),
            Ty::Sequence(Box::new(Ty::Int), o, Effect::pure()),
        )]);
        let src = "{{ x = @seq }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_option_context_load_ok() {
        // @opt : Option<Int> — Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::Int)))]);
        let src = "{{ x = @opt }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_tuple_context_load_ok() {
        // @pair : (Int, String) — Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("pair"), Ty::Tuple(vec![Ty::Int, Ty::String]))]);
        let src = "{{ x = @pair }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_object_context_load_ok() {
        // @obj : {x: Int} — Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("obj"),
            Ty::Object(FxHashMap::from_iter([(i.intern("x"), Ty::Int)])),
        )]);
        let src = "{{ x = @obj }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_fn_context_load_and_call_ok() {
        // f = @callback; f(42) — store Fn in variable, then call.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("callback"),
            Ty::Fn {
                params: vec![p(&i, Ty::Int)],
                ret: Box::new(Ty::String),
                effect: Effect::pure(),
                captures: vec![],
            },
        )]);
        let src = "{{ f = @callback }}{{ f(42) }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lazy_list_of_fn_context_load_ok() {
        // @fns : List<Fn(Int)->Int> — Lazy tier (List is Lazy), allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("fns"),
            Ty::List(Box::new(Ty::Fn {
                params: vec![p(&i, Ty::Int)],
                ret: Box::new(Ty::Int),
                effect: Effect::pure(),
                captures: vec![],
            })),
        )]);
        let src = "{{ x = @fns }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    // ── Unpure context load tests (Opaque — must be rejected) ──

    #[test]
    fn unpure_opaque_context_load_rejected() {
        // @conn : Opaque("Connection") — Unpure tier, rejected in non-call position.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("conn"), Ty::Opaque("Connection".into()))]);
        let src = "{{ x = @conn }}{{_}}{{/}}";
        let result = check_with_interner(src, &ctx, &i);
        assert!(result.is_err(), "Opaque context load should be rejected");
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(&e.kind, MirErrorKind::NonPureContextLoad { .. }))
        );
    }

    #[test]
    fn unpure_opaque_in_argument_also_rejected() {
        // @handler(@conn) — @conn is Opaque, rejected even in argument position.
        // Arguments are checked with allow_non_pure=false.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([
            (i.intern("conn"), Ty::Opaque("Connection".into())),
            (
                i.intern("handler"),
                Ty::Fn {
                    params: vec![p(&i, Ty::Opaque("Connection".into()))],
                    ret: Box::new(Ty::String),
                    effect: Effect::pure(),
                    captures: vec![],
                },
            ),
        ]);
        let src = "{{ @handler(@conn) }}";
        let result = check_with_interner(src, &ctx, &i);
        assert!(result.is_err(), "Opaque in argument should be rejected");
    }

    // ── Pure context load tests (scalars — always ok) ──

    #[test]
    fn pure_int_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("count"), Ty::Int)]);
        let src = "{{ x = @count }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn pure_string_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("msg"), Ty::String)]);
        let src = "{{ x = @msg }}{{_}}{{/}}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    // ── Lambda captures type tracking tests ──

    #[test]
    fn lambda_captures_outer_variable() {
        // Lambda capturing an outer value should have captures tracked.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([
            (i.intern("items"), Ty::List(Box::new(Ty::Int))),
            (i.intern("threshold"), Ty::Int),
        ]);
        // threshold is captured by the lambda
        let src = "{{ @items | filter(x -> x > @threshold) | collect | len | to_string }}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn lambda_no_capture_local_params() {
        // Lambda params are local, not captures.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ @items | map(x -> x + 1) | collect | len | to_string }}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }

    #[test]
    fn nested_lambda_captures() {
        // Nested lambdas: inner captures from outer lambda's scope.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([
            (i.intern("items"), Ty::List(Box::new(Ty::Int))),
            (i.intern("factor"), Ty::Int),
        ]);
        // Inner lambda captures @factor from context (not from outer lambda scope).
        let src = "{{ @items | map(x -> x * @factor) | collect | len | to_string }}";
        assert!(check_with_interner(src, &ctx, &i).is_ok());
    }
}
