use acvus_ast::{
    AstId, BinOp, Expr, IterBlock, Literal, MatchBlock, Node, ObjectExprField, ObjectPatternField,
    Pattern, RefKind, Span, Template, TupleElem, TuplePatternElem,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::error::{MirError, MirErrorKind};
use crate::graph::QualifiedRef;
use crate::ir::CastKind;
use crate::ty::{
    Effect, EffectTarget, EffectTerm, InferEffect, InferTy, Materiality,
    Param, ParamTerm, Polarity, Solver, Ty, TyTerm, TypeEnv, TypeRegistry,
    lift_ty,
};
use crate::variant::VariantPayload;

/// Maps each AST node id to its inferred type.
pub type TypeMap = FxHashMap<AstId, Ty>;

/// Maps expression AST ids to the coercion needed at that point.
/// Produced by the type checker, consumed by the lowerer.
pub type CoercionMap = Vec<(AstId, CastKind)>;

// ── TypeResolution: boundary between TypeChecker and Lowerer ──────────

/// Result of type checking a single script or template.
///
/// Contains concrete `Ty` (frozen from `InferTy` at `check_template`/`check_script`).
/// `Ty = TyTerm<Concrete>` cannot contain unresolved variables by construction
/// (`TyVar = Infallible`), so completeness is guaranteed structurally.
#[derive(Debug, Clone)]
pub struct TypeResolution {
    pub type_map: TypeMap,
    pub coercion_map: CoercionMap,
    pub tail_ty: Ty,
    /// Effect of the function body.
    /// Contains reads/writes as QualifiedRef sets.
    pub body_effect: Effect,
    /// Extern parameters ($name) discovered during typecheck.
    pub extern_params: Vec<(Astr, Ty)>,
}

impl TypeResolution {
    fn new(
        type_map: TypeMap,
        coercion_map: CoercionMap,
        tail_ty: Ty,
        body_effect: Effect,
        extern_params: Vec<(Astr, Ty)>,
    ) -> Self {
        Self {
            type_map,
            coercion_map,
            tail_ty,
            body_effect,
            extern_params,
        }
    }
}

/// Check that a function's body effect satisfies the constraint.
/// `allowed` is the upper bound: body must not exceed it.
pub fn check_effect_constraint(
    body_effect: &Effect,
    allowed: &crate::ty::EffectConstraint,
) -> Result<(), MirError> {
    use crate::ty::EffectCap;

    let actual = match body_effect {
        Effect::Resolved(set) => set,
        // Concrete phase: EffectVar = Infallible — uninhabitable by construction.
        Effect::Var(v) => match *v {},
    };

    let mut violations = Vec::new();

    match &allowed.reads {
        EffectCap::Any => {} // all reads allowed
        EffectCap::Only(allowed_reads) => {
            if !allowed_reads.is_superset(&actual.reads) {
                let forbidden: Vec<_> = actual.reads.difference(allowed_reads).collect();
                violations.push(format!("reads from {:?}", forbidden));
            }
        }
    }
    match &allowed.writes {
        EffectCap::Any => {} // all writes allowed
        EffectCap::Only(allowed_writes) => {
            if !allowed_writes.is_superset(&actual.writes) {
                let forbidden: Vec<_> = actual.writes.difference(allowed_writes).collect();
                violations.push(format!("writes to {:?}", forbidden));
            }
        }
    }
    if violations.is_empty() {
        Ok(())
    } else {
        Err(MirError {
            kind: MirErrorKind::EffectViolation {
                detail: violations.join(", "),
            },
            span: Span::ZERO,
        })
    }
}

struct LambdaScope {
    depth: usize,
    captures: Vec<InferTy>,
    effect: InferEffect,
}

/// State only active in analysis mode (partial inference for unknown contexts/params).
struct AnalysisState {
    /// Cached fresh Vars for unknown context entries.
    infer_vars: FxHashMap<Astr, InferTy>,
    /// Declared parameter types from Signature, consumed in order as $params are discovered.
    declared_param_types: Vec<Ty>,
    /// Next index into declared_param_types.
    next_declared_param: usize,
}

pub struct TypeChecker<'a, 's> {
    /// Interner for string interning.
    interner: &'a Interner,
    /// Unified type environment: contexts + functions.
    env: &'a TypeEnv,
    /// Type registry for coercion rules.
    registry: &'a TypeRegistry,
    /// Namespace this function belongs to.
    namespace: Option<Astr>,
    /// Stack of scopes: each scope maps variable names to types.
    scopes: Vec<FxHashMap<Astr, InferTy>>,
    /// Extern parameter types (`$name`, inferred at first use).
    /// SmallVec to preserve insertion order — iteration order must match Signature order.
    param_types: smallvec::SmallVec<[(Astr, InferTy); 4]>,
    /// Solver state (borrowed — may be shared across compilations).
    solver: &'s mut Solver,
    /// Accumulated type map (internal, uses InferTy during inference).
    type_map: FxHashMap<AstId, InferTy>,
    /// Accumulated coercion records (span → CastKind).
    coercion_map: CoercionMap,
    /// Accumulated errors.
    errors: Vec<MirError>,
    /// Analysis mode state. `None` = normal mode, `Some` = partial inference enabled.
    analysis: Option<AnalysisState>,
    /// Stack of active lambda scopes. Each entry is (scope_depth, captures, effect).
    /// Nested lambdas push onto this stack; lookups record captures in ALL
    /// enclosing lambdas whose scope depth is exceeded.
    lambda_stack: Vec<LambdaScope>,
    /// Maps lambda expression AstId → body expression AstId.
    /// Used by `detect_fn_ret_coercion` to register coercions on the
    /// correct id (body, not lambda) so the lowerer's `maybe_cast`
    /// naturally inserts a Cast at the lambda return site.
    lambda_body_ids: FxHashMap<AstId, AstId>,
    /// Top-level body effect. Tracks reads/writes as QualifiedRef.
    /// Starts as pure (empty EffectSet); updated as context access and effectful calls occur.
    body_effect: InferEffect,
}

impl<'a, 's> TypeChecker<'a, 's> {
    pub fn new(interner: &'a Interner, env: &'a TypeEnv, registry: &'a TypeRegistry, solver: &'s mut Solver) -> Self {
        Self {
            interner,
            scopes: vec![FxHashMap::default()],
            env,
            registry,
            namespace: None,
            param_types: smallvec::smallvec![],
            solver,
            type_map: FxHashMap::default(),
            coercion_map: CoercionMap::default(),
            errors: Vec::new(),
            analysis: None,
            lambda_stack: Vec::new(),
            lambda_body_ids: FxHashMap::default(),
            body_effect: EffectTerm::Resolved(Default::default()),
        }
    }

    /// Pre-bind function parameters as local variables.
    /// Called before typecheck to inject parameter names+types from Signature.
    pub fn with_params(mut self, params: &[Param]) -> Self {
        for param in params {
            self.param_types.push((param.name, lift_ty(&param.ty)));
        }
        self
    }

    /// Set the namespace for context lookups.
    pub fn with_namespace(mut self, namespace: Option<Astr>) -> Self {
        self.namespace = namespace;
        self
    }

    /// Enable analysis mode: unknown `@context` refs produce fresh type
    /// variables instead of errors, allowing partial type inference.
    pub fn with_analysis_mode(mut self) -> Self {
        self.analysis = Some(AnalysisState {
            infer_vars: FxHashMap::default(),
            declared_param_types: Vec::new(),
            next_declared_param: 0,
        });
        self
    }

    /// Provide declared parameter types from Signature.
    /// In analysis mode, these are consumed in order as free params are discovered.
    pub fn with_declared_param_types(mut self, types: Vec<Ty>) -> Self {
        if let Some(ref mut state) = self.analysis {
            state.declared_param_types = types;
        }
        self
    }

    /// Freeze an InferTy to concrete Ty, falling back to Ty::error() on failure.
    /// Used for error reporting where we need concrete types.
    fn freeze_or_error(&self, ty: &InferTy) -> Ty {
        self.solver.freeze_ty(ty).unwrap_or_else(|_| Ty::error())
    }

    /// Freeze the internal InferTy type_map to a concrete TypeMap.
    fn freeze_type_map(&self) -> TypeMap {
        self.type_map
            .iter()
            .map(|(id, ty)| {
                let resolved = self.solver.resolve_ty(ty);
                (*id, self.solver.freeze_ty(&resolved).unwrap_or_else(|_| Ty::error()))
            })
            .collect()
    }

    /// Freeze an InferEffect to concrete Effect, defaulting to pure on failure.
    fn freeze_effect_or_pure(&self, e: &InferEffect) -> Effect {
        self.solver.freeze_effect(e).unwrap_or_else(|_| Effect::pure())
    }

    /// Construct an InferTy error token.
    fn infer_error() -> InferTy {
        TyTerm::Error(crate::ty::ErrorToken::new())
    }

    /// Check if an InferTy is an error.
    fn is_error(ty: &InferTy) -> bool {
        matches!(ty, TyTerm::Error(_))
    }

    /// Check if an InferTy is an unresolved variable.
    fn is_var(ty: &InferTy) -> bool {
        matches!(ty, TyTerm::Var(_))
    }

    /// Type check a template. Consumes self, returns TypeResolution.
    /// Template tail type is always String (templates emit text).
    pub fn check_template(
        mut self,
        template: &Template,
    ) -> Result<TypeResolution, Vec<MirError>> {
        self.check_nodes(&template.body);
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self.freeze_type_map();
        let extern_params: Vec<(Astr, Ty)> = self
            .param_types
            .iter()
            .map(|(name, ty)| {
                let resolved = self.solver.resolve_ty(ty);
                (*name, self.freeze_or_error(&resolved))
            })
            .collect();
        let body_effect = self.freeze_effect_or_pure(&self.body_effect.clone());
        Ok(TypeResolution::new(
            resolved,
            self.coercion_map,
            Ty::String,
            body_effect,
            extern_params,
        ))
    }

    /// Type check a script. Consumes self, returns TypeResolution.
    /// `expected_tail`: if provided, the script's tail expression is unified with this type.
    pub fn check_script(
        mut self,
        script: &acvus_ast::Script,
        expected_tail: Option<&Ty>,
    ) -> Result<TypeResolution, Vec<MirError>> {
        for stmt in &script.stmts {
            self.check_stmt(stmt);
        }
        let tail_ty = if let Some(tail) = &script.tail {
            let ty = self.check_expr(false, tail);
            if let Some(expected) = expected_tail {
                let expected_infer = lift_ty(expected);
                if self
                    .unify_covariant(&ty, &expected_infer, Some(tail.id()))
                    .is_err()
                {
                    let resolved = self.solver.resolve_ty(&ty);
                    let expected_resolved = self.solver.resolve_ty(&expected_infer);
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.freeze_or_error(&expected_resolved),
                            got: self.freeze_or_error(&resolved),
                        },
                        tail.span(),
                    );
                }
            }
            ty
        } else {
            TyTerm::Unit
        };
        if !self.errors.is_empty() {
            return Err(self.errors);
        }
        let resolved: TypeMap = self.freeze_type_map();
        let extern_params: Vec<(Astr, Ty)> = self
            .param_types
            .iter()
            .map(|(name, ty)| {
                let resolved = self.solver.resolve_ty(ty);
                (*name, self.freeze_or_error(&resolved))
            })
            .collect();
        let frozen_tail = self.freeze_or_error(&self.solver.resolve_ty(&tail_ty));
        let body_effect = self.freeze_effect_or_pure(&self.body_effect.clone());
        Ok(TypeResolution::new(
            resolved,
            self.coercion_map,
            frozen_tail,
            body_effect,
            extern_params,
        ))
    }

    /// Unify `value_ty` with `expected_ty` in covariant position, recording
    /// any coercion needed at `span`. This is the single entry point for all
    /// covariant unification — ensures coercion detection is consistent.
    fn unify_covariant(
        &mut self,
        value_ty: &InferTy,
        expected_ty: &InferTy,
        coerce_id: Option<AstId>,
    ) -> Result<(), (InferTy, InferTy)> {
        self.solver.last_extern_cast = None;
        let result = self.solver.unify_ty(value_ty, expected_ty, Polarity::Covariant, self.registry);
        if result.is_ok()
            && let Some(id) = coerce_id
        {
            // ExternCast takes priority — set by try_coerce_infer inside unify_ty.
            if let Some(fn_ref) = self.solver.last_extern_cast.take() {
                let resolved_exp = self.solver.resolve_ty(expected_ty);
                let ret_ty = self.solver.freeze_ty(&resolved_exp).unwrap_or_else(|_| Ty::error());
                self.coercion_map.push((id, CastKind::Extern { fn_ref, ret_ty }));
            } else {
                let resolved_val = self.solver.resolve_ty(value_ty);
                let resolved_exp = self.solver.resolve_ty(expected_ty);
                if let (Ok(frozen_val), Ok(frozen_exp)) =
                    (self.solver.freeze_ty(&resolved_val), self.solver.freeze_ty(&resolved_exp))
                {
                    if let Some(kind) = CastKind::between(&frozen_val, &frozen_exp) {
                        self.coercion_map.push((id, kind));
                    }
                }
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

    fn define_var(&mut self, name: Astr, ty: InferTy) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup_var(&mut self, name: Astr) -> Option<InferTy> {
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

    /// Walk a field path on a type, resolving each step.
    /// Returns the leaf type, or an error type if any step fails.
    fn resolve_field_path(&self, base: &InferTy, path: &[Astr]) -> InferTy {
        let mut current = base.clone();
        for field in path {
            let resolved = self.solver.resolve_ty(&current);
            match resolved {
                TyTerm::Object(fields) => {
                    if let Some(field_ty) = fields.get(field) {
                        current = field_ty.clone();
                    } else {
                        return Self::infer_error();
                    }
                }
                _ => return Self::infer_error(),
            }
        }
        current
    }

    fn error(&mut self, kind: MirErrorKind, span: Span) {
        self.errors.push(MirError { kind, span });
    }

    fn record(&mut self, id: AstId, ty: InferTy) {
        self.type_map.insert(id, ty);
    }

    /// Record the type at an AST id and return it.
    fn record_ret(&mut self, id: AstId, ty: InferTy) -> InferTy {
        self.record(id, ty.clone());
        ty
    }

    /// Record a context read in the body effect.
    fn record_context_read(&mut self, qref: QualifiedRef) {
        if let EffectTerm::Resolved(ref mut set) = self.body_effect {
            set.reads.insert(EffectTarget::Context(qref));
        }
    }

    /// Record a context write in the body effect.
    fn record_context_write(&mut self, qref: QualifiedRef) {
        if let EffectTerm::Resolved(ref mut set) = self.body_effect {
            set.writes.insert(EffectTarget::Context(qref));
        }
    }

    fn resolve_context_type(&mut self, qref: QualifiedRef, span: Span) -> InferTy {
        if let Some(ty) = self.env.contexts.get(&qref) {
            return ty.clone();
        }
        if let Some(ref mut state) = self.analysis {
            let solver = &mut *self.solver;
            return state
                .infer_vars
                .entry(qref.name)
                .or_insert_with(|| solver.fresh_ty_var())
                .clone();
        }
        self.error(
            MirErrorKind::UndefinedContext(self.interner.resolve(qref.name).to_string()),
            span,
        );
        Self::infer_error()
    }

    fn binop_error(&mut self, op: &'static str, left: InferTy, right: InferTy, span: Span) {
        self.error(
            MirErrorKind::TypeMismatchBinOp {
                op,
                left: self.freeze_or_error(&left),
                right: self.freeze_or_error(&right),
            },
            span,
        );
    }

    /// Check arity and unify argument types against parameter types.
    /// Returns `true` if arity matched, `false` (with error emitted) if not.
    fn check_args(
        &mut self,
        func: &str,
        arg_types: &[InferTy],
        arg_spans: &[Span],
        arg_ids: &[AstId],
        param_tys: &[InferTy],
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
            let _span = arg_spans.get(i).copied().unwrap_or(call_span);
            let coerce_id = arg_ids.get(i).copied();
            if self.unify_covariant(at, pt, coerce_id).is_err() {
                let resolved_pt = self.solver.resolve_ty(pt);
                let resolved_at = self.solver.resolve_ty(at);
                self.error(
                    MirErrorKind::UnificationFailure {
                        expected: self.freeze_or_error(&resolved_pt),
                        got: self.freeze_or_error(&resolved_at),
                    },
                    call_span,
                );
            }

            // Detect Fn return-type coercion: when arg is a lambda whose
            // return type was coerced (e.g. Deque → Iterator), register the
            // coercion on the lambda body expression's id so the lowerer
            // can insert a Cast at the return site.
            if let Some(&arg_id) = arg_ids.get(i) {
                self.detect_fn_ret_coercion(at, pt, arg_id);
            }
        }
        true
    }

    /// If `arg_ty` and `param_ty` are both `Fn` and their resolved return
    /// types require a Cast, look up the lambda body span from the type_map
    /// and register the coercion there.
    fn detect_fn_ret_coercion(&mut self, arg_ty: &InferTy, param_ty: &InferTy, lambda_id: AstId) {
        let resolved_arg = self.solver.resolve_ty(arg_ty);
        let resolved_param = self.solver.resolve_ty(param_ty);
        let (TyTerm::Fn { ret: arg_ret, .. }, TyTerm::Fn { ret: param_ret, .. }) =
            (&resolved_arg, &resolved_param)
        else {
            return;
        };
        if let (Ok(frozen_arg_ret), Ok(frozen_param_ret)) =
            (self.solver.freeze_ty(arg_ret), self.solver.freeze_ty(param_ret))
        {
            if let Some(kind) = CastKind::between(&frozen_arg_ret, &frozen_param_ret) {
                // Register the coercion on the lambda BODY span (not the lambda
                // expression span). This way the lowerer's `lower_expr(body)` →
                // `maybe_cast(body.span(), val)` naturally picks it up and inserts
                // a Cast before Return.
                if let Some(&body_id) = self.lambda_body_ids.get(&lambda_id) {
                    self.coercion_map.push((body_id, kind));
                }
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
            Node::InlineExpr { expr, span, .. } => {
                let ty = self.check_expr(false, expr);
                let resolved = self.solver.resolve_ty(&ty);
                match &resolved {
                    TyTerm::String | TyTerm::Error(_) => {}
                    TyTerm::Var(_) => {
                        if self
                            .unify_covariant(&ty, &TyTerm::String, Some(expr.id()))
                            .is_err()
                        {
                            self.error(MirErrorKind::EmitNotString { actual: self.freeze_or_error(&resolved) }, *span);
                        }
                    }
                    _ => self.error(MirErrorKind::EmitNotString { actual: self.freeze_or_error(&resolved) }, *span),
                }
            }
            Node::MatchBlock(mb) => self.check_match_block(mb),
            Node::IterBlock(ib) => self.check_iter_block(ib),
        }
    }

    /// Type-check a single script statement.
    fn check_stmt(&mut self, stmt: &acvus_ast::Stmt) {
        match stmt {
            acvus_ast::Stmt::Bind {
                id,
                name,
                expr,
                span: _,
            } => {
                let ty = self.check_expr(false, expr);
                self.define_var(*name, ty.clone());
                self.record(*id, ty);
            }
            acvus_ast::Stmt::ContextStore {
                id,
                name,
                path,
                expr,
                span,
            } => {
                let ty = self.check_expr(false, expr);
                self.record_context_write(*name);
                let ctx_ty = self
                    .env
                    .contexts
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| self.solver.fresh_ty_var());
                // Walk path to get the target field type.
                let target_ty = self.resolve_field_path(&ctx_ty, path);
                if self.solver.unify_ty(&ty, &target_ty, Polarity::Invariant, self.registry).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.freeze_or_error(&target_ty),
                            got: self.freeze_or_error(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::VarFieldStore {
                id,
                name,
                path,
                expr,
                span,
            } => {
                let ty = self.check_expr(false, expr);
                let var_ty = self.lookup_var(*name).unwrap_or_else(|| {
                    self.error(MirErrorKind::UndefinedVariable(self.interner.resolve(*name).to_string()), *span);
                    Self::infer_error()
                });
                let target_ty = self.resolve_field_path(&var_ty, path);
                if self.solver.unify_ty(&ty, &target_ty, Polarity::Invariant, self.registry).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.freeze_or_error(&target_ty),
                            got: self.freeze_or_error(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::Expr(expr) => {
                self.check_expr(false, expr);
            }
            acvus_ast::Stmt::MatchBind {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                let source_ty = self.check_expr(false, source);
                let resolved_source = self.solver.resolve_ty(&source_ty);
                self.push_scope();
                self.check_pattern(pattern, &resolved_source, *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            acvus_ast::Stmt::Iterate {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                let source_ty = self.check_expr(false, source);
                let resolved = self.solver.resolve_ty(&source_ty);
                let elem_ty = match &resolved {
                    TyTerm::List(inner) | TyTerm::Deque(inner, _) => inner.as_ref().clone(),
                    TyTerm::Range => TyTerm::Int,
                    TyTerm::Error(_) => Self::infer_error(),
                    _ => {
                        self.error(MirErrorKind::SourceNotIterable { actual: self.freeze_or_error(&resolved) }, *span);
                        return;
                    }
                };
                self.push_scope();
                self.check_pattern(pattern, &elem_ty, *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }

            // ── Script mode statements ──────────────────────────────

            acvus_ast::Stmt::LetBind {
                id,
                name,
                expr,
                span: _,
            } => {
                let ty = self.check_expr(false, expr);
                self.define_var(*name, ty.clone());
                self.record(*id, ty);
            }
            acvus_ast::Stmt::LetUninit {
                id,
                name,
                span: _,
            } => {
                let ty = self.solver.fresh_ty_var();
                self.define_var(*name, ty.clone());
                self.record(*id, ty);
            }
            acvus_ast::Stmt::Assign {
                id,
                name,
                expr,
                span,
            } => {
                let ty = self.check_expr(false, expr);
                let var_ty = self.lookup_var(*name).unwrap_or_else(|| {
                    self.error(MirErrorKind::UndefinedVariable(self.interner.resolve(*name).to_string()), *span);
                    Self::infer_error()
                });
                if self.solver.unify_ty(&ty, &var_ty, Polarity::Invariant, self.registry).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: self.freeze_or_error(&var_ty),
                            got: self.freeze_or_error(&ty),
                        },
                        *span,
                    );
                }
                self.record(*id, ty);
            }
            acvus_ast::Stmt::For {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                let source_ty = self.check_expr(false, source);
                let resolved = self.solver.resolve_ty(&source_ty);
                let elem_ty = match &resolved {
                    TyTerm::List(inner) | TyTerm::Deque(inner, _) => inner.as_ref().clone(),
                    TyTerm::Range => TyTerm::Int,
                    TyTerm::Error(_) => Self::infer_error(),
                    _ => {
                        self.error(MirErrorKind::SourceNotIterable { actual: self.freeze_or_error(&resolved) }, *span);
                        return;
                    }
                };
                self.push_scope();
                self.check_pattern(pattern, &elem_ty, *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            acvus_ast::Stmt::While {
                cond, body, span, ..
            } => {
                let cond_ty = self.check_expr(false, cond);
                if self.solver.unify_ty(&cond_ty, &TyTerm::Bool, Polarity::Invariant, self.registry).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: Ty::Bool,
                            got: self.freeze_or_error(&cond_ty),
                        },
                        *span,
                    );
                }
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
            acvus_ast::Stmt::WhileLet {
                pattern,
                source,
                body,
                span,
                ..
            } => {
                let source_ty = self.check_expr(false, source);
                let resolved = self.solver.resolve_ty(&source_ty);
                self.push_scope();
                self.check_pattern(pattern, &resolved, *span);
                for s in body {
                    self.check_stmt(s);
                }
                self.pop_scope();
            }
        }
    }

    fn check_match_block(&mut self, mb: &MatchBlock) {
        // Body-less variable binding: define in current scope (no push/pop).
        if self.is_bodyless_var_binding(mb) {
            let source_ty = self.check_expr(false, &mb.source);
            if matches!(&mb.arms[0].pattern, Pattern::Variant { .. }) {
                self.check_pattern(&mb.arms[0].pattern, &source_ty, mb.arms[0].tag_span);
            } else {
                let resolved_source = self.solver.resolve_ty(&source_ty);
                self.check_pattern(&mb.arms[0].pattern, &resolved_source, mb.arms[0].tag_span);
            }
            return;
        }

        let source_ty = self.check_expr(false, &mb.source);
        let resolved_source = self.solver.resolve_ty(&source_ty);

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
        let resolved = self.solver.resolve_ty(&source_ty);

        let elem_ty = match &resolved {
            TyTerm::List(inner) | TyTerm::Deque(inner, _) => inner.as_ref().clone(),
            TyTerm::Range => TyTerm::Int,
            TyTerm::Error(_) => Self::infer_error(),
            _ => {
                self.error(
                    MirErrorKind::SourceNotIterable { actual: self.freeze_or_error(&resolved) },
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
    fn pattern_match_type(&self, pattern: &Pattern, source_ty: &InferTy) -> InferTy {
        match pattern {
            Pattern::List { .. } | Pattern::Tuple { .. } | Pattern::Variant { .. } => {
                // List/Tuple patterns destructure the source as a whole.
                source_ty.clone()
            }
            _ => {
                // Other patterns match iterated elements.
                match source_ty {
                    TyTerm::List(inner) | TyTerm::Deque(inner, _) => inner.as_ref().clone(),
                    TyTerm::Range => TyTerm::Int,
                    _ => source_ty.clone(),
                }
            }
        }
    }

    fn check_expr(&mut self, allow_non_pure: bool, expr: &Expr) -> InferTy {
        match expr {
            Expr::Literal { id, value, span } => {
                let ty = match value {
                    Literal::Int(_) => TyTerm::Int,
                    Literal::Float(_) => TyTerm::Float,
                    Literal::String(_) => TyTerm::String,
                    Literal::Bool(_) => TyTerm::Bool,
                    Literal::Byte(_) => TyTerm::Byte,
                    Literal::Unit => TyTerm::Unit,
                    Literal::List(elems) => {
                        if elems.is_empty() {
                            let elem = self.solver.fresh_ty_var();
                            let origin = self.solver.alloc_identity(false);
                            TyTerm::Deque(Box::new(elem), Box::new(origin))
                        } else {
                            let first_ty = self.literal_ty(&elems[0]);
                            for elem in &elems[1..] {
                                let elem_ty = self.literal_ty(elem);
                                if self
                                    .unify_covariant(&elem_ty, &first_ty, Some(*id))
                                    .is_err()
                                {
                                    let resolved_first = self.solver.resolve_ty(&first_ty);
                                    let resolved_elem = self.solver.resolve_ty(&elem_ty);
                                    self.error(
                                        MirErrorKind::HeterogeneousList {
                                            expected: self.freeze_or_error(&resolved_first),
                                            got: self.freeze_or_error(&resolved_elem),
                                        },
                                        *span,
                                    );
                                }
                            }
                            let origin = self.solver.alloc_identity(false);
                            TyTerm::Deque(Box::new(self.solver.resolve_ty(&first_ty)), Box::new(origin))
                        }
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::ContextRef {
                id,
                name: qref,
                span,
            } => {
                let ty = self.resolve_context_type(*qref, *span);
                self.record_context_read(*qref);
                // Check materiality: need to freeze to check concrete type properties.
                let ty = if !allow_non_pure && !Self::is_error(&ty) && !Self::is_var(&ty) {
                    let frozen = self.freeze_or_error(&ty);
                    if frozen.materiality() == Materiality::Ephemeral {
                        self.error(
                            MirErrorKind::NonPureContextLoad {
                                name: self.interner.resolve(qref.name).to_string(),
                                ty: frozen,
                            },
                            *span,
                        );
                        Self::infer_error()
                    } else {
                        ty
                    }
                } else {
                    ty
                };
                self.record_ret(*id, ty)
            }

            Expr::Ident {
                id,
                name,
                ref_kind,
                span,
            } => {
                let ty = match ref_kind {
                    RefKind::ExternParam => {
                        match self.param_types.iter().find(|(n, _)| *n == name.name) {
                            Some((_, ty)) => ty.clone(),
                            None => {
                                if let Some(ref mut state) = self.analysis {
                                    // In analysis mode, unknown $params are inferred.
                                    // Use declared type from Signature if available.
                                    let ty = if state.next_declared_param
                                        < state.declared_param_types.len()
                                    {
                                        let t = &state.declared_param_types[state.next_declared_param];
                                        let lifted = lift_ty(t);
                                        state.next_declared_param += 1;
                                        lifted
                                    } else {
                                        self.solver.fresh_ty_var()
                                    };
                                    self.param_types.push((name.name, ty.clone()));
                                    ty
                                } else {
                                    self.error(
                                        MirErrorKind::UndefinedVariable(format!(
                                            "${}",
                                            self.interner.resolve(name.name)
                                        )),
                                        *span,
                                    );
                                    Self::infer_error()
                                }
                            }
                        }
                    }
                    RefKind::Value => match self.lookup_var(name.name) {
                        Some(ty) => ty,
                        None => {
                            // Undefined local variable — always an error.
                            // Use $name for extern params, @name for context.
                            self.error(
                                MirErrorKind::UndefinedVariable(
                                    self.interner.resolve(name.name).to_string(),
                                ),
                                *span,
                            );
                            Self::infer_error()
                        }
                    },
                };
                self.record_ret(*id, ty)
            }

            Expr::BinaryOp {
                id,
                left,
                op,
                right,
                span,
            } => {
                let lt = self.check_expr(false, left);
                let rt = self.check_expr(false, right);
                let lt = self.solver.resolve_ty(&lt);
                let rt = self.solver.resolve_ty(&rt);

                // Early guard: if either operand is Error, suppress cascading errors.
                if Self::is_error(&lt) || Self::is_error(&rt) {
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
                        | BinOp::Shr => Self::infer_error(),
                        _ => TyTerm::Bool,
                    };
                    return self.record_ret(*id, ty);
                }

                let ty = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        if self.solver.unify_ty(&lt, &rt, Polarity::Invariant, self.registry).is_err() {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                            return self.record_ret(*id, Self::infer_error());
                        }
                        let rl = self.solver.resolve_ty(&lt);
                        match &rl {
                            TyTerm::Int | TyTerm::Float | TyTerm::Var(_) => rl,
                            TyTerm::String if *op == BinOp::Add => TyTerm::String,
                            _ => {
                                self.binop_error(op_str(*op), rl, self.solver.resolve_ty(&rt), *span);
                                Self::infer_error()
                            }
                        }
                    }
                    BinOp::Eq | BinOp::Neq => {
                        if self.solver.unify_ty(&lt, &rt, Polarity::Invariant, self.registry).is_err() {
                            self.binop_error(op_str(*op), lt, rt, *span);
                        }
                        TyTerm::Bool
                    }
                    BinOp::And | BinOp::Or => {
                        let lok = self.unify_covariant(&lt, &TyTerm::Bool, None).is_ok();
                        let rok = self.unify_covariant(&rt, &TyTerm::Bool, None).is_ok();
                        if !lok || !rok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                        }
                        TyTerm::Bool
                    }
                    BinOp::Xor | BinOp::BitAnd | BinOp::BitOr | BinOp::Shl | BinOp::Shr => {
                        let lok = self.unify_covariant(&lt, &TyTerm::Int, None).is_ok();
                        let rok = self.unify_covariant(&rt, &TyTerm::Int, None).is_ok();
                        if !lok || !rok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                        }
                        TyTerm::Int
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::Lte | BinOp::Gte => {
                        let ok = self.solver.unify_ty(&lt, &rt, Polarity::Invariant, self.registry).is_ok()
                            && matches!(
                                self.solver.resolve_ty(&lt),
                                TyTerm::Int | TyTerm::Float | TyTerm::Var(_)
                            );
                        if !ok {
                            self.binop_error(
                                op_str(*op),
                                self.solver.resolve_ty(&lt),
                                self.solver.resolve_ty(&rt),
                                *span,
                            );
                        }
                        TyTerm::Bool
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::UnaryOp {
                id,
                op,
                operand,
                span,
            } => {
                let ot = self.check_expr(false, operand);
                let ot = self.solver.resolve_ty(&ot);

                // Early guard: if operand is Error, suppress cascading errors.
                if Self::is_error(&ot) {
                    let ty = match op {
                        acvus_ast::UnaryOp::Neg => Self::infer_error(),
                        acvus_ast::UnaryOp::Not => TyTerm::Bool,
                    };
                    return self.record_ret(*id, ty);
                }

                let ty = match op {
                    acvus_ast::UnaryOp::Neg => match &ot {
                        TyTerm::Int => TyTerm::Int,
                        TyTerm::Float => TyTerm::Float,
                        TyTerm::Var(_) => ot.clone(),
                        _ => {
                            self.binop_error("-", ot, Self::infer_error(), *span);
                            Self::infer_error()
                        }
                    },
                    acvus_ast::UnaryOp::Not => {
                        match &ot {
                            TyTerm::Bool => {}
                            TyTerm::Var(_) => {
                                let _ = self.unify_covariant(&ot, &TyTerm::Bool, None);
                            }
                            _ => self.binop_error("!", ot, Self::infer_error(), *span),
                        }
                        TyTerm::Bool
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::FieldAccess {
                id,
                object,
                field,
                span,
            } => {
                let ot_raw = self.check_expr(false, object);
                let ot = self.solver.resolve_ty(&ot_raw);
                let field_key = *field;
                let field_str = || self.interner.resolve(*field).to_string();
                let ty = match &ot {
                    TyTerm::Error(_) => Self::infer_error(),
                    TyTerm::Object(fields) if fields.contains_key(&field_key) => {
                        fields[&field_key].clone()
                    }
                    TyTerm::Object(fields) => {
                        let Some(leaf_var) = self.solver.find_leaf_var(&ot_raw) else {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: self.freeze_or_error(&ot),
                                    field: field_str(),
                                },
                                *span,
                            );
                            return Self::infer_error();
                        };
                        let fresh = self.solver.fresh_ty_var();
                        let mut new_fields = fields.clone();
                        new_fields.insert(field_key, fresh.clone());
                        self.solver.bind_ty(leaf_var, TyTerm::Object(new_fields));
                        fresh
                    }
                    TyTerm::Var(_) => {
                        let fresh = self.solver.fresh_ty_var();
                        let partial_obj =
                            TyTerm::Object(FxHashMap::from_iter([(field_key, fresh.clone())]));
                        if self
                            .solver
                            .unify_ty(&ot_raw, &partial_obj, Polarity::Invariant, self.registry)
                            .is_err()
                        {
                            self.error(
                                MirErrorKind::UndefinedField {
                                    object_ty: self.freeze_or_error(&ot),
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
                                object_ty: self.freeze_or_error(&ot),
                                field: field_str(),
                            },
                            *span,
                        );
                        Self::infer_error()
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::FuncCall {
                id,
                func,
                args,
                span,
            } => {
                let ty = self.check_func_call(func, args, None, *span);
                self.record_ret(*id, ty)
            }

            Expr::Pipe {
                id,
                left,
                right,
                span,
            } => {
                // Desugar: `a | f(b, c)` → `f(a, b, c)`
                // `a | f` → `f(a)`
                let pipe_left = Some(left.as_ref());
                let ty = match right.as_ref() {
                    Expr::FuncCall { func, args, .. } => {
                        self.check_func_call(func, args, pipe_left, *span)
                    }
                    Expr::Ident {
                        ref_kind: RefKind::Value,
                        ..
                    }
                    | Expr::ContextRef { .. } => self.check_func_call(right, &[], pipe_left, *span),
                    _ => {
                        let lt = self.check_expr(false, left);
                        let rt = self.check_expr(false, right);
                        self.check_callable(
                            &rt,
                            &[],
                            &Some(lt),
                            Some(left.span()),
                            Some(left.id()),
                            *span,
                        )
                    }
                };
                self.record_ret(*id, ty)
            }

            Expr::Lambda {
                id,
                params,
                body,
                span: _,
            } => {
                self.push_scope();
                let mut param_types = Vec::new();
                for p in params {
                    let pt = self.solver.fresh_ty_var();
                    self.define_var(p.name, pt.clone());
                    self.record(p.id, pt.clone());
                    param_types.push(ParamTerm::new(p.name, pt));
                }

                // Push lambda scope for capture tracking.
                self.lambda_stack.push(LambdaScope {
                    depth: self.scopes.len() - 1,
                    captures: Vec::new(),
                    effect: EffectTerm::Resolved(Default::default()),
                });

                let ret = self.check_expr(false, body);

                // Pop this lambda's scope.
                let ls = self.lambda_stack.pop().unwrap();
                let effect = ls.effect;
                let capture_types: Vec<InferTy> = ls
                    .captures
                    .into_iter()
                    .map(|t| self.solver.resolve_ty(&t))
                    .collect();

                // Record body span so detect_fn_ret_coercion can register
                // return-site coercions on the correct expression.
                self.lambda_body_ids.insert(*id, body.id());

                self.pop_scope();
                let ty = TyTerm::Fn {
                    params: param_types,
                    ret: Box::new(ret),
                    captures: capture_types,
                    effect,
                    hint: None,
                };
                self.record_ret(*id, ty)
            }

            Expr::Paren { id, inner, span: _ } => {
                let ty = self.check_expr(false, inner);
                self.record_ret(*id, ty)
            }

            Expr::List {
                id,
                head,
                rest,
                tail,
                span,
            } => {
                let all_elems: Vec<_> = head.iter().chain(tail.iter()).collect();
                if all_elems.is_empty() && rest.is_none() {
                    // Empty list `[]` — element type unknown, use fresh var.
                    // If no hint resolves it, we report the error after resolve.
                    let elem = self.solver.fresh_ty_var();
                    let origin = self.solver.alloc_identity(false);
                    let ty = TyTerm::Deque(Box::new(elem), Box::new(origin));
                    return self.record_ret(*id, ty);
                }

                let elem_ty = match all_elems.first() {
                    Some(first) => self.check_expr(false, first),
                    None => self.solver.fresh_ty_var(), // Only `..` with no elements: fresh var.
                };

                for elem in all_elems.iter().skip(1) {
                    let et = self.check_expr(false, elem);
                    if self.unify_covariant(&et, &elem_ty, None).is_err() {
                        let resolved_elem = self.solver.resolve_ty(&elem_ty);
                        let resolved_et = self.solver.resolve_ty(&et);
                        self.error(
                            MirErrorKind::HeterogeneousList {
                                expected: self.freeze_or_error(&resolved_elem),
                                got: self.freeze_or_error(&resolved_et),
                            },
                            *span,
                        );
                    }
                }

                let origin = self.solver.alloc_identity(false);
                let ty = TyTerm::Deque(Box::new(self.solver.resolve_ty(&elem_ty)), Box::new(origin));
                self.record_ret(*id, ty)
            }

            Expr::Object {
                id,
                fields,
                span: _,
            } => {
                let mut field_types = FxHashMap::default();
                for ObjectExprField { key, value, .. } in fields {
                    let ft = self.check_expr(false, value);
                    field_types.insert(*key, ft);
                }
                let ty = TyTerm::Object(field_types);
                self.record_ret(*id, ty)
            }

            Expr::Range {
                id,
                start,
                end,
                kind: _,
                span,
            } => {
                let st = self.check_expr(false, start);
                let et = self.check_expr(false, end);
                let st = self.solver.resolve_ty(&st);
                let et = self.solver.resolve_ty(&et);
                if !matches!(&st, TyTerm::Int | TyTerm::Error(_)) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: self.freeze_or_error(&st) }, *span);
                }
                if !matches!(&et, TyTerm::Int | TyTerm::Error(_)) {
                    self.error(MirErrorKind::RangeBoundsNotInt { actual: self.freeze_or_error(&et) }, *span);
                }
                self.record_ret(*id, TyTerm::Range)
            }

            Expr::Tuple {
                id,
                elements,
                span: _,
            } => {
                let elem_types: Vec<InferTy> = elements
                    .iter()
                    .map(|elem| match elem {
                        TupleElem::Expr(e) => self.check_expr(false, e),
                        TupleElem::Wildcard(_) => self.solver.fresh_ty_var(),
                    })
                    .collect();
                let ty = TyTerm::Tuple(elem_types);
                self.record_ret(*id, ty)
            }

            Expr::Group {
                id,
                elements,
                span: _,
            } => {
                // Group is only valid as lambda param list (handled by parser).
                let Some(last) = elements.last() else {
                    self.record(*id, TyTerm::Unit);
                    return TyTerm::Unit;
                };
                for e in &elements[..elements.len() - 1] {
                    self.check_expr(false, e);
                }
                let ty = self.check_expr(false, last);
                self.record_ret(*id, ty)
            }

            Expr::Variant {
                id,
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
                                return Self::infer_error();
                            };
                            let inner_ty = self.check_expr(false, inner_expr);
                            if self
                                .unify_covariant(&type_params[*idx], &inner_ty, None)
                                .is_err()
                            {
                                let resolved_tp = self.solver.resolve_ty(&type_params[*idx]);
                                let resolved_inner = self.solver.resolve_ty(&inner_ty);
                                self.error(
                                    MirErrorKind::UnificationFailure {
                                        expected: self.freeze_or_error(&resolved_tp),
                                        got: self.freeze_or_error(&resolved_inner),
                                    },
                                    *span,
                                );
                            }
                        }
                        VariantPayload::None => {}
                    }
                    // Builtin Option → Ty::Option
                    let inner = self.solver.resolve_ty(&type_params[0]);
                    let ty = TyTerm::Option(Box::new(inner));
                    return self.record_ret(*id, ty);
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
                    return Self::infer_error();
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
                let ty = TyTerm::Enum {
                    name: *enum_name,
                    variants,
                };
                self.record_ret(*id, ty)
            }

            Expr::Block {
                id,
                stmts,
                tail,
                span: _,
            } => {
                self.push_scope();
                for stmt in stmts {
                    self.check_stmt(stmt);
                }
                let ty = self.check_expr(false, tail);
                self.pop_scope();
                self.record_ret(*id, ty)
            }

            Expr::If {
                id,
                cond,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                let cond_ty = self.check_expr(false, cond);
                if self.solver.unify_ty(&cond_ty, &TyTerm::Bool, Polarity::Invariant, self.registry).is_err() {
                    self.error(
                        MirErrorKind::UnificationFailure {
                            expected: Ty::Bool,
                            got: self.freeze_or_error(&cond_ty),
                        },
                        *span,
                    );
                }
                self.push_scope();
                for s in then_body {
                    self.check_stmt(s);
                }
                let then_ty = match then_tail {
                    Some(tail) => self.check_expr(false, tail),
                    None => TyTerm::Unit,
                };
                self.pop_scope();
                let result_ty = match else_branch {
                    Some(eb) => {
                        let else_ty = self.check_else_branch(eb);
                        if self.solver.unify_ty(&then_ty, &else_ty, Polarity::Covariant, self.registry).is_err() {
                            self.error(
                                MirErrorKind::UnificationFailure {
                                    expected: self.freeze_or_error(&then_ty),
                                    got: self.freeze_or_error(&else_ty),
                                },
                                *span,
                            );
                        }
                        then_ty
                    }
                    None => then_ty,
                };
                self.record_ret(*id, result_ty)
            }

            Expr::IfLet {
                id,
                pattern,
                source,
                then_body,
                then_tail,
                else_branch,
                span,
            } => {
                let source_ty = self.check_expr(false, source);
                let resolved = self.solver.resolve_ty(&source_ty);
                self.push_scope();
                self.check_pattern(pattern, &resolved, *span);
                for s in then_body {
                    self.check_stmt(s);
                }
                let then_ty = match then_tail {
                    Some(tail) => self.check_expr(false, tail),
                    None => TyTerm::Unit,
                };
                self.pop_scope();
                let result_ty = match else_branch {
                    Some(eb) => {
                        let else_ty = self.check_else_branch(eb);
                        if self.solver.unify_ty(&then_ty, &else_ty, Polarity::Covariant, self.registry).is_err() {
                            self.error(
                                MirErrorKind::UnificationFailure {
                                    expected: self.freeze_or_error(&then_ty),
                                    got: self.freeze_or_error(&else_ty),
                                },
                                *span,
                            );
                        }
                        then_ty
                    }
                    None => then_ty,
                };
                self.record_ret(*id, result_ty)
            }
        }
    }

    fn check_else_branch(&mut self, eb: &acvus_ast::ElseBranch) -> InferTy {
        match eb {
            acvus_ast::ElseBranch::ElseIf(expr) => self.check_expr(false, expr),
            acvus_ast::ElseBranch::Else { body, tail, .. } => {
                self.push_scope();
                for s in body {
                    self.check_stmt(s);
                }
                let ty = match tail {
                    Some(tail) => self.check_expr(false, tail),
                    None => TyTerm::Unit,
                };
                self.pop_scope();
                ty
            }
        }
    }

    fn check_func_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        pipe_left: Option<&Expr>,
        call_span: Span,
    ) -> InferTy {
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
            let resolved = self.solver.resolve_ty(&ft);
            let pipe_left_span = pipe_left.map(|e| e.span());
            let pipe_left_id = pipe_left.map(|e| e.id());
            return self.check_callable(
                &resolved,
                args,
                &pipe_ty,
                pipe_left_span,
                pipe_left_id,
                call_span,
            );
        };

        // Check named functions (builtins, externs, user-defined).
        let name_str = self.interner.resolve(name.name);
        if let Some(fn_sig) = self.env.functions.get(name) {
            let arg_types: Vec<InferTy> = pipe_ty
                .iter()
                .cloned()
                .chain(args.iter().map(|a| self.check_expr(false, a)))
                .collect();
            let arg_spans: Vec<Span> = pipe_left
                .iter()
                .map(|e| e.span())
                .chain(args.iter().map(|a| a.span()))
                .collect();
            let arg_ids: Vec<AstId> = pipe_left
                .iter()
                .map(|e| e.id())
                .chain(args.iter().map(|a| a.id()))
                .collect();

            let fn_ty = self.solver.instantiate_poly(fn_sig);
            match &fn_ty {
                TyTerm::Fn {
                    params: param_tys,
                    ret,
                    effect,
                    ..
                } => {
                    self.propagate_call_effect(effect.clone());
                    let tys: Vec<InferTy> = param_tys.iter().map(|p| p.ty.clone()).collect();
                    if !self.check_args(name_str, &arg_types, &arg_spans, &arg_ids, &tys, call_span)
                    {
                        return Self::infer_error();
                    }
                    return self.solver.resolve_ty(ret);
                }
                _ => {
                    self.error(
                        MirErrorKind::UndefinedFunction(name_str.to_string()),
                        call_span,
                    );
                    return Self::infer_error();
                }
            }
        }

        // Check local variable with function type.
        if let Some(var_ty) = self.lookup_var(name.name) {
            let resolved = self.solver.resolve_ty(&var_ty);
            let pipe_left_span = pipe_left.map(|e| e.span());
            let pipe_left_id = pipe_left.map(|e| e.id());
            return self.check_callable(
                &resolved,
                args,
                &pipe_ty,
                pipe_left_span,
                pipe_left_id,
                call_span,
            );
        }

        self.error(
            MirErrorKind::UndefinedFunction(self.interner.resolve(name.name).to_string()),
            call_span,
        );
        Self::infer_error()
    }

    /// Propagate a callee's effect to the enclosing scope.
    /// If inside a lambda, propagates to the lambda scope.
    /// Otherwise, propagates to the top-level body_effect.
    fn propagate_call_effect(&mut self, effect: InferEffect) {
        let resolved = self.solver.resolve_infer_effect(&effect);
        if let EffectTerm::Resolved(callee_set) = &resolved
            && !callee_set.is_pure()
        {
            if let Some(ls) = self.lambda_stack.last_mut() {
                if let EffectTerm::Resolved(ref mut ls_set) = ls.effect {
                    *ls_set = ls_set.union(callee_set);
                } else {
                    ls.effect = resolved.clone();
                }
            } else if let EffectTerm::Resolved(ref mut body_set) = self.body_effect {
                *body_set = body_set.union(callee_set);
            }
        }
    }

    fn check_callable(
        &mut self,
        func_ty: &InferTy,
        args: &[Expr],
        pipe_ty: &Option<InferTy>,
        pipe_left_span: Option<Span>,
        pipe_left_id: Option<AstId>,
        call_span: Span,
    ) -> InferTy {
        // Early exit for non-callable types.
        match func_ty {
            TyTerm::Fn { .. } | TyTerm::Var(_) => {}
            TyTerm::Error(_) => {
                for a in args {
                    self.check_expr(false, a);
                }
                return Self::infer_error();
            }
            _ => {
                self.error(
                    MirErrorKind::UndefinedFunction("<not callable>".to_string()),
                    call_span,
                );
                return Self::infer_error();
            }
        }

        let arg_types: Vec<InferTy> = pipe_ty
            .iter()
            .cloned()
            .chain(args.iter().map(|a| self.check_expr(false, a)))
            .collect();
        let arg_spans: Vec<Span> = pipe_left_span
            .iter()
            .copied()
            .chain(args.iter().map(|a| a.span()))
            .collect();
        let arg_ids: Vec<AstId> = pipe_left_id
            .iter()
            .copied()
            .chain(args.iter().map(|a| a.id()))
            .collect();

        match func_ty {
            TyTerm::Fn {
                params,
                ret,
                effect,
                ..
            } => {
                // Propagate effect to enclosing scope (lambda or top-level body).
                self.propagate_call_effect(effect.clone());
                let tys: Vec<InferTy> = params.iter().map(|p| p.ty.clone()).collect();
                if !self.check_args(
                    "<closure>",
                    &arg_types,
                    &arg_spans,
                    &arg_ids,
                    &tys,
                    call_span,
                ) {
                    return Self::infer_error();
                }
                self.solver.resolve_ty(ret)
            }
            TyTerm::Var(_) => {
                let ret = self.solver.fresh_ty_var();
                let dummy = self.interner.intern("_");
                let fn_ty = TyTerm::Fn {
                    params: arg_types
                        .into_iter()
                        .map(|ty| ParamTerm::new(dummy, ty))
                        .collect(),
                    ret: Box::new(ret.clone()),
                    captures: vec![],
                    effect: EffectTerm::Resolved(Default::default()),
                    hint: None,
                };
                if self.unify_covariant(func_ty, &fn_ty, None).is_err() {
                    self.error(
                        MirErrorKind::UndefinedFunction("<expr>".to_string()),
                        call_span,
                    );
                    return Self::infer_error();
                }
                self.solver.resolve_ty(&ret)
            }
            _ => unreachable!(),
        }
    }

    fn check_pattern(&mut self, pattern: &Pattern, source_ty: &InferTy, span: Span) {
        let source_resolved = self.solver.resolve_ty(source_ty);
        match pattern {
            Pattern::ContextBind { name: qref, .. } => {
                // Context write allowed — mutability will be enforced later.
                self.record_context_write(*qref);
                let ctx_ty = self
                    .env
                    .contexts
                    .get(qref)
                    .cloned()
                    .unwrap_or_else(|| self.solver.fresh_ty_var());
                if self
                    .solver
                    .unify_ty(&source_resolved, &ctx_ty, Polarity::Invariant, self.registry)
                    .is_err()
                {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.freeze_or_error(&ctx_ty),
                            source_ty: self.freeze_or_error(&source_resolved),
                        },
                        span,
                    );
                }
            }
            Pattern::Binding { name, ref_kind, .. } => match ref_kind {
                RefKind::ExternParam => {
                    self.error(
                        MirErrorKind::ExternParamAssign(self.interner.resolve(*name).to_string()),
                        span,
                    );
                }
                RefKind::Value => {
                    self.define_var(*name, source_resolved);
                }
            },

            Pattern::Literal { value, .. } => {
                let pat_ty = self.literal_ty(value);
                if self
                    .unify_covariant(&source_resolved, &pat_ty, None)
                    .is_err()
                {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.freeze_or_error(&pat_ty),
                            source_ty: self.freeze_or_error(&source_resolved),
                        },
                        span,
                    );
                }
            }

            Pattern::List { head, tail, .. } => {
                // Reuse existing element Var when source already resolves to a
                // List. Same rationale as Tuple above.
                let shallow = self.solver.shallow_resolve_ty(source_ty);
                let elem_ty = match shallow {
                    TyTerm::List(ref inner) | TyTerm::Deque(ref inner, _) => (**inner).clone(),
                    _ => {
                        let var = self.solver.fresh_ty_var();
                        let origin = self.solver.alloc_identity(false);
                        let list_ty = TyTerm::Deque(Box::new(var.clone()), Box::new(origin));
                        if self.unify_covariant(source_ty, &list_ty, None).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: self.freeze_or_error(&list_ty),
                                    source_ty: self.freeze_or_error(&source_resolved),
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
                let obj_fields = if let TyTerm::Object(obj_fields) = &source_resolved {
                    obj_fields.clone()
                } else {
                    let field_vars: FxHashMap<Astr, InferTy> = fields
                        .iter()
                        .map(|f| (f.key, self.solver.fresh_ty_var()))
                        .collect();
                    let obj_ty = TyTerm::Object(field_vars.clone());
                    if self.unify_covariant(source_ty, &obj_ty, None).is_err() {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: self.freeze_or_error(&obj_ty),
                                source_ty: self.freeze_or_error(&source_resolved),
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
                                object_ty: self.freeze_or_error(&source_resolved),
                                field: self.interner.resolve(*key).to_string(),
                            },
                            span,
                        );
                        continue;
                    };
                    let resolved = self.solver.resolve_ty(field_ty);
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
                    .unify_covariant(&source_resolved, &TyTerm::Int, None)
                    .is_err()
                {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: Ty::Int,
                            source_ty: self.freeze_or_error(&source_resolved),
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
                let shallow = self.solver.shallow_resolve_ty(source_ty);
                let elem_tys = match shallow {
                    TyTerm::Tuple(ref existing) if existing.len() == elements.len() => existing.clone(),
                    _ => {
                        let vars: Vec<InferTy> =
                            elements.iter().map(|_| self.solver.fresh_ty_var()).collect();
                        let tuple_ty = TyTerm::Tuple(vars.clone());
                        if self.unify_covariant(source_ty, &tuple_ty, None).is_err() {
                            self.error(
                                MirErrorKind::PatternTypeMismatch {
                                    pattern_ty: self.freeze_or_error(&tuple_ty),
                                    source_ty: self.freeze_or_error(&source_resolved),
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
                    let enum_ty = TyTerm::Option(Box::new(self.solver.resolve_ty(&type_params[0])));
                    if self
                        .unify_covariant(&source_resolved, &enum_ty, None)
                        .is_err()
                    {
                        self.error(
                            MirErrorKind::PatternTypeMismatch {
                                pattern_ty: self.freeze_or_error(&enum_ty),
                                source_ty: self.freeze_or_error(&source_resolved),
                            },
                            span,
                        );
                        return;
                    }

                    if let VariantPayload::TypeParam(idx) = &variant_payload {
                        let resolved_inner = self.solver.resolve_ty(&type_params[*idx]);
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
                    Some(Box::new(self.solver.fresh_ty_var()))
                } else {
                    None
                };
                let mut variants = FxHashMap::default();
                variants.insert(*tag, payload_ty.clone());
                let enum_ty = TyTerm::Enum {
                    name: *enum_name,
                    variants,
                };
                // Unify against the original (unresolved) source_ty so that
                // find_leaf_var can trace the Var chain and rebind the merged type.
                if self.unify_covariant(source_ty, &enum_ty, None).is_err() {
                    self.error(
                        MirErrorKind::PatternTypeMismatch {
                            pattern_ty: self.freeze_or_error(&enum_ty),
                            source_ty: self.freeze_or_error(&source_resolved),
                        },
                        span,
                    );
                    return;
                }

                // Bind payload pattern if present.
                if let Some(inner_pat) = payload {
                    let inner_ty = payload_ty
                        .map(|ty| self.solver.resolve_ty(&ty))
                        .unwrap_or_else(Self::infer_error);
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
    ) -> Option<(Astr, Vec<InferTy>, VariantPayload)> {
        let tag_str = self.interner.resolve(tag);
        let option_name = self.interner.intern("Option");

        // Check qualified name if present.
        if let Some(ename) = ast_enum_name
            && *ename != option_name
        {
            return None;
        }

        let payload = match tag_str {
            "Some" => VariantPayload::TypeParam(0),
            "None" => VariantPayload::None,
            _ => return None,
        };

        let type_params = vec![self.solver.fresh_ty_var()];
        Some((option_name, type_params, payload))
    }

    fn literal_ty(&mut self, lit: &Literal) -> InferTy {
        match lit {
            Literal::Int(_) => TyTerm::Int,
            Literal::Float(_) => TyTerm::Float,
            Literal::String(_) => TyTerm::String,
            Literal::Bool(_) => TyTerm::Bool,
            Literal::Byte(_) => TyTerm::Byte,
            Literal::Unit => TyTerm::Unit,
            Literal::List(elems) => {
                let origin = self.solver.alloc_identity(false);
                match elems.first() {
                    Some(first) => TyTerm::Deque(Box::new(self.literal_ty(first)), Box::new(origin)),
                    None => TyTerm::Deque(Box::new(Self::infer_error()), Box::new(origin)),
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
                self.error(MirErrorKind::RangeBoundsNotInt { actual: self.freeze_or_error(&ty) }, span);
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
    use crate::graph::types::QualifiedRef;

    /// Test helper: create a `Param` with name "_".
    fn p(i: &Interner, ty: Ty) -> Param {
        Param::new(i.intern("_"), ty)
    }

    fn check(source: &str) -> Result<TypeMap, String> {
        let interner = Interner::new();
        check_with_interner(source, &FxHashMap::default(), &interner)
    }

    fn check_with_interner(
        source: &str,
        context: &FxHashMap<Astr, Ty>,
        interner: &Interner,
    ) -> Result<TypeMap, String> {
        let template = acvus_ast::parse(interner, source).expect("parse failed");
        let mut solver = Solver::new();
        let registry = TypeRegistry::default();
        // Convert Astr-keyed context map to QualifiedRef-keyed (root namespace).
        let qref_contexts: FxHashMap<QualifiedRef, InferTy> = context
            .iter()
            .map(|(&name, ty)| (QualifiedRef::root(name), crate::ty::lift_ty(ty)))
            .collect();

        let env = crate::ty::TypeEnv {
            contexts: qref_contexts,
            functions: Default::default(),
        };
        let checker = TypeChecker::new(interner, &env, &registry, &mut solver);
        let resolution = checker.check_template(&template).map_err(|errs| {
            errs.iter()
                .map(|e| {
                    format!(
                        "[typeck] [{}..{}] {}",
                        e.span.start,
                        e.span.end,
                        e.display(interner)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        })?;
        Ok(resolution.type_map)
    }

    #[test]
    fn literal_string_emit() {
        check("{{ \"hello\" }}").unwrap();
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
        check(src).unwrap();
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
        result.unwrap();
    }

    #[test]
    fn context_read() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("name"), Ty::String)]);
        let src = "{{ @name }}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn undefined_variable() {
        let src = "{{ x = unknown }}{{_}}{{/}}";
        let result = check(src);
        assert!(result.is_err());
    }

    #[test]
    fn extern_param_write_rejected() {
        let src = "{{ $count = 42 }}";
        let err = check(src).expect_err("should reject extern param write");
        assert!(
            err.contains("$count"),
            "expected ExternParamAssign error, got: {err}"
        );
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
                hint: None,
            },
        )]);
        let src = "{{ x = @fetch_user(1) }}{{ x }}{{_}}{{/}}";
        check_with_interner(src, &context, &i).unwrap();
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
        check_with_interner(src, &context, &i).unwrap();
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
        check_with_interner(src, &context, &i).unwrap();
    }

    // ── Variant (Option) ────────────────────────────────────────────

    #[test]
    fn some_int_is_option_int() {
        let src = "{{ x = Some(42) }}{{_}}{{/}}";
        check(src).unwrap();
    }

    #[test]
    fn none_is_option() {
        let src = "{{ x = None }}{{_}}{{/}}";
        check(src).unwrap();
    }

    #[test]
    fn some_pattern_extracts_inner() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::String)))]);
        let src = "{{ Some(x) = @opt }}{{ x }}{{_}}{{/}}";
        check_with_interner(src, &context, &i).unwrap();
    }

    #[test]
    fn none_pattern_matches_option() {
        let i = Interner::new();
        let context = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::Int)))]);
        let src = "{{ None = @opt }}none{{_}}has value{{/}}";
        check_with_interner(src, &context, &i).unwrap();
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
                    hint: None,
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
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn extern_fn_bare_ref_allowed() {
        // f = @my_fn — Fn is Lazy tier, allowed in non-call position.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = "{{ f = @my_fn }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i)
            .expect("bare reference to extern fn should be allowed (Lazy tier)");
    }

    #[test]
    fn extern_fn_pipe_call_ok() {
        // "hello" | @my_fn — pipe into extern fn is a call, should be allowed.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = r#"{{ "hello" | @my_fn }}"#;
        check_with_interner(src, &ctx, &i).unwrap();
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
                hint: None,
            },
        )]);
        let src = r#"{{ "hello" | @my_fn(42) }}"#;
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn pure_context_ref_ok() {
        // @name — bare reference to pure type (String) is fine.
        let i = Interner::new();
        let ctx = extern_fn_context(&i);
        let src = "{{ @name }}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    // ── 3-tier purity: Lazy context load tests ──

    #[test]
    fn lazy_list_context_load_ok() {
        // @items : List<Int> — Lazy tier, allowed in non-call position.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("items"), Ty::List(Box::new(Ty::Int)))]);
        let src = "{{ x = @items }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    // Iterator/Sequence context load tests migrated to acvus-mir-test
    // (requires TypeRegistry + Interner for UserDefined construction).

    #[test]
    fn lazy_option_context_load_ok() {
        // @opt : Option<Int> — Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("opt"), Ty::Option(Box::new(Ty::Int)))]);
        let src = "{{ x = @opt }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn lazy_tuple_context_load_ok() {
        // @pair : (Int, String) — Lazy tier, allowed.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("pair"), Ty::Tuple(vec![Ty::Int, Ty::String]))]);
        let src = "{{ x = @pair }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
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
        check_with_interner(src, &ctx, &i).unwrap();
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
                hint: None,
            },
        )]);
        let src = "{{ f = @callback }}{{ f(42) }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
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
                hint: None,
            })),
        )]);
        let src = "{{ x = @fns }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    // ── Unpure context load tests (UserDefined — must be rejected) ──

    #[test]
    fn unpure_opaque_context_load_rejected() {
        // @conn : UserDefined — Unpure tier, rejected in non-call position.
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(
            i.intern("conn"),
            Ty::UserDefined {
                id: QualifiedRef::root(i.intern("TestOpaque")),
                type_args: vec![],
                effect_args: vec![],
            },
        )]);
        let src = "{{ x = @conn }}{{_}}{{/}}";
        let err = check_with_interner(src, &ctx, &i)
            .expect_err("UserDefined context load should be rejected");
        assert!(
            err.contains("non-pure") || err.contains("NonPure"),
            "expected NonPureContextLoad error, got: {err}"
        );
    }

    #[test]
    fn unpure_opaque_in_argument_also_rejected() {
        // @handler(@conn) — @conn is UserDefined, rejected even in argument position.
        // Arguments are checked with allow_non_pure=false.
        let i = Interner::new();
        let conn_ty = Ty::UserDefined {
            id: QualifiedRef::root(i.intern("TestOpaque")),
            type_args: vec![],
            effect_args: vec![],
        };
        let ctx = FxHashMap::from_iter([
            (i.intern("conn"), conn_ty.clone()),
            (
                i.intern("handler"),
                Ty::Fn {
                    params: vec![p(&i, conn_ty)],
                    ret: Box::new(Ty::String),
                    effect: Effect::pure(),
                    captures: vec![],
                    hint: None,
                },
            ),
        ]);
        let src = "{{ @handler(@conn) }}";
        let result = check_with_interner(src, &ctx, &i);
        assert!(
            result.is_err(),
            "UserDefined in argument should be rejected"
        );
    }

    // ── Pure context load tests (scalars — always ok) ──

    #[test]
    fn pure_int_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("count"), Ty::Int)]);
        let src = "{{ x = @count }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }

    #[test]
    fn pure_string_context_load_ok() {
        let i = Interner::new();
        let ctx = FxHashMap::from_iter([(i.intern("msg"), Ty::String)]);
        let src = "{{ x = @msg }}{{_}}{{/}}";
        check_with_interner(src, &ctx, &i).unwrap();
    }
}
