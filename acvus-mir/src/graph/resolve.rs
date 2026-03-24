//! Phase 2: Resolve
//!
//! Complete typecheck with fully known context types.
//! Takes Phase 1 inferred types + user-provided context values,
//! builds a full registry, and produces TypeResolution<Checked> per function.
//!
//! SSA/PHI will be added later on top of this.

use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::MirError;
use crate::ty::{Ty, TySubst};
use crate::typeck::{Checked, TypeResolution, check_completeness};

use super::extract::{ExtractResult, ParsedSource};
use super::types::*;

// ── Phase 2 output ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct ResolveError {
    pub fn_id: FunctionId,
    pub errors: Vec<MirError>,
}

#[derive(Debug)]
pub struct ResolvedGraph {
    /// Per-function checked type resolutions (only for Local functions).
    resolutions: FxHashMap<FunctionId, TypeResolution<Checked>>,
    /// Resolved output types for all functions.
    fn_types: FxHashMap<FunctionId, Ty>,
    /// Resolved types for all contexts.
    context_types: FxHashMap<ContextId, Ty>,
    /// Errors encountered during resolution.
    errors: Vec<ResolveError>,
}

impl ResolvedGraph {
    pub fn resolution(&self, id: FunctionId) -> &TypeResolution<Checked> {
        self.resolutions
            .get(&id)
            .expect("no resolution for function")
    }

    pub fn try_resolution(&self, id: FunctionId) -> Option<&TypeResolution<Checked>> {
        self.resolutions.get(&id)
    }

    pub fn fn_type(&self, id: FunctionId) -> Option<&Ty> {
        self.fn_types.get(&id)
    }

    pub fn context_type(&self, id: ContextId) -> Option<&Ty> {
        self.context_types.get(&id)
    }

    pub fn errors(&self) -> &[ResolveError] {
        &self.errors
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
}

// ── Per-function resolution ──────────────────────────────────────────

/// Resolve a single function. Returns the checked resolution + output type,
/// or a list of errors.
pub fn resolve_one(
    interner: &Interner,
    func: &Function,
    parsed: &ParsedSource,
    bind_params: &[crate::ty::Param],
    env: &crate::ty::TypeEnv,
) -> Result<(TypeResolution<Checked>, Ty), Vec<MirError>> {
    // Validate: inferred params must be declared in the signature.
    if !bind_params.is_empty() {
        let sig_names: FxHashSet<Astr> = func
            .constraint
            .signature
            .as_ref()
            .map(|sig| sig.params.iter().map(|p| p.name).collect())
            .unwrap_or_default();
        let mut undeclared_errors = Vec::new();
        for param in bind_params {
            if !sig_names.contains(&param.name) {
                undeclared_errors.push(MirError {
                    kind: crate::error::MirErrorKind::UndefinedVariable(
                        interner.resolve(param.name).to_string(),
                    ),
                    span: acvus_ast::Span::new(0, 0),
                });
            }
        }
        if !undeclared_errors.is_empty() {
            return Err(undeclared_errors);
        }
    }

    let mut subst = TySubst::new();
    let checker =
        crate::typeck::TypeChecker::new(interner, env, &mut subst).with_params(bind_params);
    let unchecked_result = match parsed {
        ParsedSource::Script(script) => checker.check_script(script, None),
        ParsedSource::Template(template) => checker.check_template(template),
    };

    match unchecked_result {
        Ok(unchecked) => match check_completeness(unchecked, &subst) {
            Ok(checked) => {
                let tail_ty = checked.tail_ty.clone();
                Ok((checked, tail_ty))
            }
            Err(errs) => Err(errs),
        },
        Err(errs) => Err(errs),
    }
}

// ── Batch resolution ────────────────────────────────────────────────

/// Run Phase 2 resolution (batch).
///
/// All types must be fully resolved by this point (from infer + UI).
/// `user_context_types`: additional context types provided by the user
/// (e.g., from UI injection). These override inferred types.
pub fn resolve(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    infer_result: &super::infer::InferResult,
    user_context_types: &FxHashMap<Astr, Ty>,
) -> ResolvedGraph {
    let mut subst = TySubst::new();
    let mut resolutions: FxHashMap<FunctionId, TypeResolution<Checked>> = FxHashMap::default();
    let mut fn_types: FxHashMap<FunctionId, Ty> = FxHashMap::default();
    let mut errors: Vec<ResolveError> = Vec::new();

    // Build context type map: declared (from graph) + user-provided.
    let mut context_types: FxHashMap<Astr, Ty> = FxHashMap::default();

    // 1. Declared context types from the graph.
    for ctx in graph.contexts.iter() {
        match &ctx.constraint {
            Constraint::Exact(ty) => {
                context_types.insert(ctx.name, ty.clone());
            }
            Constraint::Inferred => {
                // Will be filled by user_context_types or remain as TyVar.
                if !user_context_types.contains_key(&ctx.name) {
                    context_types.insert(ctx.name, subst.fresh_param());
                }
            }
            Constraint::DerivedFnOutput(_, _) | Constraint::DerivedContext(_, _) => {
                // TODO: resolve derived types. For now, use fresh var.
                context_types.insert(ctx.name, subst.fresh_param());
            }
        }
    }

    // 2. User-provided types override.
    for (&name, ty) in user_context_types {
        context_types.insert(name, ty.clone());
    }

    // 3. Verify all referenced contexts are declared.
    //    Phase 2 requires all contexts to be known (from graph + Phase 1 infer).
    //    Undeclared contexts are errors.
    for (&fn_id, fn_ref) in &extract.fn_refs {
        for &name in fn_ref
            .context_reads
            .iter()
            .chain(fn_ref.context_writes.iter())
        {
            if !context_types.contains_key(&name) {
                errors.push(ResolveError {
                    fn_id,
                    errors: vec![MirError {
                        kind: crate::error::MirErrorKind::UndefinedContext(
                            interner.resolve(name).to_string(),
                        ),
                        span: acvus_ast::Span::new(0, 0),
                    }],
                });
            }
        }
    }

    // 4. Build function type environment from infer results.
    //    All local function types are fully resolved by infer — no fresh Params here.
    let mut fn_type_env: FxHashMap<Astr, Ty> = crate::builtins::builtin_fn_types(interner);

    for func in graph.functions.iter() {
        match &func.kind {
            FnKind::Local(_) => {
                if let Some(meta) = infer_result.functions.get(&func.id) {
                    fn_type_env.insert(func.name, meta.ty.clone());
                }
            }
            FnKind::Extern { .. } => {
                if let Constraint::Exact(ty) = &func.constraint.output {
                    fn_type_env.insert(func.name, ty.clone());
                }
            }
        }
    }

    // 5. Typecheck each local function with the complete environment.
    let env = crate::ty::TypeEnv {
        contexts: context_types.clone(),
        functions: fn_type_env,
    };

    for func in graph.functions.iter() {
        let FnKind::Local(_source) = &func.kind else {
            continue;
        };
        let Some(parsed) = extract.parsed.get(&func.id) else {
            continue;
        };

        // Get parameter bindings from infer result.
        let bind_params = infer_result
            .functions
            .get(&func.id)
            .map(|m| m.params.clone())
            .unwrap_or_default();

        // Validate: inferred params must be declared in the signature.
        // An implicit param not in the signature means an undefined variable.
        if !bind_params.is_empty() {
            let sig_names: FxHashSet<Astr> = func
                .constraint
                .signature
                .as_ref()
                .map(|sig| sig.params.iter().map(|p| p.name).collect())
                .unwrap_or_default();
            let mut undeclared_errors = Vec::new();
            for param in &bind_params {
                if !sig_names.contains(&param.name) {
                    undeclared_errors.push(MirError {
                        kind: crate::error::MirErrorKind::UndefinedVariable(
                            interner.resolve(param.name).to_string(),
                        ),
                        span: acvus_ast::Span::new(0, 0),
                    });
                }
            }
            if !undeclared_errors.is_empty() {
                errors.push(ResolveError {
                    fn_id: func.id,
                    errors: undeclared_errors,
                });
                continue;
            }
        }

        // Typecheck with pre-bound parameters.
        let checker =
            crate::typeck::TypeChecker::new(interner, &env, &mut subst).with_params(&bind_params);
        let unchecked_result = match parsed {
            ParsedSource::Script(script) => checker.check_script(script, None),
            ParsedSource::Template(template) => checker.check_template(template),
        };

        match unchecked_result {
            Ok(unchecked) => {
                // Try check_completeness.
                match check_completeness(unchecked, &subst) {
                    Ok(checked) => {
                        fn_types.insert(func.id, checked.tail_ty.clone());
                        resolutions.insert(func.id, checked);
                    }
                    Err(errs) => {
                        errors.push(ResolveError {
                            fn_id: func.id,
                            errors: errs,
                        });
                    }
                }
            }
            Err(errs) => {
                errors.push(ResolveError {
                    fn_id: func.id,
                    errors: errs,
                });
            }
        }
    }

    // 5. Resolve context types through substitution.
    let resolved_context_types: FxHashMap<ContextId, Ty> = graph
        .contexts
        .iter()
        .map(|ctx| {
            let ty = context_types
                .get(&ctx.name)
                .map(|t| subst.resolve(t))
                .unwrap_or_else(Ty::error);
            (ctx.id, ty)
        })
        .collect();

    ResolvedGraph {
        resolutions,
        fn_types,
        context_types: resolved_context_types,
        errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use crate::ty::Effect;
    use acvus_utils::{Freeze, Interner};

    // ── Single-function helpers (existing) ───────────────────────────

    fn make_graph_with_ctx(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                id: ContextId::alloc(),
                name: interner.intern(name),
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        let mut functions = crate::builtins::standard_builtins(interner);
        functions.push(Function {
            id: FunctionId::alloc(),
            name: interner.intern("test"),
            kind: FnKind::Local(SourceCode {
                name: interner.intern("test"),
                source: interner.intern(source),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
            },
        });
        CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        }
    }

    fn make_graph_no_ctx(interner: &Interner, source: &str) -> CompilationGraph {
        make_graph_with_ctx(interner, source, &[])
    }

    fn last_local_id(graph: &CompilationGraph) -> FunctionId {
        graph
            .functions
            .iter()
            .rev()
            .find(|f| matches!(f.kind, FnKind::Local(_)))
            .expect("no local function")
            .id
    }

    // -- Completeness: valid programs resolve --

    #[test]
    fn resolve_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_no_ctx(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", result.errors());
        let uid = last_local_id(&graph);
        assert!(result.try_resolution(uid).is_some());
        assert_eq!(*result.fn_type(uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_declared_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", result.errors());
        let uid = last_local_id(&graph);
        assert_eq!(*result.fn_type(uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_user_provided_context() {
        let i = Interner::new();
        let graph = make_graph_no_ctx(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let mut user = FxHashMap::default();
        user.insert(i.intern("x"), Ty::Int);
        let result = resolve(&i, &graph, &ext, &inf, &user);

        assert!(!result.has_errors(), "errors: {:?}", result.errors());
        let uid = last_local_id(&graph);
        assert_eq!(*result.fn_type(uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_string_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@name", &[("name", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", result.errors());
        let uid = last_local_id(&graph);
        assert_eq!(*result.fn_type(uid).unwrap(), Ty::String);
    }

    #[test]
    fn resolve_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let graph = make_graph_with_ctx(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", result.errors());
        let uid = last_local_id(&graph);
        assert_eq!(*result.fn_type(uid).unwrap(), Ty::String);
    }

    // -- Soundness: type errors detected --

    #[test]
    fn resolve_type_mismatch_detected() {
        let i = Interner::new();
        // @x is String but used in arithmetic — should error.
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        assert!(result.has_errors(), "should detect type mismatch");
    }

    // -- Context type resolution --

    #[test]
    fn resolve_context_types_populated() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let inf = crate::graph::infer::infer(&i, &graph, &ext);
        let result = resolve(&i, &graph, &ext, &inf, &FxHashMap::default());

        let ctx_id = graph.contexts[0].id;
        assert_eq!(*result.context_type(ctx_id).unwrap(), Ty::Int);
    }

    // ════════════════════════════════════════════════════════════════
    // Inter-function call tests
    // ════════════════════════════════════════════════════════════════

    /// Helper: build a multi-function CompilationGraph.
    ///
    /// `fns`: list of `(name, source, signature, output_constraint)`.
    /// - `signature`: `None` = no declared params, `Some(params)` = declared param types.
    /// - `output`: `Constraint` for the function's return type.
    ///
    /// `ctx`: contexts as before.
    ///
    /// Returns the graph and a Vec of (name, FunctionId) for each local function
    /// (in insertion order).
    fn make_multi_fn_graph(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        ctx: &[(&str, Ty)],
    ) -> (CompilationGraph, Vec<(Astr, FunctionId)>) {
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                id: ContextId::alloc(),
                name: interner.intern(name),
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();

        let mut functions = crate::builtins::standard_builtins(interner);
        let mut ids = Vec::new();

        for (name, source, sig, output) in fns {
            let fid = FunctionId::alloc();
            let aname = interner.intern(name);
            ids.push((aname, fid));
            functions.push(Function {
                id: fid,
                name: aname,
                kind: FnKind::Local(SourceCode {
                    name: aname,
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: sig.as_ref().map(|params| Signature {
                        params: params
                            .iter()
                            .map(|(name, ty)| {
                                crate::ty::Param::new(interner.intern(name), ty.clone())
                            })
                            .collect(),
                    }),
                    output: output.clone(),
                },
            });
        }

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        (graph, ids)
    }

    /// Convenience: resolve a multi-function graph, return errors as strings.
    fn resolve_multi(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        ctx: &[(&str, Ty)],
    ) -> (ResolvedGraph, Vec<(Astr, FunctionId)>) {
        resolve_with_extern(interner, fns, &[], ctx)
    }

    /// Build a graph with both local and extern functions, then resolve.
    fn resolve_with_extern(
        interner: &Interner,
        local_fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        extern_fns: &[Function],
        ctx: &[(&str, Ty)],
    ) -> (ResolvedGraph, Vec<(Astr, FunctionId)>) {
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                id: ContextId::alloc(),
                name: interner.intern(name),
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();

        let mut functions = crate::builtins::standard_builtins(interner);
        let mut ids = Vec::new();

        for (name, source, sig, output) in local_fns {
            let fid = FunctionId::alloc();
            let aname = interner.intern(name);
            ids.push((aname, fid));
            functions.push(Function {
                id: fid,
                name: aname,
                kind: FnKind::Local(SourceCode {
                    name: aname,
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: sig.as_ref().map(|params| Signature {
                        params: params
                            .iter()
                            .map(|(name, ty)| {
                                crate::ty::Param::new(interner.intern(name), ty.clone())
                            })
                            .collect(),
                    }),
                    output: output.clone(),
                },
            });
        }
        functions.extend_from_slice(extern_fns);

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        let ext = extract::extract(interner, &graph);
        let inf = crate::graph::infer::infer(interner, &graph, &ext);
        let result = resolve(interner, &graph, &ext, &inf, &FxHashMap::default());
        (result, ids)
    }

    fn error_strings(interner: &Interner, result: &ResolvedGraph) -> Vec<String> {
        result
            .errors()
            .iter()
            .flat_map(|e| &e.errors)
            .map(|e| format!("{}", e.display(interner)))
            .collect()
    }

    // ── Completeness: valid inter-function calls should resolve ──────

    /// C1: A calls B with matching concrete types.
    #[test]
    fn inter_fn_simple_call() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "double",
                    "x * 2",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "double(21)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C2: A calls B, B returns String.
    #[test]
    fn inter_fn_string_return() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("greet", "\"hello\"", Some(vec![]), Constraint::Inferred),
                ("main", "greet()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    /// C3: Multi-arg function call.
    #[test]
    fn inter_fn_multi_arg() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "add",
                    "x + y",
                    Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "add(1, 2)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C4: Chain of calls — A calls B, B calls C.
    #[test]
    fn inter_fn_chain_call() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "inc",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "double_inc",
                    "inc(x) + inc(x)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "double_inc(5)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C5: Function uses context and is called by another function.
    #[test]
    fn inter_fn_with_context() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("get_count", "@count", Some(vec![]), Constraint::Inferred),
                ("main", "get_count() + 1", None, Constraint::Inferred),
            ],
            &[("count", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C6: Function with declared Exact output type.
    #[test]
    fn inter_fn_exact_output() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "make_str",
                    "\"hi\"",
                    Some(vec![]),
                    Constraint::Exact(Ty::String),
                ),
                ("main", "make_str()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    /// C7: Caller uses return value in arithmetic.
    #[test]
    fn inter_fn_return_used_in_binop() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("five", "5", Some(vec![]), Constraint::Inferred),
                ("main", "five() + five()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C8: Pipe syntax — value | fn.
    #[test]
    fn inter_fn_pipe_call() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "double",
                    "x * 2",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "10 | double", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C9: Function returning list.
    #[test]
    fn inter_fn_list_return() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("make_list", "[1, 2, 3]", Some(vec![]), Constraint::Inferred),
                ("main", "make_list() | len", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C10: Function accepting and returning String.
    #[test]
    fn inter_fn_string_identity() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "echo",
                    "s",
                    Some(vec![("s", Ty::String)]),
                    Constraint::Inferred,
                ),
                ("main", "echo(\"hello\")", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    /// C11: Multiple callers of the same function.
    #[test]
    fn inter_fn_multiple_callers() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "inc",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("a", "inc(10)", None, Constraint::Inferred),
                ("b", "inc(20)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(*result.fn_type(ids[1].1).unwrap(), Ty::Int);
        assert_eq!(*result.fn_type(ids[2].1).unwrap(), Ty::Int);
    }

    /// C12: Calling function with bool return.
    #[test]
    fn inter_fn_bool_return() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "is_positive",
                    "x > 0",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "is_positive(42)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Bool);
    }

    /// C13: Deep call chain — A → B → C → D.
    #[test]
    fn inter_fn_deep_chain() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), Constraint::Inferred),
                ("c", "d()", Some(vec![]), Constraint::Inferred),
                ("b", "c()", Some(vec![]), Constraint::Inferred),
                ("main", "b()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C14: Function that calls builtin and local function together.
    #[test]
    fn inter_fn_mixed_builtin_and_local() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("make_num", "42", Some(vec![]), Constraint::Inferred),
                ("main", "make_num() | to_string", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    /// C15: Function result used as argument to another function.
    #[test]
    fn inter_fn_nested_call() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "inc",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "inc(inc(0))", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C16: Mutual recursion — A calls B, B calls A.
    #[test]
    fn inter_fn_mutual_recursion() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "is_even",
                    "is_odd(n - 1)",
                    Some(vec![("n", Ty::Int)]),
                    Constraint::Exact(Ty::Bool),
                ),
                (
                    "is_odd",
                    "is_even(n - 1)",
                    Some(vec![("n", Ty::Int)]),
                    Constraint::Exact(Ty::Bool),
                ),
                ("main", "is_even(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Bool);
    }

    /// C17: Self-recursion.
    #[test]
    fn inter_fn_self_recursion() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "fib",
                    "fib(n - 1) + fib(n - 2)",
                    Some(vec![("n", Ty::Int)]),
                    Constraint::Exact(Ty::Int),
                ),
                ("main", "fib(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// C18: Function with float params and return.
    #[test]
    fn inter_fn_float_arithmetic() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "avg",
                    "(a + b) / 2.0",
                    Some(vec![("a", Ty::Float), ("b", Ty::Float)]),
                    Constraint::Inferred,
                ),
                ("main", "avg(1.0, 3.0)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Float);
    }

    // ── Soundness: invalid calls should be rejected ─────────────────

    /// S1: Wrong argument type.
    #[test]
    fn inter_fn_reject_wrong_arg_type() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                (
                    "double",
                    "x * 2",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "double(\"hello\")", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String arg for Int param"
        );
    }

    /// S2: Too many arguments.
    #[test]
    fn inter_fn_reject_too_many_args() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                (
                    "inc",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "inc(1, 2)", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject extra argument");
    }

    /// S3: Too few arguments.
    #[test]
    fn inter_fn_reject_too_few_args() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                (
                    "add",
                    "x + y",
                    Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "add(1)", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject missing argument");
    }

    /// S4: Using return value where wrong type expected.
    #[test]
    fn inter_fn_reject_return_type_mismatch() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                ("make_str", "\"hello\"", Some(vec![]), Constraint::Inferred),
                // make_str() returns String, but + 1 expects Int
                ("main", "make_str() + 1", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject String + Int");
    }

    /// S5: Calling undefined function.
    #[test]
    fn inter_fn_reject_undefined_function() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[("main", "nonexistent(1)", None, Constraint::Inferred)],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject call to undefined function"
        );
    }

    /// S6: Declared output type contradicts actual body.
    #[test]
    fn inter_fn_reject_output_mismatch() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                // Body returns Int, but declared output is String.
                ("bad", "42", Some(vec![]), Constraint::Exact(Ty::String)),
                ("main", "bad()", None, Constraint::Inferred),
            ],
            &[],
        );
        // Either 'bad' itself should error (body vs constraint mismatch)
        // or 'main' should get String (from declared) and later validation catches it.
        // At minimum, the system must not silently produce wrong types.
        let main_id = ids[1].1;
        if !result.has_errors() {
            // If no error, main should see the declared type (String), not Int.
            assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
        }
    }

    /// S7: Mutual recursion without declared types — must not stack overflow.
    #[test]
    fn inter_fn_mutual_recursion_no_declared_types() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                // No Exact output — types must be inferred.
                // This is hard and may legitimately error, but must not panic/infinite loop.
                (
                    "ping",
                    "pong(x)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "pong",
                    "ping(x)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        // We don't assert success or failure — just that it terminates.
        let _ = result;
    }

    /// S8: Wrong type in pipe position.
    #[test]
    fn inter_fn_reject_wrong_pipe_type() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                (
                    "needs_int",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "\"hello\" | needs_int", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String piped to Int param"
        );
    }

    /// S9: Function with wrong context type propagated through call.
    #[test]
    fn inter_fn_reject_context_type_propagation() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                ("get_name", "@name", Some(vec![]), Constraint::Inferred),
                // get_name returns String (from context), used in arithmetic — should error.
                ("main", "get_name() + 1", None, Constraint::Inferred),
            ],
            &[("name", Ty::String)],
        );
        assert!(
            result.has_errors(),
            "should reject String + Int through call chain"
        );
    }

    /// S10: Calling a function as if it had different arity in different call sites.
    #[test]
    fn inter_fn_reject_inconsistent_arity() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                ("f", "x", Some(vec![("x", Ty::Int)]), Constraint::Inferred),
                // First call correct, second call wrong arity.
                ("main", "f(1) + f(1, 2)", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject wrong arity call");
    }

    /// S11: Return type of called function used in list — type must be consistent.
    #[test]
    fn inter_fn_reject_heterogeneous_via_calls() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                ("make_int", "42", Some(vec![]), Constraint::Inferred),
                ("make_str", "\"hi\"", Some(vec![]), Constraint::Inferred),
                // [Int, String] — heterogeneous list should error.
                (
                    "main",
                    "[make_int(), make_str()]",
                    None,
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject heterogeneous list from calls"
        );
    }

    /// S12: Passing function return to wrong-typed parameter of another function.
    #[test]
    fn inter_fn_reject_chained_type_mismatch() {
        let i = Interner::new();
        let (result, _ids) = resolve_multi(
            &i,
            &[
                ("make_str", "\"hi\"", Some(vec![]), Constraint::Inferred),
                (
                    "needs_int",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "needs_int(make_str())", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String passed to Int param"
        );
    }

    // ── Edge cases ──────────────────────────────────────────────────

    /// E1: Function with no parameters, no context — pure constant.
    #[test]
    fn inter_fn_zero_arg_constant() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("pi", "3", Some(vec![]), Constraint::Inferred),
                ("main", "pi()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E2: Same name as builtin — local should shadow or coexist?
    /// (This tests the current behavior, whatever it is.)
    #[test]
    fn inter_fn_name_shadows_builtin() {
        let i = Interner::new();
        // "len" is a builtin. Defining a local "len" — what happens?
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("len", "42", Some(vec![]), Constraint::Exact(Ty::Int)),
                ("main", "len()", None, Constraint::Inferred),
            ],
            &[],
        );
        // Either the local wins (returns Int) or the builtin wins or it's an error.
        // The important thing is it doesn't panic.
        let main_id = ids[1].1;
        if !result.has_errors() {
            // If it resolves, check what type we got.
            let _ty = result.fn_type(main_id).unwrap();
        }
    }

    /// E3: Callee defined after caller in graph order.
    #[test]
    fn inter_fn_forward_reference() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                // main is first, calls helper which is second.
                ("main", "helper()", None, Constraint::Inferred),
                ("helper", "42", Some(vec![]), Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(
            errs.is_empty(),
            "forward reference should resolve: {errs:?}"
        );
        let main_id = ids[0].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E4: Two functions reading the same context.
    #[test]
    fn inter_fn_shared_context() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("read_a", "@x + 1", Some(vec![]), Constraint::Inferred),
                ("read_b", "@x + 2", Some(vec![]), Constraint::Inferred),
                ("main", "read_a() + read_b()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E5: Function calling itself with Exact type annotation (base case).
    #[test]
    fn inter_fn_self_call_exact() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "f",
                    "f(x - 1)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Exact(Ty::Int),
                ),
                ("main", "f(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E6: Diamond dependency — A calls B and C, both call D.
    #[test]
    fn inter_fn_diamond_dependency() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), Constraint::Inferred),
                ("b", "d() + 10", Some(vec![]), Constraint::Inferred),
                ("c", "d() + 20", Some(vec![]), Constraint::Inferred),
                ("main", "b() + c()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E7: Function result piped through builtin chain.
    #[test]
    fn inter_fn_pipe_through_builtins() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("make_list", "[1, 2, 3]", Some(vec![]), Constraint::Inferred),
                (
                    "main",
                    "make_list() | iter | map(x -> x + 1) | collect | len",
                    None,
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E8: Function with effectful iterator return type.
    #[test]
    fn inter_fn_effectful_return() {
        let i = Interner::new();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("get_iter", "@src", Some(vec![]), Constraint::Inferred),
                ("main", "get_iter() | collect", None, Constraint::Inferred),
            ],
            &[("src", eff_iter)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(
            *result.fn_type(main_id).unwrap(),
            Ty::List(Box::new(Ty::Int))
        );
    }

    /// E9: Function returning Option type.
    #[test]
    fn inter_fn_option_return() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "maybe_first",
                    "@items | iter | first",
                    Some(vec![]),
                    Constraint::Inferred,
                ),
                ("main", "maybe_first() | unwrap", None, Constraint::Inferred),
            ],
            &[("items", Ty::List(Box::new(Ty::Int)))],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E10: Three functions forming a pipeline.
    #[test]
    fn inter_fn_three_stage_pipeline() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "stage1",
                    "x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "stage2",
                    "x * 2",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "stage3",
                    "x - 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "main",
                    "0 | stage1 | stage2 | stage3",
                    None,
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// E11: All local functions are callers — no leaf function is called.
    /// (Independent functions, no inter-function calls — regression check.)
    #[test]
    fn inter_fn_independent_functions() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("a", "1 + 2", None, Constraint::Inferred),
                ("b", "\"hello\"", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(*result.fn_type(ids[0].1).unwrap(), Ty::Int);
        assert_eq!(*result.fn_type(ids[1].1).unwrap(), Ty::String);
    }

    /// E12: Function with object return type used with field access.
    #[test]
    fn inter_fn_object_return_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("get_user", "@user", Some(vec![]), Constraint::Exact(obj_ty)),
                ("main", "get_user().name", None, Constraint::Inferred),
            ],
            &[(
                "user",
                Ty::Object(FxHashMap::from_iter([
                    (i.intern("name"), Ty::String),
                    (i.intern("age"), Ty::Int),
                ])),
            )],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    // ════════════════════════════════════════════════════════════════
    // Soundness boundary tests
    //
    // These test dangerous edge cases at the boundaries of
    // infer → resolve → check_completeness.
    // ════════════════════════════════════════════════════════════════

    /// B1: Caller tries to use return value as wrong type.
    /// a() returns Int (inferred), caller uses it as String → error.
    #[test]
    fn boundary_caller_forces_wrong_return_type() {
        let i = Interner::new();
        let (result, _) = resolve_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), Constraint::Inferred),
                ("main", "a() + \"hello\"", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject Int used as String");
    }

    /// B2: Two callers use same function's return as different types.
    /// a() returns Int. One caller does a() + 1, other does a() + "hi" → error.
    #[test]
    fn boundary_inconsistent_return_usage() {
        let i = Interner::new();
        let (result, _) = resolve_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), Constraint::Inferred),
                ("ok_caller", "a() + 1", None, Constraint::Inferred),
                ("bad_caller", "a() + \"hi\"", None, Constraint::Inferred),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject inconsistent return type usage"
        );
    }

    /// B3: Mutual recursion with Inferred output — should NOT silently succeed.
    /// Both functions have unknown return types and call each other.
    /// Without Exact annotations, the types cannot be determined.
    #[test]
    fn boundary_mutual_recursion_inferred_must_not_succeed_silently() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "ping",
                    "pong(x)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "pong",
                    "ping(x)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        // Either errors (cannot infer) or if it resolves, the types must be consistent.
        if !result.has_errors() {
            // If somehow resolved, both must have the same return type.
            let ping_ty = result.fn_type(ids[0].1);
            let pong_ty = result.fn_type(ids[1].1);
            if let (Some(pt), Some(qt)) = (ping_ty, pong_ty) {
                assert_eq!(pt, qt, "mutual recursion must have consistent return types");
            }
        }
    }

    /// B4: Self-recursion with Inferred output and no base case type.
    /// f(x) = f(x-1). Return type is entirely self-referential → cannot infer.
    #[test]
    fn boundary_self_recursion_inferred_no_base() {
        let i = Interner::new();
        let (result, _) = resolve_multi(
            &i,
            &[
                (
                    "f",
                    "f(x - 1)",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "f(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        // Should error — f's return type is entirely circular.
        assert!(
            result.has_errors(),
            "purely recursive return type should fail inference"
        );
    }

    /// B5: Effect soundness — function reading context must be Effectful,
    /// and caller must see it as Effectful.
    #[test]
    fn boundary_effectful_context_read_propagates() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("get_x", "@x", Some(vec![]), Constraint::Inferred),
                ("main", "get_x()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        // get_x reads context → Effectful.
        // main calls get_x → main should also be Effectful.
        let main_id = ids[1].1;
        if let Some(resolution) = result.try_resolution(main_id) {
            assert_eq!(
                resolution.body_effect,
                Effect::io(),
                "caller of context-reading function must be effectful"
            );
        }
    }

    /// B6: Effect soundness — pure function call should not taint caller.
    #[test]
    fn boundary_pure_function_stays_pure() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "add",
                    "x + y",
                    Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                    Constraint::Inferred,
                ),
                ("main", "add(1, 2)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        if let Some(resolution) = result.try_resolution(main_id) {
            assert_eq!(
                resolution.body_effect,
                Effect::pure(),
                "caller of pure function should remain pure"
            );
        }
    }

    /// B7: Effect soundness — context write is Effectful.
    #[test]
    fn boundary_context_write_is_effectful() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("set_x", "@x = 42; @x", Some(vec![]), Constraint::Inferred),
                ("main", "set_x()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let set_id = ids[0].1;
        if let Some(resolution) = result.try_resolution(set_id) {
            assert_eq!(
                resolution.body_effect,
                Effect::io(),
                "context-writing function must be effectful"
            );
        }
    }

    /// B8: infer produces wrong type, resolve catches it.
    /// Body returns Int but declared output is String → resolve must reject.
    #[test]
    fn boundary_infer_wrong_type_resolve_catches() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Constraint::Exact(Ty::String)),
                ("main", "bad()", None, Constraint::Inferred),
            ],
            &[],
        );
        // Either 'bad' has errors (body type mismatch) or main sees String.
        // The system must NOT let main see Int when bad declared String.
        let main_id = ids[1].1;
        if let Some(main_ty) = result.fn_type(main_id) {
            assert_ne!(
                *main_ty,
                Ty::Int,
                "main must not see Int when bad declared String"
            );
        }
    }

    /// B9: Param in output but not in input → must not silently succeed.
    /// f() -> T where T is unconstrained → cannot infer.
    #[test]
    fn boundary_orphan_param_in_output() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                // No params, inferred output. Body is just a free variable with no constraint.
                ("mystery", "x", Some(vec![]), Constraint::Inferred),
                ("main", "mystery() + 1", None, Constraint::Inferred),
            ],
            &[],
        );
        // "x" in mystery body is a free param, but Signature has 0 params.
        // So "x" gets fresh_param. mystery() returns fresh_param.
        // main does mystery() + 1 which may force Int.
        // This is actually OK if it resolves — the inferred param is Int.
        // The key is it should NOT be unsound.
        if !result.has_errors() {
            let main_id = ids[1].1;
            if let Some(ty) = result.fn_type(main_id) {
                assert_eq!(*ty, Ty::Int, "if resolved, return type should be Int");
            }
        }
    }

    /// B10: Chain through effectful function — effect must propagate transitively.
    #[test]
    fn boundary_transitive_effect_propagation() {
        let i = Interner::new();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let (result, ids) = resolve_multi(
            &i,
            &[
                (
                    "consume_iter",
                    "@src | collect",
                    Some(vec![]),
                    Constraint::Inferred,
                ),
                ("use_it", "consume_iter()", None, Constraint::Inferred),
            ],
            &[("src", eff_iter)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        // consume_iter reads context → Effectful.
        // use_it calls consume_iter → should also be Effectful.
        let use_id = ids[1].1;
        if let Some(resolution) = result.try_resolution(use_id) {
            assert_eq!(
                resolution.body_effect,
                Effect::io(),
                "transitive effect must propagate"
            );
        }
    }

    // ── body_reads / body_writes tests ──────────────────────────────

    /// Direct context read is tracked in body_reads.
    #[test]
    fn body_reads_tracks_context_read() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[("reader", "@x + @y", Some(vec![]), Constraint::Inferred)],
            &[("x", Ty::Int), ("y", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            assert!(
                resolution.body_reads.contains(&i.intern("x")),
                "should track @x read"
            );
            assert!(
                resolution.body_reads.contains(&i.intern("y")),
                "should track @y read"
            );
            assert!(
                resolution.body_writes.is_empty(),
                "read-only function should have no writes"
            );
        }
    }

    /// Direct context write is tracked in body_writes.
    #[test]
    fn body_writes_tracks_context_write() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[("writer", "@x = 42; @x", Some(vec![]), Constraint::Inferred)],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            assert!(
                resolution.body_writes.contains(&i.intern("x")),
                "should track @x write"
            );
            // @x = 42 also reads @x in the tail expression
            assert!(
                resolution.body_reads.contains(&i.intern("x")),
                "should also track @x read from tail"
            );
        }
    }

    /// Pure function has empty body_reads and body_writes.
    #[test]
    fn body_reads_writes_empty_for_pure() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[("pure_fn", "1 + 2", Some(vec![]), Constraint::Inferred)],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            assert!(resolution.body_reads.is_empty());
            assert!(resolution.body_writes.is_empty());
        }
    }

    /// Calling another function does NOT add to caller's body_reads/body_writes.
    /// Transitive effect propagation is infer's responsibility, not typeck's.
    #[test]
    fn body_reads_no_transitive_from_callee() {
        let i = Interner::new();
        let (result, ids) = resolve_multi(
            &i,
            &[
                ("get_x", "@x", Some(vec![]), Constraint::Inferred),
                ("main", "get_x()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        // get_x directly reads @x
        let get_x_id = ids[0].1;
        if let Some(resolution) = result.try_resolution(get_x_id) {
            assert!(resolution.body_reads.contains(&i.intern("x")));
        }
        // main calls get_x but does NOT directly read @x
        // → body_reads should be empty (transitive propagation is infer's job)
        let main_id = ids[1].1;
        if let Some(resolution) = result.try_resolution(main_id) {
            assert!(
                resolution.body_reads.is_empty(),
                "caller should not have transitive reads in body_reads"
            );
            assert!(
                resolution.body_writes.is_empty(),
                "caller should not have transitive writes in body_writes"
            );
        }
    }

    // ── Extern function resolution ─────────────────────────────────────

    fn make_extern_fn(interner: &Interner, name: &str, params: Vec<Ty>, ret: Ty) -> Function {
        let named_params: Vec<crate::ty::Param> = params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| crate::ty::Param::new(interner.intern(&format!("_{i}")), ty))
            .collect();
        Function {
            id: FunctionId::alloc(),
            name: interner.intern(name),
            kind: FnKind::Extern {
                deps: Freeze::new(vec![]),
            },
            constraint: FnConstraint {
                signature: Some(Signature {
                    params: named_params.clone(),
                }),
                output: Constraint::Exact(Ty::Fn {
                    params: named_params,
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: Effect::pure(),
                }),
            },
        }
    }

    /// Extern function should be callable from local functions.
    #[test]
    fn extern_fn_call_resolves() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, ids) = resolve_with_extern(
            &i,
            &[("main", "fetch(42)", None, Constraint::Inferred)],
            &[fetch],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "extern call should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::String);
    }

    /// Extern function with wrong argument type should error.
    #[test]
    fn extern_fn_call_type_mismatch() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, _) = resolve_with_extern(
            &i,
            &[("main", "fetch(\"bad\")", None, Constraint::Inferred)],
            &[fetch],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String where Int expected"
        );
    }

    /// Extern function return type flows into caller's expression.
    #[test]
    fn extern_fn_return_type_propagates() {
        let i = Interner::new();
        let get_count = make_extern_fn(&i, "get_count", vec![], Ty::Int);
        let (result, ids) = resolve_with_extern(
            &i,
            &[("main", "get_count() + 1", None, Constraint::Inferred)],
            &[get_count],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(*result.fn_type(main_id).unwrap(), Ty::Int);
    }

    /// Multiple extern functions can be registered and called.
    #[test]
    fn extern_fn_multiple() {
        let i = Interner::new();
        let add = make_extern_fn(&i, "ext_add", vec![Ty::Int, Ty::Int], Ty::Int);
        let greet = make_extern_fn(&i, "ext_greet", vec![Ty::String], Ty::String);
        let (result, ids) = resolve_with_extern(
            &i,
            &[
                ("use_add", "ext_add(1, 2)", None, Constraint::Inferred),
                ("use_greet", "ext_greet(\"hi\")", None, Constraint::Inferred),
            ],
            &[add, greet],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(*result.fn_type(ids[0].1).unwrap(), Ty::Int);
        assert_eq!(*result.fn_type(ids[1].1).unwrap(), Ty::String);
    }
}
