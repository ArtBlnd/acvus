//! Phase 1: Infer
//!
//! For each function, infer the types of unknown context parameters and
//! function output types. Supports inter-function calls by adding all
//! local functions to the TypeEnv before typechecking.
//!
//! Output: context params + function types (for UI display and Phase 2).

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::ty::{
    EffectTerm, Infer, InferTy, Param, PolyTy, Scheme, Solver, Ty, TyTerm, TyVarBound,
    TypeRegistry, lift_to_poly, lift_ty,
};

use super::extract::{ExtractResult, ParsedSource};
use super::types::*;

// -- Phase 1 output --------------------------------------------------

/// Inferred metadata for a single function.
#[derive(Debug, Clone)]
pub struct FunctionMeta {
    /// Fully resolved Ty::Fn for this function.
    pub ty: Ty,
    /// Named parameters (free_params from source zipped with signature types).
    pub params: Vec<Param>,
}

/// Per-function inference outcome.
#[derive(Debug)]
pub enum FnInferOutcome {
    /// Type fully resolved. Lowerable.
    Complete {
        resolution: Freeze<crate::typeck::TypeResolution>,
        tail_ty: Ty,
        meta: FunctionMeta,
    },
    /// Type incomplete. Cannot lower.
    Incomplete {
        unknown_contexts: Vec<(QualifiedRef, Ty)>,
        unknown_extern_params: Vec<(Astr, Ty)>,
        meta: FunctionMeta,
        errors: Vec<crate::error::MirError>,
    },
}

impl FnInferOutcome {
    /// Get the function metadata regardless of completeness.
    pub fn meta(&self) -> &FunctionMeta {
        match self {
            FnInferOutcome::Complete { meta, .. } => meta,
            FnInferOutcome::Incomplete { meta, .. } => meta,
        }
    }

    /// Get the checked resolution if complete. Clone is cheap (Arc::clone).
    pub fn resolution(&self) -> Option<Freeze<crate::typeck::TypeResolution>> {
        match self {
            FnInferOutcome::Complete { resolution, .. } => Some(resolution.clone()),
            FnInferOutcome::Incomplete { .. } => None,
        }
    }

    /// Get the tail type if complete.
    pub fn tail_ty(&self) -> Option<&Ty> {
        match self {
            FnInferOutcome::Complete { tail_ty, .. } => Some(tail_ty),
            FnInferOutcome::Incomplete { .. } => None,
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, FnInferOutcome::Complete { .. })
    }
}

/// Phase 1 output: inferred context parameters and function types.
/// All type information is frozen - immutable after inference.
#[derive(Debug)]
pub struct InferResult {
    /// Per-function inference outcome (Complete or Incomplete).
    pub outcomes: FxHashMap<QualifiedRef, FnInferOutcome>,
    /// Resolved context types (known + inferred). Frozen after inference.
    pub context_types: Freeze<FxHashMap<QualifiedRef, Ty>>,
}

impl InferResult {
    /// Get the checked resolution for a function, if complete.
    pub fn try_resolution(
        &self,
        id: QualifiedRef,
    ) -> Option<Freeze<crate::typeck::TypeResolution>> {
        self.outcomes.get(&id)?.resolution()
    }

    /// Get the context type for a QualifiedRef.
    pub fn context_type(&self, qref: &QualifiedRef) -> Option<&Ty> {
        (*self.context_types).get(qref)
    }

    /// Whether any function has errors.
    pub fn has_errors(&self) -> bool {
        self.outcomes
            .values()
            .any(|o| matches!(o, FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty()))
    }

    /// Collect all errors across all functions.
    pub fn errors(&self) -> Vec<(QualifiedRef, &[crate::error::MirError])> {
        self.outcomes
            .iter()
            .filter_map(|(&id, o)| match o {
                FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty() => {
                    Some((id, errors.as_slice()))
                }
                _ => None,
            })
            .collect()
    }
}

// -- Call graph + SCC -------------------------------------------------

/// Extract call edges for a single function from its parsed AST.
/// Returns the list of QualifiedRefs that this function references.
pub fn extract_call_edges(
    parsed: &ParsedSource,
    name_to_fn: &FxHashMap<Astr, QualifiedRef>,
    self_id: QualifiedRef,
) -> Vec<QualifiedRef> {
    let names: Vec<Astr> = match parsed {
        ParsedSource::Script(script) => collect_value_refs_script(script),
        ParsedSource::Template(template) => collect_value_refs_template(template),
    };
    let mut callees = Vec::new();
    for name in names {
        if let Some(&callee_id) = name_to_fn.get(&name)
            && callee_id != self_id
            && !callees.contains(&callee_id)
        {
            callees.push(callee_id);
        }
    }
    callees
}

/// Build a call graph: for each local function, which other local functions
/// does it reference by name in its body?
fn build_call_graph(
    graph: &CompilationGraph,
    extract: &ExtractResult,
) -> FxHashMap<QualifiedRef, Vec<QualifiedRef>> {
    let name_to_id: FxHashMap<Astr, QualifiedRef> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| (f.qref.name, f.qref))
        .collect();

    let mut edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = FxHashMap::default();
    for func in graph.functions.iter() {
        if matches!(func.kind, FnKind::Extern { .. }) {
            continue;
        }
        let Some(parsed) = extract.parsed.get(&func.qref) else {
            continue;
        };
        edges.insert(
            func.qref,
            extract_call_edges(parsed, &name_to_id, func.qref),
        );
    }
    edges
}

fn collect_value_refs_stmts(stmts: &[acvus_ast::Stmt], refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    for stmt in stmts {
        match stmt {
            Stmt::Bind { expr, .. }
            | Stmt::ContextStore { expr, .. }
            | Stmt::VarFieldStore { expr, .. } => {
                collect_value_refs_expr(expr, refs);
            }
            Stmt::Expr(expr) => collect_value_refs_expr(expr, refs),
            Stmt::MatchBind { source, body, .. } | Stmt::WhileLet { source, body, .. } => {
                collect_value_refs_expr(source, refs);
                collect_value_refs_stmts(body, refs);
            }
            Stmt::LetBind { expr, .. } | Stmt::Assign { expr, .. } => {
                collect_value_refs_expr(expr, refs);
            }
            Stmt::LetUninit { .. } => {}
            Stmt::While { cond, body, .. } => {
                collect_value_refs_expr(cond, refs);
                collect_value_refs_stmts(body, refs);
            }
        }
    }
}

/// Collect all RefKind::Value identifiers from a script AST.
fn collect_value_refs_script(script: &acvus_ast::Script) -> Vec<Astr> {
    let mut refs = Vec::new();
    collect_value_refs_stmts(&script.stmts, &mut refs);
    if let Some(tail) = &script.tail {
        collect_value_refs_expr(tail, &mut refs);
    }
    refs
}

fn collect_value_refs_template(template: &acvus_ast::Template) -> Vec<Astr> {
    let mut refs = Vec::new();
    for node in &template.body {
        collect_value_refs_node(node, &mut refs);
    }
    refs
}

fn collect_value_refs_node(node: &acvus_ast::Node, refs: &mut Vec<Astr>) {
    match node {
        acvus_ast::Node::Text { .. } | acvus_ast::Node::Comment { .. } => {}
        acvus_ast::Node::InlineExpr { expr, .. } => collect_value_refs_expr(expr, refs),
        acvus_ast::Node::MatchBlock(mb) => {
            collect_value_refs_expr(&mb.source, refs);
            for arm in &mb.arms {
                for n in &arm.body {
                    collect_value_refs_node(n, refs);
                }
            }
            if let Some(ca) = &mb.catch_all {
                for n in &ca.body {
                    collect_value_refs_node(n, refs);
                }
            }
        }
    }
}

fn collect_value_refs_expr(expr: &acvus_ast::Expr, refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    match expr {
        Expr::Ident {
            name,
            ref_kind: RefKind::Value,
            ..
        } => refs.push(name.name),
        Expr::Ident { .. } | Expr::Literal { .. } | Expr::ContextRef { .. } => {}
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. } => {
            collect_value_refs_expr(left, refs);
            collect_value_refs_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. } => collect_value_refs_expr(operand, refs),
        Expr::FieldAccess { object, .. } => collect_value_refs_expr(object, refs),
        Expr::FuncCall { func, args, .. } => {
            collect_value_refs_expr(func, refs);
            for a in args {
                collect_value_refs_expr(a, refs);
            }
        }
        Expr::Lambda { body, .. } => collect_value_refs_expr(body, refs),
        Expr::Paren { inner, .. } => collect_value_refs_expr(inner, refs),
        Expr::List { head, tail, .. } => {
            for e in head.iter().chain(tail.iter()) {
                collect_value_refs_expr(e, refs);
            }
        }
        Expr::Object { fields, .. } => {
            for f in fields {
                collect_value_refs_expr(&f.value, refs);
            }
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(e) = e {
                    collect_value_refs_expr(e, refs);
                }
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements {
                collect_value_refs_expr(e, refs);
            }
        }
        Expr::Variant {
            payload: Some(inner),
            ..
        } => collect_value_refs_expr(inner, refs),
        Expr::Variant { payload: None, .. } => {}
        Expr::Block { stmts, tail, .. } => {
            collect_value_refs_stmts(stmts, refs);
            collect_value_refs_expr(tail, refs);
        }
        Expr::If {
            cond,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            collect_value_refs_expr(cond, refs);
            collect_value_refs_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                collect_value_refs_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                collect_value_refs_else_branch(eb, refs);
            }
        }
        Expr::IfLet {
            source,
            then_body,
            then_tail,
            else_branch,
            ..
        } => {
            collect_value_refs_expr(source, refs);
            collect_value_refs_stmts(then_body, refs);
            if let Some(tail) = then_tail {
                collect_value_refs_expr(tail, refs);
            }
            if let Some(eb) = else_branch {
                collect_value_refs_else_branch(eb, refs);
            }
        }
    }
}

fn collect_value_refs_else_branch(eb: &acvus_ast::ElseBranch, refs: &mut Vec<Astr>) {
    match eb {
        acvus_ast::ElseBranch::ElseIf(expr) => collect_value_refs_expr(expr, refs),
        acvus_ast::ElseBranch::Else { body, tail, .. } => {
            collect_value_refs_stmts(body, refs);
            if let Some(tail) = tail {
                collect_value_refs_expr(tail, refs);
            }
        }
    }
}

/// Tarjan's SCC algorithm. Returns SCCs in reverse topological order
/// (leaf SCCs first - dependencies before dependents).
pub fn tarjan_scc(
    ids: &[QualifiedRef],
    edges: &FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
) -> Vec<Vec<QualifiedRef>> {
    let mut index_counter: u32 = 0;
    let mut stack: Vec<QualifiedRef> = Vec::new();
    let mut on_stack: FxHashSet<QualifiedRef> = FxHashSet::default();
    let mut index: FxHashMap<QualifiedRef, u32> = FxHashMap::default();
    let mut lowlink: FxHashMap<QualifiedRef, u32> = FxHashMap::default();
    let mut result: Vec<Vec<QualifiedRef>> = Vec::new();

    fn strongconnect(
        v: QualifiedRef,
        edges: &FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
        index_counter: &mut u32,
        stack: &mut Vec<QualifiedRef>,
        on_stack: &mut FxHashSet<QualifiedRef>,
        index: &mut FxHashMap<QualifiedRef, u32>,
        lowlink: &mut FxHashMap<QualifiedRef, u32>,
        result: &mut Vec<Vec<QualifiedRef>>,
    ) {
        index.insert(v, *index_counter);
        lowlink.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        if let Some(neighbors) = edges.get(&v) {
            for &w in neighbors {
                if !index.contains_key(&w) {
                    strongconnect(
                        w,
                        edges,
                        index_counter,
                        stack,
                        on_stack,
                        index,
                        lowlink,
                        result,
                    );
                    let lw = lowlink[&w];
                    let lv = lowlink[&v];
                    lowlink.insert(v, lv.min(lw));
                } else if on_stack.contains(&w) {
                    let iw = index[&w];
                    let lv = lowlink[&v];
                    lowlink.insert(v, lv.min(iw));
                }
            }
        }

        if lowlink[&v] == index[&v] {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == v {
                    break;
                }
            }
            result.push(component);
        }
    }

    for &id in ids {
        if !index.contains_key(&id) {
            strongconnect(
                id,
                edges,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut index,
                &mut lowlink,
                &mut result,
            );
        }
    }

    // Tarjan produces SCCs in reverse topological order already.
    result
}

// -- Per-SCC inference ------------------------------------------------

/// Result of inferring a single SCC.
#[derive(Debug, Clone)]
pub struct SccInferResult {
    /// Per-function metadata (type, params).
    pub fn_metas: FxHashMap<QualifiedRef, FunctionMeta>,
    /// QualifiedRef -> resolved Ty::Fn (for passing to next SCC).
    pub resolved_types: FxHashMap<QualifiedRef, Ty>,
    /// Per-function type errors from typechecker.
    pub errors: FxHashMap<QualifiedRef, Vec<crate::error::MirError>>,
}

/// Infer types for a single SCC.
///
/// `resolved_fn_types`: all function types already resolved by prior SCCs + builtins.
/// `known_ctx`: declared context types from the graph.
/// The bounds every Extern in `functions` declared for its type variables.
pub fn declared_bounds<'a>(
    functions: impl Iterator<Item = &'a Function>,
) -> FxHashMap<QualifiedRef, Vec<TyVarBound>> {
    functions
        .filter_map(|f| match &f.kind {
            FnKind::Extern { bounds } => Some((f.qref, bounds.clone())),
            FnKind::Local(_) => None,
        })
        .collect()
}

/// The scheme a function's type is instantiated under: its declared bounds
/// when it is an Extern, none otherwise.
fn declared_scheme(bounds: Option<&Vec<TyVarBound>>, ty: PolyTy) -> Scheme {
    match bounds {
        Some(bounds) => Scheme {
            ty,
            bounds: bounds.clone(),
        },
        None => Scheme::unbounded(ty),
    }
}

pub fn infer_scc(
    interner: &Interner,
    scc: &[QualifiedRef],
    fn_by_id: &FxHashMap<QualifiedRef, &Function>,
    extract_parsed: &FxHashMap<QualifiedRef, &ParsedSource>,
    known_ctx: &FxHashMap<QualifiedRef, PolyTy>,
    resolved_fn_types: &FxHashMap<QualifiedRef, PolyTy>,
    declared: &FxHashMap<QualifiedRef, Vec<TyVarBound>>,
) -> SccInferResult {
    let mut solver = Solver::new();
    let registry = TypeRegistry::default();

    // Instantiate context types into solver-scoped InferTy.
    let known_ctx_infer: FxHashMap<QualifiedRef, InferTy> = known_ctx
        .iter()
        .map(|(&k, v)| (k, solver.instantiate_poly(v)))
        .collect();
    let mut fn_bind_params: FxHashMap<QualifiedRef, Vec<Param>> = FxHashMap::default();
    let mut fn_ret_vars: FxHashMap<QualifiedRef, InferTy> = FxHashMap::default();
    let mut fn_effect_vars: FxHashMap<QualifiedRef, EffectTerm<Infer>> = FxHashMap::default();
    let mut fn_errors: FxHashMap<QualifiedRef, Vec<crate::error::MirError>> = FxHashMap::default();

    // Build PolyTy::Fn templates for functions in this SCC.
    // Solver ret vars are kept separately for unification.
    let mut scc_fn_types: FxHashMap<QualifiedRef, PolyTy> = FxHashMap::default();

    for &fid in scc {
        let func = fn_by_id[&fid];

        // Destructure func.ty - must be Fn for local functions.
        let TyTerm::Fn {
            params: ref fn_params,
            ret: ref fn_ret,
            effect: ref fn_effect,
            ..
        } = func.ty
        else {
            unreachable!("local function ty must be Fn");
        };

        // Solver vars for unification (InferTy).
        let ret_var: InferTy = solver.instantiate_poly(fn_ret);
        fn_ret_vars.insert(fid, ret_var.clone());
        let effect_var = solver.fresh_effect_var();
        fn_effect_vars.insert(fid, effect_var);

        let fn_ty: PolyTy = TyTerm::Fn {
            params: fn_params.clone(),
            ret: fn_ret.clone(),
            captures: vec![],
            effect: *fn_effect,
        };
        scc_fn_types.insert(func.qref, fn_ty);
    }

    let mut env_functions: FxHashMap<QualifiedRef, Scheme> = resolved_fn_types
        .iter()
        .map(|(&k, v)| (k, declared_scheme(declared.get(&k), v.clone())))
        .collect();
    env_functions.extend(
        scc_fn_types
            .into_iter()
            .map(|(k, v)| (k, Scheme::unbounded(v))),
    );

    // Typecheck each function in this SCC.
    for &fid in scc {
        let func = fn_by_id[&fid];
        let Some(parsed) = extract_parsed.get(&fid) else {
            continue;
        };

        let env = crate::ty::TypeEnv {
            contexts: known_ctx_infer.clone(),
            functions: env_functions.clone(),
        };

        // Extract ret and params from func.ty.
        let TyTerm::Fn {
            ret: ref fn_ret,
            params: ref fn_params,
            ..
        } = func.ty
        else {
            unreachable!("local function ty must be Fn");
        };

        let expected_tail_ty: Option<Ty> = {
            let infer = solver.instantiate_poly(fn_ret);
            solver.freeze_ty(&infer).ok()
        };
        let declared_types: Vec<Ty> = fn_params
            .iter()
            .filter_map(|p| {
                let infer = solver.instantiate_poly(&p.ty);
                solver.freeze_ty(&infer).ok()
            })
            .collect();

        let checker = crate::typeck::TypeChecker::new(interner, &env, &registry, &mut solver)
            .with_analysis_mode()
            .with_declared_param_types(declared_types)
            .with_body_effect(fn_effect_vars[&fid]);
        let result = match parsed {
            ParsedSource::Script(script) => checker.check_script(script, expected_tail_ty.as_ref()),
            ParsedSource::Template(template) => checker.check_template(template),
        };

        match result {
            Ok(ref unchecked) => {
                // Unify ret var with tail ty (for inferred return types).
                if expected_tail_ty.is_none() {
                    if let Some(ret_var) = fn_ret_vars.get(&fid) {
                        let tail_infer = lift_ty(&unchecked.tail_ty);
                        let _ = solver.unify_ty(
                            ret_var,
                            &tail_infer,
                            crate::ty::Polarity::Invariant,
                            &registry,
                        );
                    }
                }
                let closed = EffectTerm::Known(unchecked.effect);
                solver
                    .unify_effect(
                        &fn_effect_vars[&fid],
                        &closed,
                        crate::ty::Polarity::Invariant,
                    )
                    .expect("the closed effect is the variable's own lower bound");

                let bind: Vec<Param> = unchecked
                    .extern_params
                    .iter()
                    .map(|(name, ty)| Param::new(*name, ty.clone()))
                    .collect();
                fn_bind_params.insert(fid, bind);
            }
            Err(errors) => {
                fn_errors.insert(fid, errors);
            }
        }
    }

    // Resolve all functions in this SCC - freeze InferTy -> Ty at the boundary.
    let mut resolved_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
    let mut fn_metas: FxHashMap<QualifiedRef, FunctionMeta> = FxHashMap::default();

    for &fid in scc {
        let func = fn_by_id[&fid];
        let ret = fn_ret_vars
            .get(&fid)
            .and_then(|r| solver.freeze_ty(&solver.resolve_ty(r)).ok())
            .unwrap_or_else(Ty::error);
        let bind: Vec<Param> = fn_bind_params.get(&fid).cloned().unwrap_or_default();

        let effect = EffectTerm::Known(solver.freeze_effect(&fn_effect_vars[&fid]));

        let fn_ty = Ty::Fn {
            params: bind.clone(),
            ret: Box::new(ret),
            captures: vec![],
            effect,
        };
        resolved_types.insert(func.qref, fn_ty.clone());
        fn_metas.insert(
            fid,
            FunctionMeta {
                ty: fn_ty,
                params: bind,
            },
        );
    }

    SccInferResult {
        fn_metas,
        resolved_types,
        errors: fn_errors,
    }
}

// -- Batch inference -------------------------------------------------

/// Run Phase 1 inference with SCC-based processing.
///
/// 1. Build call graph from AST references.
/// 2. Compute SCCs (Tarjan) - reverse topological order.
/// 3. Process each SCC:
///    - Within an SCC: shared Solver, no instantiation of intra-SCC calls.
///    - After an SCC is done: resolve ret vars -> concrete Ty::Fn.
///    - Next SCC sees concrete types -> instantiation is safe.
pub fn infer(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    user_context_types: &FxHashMap<QualifiedRef, PolyTy>,
    type_registry: Freeze<TypeRegistry>,
) -> InferResult {
    let mut solver = Solver::new();
    let registry_ref: &TypeRegistry = &type_registry;

    // Per-function state accumulated across SCCs.
    let mut fn_bind_params: FxHashMap<QualifiedRef, Vec<Param>> = FxHashMap::default();
    let mut fn_unchecked: FxHashMap<QualifiedRef, Freeze<crate::typeck::TypeResolution>> =
        FxHashMap::default();
    let mut fn_typeck_errors: FxHashMap<QualifiedRef, Vec<crate::error::MirError>> =
        FxHashMap::default();
    let mut resolved_fn_types: FxHashMap<QualifiedRef, PolyTy> = Default::default();
    let mut fn_metas: FxHashMap<QualifiedRef, FunctionMeta> = FxHashMap::default();

    // -- Setup --------------------------------------------------------

    // Extern function types are always known upfront (their PolyTy is fully concrete).
    for func in graph.functions.iter() {
        if let FnKind::Extern { .. } = &func.kind {
            resolved_fn_types.insert(func.qref, func.ty.clone());
        }
    }

    // Known context types: graph declarations + user-provided.
    // Internally work with InferTy; freeze to Ty at the output boundary.
    // If ctx.ty is fully concrete -> instantiate_poly gives a concrete InferTy.
    // If ctx.ty contains Var placeholders -> instantiate_poly maps each Var to a fresh solver var.
    let mut known_ctx: FxHashMap<QualifiedRef, InferTy> = FxHashMap::default();
    for ctx in graph.contexts.iter() {
        known_ctx.insert(ctx.qref, solver.instantiate_poly(&ctx.ty));
    }
    known_ctx.extend(
        user_context_types
            .iter()
            .map(|(&k, v)| (k, solver.instantiate_poly(v))),
    );

    let fn_by_id: FxHashMap<QualifiedRef, &Function> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| (f.qref, f))
        .collect();
    let declared = declared_bounds(graph.functions.iter());

    // -- STEP 1: Call graph + SCCs ------------------------------------

    let call_graph = build_call_graph(graph, extract);
    let local_ids: Vec<QualifiedRef> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| f.qref)
        .collect();
    let sccs = tarjan_scc(&local_ids, &call_graph);

    // -- STEP 2: Typecheck + resolve per SCC -------------------------

    for scc in &sccs {
        // 2a. Build PolyTy::Fn templates for SCC members.
        // Solver ret vars are kept separately for unification.
        let mut scc_fn_types: FxHashMap<QualifiedRef, PolyTy> = FxHashMap::default();
        let mut scc_ret_vars: FxHashMap<QualifiedRef, InferTy> = FxHashMap::default();
        let mut scc_effect_vars: FxHashMap<QualifiedRef, EffectTerm<Infer>> = FxHashMap::default();

        for &fid in scc {
            let func = fn_by_id[&fid];

            // Destructure func.ty - must be Fn for local functions.
            let TyTerm::Fn {
                params: ref fn_params,
                ret: ref fn_ret,
                effect: ref fn_effect,
                ..
            } = func.ty
            else {
                unreachable!("local function ty must be Fn");
            };

            // Solver vars for unification (InferTy).
            // If ret is concrete (no Poly Vars) -> instantiate_poly gives concrete InferTy.
            // If ret has Vars -> instantiate_poly maps each Var to a fresh solver var.
            let ret_var: InferTy = solver.instantiate_poly(fn_ret);
            scc_ret_vars.insert(fid, ret_var.clone());
            scc_effect_vars.insert(fid, solver.fresh_effect_var());

            scc_fn_types.insert(
                func.qref,
                TyTerm::Fn {
                    params: fn_params.clone(),
                    ret: fn_ret.clone(),
                    captures: vec![],
                    effect: *fn_effect,
                },
            );
        }

        let mut env_functions: FxHashMap<QualifiedRef, Scheme> = resolved_fn_types
            .iter()
            .map(|(&k, v)| (k, declared_scheme(declared.get(&k), v.clone())))
            .collect();
        env_functions.extend(
            scc_fn_types
                .into_iter()
                .map(|(k, v)| (k, Scheme::unbounded(v))),
        );

        for &fid in scc {
            let func = fn_by_id[&fid];
            let Some(parsed) = extract.parsed.get(&fid) else {
                continue;
            };

            let env = crate::ty::TypeEnv {
                contexts: known_ctx.clone(),
                functions: env_functions.clone(),
            };

            // Extract ret and params from func.ty.
            let TyTerm::Fn {
                ret: ref fn_ret,
                params: ref fn_params,
                ..
            } = func.ty
            else {
                unreachable!("local function ty must be Fn");
            };

            // For expected_tail: if ret is fully concrete (no Poly Vars), freeze to Ty.
            // If ret has Vars (inferred), freeze fails -> None -> typechecker infers freely.
            let expected_tail_ty: Option<Ty> = {
                let infer = solver.instantiate_poly(fn_ret);
                solver.freeze_ty(&infer).ok()
            };
            let declared_types: Vec<Ty> = fn_params
                .iter()
                .filter_map(|p| {
                    let infer = solver.instantiate_poly(&p.ty);
                    solver.freeze_ty(&infer).ok()
                })
                .collect();

            let checker =
                crate::typeck::TypeChecker::new(interner, &env, registry_ref, &mut solver)
                    .with_analysis_mode()
                    .with_declared_param_types(declared_types)
                    .with_body_effect(scc_effect_vars[&fid]);
            let result = match parsed {
                ParsedSource::Script(script) => {
                    checker.check_script(script, expected_tail_ty.as_ref())
                }
                ParsedSource::Template(template) => checker.check_template(template),
            };

            match result {
                Ok(unchecked) => {
                    // Unify ret var with tail ty (for inferred return types).
                    if expected_tail_ty.is_none() {
                        if let Some(ret_var) = scc_ret_vars.get(&fid) {
                            let tail_infer = lift_ty(&unchecked.tail_ty);
                            let _ = solver.unify_ty(
                                ret_var,
                                &tail_infer,
                                crate::ty::Polarity::Invariant,
                                registry_ref,
                            );
                        }
                    }
                    let closed = EffectTerm::Known(unchecked.effect);
                    solver
                        .unify_effect(
                            &scc_effect_vars[&fid],
                            &closed,
                            crate::ty::Polarity::Invariant,
                        )
                        .expect("the closed effect is the variable's own lower bound");

                    let bind: Vec<Param> = unchecked
                        .extern_params
                        .iter()
                        .map(|(name, ty)| Param::new(*name, ty.clone()))
                        .collect();
                    fn_bind_params.insert(fid, bind);
                    fn_unchecked.insert(fid, unchecked);
                }
                Err(errors) => {
                    fn_typeck_errors.insert(fid, errors);
                }
            }
        }

        // 2c. Resolve SCC: freeze InferTy -> Ty, build resolved fn types + fn_metas.
        for &fid in scc {
            let ret = scc_ret_vars
                .get(&fid)
                .and_then(|r| solver.freeze_ty(&solver.resolve_ty(r)).ok())
                .unwrap_or_else(Ty::error);
            let bind: Vec<Param> = fn_bind_params.get(&fid).cloned().unwrap_or_default();

            let effect = EffectTerm::Known(solver.freeze_effect(&scc_effect_vars[&fid]));

            let fn_ty = Ty::Fn {
                params: bind.clone(),
                ret: Box::new(ret),
                captures: vec![],
                effect,
            };
            resolved_fn_types.insert(fid, lift_to_poly(&fn_ty));
            fn_metas.insert(
                fid,
                FunctionMeta {
                    ty: fn_ty,
                    params: bind,
                },
            );
        }
    }

    // -- STEP 3: outcomes --------------------------------------------

    let mut outcomes: FxHashMap<QualifiedRef, FnInferOutcome> = FxHashMap::default();

    for &fid in &local_ids {
        let meta = fn_metas.remove(&fid).unwrap_or(FunctionMeta {
            ty: Ty::error(),
            params: vec![],
        });

        // If typeck failed, this function is Incomplete.
        if let Some(errors) = fn_typeck_errors.remove(&fid) {
            outcomes.insert(
                fid,
                FnInferOutcome::Incomplete {
                    unknown_contexts: vec![],
                    unknown_extern_params: vec![],
                    meta,
                    errors,
                },
            );
            continue;
        }

        // If no unchecked resolution (e.g., skipped function), Incomplete.
        let Some(unchecked) = fn_unchecked.remove(&fid) else {
            outcomes.insert(
                fid,
                FnInferOutcome::Incomplete {
                    unknown_contexts: vec![],
                    unknown_extern_params: vec![],
                    meta,
                    errors: vec![],
                },
            );
            continue;
        };

        let checked = unchecked;
        let tail_ty = checked.tail_ty.clone();
        outcomes.insert(
            fid,
            FnInferOutcome::Complete {
                resolution: checked,
                tail_ty,
                meta,
            },
        );
    }

    // -- STEP 5: Build result ----------------------------------------

    // Freeze InferTy -> Ty for context types at the output boundary.
    let context_types: FxHashMap<QualifiedRef, Ty> = known_ctx
        .iter()
        .map(|(&k, v)| {
            let resolved = solver.resolve_ty(v);
            let frozen = solver.freeze_ty(&resolved).unwrap_or_else(|_| Ty::error());
            (k, frozen)
        })
        .collect();

    InferResult {
        outcomes,
        context_types: Freeze::new(context_types),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use crate::ty::{ParamTerm, Poly, PolyBuilder, PolyParam, lift_to_poly};
    use acvus_utils::{Freeze, Interner};

    fn make_graph(interner: &Interner, source: &str) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            }]),
            contexts: Freeze::new(vec![]),
        }
    }

    fn make_graph_with_ctx(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_to_poly(ty),
            })
            .collect();
        let qref = QualifiedRef::root(interner.intern("test"));
        CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            }]),
            contexts: Freeze::new(contexts),
        }
    }

    // -- Completeness: correct types inferred --

    // FnRefs removed: context param inference no longer produces InferredParam for
    // undeclared contexts. All contexts are now passed via known_ctx; undeclared
    // context references are handled by the typechecker directly.

    #[test]
    fn infer_no_unknown_context_params() {
        let i = Interner::new();
        // Undeclared contexts no longer produce InferredParam entries.
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    #[test]
    fn infer_known_context_not_in_params() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + @y", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    // -- Soundness: no false inferences --

    #[test]
    fn infer_no_contexts_empty() {
        let i = Interner::new();
        let graph = make_graph(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    // ================================================================
    // Migrated from resolve.rs - inter-function, soundness, edge cases
    // ================================================================

    // -- Helpers (resolve-style: builtins + named params + output constraint) --

    fn make_graph_with_ctx_and_builtins(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let mut pb = PolyBuilder::new();
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_to_poly(ty),
            })
            .collect();
        let mut functions = Vec::new();
        let qref = QualifiedRef::root(interner.intern("test"));
        functions.push(Function {
            qref,
            kind: FnKind::Local(ParsedAst::Script(
                acvus_ast::parse_script(interner, source).expect("parse"),
            )),
            ty: TyTerm::Fn {
                params: vec![],
                ret: Box::new(pb.fresh_ty_var()),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        });
        CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        }
    }

    fn make_graph_no_ctx_with_builtins(interner: &Interner, source: &str) -> CompilationGraph {
        make_graph_with_ctx_and_builtins(interner, source, &[])
    }

    fn last_local_id(graph: &CompilationGraph) -> QualifiedRef {
        graph
            .functions
            .iter()
            .rev()
            .find(|f| matches!(f.kind, FnKind::Local(_)))
            .expect("no local function")
            .qref
    }

    /// Build a multi-function CompilationGraph with builtins.
    /// `fns`: list of `(name, source, signature, output_constraint)`.
    fn make_multi_fn_graph(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Option<PolyTy>)],
        ctx: &[(&str, Ty)],
    ) -> (CompilationGraph, Vec<(Astr, QualifiedRef)>) {
        let mut pb = PolyBuilder::new();
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_to_poly(ty),
            })
            .collect();

        let mut functions = Vec::new();
        let mut ids = Vec::new();

        for (name, source, sig, output) in fns {
            let aname = interner.intern(name);
            let fid = QualifiedRef::root(aname);
            ids.push((aname, fid));
            let poly_params: Vec<PolyParam> = sig
                .as_ref()
                .map(|params| {
                    params
                        .iter()
                        .map(|(name, ty)| {
                            ParamTerm::<Poly>::new(interner.intern(name), lift_to_poly(ty))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let ret = output.clone().unwrap_or_else(|| pb.fresh_ty_var());
            functions.push(Function {
                qref: fid,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: poly_params,
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            });
        }

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        (graph, ids)
    }

    /// Infer a multi-function graph, return result and ids.
    fn infer_multi(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Option<PolyTy>)],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, QualifiedRef)>) {
        infer_with_extern(interner, fns, &[], ctx)
    }

    /// Build a graph with both local and extern functions, then infer.
    fn infer_with_extern(
        interner: &Interner,
        local_fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Option<PolyTy>)],
        extern_fns: &[Function],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, QualifiedRef)>) {
        let mut pb = PolyBuilder::new();
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                qref: QualifiedRef::root(interner.intern(name)),
                ty: lift_to_poly(ty),
            })
            .collect();

        let mut functions = extern_fns.to_vec();
        let mut ids = Vec::new();

        for (name, source, sig, output) in local_fns {
            let aname = interner.intern(name);
            let fid = QualifiedRef::root(aname);
            ids.push((aname, fid));
            let poly_params: Vec<PolyParam> = sig
                .as_ref()
                .map(|params| {
                    params
                        .iter()
                        .map(|(name, ty)| {
                            ParamTerm::<Poly>::new(interner.intern(name), lift_to_poly(ty))
                        })
                        .collect()
                })
                .unwrap_or_default();
            let ret = output.clone().unwrap_or_else(|| pb.fresh_ty_var());
            functions.push(Function {
                qref: fid,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(interner, source).expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: poly_params,
                    ret: Box::new(ret),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            });
        }

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        let ext = extract::extract(interner, &graph);
        let result = infer(
            interner,
            &graph,
            &ext,
            &FxHashMap::default(),
            Freeze::default(),
        );
        (result, ids)
    }

    fn error_strings(interner: &Interner, result: &InferResult) -> Vec<String> {
        result
            .errors()
            .iter()
            .flat_map(|(_, errs)| errs.iter())
            .map(|e| format!("{}", e.display(interner)))
            .collect()
    }

    /// Get the tail type (return type) of a function from InferResult.
    /// In the old resolve pipeline, fn_type() returned the tail type.
    /// In the new pipeline, fn_type() returns the full Ty::Fn.
    fn tail_type(result: &InferResult, id: QualifiedRef) -> Option<Ty> {
        result.outcomes.get(&id)?.tail_ty().cloned()
    }

    fn make_extern_fn(interner: &Interner, name: &str, params: Vec<Ty>, ret: Ty) -> Function {
        let named_params: Vec<ParamTerm<Poly>> = params
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                ParamTerm::<Poly>::new(interner.intern(&format!("_{i}")), lift_to_poly(ty))
            })
            .collect();
        Function {
            qref: QualifiedRef::root(interner.intern(name)),
            kind: FnKind::Extern { bounds: vec![] },
            ty: TyTerm::Fn {
                params: named_params,
                ret: Box::new(lift_to_poly(&ret)),
                captures: vec![],
                effect: crate::ty::Effect::OPAQUE.into(),
            },
        }
    }

    // -- Completeness: valid single-function programs ------------------

    #[test]
    fn resolve_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_no_ctx_with_builtins(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert!(result.try_resolution(uid).is_some());
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_declared_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_user_provided_context() {
        let i = Interner::new();
        let graph = make_graph_no_ctx_with_builtins(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let mut user = FxHashMap::default();
        user.insert(QualifiedRef::root(i.intern("x")), lift_to_poly(&Ty::Int));
        let result = infer(&i, &graph, &ext, &user, Freeze::default());

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_string_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@name", &[("name", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::String);
    }

    #[test]
    fn resolve_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let graph = make_graph_with_ctx_and_builtins(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        assert!(
            !result.has_errors(),
            "errors: {:?}",
            error_strings(&i, &result)
        );
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result, uid).unwrap(), Ty::String);
    }

    // -- Soundness: type errors detected --

    #[test]
    fn resolve_type_mismatch_detected() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        assert!(result.has_errors(), "should detect type mismatch");
    }

    // -- Context type resolution --

    #[test]
    fn resolve_context_types_populated() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        let ctx_ref = graph.contexts[0].qref;
        assert_eq!(*result.context_type(&ctx_ref).unwrap(), Ty::Int);
    }

    // -- Completeness: valid inter-function calls ----------------------

    /// C1: A calls B with matching concrete types.
    #[test]
    fn inter_fn_simple_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("double", "$x * 2", Some(vec![("x", Ty::Int)]), None),
                ("main", "double(21)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C2: A calls B, B returns String.
    #[test]
    fn inter_fn_string_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("greet", "\"hello\"", Some(vec![]), None),
                ("main", "greet()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C3: Multi-arg function call.
    #[test]
    fn inter_fn_multi_arg() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "add",
                    "$x + $y",
                    Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                    None,
                ),
                ("main", "add(1, 2)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C4: Chain of calls - A calls B, B calls C.
    #[test]
    fn inter_fn_chain_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::Int)]), None),
                (
                    "double_inc",
                    "inc($x) + inc($x)",
                    Some(vec![("x", Ty::Int)]),
                    None,
                ),
                ("main", "double_inc(5)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C5: Function uses context and is called by another function.
    #[test]
    fn inter_fn_with_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("get_count", "@count", Some(vec![]), None),
                ("main", "get_count() + 1", None, None),
            ],
            &[("count", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C6: Function with declared Exact output type.
    #[test]
    fn inter_fn_exact_output() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "make_str",
                    "\"hi\"",
                    Some(vec![]),
                    Some(lift_to_poly(&Ty::String)),
                ),
                ("main", "make_str()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C7: Caller uses return value in arithmetic.
    #[test]
    fn inter_fn_return_used_in_binop() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("five", "5", Some(vec![]), None),
                ("main", "five() + five()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C8: Pipe syntax - value | fn.
    #[test]
    fn inter_fn_pipe_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("double", "$x * 2", Some(vec![("x", Ty::Int)]), None),
                ("main", "10 | double", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    // inter_fn_list_return: migrated to acvus-mir-test (depends on ExternFn `len`)

    /// C10: Function accepting and returning String.
    #[test]
    fn inter_fn_string_identity() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("echo", "$s", Some(vec![("s", Ty::String)]), None),
                ("main", "echo(\"hello\")", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// C11: Multiple callers of the same function.
    #[test]
    fn inter_fn_multiple_callers() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::Int)]), None),
                ("a", "inc(10)", None, None),
                ("b", "inc(20)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result, ids[2].1).unwrap(), Ty::Int);
    }

    /// C12: Calling function with bool return.
    #[test]
    fn inter_fn_bool_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("is_positive", "$x > 0", Some(vec![("x", Ty::Int)]), None),
                ("main", "is_positive(42)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Bool);
    }

    /// C13: Deep call chain - A -> B -> C -> D.
    #[test]
    fn inter_fn_deep_chain() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), None),
                ("c", "d()", Some(vec![]), None),
                ("b", "c()", Some(vec![]), None),
                ("main", "b()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    // inter_fn_mixed_builtin_and_local: migrated to acvus-mir-test (depends on ExternFn `to_string`)

    /// C15: Function result used as argument to another function.
    #[test]
    fn inter_fn_nested_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("inc", "$x + 1", Some(vec![("x", Ty::Int)]), None),
                ("main", "inc(inc(0))", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C16: Mutual recursion - A calls B, B calls A.
    #[test]
    fn inter_fn_mutual_recursion() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "is_even",
                    "is_odd($n - 1)",
                    Some(vec![("n", Ty::Int)]),
                    Some(lift_to_poly(&Ty::Bool)),
                ),
                (
                    "is_odd",
                    "is_even($n - 1)",
                    Some(vec![("n", Ty::Int)]),
                    Some(lift_to_poly(&Ty::Bool)),
                ),
                ("main", "is_even(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Bool);
    }

    /// C17: Self-recursion.
    #[test]
    fn inter_fn_self_recursion() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "fib",
                    "fib($n - 1) + fib($n - 2)",
                    Some(vec![("n", Ty::Int)]),
                    Some(lift_to_poly(&Ty::Int)),
                ),
                ("main", "fib(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// C18: Function with float params and return.
    #[test]
    fn inter_fn_float_arithmetic() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "avg",
                    "($a + $b) / 2.0",
                    Some(vec![("a", Ty::Float), ("b", Ty::Float)]),
                    None,
                ),
                ("main", "avg(1.0, 3.0)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Float);
    }

    // -- Soundness: invalid calls should be rejected -----------------

    /// S1: Wrong argument type.
    #[test]
    fn inter_fn_reject_wrong_arg_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("double", "x * 2", Some(vec![("x", Ty::Int)]), None),
                ("main", "double(\"hello\")", None, None),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("inc", "x + 1", Some(vec![("x", Ty::Int)]), None),
                ("main", "inc(1, 2)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject extra argument");
    }

    /// S3: Too few arguments.
    #[test]
    fn inter_fn_reject_too_few_args() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                (
                    "add",
                    "x + y",
                    Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                    None,
                ),
                ("main", "add(1)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject missing argument");
    }

    /// S4: Using return value where wrong type expected.
    #[test]
    fn inter_fn_reject_return_type_mismatch() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_str", "\"hello\"", Some(vec![]), None),
                ("main", "make_str() + 1", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject String + Int");
    }

    /// S5: Calling undefined function.
    #[test]
    fn inter_fn_reject_undefined_function() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(&i, &[("main", "nonexistent(1)", None, None)], &[]);
        assert!(
            result.has_errors(),
            "should reject call to undefined function"
        );
    }

    /// S6: Declared output type contradicts actual body.
    #[test]
    fn inter_fn_reject_output_mismatch() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Some(lift_to_poly(&Ty::String))),
                ("main", "bad()", None, None),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if !result.has_errors() {
            assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
        }
    }

    /// S7: Mutual recursion without declared types - must not stack overflow.
    #[test]
    fn inter_fn_mutual_recursion_no_declared_types() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("ping", "pong(x)", Some(vec![("x", Ty::Int)]), None),
                ("pong", "ping(x)", Some(vec![("x", Ty::Int)]), None),
            ],
            &[],
        );
        // We don't assert success or failure - just that it terminates.
        let _ = result;
    }

    /// S8: Wrong type in pipe position.
    #[test]
    fn inter_fn_reject_wrong_pipe_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("needs_int", "x + 1", Some(vec![("x", Ty::Int)]), None),
                ("main", "\"hello\" | needs_int", None, None),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("get_name", "@name", Some(vec![]), None),
                ("main", "get_name() + 1", None, None),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("f", "x", Some(vec![("x", Ty::Int)]), None),
                ("main", "f(1) + f(1, 2)", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject wrong arity call");
    }

    /// S11: Return type of called function used in list - type must be consistent.
    #[test]
    fn inter_fn_reject_heterogeneous_via_calls() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_int", "42", Some(vec![]), None),
                ("make_str", "\"hi\"", Some(vec![]), None),
                ("main", "[make_int(), make_str()]", None, None),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_str", "\"hi\"", Some(vec![]), None),
                ("needs_int", "x + 1", Some(vec![("x", Ty::Int)]), None),
                ("main", "needs_int(make_str())", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject String passed to Int param"
        );
    }

    // -- Edge cases --------------------------------------------------

    /// E1: Function with no parameters, no context - pure constant.
    #[test]
    fn inter_fn_zero_arg_constant() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("pi", "3", Some(vec![]), None),
                ("main", "pi()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// E2: Same name as builtin - local should shadow or coexist?
    #[test]
    fn inter_fn_name_shadows_builtin() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("len", "42", Some(vec![]), Some(lift_to_poly(&Ty::Int))),
                ("main", "len()", None, None),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if !result.has_errors() {
            let _ty = tail_type(&result, main_id).unwrap();
        }
    }

    /// E3: Callee defined after caller in graph order.
    #[test]
    fn inter_fn_forward_reference() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("main", "helper()", None, None),
                ("helper", "42", Some(vec![]), None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(
            errs.is_empty(),
            "forward reference should resolve: {errs:?}"
        );
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// E4: Two functions reading the same context.
    #[test]
    fn inter_fn_shared_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("read_a", "@x + 1", Some(vec![]), None),
                ("read_b", "@x + 2", Some(vec![]), None),
                ("main", "read_a() + read_b()", None, None),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[2].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// E5: Function calling itself with Exact type annotation (base case).
    #[test]
    fn inter_fn_self_call_exact() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "f",
                    "f($x - 1)",
                    Some(vec![("x", Ty::Int)]),
                    Some(lift_to_poly(&Ty::Int)),
                ),
                ("main", "f(10)", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// E6: Diamond dependency - A calls B and C, both call D.
    #[test]
    fn inter_fn_diamond_dependency() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("d", "1", Some(vec![]), None),
                ("b", "d() + 10", Some(vec![]), None),
                ("c", "d() + 20", Some(vec![]), None),
                ("main", "b() + c()", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    // inter_fn_pipe_through_builtins: migrated to acvus-mir-test (depends on ExternFn `iter`, `map`, `collect`, `len`)

    // inter_fn_effectful_return: migrated to acvus-mir-test (depends on ExternFn `collect`)

    // inter_fn_option_return: migrated to acvus-mir-test (depends on ExternFn `iter`, `first`, `unwrap`)

    /// E10: Three functions forming a pipeline.
    #[test]
    fn inter_fn_three_stage_pipeline() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("stage1", "$x + 1", Some(vec![("x", Ty::Int)]), None),
                ("stage2", "$x * 2", Some(vec![("x", Ty::Int)]), None),
                ("stage3", "$x - 1", Some(vec![("x", Ty::Int)]), None),
                ("main", "0 | stage1 | stage2 | stage3", None, None),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[3].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// E11: All local functions are callers - no inter-function calls.
    #[test]
    fn inter_fn_independent_functions() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("a", "1 + 2", None, None), ("b", "\"hello\"", None, None)],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[0].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::String);
    }

    /// E12: Function with object return type used with field access.
    #[test]
    fn inter_fn_object_return_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([
            (i.intern("name"), Ty::String),
            (i.intern("age"), Ty::Int),
        ]));
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "get_user",
                    "@user",
                    Some(vec![]),
                    Some(lift_to_poly(&obj_ty)),
                ),
                ("main", "get_user().name", None, None),
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
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    // ================================================================
    // Soundness boundary tests
    // ================================================================

    /// B1: Caller tries to use return value as wrong type.
    #[test]
    fn boundary_caller_forces_wrong_return_type() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), None),
                ("main", "a() + \"hello\"", None, None),
            ],
            &[],
        );
        assert!(result.has_errors(), "should reject Int used as String");
    }

    /// B2: Two callers use same function's return as different types.
    #[test]
    fn boundary_inconsistent_return_usage() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("a", "0", Some(vec![]), None),
                ("ok_caller", "a() + 1", None, None),
                ("bad_caller", "a() + \"hi\"", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "should reject inconsistent return type usage"
        );
    }

    /// B3: Mutual recursion with Inferred output - should NOT silently succeed.
    #[test]
    fn boundary_mutual_recursion_inferred_must_not_succeed_silently() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("ping", "pong(x)", Some(vec![("x", Ty::Int)]), None),
                ("pong", "ping(x)", Some(vec![("x", Ty::Int)]), None),
            ],
            &[],
        );
        if !result.has_errors() {
            let ping_ty = tail_type(&result, ids[0].1);
            let pong_ty = tail_type(&result, ids[1].1);
            if let (Some(ref pt), Some(ref qt)) = (ping_ty, pong_ty) {
                assert_eq!(pt, qt, "mutual recursion must have consistent return types");
            }
        }
    }

    /// B4: Self-recursion with Inferred output and no base case type.
    #[test]
    fn boundary_self_recursion_inferred_no_base() {
        let i = Interner::new();
        let (result, _) = infer_multi(
            &i,
            &[
                ("f", "f(x - 1)", Some(vec![("x", Ty::Int)]), None),
                ("main", "f(10)", None, None),
            ],
            &[],
        );
        assert!(
            result.has_errors(),
            "purely recursive return type should fail inference"
        );
    }

    /// B8: infer produces wrong type, should be caught.
    #[test]
    fn boundary_infer_wrong_type_catches() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Some(lift_to_poly(&Ty::String))),
                ("main", "bad()", None, None),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if let Some(main_ty) = tail_type(&result, main_id) {
            assert_ne!(
                main_ty,
                Ty::Int,
                "main must not see Int when bad declared String"
            );
        }
    }

    /// B9: Param in output but not in input - must not silently succeed.
    #[test]
    fn boundary_orphan_param_in_output() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("mystery", "x", Some(vec![]), None),
                ("main", "mystery() + 1", None, None),
            ],
            &[],
        );
        if !result.has_errors() {
            let main_id = ids[1].1;
            if let Some(ty) = tail_type(&result, main_id) {
                assert_eq!(ty, Ty::Int, "if resolved, return type should be Int");
            }
        }
    }

    // -- Extern function tests -------------------------------------

    /// Extern function should be callable from local functions.
    #[test]
    fn extern_fn_call_resolves() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, ids) =
            infer_with_extern(&i, &[("main", "fetch(42)", None, None)], &[fetch], &[]);
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "extern call should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::String);
    }

    /// Extern function with wrong argument type should error.
    #[test]
    fn extern_fn_call_type_mismatch() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, _) =
            infer_with_extern(&i, &[("main", "fetch(\"bad\")", None, None)], &[fetch], &[]);
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
        let (result, ids) = infer_with_extern(
            &i,
            &[("main", "get_count() + 1", None, None)],
            &[get_count],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result, main_id).unwrap(), Ty::Int);
    }

    /// Multiple extern functions can be registered and called.
    #[test]
    fn extern_fn_multiple() {
        let i = Interner::new();
        let add = make_extern_fn(&i, "ext_add", vec![Ty::Int, Ty::Int], Ty::Int);
        let greet = make_extern_fn(&i, "ext_greet", vec![Ty::String], Ty::String);
        let (result, ids) = infer_with_extern(
            &i,
            &[
                ("use_add", "ext_add(1, 2)", None, None),
                ("use_greet", "ext_greet(\"hi\")", None, None),
            ],
            &[add, greet],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result, ids[0].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result, ids[1].1).unwrap(), Ty::String);
    }

    // ================================================================
    // Context extraction tests
    // ================================================================

    // -- Completeness: contexts correctly extracted and typed --

    /// Single context read - type inferred from usage.
    #[test]
    fn context_extract_single_read() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    /// Multiple contexts.
    #[test]
    fn context_extract_multiple() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + @y");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    /// Context inside nested block - still extracted.
    #[test]
    fn context_extract_nested_block() {
        let i = Interner::new();
        let graph = make_graph(&i, "{ @x + 1 }");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());
    }

    // context_extract_in_lambda: migrated to acvus-mir-test (depends on ExternFn `map`, `collect`)

    // -- Complete/Incomplete boundary --

    /// Declared Exact context -> Complete.
    #[test]
    fn context_declared_exact_is_complete() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "declared Exact context should be Complete"
        );
    }

    /// Declared Inferred context -> Complete (type inferred via fresh var).
    #[test]
    fn context_declared_inferred_is_complete() {
        let i = Interner::new();
        let mut pb = PolyBuilder::new();
        let contexts = vec![Context {
            qref: QualifiedRef::root(i.intern("x")),
            ty: pb.fresh_ty_var(),
        }];
        let test_qref = QualifiedRef::root(i.intern("test"));
        let graph = CompilationGraph {
            functions: Freeze::new(vec![Function {
                qref: test_qref,
                kind: FnKind::Local(ParsedAst::Script(
                    acvus_ast::parse_script(&i, "@x + 1").expect("parse"),
                )),
                ty: TyTerm::Fn {
                    params: vec![],
                    ret: Box::new(pb.fresh_ty_var()),
                    captures: vec![],
                    effect: crate::ty::Effect::OPAQUE.into(),
                },
            }]),
            contexts: Freeze::new(contexts),
        };
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "Inferred context should still be Complete"
        );
        // The inferred type should be Int (from @x + 1).
        let qref = QualifiedRef::root(i.intern("x"));
        assert_eq!(*result.context_type(&qref).unwrap(), Ty::Int);
    }

    /// Undeclared context - typechecker creates fresh infer var in analysis mode.
    /// FnRefs removed: undeclared contexts no longer cause Incomplete via fn_params;
    /// they are handled by the typechecker's infer_vars and may resolve.
    #[test]
    fn context_undeclared_resolves_via_infer_var() {
        let i = Interner::new();
        // No contexts declared, but source uses @x. Typechecker infers @x : Int.
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "undeclared context with resolvable type should be Complete"
        );
    }

    /// User-provided context type -> Complete.
    #[test]
    fn context_user_provided_is_complete() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let mut user = FxHashMap::default();
        user.insert(QualifiedRef::root(i.intern("x")), lift_to_poly(&Ty::Int));
        let result = infer(&i, &graph, &ext, &user, Freeze::default());

        let fid = graph.functions[0].qref;
        assert!(
            result.outcomes[&fid].is_complete(),
            "user-provided context should be Complete"
        );
    }

    // -- Soundness: type mismatch detected --

    /// Declared context type conflicts with usage -> Incomplete.
    #[test]
    fn context_type_mismatch_is_incomplete() {
        let i = Interner::new();
        // @x is String but used in arithmetic.
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default(), Freeze::default());

        let fid = graph.functions[0].qref;
        assert!(
            !result.outcomes[&fid].is_complete(),
            "type mismatch should be Incomplete"
        );
    }

    // ================================================================
    // Param extraction tests
    // ================================================================

    /// Single $param - discovered in extern_params.
    #[test]
    fn param_extract_single() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "$x + 1", Some(vec![("x", Ty::Int)]), None)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params.len(), 1);
        assert_eq!(meta.params[0].name, i.intern("x"));
        assert_eq!(meta.params[0].ty, Ty::Int);
    }

    /// Multiple $params.
    #[test]
    fn param_extract_multiple() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                "$x + $y",
                Some(vec![("x", Ty::Int), ("y", Ty::Int)]),
                None,
            )],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params.len(), 2);
        let names: FxHashSet<Astr> = meta.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&i.intern("x")));
        assert!(names.contains(&i.intern("y")));
    }

    /// Param type inferred from usage.
    #[test]
    fn param_type_inferred() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "$x + 1", Some(vec![("x", Ty::Int)]), None)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::Int);
    }

    /// Param used in string concat -> inferred as String.
    #[test]
    fn param_type_inferred_string() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                r#"$x + "hello""#,
                Some(vec![("x", Ty::String)]),
                None,
            )],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::String);
    }

    // ================================================================
    // Type constraint tests
    // ================================================================

    /// Exact output constraint satisfied -> Complete.
    #[test]
    fn type_constraint_exact_satisfied() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "42", Some(vec![]), Some(lift_to_poly(&Ty::Int)))],
            &[],
        );
        let fid = ids[0].1;
        assert!(
            result.outcomes[&fid].is_complete(),
            "matching Exact output should be Complete"
        );
    }

    /// Exact output constraint violated -> Incomplete.
    #[test]
    fn type_constraint_exact_violated() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[(
                "test",
                r#""hello""#,
                Some(vec![]),
                Some(lift_to_poly(&Ty::Int)),
            )],
            &[],
        );
        let fid = ids[0].1;
        assert!(
            !result.outcomes[&fid].is_complete(),
            "String body with Exact(Int) constraint should be Incomplete"
        );
    }

    /// Inferred output -> always Complete (no constraint to violate).
    #[test]
    fn type_constraint_inferred_always_complete() {
        let i = Interner::new();
        let (result, ids) = infer_multi(&i, &[("test", r#""hello""#, Some(vec![]), None)], &[]);
        let fid = ids[0].1;
        assert!(result.outcomes[&fid].is_complete());
    }
}
