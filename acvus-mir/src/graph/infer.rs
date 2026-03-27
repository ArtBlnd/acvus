//! Phase 1: Infer
//!
//! For each function, infer the types of unknown context parameters and
//! function output types. Supports inter-function calls by adding all
//! local functions to the TypeEnv before typechecking.
//!
//! Output: context params + function types (for UI display and Phase 2).

use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use std::collections::BTreeSet;

use crate::ty::{Effect, EffectSet, Param, Ty, TySubst};

use super::extract::{ExtractResult, ParsedSource};
use super::types::*;

// ── Phase 1 output ──────────────────────────────────────────────────

/// Inferred context parameter for a single context.
#[derive(Debug, Clone)]
pub struct InferredParam {
    pub name: QualifiedRef,
    pub ty: Ty,
}

/// Inferred metadata for a single function.
#[derive(Debug, Clone)]
pub struct FunctionMeta {
    /// Fully resolved Ty::Fn for this function.
    pub ty: Ty,
    /// Named parameters (free_params from source zipped with signature types).
    pub params: Vec<Param>,
    /// Transitive effect set (reads/writes with QualifiedRef).
    /// Includes both direct access and access through callees.
    pub effect: EffectSet,
}

/// Per-function inference outcome.
#[derive(Debug)]
pub enum FnInferOutcome {
    /// Type fully resolved. Lowerable.
    Complete {
        resolution: crate::typeck::TypeResolution<crate::typeck::Checked>,
        tail_ty: Ty,
        meta: FunctionMeta,
    },
    /// Type incomplete. Cannot lower.
    Incomplete {
        unknown_contexts: Vec<InferredParam>,
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

    /// Get the checked resolution if complete.
    pub fn resolution(&self) -> Option<&crate::typeck::TypeResolution<crate::typeck::Checked>> {
        match self {
            FnInferOutcome::Complete { resolution, .. } => Some(resolution),
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
#[derive(Debug)]
pub struct InferResult {
    /// Per-function inference outcome (Complete or Incomplete).
    pub outcomes: FxHashMap<FunctionId, FnInferOutcome>,
    /// Per-function inferred context parameters (for UI display).
    pub fn_params: FxHashMap<FunctionId, Vec<InferredParam>>,
    /// All context parameters across all functions, deduplicated.
    pub all_params: Vec<InferredParam>,
    /// Resolved context types (known + inferred).
    pub context_types: FxHashMap<QualifiedRef, Ty>,
}

impl InferResult {
    /// Get the checked resolution for a function, if complete.
    pub fn try_resolution(&self, id: FunctionId) -> Option<&crate::typeck::TypeResolution<crate::typeck::Checked>> {
        self.outcomes.get(&id)?.resolution()
    }

    /// Get the function type for a function.
    pub fn fn_type(&self, id: FunctionId) -> Option<&Ty> {
        self.outcomes.get(&id).map(|o| &o.meta().ty)
    }

    /// Get the context type for a QualifiedRef.
    pub fn context_type(&self, qref: &QualifiedRef) -> Option<&Ty> {
        self.context_types.get(qref)
    }

    /// Whether any function has errors.
    pub fn has_errors(&self) -> bool {
        self.outcomes.values().any(|o| matches!(o, FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty()))
    }

    /// Collect all errors across all functions.
    pub fn errors(&self) -> Vec<(FunctionId, &[crate::error::MirError])> {
        self.outcomes.iter().filter_map(|(&id, o)| {
            match o {
                FnInferOutcome::Incomplete { errors, .. } if !errors.is_empty() => {
                    Some((id, errors.as_slice()))
                }
                _ => None,
            }
        }).collect()
    }
}

// ── Call graph + SCC ─────────────────────────────────────────────────

/// Extract call edges for a single function from its parsed AST.
/// Returns the list of FunctionIds that this function references.
pub fn extract_call_edges(
    parsed: &ParsedSource,
    name_to_fn: &FxHashMap<Astr, FunctionId>,
    self_id: FunctionId,
) -> Vec<FunctionId> {
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
) -> FxHashMap<FunctionId, Vec<FunctionId>> {
    let name_to_id: FxHashMap<Astr, FunctionId> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| (f.name, f.id))
        .collect();

    let mut edges: FxHashMap<FunctionId, Vec<FunctionId>> = FxHashMap::default();
    for func in graph.functions.iter() {
        let FnKind::Local(_) = &func.kind else {
            continue;
        };
        let Some(parsed) = extract.parsed.get(&func.id) else {
            continue;
        };
        edges.insert(func.id, extract_call_edges(parsed, &name_to_id, func.id));
    }
    edges
}

fn collect_value_refs_stmts(stmts: &[acvus_ast::Stmt], refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    for stmt in stmts {
        match stmt {
            Stmt::Bind { expr, .. } | Stmt::ContextStore { expr, .. } => {
                collect_value_refs_expr(expr, refs);
            }
            Stmt::Expr(expr) => collect_value_refs_expr(expr, refs),
            Stmt::MatchBind { source, body, .. } | Stmt::Iterate { source, body, .. } => {
                collect_value_refs_expr(source, refs);
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
        acvus_ast::Node::IterBlock(ib) => {
            collect_value_refs_expr(&ib.source, refs);
            for n in &ib.body {
                collect_value_refs_node(n, refs);
            }
            if let Some(ca) = &ib.catch_all {
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
        } => refs.push(*name),
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
        Expr::Range { start, end, .. } => {
            collect_value_refs_expr(start, refs);
            collect_value_refs_expr(end, refs);
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
    }
}

/// Tarjan's SCC algorithm. Returns SCCs in reverse topological order
/// (leaf SCCs first — dependencies before dependents).
pub fn tarjan_scc(
    ids: &[FunctionId],
    edges: &FxHashMap<FunctionId, Vec<FunctionId>>,
) -> Vec<Vec<FunctionId>> {
    let mut index_counter: u32 = 0;
    let mut stack: Vec<FunctionId> = Vec::new();
    let mut on_stack: FxHashSet<FunctionId> = FxHashSet::default();
    let mut index: FxHashMap<FunctionId, u32> = FxHashMap::default();
    let mut lowlink: FxHashMap<FunctionId, u32> = FxHashMap::default();
    let mut result: Vec<Vec<FunctionId>> = Vec::new();

    fn strongconnect(
        v: FunctionId,
        edges: &FxHashMap<FunctionId, Vec<FunctionId>>,
        index_counter: &mut u32,
        stack: &mut Vec<FunctionId>,
        on_stack: &mut FxHashSet<FunctionId>,
        index: &mut FxHashMap<FunctionId, u32>,
        lowlink: &mut FxHashMap<FunctionId, u32>,
        result: &mut Vec<Vec<FunctionId>>,
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

// ── Per-SCC inference ────────────────────────────────────────────────

/// Result of inferring a single SCC.
#[derive(Debug, Clone)]
pub struct SccInferResult {
    /// Per-function metadata (type, params, effect).
    pub fn_metas: FxHashMap<FunctionId, FunctionMeta>,
    /// Per-function inferred context parameters.
    pub fn_params: FxHashMap<FunctionId, Vec<InferredParam>>,
    /// Function name → resolved Ty::Fn (for passing to next SCC).
    pub resolved_types: FxHashMap<Astr, Ty>,
}

/// Infer types for a single SCC.
///
/// `resolved_fn_types`: all function types already resolved by prior SCCs + builtins.
/// `known_ctx`: declared context types from the graph.
pub fn infer_scc(
    interner: &Interner,
    scc: &[FunctionId],
    fn_by_id: &FxHashMap<FunctionId, &Function>,
    extract_refs: &FxHashMap<FunctionId, super::extract::FnRefs>,
    extract_parsed: &FxHashMap<FunctionId, &ParsedSource>,
    known_ctx: &FxHashMap<QualifiedRef, Ty>,
    resolved_fn_types: &FxHashMap<Astr, Ty>,
) -> SccInferResult {
    let mut subst = TySubst::new();
    let mut fn_params: FxHashMap<FunctionId, Vec<InferredParam>> = FxHashMap::default();
    let mut fn_bind_params: FxHashMap<FunctionId, Vec<Param>> = FxHashMap::default();
    let mut fn_ret_vars: FxHashMap<FunctionId, Ty> = FxHashMap::default();
    let mut fn_effect_vars: FxHashMap<FunctionId, Effect> = FxHashMap::default();
    let mut fn_direct_effects: FxHashMap<FunctionId, EffectSet> = FxHashMap::default();

    // Build Ty::Fn for functions in this SCC (with fresh ret/effect vars).
    let mut scc_fn_types: FxHashMap<Astr, Ty> = FxHashMap::default();

    for &fid in scc {
        let func = fn_by_id[&fid];
        let ret = match &func.constraint.output {
            Constraint::Exact(ty) => ty.clone(),
            _ => subst.fresh_param(),
        };
        let effect = subst.fresh_effect_var();
        fn_ret_vars.insert(fid, ret.clone());
        fn_effect_vars.insert(fid, effect.clone());

        let sig_params: Vec<Param> = func
            .constraint
            .signature
            .as_ref()
            .map(|s| s.params.clone())
            .unwrap_or_default();

        let fn_ty = Ty::Fn {
            params: sig_params,
            ret: Box::new(ret),
            captures: vec![],
            effect,
        };
        scc_fn_types.insert(func.name, fn_ty);
    }

    // Build TypeEnv: already-resolved functions + this SCC's unresolved functions.
    let mut env_functions = resolved_fn_types.clone();
    env_functions.extend(scc_fn_types);

    // Typecheck each function in this SCC.
    for &fid in scc {
        let func = fn_by_id[&fid];
        let Some(fn_ref) = extract_refs.get(&fid) else {
            continue;
        };
        let Some(parsed) = extract_parsed.get(&fid) else {
            continue;
        };

        let mut ctx_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
        let mut unknown_vars: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();

        for r in &fn_ref.context_reads {
            if let Some(ty) = known_ctx.get(r) {
                ctx_types.insert(*r, ty.clone());
            } else {
                let var = subst.fresh_param();
                unknown_vars.insert(*r, var.clone());
                ctx_types.insert(*r, var);
            }
        }

        let env = crate::ty::TypeEnv {
            contexts: ctx_types,
            functions: env_functions.clone(),
        };

        let expected_tail = fn_ret_vars.get(&fid);
        let declared_types: Vec<Ty> = func
            .constraint
            .signature
            .as_ref()
            .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
            .unwrap_or_default();

        let checker = crate::typeck::TypeChecker::new(interner, &env, &mut subst)
            .with_analysis_mode()
            .with_declared_param_types(declared_types);
        let result = match parsed {
            ParsedSource::Script(script) => checker.check_script(script, expected_tail),
            ParsedSource::Template(template) => checker.check_template(template),
        };

        if let Ok(ref unchecked) = result {
            if let Some(effect_var) = fn_effect_vars.get(&fid) {
                let _ = subst.unify_effect(effect_var, &unchecked.body_effect);
            }
            // Extract direct effects from body_effect.
            if let Effect::Resolved(ref effect_set) = unchecked.body_effect {
                fn_direct_effects.insert(fid, effect_set.clone());
            }

            let bind: Vec<Param> = unchecked
                .extern_params
                .iter()
                .map(|(name, ty)| Param::new(*name, subst.resolve(ty)))
                .collect();
            fn_bind_params.insert(fid, bind);
        }

        let params: Vec<InferredParam> = unknown_vars
            .into_iter()
            .map(|(name, var)| InferredParam {
                name,
                ty: subst.resolve(&var),
            })
            .collect();
        fn_params.insert(fid, params);
    }

    // Resolve all functions in this SCC.
    let mut resolved_types: FxHashMap<Astr, Ty> = FxHashMap::default();
    let mut fn_metas: FxHashMap<FunctionId, FunctionMeta> = FxHashMap::default();

    for &fid in scc {
        let func = fn_by_id[&fid];
        let ret = fn_ret_vars
            .get(&fid)
            .map(|r| subst.resolve(r))
            .unwrap_or_else(Ty::error);
        let effect = fn_effect_vars
            .get(&fid)
            .map(|e| subst.resolve_effect(e))
            .unwrap_or_else(Effect::pure);
        let bind: Vec<Param> = fn_bind_params
            .get(&fid)
            .map(|b| {
                b.iter()
                    .map(|p| Param::new(p.name, subst.resolve(&p.ty)))
                    .collect()
            })
            .unwrap_or_default();

        let fn_ty = Ty::Fn {
            params: bind.clone(),
            ret: Box::new(ret),
            captures: vec![],
            effect,
        };
        resolved_types.insert(func.name, fn_ty.clone());
        fn_metas.insert(
            fid,
            FunctionMeta {
                ty: fn_ty,
                params: bind,
                effect: EffectSet::default(), // filled by effect propagation later
            },
        );
    }

    SccInferResult {
        fn_metas,
        fn_params,
        resolved_types,
    }
}

// ── Batch inference ─────────────────────────────────────────────────

/// Run Phase 1 inference with SCC-based processing.
///
/// 1. Build call graph from AST references.
/// 2. Compute SCCs (Tarjan) — reverse topological order.
/// 3. Process each SCC:
///    - Within an SCC: shared TySubst, no instantiation of intra-SCC calls.
///    - After an SCC is done: resolve ret vars → concrete Ty::Fn.
///    - Next SCC sees concrete types → instantiation is safe.
pub fn infer(
    interner: &Interner,
    graph: &CompilationGraph,
    extract: &ExtractResult,
    user_context_types: &FxHashMap<QualifiedRef, Ty>,
) -> InferResult {
    let mut subst = TySubst::new();
    let mut fn_params: FxHashMap<FunctionId, Vec<InferredParam>> = FxHashMap::default();
    let mut fn_bind_params: FxHashMap<FunctionId, Vec<Param>> = FxHashMap::default();
    // Direct effects per function, collected from typeck body_effect.
    let mut fn_direct_effects: FxHashMap<FunctionId, EffectSet> = FxHashMap::default();
    // Typeck results per function — stored for check_completeness after SCC completes.
    let mut fn_unchecked: FxHashMap<FunctionId, crate::typeck::TypeResolution<crate::typeck::Unchecked>> = FxHashMap::default();
    // Typeck errors per function (from analysis mode).
    let mut fn_typeck_errors: FxHashMap<FunctionId, Vec<crate::error::MirError>> = FxHashMap::default();

    // Resolved function types — populated as SCCs complete.
    let mut resolved_fn_types: FxHashMap<Astr, Ty> = crate::builtins::builtin_fn_types(interner);

    // Add extern function types (these are always known upfront).
    for func in graph.functions.iter() {
        if let FnKind::Extern = &func.kind {
            if let Constraint::Exact(ty) = &func.constraint.output {
                resolved_fn_types.insert(func.name, ty.clone());
            }
        }
    }

    // Collect known context types: graph declarations + user-provided.
    let mut known_ctx: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
    for ctx in graph.contexts.iter() {
        match &ctx.constraint {
            Constraint::Exact(ty) => {
                known_ctx.insert(ctx.qualified_ref(), ty.clone());
            }
            Constraint::Inferred => {
                // Declared but type unknown — use fresh var. Infer will determine the type.
                known_ctx.insert(ctx.qualified_ref(), subst.fresh_param());
            }
            Constraint::DerivedFnOutput(_, _) | Constraint::DerivedContext(_, _) => {
                // TODO: resolve derived types. For now, fresh var.
                known_ctx.insert(ctx.qualified_ref(), subst.fresh_param());
            }
        }
    }
    // User-provided types override.
    known_ctx.extend(user_context_types.iter().map(|(&k, v)| (k, v.clone())));

    // Map FunctionId → Function for lookup.
    let fn_by_id: FxHashMap<FunctionId, &Function> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| (f.id, f))
        .collect();

    // ── STEP 1: Build call graph and compute SCCs ───────────────────

    let call_graph = build_call_graph(graph, extract);
    let local_ids: Vec<FunctionId> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| f.id)
        .collect();
    let sccs = tarjan_scc(&local_ids, &call_graph);

    // ── STEP 2: Process each SCC in topological order ───────────────

    // Track ret vars and effect vars per function (across all SCCs).
    let mut fn_ret_vars: FxHashMap<FunctionId, Ty> = FxHashMap::default();
    let mut fn_effect_vars: FxHashMap<FunctionId, Effect> = FxHashMap::default();

    for scc in &sccs {
        // Build Ty::Fn for functions in this SCC (with fresh ret/effect vars).
        let mut scc_fn_types: FxHashMap<Astr, Ty> = FxHashMap::default();

        for &fid in scc {
            let func = fn_by_id[&fid];

            let ret = match &func.constraint.output {
                Constraint::Exact(ty) => ty.clone(),
                _ => subst.fresh_param(),
            };
            let effect = subst.fresh_effect_var();
            fn_ret_vars.insert(fid, ret.clone());
            fn_effect_vars.insert(fid, effect.clone());

            let sig_params: Vec<Param> = func
                .constraint
                .signature
                .as_ref()
                .map(|s| s.params.clone())
                .unwrap_or_default();

            let fn_ty = Ty::Fn {
                params: sig_params,
                ret: Box::new(ret),
                captures: vec![],
                effect,
            };
            scc_fn_types.insert(func.name, fn_ty);
        }

        // Build TypeEnv: already-resolved functions + this SCC's unresolved functions.
        let mut env_functions = resolved_fn_types.clone();
        env_functions.extend(scc_fn_types);

        // Typecheck each function in this SCC.
        for &fid in scc {
            let func = fn_by_id[&fid];
            let Some(fn_ref) = extract.fn_refs.get(&fid) else {
                continue;
            };
            let Some(parsed) = extract.parsed.get(&fid) else {
                continue;
            };

            // Build context types.
            let mut ctx_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
            let mut unknown_vars: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();

            for r in &fn_ref.context_reads {
                if let Some(ty) = known_ctx.get(r) {
                    ctx_types.insert(*r, ty.clone());
                } else {
                    let var = subst.fresh_param();
                    unknown_vars.insert(*r, var.clone());
                    ctx_types.insert(*r, var);
                }
            }

            let env = crate::ty::TypeEnv {
                contexts: ctx_types,
                functions: env_functions.clone(),
            };

            let expected_tail = fn_ret_vars.get(&fid);
            let declared_types: Vec<Ty> = func
                .constraint
                .signature
                .as_ref()
                .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
                .unwrap_or_default();

            let checker = crate::typeck::TypeChecker::new(interner, &env, &mut subst)
                .with_analysis_mode()
                .with_declared_param_types(declared_types);
            let result = match parsed {
                ParsedSource::Script(script) => checker.check_script(script, expected_tail),
                ParsedSource::Template(template) => checker.check_template(template),
            };

            match result {
                Ok(unchecked) => {
                    if let Some(effect_var) = fn_effect_vars.get(&fid) {
                        let _ = subst.unify_effect(effect_var, &unchecked.body_effect);
                    }
                    // Collect direct effects from typeck body_effect.
                    if let Effect::Resolved(ref effect_set) = unchecked.body_effect {
                        fn_direct_effects.insert(fid, effect_set.clone());
                    }

                    let bind: Vec<Param> = unchecked
                        .extern_params
                        .iter()
                        .map(|(name, ty)| Param::new(*name, subst.resolve(ty)))
                        .collect();
                    fn_bind_params.insert(fid, bind);

                    // Store unchecked resolution for check_completeness after SCC.
                    fn_unchecked.insert(fid, unchecked);
                }
                Err(errors) => {
                    fn_typeck_errors.insert(fid, errors);
                }
            }

            let params: Vec<InferredParam> = unknown_vars
                .into_iter()
                .map(|(name, var)| InferredParam {
                    name,
                    ty: subst.resolve(&var),
                })
                .collect();
            fn_params.insert(fid, params);
        }

        // SCC complete: resolve all functions in this SCC and add to resolved_fn_types.
        for &fid in scc {
            let func = fn_by_id[&fid];

            let ret = fn_ret_vars
                .get(&fid)
                .map(|r| subst.resolve(r))
                .unwrap_or_else(Ty::error);
            let effect = fn_effect_vars
                .get(&fid)
                .map(|e| subst.resolve_effect(e))
                .unwrap_or_else(Effect::pure);

            let resolved_bind: Vec<Param> = fn_bind_params
                .get(&fid)
                .map(|bind| {
                    bind.iter()
                        .map(|p| Param::new(p.name, subst.resolve(&p.ty)))
                        .collect()
                })
                .unwrap_or_default();

            let resolved_fn_ty = Ty::Fn {
                params: resolved_bind,
                ret: Box::new(ret),
                captures: vec![],
                effect,
            };
            resolved_fn_types.insert(func.name, resolved_fn_ty);
        }
    }

    // ── STEP 3: Build FunctionMeta + compute transitive effects ─────

    let mut all_map: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
    for params in fn_params.values() {
        for param in params {
            all_map
                .entry(param.name)
                .or_insert_with(|| param.ty.clone());
        }
    }
    let all_params: Vec<InferredParam> = all_map
        .into_iter()
        .map(|(name, ty)| InferredParam { name, ty })
        .collect();

    // Build FunctionMeta for each local function (initially with empty transitive effects).
    let mut fn_metas: FxHashMap<FunctionId, FunctionMeta> = FxHashMap::default();

    for &fid in &local_ids {
        let ret = fn_ret_vars
            .get(&fid)
            .map(|r| subst.resolve(r))
            .unwrap_or_else(Ty::error);
        let effect = fn_effect_vars
            .get(&fid)
            .map(|e| subst.resolve_effect(e))
            .unwrap_or_else(Effect::pure);
        let bind: Vec<Param> = fn_bind_params
            .get(&fid)
            .map(|b| {
                b.iter()
                    .map(|p| Param::new(p.name, subst.resolve(&p.ty)))
                    .collect()
            })
            .unwrap_or_default();

        let fn_ty = Ty::Fn {
            params: bind.clone(),
            ret: Box::new(ret),
            captures: vec![],
            effect,
        };
        fn_metas.insert(
            fid,
            FunctionMeta {
                ty: fn_ty,
                params: bind,
                effect: EffectSet::default(), // filled below
            },
        );
    }

    // Seed with direct effects + parameter-carried effects.
    for &fid in &local_ids {
        let mut effect = fn_direct_effects
            .get(&fid)
            .cloned()
            .unwrap_or_default();

        if let Some(meta) = fn_metas.get(&fid) {
            for param in &meta.params {
                if let Some(param_effect) = param.ty.carried_effect()
                    && let Effect::Resolved(param_set) = param_effect
                {
                    effect = effect.union(param_set);
                }
            }
        }

        if let Some(meta) = fn_metas.get_mut(&fid) {
            meta.effect = effect;
        }
    }

    // Propagate transitive effects through call graph in SCC topological order.
    for scc in &sccs {
        loop {
            let mut changed = false;
            for &fid in scc {
                if let Some(callees) = call_graph.get(&fid) {
                    for &callee_id in callees {
                        let callee_effect = fn_metas
                            .get(&callee_id)
                            .map(|m| m.effect.clone())
                            .unwrap_or_default();
                        let current = fn_metas
                            .get(&fid)
                            .map(|m| m.effect.clone())
                            .unwrap_or_default();
                        let merged = current.union(&callee_effect);
                        if merged != current
                            && let Some(meta) = fn_metas.get_mut(&fid)
                        {
                            meta.effect = merged;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    // ── STEP 4: check_completeness + effect constraint → outcomes ──

    let mut outcomes: FxHashMap<FunctionId, FnInferOutcome> = FxHashMap::default();

    for &fid in &local_ids {
        let meta = fn_metas.remove(&fid).unwrap_or(FunctionMeta {
            ty: Ty::error(),
            params: vec![],
            effect: EffectSet::default(),
        });

        // If typeck failed, this function is Incomplete.
        if let Some(errors) = fn_typeck_errors.remove(&fid) {
            let unknown_contexts = fn_params.get(&fid).cloned().unwrap_or_default();
            outcomes.insert(fid, FnInferOutcome::Incomplete {
                unknown_contexts,
                unknown_extern_params: vec![],
                meta,
                errors,
            });
            continue;
        }

        // If no unchecked resolution (e.g., skipped function), Incomplete.
        let Some(unchecked) = fn_unchecked.remove(&fid) else {
            outcomes.insert(fid, FnInferOutcome::Incomplete {
                unknown_contexts: fn_params.get(&fid).cloned().unwrap_or_default(),
                unknown_extern_params: vec![],
                meta,
                errors: vec![],
            });
            continue;
        };

        // If this function references undeclared contexts, it's Incomplete.
        let unknown_ctxs = fn_params.get(&fid).cloned().unwrap_or_default();
        if !unknown_ctxs.is_empty() {
            outcomes.insert(fid, FnInferOutcome::Incomplete {
                unknown_contexts: unknown_ctxs,
                unknown_extern_params: unchecked.extern_params.clone(),
                meta,
                errors: vec![],
            });
            continue;
        }

        // Try check_completeness.
        match crate::typeck::check_completeness(unchecked, &subst) {
            Ok(checked) => {
                // Check effect constraint if present.
                let func = fn_by_id.get(&fid);
                if let Some(func) = func {
                    if let Some(ref allowed) = func.constraint.effect {
                        if let Err(err) = crate::typeck::check_effect_constraint(&checked.body_effect, allowed) {
                            outcomes.insert(fid, FnInferOutcome::Incomplete {
                                unknown_contexts: fn_params.get(&fid).cloned().unwrap_or_default(),
                                unknown_extern_params: checked.extern_params.clone(),
                                meta,
                                errors: vec![err],
                            });
                            continue;
                        }
                    }
                }

                let tail_ty = checked.tail_ty.clone();
                outcomes.insert(fid, FnInferOutcome::Complete {
                    resolution: checked,
                    tail_ty,
                    meta,
                });
            }
            Err(errors) => {
                outcomes.insert(fid, FnInferOutcome::Incomplete {
                    unknown_contexts: fn_params.get(&fid).cloned().unwrap_or_default(),
                    unknown_extern_params: vec![],
                    meta,
                    errors,
                });
            }
        }
    }

    // ── STEP 5: Build context_types ─────────────────────────────────

    let mut context_types: FxHashMap<QualifiedRef, Ty> = known_ctx;
    // Resolve any type vars in context types with final subst state.
    for ty in context_types.values_mut() {
        *ty = subst.resolve(ty);
    }

    InferResult {
        outcomes,
        fn_params,
        all_params,
        context_types,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use acvus_utils::{Freeze, Interner};

    fn make_graph(interner: &Interner, source: &str) -> CompilationGraph {
        CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(vec![Function {
                id: FunctionId::alloc(),
                name: interner.intern("test"),
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: interner.intern("test"),
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
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
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(vec![Function {
                id: FunctionId::alloc(),
                name: interner.intern("test"),
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: interner.intern("test"),
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                },
            }]),
            contexts: Freeze::new(contexts),
        }
    }

    // -- Completeness: correct types inferred --

    #[test]
    fn infer_int_from_arithmetic() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].name, QualifiedRef::root(i.intern("x")));
        assert_eq!(result.all_params[0].ty, Ty::Int);
    }

    #[test]
    fn infer_string_from_concat() {
        let i = Interner::new();
        let graph = make_graph(&i, r#"@x + "hello""#);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].ty, Ty::String);
    }

    #[test]
    fn infer_multiple_params_with_literal() {
        let i = Interner::new();
        // @a + @b + 1 — the literal 1 forces both to Int.
        let graph = make_graph(&i, "@a + @b + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 2);
        for p in &result.all_params {
            assert_eq!(p.ty, Ty::Int);
        }
    }

    #[test]
    fn infer_unknown_remains_var() {
        let i = Interner::new();
        // @a + @b — no literal, type cannot be fully resolved.
        let graph = make_graph(&i, "@a + @b");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 2);
        // At least one should still be a Param (unresolved).
        let has_param = result
            .all_params
            .iter()
            .any(|p| matches!(p.ty, Ty::Param { .. }));
        assert!(
            has_param,
            "without a literal, type should remain unresolved"
        );
    }

    // -- Completeness: known context not re-inferred --

    #[test]
    fn infer_skips_known_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + @y", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        // Only @y should be inferred, @x is already known.
        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].name, QualifiedRef::root(i.intern("y")));
        assert_eq!(result.all_params[0].ty, Ty::Int);
    }

    // -- Soundness: no false inferences --

    #[test]
    fn infer_no_contexts_empty() {
        let i = Interner::new();
        let graph = make_graph(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(result.all_params.is_empty());
    }

    // ── Effect propagation tests ────────────────────────────────────

    /// Helper: build multi-function graph.
    /// Each entry: (name, source, params, output_constraint)
    fn make_multi_graph(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<Ty>>)],
        ctx: &[(&str, Ty)],
    ) -> (CompilationGraph, Vec<(Astr, FunctionId)>) {
        let mut functions = Vec::new();
        let mut ids = Vec::new();
        for &(name, source, ref params) in fns {
            let fid = FunctionId::alloc();
            let aname = interner.intern(name);
            let sig = params.as_ref().map(|p| Signature {
                params: p
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| Param::new(interner.intern(&format!("_{i}")), ty.clone()))
                    .collect(),
            });
            functions.push(Function {
                id: fid,
                name: aname,
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: aname,
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: sig,
                    output: Constraint::Inferred,
                    effect: None,
                },
            });
            ids.push((aname, fid));
        }
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        let graph = CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        (graph, ids)
    }

    #[test]
    fn effect_direct_read_tracked() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[("reader", "@x + @y", Some(vec![]))],
            &[("x", Ty::Int), ("y", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = ids[0].1;
        let effect = &result.outcomes[&fid].meta().effect;
        let ctx_x = QualifiedRef::root(i.intern("x"));
        let ctx_y = QualifiedRef::root(i.intern("y"));
        assert!(effect.reads.contains(&ctx_x), "should have @x in reads");
        assert!(effect.reads.contains(&ctx_y), "should have @y in reads");
        assert!(effect.writes.is_empty(), "read-only should have no writes");
    }

    #[test]
    fn effect_direct_write_tracked() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[("writer", "@x = 42; @x", Some(vec![]))],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = ids[0].1;
        let effect = &result.outcomes[&fid].meta().effect;
        let ctx_x = QualifiedRef::root(i.intern("x"));
        assert!(effect.writes.contains(&ctx_x), "should have @x in writes");
        assert!(
            effect.reads.contains(&ctx_x),
            "should also have @x in reads (tail)"
        );
    }

    #[test]
    fn effect_pure_function_empty() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(&i, &[("pure_fn", "1 + 2", Some(vec![]))], &[]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = ids[0].1;
        let effect = &result.outcomes[&fid].meta().effect;
        assert!(effect.is_pure(), "pure function should have empty effect");
    }

    #[test]
    fn effect_transitive_through_callee() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[("get_x", "@x", Some(vec![])), ("main", "get_x()", None)],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_x = QualifiedRef::root(i.intern("x"));

        // get_x directly reads @x
        let get_x_effect = &result.outcomes[&ids[0].1].meta().effect;
        assert!(get_x_effect.reads.contains(&ctx_x));

        // main calls get_x → transitive read of @x
        let main_effect = &result.outcomes[&ids[1].1].meta().effect;
        assert!(
            main_effect.reads.contains(&ctx_x),
            "caller should transitively inherit callee's reads"
        );
    }

    #[test]
    fn effect_transitive_deep_chain() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("read_x", "@x", Some(vec![])),
                ("mid", "read_x()", Some(vec![])),
                ("top", "mid()", None),
            ],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_x = QualifiedRef::root(i.intern("x"));
        // top → mid → read_x → @x. All should have @x in reads.
        for (_, fid) in &ids {
            let effect = &result.outcomes[fid].meta().effect;
            assert!(
                effect.reads.contains(&ctx_x),
                "transitive chain should propagate reads"
            );
        }
    }

    #[test]
    fn effect_no_false_transitive() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("read_x", "@x", Some(vec![])),
                ("pure_fn", "1 + 2", Some(vec![])),
                ("main", "pure_fn()", None),
            ],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_x = QualifiedRef::root(i.intern("x"));
        // main calls pure_fn (not read_x) → should NOT have @x
        let main_effect = &result.outcomes[&ids[2].1].meta().effect;
        assert!(
            !main_effect.reads.contains(&ctx_x),
            "should not have false transitive reads"
        );
        assert!(main_effect.is_pure());
    }

    #[test]
    fn effect_scc_cycle_converges() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                // a reads @x, calls b
                ("a", "@x + b()", Some(vec![])),
                // b reads @y, calls a
                ("b", "@y + a()", Some(vec![])),
            ],
            &[("x", Ty::Int), ("y", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_x = QualifiedRef::root(i.intern("x"));
        let ctx_y = QualifiedRef::root(i.intern("y"));

        // a and b are in the same SCC.
        // After fixpoint: both should have reads = {@x, @y}.
        let a_effect = &result.outcomes[&ids[0].1].meta().effect;
        let b_effect = &result.outcomes[&ids[1].1].meta().effect;
        assert!(a_effect.reads.contains(&ctx_x), "a should read @x (direct)");
        assert!(
            a_effect.reads.contains(&ctx_y),
            "a should read @y (transitive from b)"
        );
        assert!(
            b_effect.reads.contains(&ctx_x),
            "b should read @x (transitive from a)"
        );
        assert!(b_effect.reads.contains(&ctx_y), "b should read @y (direct)");
    }

    #[test]
    fn effect_mixed_reads_writes() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("writer", "@x = 10; @x", Some(vec![])),
                ("caller", "writer()", None),
            ],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_x = QualifiedRef::root(i.intern("x"));

        // writer writes @x → caller should transitively inherit
        let caller_effect = &result.outcomes[&ids[1].1].meta().effect;
        assert!(caller_effect.writes.contains(&ctx_x), "transitive write");
        assert!(caller_effect.reads.contains(&ctx_x), "transitive read");
    }

    // ── Parameter effect union tests ────────────────────────────────

    #[test]
    fn effect_param_effectful_iterator_propagates() {
        // Function takes an effectful iterator param → function is effectful.
        let i = Interner::new();
        let ctx_x = QualifiedRef::root(i.intern("effect_x"));
        let iter_effect = Effect::Resolved(EffectSet {
            reads: [ctx_x].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            // takes an iterator param, returns element
            &[(
                "consumer",
                "$_0",
                Some(vec![Ty::Iterator(Box::new(Ty::Int), iter_effect)]),
            )],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let effect = &result.outcomes[&ids[0].1].meta().effect;
        assert!(
            effect.reads.contains(&ctx_x),
            "function should inherit effectful iterator param's reads"
        );
    }

    #[test]
    fn effect_param_pure_iterator_no_effect() {
        // Function takes a pure iterator param → no param effect contribution.
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[(
                "processor",
                "$_0",
                Some(vec![Ty::Iterator(Box::new(Ty::Int), Effect::pure())]),
            )],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let effect = &result.outcomes[&ids[0].1].meta().effect;
        assert!(
            effect.is_pure(),
            "pure iterator param should not make function effectful"
        );
    }

    #[test]
    fn effect_param_effectful_fn_propagates() {
        // Function takes a Fn param with effect → function inherits it.
        let i = Interner::new();
        let ctx_y = QualifiedRef::root(i.intern("effect_y"));
        let fn_effect = Effect::Resolved(EffectSet {
            writes: [ctx_y].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            &[(
                "caller",
                "$_0",
                Some(vec![Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: fn_effect,
                }]),
            )],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let effect = &result.outcomes[&ids[0].1].meta().effect;
        assert!(
            effect.writes.contains(&ctx_y),
            "function should inherit effectful Fn param's writes"
        );
    }

    #[test]
    fn effect_param_plus_direct_access_union() {
        // Two separate functions: one reads @a, one takes effectful param.
        // Caller calls both → transitive union = reads {@a} ∪ writes {@b}.
        let i = Interner::new();
        let ctx_b = QualifiedRef::root(i.intern("effect_b"));
        let fn_effect = Effect::Resolved(EffectSet {
            writes: [ctx_b].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("read_a", "@a", Some(vec![])),
                (
                    "use_fn",
                    "$_0",
                    Some(vec![Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: fn_effect,
                    }]),
                ),
                // caller invokes both → gets both effects transitively
                ("caller", "read_a() + use_fn(read_a)", None),
            ],
            &[("a", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_a = QualifiedRef::root(i.intern("a"));
        let caller_effect = &result.outcomes[&ids[2].1].meta().effect;
        assert!(
            caller_effect.reads.contains(&ctx_a),
            "transitive read of @a from read_a"
        );
        assert!(
            caller_effect.writes.contains(&ctx_b),
            "param Fn's write of @b from use_fn"
        );
    }

    #[test]
    fn effect_multiple_effectful_params_union() {
        // Function takes two effectful Fn params via two separate functions.
        let i = Interner::new();
        let ctx_x = QualifiedRef::root(i.intern("effect_x"));
        let ctx_y = QualifiedRef::root(i.intern("effect_y"));
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                (
                    "fn_a",
                    "$_0",
                    Some(vec![Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: Effect::Resolved(EffectSet {
                            reads: [ctx_x].into_iter().collect(),
                            ..Default::default()
                        }),
                    }]),
                ),
                (
                    "fn_b",
                    "$_0",
                    Some(vec![Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: Effect::Resolved(EffectSet {
                            writes: [ctx_y].into_iter().collect(),
                            ..Default::default()
                        }),
                    }]),
                ),
                // caller gets both effects transitively
                ("caller", "fn_a(fn_b) + fn_b(fn_a)", None),
            ],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        // fn_a has param effect reads @x
        let fn_a_effect = &result.outcomes[&ids[0].1].meta().effect;
        assert!(fn_a_effect.reads.contains(&ctx_x), "fn_a param's read");

        // fn_b has param effect writes @y
        let fn_b_effect = &result.outcomes[&ids[1].1].meta().effect;
        assert!(fn_b_effect.writes.contains(&ctx_y), "fn_b param's write");
    }

    // ════════════════════════════════════════════════════════════════
    // Migrated from resolve.rs — inter-function, soundness, edge cases
    // ════════════════════════════════════════════════════════════════

    // ── Helpers (resolve-style: builtins + named params + output constraint) ──

    fn make_graph_with_ctx_and_builtins(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
    ) -> CompilationGraph {
        let contexts = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        let mut functions = crate::builtins::standard_builtins(interner);
        functions.push(Function {
            id: FunctionId::alloc(),
            name: interner.intern("test"),
            namespace: None,
            kind: FnKind::Local(SourceCode {
                name: interner.intern("test"),
                source: interner.intern(source),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: None,
            },
        });
        CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        }
    }

    fn make_graph_no_ctx_with_builtins(interner: &Interner, source: &str) -> CompilationGraph {
        make_graph_with_ctx_and_builtins(interner, source, &[])
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

    /// Build a multi-function CompilationGraph with builtins.
    /// `fns`: list of `(name, source, signature, output_constraint)`.
    fn make_multi_fn_graph(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        ctx: &[(&str, Ty)],
    ) -> (CompilationGraph, Vec<(Astr, FunctionId)>) {
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
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
                namespace: None,
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
                    effect: None,
                },
            });
        }

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            namespaces: Default::default(),
        };
        (graph, ids)
    }

    /// Infer a multi-function graph, return result and ids.
    fn infer_multi(
        interner: &Interner,
        fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, FunctionId)>) {
        infer_with_extern(interner, fns, &[], ctx)
    }

    /// Build a graph with both local and extern functions, then infer.
    fn infer_with_extern(
        interner: &Interner,
        local_fns: &[(&str, &str, Option<Vec<(&str, Ty)>>, Constraint)],
        extern_fns: &[Function],
        ctx: &[(&str, Ty)],
    ) -> (InferResult, Vec<(Astr, FunctionId)>) {
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
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
                namespace: None,
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
                    effect: None,
                },
            });
        }
        functions.extend_from_slice(extern_fns);

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            namespaces: Default::default(),
        };
        let ext = extract::extract(interner, &graph);
        let result = infer(interner, &graph, &ext, &FxHashMap::default());
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

    /// Helper: extract the resolved EffectSet from a TypeResolution's body_effect.
    fn extract_effect_set(
        resolution: &crate::typeck::TypeResolution<crate::typeck::Checked>,
    ) -> &EffectSet {
        match &resolution.body_effect {
            Effect::Resolved(set) => set,
            other => panic!("expected Effect::Resolved, got {other:?}"),
        }
    }

    /// Get the tail type (return type) of a function from InferResult.
    /// In the old resolve pipeline, fn_type() returned the tail type.
    /// In the new pipeline, fn_type() returns the full Ty::Fn.
    fn tail_type(result: &InferResult, id: FunctionId) -> Option<Ty> {
        result.outcomes.get(&id)?.tail_ty().cloned()
    }

    fn make_extern_fn(
        interner: &Interner,
        name: &str,
        params: Vec<Ty>,
        ret: Ty,
    ) -> Function {
        let named_params: Vec<crate::ty::Param> = params
            .into_iter()
            .enumerate()
            .map(|(i, ty)| crate::ty::Param::new(interner.intern(&format!("_{i}")), ty))
            .collect();
        Function {
            id: FunctionId::alloc(),
            name: interner.intern(name),
            namespace: None,
            kind: FnKind::Extern,
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
                effect: None,
            },
        }
    }

    /// Helper: infer a single function with an effect constraint.
    fn infer_with_effect(
        interner: &Interner,
        source: &str,
        ctx: &[(&str, Ty)],
        effect: EffectSet,
    ) -> (InferResult, FunctionId) {
        let contexts: Vec<Context> = ctx
            .iter()
            .map(|(name, ty)| Context {
                name: interner.intern(name),
                namespace: None,
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();

        let mut functions = crate::builtins::standard_builtins(interner);
        let fid = FunctionId::alloc();
        let name = interner.intern("test");
        functions.push(Function {
            id: fid,
            name,
            namespace: None,
            kind: FnKind::Local(SourceCode {
                name,
                source: interner.intern(source),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: Some(effect),
            },
        });

        let graph = CompilationGraph {
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
            namespaces: Default::default(),
        };
        let ext = extract::extract(interner, &graph);
        let result = infer(interner, &graph, &ext, &FxHashMap::default());
        (result, fid)
    }

    // ── Completeness: valid single-function programs ──────────────────

    #[test]
    fn resolve_simple_arithmetic() {
        let i = Interner::new();
        let graph = make_graph_no_ctx_with_builtins(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", error_strings(&i, &result));
        let uid = last_local_id(&graph);
        assert!(result.try_resolution(uid).is_some());
        assert_eq!(tail_type(&result,uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_declared_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", error_strings(&i, &result));
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result,uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_with_user_provided_context() {
        let i = Interner::new();
        let graph = make_graph_no_ctx_with_builtins(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let mut user = FxHashMap::default();
        user.insert(QualifiedRef::root(i.intern("x")), Ty::Int);
        let result = infer(&i, &graph, &ext, &user);

        assert!(!result.has_errors(), "errors: {:?}", error_strings(&i, &result));
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result,uid).unwrap(), Ty::Int);
    }

    #[test]
    fn resolve_string_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@name", &[("name", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", error_strings(&i, &result));
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result,uid).unwrap(), Ty::String);
    }

    #[test]
    fn resolve_context_field_access() {
        let i = Interner::new();
        let obj_ty = Ty::Object(FxHashMap::from_iter([(i.intern("name"), Ty::String)]));
        let graph = make_graph_with_ctx_and_builtins(&i, "@user.name", &[("user", obj_ty)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(!result.has_errors(), "errors: {:?}", error_strings(&i, &result));
        let uid = last_local_id(&graph);
        assert_eq!(tail_type(&result,uid).unwrap(), Ty::String);
    }

    // -- Soundness: type errors detected --

    #[test]
    fn resolve_type_mismatch_detected() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert!(result.has_errors(), "should detect type mismatch");
    }

    // -- Context type resolution --

    #[test]
    fn resolve_context_types_populated() {
        let i = Interner::new();
        let graph = make_graph_with_ctx_and_builtins(&i, "@x", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let ctx_ref = graph.contexts[0].qualified_ref();
        assert_eq!(*result.context_type(&ctx_ref).unwrap(), Ty::Int);
    }

    // ── Completeness: valid inter-function calls ──────────────────────

    /// C1: A calls B with matching concrete types.
    #[test]
    fn inter_fn_simple_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "double",
                    "$x * 2",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C2: A calls B, B returns String.
    #[test]
    fn inter_fn_string_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
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
                    Constraint::Inferred,
                ),
                ("main", "add(1, 2)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C4: Chain of calls — A calls B, B calls C.
    #[test]
    fn inter_fn_chain_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "inc",
                    "$x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "double_inc",
                    "inc($x) + inc($x)",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C5: Function uses context and is called by another function.
    #[test]
    fn inter_fn_with_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
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
                    Constraint::Exact(Ty::String),
                ),
                ("main", "make_str()", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
    }

    /// C7: Caller uses return value in arithmetic.
    #[test]
    fn inter_fn_return_used_in_binop() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C8: Pipe syntax — value | fn.
    #[test]
    fn inter_fn_pipe_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "double",
                    "$x * 2",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C9: Function returning list.
    #[test]
    fn inter_fn_list_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C10: Function accepting and returning String.
    #[test]
    fn inter_fn_string_identity() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "echo",
                    "$s",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
    }

    /// C11: Multiple callers of the same function.
    #[test]
    fn inter_fn_multiple_callers() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "inc",
                    "$x + 1",
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
        assert_eq!(tail_type(&result,ids[1].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result,ids[2].1).unwrap(), Ty::Int);
    }

    /// C12: Calling function with bool return.
    #[test]
    fn inter_fn_bool_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "is_positive",
                    "$x > 0",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Bool);
    }

    /// C13: Deep call chain — A → B → C → D.
    #[test]
    fn inter_fn_deep_chain() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C14: Function that calls builtin and local function together.
    #[test]
    fn inter_fn_mixed_builtin_and_local() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
    }

    /// C15: Function result used as argument to another function.
    #[test]
    fn inter_fn_nested_call() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "inc",
                    "$x + 1",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// C16: Mutual recursion — A calls B, B calls A.
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
                    Constraint::Exact(Ty::Bool),
                ),
                (
                    "is_odd",
                    "is_even($n - 1)",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Bool);
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
                    Constraint::Exact(Ty::Int),
                ),
                ("main", "fib(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
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
                    Constraint::Inferred,
                ),
                ("main", "avg(1.0, 3.0)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Float);
    }

    // ── Soundness: invalid calls should be rejected ─────────────────

    /// S1: Wrong argument type.
    #[test]
    fn inter_fn_reject_wrong_arg_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
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
        let (result, _ids) = infer_multi(
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
        let (result, _ids) = infer_multi(
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_str", "\"hello\"", Some(vec![]), Constraint::Inferred),
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
        let (result, _ids) = infer_multi(
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
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Constraint::Exact(Ty::String)),
                ("main", "bad()", None, Constraint::Inferred),
            ],
            &[],
        );
        let main_id = ids[1].1;
        if !result.has_errors() {
            assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
        }
    }

    /// S7: Mutual recursion without declared types — must not stack overflow.
    #[test]
    fn inter_fn_mutual_recursion_no_declared_types() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
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
        // We don't assert success or failure — just that it terminates.
        let _ = result;
    }

    /// S8: Wrong type in pipe position.
    #[test]
    fn inter_fn_reject_wrong_pipe_type() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("get_name", "@name", Some(vec![]), Constraint::Inferred),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("f", "x", Some(vec![("x", Ty::Int)]), Constraint::Inferred),
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
        let (result, _ids) = infer_multi(
            &i,
            &[
                ("make_int", "42", Some(vec![]), Constraint::Inferred),
                ("make_str", "\"hi\"", Some(vec![]), Constraint::Inferred),
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
        let (result, _ids) = infer_multi(
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
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E2: Same name as builtin — local should shadow or coexist?
    #[test]
    fn inter_fn_name_shadows_builtin() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("len", "42", Some(vec![]), Constraint::Exact(Ty::Int)),
                ("main", "len()", None, Constraint::Inferred),
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E4: Two functions reading the same context.
    #[test]
    fn inter_fn_shared_context() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
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
                    Constraint::Exact(Ty::Int),
                ),
                ("main", "f(10)", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E6: Diamond dependency — A calls B and C, both call D.
    #[test]
    fn inter_fn_diamond_dependency() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E7: Function result piped through builtin chain.
    #[test]
    fn inter_fn_pipe_through_builtins() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("make_list", "[1, 2, 3]", Some(vec![]), Constraint::Inferred),
                (
                    "main",
                    "make_list() | iter | map(|x| -> x + 1) | collect | len",
                    None,
                    Constraint::Inferred,
                ),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E8: Function with effectful iterator return type.
    #[test]
    fn inter_fn_effectful_return() {
        let i = Interner::new();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let (result, ids) = infer_multi(
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
            tail_type(&result,main_id).unwrap(),
            Ty::List(Box::new(Ty::Int))
        );
    }

    /// E9: Function returning Option type.
    #[test]
    fn inter_fn_option_return() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E10: Three functions forming a pipeline.
    #[test]
    fn inter_fn_three_stage_pipeline() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "stage1",
                    "$x + 1",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "stage2",
                    "$x * 2",
                    Some(vec![("x", Ty::Int)]),
                    Constraint::Inferred,
                ),
                (
                    "stage3",
                    "$x - 1",
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
    }

    /// E11: All local functions are callers — no inter-function calls.
    #[test]
    fn inter_fn_independent_functions() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("a", "1 + 2", None, Constraint::Inferred),
                ("b", "\"hello\"", None, Constraint::Inferred),
            ],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result,ids[0].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result,ids[1].1).unwrap(), Ty::String);
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
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
    }

    // ════════════════════════════════════════════════════════════════
    // Soundness boundary tests
    // ════════════════════════════════════════════════════════════════

    /// B1: Caller tries to use return value as wrong type.
    #[test]
    fn boundary_caller_forces_wrong_return_type() {
        let i = Interner::new();
        let (result, _) = infer_multi(
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
    #[test]
    fn boundary_inconsistent_return_usage() {
        let i = Interner::new();
        let (result, _) = infer_multi(
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
    #[test]
    fn boundary_mutual_recursion_inferred_must_not_succeed_silently() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        assert!(
            result.has_errors(),
            "purely recursive return type should fail inference"
        );
    }

    /// B5: Effect soundness — function reading context must be Effectful.
    #[test]
    fn boundary_effectful_context_read_propagates() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("get_x", "@x", Some(vec![]), Constraint::Inferred),
                ("main", "get_x()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[1].1;
        let main_effect = &result.outcomes[&main_id].meta().effect;
        assert!(
            !main_effect.is_pure(),
            "caller of context-reading function must be effectful"
        );
    }

    /// B6: Effect soundness — pure function call should not taint caller.
    #[test]
    fn boundary_pure_function_stays_pure() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                (
                    "add",
                    "$x + $y",
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
        let main_effect = &result.outcomes[&main_id].meta().effect;
        assert!(
            main_effect.is_pure(),
            "caller of pure function should remain pure"
        );
    }

    /// B7: Effect soundness — context write is Effectful.
    #[test]
    fn boundary_context_write_is_effectful() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
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
        let set_effect = &result.outcomes[&set_id].meta().effect;
        assert!(
            !set_effect.writes.is_empty(),
            "context-writing function must have writes in effect"
        );
        assert!(
            set_effect
                .writes
                .contains(&QualifiedRef::root(i.intern("x"))),
            "writes should contain @x"
        );
    }

    /// B8: infer produces wrong type, should be caught.
    #[test]
    fn boundary_infer_wrong_type_catches() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("bad", "42", Some(vec![]), Constraint::Exact(Ty::String)),
                ("main", "bad()", None, Constraint::Inferred),
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

    /// B9: Param in output but not in input — must not silently succeed.
    #[test]
    fn boundary_orphan_param_in_output() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("mystery", "x", Some(vec![]), Constraint::Inferred),
                ("main", "mystery() + 1", None, Constraint::Inferred),
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

    /// B10: Chain through effectful function — effect must propagate transitively.
    #[test]
    fn boundary_transitive_effect_propagation() {
        let i = Interner::new();
        let eff_iter = Ty::Iterator(Box::new(Ty::Int), Effect::io());
        let (result, ids) = infer_multi(
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
        let use_id = ids[1].1;
        let use_effect = &result.outcomes[&use_id].meta().effect;
        assert!(
            !use_effect.is_pure(),
            "transitive effect must propagate"
        );
    }

    // ── body_effect (reads / writes) tests ─────────────────────────

    /// Direct context read is tracked in body_effect reads.
    #[test]
    fn body_effect_tracks_context_read() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("reader", "@x + @y", Some(vec![]), Constraint::Inferred)],
            &[("x", Ty::Int), ("y", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            let effect_set = extract_effect_set(resolution);
            assert!(
                effect_set
                    .reads
                    .contains(&QualifiedRef::root(i.intern("x"))),
                "should track @x read"
            );
            assert!(
                effect_set
                    .reads
                    .contains(&QualifiedRef::root(i.intern("y"))),
                "should track @y read"
            );
            assert!(
                effect_set.writes.is_empty(),
                "read-only function should have no writes"
            );
        }
    }

    /// Direct context write is tracked in body_effect writes.
    #[test]
    fn body_effect_tracks_context_write() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("writer", "@x = 42; @x", Some(vec![]), Constraint::Inferred)],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            let effect_set = extract_effect_set(resolution);
            assert!(
                effect_set
                    .writes
                    .contains(&QualifiedRef::root(i.intern("x"))),
                "should track @x write"
            );
            assert!(
                effect_set
                    .reads
                    .contains(&QualifiedRef::root(i.intern("x"))),
                "should also track @x read from tail"
            );
        }
    }

    /// Pure function has empty body_effect reads and writes.
    #[test]
    fn body_effect_empty_for_pure() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("pure_fn", "1 + 2", Some(vec![]), Constraint::Inferred)],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let fid = ids[0].1;
        if let Some(resolution) = result.try_resolution(fid) {
            let effect_set = extract_effect_set(resolution);
            assert!(effect_set.reads.is_empty());
            assert!(effect_set.writes.is_empty());
        }
    }

    /// Calling an effectful function propagates its effect to the caller.
    #[test]
    fn body_effect_propagates_from_callee() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("get_x", "@x", Some(vec![]), Constraint::Inferred),
                ("main", "get_x()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let get_x_id = ids[0].1;
        if let Some(resolution) = result.try_resolution(get_x_id) {
            let effect_set = extract_effect_set(resolution);
            assert!(effect_set.reads.contains(&QualifiedRef::root(i.intern("x"))));
        }
        let main_id = ids[1].1;
        if let Some(resolution) = result.try_resolution(main_id) {
            let effect_set = extract_effect_set(resolution);
            assert!(
                !effect_set.is_pure(),
                "caller of effectful function should not be pure"
            );
        }
    }

    // ── Extern function tests ─────────────────────────────────────

    /// Extern function should be callable from local functions.
    #[test]
    fn extern_fn_call_resolves() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, ids) = infer_with_extern(
            &i,
            &[("main", "fetch(42)", None, Constraint::Inferred)],
            &[fetch],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "extern call should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::String);
    }

    /// Extern function with wrong argument type should error.
    #[test]
    fn extern_fn_call_type_mismatch() {
        let i = Interner::new();
        let fetch = make_extern_fn(&i, "fetch", vec![Ty::Int], Ty::String);
        let (result, _) = infer_with_extern(
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
        let (result, ids) = infer_with_extern(
            &i,
            &[("main", "get_count() + 1", None, Constraint::Inferred)],
            &[get_count],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        let main_id = ids[0].1;
        assert_eq!(tail_type(&result,main_id).unwrap(), Ty::Int);
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
                ("use_add", "ext_add(1, 2)", None, Constraint::Inferred),
                ("use_greet", "ext_greet(\"hi\")", None, Constraint::Inferred),
            ],
            &[add, greet],
            &[],
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "should resolve: {errs:?}");
        assert_eq!(tail_type(&result,ids[0].1).unwrap(), Ty::Int);
        assert_eq!(tail_type(&result,ids[1].1).unwrap(), Ty::String);
    }

    // ── Effect constraint tests ──────────────────────────────────────

    /// Pure function with pure constraint: should pass.
    #[test]
    fn effect_constraint_pure_passes() {
        let i = Interner::new();
        let (result, fid) = infer_with_effect(&i, "1 + 2", &[], EffectSet::default());
        let errs = error_strings(&i, &result);
        assert!(
            errs.is_empty(),
            "pure function should pass pure constraint: {errs:?}"
        );
        assert!(result.try_resolution(fid).is_some());
    }

    /// Context-reading function with pure constraint: should reject.
    #[test]
    fn effect_constraint_context_read_rejected_by_pure() {
        let i = Interner::new();
        let (result, _fid) = infer_with_effect(
            &i,
            "@x + 1",
            &[("x", Ty::Int)],
            EffectSet::default(),
        );
        let errs = error_strings(&i, &result);
        assert!(
            !errs.is_empty(),
            "context read should violate pure constraint"
        );
        assert!(errs.iter().any(|e| e.contains("effect constraint violated")));
    }

    /// Context-reading function with read-allowed constraint: should pass.
    #[test]
    fn effect_constraint_context_read_passes() {
        let i = Interner::new();
        let qref = QualifiedRef {
            name: i.intern("x"),
            namespace: None,
        };
        let allowed = EffectSet {
            reads: std::collections::BTreeSet::from([qref]),
            ..EffectSet::default()
        };
        let (result, fid) = infer_with_effect(&i, "@x + 1", &[("x", Ty::Int)], allowed);
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "read should be allowed: {errs:?}");
        assert!(result.try_resolution(fid).is_some());
    }

    /// Context-writing function with read-only constraint: should reject.
    #[test]
    fn effect_constraint_context_write_rejected_by_read_only() {
        let i = Interner::new();
        let qref = QualifiedRef {
            name: i.intern("x"),
            namespace: None,
        };
        let allowed = EffectSet {
            reads: std::collections::BTreeSet::from([qref]),
            ..EffectSet::default()
        };
        let (result, _fid) = infer_with_effect(
            &i,
            "@x = 42; @x",
            &[("x", Ty::Int)],
            allowed,
        );
        let errs = error_strings(&i, &result);
        assert!(
            !errs.is_empty(),
            "context write should violate read-only constraint"
        );
        assert!(errs.iter().any(|e| e.contains("effect constraint violated")));
    }

    /// Context-writing function with write-allowed constraint: should pass.
    #[test]
    fn effect_constraint_context_write_passes() {
        let i = Interner::new();
        let qref = QualifiedRef {
            name: i.intern("x"),
            namespace: None,
        };
        let allowed = EffectSet {
            reads: std::collections::BTreeSet::from([qref]),
            writes: std::collections::BTreeSet::from([qref]),
            ..EffectSet::default()
        };
        let (result, fid) = infer_with_effect(
            &i,
            "@x = 42; @x",
            &[("x", Ty::Int)],
            allowed,
        );
        let errs = error_strings(&i, &result);
        assert!(errs.is_empty(), "write should be allowed: {errs:?}");
        assert!(result.try_resolution(fid).is_some());
    }

    /// Writing to context not in allowed set: should reject.
    #[test]
    fn effect_constraint_write_to_unknown_rejected() {
        let i = Interner::new();
        let qref_x = QualifiedRef {
            name: i.intern("x"),
            namespace: None,
        };
        let allowed = EffectSet {
            reads: std::collections::BTreeSet::from([qref_x]),
            writes: std::collections::BTreeSet::from([qref_x]),
            ..EffectSet::default()
        };
        let (result, _fid) = infer_with_effect(
            &i,
            "@y = 42; @x",
            &[("x", Ty::Int), ("y", Ty::Int)],
            allowed,
        );
        let errs = error_strings(&i, &result);
        assert!(
            !errs.is_empty(),
            "write to @y should violate constraint"
        );
        assert!(errs.iter().any(|e| e.contains("effect constraint violated")));
    }

    /// No effect constraint (None): should always pass regardless of effects.
    #[test]
    fn effect_constraint_none_always_passes() {
        let i = Interner::new();
        let (result, _ids) = infer_multi(
            &i,
            &[("test", "@x = 42; @x", None, Constraint::Inferred)],
            &[("x", Ty::Int)],
        );
        let errs = error_strings(&i, &result);
        assert!(
            errs.is_empty(),
            "no effect constraint should pass: {errs:?}"
        );
    }

    // ════════════════════════════════════════════════════════════════
    // Context extraction tests
    // ════════════════════════════════════════════════════════════════

    // -- Completeness: contexts correctly extracted and typed --

    /// Single context read — type inferred from usage.
    #[test]
    fn context_extract_single_read() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        let params = &result.fn_params[&fid];
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name, QualifiedRef::root(i.intern("x")));
        assert_eq!(params[0].ty, Ty::Int);
    }

    /// Multiple contexts — both extracted.
    #[test]
    fn context_extract_multiple() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + @y");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 2);
        let names: FxHashSet<QualifiedRef> = result.all_params.iter().map(|p| p.name).collect();
        assert!(names.contains(&QualifiedRef::root(i.intern("x"))));
        assert!(names.contains(&QualifiedRef::root(i.intern("y"))));
    }

    /// Context store — writes tracked in effect.
    #[test]
    fn context_extract_store_writes() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x = 42; @x", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        let effect = &result.outcomes[&fid].meta().effect;
        let qref = QualifiedRef::root(i.intern("x"));
        assert!(effect.writes.contains(&qref), "should track context write");
        assert!(effect.reads.contains(&qref), "should track context read");
    }

    /// Context inside nested block — still extracted.
    #[test]
    fn context_extract_nested_block() {
        let i = Interner::new();
        let graph = make_graph(&i, "{ @x + 1 }");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].name, QualifiedRef::root(i.intern("x")));
    }

    /// Context inside lambda — still extracted.
    #[test]
    fn context_extract_in_lambda() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(
            &i,
            "@items | map(|x| -> x + @offset) | collect",
            &[("items", Ty::List(Box::new(Ty::Int))), ("offset", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        let effect = &result.outcomes[&fid].meta().effect;
        let qref = QualifiedRef::root(i.intern("offset"));
        assert!(effect.reads.contains(&qref), "lambda context read should be tracked");
    }

    // -- Complete/Incomplete boundary --

    /// Declared Exact context → Complete.
    #[test]
    fn context_declared_exact_is_complete() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        assert!(result.outcomes[&fid].is_complete(), "declared Exact context should be Complete");
    }

    /// Declared Inferred context → Complete (type inferred via fresh var).
    #[test]
    fn context_declared_inferred_is_complete() {
        let i = Interner::new();
        let contexts = vec![Context {
            name: i.intern("x"),
            namespace: None,
            constraint: Constraint::Inferred,
        }];
        let graph = CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(vec![Function {
                id: FunctionId::alloc(),
                name: i.intern("test"),
                namespace: None,
                kind: FnKind::Local(SourceCode {
                    name: i.intern("test"),
                    source: i.intern("@x + 1"),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: None,
                    output: Constraint::Inferred,
                    effect: None,
                },
            }]),
            contexts: Freeze::new(contexts),
        };
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        assert!(result.outcomes[&fid].is_complete(), "Inferred context should still be Complete");
        // The inferred type should be Int (from @x + 1).
        let qref = QualifiedRef::root(i.intern("x"));
        assert_eq!(*result.context_type(&qref).unwrap(), Ty::Int);
    }

    /// Undeclared context (not in graph at all) → Incomplete.
    #[test]
    fn context_undeclared_is_incomplete() {
        let i = Interner::new();
        // No contexts declared, but source uses @x.
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        assert!(!result.outcomes[&fid].is_complete(), "undeclared context should be Incomplete");
    }

    /// User-provided context type → Complete.
    #[test]
    fn context_user_provided_is_complete() {
        let i = Interner::new();
        let graph = make_graph(&i, "@x + 1");
        let ext = extract::extract(&i, &graph);
        let mut user = FxHashMap::default();
        user.insert(QualifiedRef::root(i.intern("x")), Ty::Int);
        let result = infer(&i, &graph, &ext, &user);

        let fid = graph.functions[0].id;
        assert!(result.outcomes[&fid].is_complete(), "user-provided context should be Complete");
    }

    // -- Soundness: type mismatch detected --

    /// Declared context type conflicts with usage → Incomplete.
    #[test]
    fn context_type_mismatch_is_incomplete() {
        let i = Interner::new();
        // @x is String but used in arithmetic.
        let graph = make_graph_with_ctx(&i, "@x + 1", &[("x", Ty::String)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        let fid = graph.functions[0].id;
        assert!(!result.outcomes[&fid].is_complete(), "type mismatch should be Incomplete");
    }

    // ════════════════════════════════════════════════════════════════
    // Param extraction tests
    // ════════════════════════════════════════════════════════════════

    /// Single $param — discovered in extern_params.
    #[test]
    fn param_extract_single() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "$x + 1", Some(vec![("x", Ty::Int)]), Constraint::Inferred)],
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
            &[("test", "$x + $y", Some(vec![("x", Ty::Int), ("y", Ty::Int)]), Constraint::Inferred)],
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
            &[("test", "$x + 1", Some(vec![("x", Ty::Int)]), Constraint::Inferred)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::Int);
    }

    /// Param used in string concat → inferred as String.
    #[test]
    fn param_type_inferred_string() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", r#"$x + "hello""#, Some(vec![("x", Ty::String)]), Constraint::Inferred)],
            &[],
        );
        let fid = ids[0].1;
        let meta = result.outcomes[&fid].meta();
        assert_eq!(meta.params[0].ty, Ty::String);
    }

    // ════════════════════════════════════════════════════════════════
    // Type constraint tests
    // ════════════════════════════════════════════════════════════════

    /// Exact output constraint satisfied → Complete.
    #[test]
    fn type_constraint_exact_satisfied() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", "42", Some(vec![]), Constraint::Exact(Ty::Int))],
            &[],
        );
        let fid = ids[0].1;
        assert!(result.outcomes[&fid].is_complete(), "matching Exact output should be Complete");
    }

    /// Exact output constraint violated → Incomplete.
    #[test]
    fn type_constraint_exact_violated() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", r#""hello""#, Some(vec![]), Constraint::Exact(Ty::Int))],
            &[],
        );
        let fid = ids[0].1;
        assert!(
            !result.outcomes[&fid].is_complete(),
            "String body with Exact(Int) constraint should be Incomplete"
        );
    }

    /// Inferred output → always Complete (no constraint to violate).
    #[test]
    fn type_constraint_inferred_always_complete() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[("test", r#""hello""#, Some(vec![]), Constraint::Inferred)],
            &[],
        );
        let fid = ids[0].1;
        assert!(result.outcomes[&fid].is_complete());
    }

    // ════════════════════════════════════════════════════════════════
    // Transitive effect tests
    // ════════════════════════════════════════════════════════════════

    /// Callee reads context → caller's transitive effect includes the read.
    #[test]
    fn transitive_effect_callee_read_propagates() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("reader", "@x", Some(vec![]), Constraint::Inferred),
                ("caller", "reader()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let caller_id = ids[1].1;
        let effect = &result.outcomes[&caller_id].meta().effect;
        let qref = QualifiedRef::root(i.intern("x"));
        assert!(
            effect.reads.contains(&qref),
            "caller should inherit callee's context read in transitive effect"
        );
    }

    /// Callee writes context → caller's transitive effect includes the write.
    #[test]
    fn transitive_effect_callee_write_propagates() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("writer", "@x = 42; @x", Some(vec![]), Constraint::Inferred),
                ("caller", "writer()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let caller_id = ids[1].1;
        let effect = &result.outcomes[&caller_id].meta().effect;
        let qref = QualifiedRef::root(i.intern("x"));
        assert!(
            effect.writes.contains(&qref),
            "caller should inherit callee's context write in transitive effect"
        );
    }

    /// Deep chain: A → B → C reads @x. A's transitive effect should include the read.
    #[test]
    fn transitive_effect_deep_chain() {
        let i = Interner::new();
        let (result, ids) = infer_multi(
            &i,
            &[
                ("leaf", "@x", Some(vec![]), Constraint::Inferred),
                ("mid", "leaf()", Some(vec![]), Constraint::Inferred),
                ("top", "mid()", None, Constraint::Inferred),
            ],
            &[("x", Ty::Int)],
        );
        let top_id = ids[2].1;
        let effect = &result.outcomes[&top_id].meta().effect;
        let qref = QualifiedRef::root(i.intern("x"));
        assert!(
            effect.reads.contains(&qref),
            "top-level caller should inherit transitive effect through chain"
        );
    }

    /// Effect constraint with transitive effect — callee reads @x,
    /// caller has pure constraint → should be Incomplete.
    #[test]
    fn effect_constraint_transitive_violation() {
        let i = Interner::new();
        let qref_x = QualifiedRef::root(i.intern("x"));

        // "reader" reads @x (no constraint).
        // "caller" calls reader() but has pure constraint.
        let contexts: Vec<Context> = vec![Context {
            name: i.intern("x"),
            namespace: None,
            constraint: Constraint::Exact(Ty::Int),
        }];
        let mut functions = crate::builtins::standard_builtins(&i);
        let reader_id = FunctionId::alloc();
        functions.push(Function {
            id: reader_id,
            name: i.intern("reader"),
            namespace: None,
            kind: FnKind::Local(SourceCode {
                name: i.intern("reader"),
                source: i.intern("@x"),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: Some(Signature { params: vec![] }),
                output: Constraint::Inferred,
                effect: None,
            },
        });
        let caller_id = FunctionId::alloc();
        functions.push(Function {
            id: caller_id,
            name: i.intern("caller"),
            namespace: None,
            kind: FnKind::Local(SourceCode {
                name: i.intern("caller"),
                source: i.intern("reader()"),
                kind: SourceKind::Script,
            }),
            constraint: FnConstraint {
                signature: None,
                output: Constraint::Inferred,
                effect: Some(EffectSet::default()), // pure constraint
            },
        });
        let graph = CompilationGraph {
            namespaces: Freeze::new(vec![]),
            functions: Freeze::new(functions),
            contexts: Freeze::new(contexts),
        };
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext, &FxHashMap::default());

        // reader should be Complete (no constraint).
        assert!(result.outcomes[&reader_id].is_complete(), "reader has no constraint");
        // caller should be Incomplete — transitive read violates pure.
        assert!(
            !result.outcomes[&caller_id].is_complete(),
            "caller with pure constraint should be Incomplete when callee reads context"
        );
    }
}
