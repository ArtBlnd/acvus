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
    pub name: Astr,
    pub ty: Ty,
}

/// Inferred metadata for a single function.
#[derive(Debug, Clone)]
pub struct FunctionMeta {
    /// Fully resolved Ty::Fn for this function.
    pub ty: Ty,
    /// Named parameters (free_params from source zipped with signature types).
    pub params: Vec<Param>,
    /// Transitive effect set (reads/writes with ContextId).
    /// Includes both direct access and access through callees.
    pub effect: EffectSet,
}

/// Phase 1 output: inferred context parameters and function types.
#[derive(Debug)]
pub struct InferResult {
    /// Per-function inferred parameters.
    pub fn_params: FxHashMap<FunctionId, Vec<InferredParam>>,
    /// All context parameters across all functions, deduplicated.
    /// If the same context name appears in multiple units, the types must agree.
    pub all_params: Vec<InferredParam>,
    /// Per-function inferred metadata (type, params, effect).
    pub functions: FxHashMap<FunctionId, FunctionMeta>,
}

// ── Call graph + SCC ─────────────────────────────────────────────────

/// Build a call graph: for each local function, which other local functions
/// does it reference by name in its body?
fn build_call_graph(
    graph: &CompilationGraph,
    extract: &ExtractResult,
) -> FxHashMap<FunctionId, Vec<FunctionId>> {
    // name → FunctionId for local functions only.
    let name_to_id: FxHashMap<Astr, FunctionId> = graph
        .functions
        .iter()
        .filter(|f| matches!(f.kind, FnKind::Local(_)))
        .map(|f| (f.name, f.id))
        .collect();

    let mut edges: FxHashMap<FunctionId, Vec<FunctionId>> = FxHashMap::default();

    for func in graph.functions.iter() {
        let FnKind::Local(_) = &func.kind else { continue; };
        let Some(parsed) = extract.parsed.get(&func.id) else { continue; };

        let mut callees = Vec::new();
        let names: Vec<Astr> = match parsed {
            ParsedSource::Script(script) => collect_value_refs_script(script),
            ParsedSource::Template(template) => collect_value_refs_template(template),
        };
        for name in names {
            if let Some(&callee_id) = name_to_id.get(&name) {
                if callee_id != func.id {
                    // Skip self-calls for graph building (self-recursion is
                    // handled within an SCC naturally).
                    if !callees.contains(&callee_id) {
                        callees.push(callee_id);
                    }
                }
            }
        }
        edges.insert(func.id, callees);
    }
    edges
}

/// Collect all RefKind::Value identifiers from a script AST.
fn collect_value_refs_script(script: &acvus_ast::Script) -> Vec<Astr> {
    use acvus_ast::*;
    let mut refs = Vec::new();
    for stmt in &script.stmts {
        match stmt {
            Stmt::Bind { expr, .. } | Stmt::ContextStore { expr, .. } => {
                collect_value_refs_expr(expr, &mut refs);
            }
            Stmt::Expr(expr) => collect_value_refs_expr(expr, &mut refs),
        }
    }
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
                for n in &arm.body { collect_value_refs_node(n, refs); }
            }
            if let Some(ca) = &mb.catch_all {
                for n in &ca.body { collect_value_refs_node(n, refs); }
            }
        }
        acvus_ast::Node::IterBlock(ib) => {
            collect_value_refs_expr(&ib.source, refs);
            for n in &ib.body { collect_value_refs_node(n, refs); }
            if let Some(ca) = &ib.catch_all {
                for n in &ca.body { collect_value_refs_node(n, refs); }
            }
        }
    }
}

fn collect_value_refs_expr(expr: &acvus_ast::Expr, refs: &mut Vec<Astr>) {
    use acvus_ast::*;
    match expr {
        Expr::Ident { name, ref_kind: RefKind::Value, .. } => refs.push(*name),
        Expr::Ident { .. } | Expr::Literal { .. } => {}
        Expr::BinaryOp { left, right, .. } | Expr::Pipe { left, right, .. } => {
            collect_value_refs_expr(left, refs);
            collect_value_refs_expr(right, refs);
        }
        Expr::UnaryOp { operand, .. } => collect_value_refs_expr(operand, refs),
        Expr::FieldAccess { object, .. } => collect_value_refs_expr(object, refs),
        Expr::FuncCall { func, args, .. } => {
            collect_value_refs_expr(func, refs);
            for a in args { collect_value_refs_expr(a, refs); }
        }
        Expr::Lambda { body, .. } => collect_value_refs_expr(body, refs),
        Expr::Paren { inner, .. } => collect_value_refs_expr(inner, refs),
        Expr::List { head, tail, .. } => {
            for e in head.iter().chain(tail.iter()) { collect_value_refs_expr(e, refs); }
        }
        Expr::Object { fields, .. } => {
            for f in fields { collect_value_refs_expr(&f.value, refs); }
        }
        Expr::Range { start, end, .. } => {
            collect_value_refs_expr(start, refs);
            collect_value_refs_expr(end, refs);
        }
        Expr::Tuple { elements, .. } => {
            for e in elements {
                if let TupleElem::Expr(e) = e { collect_value_refs_expr(e, refs); }
            }
        }
        Expr::Group { elements, .. } => {
            for e in elements { collect_value_refs_expr(e, refs); }
        }
        Expr::Variant { payload: Some(inner), .. } => collect_value_refs_expr(inner, refs),
        Expr::Variant { payload: None, .. } => {}
        Expr::Block { stmts, tail, .. } => {
            for stmt in stmts {
                match stmt {
                    Stmt::Bind { expr, .. } | Stmt::ContextStore { expr, .. } => {
                        collect_value_refs_expr(expr, refs);
                    }
                    Stmt::Expr(e) => collect_value_refs_expr(e, refs),
                }
            }
            collect_value_refs_expr(tail, refs);
        }
    }
}

/// Tarjan's SCC algorithm. Returns SCCs in reverse topological order
/// (leaf SCCs first — dependencies before dependents).
fn tarjan_scc(ids: &[FunctionId], edges: &FxHashMap<FunctionId, Vec<FunctionId>>) -> Vec<Vec<FunctionId>> {
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
                    strongconnect(w, edges, index_counter, stack, on_stack, index, lowlink, result);
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
                if w == v { break; }
            }
            result.push(component);
        }
    }

    for &id in ids {
        if !index.contains_key(&id) {
            strongconnect(id, edges, &mut index_counter, &mut stack, &mut on_stack, &mut index, &mut lowlink, &mut result);
        }
    }

    // Tarjan produces SCCs in reverse topological order already.
    result
}

// ── Inference ───────────────────────────────────────────────────────

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
) -> InferResult {
    let mut subst = TySubst::new();
    let mut fn_params: FxHashMap<FunctionId, Vec<InferredParam>> = FxHashMap::default();
    let mut fn_bind_params: FxHashMap<FunctionId, Vec<Param>> = FxHashMap::default();
    // Direct context access per function (Astr), collected from typeck results.
    let mut fn_direct_reads: FxHashMap<FunctionId, FxHashSet<Astr>> = FxHashMap::default();
    let mut fn_direct_writes: FxHashMap<FunctionId, FxHashSet<Astr>> = FxHashMap::default();

    // Resolved function types — populated as SCCs complete.
    let mut resolved_fn_types: FxHashMap<Astr, Ty> = crate::builtins::builtin_fn_types(interner);

    // Collect known context types.
    let known_ctx: FxHashMap<Astr, Ty> = graph
        .contexts
        .iter()
        .filter_map(|ctx| {
            if let Constraint::Exact(ty) = &ctx.constraint {
                Some((ctx.name, ty.clone()))
            } else {
                None
            }
        })
        .collect();

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
            let Some(fn_ref) = extract.fn_refs.get(&fid) else { continue; };
            let Some(parsed) = extract.parsed.get(&fid) else { continue; };

            // Build context types.
            let mut ctx_types: FxHashMap<Astr, Ty> = FxHashMap::default();
            let mut unknown_vars: FxHashMap<Astr, Ty> = FxHashMap::default();

            for &name in &fn_ref.context_reads {
                if let Some(ty) = known_ctx.get(&name) {
                    ctx_types.insert(name, ty.clone());
                } else {
                    let var = subst.fresh_param();
                    unknown_vars.insert(name, var.clone());
                    ctx_types.insert(name, var);
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
                // Collect direct context access from typeck.
                fn_direct_reads.insert(fid, unchecked.body_reads.clone());
                fn_direct_writes.insert(fid, unchecked.body_writes.clone());

                let bind: Vec<Param> = unchecked
                    .inferred_params
                    .iter()
                    .map(|(name, ty)| Param::new(*name, subst.resolve(ty)))
                    .collect();
                fn_bind_params.insert(fid, bind);
            }

            let params: Vec<InferredParam> = unknown_vars
                .into_iter()
                .map(|(name, var)| InferredParam { name, ty: subst.resolve(&var) })
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

    // ── STEP 3: Collect results ─────────────────────────────────────

    let mut all_map: FxHashMap<Astr, Ty> = FxHashMap::default();
    for params in fn_params.values() {
        for param in params {
            all_map.entry(param.name).or_insert_with(|| param.ty.clone());
        }
    }
    let all_params: Vec<InferredParam> = all_map
        .into_iter()
        .map(|(name, ty)| InferredParam { name, ty })
        .collect();

    let mut functions: FxHashMap<FunctionId, FunctionMeta> = FxHashMap::default();

    for &fid in &local_ids {
        let ret = fn_ret_vars.get(&fid).map(|r| subst.resolve(r)).unwrap_or_else(Ty::error);
        let effect = fn_effect_vars.get(&fid).map(|e| subst.resolve_effect(e)).unwrap_or_else(Effect::pure);
        let bind: Vec<Param> = fn_bind_params
            .get(&fid)
            .map(|b| b.iter().map(|p| Param::new(p.name, subst.resolve(&p.ty))).collect())
            .unwrap_or_default();

        let fn_ty = Ty::Fn {
            params: bind.clone(),
            ret: Box::new(ret),
            captures: vec![],
            effect,
        };
        functions.insert(fid, FunctionMeta {
            ty: fn_ty,
            params: bind,
            effect: EffectSet::default(), // filled in step 4
        });
    }

    // ── STEP 4: Compute transitive effects ─────────────────────────
    //
    // Build name→ContextId mapping, convert direct accesses (Astr) to ContextId,
    // then propagate through call graph in SCC order (fixpoint within each SCC).

    let name_to_ctx_id: FxHashMap<Astr, ContextId> = graph
        .contexts
        .iter()
        .map(|c| (c.name, c.id))
        .collect();

    // Convert direct accesses: Astr → ContextId and store in function meta.
    // Also union parameter effects (over-approximation: any effectful param contributes).
    for &fid in &local_ids {
        let reads: BTreeSet<ContextId> = fn_direct_reads
            .get(&fid)
            .map(|names| {
                names.iter().filter_map(|n| name_to_ctx_id.get(n).copied()).collect()
            })
            .unwrap_or_default();
        let writes: BTreeSet<ContextId> = fn_direct_writes
            .get(&fid)
            .map(|names| {
                names.iter().filter_map(|n| name_to_ctx_id.get(n).copied()).collect()
            })
            .unwrap_or_default();
        let mut effect = EffectSet { reads, writes, io: false };

        // Union effects carried by parameters (Iterator<T,E>, Fn{effect}, Sequence<T,_,E>).
        if let Some(meta) = functions.get(&fid) {
            for param in &meta.params {
                if let Some(param_effect) = param.ty.carried_effect() {
                    if let Effect::Resolved(param_set) = param_effect {
                        effect = effect.union(param_set);
                    }
                }
            }
        }

        if let Some(meta) = functions.get_mut(&fid) {
            meta.effect = effect;
        }
    }

    // Propagate transitive effects through call graph in SCC topological order.
    // Within an SCC: fixpoint iteration (union is monotonic, bounded by finite context set).
    for scc in &sccs {
        loop {
            let mut changed = false;
            for &fid in scc {
                if let Some(callees) = call_graph.get(&fid) {
                    for &callee_id in callees {
                        let callee_effect = functions.get(&callee_id)
                            .map(|m| m.effect.clone())
                            .unwrap_or_default();
                        let current = functions.get(&fid)
                            .map(|m| m.effect.clone())
                            .unwrap_or_default();
                        let merged = current.union(&callee_effect);
                        if merged != current {
                            if let Some(meta) = functions.get_mut(&fid) {
                                meta.effect = merged;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    InferResult {
        fn_params,
        all_params,
        functions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::extract;
    use acvus_utils::{Freeze, Interner};

    fn make_graph(interner: &Interner, source: &str) -> CompilationGraph {
        CompilationGraph {
            functions: Freeze::new(vec![Function {
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
                id: ContextId::alloc(),
                name: interner.intern(name),
                constraint: Constraint::Exact(ty.clone()),
            })
            .collect();
        CompilationGraph {
            functions: Freeze::new(vec![Function {
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
        let result = infer(&i, &graph, &ext);

        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].name, i.intern("x"));
        assert_eq!(result.all_params[0].ty, Ty::Int);
    }

    #[test]
    fn infer_string_from_concat() {
        let i = Interner::new();
        let graph = make_graph(&i, r#"@x + "hello""#);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].ty, Ty::String);
    }

    #[test]
    fn infer_multiple_params_with_literal() {
        let i = Interner::new();
        // @a + @b + 1 — the literal 1 forces both to Int.
        let graph = make_graph(&i, "@a + @b + 1");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

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
        let result = infer(&i, &graph, &ext);

        assert_eq!(result.all_params.len(), 2);
        // At least one should still be a Param (unresolved).
        let has_param = result.all_params.iter().any(|p| matches!(p.ty, Ty::Param { .. }));
        assert!(has_param, "without a literal, type should remain unresolved");
    }

    // -- Completeness: known context not re-inferred --

    #[test]
    fn infer_skips_known_context() {
        let i = Interner::new();
        let graph = make_graph_with_ctx(&i, "@x + @y", &[("x", Ty::Int)]);
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        // Only @y should be inferred, @x is already known.
        assert_eq!(result.all_params.len(), 1);
        assert_eq!(result.all_params[0].name, i.intern("y"));
        assert_eq!(result.all_params[0].ty, Ty::Int);
    }

    // -- Soundness: no false inferences --

    #[test]
    fn infer_no_contexts_empty() {
        let i = Interner::new();
        let graph = make_graph(&i, "1 + 2");
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

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
            let sig = params.as_ref().map(|p| {
                Signature {
                    params: p.iter().enumerate().map(|(i, ty)| {
                        Param::new(interner.intern(&format!("_{i}")), ty.clone())
                    }).collect(),
                }
            });
            functions.push(Function {
                id: fid,
                name: aname,
                kind: FnKind::Local(SourceCode {
                    name: aname,
                    source: interner.intern(source),
                    kind: SourceKind::Script,
                }),
                constraint: FnConstraint {
                    signature: sig,
                    output: Constraint::Inferred,
                },
            });
            ids.push((aname, fid));
        }
        let contexts = ctx.iter().map(|(name, ty)| Context {
            id: ContextId::alloc(),
            name: interner.intern(name),
            constraint: Constraint::Exact(ty.clone()),
        }).collect();
        let graph = CompilationGraph {
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
        let result = infer(&i, &graph, &ext);

        let fid = ids[0].1;
        let effect = &result.functions[&fid].effect;
        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;
        let ctx_y = graph.contexts.iter().find(|c| c.name == i.intern("y")).unwrap().id;
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
        let result = infer(&i, &graph, &ext);

        let fid = ids[0].1;
        let effect = &result.functions[&fid].effect;
        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;
        assert!(effect.writes.contains(&ctx_x), "should have @x in writes");
        assert!(effect.reads.contains(&ctx_x), "should also have @x in reads (tail)");
    }

    #[test]
    fn effect_pure_function_empty() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[("pure_fn", "1 + 2", Some(vec![]))],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let fid = ids[0].1;
        let effect = &result.functions[&fid].effect;
        assert!(effect.is_pure(), "pure function should have empty effect");
    }

    #[test]
    fn effect_transitive_through_callee() {
        let i = Interner::new();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("get_x", "@x", Some(vec![])),
                ("main", "get_x()", None),
            ],
            &[("x", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;

        // get_x directly reads @x
        let get_x_effect = &result.functions[&ids[0].1].effect;
        assert!(get_x_effect.reads.contains(&ctx_x));

        // main calls get_x → transitive read of @x
        let main_effect = &result.functions[&ids[1].1].effect;
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
        let result = infer(&i, &graph, &ext);

        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;
        // top → mid → read_x → @x. All should have @x in reads.
        for (_, fid) in &ids {
            let effect = &result.functions[fid].effect;
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
        let result = infer(&i, &graph, &ext);

        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;
        // main calls pure_fn (not read_x) → should NOT have @x
        let main_effect = &result.functions[&ids[2].1].effect;
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
        let result = infer(&i, &graph, &ext);

        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;
        let ctx_y = graph.contexts.iter().find(|c| c.name == i.intern("y")).unwrap().id;

        // a and b are in the same SCC.
        // After fixpoint: both should have reads = {@x, @y}.
        let a_effect = &result.functions[&ids[0].1].effect;
        let b_effect = &result.functions[&ids[1].1].effect;
        assert!(a_effect.reads.contains(&ctx_x), "a should read @x (direct)");
        assert!(a_effect.reads.contains(&ctx_y), "a should read @y (transitive from b)");
        assert!(b_effect.reads.contains(&ctx_x), "b should read @x (transitive from a)");
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
        let result = infer(&i, &graph, &ext);

        let ctx_x = graph.contexts.iter().find(|c| c.name == i.intern("x")).unwrap().id;

        // writer writes @x → caller should transitively inherit
        let caller_effect = &result.functions[&ids[1].1].effect;
        assert!(caller_effect.writes.contains(&ctx_x), "transitive write");
        assert!(caller_effect.reads.contains(&ctx_x), "transitive read");
    }

    // ── Parameter effect union tests ────────────────────────────────

    #[test]
    fn effect_param_effectful_iterator_propagates() {
        // Function takes an effectful iterator param → function is effectful.
        let i = Interner::new();
        let ctx_x = ContextId::alloc();
        let iter_effect = Effect::Resolved(EffectSet {
            reads: [ctx_x].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            // takes an iterator param, returns element
            &[("consumer", "_0", Some(vec![
                Ty::Iterator(Box::new(Ty::Int), iter_effect),
            ]))],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let effect = &result.functions[&ids[0].1].effect;
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
            &[("processor", "_0", Some(vec![
                Ty::Iterator(Box::new(Ty::Int), Effect::pure()),
            ]))],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let effect = &result.functions[&ids[0].1].effect;
        assert!(effect.is_pure(), "pure iterator param should not make function effectful");
    }

    #[test]
    fn effect_param_effectful_fn_propagates() {
        // Function takes a Fn param with effect → function inherits it.
        let i = Interner::new();
        let ctx_y = ContextId::alloc();
        let fn_effect = Effect::Resolved(EffectSet {
            writes: [ctx_y].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            &[("caller", "_0", Some(vec![
                Ty::Fn {
                    params: vec![],
                    ret: Box::new(Ty::Int),
                    captures: vec![],
                    effect: fn_effect,
                },
            ]))],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let effect = &result.functions[&ids[0].1].effect;
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
        let ctx_b = ContextId::alloc();
        let fn_effect = Effect::Resolved(EffectSet {
            writes: [ctx_b].into_iter().collect(),
            ..Default::default()
        });
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("read_a", "@a", Some(vec![])),
                ("use_fn", "_0", Some(vec![
                    Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: fn_effect,
                    },
                ])),
                // caller invokes both → gets both effects transitively
                ("caller", "read_a() + use_fn(read_a)", None),
            ],
            &[("a", Ty::Int)],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        let ctx_a = graph.contexts.iter().find(|c| c.name == i.intern("a")).unwrap().id;
        let caller_effect = &result.functions[&ids[2].1].effect;
        assert!(caller_effect.reads.contains(&ctx_a), "transitive read of @a from read_a");
        assert!(caller_effect.writes.contains(&ctx_b), "param Fn's write of @b from use_fn");
    }

    #[test]
    fn effect_multiple_effectful_params_union() {
        // Function takes two effectful Fn params via two separate functions.
        let i = Interner::new();
        let ctx_x = ContextId::alloc();
        let ctx_y = ContextId::alloc();
        let (graph, ids) = make_multi_graph(
            &i,
            &[
                ("fn_a", "_0", Some(vec![
                    Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: Effect::Resolved(EffectSet {
                            reads: [ctx_x].into_iter().collect(),
                            ..Default::default()
                        }),
                    },
                ])),
                ("fn_b", "_0", Some(vec![
                    Ty::Fn {
                        params: vec![],
                        ret: Box::new(Ty::Int),
                        captures: vec![],
                        effect: Effect::Resolved(EffectSet {
                            writes: [ctx_y].into_iter().collect(),
                            ..Default::default()
                        }),
                    },
                ])),
                // caller gets both effects transitively
                ("caller", "fn_a(fn_b) + fn_b(fn_a)", None),
            ],
            &[],
        );
        let ext = extract::extract(&i, &graph);
        let result = infer(&i, &graph, &ext);

        // fn_a has param effect reads @x
        let fn_a_effect = &result.functions[&ids[0].1].effect;
        assert!(fn_a_effect.reads.contains(&ctx_x), "fn_a param's read");

        // fn_b has param effect writes @y
        let fn_b_effect = &result.functions[&ids[1].1].effect;
        assert!(fn_b_effect.writes.contains(&ctx_y), "fn_b param's write");
    }
}
