//! Incremental compilation graph.
//!
//! Manages per-function extract/infer caches with dirty tracking.
//! On source change: re-extract -> diff call edges -> re-SCC if needed ->
//! re-infer dirty SCCs (with early cutoff), then lower and optimize.

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::Refusal;
use crate::ir::MirModule;
use crate::ty::{PolyTy, Sources, Ty, TypeRegistry, lift_to_poly};

use super::extract::{ParsedSource, extract_one};
use super::infer::{FnInferOutcome, SccInferResult, extract_call_edges, infer_scc, tarjan_scc};
use super::lower::lower_one;
use super::optimize::{Opt, optimize};
use super::types::*;

// -- Cached entries --------------------------------------------------

struct ExtractEntry {
    parsed: ParsedSource,
}

/// A pre-SSA body, lowered and bound, kept so that a change elsewhere in the
/// graph re-optimizes this function without re-lowering it.
struct LowerEntry {
    module: MirModule,
    refusals: Vec<Refusal>,
}

struct OptimizedEntry {
    /// Read off the code the passes left, which is what makes this the set
    /// RFC-0071 rule 5 calls required.
    inputs: Vec<ContextInfo>,
    refusals: Vec<Refusal>,
}

// -- IncrementalGraph ------------------------------------------------

/// Inference is per-SCC, and optimization is whole-graph on every change.
/// A compilation here is one editor's open documents and the library they
/// call, so re-running the passes over all of them costs less than the
/// cross-module invalidation a per-function optimization cache would need,
/// and `graph::optimize::optimize` inlines and borrow-checks across modules
/// in one call regardless.
pub struct IncrementalGraph {
    interner: Interner,
    /// The sources of this compilation, shared by every solver it runs.
    sources: Sources,
    /// The user-defined types and cast rules of this compilation.
    type_registry: TypeRegistry,
    bindings: Bindings,

    // -- Source data --
    functions: FxHashMap<QualifiedRef, Function>,
    contexts: FxHashMap<QualifiedRef, Context>,
    /// The same fact `CompilationGraph::entry` carries, for a graph built by
    /// accumulation: unset until a host says which body it starts.
    entry: Option<QualifiedRef>,

    // -- Phase 0: Extract cache --
    extract_cache: FxHashMap<QualifiedRef, ExtractEntry>,

    // -- Call graph --
    call_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
    reverse_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
    scc_order: Vec<Vec<QualifiedRef>>,
    fn_to_scc: FxHashMap<QualifiedRef, usize>,

    // -- Phase 1: Infer cache (per SCC index) --
    infer_cache: Vec<Option<SccInferResult>>,

    // -- Phase 3: Lower cache --
    lower_cache: FxHashMap<QualifiedRef, LowerEntry>,

    // -- Phase 5: Optimize --
    optimized: FxHashMap<QualifiedRef, OptimizedEntry>,

    // -- Diagnostics --
    diagnostics: FxHashMap<QualifiedRef, Vec<Refusal>>,
}

impl IncrementalGraph {
    pub fn new(interner: &Interner) -> Self {
        Self::with_type_registry(interner, TypeRegistry::new())
    }

    pub fn with_type_registry(interner: &Interner, type_registry: TypeRegistry) -> Self {
        Self {
            interner: interner.clone(),
            sources: Sources::new(),
            type_registry,
            bindings: Bindings::default(),
            functions: FxHashMap::default(),
            contexts: FxHashMap::default(),
            entry: None,
            extract_cache: FxHashMap::default(),
            call_edges: FxHashMap::default(),
            reverse_edges: FxHashMap::default(),
            scc_order: Vec::new(),
            fn_to_scc: FxHashMap::default(),
            infer_cache: Vec::new(),
            lower_cache: FxHashMap::default(),
            optimized: FxHashMap::default(),
            diagnostics: FxHashMap::default(),
        }
    }

    // -- Namespace management -----------------------------------------

    pub fn remove_namespace(&mut self, ns_name: Astr) {
        // Remove all functions and contexts in this namespace.
        let fn_refs: Vec<QualifiedRef> = self
            .functions
            .values()
            .filter(|f| f.qref.namespace == Some(ns_name))
            .map(|f| f.qref)
            .collect();
        for qref in fn_refs {
            self.remove_function(qref);
        }
        let ctx_refs: Vec<QualifiedRef> = self
            .contexts
            .iter()
            .filter(|(_, c)| c.qref.namespace == Some(ns_name))
            .map(|(qref, _)| *qref)
            .collect();
        for qref in ctx_refs {
            self.remove_context(qref);
        }
    }

    // -- Registration ------------------------------------------------

    pub fn set_entry(&mut self, qref: QualifiedRef) {
        self.entry = Some(qref);
        self.invalidate_all_infer();
        self.recompile();
    }

    pub fn add_function(&mut self, func: Function) {
        let qref = func.qref;
        self.functions.insert(qref, func);
        self.run_extract(qref);
        self.rebuild_graph();
    }

    pub fn remove_function(&mut self, qref: QualifiedRef) {
        if self.functions.remove(&qref).is_some() {
            self.extract_cache.remove(&qref);
            self.call_edges.remove(&qref);
            self.diagnostics.remove(&qref);
            self.lower_cache.remove(&qref);
            self.optimized.remove(&qref);
            self.remove_reverse_edges(qref);
            self.rebuild_graph();
        }
    }

    /// Bind a `$` name to a constant. Every function is re-inferred: the
    /// binding gives the name a type, which is what the bodies reading it
    /// were checked against.
    pub fn bind_input(&mut self, name: Astr, value: acvus_ast::Literal) {
        self.bindings.bind(name, value);
        self.invalidate_all_infer();
        self.recompile();
    }

    /// The arms a fold of this name decided against are code again, so every
    /// name they read is required once more (RFC-0071 rule 5).
    pub fn unbind_input(&mut self, name: Astr) {
        self.bindings.unbind(name);
        self.invalidate_all_infer();
        self.recompile();
    }

    pub fn add_context(&mut self, ctx: Context) {
        let qref = ctx.qref;
        self.contexts.insert(qref, ctx);
        // Context change can affect all infer - full rebuild.
        self.invalidate_all_infer();
        self.recompile();
    }

    pub fn remove_context(&mut self, qref: QualifiedRef) {
        if self.contexts.remove(&qref).is_some() {
            self.invalidate_all_infer();
            self.recompile();
        }
    }

    // -- Source update (main incremental entry point) ----------------

    pub fn update_ast(&mut self, qref: QualifiedRef, ast: ParsedAst) {
        let Some(func) = self.functions.get_mut(&qref) else {
            return;
        };
        match &mut func.kind {
            FnKind::Local(existing) => *existing = ast,
            FnKind::Extern { .. } => return,
        }

        // 1. Re-extract.
        let old_edges = self.call_edges.get(&qref).cloned();
        self.run_extract(qref);

        // 2. Check if call edges changed.
        let new_edges = self.call_edges.get(&qref);
        let edges_changed = old_edges.as_ref() != new_edges;

        if edges_changed {
            // SCC structure may have changed - full rebuild.
            self.rebuild_graph();
        } else {
            // SCC unchanged - only re-infer the affected SCC + propagate.
            self.dirty_propagate(qref);
            self.settle();
        }
    }

    // -- Queries -----------------------------------------------------

    /// What every stage that can refuse this function refused: typeck, lower
    /// and validate, the set `acvus check` reports for the same source
    /// (RFC-0031). `acvus-lsp/tests/equivalence.rs` holds the two equal.
    pub fn diagnostics(&self, qref: QualifiedRef) -> &[Refusal] {
        self.diagnostics
            .get(&qref)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn all_diagnostics(&self) -> impl Iterator<Item = (QualifiedRef, &[Refusal])> {
        self.diagnostics
            .iter()
            .map(|(&qref, errs)| (qref, errs.as_slice()))
    }

    /// The inputs this one function requires: neither a `$` a binding fixed
    /// nor one whose type closed to `!` is among them (RFC-0071 rule 5).
    ///
    /// A function that lowered answers with the set its surviving code reads,
    /// which is the fold's and is what `acvus check` reports. A function
    /// inference left `Incomplete`, or lowering refused, has no body for the
    /// fold to run on, so it answers with the wider set the solve closed: a
    /// name only the arm a binding decided against reads is still among them.
    pub fn context_info(&self, qref: QualifiedRef) -> Vec<ContextInfo> {
        if let Some(entry) = self.optimized.get(&qref) {
            return entry.inputs.clone();
        }
        let Some(meta) = self.fn_meta(qref) else {
            return Vec::new();
        };
        meta.params
            .iter()
            .filter(|param| param.ty != Ty::Never)
            .filter(|param| self.bindings.get(param.name).is_none())
            .map(|param| ContextInfo {
                name: QualifiedRef::root(param.name),
                ty: param.ty.clone(),
            })
            .collect()
    }

    /// The inputs a run starting at `qref` requires: `context_info` of it
    /// and of every function it reaches through calls, since a `$` is shared
    /// by the whole graph (RFC-0071 rule 4).
    pub fn required_inputs(&self, qref: QualifiedRef) -> Vec<ContextInfo> {
        let mut seen: FxHashSet<QualifiedRef> = FxHashSet::default();
        let mut work = vec![qref];
        let mut required: Vec<ContextInfo> = Vec::new();
        while let Some(at) = work.pop() {
            if !seen.insert(at) {
                continue;
            }
            for input in self.context_info(at) {
                if !required.iter().any(|held| held.name == input.name) {
                    required.push(input);
                }
            }
            work.extend(self.call_edges.get(&at).into_iter().flatten());
        }
        required
    }

    fn outcome(&self, qref: QualifiedRef) -> Option<&FnInferOutcome> {
        let &scc = self.fn_to_scc.get(&qref)?;
        self.infer_cache[scc].as_ref()?.outcomes.get(&qref)
    }

    fn fn_meta(&self, qref: QualifiedRef) -> Option<&super::infer::FunctionMeta> {
        Some(self.outcome(qref)?.meta())
    }

    pub fn resolution(&self, qref: QualifiedRef) -> Option<Freeze<crate::typeck::TypeResolution>> {
        self.outcome(qref)?.resolution()
    }

    pub fn function(&self, qref: QualifiedRef) -> Option<&Function> {
        self.functions.get(&qref)
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    // -- Resolution ---------------------------------------------------

    /// Resolve a function name.
    /// - `qualifier = None` -> unqualified, root only.
    /// - `qualifier = Some(ns_name)` -> qualified, specific namespace only.
    pub fn resolve_fn(&self, qualifier: Option<Astr>, name: Astr) -> Option<QualifiedRef> {
        let qref = match qualifier {
            None => QualifiedRef::root(name),
            Some(ns_name) => QualifiedRef::qualified(ns_name, name),
        };
        if self.functions.contains_key(&qref) {
            Some(qref)
        } else {
            None
        }
    }

    /// Resolve a context name to its QualifiedRef.
    /// - `qualifier = None` -> unqualified, root only.
    /// - `qualifier = Some(ns_name)` -> qualified, specific namespace only.
    pub fn resolve_ctx(&self, qualifier: Option<Astr>, name: Astr) -> Option<QualifiedRef> {
        let qref = match qualifier {
            None => QualifiedRef::root(name),
            Some(ns_name) => QualifiedRef::qualified(ns_name, name),
        };
        if self.contexts.contains_key(&qref) {
            Some(qref)
        } else {
            None
        }
    }

    /// All contexts visible from a namespace (own namespace + root).
    /// Used by LSP for completions.
    pub fn visible_contexts(&self, ns: Option<Astr>) -> Vec<(Option<Astr>, Astr, &Context)> {
        self.contexts
            .values()
            .filter(|c| c.qref.namespace.is_none() || c.qref.namespace == ns)
            .map(|c| (c.qref.namespace, c.qref.name, c))
            .collect()
    }

    /// All functions callable from a namespace:
    /// - Root functions (unqualified)
    /// - Same-namespace functions (would need qualified, but are accessible)
    /// Used by LSP for completions.
    pub fn visible_functions(&self, ns: Option<Astr>) -> Vec<(Option<Astr>, Astr, &Function)> {
        self.functions
            .values()
            .filter(|f| f.qref.namespace.is_none() || f.qref.namespace == ns)
            .map(|f| (f.qref.namespace, f.qref.name, f))
            .collect()
    }

    // -- Internal: Extract -------------------------------------------

    fn run_extract(&mut self, qref: QualifiedRef) {
        let Some(func) = self.functions.get(&qref) else {
            return;
        };

        // Run extract.
        if let Some(parsed) = extract_one(&self.interner, func) {
            // Update call edges.
            // TODO: qualified call edges once AST supports ns:func() syntax.
            // For now, only unqualified (root) names are resolved.
            let root_fn_names: FxHashMap<Astr, QualifiedRef> = self
                .functions
                .iter()
                .filter(|(q, f)| q.namespace.is_none() && matches!(f.kind, FnKind::Local(_)))
                .map(|(&q, _)| (q.name, q))
                .collect();
            let new_edges = extract_call_edges(&parsed, &root_fn_names, qref);
            self.remove_reverse_edges(qref);
            for &callee in &new_edges {
                self.reverse_edges.entry(callee).or_default().push(qref);
            }
            self.call_edges.insert(qref, new_edges);

            self.extract_cache.insert(qref, ExtractEntry { parsed });
        } else {
            // Parse failed - clear caches.
            self.extract_cache.remove(&qref);
            self.call_edges.remove(&qref);
            self.remove_reverse_edges(qref);
        }
    }

    fn remove_reverse_edges(&mut self, qref: QualifiedRef) {
        if let Some(old_callees) = self.call_edges.get(&qref) {
            let old_callees = old_callees.clone();
            for callee in old_callees {
                if let Some(rev) = self.reverse_edges.get_mut(&callee) {
                    rev.retain(|&x| x != qref);
                }
            }
        }
    }

    // -- Internal: Graph rebuild (SCC) -------------------------------

    fn rebuild_graph(&mut self) {
        let local_qrefs: Vec<QualifiedRef> = self
            .functions
            .values()
            .filter(|f| matches!(f.kind, FnKind::Local(_)))
            .map(|f| f.qref)
            .collect();

        self.scc_order = tarjan_scc(&local_qrefs, &self.call_edges);
        self.fn_to_scc.clear();
        for (idx, scc) in self.scc_order.iter().enumerate() {
            for &fid in scc {
                self.fn_to_scc.insert(fid, idx);
            }
        }

        // Rebuild all infer caches.
        self.infer_cache = vec![None; self.scc_order.len()];
        self.lower_cache.clear();

        self.recompile();
    }

    // -- Internal: Infer ---------------------------------------------

    fn run_infer(&mut self) {
        let known_ctx = self.known_context_types();
        let mut resolved_fn_types: FxHashMap<QualifiedRef, PolyTy> = FxHashMap::default();
        // Seed with extern function types (always known upfront).
        for (qref, func) in &self.functions {
            if let FnKind::Extern { .. } = &func.kind {
                resolved_fn_types.insert(*qref, func.ty.clone());
            }
        }

        let fn_by_id: FxHashMap<QualifiedRef, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(_)))
            .map(|(&qref, f)| (qref, f))
            .collect();

        let extract_parsed: FxHashMap<QualifiedRef, &ParsedSource> = self
            .extract_cache
            .iter()
            .map(|(&qref, e)| (qref, &e.parsed))
            .collect();

        for (scc_idx, scc) in self.scc_order.iter().enumerate() {
            // Skip already-cached SCCs.
            if self.infer_cache[scc_idx].is_some() {
                // Still need to accumulate resolved types for subsequent SCCs.
                if let Some(ref cached) = self.infer_cache[scc_idx] {
                    resolved_fn_types.extend(
                        cached
                            .resolved_types
                            .iter()
                            .map(|(&k, v)| (k, lift_to_poly(v))),
                    );
                }
                continue;
            }

            let parsed_owned: FxHashMap<QualifiedRef, &ParsedSource> = scc
                .iter()
                .filter_map(|fid| extract_parsed.get(fid).map(|p| (*fid, *p)))
                .collect();

            let result = infer_scc(
                &self.interner,
                scc,
                self.entry,
                &self.bindings,
                &fn_by_id,
                &parsed_owned,
                &known_ctx,
                &resolved_fn_types,
                &super::infer::declared_bounds(self.functions.values()),
                &mut self.sources,
                &self.type_registry,
            );

            resolved_fn_types.extend(
                result
                    .resolved_types
                    .iter()
                    .map(|(&k, v)| (k, lift_to_poly(v))),
            );
            self.infer_cache[scc_idx] = Some(result);
        }
    }

    fn dirty_propagate(&mut self, changed_fn: QualifiedRef) {
        let Some(&start_scc) = self.fn_to_scc.get(&changed_fn) else {
            return;
        };

        // Mark this SCC as dirty.
        let _old_result = self.infer_cache[start_scc].take();

        // Re-run infer from this SCC onwards.
        let known_ctx = self.known_context_types();
        let mut resolved_fn_types: FxHashMap<QualifiedRef, PolyTy> = FxHashMap::default();
        // Seed with extern function types.
        for (qref, func) in &self.functions {
            if let FnKind::Extern { .. } = &func.kind {
                resolved_fn_types.insert(*qref, func.ty.clone());
            }
        }

        let fn_by_id: FxHashMap<QualifiedRef, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(_)))
            .map(|(&qref, f)| (qref, f))
            .collect();

        let extract_parsed: FxHashMap<QualifiedRef, &ParsedSource> = self
            .extract_cache
            .iter()
            .map(|(&qref, e)| (qref, &e.parsed))
            .collect();

        // Accumulate resolved types from prior SCCs.
        for scc_idx in 0..start_scc {
            if let Some(ref cached) = self.infer_cache[scc_idx] {
                resolved_fn_types.extend(
                    cached
                        .resolved_types
                        .iter()
                        .map(|(&k, v)| (k, lift_to_poly(v))),
                );
            }
        }

        // Re-infer from start_scc onwards (with early cutoff).
        let mut dirty_sccs: FxHashSet<usize> = FxHashSet::default();
        dirty_sccs.insert(start_scc);

        for scc_idx in start_scc..self.scc_order.len() {
            if !dirty_sccs.contains(&scc_idx) {
                // Not dirty - use cached result.
                if let Some(ref cached) = self.infer_cache[scc_idx] {
                    resolved_fn_types.extend(
                        cached
                            .resolved_types
                            .iter()
                            .map(|(&k, v)| (k, lift_to_poly(v))),
                    );
                }
                continue;
            }

            let scc = &self.scc_order[scc_idx];
            let old_types = self.infer_cache[scc_idx]
                .as_ref()
                .map(|r| r.resolved_types.clone());

            let parsed_for_scc: FxHashMap<QualifiedRef, &ParsedSource> = scc
                .iter()
                .filter_map(|fid| extract_parsed.get(fid).map(|p| (*fid, *p)))
                .collect();

            let result = infer_scc(
                &self.interner,
                scc,
                self.entry,
                &self.bindings,
                &fn_by_id,
                &parsed_for_scc,
                &known_ctx,
                &resolved_fn_types,
                &super::infer::declared_bounds(self.functions.values()),
                &mut self.sources,
                &self.type_registry,
            );

            // Early cutoff: if types didn't change, don't propagate.
            let types_changed = old_types
                .as_ref()
                .map(|old| old != &result.resolved_types)
                .unwrap_or(true);

            if types_changed {
                // Mark dependent SCCs as dirty.
                for &fid in scc {
                    if let Some(callers) = self.reverse_edges.get(&fid) {
                        for &caller in callers {
                            if let Some(&caller_scc) = self.fn_to_scc.get(&caller)
                                && caller_scc > scc_idx
                            {
                                dirty_sccs.insert(caller_scc);
                            }
                        }
                    }
                }
            }

            // A re-inferred body is lowered again: its resolution is new.
            for &fid in scc {
                self.lower_cache.remove(&fid);
            }

            resolved_fn_types.extend(
                result
                    .resolved_types
                    .iter()
                    .map(|(&k, v)| (k, lift_to_poly(v))),
            );
            self.infer_cache[scc_idx] = Some(result);
        }
    }

    // -- Internal: Lower + optimize -----------------------------------

    fn recompile(&mut self) {
        self.run_infer();
        self.settle();
    }

    /// The modules are optimized at `Opt::Full`, the level `acvus check`
    /// reads the required inputs at: a constant a bound `$` puts behind a
    /// field or a loop is what `sroa` and the motion passes expose, and a
    /// set read at a lower level would be wider than the one the batch
    /// path reports.
    fn settle(&mut self) {
        let lowerable: FxHashSet<QualifiedRef> = self
            .extract_cache
            .keys()
            .copied()
            .filter(|&qref| matches!(self.outcome(qref), Some(FnInferOutcome::Complete { .. })))
            .collect();
        self.lower_cache.retain(|qref, _| lowerable.contains(qref));

        let to_lower: Vec<QualifiedRef> = self
            .extract_cache
            .keys()
            .copied()
            .filter(|qref| !self.lower_cache.contains_key(qref))
            .collect();
        for qref in to_lower {
            let lowered = {
                let Some(outcome) = self.outcome(qref) else {
                    continue;
                };
                lower_one(
                    &self.interner,
                    &self.extract_cache[&qref].parsed,
                    outcome,
                    &self.bindings,
                )
            };
            let Some(lowered) = lowered else {
                continue;
            };
            self.lower_cache.insert(
                qref,
                LowerEntry {
                    module: lowered.module,
                    refusals: lowered.errors.into_iter().map(Refusal::Mir).collect(),
                },
            );
        }

        // A body lowering refused is not optimized, as `acvus check` does not
        // optimize a graph a stage before it refused.
        let modules: FxHashMap<QualifiedRef, MirModule> = self
            .lower_cache
            .iter()
            .filter(|(_, entry)| entry.refusals.is_empty())
            .map(|(&qref, entry)| (qref, entry.module.clone()))
            .collect();
        let result = optimize(&self.interner, modules, Opt::Full);

        let mut refused: FxHashMap<QualifiedRef, Vec<Refusal>> = FxHashMap::default();
        for (qref, errors) in result.errors {
            refused
                .entry(qref)
                .or_default()
                .extend(errors.into_iter().map(Refusal::Invalid));
        }
        self.optimized = result
            .inputs
            .into_iter()
            .map(|(qref, inputs)| {
                let refusals = refused.remove(&qref).unwrap_or_default();
                (qref, OptimizedEntry { inputs, refusals })
            })
            .collect();
        assert!(
            refused.is_empty(),
            "optimize refused {:?}, which it reported no inputs for",
            refused.keys().collect::<Vec<_>>()
        );

        self.rebuild_diagnostics();
    }

    fn rebuild_diagnostics(&mut self) {
        let stages = self
            .infer_cache
            .iter()
            .flatten()
            .flat_map(|scc| scc.errors())
            .map(|(qref, errors)| {
                let refusals: Vec<Refusal> = errors.iter().cloned().map(Refusal::Mir).collect();
                (qref, refusals)
            });
        let mut diagnostics: FxHashMap<QualifiedRef, Vec<Refusal>> = stages.collect();
        for (&qref, entry) in &self.lower_cache {
            diagnostics
                .entry(qref)
                .or_default()
                .extend(entry.refusals.iter().cloned());
        }
        for (&qref, entry) in &self.optimized {
            diagnostics
                .entry(qref)
                .or_default()
                .extend(entry.refusals.iter().cloned());
        }
        diagnostics.retain(|_, refusals| !refusals.is_empty());
        self.diagnostics = diagnostics;
    }

    // -- Helpers -----------------------------------------------------

    fn known_context_types(&self) -> FxHashMap<QualifiedRef, PolyTy> {
        self.contexts
            .values()
            .map(|ctx| (ctx.qref, ctx.ty.clone()))
            .collect()
    }

    fn invalidate_all_infer(&mut self) {
        for slot in &mut self.infer_cache {
            *slot = None;
        }
        self.lower_cache.clear();
    }

    /// Build a snapshot InferResult for compatibility with batch APIs.
    pub fn infer_result(&mut self) -> super::infer::InferResult {
        let outcomes: FxHashMap<QualifiedRef, FnInferOutcome> = self
            .infer_cache
            .iter()
            .flatten()
            .flat_map(|scc| scc.outcomes.iter())
            .map(|(&fid, outcome)| (fid, outcome.clone()))
            .collect();

        super::infer::InferResult {
            outcomes,
            context_types: {
                // PolyTy -> InferTy (instantiate) -> Ty (freeze) at the output boundary.
                let known = self.known_context_types();
                let signatures = FxHashMap::default();
                let mut solver =
                    crate::ty::Solver::new(&mut self.sources, &self.type_registry, &signatures);
                Freeze::new(
                    known
                        .into_iter()
                        .map(|(k, v)| {
                            let infer = solver.instantiate_poly(&v);
                            (k, solver.freeze_ty(&infer).unwrap_or_else(|_| Ty::error()))
                        })
                        .collect(),
                )
            },
        }
    }
}
