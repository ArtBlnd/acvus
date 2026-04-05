//! Incremental compilation graph.
//!
//! Manages per-function extract/infer caches with dirty tracking.
//! On source change: re-extract → diff call edges → re-SCC if needed →
//! re-infer dirty SCCs (with early cutoff).

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::MirError;
use crate::ty::{EffectSet, Ty};

use super::extract::{ExtractResult, ParsedSource, extract_one};
use super::infer::{SccInferResult, extract_call_edges, infer_scc, tarjan_scc};
use super::types::*;

// ── Cached entries ──────────────────────────────────────────────────

struct ExtractEntry {
    parsed: ParsedSource,
}

// ── Context info (public output) ────────────────────────────────────

/// Information about a context/param that must be injected externally.
#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub name: QualifiedRef,
    pub ty: Ty,
}

// ── IncrementalGraph ────────────────────────────────────────────────

pub struct IncrementalGraph {
    interner: Interner,

    // ── Source data ──
    functions: FxHashMap<QualifiedRef, Function>,
    contexts: FxHashMap<QualifiedRef, Context>,

    // ── Phase 0: Extract cache ──
    extract_cache: FxHashMap<QualifiedRef, ExtractEntry>,

    // ── Call graph ──
    call_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
    reverse_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>>,
    scc_order: Vec<Vec<QualifiedRef>>,
    fn_to_scc: FxHashMap<QualifiedRef, usize>,

    // ── Phase 1: Infer cache (per SCC index) ──
    infer_cache: Vec<Option<SccInferResult>>,

    // ── Diagnostics ──
    diagnostics: FxHashMap<QualifiedRef, Vec<MirError>>,
}

impl IncrementalGraph {
    pub fn new(interner: &Interner) -> Self {
        Self {
            interner: interner.clone(),
            functions: FxHashMap::default(),
            contexts: FxHashMap::default(),
            extract_cache: FxHashMap::default(),
            call_edges: FxHashMap::default(),
            reverse_edges: FxHashMap::default(),
            scc_order: Vec::new(),
            fn_to_scc: FxHashMap::default(),
            infer_cache: Vec::new(),
            diagnostics: FxHashMap::default(),
        }
    }

    // ── Namespace management ─────────────────────────────────────────

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

    // ── Registration ────────────────────────────────────────────────

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
            self.remove_reverse_edges(qref);
            self.rebuild_graph();
        }
    }

    pub fn add_context(&mut self, ctx: Context) {
        let qref = ctx.qref;
        self.contexts.insert(qref, ctx);
        // Context change can affect all infer — full rebuild.
        self.invalidate_all_infer();
        self.run_infer();
    }

    pub fn remove_context(&mut self, qref: QualifiedRef) {
        if self.contexts.remove(&qref).is_some() {
            self.invalidate_all_infer();
            self.run_infer();
        }
    }

    // ── Source update (main incremental entry point) ────────────────

    pub fn update_ast(&mut self, qref: QualifiedRef, ast: ParsedAst) {
        let Some(func) = self.functions.get_mut(&qref) else {
            return;
        };
        match &mut func.kind {
            FnKind::Local(existing) => *existing = ast,
            FnKind::Extern => return,
        }

        // 1. Re-extract.
        let old_edges = self.call_edges.get(&qref).cloned();
        self.run_extract(qref);

        // 2. Check if call edges changed.
        let new_edges = self.call_edges.get(&qref);
        let edges_changed = old_edges.as_ref() != new_edges;

        if edges_changed {
            // SCC structure may have changed — full rebuild.
            self.rebuild_graph();
        } else {
            // SCC unchanged — only re-infer the affected SCC + propagate.
            self.dirty_propagate(qref);
        }
    }

    // ── Queries ─────────────────────────────────────────────────────

    pub fn diagnostics(&self, qref: QualifiedRef) -> &[MirError] {
        self.diagnostics
            .get(&qref)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn all_diagnostics(&self) -> impl Iterator<Item = (QualifiedRef, &[MirError])> {
        self.diagnostics
            .iter()
            .map(|(&qref, errs)| (qref, errs.as_slice()))
    }

    /// Get context/param info that must be injected externally for a function.
    pub fn context_info(&self, _qref: QualifiedRef) -> Vec<ContextInfo> {
        // TODO: context param inference removed — reconstruct from fn_metas if needed.
        vec![]
    }

    // TODO: resolution now comes from InferResult outcomes.
    // SccInferResult needs to store resolutions for this to work.
    // For now, always returns None.
    pub fn resolution(&self, _qref: QualifiedRef) -> Option<()> {
        None
    }

    pub fn function(&self, qref: QualifiedRef) -> Option<&Function> {
        self.functions.get(&qref)
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    // ── Resolution ───────────────────────────────────────────────────

    /// Resolve a function name.
    /// - `qualifier = None` → unqualified, root only.
    /// - `qualifier = Some(ns_name)` → qualified, specific namespace only.
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
    /// - `qualifier = None` → unqualified, root only.
    /// - `qualifier = Some(ns_name)` → qualified, specific namespace only.
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

    // ── Internal: Extract ───────────────────────────────────────────

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
            // Parse failed — clear caches.
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

    // ── Internal: Graph rebuild (SCC) ───────────────────────────────

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
        self.diagnostics.clear();

        self.run_infer();
    }

    // ── Internal: Infer ─────────────────────────────────────────────

    fn run_infer(&mut self) {
        let known_ctx = self.known_context_types();
        let mut resolved_fn_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
        // Seed with extern function types (always known upfront).
        for (qref, func) in &self.functions {
            if let FnKind::Extern = &func.kind {
                if let Constraint::Exact(ty) = &func.constraint.output {
                    resolved_fn_types.insert(*qref, ty.clone());
                }
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
                    resolved_fn_types.extend(cached.resolved_types.clone());
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
                &fn_by_id,
                &parsed_owned,
                &known_ctx,
                &resolved_fn_types,
            );

            resolved_fn_types.extend(result.resolved_types.clone());
            for (qref, errs) in &result.errors {
                self.diagnostics.insert(*qref, errs.clone());
            }
            self.infer_cache[scc_idx] = Some(result);
        }

        // Effect propagation across SCCs.
        self.propagate_effects();
    }

    fn dirty_propagate(&mut self, changed_fn: QualifiedRef) {
        let Some(&start_scc) = self.fn_to_scc.get(&changed_fn) else {
            return;
        };

        // Mark this SCC as dirty.
        let _old_result = self.infer_cache[start_scc].take();

        // Re-run infer from this SCC onwards.
        let known_ctx = self.known_context_types();
        let mut resolved_fn_types: FxHashMap<QualifiedRef, Ty> = FxHashMap::default();
        // Seed with extern function types.
        for (qref, func) in &self.functions {
            if let FnKind::Extern = &func.kind {
                if let Constraint::Exact(ty) = &func.constraint.output {
                    resolved_fn_types.insert(*qref, ty.clone());
                }
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
                resolved_fn_types.extend(cached.resolved_types.clone());
            }
        }

        // Re-infer from start_scc onwards (with early cutoff).
        let mut dirty_sccs: FxHashSet<usize> = FxHashSet::default();
        dirty_sccs.insert(start_scc);

        for scc_idx in start_scc..self.scc_order.len() {
            if !dirty_sccs.contains(&scc_idx) {
                // Not dirty — use cached result.
                if let Some(ref cached) = self.infer_cache[scc_idx] {
                    resolved_fn_types.extend(cached.resolved_types.clone());
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
                &fn_by_id,
                &parsed_for_scc,
                &known_ctx,
                &resolved_fn_types,
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

            // Update diagnostics: clear old errors for this SCC, add new ones.
            for &fid in scc {
                self.diagnostics.remove(&fid);
            }
            for (qref, errs) in &result.errors {
                self.diagnostics.insert(*qref, errs.clone());
            }

            resolved_fn_types.extend(result.resolved_types.clone());
            self.infer_cache[scc_idx] = Some(result);
        }

        // Effect propagation.
        self.propagate_effects();
    }

    fn propagate_effects(&mut self) {
        // Seed fn_metas with direct effects computed by the typechecker during infer.
        for scc_idx in 0..self.scc_order.len() {
            let scc = &self.scc_order[scc_idx];
            let Some(ref mut scc_result) = self.infer_cache[scc_idx] else {
                continue;
            };

            for &fid in scc {
                if let Some(direct) = scc_result.fn_direct_effects.get(&fid)
                    && let Some(meta) = scc_result.fn_metas.get_mut(&fid)
                {
                    meta.effect = direct.clone();
                }
            }
        }

        // Propagate through call graph in SCC order (fixpoint within each SCC).
        // We need to collect all fn_metas into one map for propagation.
        let mut all_effects: FxHashMap<QualifiedRef, EffectSet> = FxHashMap::default();
        for scc_result in self.infer_cache.iter().flatten() {
            for (&fid, meta) in &scc_result.fn_metas {
                all_effects.insert(fid, meta.effect.clone());
            }
        }

        for scc in &self.scc_order {
            loop {
                let mut changed = false;
                for &fid in scc {
                    if let Some(callees) = self.call_edges.get(&fid) {
                        for &callee_id in callees {
                            let callee_effect =
                                all_effects.get(&callee_id).cloned().unwrap_or_default();
                            let current = all_effects.get(&fid).cloned().unwrap_or_default();
                            let merged = current.union(&callee_effect);
                            if merged != current {
                                all_effects.insert(fid, merged);
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

        // Write back to infer_cache.
        for scc_result in self.infer_cache.iter_mut().flatten() {
            for (fid, meta) in scc_result.fn_metas.iter_mut() {
                if let Some(effect) = all_effects.get(fid) {
                    meta.effect = effect.clone();
                }
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn known_context_types(&self) -> FxHashMap<QualifiedRef, Ty> {
        self.contexts
            .values()
            .filter_map(|ctx| {
                if let Constraint::Exact(ty) = &ctx.constraint {
                    Some((ctx.qref, ty.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn invalidate_all_infer(&mut self) {
        for slot in &mut self.infer_cache {
            *slot = None;
        }
        self.diagnostics.clear();
    }

    /// Build a snapshot ExtractResult for compatibility with batch APIs.
    pub fn extract_result(&self) -> ExtractResult {
        let parsed = FxHashMap::default();
        // ParsedSource is not Clone — we need to handle this.
        // For now, skip parsed in snapshot (batch lower can re-extract if needed).
        ExtractResult { parsed }
    }

    /// Build a snapshot InferResult for compatibility with batch APIs.
    pub fn infer_result(&self) -> super::infer::InferResult {
        let mut outcomes: FxHashMap<QualifiedRef, super::infer::FnInferOutcome> =
            FxHashMap::default();

        for scc_result in self.infer_cache.iter().flatten() {
            // Convert SccInferResult metas to Incomplete outcomes (temporary — incremental
            // does not yet run check_completeness; this will be reworked in Step 6).
            for (&fid, meta) in &scc_result.fn_metas {
                outcomes.insert(
                    fid,
                    super::infer::FnInferOutcome::Incomplete {
                        unknown_contexts: vec![],
                        unknown_extern_params: vec![],
                        meta: meta.clone(),
                        errors: vec![],
                    },
                );
            }
        }

        let mut fn_metadata: FxHashMap<QualifiedRef, super::types::FnMetadata> =
            FxHashMap::default();
        for (&qref, outcome) in &outcomes {
            let hint = self.functions.get(&qref).and_then(|f| f.constraint.hint);
            fn_metadata.insert(
                qref,
                super::types::FnMetadata {
                    ty: outcome.meta().ty.clone(),
                    hint,
                },
            );
        }

        super::infer::InferResult {
            outcomes,
            context_types: Freeze::new(self.known_context_types()),
            fn_metadata: Freeze::new(fn_metadata),
        }
    }
}
