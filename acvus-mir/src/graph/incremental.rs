//! Incremental compilation graph.
//!
//! Manages per-function extract/infer caches with dirty tracking.
//! On source change: re-extract → diff call edges → re-SCC if needed →
//! re-infer dirty SCCs (with early cutoff).

use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::MirError;
use crate::ty::{EffectSet, Ty};

use super::extract::{ExtractResult, FnRefs, ParsedSource, extract_one};
use super::infer::{InferredParam, SccInferResult, extract_call_edges, infer_scc, tarjan_scc};
use super::types::*;

// ── Cached entries ──────────────────────────────────────────────────

struct ExtractEntry {
    source_hash: u64,
    refs: FnRefs,
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

    // ── Namespaces ──
    namespaces: FxHashMap<NamespaceId, Namespace>,
    name_to_ns: FxHashMap<Astr, NamespaceId>,

    // ── Source data ──
    functions: FxHashMap<FunctionId, Function>,
    contexts: FxHashMap<QualifiedRef, Context>,
    /// (namespace, name) → FunctionId. namespace=None means root.
    name_to_fn: FxHashMap<(Option<NamespaceId>, Astr), FunctionId>,

    // ── Phase 0: Extract cache ──
    extract_cache: FxHashMap<FunctionId, ExtractEntry>,

    // ── Call graph ──
    call_edges: FxHashMap<FunctionId, Vec<FunctionId>>,
    reverse_edges: FxHashMap<FunctionId, Vec<FunctionId>>,
    scc_order: Vec<Vec<FunctionId>>,
    fn_to_scc: FxHashMap<FunctionId, usize>,

    // ── Phase 1: Infer cache (per SCC index) ──
    infer_cache: Vec<Option<SccInferResult>>,

    // ── Diagnostics ──
    diagnostics: FxHashMap<FunctionId, Vec<MirError>>,
}

impl IncrementalGraph {
    pub fn new(interner: &Interner) -> Self {
        Self {
            interner: interner.clone(),
            namespaces: FxHashMap::default(),
            name_to_ns: FxHashMap::default(),
            functions: FxHashMap::default(),
            contexts: FxHashMap::default(),
            name_to_fn: FxHashMap::default(),
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

    pub fn add_namespace(&mut self, ns: Namespace) {
        self.name_to_ns.insert(ns.name, ns.id);
        self.namespaces.insert(ns.id, ns);
    }

    pub fn remove_namespace(&mut self, id: NamespaceId) {
        if let Some(ns) = self.namespaces.remove(&id) {
            self.name_to_ns.remove(&ns.name);
            // Remove all functions and contexts in this namespace.
            let fn_ids: Vec<FunctionId> = self
                .functions
                .values()
                .filter(|f| f.namespace == Some(id))
                .map(|f| f.id)
                .collect();
            for fid in fn_ids {
                self.remove_function(fid);
            }
            let ctx_refs: Vec<QualifiedRef> = self
                .contexts
                .iter()
                .filter(|(_, c)| c.namespace == Some(id))
                .map(|(qref, _)| *qref)
                .collect();
            for qref in ctx_refs {
                self.remove_context(qref);
            }
        }
    }

    // ── Registration ────────────────────────────────────────────────

    pub fn add_function(&mut self, func: Function) {
        let key = (func.namespace, func.name);
        self.name_to_fn.insert(key, func.id);
        let id = func.id;
        self.functions.insert(id, func);
        self.run_extract(id);
        self.rebuild_graph();
    }

    pub fn remove_function(&mut self, id: FunctionId) {
        if let Some(func) = self.functions.remove(&id) {
            let key = (func.namespace, func.name);
            self.name_to_fn.remove(&key);
            self.extract_cache.remove(&id);
            self.call_edges.remove(&id);
            self.diagnostics.remove(&id);
            self.remove_reverse_edges(id);
            self.rebuild_graph();
        }
    }

    pub fn add_context(&mut self, ctx: Context) {
        let qref = ctx.qualified_ref();
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

    pub fn update_source(&mut self, id: FunctionId, source: Astr) {
        let Some(func) = self.functions.get_mut(&id) else {
            return;
        };
        match &mut func.kind {
            FnKind::Local(src) => src.source = source,
            FnKind::LocalAst(_) => return, // AST functions are not updated via source string
            FnKind::Extern => return,
        }

        // 1. Re-extract.
        let old_edges = self.call_edges.get(&id).cloned();
        self.run_extract(id);

        // 2. Check if call edges changed.
        let new_edges = self.call_edges.get(&id);
        let edges_changed = old_edges.as_ref() != new_edges;

        if edges_changed {
            // SCC structure may have changed — full rebuild.
            self.rebuild_graph();
        } else {
            // SCC unchanged — only re-infer the affected SCC + propagate.
            self.dirty_propagate(id);
        }
    }

    // ── Queries ─────────────────────────────────────────────────────

    pub fn diagnostics(&self, id: FunctionId) -> &[MirError] {
        self.diagnostics
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn all_diagnostics(&self) -> impl Iterator<Item = (FunctionId, &[MirError])> {
        self.diagnostics
            .iter()
            .map(|(&id, errs)| (id, errs.as_slice()))
    }

    /// Get context/param info that must be injected externally for a function.
    pub fn context_info(&self, id: FunctionId) -> Vec<ContextInfo> {
        // Collect from infer results: fn_params for this function.
        let scc_idx = match self.fn_to_scc.get(&id) {
            Some(&idx) => idx,
            None => return vec![],
        };
        let scc_result = match self.infer_cache.get(scc_idx) {
            Some(Some(r)) => r,
            _ => return vec![],
        };
        scc_result
            .fn_params
            .get(&id)
            .map(|params| {
                params
                    .iter()
                    .map(|p| ContextInfo {
                        name: p.name,
                        ty: p.ty.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // TODO: resolution now comes from InferResult outcomes.
    // SccInferResult needs to store resolutions for this to work.
    // For now, always returns None.
    pub fn resolution(&self, _id: FunctionId) -> Option<()> {
        None
    }

    pub fn function(&self, id: FunctionId) -> Option<&Function> {
        self.functions.get(&id)
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    pub fn namespace(&self, id: NamespaceId) -> Option<&Namespace> {
        self.namespaces.get(&id)
    }

    pub fn namespace_by_name(&self, name: Astr) -> Option<&Namespace> {
        self.name_to_ns
            .get(&name)
            .and_then(|id| self.namespaces.get(id))
    }

    // ── Resolution ───────────────────────────────────────────────────

    /// Resolve a function name.
    /// - `qualifier = None` → unqualified, root only.
    /// - `qualifier = Some(ns_name)` → qualified, specific namespace only.
    pub fn resolve_fn(&self, qualifier: Option<Astr>, name: Astr) -> Option<FunctionId> {
        match qualifier {
            None => self.name_to_fn.get(&(None, name)).copied(),
            Some(ns_name) => {
                let ns_id = self.name_to_ns.get(&ns_name)?;
                self.name_to_fn.get(&(Some(*ns_id), name)).copied()
            }
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
    pub fn visible_contexts(
        &self,
        ns: Option<NamespaceId>,
    ) -> Vec<(Option<NamespaceId>, Astr, &Context)> {
        self.contexts
            .values()
            .filter(|c| c.namespace.is_none() || c.namespace == ns)
            .map(|c| (c.namespace, c.name, c))
            .collect()
    }

    /// All functions callable from a namespace:
    /// - Root functions (unqualified)
    /// - Same-namespace functions (would need qualified, but are accessible)
    /// Used by LSP for completions.
    pub fn visible_functions(
        &self,
        ns: Option<NamespaceId>,
    ) -> Vec<(Option<NamespaceId>, Astr, &Function)> {
        self.functions
            .values()
            .filter(|f| f.namespace.is_none() || f.namespace == ns)
            .map(|f| (f.namespace, f.name, f))
            .collect()
    }

    // ── Internal: Extract ───────────────────────────────────────────

    fn run_extract(&mut self, id: FunctionId) {
        let Some(func) = self.functions.get(&id) else {
            return;
        };

        // Check source hash for early skip.
        let source_hash = if let FnKind::Local(src) = &func.kind {
            hash_astr(src.source)
        } else {
            return;
        };

        if let Some(entry) = self.extract_cache.get(&id)
            && entry.source_hash == source_hash
        {
            return; // No change.
        }

        // Run extract.
        if let Some((refs, parsed)) = extract_one(&self.interner, func) {
            // Update call edges.
            // TODO: qualified call edges once AST supports ns:func() syntax.
            // For now, only unqualified (root) names are resolved.
            let root_fn_names: FxHashMap<Astr, FunctionId> = self
                .name_to_fn
                .iter()
                .filter(|((ns, _), _)| ns.is_none())
                .map(|((_, name), &id)| (*name, id))
                .collect();
            let new_edges = extract_call_edges(&parsed, &root_fn_names, id);
            self.remove_reverse_edges(id);
            for &callee in &new_edges {
                self.reverse_edges.entry(callee).or_default().push(id);
            }
            self.call_edges.insert(id, new_edges);

            self.extract_cache.insert(
                id,
                ExtractEntry {
                    source_hash,
                    refs,
                    parsed,
                },
            );
        } else {
            // Parse failed — clear caches.
            self.extract_cache.remove(&id);
            self.call_edges.remove(&id);
            self.remove_reverse_edges(id);
        }
    }

    fn remove_reverse_edges(&mut self, id: FunctionId) {
        if let Some(old_callees) = self.call_edges.get(&id) {
            let old_callees = old_callees.clone();
            for callee in old_callees {
                if let Some(rev) = self.reverse_edges.get_mut(&callee) {
                    rev.retain(|&x| x != id);
                }
            }
        }
    }

    // ── Internal: Graph rebuild (SCC) ───────────────────────────────

    fn rebuild_graph(&mut self) {
        let local_ids: Vec<FunctionId> = self
            .functions
            .values()
            .filter(|f| matches!(f.kind, FnKind::Local(_)))
            .map(|f| f.id)
            .collect();

        self.scc_order = tarjan_scc(&local_ids, &self.call_edges);
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
        let mut resolved_fn_types = crate::builtins::builtin_fn_types(&self.interner);

        let fn_by_id: FxHashMap<FunctionId, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(_)))
            .map(|(&id, f)| (id, f))
            .collect();

        let extract_refs: FxHashMap<FunctionId, &FnRefs> = self
            .extract_cache
            .iter()
            .map(|(&id, e)| (id, &e.refs))
            .collect();
        let extract_parsed: FxHashMap<FunctionId, &ParsedSource> = self
            .extract_cache
            .iter()
            .map(|(&id, e)| (id, &e.parsed))
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

            let refs_owned: FxHashMap<FunctionId, FnRefs> = scc
                .iter()
                .filter_map(|fid| extract_refs.get(fid).map(|r| (*fid, (*r).clone())))
                .collect();
            let parsed_owned: FxHashMap<FunctionId, &ParsedSource> = scc
                .iter()
                .filter_map(|fid| extract_parsed.get(fid).map(|p| (*fid, *p)))
                .collect();

            let result = infer_scc(
                &self.interner,
                scc,
                &fn_by_id,
                &refs_owned,
                &parsed_owned,
                &known_ctx,
                &resolved_fn_types,
            );

            resolved_fn_types.extend(result.resolved_types.clone());
            self.infer_cache[scc_idx] = Some(result);
        }

        // Effect propagation across SCCs.
        self.propagate_effects();
    }

    fn dirty_propagate(&mut self, changed_fn: FunctionId) {
        let Some(&start_scc) = self.fn_to_scc.get(&changed_fn) else {
            return;
        };

        // Mark this SCC as dirty.
        let _old_result = self.infer_cache[start_scc].take();

        // Re-run infer from this SCC onwards.
        let known_ctx = self.known_context_types();
        let mut resolved_fn_types = crate::builtins::builtin_fn_types(&self.interner);

        let fn_by_id: FxHashMap<FunctionId, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(_)))
            .map(|(&id, f)| (id, f))
            .collect();

        let extract_refs: FxHashMap<FunctionId, FnRefs> = self
            .extract_cache
            .iter()
            .map(|(&id, e)| (id, e.refs.clone()))
            .collect();
        let extract_parsed: FxHashMap<FunctionId, &ParsedSource> = self
            .extract_cache
            .iter()
            .map(|(&id, e)| (id, &e.parsed))
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

            let refs_for_scc: FxHashMap<FunctionId, FnRefs> = scc
                .iter()
                .filter_map(|fid| extract_refs.get(fid).map(|r| (*fid, r.clone())))
                .collect();
            let parsed_for_scc: FxHashMap<FunctionId, &ParsedSource> = scc
                .iter()
                .filter_map(|fid| extract_parsed.get(fid).map(|p| (*fid, *p)))
                .collect();

            let result = infer_scc(
                &self.interner,
                scc,
                &fn_by_id,
                &refs_for_scc,
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

            resolved_fn_types.extend(result.resolved_types.clone());
            self.infer_cache[scc_idx] = Some(result);
        }

        // Effect propagation.
        self.propagate_effects();
    }

    fn propagate_effects(&mut self) {
        // TODO: namespace-aware context resolution once AST supports @ns:name.
        // For now, only root contexts are resolved by name.
        let known_ctx_names: FxHashSet<Astr> = self
            .contexts
            .iter()
            .filter(|(qref, _)| qref.namespace.is_none())
            .map(|(qref, _)| qref.name)
            .collect();

        // Collect direct reads/writes from extract cache.
        for scc_idx in 0..self.scc_order.len() {
            let scc = &self.scc_order[scc_idx];
            let Some(ref mut scc_result) = self.infer_cache[scc_idx] else {
                continue;
            };

            for &fid in scc {
                let Some(extract_entry) = self.extract_cache.get(&fid) else {
                    continue;
                };
                let reads: std::collections::BTreeSet<QualifiedRef> = extract_entry
                    .refs
                    .context_reads
                    .iter()
                    .filter(|r| known_ctx_names.contains(&r.name))
                    .copied()
                    .collect();
                let writes: std::collections::BTreeSet<QualifiedRef> = extract_entry
                    .refs
                    .context_writes
                    .iter()
                    .filter(|r| known_ctx_names.contains(&r.name))
                    .copied()
                    .collect();
                if let Some(meta) = scc_result.fn_metas.get_mut(&fid) {
                    meta.effect = EffectSet {
                        reads,
                        writes,
                        io: false,
                        self_modifying: false,
                    };
                }
            }
        }

        // Propagate through call graph in SCC order (fixpoint within each SCC).
        // We need to collect all fn_metas into one map for propagation.
        let mut all_effects: FxHashMap<FunctionId, EffectSet> = FxHashMap::default();
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
                    Some((ctx.qualified_ref(), ty.clone()))
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
        let mut fn_refs = FxHashMap::default();
        let parsed = FxHashMap::default();
        for (&id, entry) in &self.extract_cache {
            fn_refs.insert(id, entry.refs.clone());
            // ParsedSource is not Clone — we need to handle this.
            // For now, skip parsed in snapshot (batch lower can re-extract if needed).
        }
        ExtractResult { fn_refs, parsed }
    }

    /// Build a snapshot InferResult for compatibility with batch APIs.
    pub fn infer_result(&self) -> super::infer::InferResult {
        let mut fn_params: FxHashMap<FunctionId, Vec<InferredParam>> = FxHashMap::default();
        let mut outcomes: FxHashMap<FunctionId, super::infer::FnInferOutcome> =
            FxHashMap::default();

        for scc_result in self.infer_cache.iter().flatten() {
            fn_params.extend(scc_result.fn_params.clone());
            // Convert SccInferResult metas to Incomplete outcomes (temporary — incremental
            // does not yet run check_completeness; this will be reworked in Step 6).
            for (&fid, meta) in &scc_result.fn_metas {
                outcomes.insert(
                    fid,
                    super::infer::FnInferOutcome::Incomplete {
                        unknown_contexts: scc_result
                            .fn_params
                            .get(&fid)
                            .cloned()
                            .unwrap_or_default(),
                        unknown_extern_params: vec![],
                        meta: meta.clone(),
                        errors: vec![],
                    },
                );
            }
        }

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

        super::infer::InferResult {
            outcomes,
            fn_params,
            all_params,
            context_types: self.known_context_types(),
        }
    }
}

fn hash_astr(s: Astr) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}
