//! Incremental compilation graph.
//!
//! Manages per-function extract/infer caches with dirty tracking.
//! On source change: re-extract -> diff call edges -> re-SCC if needed ->
//! re-infer dirty SCCs (with early cutoff), then lower and optimize.

use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::Refusal;
use crate::ir::MirModule;
use crate::laws::LawTable;
use crate::ty::{PolyTy, Sources, Ty, TypeRegistry, lift_to_poly};
use crate::typeck::ProbeProduct;

use super::extract::{ParsedSource, extract, extract_one};
use super::infer::{
    FnInferOutcome, Probe, SccInferResult, extract_call_edges, infer_scc, solve_contexts,
    tarjan_scc,
};
use super::lower::{close_fetched_first, lower_one};
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
    types: Freeze<TypeRegistry>,
    bindings: Bindings,

    // -- Source data --
    functions: FxHashMap<QualifiedRef, Function>,
    contexts: FxHashMap<QualifiedRef, Context>,
    /// `contexts` with each open one at the type the whole graph solves it
    /// to, which every component is inferred against.
    solved: Vec<Context>,
    /// The same fact `CompilationGraph::entries` carries, for a graph built
    /// by accumulation: empty until a host says which bodies it starts.
    entries: Vec<QualifiedRef>,

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
    pub fn new(interner: &Interner, graph: CompilationGraph) -> Self {
        let CompilationGraph {
            functions,
            contexts,
            types,
            bindings,
            entries,
        } = graph;
        let mut this = Self {
            interner: interner.clone(),
            sources: Sources::new(),
            types,
            bindings,
            functions: functions.iter().map(|f| (f.qref, f.clone())).collect(),
            contexts: contexts.iter().map(|c| (c.qref, c.clone())).collect(),
            solved: contexts.to_vec(),
            entries,
            extract_cache: FxHashMap::default(),
            call_edges: FxHashMap::default(),
            reverse_edges: FxHashMap::default(),
            scc_order: Vec::new(),
            fn_to_scc: FxHashMap::default(),
            infer_cache: Vec::new(),
            lower_cache: FxHashMap::default(),
            optimized: FxHashMap::default(),
            diagnostics: FxHashMap::default(),
        };
        let qrefs: Vec<QualifiedRef> = this.functions.keys().copied().collect();
        for qref in qrefs {
            this.run_extract(qref);
        }
        this.rebuild_graph();
        this
    }

    // -- Registration ------------------------------------------------

    pub fn add_function(&mut self, func: Function) {
        let qref = func.qref;
        self.functions.insert(qref, func);
        self.run_extract(qref);
        self.redraw_call_edges();
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
            self.redraw_call_edges();
            self.rebuild_graph();
        }
    }

    // -- Source update (main incremental entry point) ----------------

    pub fn update_ast(&mut self, qref: QualifiedRef, ast: ParsedAst) {
        let Some(func) = self.functions.get_mut(&qref) else {
            return;
        };
        match &mut func.kind {
            FnKind::Local(existing, _) => *existing = ast,
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
        } else if self.resolve_contexts() {
            self.infer_cache = vec![None; self.scc_order.len()];
            self.lower_cache.clear();
            self.run_infer();
            self.settle();
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

    pub fn inferred_ty(&self, qref: QualifiedRef) -> Option<&Ty> {
        Some(&self.fn_meta(qref)?.ty)
    }

    pub fn resolution(&self, qref: QualifiedRef) -> Option<Freeze<crate::typeck::TypeResolution>> {
        self.outcome(qref)?.resolution()
    }

    pub fn view(&self, qref: QualifiedRef) -> Option<Freeze<crate::typeck::BodyView>> {
        self.outcome(qref)?.view()
    }

    pub fn function(&self, qref: QualifiedRef) -> Option<&Function> {
        self.functions.get(&qref)
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// The contexts a body can name: `@name` is the root context `name`,
    /// the one key the checker reads it at, and the grammar writes no
    /// qualified context.
    pub fn visible_contexts(&self) -> impl Iterator<Item = &Context> {
        self.contexts
            .values()
            .filter(|context| context.qref.namespace.is_none())
    }

    /// Every function, which is every one a body can call: a bare name
    /// reaches the functions of that name in every namespace
    /// (`TypeEnv::resolve_fn`), and `ns::f` the one in `ns`.
    pub fn functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.values()
    }

    // -- Probe --------------------------------------------------------

    /// What the checker sees at `marker` with `probed` as the local body of
    /// its `qref`, which the graph may not hold, as while a document's text
    /// does not parse, and the rest of the graph as it stands. `None` where
    /// `qref` is an extern or `probed` is not local.
    pub fn probe(&self, probed: Function, marker: acvus_ast::AstId) -> Option<ProbeProduct> {
        let qref = probed.qref;
        self.check_as(probed, Some(Probe { body: qref, marker }))?
            .probe
    }

    /// The view of `probed` checked as the local body of its `qref`, as
    /// `probe` checks it: what the graph would record were the body
    /// replaced, while the graph keeps the body it holds. `None` where
    /// `qref` is an extern or `probed` is not local.
    pub fn view_as(&self, probed: Function) -> Option<Freeze<crate::typeck::BodyView>> {
        let qref = probed.qref;
        self.check_as(probed, None)?
            .outcomes
            .get(&qref)
            .expect("the checked SCC holds the replaced body")
            .view()
    }

    /// `probed`'s SCC inferred again, from the call edges its AST has, on
    /// a copy of the sources, so nothing the graph holds changes.
    fn check_as(&self, probed: Function, probe: Option<Probe>) -> Option<SccInferResult> {
        let qref = probed.qref;
        if let Some(Function {
            kind: FnKind::Extern { .. },
            ..
        }) = self.functions.get(&qref)
        {
            return None;
        }
        let parsed = extract_one(&self.interner, &probed)?;

        let mut call_edges = self.call_edges.clone();
        call_edges.insert(
            qref,
            extract_call_edges(&parsed, &self.root_fn_names(), qref),
        );
        let local_qrefs: Vec<QualifiedRef> = self
            .functions
            .values()
            .filter(|f| f.qref != qref && matches!(f.kind, FnKind::Local(..)))
            .map(|f| f.qref)
            .chain(std::iter::once(qref))
            .collect();
        let scc_order = tarjan_scc(&local_qrefs, &call_edges);
        let at = scc_order
            .iter()
            .position(|scc| scc.contains(&qref))
            .expect("a local function is in one SCC");

        // An SCC before the probe's reaches no body the probe changed, so
        // its members' types are the ones the graph settled.
        let mut resolved_fn_types = self.extern_fn_types();
        for member in scc_order[..at].iter().flatten() {
            let settled = self.fn_to_scc[member];
            let resolved = &self.infer_cache[settled]
                .as_ref()
                .expect("every SCC was inferred by the last settle")
                .resolved_types[member];
            resolved_fn_types.insert(*member, lift_to_poly(resolved));
        }

        let fn_by_id: FxHashMap<QualifiedRef, &Function> = self
            .functions
            .iter()
            .filter(|(member, f)| **member != qref && matches!(f.kind, FnKind::Local(..)))
            .map(|(&member, f)| (member, f))
            .chain(std::iter::once((qref, &probed)))
            .collect();
        let parsed_for_scc: FxHashMap<QualifiedRef, &ParsedSource> = scc_order[at]
            .iter()
            .filter_map(|&member| match member == qref {
                true => Some((member, &parsed)),
                false => self
                    .extract_cache
                    .get(&member)
                    .map(|entry| (member, &entry.parsed)),
            })
            .collect();

        let mut sources = self.sources.clone();
        Some(infer_scc(
            &self.interner,
            &scc_order[at],
            &self.entries,
            &self.bindings,
            &fn_by_id,
            &parsed_for_scc,
            &self.solved,
            &resolved_fn_types,
            &super::infer::declared_bounds(self.functions.values()),
            &mut sources,
            &self.types,
            probe,
        ))
    }

    // -- Internal: Extract -------------------------------------------

    fn run_extract(&mut self, qref: QualifiedRef) {
        let Some(func) = self.functions.get(&qref) else {
            return;
        };

        // Run extract.
        if let Some(parsed) = extract_one(&self.interner, func) {
            let new_edges = extract_call_edges(&parsed, &self.root_fn_names(), qref);
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

    // TODO: qualified call edges once AST supports ns:func() syntax.
    // For now, only unqualified (root) names are resolved.
    fn root_fn_names(&self) -> FxHashMap<Astr, QualifiedRef> {
        self.functions
            .iter()
            .filter(|(q, f)| q.namespace.is_none() && matches!(f.kind, FnKind::Local(..)))
            .map(|(&q, _)| (q.name, q))
            .collect()
    }

    /// Every function's call edges against the functions the graph holds
    /// now, since a name a body calls reaches a function added after it,
    /// and stops reaching one removed.
    fn redraw_call_edges(&mut self) {
        let names = self.root_fn_names();
        let call_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = self
            .extract_cache
            .iter()
            .map(|(&caller, entry)| (caller, extract_call_edges(&entry.parsed, &names, caller)))
            .collect();
        self.reverse_edges.clear();
        for (&caller, callees) in &call_edges {
            for &callee in callees {
                self.reverse_edges.entry(callee).or_default().push(caller);
            }
        }
        self.call_edges = call_edges;
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
            .filter(|f| matches!(f.kind, FnKind::Local(..)))
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
        let mut resolved_fn_types = self.extern_fn_types();

        let fn_by_id: FxHashMap<QualifiedRef, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(..)))
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
                &self.entries,
                &self.bindings,
                &fn_by_id,
                &parsed_owned,
                &self.solved,
                &resolved_fn_types,
                &super::infer::declared_bounds(self.functions.values()),
                &mut self.sources,
                &self.types,
                None,
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
        let mut resolved_fn_types = self.extern_fn_types();

        let fn_by_id: FxHashMap<QualifiedRef, &Function> = self
            .functions
            .iter()
            .filter(|(_, f)| matches!(f.kind, FnKind::Local(..)))
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
                &self.entries,
                &self.bindings,
                &fn_by_id,
                &parsed_for_scc,
                &self.solved,
                &resolved_fn_types,
                &super::infer::declared_bounds(self.functions.values()),
                &mut self.sources,
                &self.types,
                None,
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
        self.resolve_contexts();
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
        let mut modules: FxHashMap<QualifiedRef, MirModule> = self
            .lower_cache
            .iter()
            .filter(|(_, entry)| entry.refusals.is_empty())
            .map(|(&qref, entry)| (qref, entry.module.clone()))
            .collect();
        close_fetched_first(&mut modules);
        let laws = LawTable::of(self.functions.values());
        let result = optimize(&self.interner, &laws, modules, Opt::Full);

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

    fn extern_fn_types(&self) -> FxHashMap<QualifiedRef, PolyTy> {
        self.functions
            .iter()
            .filter(|(_, func)| matches!(func.kind, FnKind::Extern { .. }))
            .map(|(&qref, func)| (qref, func.ty.clone()))
            .collect()
    }

    /// Solve the open contexts from every body the graph holds now, as
    /// `infer` does; whether a type changed, in which case every component
    /// is inferred again.
    fn resolve_contexts(&mut self) -> bool {
        let graph = CompilationGraph {
            functions: Freeze::new(self.functions.values().cloned().collect()),
            contexts: Freeze::new(self.contexts.values().cloned().collect()),
            types: self.types.clone(),
            bindings: self.bindings.clone(),
            entries: self.entries.clone(),
        };
        let solved = solve_contexts(&self.interner, &graph, &extract(&self.interner, &graph));
        let same = |a: &[Context], b: &[Context]| {
            a.len() == b.len()
                && a.iter().all(|a| b.iter().any(|b| a.qref == b.qref && a.ty == b.ty))
        };
        let changed = !same(&self.solved, &solved);
        self.solved = solved;
        changed
    }
}
