//! Incremental compilation graph.
//!
//! Manages per-function extract/infer caches with dirty tracking.
//! On source change: re-extract -> diff call edges -> re-SCC if needed ->
//! re-infer dirty SCCs (with early cutoff), then lower and optimize.

use acvus_utils::{Freeze, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::Refusal;
use crate::ir::MirModule;
use crate::laws::LawTable;
use crate::ty::{InputParam, PolyTy, Sources, Ty, TypeRegistry, lift_to_poly};
use crate::typeck::ProbeProduct;

use super::extract::{ParsedSource, extract, extract_one, lift_one};
use super::infer::{
    CallTargets, FnInferOutcome, LocalFunction, Probe, SccInferResult, extract_call_edges,
    infer_scc, solve_contexts, tarjan_scc, typed_together,
};
use super::lower::{inputs_of, lower_one};
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
    access: Access,

    // -- Source data --
    functions: FxHashMap<QualifiedRef, Function>,
    /// The functions the lift made of the scripts' `fn`s (`graph::lift`),
    /// kept apart from `functions`, which are the ones the host gave.
    lifted: FxHashMap<QualifiedRef, Function>,
    facts: FxHashMap<QualifiedRef, LiftFacts>,
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
        acvus_utils::grow(|| Self::new_level(interner, graph))
    }

    fn new_level(interner: &Interner, graph: CompilationGraph) -> Self {
        let solved = solve_contexts(interner, &graph, &extract(interner, &graph));
        let CompilationGraph {
            functions,
            contexts,
            types,
            bindings,
            access,
            entries,
        } = graph;
        let mut this = Self {
            interner: interner.clone(),
            sources: Sources::new(),
            types,
            bindings,
            access,
            functions: functions.iter().map(|f| (f.qref, f.clone())).collect(),
            lifted: FxHashMap::default(),
            facts: FxHashMap::default(),
            contexts: contexts.iter().map(|c| (c.qref, c.clone())).collect(),
            solved,
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
        this.relift_every_script();
        let qrefs: Vec<QualifiedRef> = this.all_functions().map(|f| f.qref).collect();
        for qref in qrefs {
            this.run_extract(qref);
        }
        this.order_sccs();
        this.infer_all();
        this
    }

    // -- Registration ------------------------------------------------

    /// A function the host adds may take a name a script's `fn` has, so
    /// every script is lifted again.
    pub fn add_function(&mut self, func: Function) {
        acvus_utils::grow(|| self.add_function_level(func))
    }

    fn add_function_level(&mut self, func: Function) {
        let qref = func.qref;
        self.functions.insert(qref, func);
        self.run_extract(qref);
        self.relift_every_script();
        self.redraw_call_edges();
        self.rebuild_graph();
    }

    pub fn remove_function(&mut self, qref: QualifiedRef) {
        acvus_utils::grow(|| self.remove_function_level(qref))
    }

    fn remove_function_level(&mut self, qref: QualifiedRef) {
        if self.functions.remove(&qref).is_some() {
            self.forget(qref);
            self.relift_every_script();
            self.redraw_call_edges();
            self.rebuild_graph();
        }
    }

    fn forget(&mut self, qref: QualifiedRef) {
        self.extract_cache.remove(&qref);
        self.call_edges.remove(&qref);
        self.diagnostics.remove(&qref);
        self.lower_cache.remove(&qref);
        self.optimized.remove(&qref);
        self.facts.remove(&qref);
        self.remove_reverse_edges(qref);
    }

    /// The functions the host gave and the ones the lift made.
    fn all_functions(&self) -> impl Iterator<Item = &Function> {
        self.functions.values().chain(self.lifted.values())
    }

    /// Every script the graph holds, and every one whose instances it still
    /// holds after the script itself was removed.
    fn relift_every_script(&mut self) {
        let mut scripts: Vec<QualifiedRef> = self
            .functions
            .keys()
            .copied()
            .chain(self.lifted.keys().filter_map(|qref| qref.declaring_script()))
            .collect();
        scripts.sort();
        scripts.dedup();
        for script in scripts {
            self.relift(script);
        }
    }

    /// Replace what the lift made of `script` with a lift of its body as
    /// it is now; whether the script has or had a `fn`.
    fn relift(&mut self, script: QualifiedRef) -> bool {
        let stale: Vec<QualifiedRef> = self
            .lifted
            .keys()
            .copied()
            .filter(|qref| qref.declaring_script() == Some(script))
            .collect();
        let had = !stale.is_empty();
        for qref in stale {
            self.lifted.remove(&qref);
            self.forget(qref);
        }
        self.facts.remove(&script);
        let Some(func) = self.functions.get(&script) else {
            return had;
        };
        let declared: Vec<QualifiedRef> = self.functions.keys().copied().collect();
        let Some(lift) = lift_one(&self.interner, func, &declared, &self.types) else {
            return had;
        };
        let has = !lift.functions.is_empty();
        self.facts.extend(lift.facts);
        for instance in lift.functions {
            let qref = instance.qref;
            self.lifted.insert(qref, instance);
            self.run_extract(qref);
        }
        had || has
    }

    // -- Source update (main incremental entry point) ----------------

    pub fn update_ast(&mut self, qref: QualifiedRef, ast: ParsedAst) {
        acvus_utils::grow(|| self.update_ast_level(qref, ast))
    }

    fn update_ast_level(&mut self, qref: QualifiedRef, ast: ParsedAst) {
        let Some(func) = self.functions.get_mut(&qref) else {
            return;
        };
        match &mut func.kind {
            FnKind::Local(existing, _) => *existing = ast,
            FnKind::Extern { .. } => return,
        }
        if self.relift(qref) {
            self.run_extract(qref);
            self.redraw_call_edges();
            self.rebuild_graph();
            return;
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
            self.infer_all();
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
        meta.inputs
            .iter()
            .filter(|input| input.ty != Ty::Never)
            .map(|input| ContextInfo {
                name: QualifiedRef::root(input.name),
                ty: input.ty.clone(),
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

    /// A function the host gave, or one the lift made of a script's `fn`.
    pub fn function(&self, qref: QualifiedRef) -> Option<&Function> {
        self.functions.get(&qref).or_else(|| self.lifted.get(&qref))
    }

    /// The instances the lift made of `script`'s `fn`s.
    pub fn instances_of(&self, script: QualifiedRef) -> impl Iterator<Item = &Function> {
        self.lifted
            .values()
            .filter(move |function| function.qref.declaring_script() == Some(script))
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
        acvus_utils::grow(|| self.probe_level(probed, marker))
    }

    fn probe_level(&self, probed: Function, marker: acvus_ast::AstId) -> Option<ProbeProduct> {
        let qref = probed.qref;
        self.check_as(probed, Some(Probe { body: qref, marker }))?
            .probe
    }

    /// The view of `probed` checked as the local body of its `qref`, as
    /// `probe` checks it: what the graph would record were the body
    /// replaced, while the graph keeps the body it holds. `None` where
    /// `qref` is an extern or `probed` is not local.
    pub fn view_as(&self, probed: Function) -> Option<Freeze<crate::typeck::BodyView>> {
        acvus_utils::grow(|| self.view_as_level(probed))
    }

    fn view_as_level(&self, probed: Function) -> Option<Freeze<crate::typeck::BodyView>> {
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
        let probed_local = LocalFunction::of(&probed)?;
        let parsed = extract_one(&self.interner, &probed)?;
        let declared: Vec<QualifiedRef> = self.functions.keys().copied().collect();
        let lift = lift_one(&self.interner, &probed, &declared, &self.types);
        let (instances, lift_facts) = match lift {
            Some(lift) => (lift.functions, lift.facts),
            None => (Vec::new(), FxHashMap::default()),
        };
        let replaced = |member: &QualifiedRef| {
            *member == qref || member.declaring_script() == Some(qref)
        };
        let mut facts: FxHashMap<QualifiedRef, LiftFacts> = self
            .facts
            .iter()
            .filter(|(member, _)| !replaced(member))
            .map(|(&member, facts)| (member, facts.clone()))
            .chain(lift_facts)
            .collect();
        let probed_parsed: FxHashMap<QualifiedRef, ParsedSource> = instances
            .iter()
            .filter_map(|instance| Some((instance.qref, extract_one(&self.interner, instance)?)))
            .chain(std::iter::once((qref, parsed)))
            .collect();

        let names = self.root_fn_names();
        let mut call_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = self
            .call_edges
            .iter()
            .filter(|(member, _)| !replaced(member))
            .map(|(&member, edges)| (member, edges.clone()))
            .collect();
        for (&member, source) in &probed_parsed {
            call_edges.insert(
                member,
                extract_call_edges(source, &names, member, facts.get(&member)),
            );
        }
        facts.retain(|member, _| {
            self.functions.contains_key(member)
                || self.lifted.contains_key(member) && !replaced(member)
                || probed_parsed.contains_key(member)
        });
        let local_qrefs: Vec<QualifiedRef> = self
            .all_functions()
            .filter(|f| !replaced(&f.qref) && matches!(f.kind, FnKind::Local(..)))
            .map(|f| f.qref)
            .chain(probed_parsed.keys().copied())
            .collect();
        let scc_order = tarjan_scc(&local_qrefs, &typed_together(&call_edges, &facts));
        let at = scc_order
            .iter()
            .position(|scc| scc.contains(&qref))
            .expect("a local function is in one SCC");

        // An SCC before the probe's reaches no body the probe changed, so
        // its members' types are the ones the graph settled.
        let mut resolved_fn_types = self.extern_fn_types();
        let mut resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> = FxHashMap::default();
        for member in scc_order[..at].iter().flatten() {
            let Some(&settled) = self.fn_to_scc.get(member) else {
                continue;
            };
            let inferred = self.infer_cache[settled]
                .as_ref()
                .expect("every SCC was inferred by the last settle");
            resolved_fn_types.insert(*member, lift_to_poly(&inferred.resolved_types[member]));
            resolved_inputs.insert(*member, inferred.resolved_inputs[member].clone());
        }

        let fn_by_id: FxHashMap<QualifiedRef, LocalFunction<'_>> = self
            .all_functions()
            .filter(|f| !replaced(&f.qref))
            .filter_map(|f| Some((f.qref, LocalFunction::of(f)?)))
            .chain(instances.iter().map(|instance| {
                let local = LocalFunction::of(instance).expect("an instance of a `fn` has a body");
                (instance.qref, local)
            }))
            .chain(std::iter::once((qref, probed_local)))
            .collect();
        let parsed_for_scc: FxHashMap<QualifiedRef, &ParsedSource> = scc_order[at]
            .iter()
            .filter_map(|&member| match probed_parsed.get(&member) {
                Some(source) => Some((member, source)),
                None => self
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
            &resolved_inputs,
            &super::infer::declared_bounds(self.functions.values()),
            &facts,
            &mut sources,
            &self.types,
            probe,
            self.access,
        ))
    }

    // -- Internal: Extract -------------------------------------------

    fn run_extract(&mut self, qref: QualifiedRef) {
        let Some(func) = self.function(qref) else {
            return;
        };

        // Run extract.
        if let Some(parsed) = extract_one(&self.interner, func) {
            let new_edges =
                extract_call_edges(&parsed, &self.root_fn_names(), qref, self.facts.get(&qref));
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
    fn root_fn_names(&self) -> CallTargets {
        CallTargets::of(
            self.all_functions()
                .filter(|f| matches!(f.kind, FnKind::Local(..)))
                .map(|f| &f.qref),
        )
    }

    /// Every function's call edges against the functions the graph holds
    /// now, since a name a body calls reaches a function added after it,
    /// and stops reaching one removed.
    fn redraw_call_edges(&mut self) {
        let names = self.root_fn_names();
        let call_edges: FxHashMap<QualifiedRef, Vec<QualifiedRef>> = self
            .extract_cache
            .iter()
            .map(|(&caller, entry)| {
                let edges = extract_call_edges(&entry.parsed, &names, caller, self.facts.get(&caller));
                (caller, edges)
            })
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
        self.order_sccs();
        self.resolve_contexts();
        self.infer_all();
    }

    fn order_sccs(&mut self) {
        let local_qrefs: Vec<QualifiedRef> = self
            .all_functions()
            .filter(|f| matches!(f.kind, FnKind::Local(..)))
            .map(|f| f.qref)
            .collect();

        self.scc_order = tarjan_scc(&local_qrefs, &typed_together(&self.call_edges, &self.facts));
        self.fn_to_scc.clear();
        for (idx, scc) in self.scc_order.iter().enumerate() {
            for &fid in scc {
                self.fn_to_scc.insert(fid, idx);
            }
        }
    }

    fn infer_all(&mut self) {
        self.infer_cache = vec![None; self.scc_order.len()];
        self.lower_cache.clear();
        self.run_infer();
        self.settle();
    }

    // -- Internal: Infer ---------------------------------------------

    fn run_infer(&mut self) {
        let mut resolved_fn_types = self.extern_fn_types();
        let mut resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> = FxHashMap::default();

        let fn_by_id: FxHashMap<QualifiedRef, LocalFunction<'_>> = self
            .functions
            .values()
            .chain(self.lifted.values())
            .filter_map(|f| Some((f.qref, LocalFunction::of(f)?)))
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
                    resolved_inputs.extend(cached.resolved_inputs.clone());
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
                &resolved_inputs,
                &super::infer::declared_bounds(self.functions.values()),
                &self.facts,
                &mut self.sources,
                &self.types,
                None,
                self.access,
            );

            resolved_fn_types.extend(
                result
                    .resolved_types
                    .iter()
                    .map(|(&k, v)| (k, lift_to_poly(v))),
            );
            resolved_inputs.extend(result.resolved_inputs.clone());
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
        let mut resolved_inputs: FxHashMap<QualifiedRef, Vec<InputParam>> = FxHashMap::default();

        let fn_by_id: FxHashMap<QualifiedRef, LocalFunction<'_>> = self
            .functions
            .values()
            .chain(self.lifted.values())
            .filter_map(|f| Some((f.qref, LocalFunction::of(f)?)))
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
                resolved_inputs.extend(cached.resolved_inputs.clone());
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
                    resolved_inputs.extend(cached.resolved_inputs.clone());
                }
                continue;
            }

            let scc = &self.scc_order[scc_idx];
            let old_types = self.infer_cache[scc_idx]
                .as_ref()
                .map(|r| (r.resolved_types.clone(), r.resolved_inputs.clone()));

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
                &resolved_inputs,
                &super::infer::declared_bounds(self.functions.values()),
                &self.facts,
                &mut self.sources,
                &self.types,
                None,
                self.access,
            );

            // Early cutoff: if types didn't change, don't propagate.
            let types_changed = old_types
                .as_ref()
                .map(|(types, inputs)| {
                    types != &result.resolved_types || inputs != &result.resolved_inputs
                })
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
            resolved_inputs.extend(result.resolved_inputs.clone());
            self.infer_cache[scc_idx] = Some(result);
        }
    }

    // -- Internal: Lower + optimize -----------------------------------

    /// The modules are optimized at `Opt::Full`, the level `acvus check`
    /// reads the required inputs at: a constant a bound `$` puts behind a
    /// field or a loop is what `sroa` and the motion passes expose, and a
    /// set read at a lower level would be wider than the one the batch
    /// path reports.
    fn settle(&mut self) {
        // A script and the instances of its `fn`s are one source, which
        // `acvus check` refuses at typeck when any of them is refused, so
        // none of them is lowered then.
        let refused_sources: FxHashSet<QualifiedRef> = self
            .infer_cache
            .iter()
            .flatten()
            .flat_map(|scc| scc.errors())
            .map(|(qref, _)| qref.written_in())
            .collect();
        let lowerable: FxHashSet<QualifiedRef> = self
            .extract_cache
            .keys()
            .copied()
            .filter(|&qref| matches!(self.outcome(qref), Some(FnInferOutcome::Complete { .. })))
            .filter(|qref| !refused_sources.contains(&qref.written_in()))
            .collect();
        self.lower_cache.retain(|qref, _| lowerable.contains(qref));

        let to_lower: Vec<QualifiedRef> = lowerable
            .iter()
            .copied()
            .filter(|qref| !self.lower_cache.contains_key(qref))
            .collect();
        let callee_inputs = inputs_of(
            self.infer_cache
                .iter()
                .flatten()
                .flat_map(|inferred| &inferred.outcomes),
        );
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
                    &callee_inputs,
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
        let laws = LawTable::of(self.functions.values(), &self.types);
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

    /// Each refusal is the diagnostic of the function whose source it is
    /// written in: an instance's, of the script that declares the `fn`.
    /// Every instance is checked on its own (`graph::lift`), so one fault
    /// of a `fn`'s body is refused by each; the script holds it once.
    fn rebuild_diagnostics(&mut self) {
        let by_function = |stage: Vec<(QualifiedRef, Vec<Refusal>)>| {
            let mut stage = stage;
            stage.sort_by_key(|(qref, _)| *qref);
            stage
        };
        let checked = by_function(
            self.infer_cache
                .iter()
                .flatten()
                .flat_map(|scc| scc.errors())
                .map(|(qref, errors)| (qref, errors.iter().cloned().map(Refusal::Mir).collect()))
                .collect(),
        );
        let lowered = by_function(
            self.lower_cache
                .iter()
                .map(|(&qref, entry)| (qref, entry.refusals.clone()))
                .collect(),
        );
        let optimized = by_function(
            self.optimized
                .iter()
                .map(|(&qref, entry)| (qref, entry.refusals.clone()))
                .collect(),
        );
        let mut diagnostics: FxHashMap<QualifiedRef, Vec<Refusal>> = FxHashMap::default();
        for (qref, refusals) in checked.into_iter().chain(lowered).chain(optimized) {
            let held = diagnostics.entry(qref.written_in()).or_default();
            for refusal in refusals {
                if !held.iter().any(|earlier| self.same_fault(earlier, &refusal)) {
                    held.push(refusal);
                }
            }
        }
        diagnostics.retain(|_, refusals| !refusals.is_empty());
        self.diagnostics = diagnostics;
    }

    fn same_fault(&self, left: &Refusal, right: &Refusal) -> bool {
        left.span() == right.span()
            && left.labels() == right.labels()
            && left.display(&self.interner).to_string() == right.display(&self.interner).to_string()
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
            access: self.access,
            entries: self.entries.clone(),
        };
        let solved = solve_contexts(&self.interner, &graph, &extract(&self.interner, &graph));
        let same = |a: &[Context], b: &[Context]| {
            a.len() == b.len()
                && a.iter()
                    .all(|a| b.iter().any(|b| a.qref == b.qref && a.ty == b.ty))
        };
        let changed = !same(&self.solved, &solved);
        self.solved = solved;
        changed
    }
}
