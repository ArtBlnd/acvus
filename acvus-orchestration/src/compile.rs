

use std::collections::VecDeque;

use acvus_mir::context_registry::{ContextTypeRegistry, PartialContextTypeRegistry, RegistryConflictError};
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::{Effect, FnKind, Ty};
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::reachable_context::{
    ContextKeyPartition, KnownValue, partition_context_keys, reachable_context_keys,
};
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};
use acvus_utils::{Astr, Interner};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::TokenBudget;
use crate::convert::value_to_known;
use crate::dsl::{ContextScope, Execution, FnParam, MessageSpec, NodeLocalTypes, NodeSpec, Persistency};
use crate::error::{OrchError, OrchErrorKind};
use crate::kind::{
    CompiledNodeKind, NodeKind, compile_expr, compile_llm, compile_llm_cache, compile_plain,
    parse_type_name,
};
use crate::storage::EntryRef;

/// Compiled execution strategy.
#[derive(Debug, Clone)]
pub enum CompiledExecution {
    Always,
    OncePerTurn,
    IfModified { key: CompiledScript },
}

/// Compiled persistency mode.
#[derive(Debug, Clone)]
pub enum CompiledPersistency {
    Ephemeral,
    Snapshot,
    Deque { bind: CompiledScript },
    Diff { bind: CompiledScript },
}

/// Compiled strategy — groups execution, persistency, initial_value, retry, and assert.
#[derive(Debug, Clone)]
pub struct CompiledStrategy {
    pub execution: CompiledExecution,
    pub persistency: CompiledPersistency,
    pub initial_value: Option<CompiledScript>,
    pub retry: u32,
    pub assert: Option<CompiledScript>,
}

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub name: Astr,
    pub kind: CompiledNodeKind,
    pub all_context_keys: FxHashSet<Astr>,
    pub strategy: CompiledStrategy,
    pub is_function: bool,
    pub fn_params: Vec<FnParam>,
}

/// Compiled expression (Script → MIR).
#[derive(Debug, Clone)]
pub struct CompiledScript {
    pub module: MirModule,
    pub context_keys: FxHashSet<Astr>,
    pub val_def: ValDefMap,
}

/// A compiled message entry.
#[derive(Debug, Clone)]
pub enum CompiledMessage {
    Block(CompiledBlock),
    Iterator {
        expr: CompiledScript,
        slice: Option<Vec<i64>>,
        role: Option<Astr>,
        token_budget: Option<TokenBudget>,
    },
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub role: Astr,
    pub module: MirModule,
    pub context_keys: FxHashSet<Astr>,
    pub val_def: ValDefMap,
}

impl CompiledBlock {
    /// Context keys still needed on live execution paths, given known values.
    ///
    /// Uses dead branch pruning: if a known value resolves a branch condition,
    /// context loads in the dead branch are excluded.
    pub fn required_context_keys(&self, known: &FxHashMap<Astr, KnownValue>) -> FxHashSet<Astr> {
        reachable_context_keys(&self.module, known, &self.val_def)
    }
}

impl CompiledNode {
    /// Context keys still needed across all blocks in this node, given known values.
    ///
    /// Aggregates results from all message blocks and the key module (if present).
    /// Keys in `resolvable` (e.g. dependency node names) are excluded.
    pub fn required_context_keys(
        &self,
        known: &FxHashMap<Astr, KnownValue>,
        resolvable: &FxHashSet<Astr>,
    ) -> FxHashSet<Astr> {
        let mut needed = FxHashSet::default();
        for msg in self.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                needed.extend(block.required_context_keys(known));
            }
        }
        needed.retain(|k| !resolvable.contains(k));
        needed
    }

    /// Context keys that must be provided externally.
    ///
    /// Reads already-resolved values from `storage` for dead branch pruning,
    /// and excludes keys in `resolvable` (dependency nodes that auto-resolve).
    pub fn required_external_keys(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
        resolvable: &FxHashSet<Astr>,
    ) -> FxHashSet<Astr> {
        let known = self.known_from_entry(interner, entry);
        self.required_context_keys(&known, resolvable)
    }

    /// Partition context keys into eager (definitely needed) and lazy
    /// (conditionally needed), excluding resolvable dependency nodes.
    pub fn partition_external_keys(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
        resolvable: &FxHashSet<Astr>,
    ) -> ContextKeyPartition {
        let known = self.known_from_entry(interner, entry);
        let mut merged = ContextKeyPartition::default();
        for msg in self.kind.messages() {
            if let CompiledMessage::Block(block) = msg {
                let p = partition_context_keys(&block.module, &known, &block.val_def);
                merged.eager.extend(p.eager);
                merged.lazy.extend(p.lazy);
            }
        }
        merged.eager.retain(|k| !resolvable.contains(k));
        merged
            .lazy
            .retain(|k| !resolvable.contains(k) && !merged.eager.contains(k));
        merged
    }

    pub(crate) fn known_from_entry(
        &self,
        interner: &Interner,
        entry: &dyn EntryRef<'_>,
    ) -> FxHashMap<Astr, KnownValue> {
        self.all_context_keys
            .iter()
            .filter_map(|k| {
                let arc = entry.get(interner.resolve(*k))?;
                let known = value_to_known(&arc)?;
                Some((*k, known))
            })
            .collect()
    }
}

/// Compile an expression string (script syntax) with type checking.
/// Returns the compiled script and its tail expression type.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> Result<(CompiledScript, Ty), OrchError> {
    compile_script_with_hint(interner, source, registry, None)
}

/// Compile a script with an optional expected tail type hint for unification.
pub fn compile_script_with_hint(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(CompiledScript, Ty), OrchError> {
    let script = acvus_ast::parse_script(interner, source).map_err(|e| {
        OrchError::new(OrchErrorKind::ScriptParse {
            error: format!("{e}"),
        })
    })?;
    let (module, _hints, tail_ty) = acvus_mir::compile_script_with_hint(
        interner,
        &script,
        registry,
        expected_tail,
    )
    .map_err(|errs| {
        OrchError::new(OrchErrorKind::ScriptCompile {
            context: source.to_string(),
            errors: errs,
        })
    })?;
    let context_keys = extract_context_keys(&module);
    let val_def = ValDefMapAnalysis.run(&module, ());
    Ok((
        CompiledScript {
            module,
            context_keys,
            val_def,
        },
        tail_ty,
    ))
}

// ── Script output type expectations ──────────────────────────────────
//
//   Field               Expected type            Notes
//   ──────────────────  ───────────────────────  ──────────────────────────
//   iterator + body     List<T>                  T bound to context for body
//   iterator (no body)  List<MESSAGE_ELEM_TY>    elements used as messages directly
//   cache_key           String
//   bind script         (any)
//

/// Expect the tail type to be `List<T>`. Returns the inner `T`.
pub(crate) fn expect_list(context: &str, ty: Ty) -> Result<Ty, OrchError> {
    match ty {
        Ty::List(inner) | Ty::Deque(inner, _) => Ok(*inner),
        Ty::Error => Ok(Ty::Error),
        other => Err(OrchError::new(OrchErrorKind::ScriptTypeMismatch {
            context: context.into(),
            expected: Ty::List(Box::new(Ty::Infer)),
            got: other,
        })),
    }
}

/// Expect the tail type to be exactly `expected`.
pub(crate) fn expect_ty(context: &str, ty: &Ty, expected: &Ty) -> Result<(), OrchError> {
    if matches!(ty, Ty::Error) {
        return Ok(());
    }
    let mut subst = acvus_mir::ty::TySubst::new();
    match subst.unify(ty, expected, acvus_mir::ty::Polarity::Covariant) {
        Ok(()) => Ok(()),
        Err(_) => Err(OrchError::new(OrchErrorKind::ScriptTypeMismatch {
            context: context.into(),
            expected: expected.clone(),
            got: ty.clone(),
        })),
    }
}

/// Compile a template source string into a `CompiledBlock`.
pub(crate) fn compile_template(
    interner: &Interner,
    source: &str,
    block_idx: usize,
    registry: &ContextTypeRegistry,
) -> Result<CompiledBlock, OrchError> {
    let ast = acvus_ast::parse(interner, source).map_err(|e| {
        OrchError::new(OrchErrorKind::TemplateParse {
            block: block_idx,
            error: format!("{e}"),
        })
    })?;

    let (module, _hints) = acvus_mir::compile(
        interner,
        &ast,
        registry,
    )
    .map_err(|errs| {
        OrchError::new(OrchErrorKind::TemplateCompile {
            block: block_idx,
            errors: errs,
        })
    })?;

    let context_keys = extract_context_keys(&module);
    let val_def = ValDefMapAnalysis.run(&module, ());

    Ok(CompiledBlock {
        role: interner.intern(""),
        module,
        context_keys,
        val_def,
    })
}

/// Compile messages from a message spec list.
pub(crate) fn compile_messages(
    interner: &Interner,
    messages: &[MessageSpec],
    registry: &ContextTypeRegistry,
    iterator_elem_ty: &Ty,
) -> Result<(Vec<CompiledMessage>, FxHashSet<Astr>), Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = FxHashSet::default();
    let mut errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, source } => {
                let block = match compile_template(interner, source, i, registry) {
                    Ok(b) => b,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                all_context_keys.extend(block.context_keys.iter().copied());
                compiled_messages.push(CompiledMessage::Block(CompiledBlock {
                    role: *role,
                    ..block
                }));
            }
            MessageSpec::Iterator {
                key,
                slice,
                role,
                token_budget,
            } => {
                let ctx = format!("iterator (block {i})");
                let (expr, tail_ty) =
                    match compile_script(interner, interner.resolve(*key), registry) {
                        Ok(v) => v,
                        Err(e) => {
                            errors.push(e);
                            continue;
                        }
                    };
                let elem_ty = match expect_list(&ctx, tail_ty) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                if let Err(e) = expect_ty(&ctx, &elem_ty, iterator_elem_ty) {
                    errors.push(e);
                    continue;
                }

                all_context_keys.extend(expr.context_keys.iter().copied());
                compiled_messages.push(CompiledMessage::Iterator {
                    expr,
                    slice: slice.clone(),
                    role: *role,
                    token_budget: token_budget.clone(),
                });
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok((compiled_messages, all_context_keys))
}

/// Compile a node spec into a `CompiledNode`.
///
/// `registry` must already include @self (if applicable) via `build_node_context`.
/// `initial_value` is the compiled initial_value script.
/// Each message's `source` field is compiled directly — no file I/O.
pub fn compile_node(
    interner: &Interner,
    spec: &NodeSpec,
    registry: &ContextTypeRegistry,
    initial_value: Option<CompiledScript>,
    compiled_execution: CompiledExecution,
    compiled_persistency: CompiledPersistency,
    compiled_assert: Option<CompiledScript>,
) -> Result<CompiledNode, Vec<OrchError>> {
    let (kind, mut all_context_keys): (_, FxHashSet<_>) = match &spec.kind {
        NodeKind::Plain(plain_spec) => {
            let (compiled, keys) = compile_plain(interner, plain_spec, registry)?;
            (CompiledNodeKind::Plain(compiled), keys)
        }
        NodeKind::Llm(llm_spec) => {
            let (compiled, keys) = compile_llm(interner, llm_spec, registry)?;
            (CompiledNodeKind::Llm(compiled), keys)
        }
        NodeKind::LlmCache(cache_spec) => {
            let (compiled, keys) =
                compile_llm_cache(interner, cache_spec, registry)?;
            (CompiledNodeKind::LlmCache(compiled), keys)
        }
        NodeKind::Expr(expr_spec) => {
            let (compiled, keys) = compile_expr(interner, expr_spec, registry)?;
            (CompiledNodeKind::Expr(compiled), keys)
        }
    };

    // initial_value context keys contribute to dependencies
    if let Some(ref iv) = initial_value {
        all_context_keys.extend(iv.context_keys.iter().copied());
    }

    // assert context keys contribute
    if let Some(ref compiled_assert) = compiled_assert {
        all_context_keys.extend(compiled_assert.context_keys.iter().copied());
    }

    // execution context keys contribute
    match &compiled_execution {
        CompiledExecution::Always | CompiledExecution::OncePerTurn => {}
        CompiledExecution::IfModified { key } => {
            all_context_keys.extend(key.context_keys.iter().copied());
        }
    }

    // persistency context keys contribute
    match &compiled_persistency {
        CompiledPersistency::Ephemeral | CompiledPersistency::Snapshot => {}
        CompiledPersistency::Deque { bind } | CompiledPersistency::Diff { bind } => {
            all_context_keys.extend(bind.context_keys.iter().copied());
        }
    }

    Ok(CompiledNode {
        name: spec.name,
        kind,
        all_context_keys,
        strategy: CompiledStrategy {
            execution: compiled_execution,
            persistency: compiled_persistency,
            initial_value,
            retry: spec.strategy.retry,
            assert: compiled_assert,
        },
        is_function: spec.is_function,
        fn_params: spec.fn_params.clone(),
    })
}

/// Result of computing external context types from node specs.
///
/// `context_types` contains all externally-visible types:
/// - injected types (from project.toml / bindings)
/// - `@nodeName` → stored type (from self_bind tail)
/// - `@turn_index` → Int (system context)
///
/// Local types (`@self`, `@raw`) are NOT included — use `NodeLocalTypes` from dsl.

pub struct ExternalContextEnv {
    pub registry: PartialContextTypeRegistry,
    /// Types of values stored in storage (node self types + @turn_index).
    /// Does not include injected types (those come from the resolver, not storage).
    pub storage_types: FxHashMap<Astr, Ty>,
    /// Per-node local types, indexed by node name.
    pub node_locals: FxHashMap<Astr, NodeLocalTypes>,
    pub(crate) stored_types: Vec<Ty>,
}

/// Resolve `NodeLocalTypes` for a single node — the SINGLE source of truth for @self and @raw types.
///
/// `raw_ty` is the node's raw output type — passed explicitly because Expr nodes may have
/// their type resolved externally (via topo sort) rather than from the spec.
///
/// - `@raw` = `raw_ty` (what the node produces each execution)
/// - `@self`:
///   - Deque/Diff with bind: two-pass — compile bind with `@self = Infer`, use tail_ty as self_ty
///   - Ephemeral/Snapshot with initial_value: initial_value's return type
///   - No initial_value: self_ty = raw_ty (fallback, @self not exposed without initial_value)
fn resolve_node_locals(
    interner: &Interner,
    spec: &NodeSpec,
    base_registry: &ContextTypeRegistry,
    raw_ty: Ty,
) -> Result<NodeLocalTypes, Vec<OrchError>> {
    let map_conflict = |e: RegistryConflictError| {
        vec![OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })]
    };

    let self_ty = match &spec.strategy.persistency {
        Persistency::Deque { bind } | Persistency::Diff { bind } => {
            // Two-pass: compile bind with @self = Infer → typechecker infers self_ty from usage
            let temp_locals = NodeLocalTypes { raw_ty: raw_ty.clone(), self_ty: Ty::Infer };
            let bind_reg = spec.build_node_context(interner, base_registry, ContextScope::Bind, Some(&temp_locals))
                .map_err(map_conflict)?;
            let resolved_self = compile_script_with_hint(
                interner,
                interner.resolve(*bind),
                &bind_reg,
                None,
            )
            .map(|(_, ty)| ty)
            .unwrap_or(Ty::Error);

            // Validate initial_value against resolved self_ty
            if let Some(ref init_src) = spec.strategy.initial_value {
                let init_reg = spec.build_node_context(interner, base_registry, ContextScope::InitialValue, None)
                    .map_err(map_conflict)?;
                let hint = if resolved_self != Ty::Error { Some(&resolved_self) } else { None };
                let _ = compile_script_with_hint(interner, interner.resolve(*init_src), &init_reg, hint);
            }

            resolved_self
        }
        _ => {
            // Ephemeral/Snapshot: self_ty from initial_value if present, else raw_ty
            if let Some(ref init_src) = spec.strategy.initial_value {
                let init_reg = spec.build_node_context(interner, base_registry, ContextScope::InitialValue, None)
                    .map_err(map_conflict)?;
                compile_script_with_hint(interner, interner.resolve(*init_src), &init_reg, None)
                    .map(|(_, ty)| ty)
                    .unwrap_or(Ty::Error)
            } else {
                raw_ty.clone()
            }
        }
    };

    Ok(NodeLocalTypes { raw_ty, self_ty })
}

/// Wrap a type as `Ty::Fn` if the node is a function node.
fn wrap_fn_ty(spec: &NodeSpec, ty: Ty) -> Ty {
    if spec.is_function {
        let param_types: Vec<Ty> = spec.fn_params.iter().map(|p| p.ty.clone()).collect();
        Ty::Fn {
            params: param_types,
            ret: Box::new(ty),
            kind: FnKind::Extern,
            captures: vec![],
            effect: Effect::Pure,
        }
    } else {
        ty
    }
}

/// Compute all externally-visible context types from node specs.
///
/// Pipeline:
/// 1. Build a temporary `working_ctx` with raw types (for intermediate compilation)
/// 2. Resolve Expr node types via topo sort (progressively adding to `working_ctx`)
/// 3. Compute node_locals → determine final `@name` types (`stored_types`, immutable)
/// 4. Register everything into registry — once per key
///
/// The result can be used for typechecking binding scripts or passed
/// to `compile_nodes_with_env` for full compilation.
pub fn compute_external_context_env(
    interner: &Interner,
    specs: &[NodeSpec],
    mut registry: PartialContextTypeRegistry,
) -> Result<ExternalContextEnv, Vec<OrchError>> {
    let map_conflict = |e: acvus_mir::context_registry::RegistryConflictError| {
        vec![OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })]
    };

    // ── Phase 1: Register system types into registry ───────────────────
    //
    // Add @turn_index and concrete node raw types directly to the registry.
    // For intermediate compilation (Phase 2-3), create temporary full registries
    // via `registry.to_full()`.

    let raw_types: Vec<Ty> = specs
        .iter()
        .map(|s| s.kind.raw_output_ty(interner))
        .collect();

    registry
        .insert_system(interner.intern("turn_index"), Ty::Int)
        .map_err(map_conflict)?;

    for (spec, ty) in specs.iter().zip(raw_types.iter()) {
        if *ty == Ty::Infer {
            continue;
        }
        registry.insert_system(spec.name, wrap_fn_ty(spec, ty.clone()))
            .map_err(map_conflict)?;
    }

    // ── Phase 2: Resolve Expr node types via topo sort ───────────────────
    //
    // Expr nodes start with raw_type = Infer. Topo sort compiles them in
    // dependency order, progressively adding resolved types to the registry.
    // expr_resolved[idx] holds the resolved raw type for each Expr node.

    let infer_indices: Vec<usize> = (0..specs.len())
        .filter(|&i| raw_types[i] == Ty::Infer)
        .collect();

    let mut expr_resolved: FxHashMap<usize, Ty> = FxHashMap::default();

    if !infer_indices.is_empty() {
        let node_name_to_idx: FxHashMap<Astr, usize> =
            specs.iter().enumerate().map(|(i, s)| (s.name, i)).collect();
        let infer_set: FxHashSet<usize> = infer_indices.iter().copied().collect();

        let n = specs.len();
        let mut deps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];
        let mut rdeps: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); n];

        // Use a temporary full registry for analysis extraction
        let temp_reg = registry.to_full();
        for &idx in &infer_indices {
            if let NodeKind::Expr(expr_spec) = &specs[idx].kind {
                let keys =
                    analysis_extract_script_keys(interner, &expr_spec.source, &temp_reg);
                for key in keys {
                    if let Some(&dep_idx) = node_name_to_idx.get(&key) {
                        if infer_set.contains(&dep_idx) && dep_idx != idx {
                            deps[idx].insert(dep_idx);
                            rdeps[dep_idx].insert(idx);
                        }
                    }
                }
            }
        }

        // Kahn's algorithm on Infer nodes
        let mut in_degree: FxHashMap<usize, usize> = infer_indices
            .iter()
            .map(|&i| (i, deps[i].len()))
            .collect();
        let mut queue: VecDeque<usize> = infer_indices
            .iter()
            .filter(|&&i| deps[i].is_empty())
            .copied()
            .collect();
        let mut topo = Vec::with_capacity(infer_indices.len());

        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in &rdeps[u] {
                if let Some(deg) = in_degree.get_mut(&v) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }

        if topo.len() != infer_indices.len() {
            let in_cycle: Vec<String> = infer_indices
                .iter()
                .filter(|&&i| in_degree[&i] > 0)
                .map(|&i| interner.resolve(specs[i].name).to_string())
                .collect();
            return Err(vec![OrchError::new(OrchErrorKind::CycleDetected {
                nodes: in_cycle,
            })]);
        }

        for &idx in &topo {
            if let NodeKind::Expr(expr_spec) = &specs[idx].kind {
                let hint = match &expr_spec.output_ty {
                    Ty::Infer => None,
                    ty => Some(ty),
                };
                let full_reg = registry.to_full();
                let temp_raw_ty = specs[idx].kind.raw_output_ty(interner);
                let temp_locals = resolve_node_locals(interner, &specs[idx], &full_reg, temp_raw_ty)?;
                let expr_reg = specs[idx].build_node_context(interner, &full_reg, ContextScope::Body, Some(&temp_locals))
                    .map_err(map_conflict)?;
                let resolved = compile_script_with_hint(
                    interner,
                    &expr_spec.source,
                    &expr_reg,
                    hint,
                )
                .map(|(_, ty)| ty)
                .unwrap_or(Ty::Error);

                registry.insert_system(specs[idx].name, wrap_fn_ty(&specs[idx], resolved.clone()))
                    .map_err(map_conflict)?;
                expr_resolved.insert(idx, resolved);
            }
        }
    }

    // ── Phase 3: Compute node_locals → final stored_types (immutable) ────
    //
    // For each node, resolve @self and @raw via resolve_node_locals,
    // then determine the final @name type.
    // Use the now-complete registry for compilation.

    let full_reg = registry.to_full();
    let mut node_locals = FxHashMap::default();
    let mut stored_types: Vec<Ty> = Vec::with_capacity(specs.len());

    for (i, spec) in specs.iter().enumerate() {
        let raw_ty = expr_resolved.get(&i).cloned().unwrap_or_else(|| raw_types[i].clone());
        let locals = resolve_node_locals(interner, spec, &full_reg, raw_ty)?;
        let name_ty = if spec.strategy.initial_value.is_some() {
            locals.self_ty.clone()
        } else {
            locals.raw_ty.clone()
        };
        stored_types.push(name_ty);
        node_locals.insert(spec.name, locals);
    }

    // ── Phase 3b: Validate purity — non-Ephemeral nodes must have pure stored types ──
    for (i, spec) in specs.iter().enumerate() {
        if !matches!(spec.strategy.persistency, Persistency::Ephemeral) && !stored_types[i].is_pureable() {
            let persistency_name = match &spec.strategy.persistency {
                Persistency::Ephemeral => unreachable!(),
                Persistency::Snapshot => "Snapshot",
                Persistency::Deque { .. } => "Deque",
                Persistency::Diff { .. } => "Diff",
            };
            return Err(vec![OrchError::new(OrchErrorKind::UnpureStoredType {
                node: spec.name,
                persistency: persistency_name.to_string(),
                ty: stored_types[i].clone(),
            })]);
        }
    }

    let storage_types = registry.system().clone();

    Ok(ExternalContextEnv {
        registry,
        storage_types,
        node_locals,
        stored_types,
    })
}

/// Compile multiple node specs, merging their output types into context automatically.
///
/// `injected_types` are externally declared context types (from project.toml).
/// Node stored types are derived from `initial_value` scripts.
pub fn compile_nodes(
    interner: &Interner,
    specs: &[NodeSpec],
    registry: PartialContextTypeRegistry,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let env = compute_external_context_env(interner, specs, registry)?;
    compile_nodes_with_env(interner, specs, env)
}

/// Compile nodes using a pre-computed external context environment.
///
/// Use this when you already called `compute_external_context_env` (e.g. for
/// typechecking binding scripts) and want to avoid recomputation.
pub fn compile_nodes_with_env(
    interner: &Interner,
    specs: &[NodeSpec],
    env: ExternalContextEnv,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let base_registry = env.registry.to_full();
    let stored_types = env.stored_types;
    let mut errors = Vec::new();

    let map_conflict = |e: RegistryConflictError| {
        OrchError::new(OrchErrorKind::RegistryConflict {
            key: e.key,
            tier_a: e.tier_a,
            tier_b: e.tier_b,
        })
    };

    // Compile initial_value scripts — all node kinds.
    let mut initial_value_scripts: Vec<Option<CompiledScript>> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let Some(ref init_src) = spec.strategy.initial_value else {
            initial_value_scripts.push(None);
            continue;
        };
        let hint = match &stored_types[i] {
            Ty::Error => None,
            ty => Some(ty),
        };
        let init_reg = match spec.build_node_context(interner, &base_registry, ContextScope::InitialValue, env.node_locals.get(&spec.name)) {
            Ok(r) => r,
            Err(e) => {
                errors.push(map_conflict(e));
                initial_value_scripts.push(None);
                continue;
            }
        };
        let (script, init_ty) = match compile_script_with_hint(
            interner,
            interner.resolve(*init_src),
            &init_reg,
            hint,
        ) {
            Ok(v) => v,
            Err(e) => {
                errors.push(e);
                initial_value_scripts.push(None);
                continue;
            }
        };
        if let Err(e) = expect_ty(
            &format!("node '{}' initial_value type", interner.resolve(spec.name)),
            &init_ty,
            &stored_types[i],
        ) {
            errors.push(e);
        }
        initial_value_scripts.push(Some(script));
    }

    // Compile execution for each node
    let mut compiled_executions: Vec<CompiledExecution> = Vec::new();
    for spec in specs.iter() {
        let exec = match &spec.strategy.execution {
            Execution::Always => CompiledExecution::Always,
            Execution::OncePerTurn => CompiledExecution::OncePerTurn,
            Execution::IfModified { key } => {
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*key),
                    &base_registry,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_executions.push(CompiledExecution::Always);
                        continue;
                    }
                };
                CompiledExecution::IfModified { key: script }
            }
        };
        compiled_executions.push(exec);
    }

    // Compile persistency for each node
    let mut compiled_persistencies: Vec<CompiledPersistency> = Vec::new();
    for spec in specs.iter() {
        let persistency = match &spec.strategy.persistency {
            Persistency::Ephemeral => CompiledPersistency::Ephemeral,
            Persistency::Snapshot => CompiledPersistency::Snapshot,
            Persistency::Deque { bind } | Persistency::Diff { bind } => {
                // bind context: @self + @raw + all context (via ContextScope::Bind)
                let bind_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Bind, env.node_locals.get(&spec.name)) {
                    Ok(r) => r,
                    Err(e) => {
                        errors.push(map_conflict(e));
                        compiled_persistencies.push(CompiledPersistency::Ephemeral);
                        continue;
                    }
                };
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*bind),
                    &bind_reg,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_persistencies.push(CompiledPersistency::Ephemeral);
                        continue;
                    }
                };
                match &spec.strategy.persistency {
                    Persistency::Deque { .. } => CompiledPersistency::Deque { bind: script },
                    Persistency::Diff { .. } => CompiledPersistency::Diff { bind: script },
                    _ => unreachable!(),
                }
            }
        };
        compiled_persistencies.push(persistency);
    }

    // Compile assert for each node (Bind scope: @self + @raw)
    let mut compiled_asserts: Vec<Option<CompiledScript>> = Vec::new();
    for spec in specs.iter() {
        let compiled_assert = if let Some(ref assert_src) = spec.strategy.assert {
            let assert_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Bind, env.node_locals.get(&spec.name)) {
                Ok(r) => r,
                Err(e) => {
                    errors.push(map_conflict(e));
                    compiled_asserts.push(None);
                    continue;
                }
            };
            let (script, _ty) = match compile_script_with_hint(
                interner,
                interner.resolve(*assert_src),
                &assert_reg,
                Some(&Ty::Bool),
            ) {
                Ok(v) => v,
                Err(e) => {
                    errors.push(e);
                    compiled_asserts.push(None);
                    continue;
                }
            };
            Some(script)
        } else {
            None
        };
        compiled_asserts.push(compiled_assert);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool param types must be injected from the caller (LlmSpec) side because
    // the target node alone cannot know what types its params will have — the
    // param types are declared in ToolBinding, not in the target node itself.
    let mut tool_param_types: FxHashMap<Astr, Vec<(Astr, Ty)>> = FxHashMap::default();
    for spec in specs {
        let NodeKind::Llm(llm_spec) = &spec.kind else {
            continue;
        };
        for tool in &llm_spec.tools {
            let params: Vec<(Astr, Ty)> = tool
                .params
                .iter()
                .filter_map(|(k, v)| Some((interner.intern(k), parse_type_name(&v.ty)?)))
                .collect();
            tool_param_types
                .entry(interner.intern(&tool.node))
                .or_default()
                .extend(params);
        }
    }

    let mut nodes = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let mut node_reg = match spec.build_node_context(interner, &base_registry, ContextScope::Body, env.node_locals.get(&spec.name)) {
            Ok(r) => r,
            Err(e) => {
                errors.push(map_conflict(e));
                continue;
            }
        };
        if let Some(params) = tool_param_types.get(&spec.name) {
            node_reg = match node_reg.with_extra_scoped(params.iter().map(|(k, v)| (*k, v.clone()))) {
                Ok(r) => r,
                Err(e) => {
                    errors.push(map_conflict(e));
                    continue;
                }
            };
        }
        // Validate fn_param names don't conflict with context keys
        // (fn_params are already in the registry via build_node_context, so
        // conflicts would have been caught by with_extra_scoped above)
        let initial_value = initial_value_scripts[i].clone();
        let compiled_execution = compiled_executions[i].clone();
        let compiled_persistency = compiled_persistencies[i].clone();
        let compiled_assert = compiled_asserts[i].clone();
        match compile_node(
            interner,
            spec,
            &node_reg,
            initial_value,
            compiled_execution,
            compiled_persistency,
            compiled_assert,
        ) {
            Ok(node) => nodes.push(node),
            Err(errs) => {
                errors.extend(errs);
                continue;
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool targets are not captured in all_context_keys (they are invoked
    // dynamically by the model, not via @ref in templates), so we validate
    // their existence separately.
    let node_names: FxHashSet<Astr> = nodes.iter().map(|n| n.name).collect();
    for node in &nodes {
        if let CompiledNodeKind::Llm(llm) = &node.kind {
            for tool in &llm.tools {
                let tool_node = interner.intern(&tool.node);
                if !node_names.contains(&tool_node) {
                    errors.push(OrchError::new(OrchErrorKind::ToolTargetNotFound {
                        tool: tool.name.clone(),
                        target: tool.node.clone(),
                    }));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(nodes)
    } else {
        Err(errors)
    }
}

/// Extract context keys from a script source using analysis mode.
///
/// Analysis mode assigns fresh type variables for unknown `@context` refs,
/// so context keys can be discovered even before all types are known.
fn analysis_extract_script_keys(
    interner: &Interner,
    source: &str,
    registry: &ContextTypeRegistry,
) -> FxHashSet<Astr> {
    let Ok(script) = acvus_ast::parse_script(interner, source) else {
        return FxHashSet::default();
    };
    let (module, _, _, _) = acvus_mir::compile_script_analysis_with_tail_partial(
        interner,
        &script,
        registry,
        None,
    );
    extract_context_keys(&module)
}

/// Extract all context keys referenced by `ContextLoad` instructions in a module.
fn extract_context_keys(module: &MirModule) -> FxHashSet<Astr> {
    let mut keys = FxHashSet::default();

    for inst in &module.main.insts {
        if let InstKind::ContextLoad { name, .. } = &inst.kind {
            keys.insert(*name);
        }
    }

    for closure in module.closures.values() {
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind {
                keys.insert(*name);
            }
        }
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::ty::Ty;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_utils::Interner;
    use rustc_hash::FxHashMap;
    use crate::dsl::*;
    use crate::kind::{NodeKind, LlmSpec};
    use crate::MessageSpec as MsgSpec;

    #[test]
    fn deque_bind_self_extend_raw() {
        let interner = Interner::new();
        let registry = PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ).unwrap();

        let spec = NodeSpec {
            name: interner.intern("chat"),
            kind: NodeKind::Llm(LlmSpec {
                api: crate::ApiKind::OpenAI,
                provider: String::new(),
                model: String::new(),
                messages: vec![MsgSpec::Block {
                    role: interner.intern("user"),
                    source: "hello".into(),
                }],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
                max_tokens: Default::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Deque {
                    bind: interner.intern("@self | extend(@raw)"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let env = compute_external_context_env(&interner, &[spec], registry)
            .expect("compute_external_context_env should succeed");

        let chat_name = interner.intern("chat");
        let locals = env.node_locals.get(&chat_name).expect("should have locals for chat");
        eprintln!("self_ty = {}", locals.self_ty.display(&interner));
        eprintln!("raw_ty = {}", locals.raw_ty.display(&interner));

        // self_ty should be Deque, not List
        assert!(
            matches!(&locals.self_ty, Ty::Deque(_, _)),
            "self_ty should be Deque, got: {}",
            locals.self_ty.display(&interner),
        );

        // Simulate pomollu-engine bind typecheck: re-check bind with hint = self_ty
        let full_reg = env.registry.to_full();
        let chat_spec = NodeSpec {
            name: chat_name,
            kind: NodeKind::Llm(LlmSpec {
                api: crate::ApiKind::OpenAI,
                provider: String::new(),
                model: String::new(),
                messages: vec![],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
                max_tokens: Default::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Deque {
                    bind: interner.intern("@self | extend(@raw)"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };
        let bind_reg = chat_spec.build_node_context(&interner, &full_reg, ContextScope::Bind, Some(locals)).unwrap();
        eprintln!("bind_ctx @self = {:?}", bind_reg.scoped().get(&interner.intern("self")).map(|t| format!("{}", t.display(&interner))));
        eprintln!("bind_ctx @raw = {:?}", bind_reg.scoped().get(&interner.intern("raw")).map(|t| format!("{}", t.display(&interner))));

        let hint = Some(&locals.self_ty);
        let result = compile_script_with_hint(
            &interner,
            "@self | extend(@raw)",
            &bind_reg,
            hint.as_ref().map(|t| *t),
        );
        match &result {
            Ok((_, ty)) => eprintln!("bind typecheck OK, tail = {}", ty.display(&interner)),
            Err(e) => eprintln!("bind typecheck FAILED: {}", e.display(&interner)),
        }
        assert!(result.is_ok(), "bind re-typecheck should succeed");

        // Also check initial_value
        let init_reg = chat_spec.build_node_context(&interner, &full_reg, ContextScope::InitialValue, Some(locals)).unwrap();
        let init_result = compile_script_with_hint(
            &interner,
            "[]",
            &init_reg,
            hint.as_ref().map(|t| *t),
        );
        match &init_result {
            Ok((_, ty)) => eprintln!("init typecheck OK, tail = {}", ty.display(&interner)),
            Err(e) => eprintln!("init typecheck FAILED: {}", e.display(&interner)),
        }
    }
}
