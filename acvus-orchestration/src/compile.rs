use std::collections::HashSet;

use acvus_ast::Literal;
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::reachable_context::{
    ContextKeyPartition, partition_context_keys, reachable_context_keys,
};
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::TokenBudget;
use crate::convert::value_to_literal;
use crate::dsl::{MessageSpec, NodeSpec, Strategy};
use crate::error::{OrchError, OrchErrorKind};
use crate::kind::{
    CompiledNodeKind, NodeKind, compile_expr, compile_llm, compile_llm_cache, compile_plain,
    parse_type_name,
};
use crate::storage::Storage;

/// Compiled self specification for a node.
///
/// Stored value = raw output (identity). No bind transformation.
/// When `initial_value` is `Some`, `@self` is available in the node body.
#[derive(Debug, Clone)]
pub struct CompiledSelf {
    pub initial_value: Option<CompiledScript>,
}

/// Compiled execution strategy.
#[derive(Debug, Clone)]
pub enum CompiledStrategy {
    Always,
    OncePerTurn,
    IfModified { key: CompiledScript },
    History { history_bind: CompiledScript },
}

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub name: Astr,
    pub kind: CompiledNodeKind,
    pub all_context_keys: HashSet<Astr>,
    pub self_spec: CompiledSelf,
    pub strategy: CompiledStrategy,
    pub retry: u32,
    pub assert: Option<CompiledScript>,
}

/// Compiled expression (Script → MIR).
#[derive(Debug, Clone)]
pub struct CompiledScript {
    pub module: MirModule,
    pub context_keys: HashSet<Astr>,
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
    pub context_keys: HashSet<Astr>,
    pub val_def: ValDefMap,
}

impl CompiledBlock {
    /// Context keys still needed on live execution paths, given known values.
    ///
    /// Uses dead branch pruning: if a known value resolves a branch condition,
    /// context loads in the dead branch are excluded.
    pub fn required_context_keys(&self, known: &FxHashMap<Astr, Literal>) -> HashSet<Astr> {
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
        known: &FxHashMap<Astr, Literal>,
        resolvable: &HashSet<Astr>,
    ) -> HashSet<Astr> {
        let mut needed = HashSet::new();
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
    pub fn required_external_keys<S>(
        &self,
        interner: &Interner,
        storage: &S,
        resolvable: &HashSet<Astr>,
    ) -> HashSet<Astr>
    where
        S: Storage,
    {
        let known = self.known_from_storage(interner, storage);
        self.required_context_keys(&known, resolvable)
    }

    /// Partition context keys into eager (definitely needed) and lazy
    /// (conditionally needed), excluding resolvable dependency nodes.
    pub fn partition_external_keys<S>(
        &self,
        interner: &Interner,
        storage: &S,
        resolvable: &HashSet<Astr>,
    ) -> ContextKeyPartition
    where
        S: Storage,
    {
        let known = self.known_from_storage(interner, storage);
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

    pub(crate) fn known_from_storage<S>(
        &self,
        interner: &Interner,
        storage: &S,
    ) -> FxHashMap<Astr, Literal>
    where
        S: Storage,
    {
        self.all_context_keys
            .iter()
            .filter_map(|k| {
                let arc = storage.get(interner.resolve(*k))?;
                let lit = value_to_literal(&arc)?;
                Some((*k, lit))
            })
            .collect()
    }
}

/// Compile an expression string (script syntax) with type checking.
/// Returns the compiled script and its tail expression type.
pub fn compile_script(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledScript, Ty), OrchError> {
    compile_script_with_hint(interner, source, context_types, registry, None)
}

/// Compile a script with an optional expected tail type hint for unification.
pub fn compile_script_with_hint(
    interner: &Interner,
    source: &str,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
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
        context_types,
        registry,
        &acvus_mir::user_type::UserTypeRegistry::new(),
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
//   history store       (any)                    type inferred → @turn.history.{node} = List<T>
//   bind script         (any)
//

/// Expect the tail type to be `List<T>`. Returns the inner `T`.
pub(crate) fn expect_list(context: &str, ty: Ty) -> Result<Ty, OrchError> {
    match ty {
        Ty::List(inner) => Ok(*inner),
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
    if matches!(ty, Ty::Error) || ty == expected {
        Ok(())
    } else {
        Err(OrchError::new(OrchErrorKind::ScriptTypeMismatch {
            context: context.into(),
            expected: expected.clone(),
            got: ty.clone(),
        }))
    }
}

/// Compile a template source string into a `CompiledBlock`.
pub(crate) fn compile_template(
    interner: &Interner,
    source: &str,
    block_idx: usize,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
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
        context_types,
        registry,
        &acvus_mir::user_type::UserTypeRegistry::new(),
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
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
    iterator_elem_ty: &Ty,
) -> Result<(Vec<CompiledMessage>, HashSet<Astr>), Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, source } => {
                let block = match compile_template(interner, source, i, context_types, registry) {
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
                    match compile_script(interner, interner.resolve(*key), context_types, registry)
                    {
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
/// `stored_ty` is the node's stored type (derived from initial_value in compile_nodes).
/// Each message's `source` field is compiled directly — no file I/O.
pub fn compile_node(
    interner: &Interner,
    spec: &NodeSpec,
    context_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
    compiled_self: CompiledSelf,
    compiled_strategy: CompiledStrategy,
    stored_ty: &Ty,
) -> Result<CompiledNode, Vec<OrchError>> {
    let (kind, mut all_context_keys) = match &spec.kind {
        NodeKind::Plain(plain_spec) => {
            let (compiled, keys) = compile_plain(interner, plain_spec, context_types, registry)?;
            (CompiledNodeKind::Plain(compiled), keys)
        }
        NodeKind::Llm(llm_spec) => {
            let (compiled, keys) = compile_llm(interner, llm_spec, context_types, registry)?;
            (CompiledNodeKind::Llm(compiled), keys)
        }
        NodeKind::LlmCache(cache_spec) => {
            let (compiled, keys) =
                compile_llm_cache(interner, cache_spec, context_types, registry)?;
            (CompiledNodeKind::LlmCache(compiled), keys)
        }
        NodeKind::Expr(expr_spec) => {
            let (compiled, keys) = compile_expr(interner, expr_spec, context_types, registry)?;
            (CompiledNodeKind::Expr(compiled), keys)
        }
    };

    // self_spec context keys contribute to dependencies
    if let Some(ref iv) = compiled_self.initial_value {
        all_context_keys.extend(iv.context_keys.iter().copied());
    }

    // assert context keys contribute
    let compiled_assert = if let Some(ref assert_src) = spec.assert {
        // assert context: @self = stored value (= raw output), plus all context
        let mut assert_ctx = context_types.clone();
        assert_ctx.insert(interner.intern("self"), stored_ty.clone());
        let (script, _ty) = compile_script_with_hint(
            interner,
            interner.resolve(*assert_src),
            &assert_ctx,
            registry,
            Some(&Ty::Bool),
        )
        .map_err(|e| vec![e])?;
        all_context_keys.extend(script.context_keys.iter().copied());
        Some(script)
    } else {
        None
    };

    // strategy context keys contribute
    match &compiled_strategy {
        CompiledStrategy::Always | CompiledStrategy::OncePerTurn => {}
        CompiledStrategy::History { history_bind } => {
            all_context_keys.extend(history_bind.context_keys.iter().copied());
        }
        CompiledStrategy::IfModified { key } => {
            all_context_keys.extend(key.context_keys.iter().copied());
        }
    }

    Ok(CompiledNode {
        name: spec.name,
        kind,
        all_context_keys,
        self_spec: compiled_self,
        strategy: compiled_strategy,
        retry: spec.retry,
        assert: compiled_assert,
    })
}

/// Result of computing external context types from node specs.
///
/// `context_types` contains all externally-visible types:
/// - injected types (from project.toml / bindings)
/// - `@nodeName` → stored type (from self_bind tail)
/// - `@turn` → computed from history nodes
///
/// Local types (`@self`, `@raw`) are NOT included.
/// Per-node local types visible inside the node (e.g. @raw, @self).
#[derive(Debug, Clone)]
pub struct NodeLocalTypes {
    pub raw_ty: Ty,
    pub self_ty: Ty,
}

pub struct ExternalContextEnv {
    pub context_types: FxHashMap<Astr, Ty>,
    /// Types of values stored in storage (node self types + @turn).
    /// Does not include injected types (those come from the resolver, not storage).
    pub storage_types: FxHashMap<Astr, Ty>,
    /// Per-node local types, indexed by node name.
    pub node_locals: FxHashMap<Astr, NodeLocalTypes>,
    pub(crate) stored_types: Vec<Ty>,
}

/// Compute all externally-visible context types from node specs.
///
/// This performs steps 1-3 of the full compilation pipeline:
/// 1. Determine stored types (= raw_output_ty, identity)
/// 2. Register stored types as context (`@nodeName`)
/// 3. Compile history_bind → compute `@turn` type
///
/// The result can be used for typechecking binding scripts or passed
/// to `compile_nodes_with_env` for full compilation.
pub fn compute_external_context_env(
    interner: &Interner,
    specs: &[NodeSpec],
    injected_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<ExternalContextEnv, Vec<OrchError>> {
    let mut context_types = injected_types.clone();

    // 1. stored type = raw_output_ty for concrete nodes (LLM, Plain, LlmCache)
    //    Expr nodes start as Ty::Infer — resolved in step 3 after @turn is available.
    let mut stored_types: Vec<Ty> = specs
        .iter()
        .map(|s| s.kind.raw_output_ty(interner))
        .collect();

    // 2. Register concrete (non-Infer) stored types so other nodes can reference @name
    for (spec, ty) in specs.iter().zip(stored_types.iter()) {
        if *ty != Ty::Infer {
            context_types.insert(spec.name, ty.clone());
        }
    }

    // 3. Compile history_bind → compute @turn type for History nodes
    let history_specs: Vec<(usize, &str)> = specs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.strategy {
            Strategy::History { history_bind } => Some((i, interner.resolve(*history_bind))),
            _ => None,
        })
        .collect();
    if !history_specs.is_empty() {
        let store_ctx = context_types.clone();
        let mut entry_fields = FxHashMap::default();
        for &(i, bind_src) in &history_specs {
            let mut hist_ctx = store_ctx.clone();
            hist_ctx.insert(interner.intern("self"), stored_types[i].clone());
            let ty = compile_script(interner, bind_src, &hist_ctx, registry)
                .map(|(_, ty)| ty)
                .unwrap_or(Ty::Error);
            entry_fields.insert(specs[i].name, ty);
        }
        let history_ty = Ty::List(Box::new(Ty::Object(entry_fields)));
        let turn_fields = FxHashMap::from_iter([
            (interner.intern("index"), Ty::Int),
            (interner.intern("history"), history_ty),
        ]);
        context_types.insert(interner.intern("turn"), Ty::Object(turn_fields));
    }

    // 4. Resolve Expr node types: compile source with full context (including @turn)
    for (i, spec) in specs.iter().enumerate() {
        if stored_types[i] != Ty::Infer {
            continue;
        }
        if let NodeKind::Expr(expr_spec) = &spec.kind {
            stored_types[i] = compile_script(interner, &expr_spec.source, &context_types, registry)
                .map(|(_, ty)| ty)
                .unwrap_or(Ty::Error);
        }
        context_types.insert(spec.name, stored_types[i].clone());
    }

    let mut node_locals = FxHashMap::default();
    for (spec, stored_ty) in specs.iter().zip(stored_types.iter()) {
        node_locals.insert(
            spec.name,
            NodeLocalTypes {
                raw_ty: spec.kind.raw_output_ty(interner),
                self_ty: stored_ty.clone(),
            },
        );
    }

    // storage_types = context_types minus injected_types
    let storage_types: FxHashMap<Astr, Ty> = context_types
        .iter()
        .filter(|(k, _)| !injected_types.contains_key(k))
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    Ok(ExternalContextEnv {
        context_types,
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
    injected_types: &FxHashMap<Astr, Ty>,
    registry: &ExternRegistry,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let env = compute_external_context_env(interner, specs, injected_types, registry)?;
    compile_nodes_with_env(interner, specs, registry, env)
}

/// Compile nodes using a pre-computed external context environment.
///
/// Use this when you already called `compute_external_context_env` (e.g. for
/// typechecking binding scripts) and want to avoid recomputation.
pub fn compile_nodes_with_env(
    interner: &Interner,
    specs: &[NodeSpec],
    registry: &ExternRegistry,
    env: ExternalContextEnv,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let context_types = env.context_types;
    let stored_types = env.stored_types;
    let mut errors = Vec::new();

    // Compile initial_value scripts (only when Some) with stored type as expected tail hint
    let mut initial_value_scripts: Vec<Option<CompiledScript>> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let Some(ref init_src) = spec.self_spec.initial_value else {
            initial_value_scripts.push(None);
            continue;
        };
        let hint = match &stored_types[i] {
            Ty::Error => None,
            ty => Some(ty),
        };
        let (script, init_ty) = match compile_script_with_hint(
            interner,
            interner.resolve(*init_src),
            &context_types,
            registry,
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

    // Compile strategy for each node
    let mut compiled_strategies: Vec<CompiledStrategy> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let strat = match &spec.strategy {
            Strategy::Always => CompiledStrategy::Always,
            Strategy::OncePerTurn => CompiledStrategy::OncePerTurn,
            Strategy::History { history_bind } => {
                let mut hist_ctx = context_types.clone();
                hist_ctx.insert(interner.intern("self"), stored_types[i].clone());
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*history_bind),
                    &hist_ctx,
                    registry,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_strategies.push(CompiledStrategy::Always);
                        continue;
                    }
                };
                CompiledStrategy::History {
                    history_bind: script,
                }
            }
            Strategy::IfModified { key } => {
                let (script, _ty) = match compile_script(
                    interner,
                    interner.resolve(*key),
                    &context_types,
                    registry,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        errors.push(e);
                        compiled_strategies.push(CompiledStrategy::Always);
                        continue;
                    }
                };
                CompiledStrategy::IfModified { key: script }
            }
        };
        compiled_strategies.push(strat);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool param types must be injected from the caller (LlmSpec) side because
    // the target node alone cannot know what types its params will have — the
    // param types are declared in ToolBinding, not in the target node itself.
    let mut tool_param_types: FxHashMap<Astr, FxHashMap<Astr, Ty>> = FxHashMap::default();
    for spec in specs {
        let NodeKind::Llm(llm_spec) = &spec.kind else {
            continue;
        };
        for tool in &llm_spec.tools {
            let params: FxHashMap<Astr, Ty> = tool
                .params
                .iter()
                .filter_map(|(k, v)| Some((interner.intern(k), parse_type_name(v)?)))
                .collect();
            tool_param_types
                .entry(interner.intern(&tool.node))
                .or_default()
                .extend(params);
        }
    }

    let mut nodes = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let mut node_ctx = context_types.clone();
        if let Some(params) = tool_param_types.get(&spec.name) {
            node_ctx.extend(params.iter().map(|(k, v)| (*k, v.clone())));
        }
        // When initial_value is Some, @self is available in the node body
        if initial_value_scripts[i].is_some() {
            node_ctx.insert(interner.intern("self"), stored_types[i].clone());
        }
        let compiled_self = CompiledSelf {
            initial_value: initial_value_scripts[i].clone(),
        };
        let compiled_strategy = compiled_strategies[i].clone();
        match compile_node(
            interner,
            spec,
            &node_ctx,
            registry,
            compiled_self,
            compiled_strategy,
            &stored_types[i],
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
    let node_names: HashSet<Astr> = nodes.iter().map(|n| n.name).collect();
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

/// Extract all context keys referenced by `ContextLoad` instructions in a module.
fn extract_context_keys(module: &MirModule) -> HashSet<Astr> {
    let mut keys = HashSet::new();

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
