use std::collections::{BTreeMap, HashMap, HashSet};

use acvus_ast::Literal;
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::reachable_context::{
    ContextKeyPartition, partition_context_keys, reachable_context_keys,
};
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};

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
#[derive(Debug, Clone)]
pub struct CompiledSelf {
    pub self_bind: CompiledScript,
    pub initial_value: CompiledScript,
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
    pub name: String,
    pub kind: CompiledNodeKind,
    pub all_context_keys: HashSet<String>,
    pub self_spec: CompiledSelf,
    pub strategy: CompiledStrategy,
}

/// Compiled expression (Script → MIR).
#[derive(Debug, Clone)]
pub struct CompiledScript {
    pub module: MirModule,
    pub context_keys: HashSet<String>,
}

/// A compiled message entry.
#[derive(Debug, Clone)]
pub enum CompiledMessage {
    Block(CompiledBlock),
    Iterator {
        expr: CompiledScript,
        slice: Option<Vec<i64>>,
        role: Option<String>,
        token_budget: Option<TokenBudget>,
    },
}

/// A compiled message block within a node.
#[derive(Debug, Clone)]
pub struct CompiledBlock {
    pub role: String,
    pub module: MirModule,
    pub context_keys: HashSet<String>,
    pub val_def: ValDefMap,
}

impl CompiledBlock {
    /// Context keys still needed on live execution paths, given known values.
    ///
    /// Uses dead branch pruning: if a known value resolves a branch condition,
    /// context loads in the dead branch are excluded.
    pub fn required_context_keys(&self, known: &HashMap<String, Literal>) -> HashSet<String> {
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
        known: &HashMap<String, Literal>,
        resolvable: &HashSet<String>,
    ) -> HashSet<String> {
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
        storage: &S,
        resolvable: &HashSet<String>,
    ) -> HashSet<String>
    where
        S: Storage,
    {
        let known = self.known_from_storage(storage);
        self.required_context_keys(&known, resolvable)
    }

    /// Partition context keys into eager (definitely needed) and lazy
    /// (conditionally needed), excluding resolvable dependency nodes.
    pub fn partition_external_keys<S>(
        &self,
        storage: &S,
        resolvable: &HashSet<String>,
    ) -> ContextKeyPartition
    where
        S: Storage,
    {
        let known = self.known_from_storage(storage);
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

    fn known_from_storage<S>(&self, storage: &S) -> HashMap<String, Literal>
    where
        S: Storage,
    {
        self.all_context_keys
            .iter()
            .filter_map(|k| {
                let arc = storage.get(k)?;
                let lit = value_to_literal(&arc)?;
                Some((k.clone(), lit))
            })
            .collect()
    }
}

/// Compile an expression string (script syntax) with type checking.
/// Returns the compiled script and its tail expression type.
pub fn compile_script(
    source: &str,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(CompiledScript, Ty), OrchError> {
    compile_script_with_hint(source, context_types, registry, None)
}

/// Compile a script with an optional expected tail type hint for unification.
pub fn compile_script_with_hint(
    source: &str,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
    expected_tail: Option<&Ty>,
) -> Result<(CompiledScript, Ty), OrchError> {
    let script = acvus_ast::parse_script(source).map_err(|e| {
        OrchError::new(OrchErrorKind::ScriptParse {
            error: format!("{e}"),
        })
    })?;
    let (module, _hints, tail_ty) = acvus_mir::compile_script_with_hint(
        &script,
        context_types.clone(),
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
    Ok((
        CompiledScript {
            module,
            context_keys,
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
//   history store       (any)                    type inferred → @history.{node} = List<T>
//   bind script         (any)
//

/// Expect the tail type to be `List<T>`. Returns the inner `T`.
pub(crate) fn expect_list(context: &str, ty: Ty) -> Result<Ty, OrchError> {
    match ty {
        Ty::List(inner) => Ok(*inner),
        Ty::Error => Ok(Ty::Error),
        other => Err(OrchError::new(OrchErrorKind::ScriptTypeMismatch {
            context: context.into(),
            expected: "List<_>".into(),
            got: format!("{other}"),
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
            expected: format!("{expected}"),
            got: format!("{ty}"),
        }))
    }
}

/// Compile a template source string into a `CompiledBlock`.
pub(crate) fn compile_template(
    source: &str,
    block_idx: usize,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledBlock, OrchError> {
    let ast = acvus_ast::parse(source).map_err(|e| {
        OrchError::new(OrchErrorKind::TemplateParse {
            block: block_idx,
            error: format!("{e}"),
        })
    })?;

    let (module, _hints) = acvus_mir::compile(
        &ast,
        context_types.clone(),
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
        role: String::new(),
        module,
        context_keys,
        val_def,
    })
}

/// Compile messages from a message spec list.
pub(crate) fn compile_messages(
    messages: &[MessageSpec],
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
    iterator_elem_ty: &Ty,
) -> Result<(Vec<CompiledMessage>, HashSet<String>), Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, source } => {
                let block = match compile_template(source, i, context_types, registry) {
                    Ok(b) => b,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };
                all_context_keys.extend(block.context_keys.iter().cloned());
                compiled_messages.push(CompiledMessage::Block(CompiledBlock {
                    role: role.clone(),
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
                let (expr, tail_ty) = match compile_script(key, context_types, registry) {
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

                all_context_keys.extend(expr.context_keys.iter().cloned());
                compiled_messages.push(CompiledMessage::Iterator {
                    expr,
                    slice: slice.clone(),
                    role: role.clone(),
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
    spec: &NodeSpec,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
    compiled_self: CompiledSelf,
    compiled_strategy: CompiledStrategy,
) -> Result<CompiledNode, Vec<OrchError>> {
    let (kind, mut all_context_keys) = match &spec.kind {
        NodeKind::Plain(plain_spec) => {
            let (compiled, keys) = compile_plain(plain_spec, context_types, registry)?;
            (CompiledNodeKind::Plain(compiled), keys)
        }
        NodeKind::Llm(llm_spec) => {
            let (compiled, keys) = compile_llm(llm_spec, context_types, registry)?;
            (CompiledNodeKind::Llm(compiled), keys)
        }
        NodeKind::LlmCache(cache_spec) => {
            let (compiled, keys) = compile_llm_cache(cache_spec, context_types, registry)?;
            (CompiledNodeKind::LlmCache(compiled), keys)
        }
        NodeKind::Expr(expr_spec) => {
            let (compiled, keys) = compile_expr(expr_spec, context_types, registry)?;
            (CompiledNodeKind::Expr(compiled), keys)
        }
    };

    // self_spec context keys contribute to dependencies
    all_context_keys.extend(compiled_self.self_bind.context_keys.iter().cloned());
    all_context_keys.extend(compiled_self.initial_value.context_keys.iter().cloned());

    // strategy context keys contribute
    match &compiled_strategy {
        CompiledStrategy::Always | CompiledStrategy::OncePerTurn => {}
        CompiledStrategy::History { history_bind } => {
            all_context_keys.extend(history_bind.context_keys.iter().cloned());
        }
        CompiledStrategy::IfModified { key } => {
            all_context_keys.extend(key.context_keys.iter().cloned());
        }
    }

    Ok(CompiledNode {
        name: spec.name.clone(),
        kind,
        all_context_keys,
        self_spec: compiled_self,
        strategy: compiled_strategy,
    })
}

/// Compile multiple node specs, merging their output types into context automatically.
///
/// `injected_types` are externally declared context types (from project.toml).
/// Node stored types are derived from `initial_value` scripts.
pub fn compile_nodes(
    specs: &[NodeSpec],
    injected_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let mut context_types = injected_types.clone();
    let mut errors = Vec::new();

    // 1. Compile self_bind first (with @self = raw_output_ty as initial guess)
    //    → self_bind tail type determines stored type.
    //    Then compile initial_value and verify compatibility.
    let mut stored_types: Vec<Ty> = Vec::new();
    let mut bind_scripts: Vec<CompiledScript> = Vec::new();
    for spec in specs {
        let raw_ty = spec.kind.raw_output_ty();
        let mut bind_ctx = context_types.clone();
        // First pass: @self = raw_ty (best guess before stored type is known)
        bind_ctx.insert("self".into(), raw_ty.clone());
        bind_ctx.insert("raw".into(), raw_ty);
        let (script, tail_ty) = match compile_script(&spec.self_spec.self_bind, &bind_ctx, registry)
        {
            Ok(v) => v,
            Err(e) => {
                errors.push(e);
                stored_types.push(Ty::Error);
                bind_scripts.push(dummy_script());
                continue;
            }
        };
        stored_types.push(tail_ty);
        bind_scripts.push(script);
    }

    // 2. Register stored types as context types (so other nodes can reference @name)
    for (spec, ty) in specs.iter().zip(stored_types.iter()) {
        context_types.insert(spec.name.clone(), ty.clone());
    }

    // 3. Compile initial_value scripts with stored type as expected tail hint
    let mut initial_value_scripts: Vec<CompiledScript> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let hint = match &stored_types[i] {
            Ty::Error => None,
            ty => Some(ty),
        };
        let (script, init_ty) = match compile_script_with_hint(
            &spec.self_spec.initial_value,
            &context_types,
            registry,
            hint,
        ) {
            Ok(v) => v,
            Err(e) => {
                errors.push(e);
                initial_value_scripts.push(dummy_script());
                continue;
            }
        };
        if let Err(e) = expect_ty(
            &format!("node '{}' initial_value type", spec.name),
            &init_ty,
            &stored_types[i],
        ) {
            errors.push(e);
        }
        initial_value_scripts.push(script);
    }

    // 4. Compile strategy scripts + inject @history types for History nodes
    let history_specs: Vec<(usize, &str)> = specs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| match &s.strategy {
            Strategy::History { history_bind } => Some((i, history_bind.as_str())),
            _ => None,
        })
        .collect();
    if !history_specs.is_empty() {
        // Compile history_bind scripts to infer element type
        // history_bind context: @self = completed stored value, @raw = raw output, plus all context
        let store_ctx = context_types.clone();
        let mut history_fields = BTreeMap::new();
        for &(i, bind_src) in &history_specs {
            let mut hist_ctx = store_ctx.clone();
            hist_ctx.insert("self".into(), stored_types[i].clone());
            hist_ctx.insert("raw".into(), specs[i].kind.raw_output_ty());
            let ty = compile_script(bind_src, &hist_ctx, registry)
                .map(|(_, ty)| ty)
                .unwrap_or(Ty::Error);
            history_fields.insert(specs[i].name.clone(), Ty::List(Box::new(ty)));
        }
        context_types.insert("history".into(), Ty::Object(history_fields));
        context_types.insert("index".into(), Ty::Int);
    }

    // 5. Compile strategy for each node
    let mut compiled_strategies: Vec<CompiledStrategy> = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let strat = match &spec.strategy {
            Strategy::Always => CompiledStrategy::Always,
            Strategy::OncePerTurn => CompiledStrategy::OncePerTurn,
            Strategy::History { history_bind } => {
                let mut hist_ctx = context_types.clone();
                hist_ctx.insert("self".into(), stored_types[i].clone());
                hist_ctx.insert("raw".into(), spec.kind.raw_output_ty());
                let (script, _ty) = match compile_script(history_bind, &hist_ctx, registry) {
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
                let (script, _ty) = match compile_script(key, &context_types, registry) {
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
    let mut tool_param_types: HashMap<String, HashMap<String, Ty>> = HashMap::new();
    for spec in specs {
        let NodeKind::Llm(llm_spec) = &spec.kind else {
            continue;
        };
        for tool in &llm_spec.tools {
            let params: HashMap<String, Ty> = tool
                .params
                .iter()
                .filter_map(|(k, v)| Some((k.clone(), parse_type_name(v)?)))
                .collect();
            tool_param_types
                .entry(tool.node.clone())
                .or_default()
                .extend(params);
        }
    }

    let mut nodes = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let mut node_ctx = context_types.clone();
        if let Some(params) = tool_param_types.get(&spec.name) {
            node_ctx.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        let compiled_self = CompiledSelf {
            self_bind: bind_scripts[i].clone(),
            initial_value: initial_value_scripts[i].clone(),
        };
        let compiled_strategy = compiled_strategies[i].clone();
        match compile_node(spec, &node_ctx, registry, compiled_self, compiled_strategy) {
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
    let node_names: HashSet<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
    for node in &nodes {
        if let CompiledNodeKind::Llm(llm) = &node.kind {
            for tool in &llm.tools {
                if !node_names.contains(tool.node.as_str()) {
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

/// Create a dummy compiled script (used as placeholder when errors are accumulated).
fn dummy_script() -> CompiledScript {
    CompiledScript {
        module: MirModule {
            main: Default::default(),
            closures: HashMap::new(),
            tag_names: Vec::new(),
            extern_names: HashMap::new(),
        },
        context_keys: HashSet::new(),
    }
}

/// Extract all context keys referenced by `ContextLoad` instructions in a module.
fn extract_context_keys(module: &MirModule) -> HashSet<String> {
    let mut keys = HashSet::new();

    for inst in &module.main.insts {
        if let InstKind::ContextLoad { name, .. } = &inst.kind {
            keys.insert(name.clone());
        }
    }

    for closure in module.closures.values() {
        for inst in &closure.body.insts {
            if let InstKind::ContextLoad { name, .. } = &inst.kind {
                keys.insert(name.clone());
            }
        }
    }

    keys
}
