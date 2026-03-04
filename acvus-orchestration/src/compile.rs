use std::collections::{BTreeMap, HashMap, HashSet};

use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::{InstKind, MirModule};
use acvus_mir::ty::Ty;
use acvus_ast::Literal;
use acvus_mir_pass::AnalysisPass;
use acvus_mir_pass::analysis::reachable_context::{ContextKeyPartition, partition_context_keys, reachable_context_keys};
use acvus_mir_pass::analysis::val_def::{ValDefMap, ValDefMapAnalysis};

use crate::dsl::{GenerationParams, MessageSpec, NodeKind, NodeSpec, Strategy, StrategyMode, ToolBinding};
use crate::error::{OrchError, OrchErrorKind};
use crate::executor::value_to_literal;
use crate::storage::Storage;

/// Compiled node kind — mirrors `NodeKind` but with compiled messages.
#[derive(Debug, Clone)]
pub enum CompiledNodeKind {
    Plain {
        block: CompiledBlock,
    },
    Llm {
        provider: String,
        model: String,
        messages: Vec<CompiledMessage>,
        tools: Vec<CompiledToolBinding>,
        generation: GenerationParams,
        cache_key: Option<CompiledScript>,
        max_tokens: Option<u32>,
    },
    LlmCache {
        provider: String,
        model: String,
        messages: Vec<CompiledMessage>,
        ttl: String,
        cache_config: HashMap<String, serde_json::Value>,
    },
}

impl CompiledNodeKind {
    pub fn messages(&self) -> &[CompiledMessage] {
        match self {
            Self::Plain { .. } => &[],
            Self::Llm { messages, .. } | Self::LlmCache { messages, .. } => messages,
        }
    }
}

/// A compiled tool binding with resolved types.
#[derive(Debug, Clone)]
pub struct CompiledToolBinding {
    pub name: String,
    pub description: String,
    pub node: String,
    pub params: HashMap<String, Ty>,
}

/// Compiled history specification for a node.
#[derive(Debug, Clone)]
pub struct CompiledHistory {
    pub store: CompiledScript,
}

/// A compiled orchestration node.
#[derive(Debug, Clone)]
pub struct CompiledNode {
    pub name: String,
    pub kind: CompiledNodeKind,
    pub all_context_keys: HashSet<String>,
    pub strategy: Strategy,
    pub bind_module: Option<CompiledScript>,
    pub history: Option<CompiledHistory>,
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
        block: Option<CompiledBlock>,
        slice: Option<Vec<i64>>,
        bind: Option<String>,
        role: Option<String>,
        token_budget: Option<crate::dsl::TokenBudget>,
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
            match msg {
                CompiledMessage::Block(block) => {
                    needed.extend(block.required_context_keys(known));
                }
                CompiledMessage::Iterator { block: Some(block), .. } => {
                    needed.extend(block.required_context_keys(known));
                }
                _ => {}
            }
        }
        if let Some(bind_script) = &self.bind_module {
            needed.extend(bind_script.context_keys.iter().cloned());
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
            match msg {
                CompiledMessage::Block(block) => {
                    let p = partition_context_keys(&block.module, &known, &block.val_def);
                    merged.eager.extend(p.eager);
                    merged.lazy.extend(p.lazy);
                }
                CompiledMessage::Iterator { block: Some(block), .. } => {
                    let p = partition_context_keys(&block.module, &known, &block.val_def);
                    merged.eager.extend(p.eager);
                    merged.lazy.extend(p.lazy);
                }
                _ => {}
            }
        }
        if let Some(bind_script) = &self.bind_module {
            // Script has no branch pruning — all context keys are eager.
            merged.eager.extend(bind_script.context_keys.iter().cloned());
        }
        merged.eager.retain(|k| !resolvable.contains(k));
        merged.lazy.retain(|k| !resolvable.contains(k) && !merged.eager.contains(k));
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
    let script = acvus_ast::parse_script(source).map_err(|e| {
        OrchError::new(OrchErrorKind::ScriptParse {
            error: format!("{e}"),
        })
    })?;
    let (module, _hints, tail_ty) =
        acvus_mir::compile_script(&script, context_types.clone(), registry).map_err(|errs| {
            OrchError::new(OrchErrorKind::ScriptCompile {
                context: source.to_string(),
                errors: errs,
            })
        })?;
    let context_keys = extract_context_keys(&module);
    Ok((CompiledScript { module, context_keys }, tail_ty))
}

// ── Script output type expectations ──────────────────────────────────
//
//   Field          Expected type   Notes
//   ─────────────  ──────────────  ──────────────────────────────────
//   iterator       List<T>         T becomes the element type for body
//   cache_key      String
//   history store  (any)           type inferred → @history.{node} = List<T>
//   bind script    (any)
//

/// Expect the tail type to be `List<T>`. Returns the inner `T`.
fn expect_list(
    context: &str,
    ty: Ty,
) -> Result<Ty, OrchError> {
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
fn expect_ty(
    context: &str,
    ty: &Ty,
    expected: &Ty,
) -> Result<(), OrchError> {
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
pub fn compile_template(
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

    let (module, _hints) = acvus_mir::compile(&ast, context_types.clone(), registry).map_err(
        |errs| {
            OrchError::new(OrchErrorKind::TemplateCompile {
                block: block_idx,
                errors: errs,
            })
        },
    )?;

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
fn compile_messages(
    messages: &[MessageSpec],
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<(Vec<CompiledMessage>, HashSet<String>), Vec<OrchError>> {
    let mut compiled_messages = Vec::new();
    let mut all_context_keys = HashSet::new();
    let mut errors = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            MessageSpec::Block { role, source } => {
                match compile_template(source, i, context_types, registry) {
                    Ok(block) => {
                        all_context_keys.extend(block.context_keys.iter().cloned());
                        compiled_messages.push(CompiledMessage::Block(CompiledBlock {
                            role: role.clone(),
                            ..block
                        }));
                    }
                    Err(e) => errors.push(e),
                }
            }
            MessageSpec::Iterator { key, source, slice, bind, role, token_budget } => {
                let (expr, elem_ty) = match compile_script(key, context_types, registry) {
                    Ok((script, tail_ty)) => {
                        match expect_list(&format!("iterator (block {i})"), tail_ty) {
                            Ok(inner) => (script, inner),
                            Err(e) => {
                                errors.push(e);
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                };

                let block = if let Some(src) = source {
                    let mut iter_types = context_types.clone();
                    if let Some(bind_name) = bind {
                        iter_types.insert(bind_name.clone(), elem_ty.clone());
                    } else {
                        // destructure element fields into context
                        if let Ty::Object(fields) = &elem_ty {
                            for (k, v) in fields {
                                iter_types.insert(k.clone(), v.clone());
                            }
                        }
                    }

                    match compile_template(src, i, &iter_types, registry) {
                        Ok(block) => Some(block),
                        Err(e) => {
                            errors.push(e);
                            None
                        }
                    }
                } else {
                    None
                };

                all_context_keys.extend(expr.context_keys.iter().cloned());
                compiled_messages.push(CompiledMessage::Iterator {
                    expr,
                    block,
                    slice: slice.clone(),
                    bind: bind.clone(),
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

/// Compile tool bindings, converting param type name strings to `Ty`.
fn compile_tool_bindings(tools: &[ToolBinding]) -> Result<Vec<CompiledToolBinding>, Vec<OrchError>> {
    let mut compiled = Vec::new();
    let mut errors = Vec::new();

    for tool in tools {
        let mut params = HashMap::new();
        for (param_name, type_name) in &tool.params {
            match parse_type_name(type_name) {
                Some(ty) => { params.insert(param_name.clone(), ty); }
                None => {
                    errors.push(OrchError::new(OrchErrorKind::ToolParamType {
                        tool: tool.name.clone(),
                        param: param_name.clone(),
                        type_name: type_name.clone(),
                    }));
                }
            }
        }
        compiled.push(CompiledToolBinding {
            name: tool.name.clone(),
            description: tool.description.clone(),
            node: tool.node.clone(),
            params,
        });
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(compiled)
}

/// Parse a type name string into a `Ty`.
fn parse_type_name(name: &str) -> Option<Ty> {
    match name {
        "string" => Some(Ty::String),
        "int" => Some(Ty::Int),
        "float" => Some(Ty::Float),
        "bool" => Some(Ty::Bool),
        _ => None,
    }
}

/// Compile a node spec into a `CompiledNode`.
///
/// Each message's `source` field is compiled directly — no file I/O.
pub fn compile_node(
    spec: &NodeSpec,
    context_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<CompiledNode, Vec<OrchError>> {
    // Compile bind script for if-modified strategy
    let bind_module = if matches!(spec.strategy.mode, StrategyMode::IfModified) {
        if let Some(bind_src) = &spec.strategy.bind_source {
            let (script, _ty) = compile_script(bind_src, context_types, registry)
                .map_err(|e| vec![e])?;
            Some(script)
        } else {
            None
        }
    } else {
        None
    };

    let (kind, mut all_context_keys) = match &spec.kind {
        NodeKind::Plain { source } => {
            match compile_template(source, 0, context_types, registry) {
                Ok(block) => {
                    let keys = block.context_keys.clone();
                    (CompiledNodeKind::Plain { block }, keys)
                }
                Err(e) => return Err(vec![e]),
            }
        }
        NodeKind::Llm { provider, model, messages, tools, generation, cache_key, max_tokens } => {
            let (compiled_messages, keys) = compile_messages(messages, context_types, registry)?;
            let compiled_tools = compile_tool_bindings(tools)?;
            let mut all_keys = keys;
            let compiled_cache_key = match cache_key {
                Some(ck) => {
                    let (expr, ck_ty) = compile_script(ck, context_types, registry)
                        .map_err(|e| vec![e])?;
                    expect_ty("cache_key", &ck_ty, &Ty::String)
                        .map_err(|e| vec![e])?;
                    all_keys.extend(expr.context_keys.iter().cloned());
                    Some(expr)
                }
                None => None,
            };
            (
                CompiledNodeKind::Llm {
                    provider: provider.clone(),
                    model: model.clone(),
                    messages: compiled_messages,
                    tools: compiled_tools,
                    generation: generation.clone(),
                    cache_key: compiled_cache_key,
                    max_tokens: *max_tokens,
                },
                all_keys,
            )
        }
        NodeKind::LlmCache { provider, model, messages, ttl, cache_config } => {
            let (compiled_messages, keys) = compile_messages(messages, context_types, registry)?;
            (
                CompiledNodeKind::LlmCache {
                    provider: provider.clone(),
                    model: model.clone(),
                    messages: compiled_messages,
                    ttl: ttl.clone(),
                    cache_config: cache_config.clone(),
                },
                keys,
            )
        }
    };

    // bind_module context keys also contribute
    if let Some(bind_script) = &bind_module {
        all_context_keys.extend(bind_script.context_keys.iter().cloned());
    }

    // Compile history store script with type checking.
    let history = match &spec.history {
        Some(hs) => {
            let (store, _ty) = compile_script(&hs.store, context_types, registry)
                .map_err(|e| vec![e])?;
            all_context_keys.extend(store.context_keys.iter().cloned());
            Some(CompiledHistory { store })
        }
        None => None,
    };

    Ok(CompiledNode {
        name: spec.name.clone(),
        kind,
        all_context_keys,
        strategy: spec.strategy.clone(),
        bind_module,
        history,
    })
}

/// Compile multiple node specs, merging their output types into context automatically.
///
/// `injected_types` are externally declared context types (from project.toml).
/// Each node name is also added as `Ty::String` context.
pub fn compile_nodes(
    specs: &[NodeSpec],
    injected_types: &HashMap<String, Ty>,
    registry: &ExternRegistry,
) -> Result<Vec<CompiledNode>, Vec<OrchError>> {
    let mut context_types = injected_types.clone();
    for spec in specs {
        context_types.insert(spec.name.clone(), spec.kind.output_ty());
    }

    // Inject @history type for nodes with history.
    // First compile each store expression with type checking to infer its element type.
    // @history.{node} = List<store_type>.
    let history_specs: Vec<(&str, &str)> = specs
        .iter()
        .filter_map(|s| s.history.as_ref().map(|h| (s.name.as_str(), h.store.as_str())))
        .collect();
    if !history_specs.is_empty() {
        // Build a temporary context_types for store type inference.
        // Store expressions can reference node outputs (@chat, @input, etc.) but not @history itself.
        let store_ctx = context_types.clone();
        let mut history_fields = BTreeMap::new();
        for &(name, store_src) in &history_specs {
            match compile_script(store_src, &store_ctx, registry) {
                Ok((_script, ty)) => {
                    history_fields.insert(name.to_string(), Ty::List(Box::new(ty)));
                }
                Err(_) => {
                    // Fallback: type check failed, use Error type (unifies with anything)
                    history_fields.insert(name.to_string(), Ty::List(Box::new(Ty::Error)));
                }
            }
        }
        context_types.insert("history".into(), Ty::Object(history_fields));
        context_types.insert("index".into(), Ty::Int);
    }

    // Collect tool param types: target node name → param name → Ty.
    // When a tool targets a node, the tool's params become context types
    // available to that node at compile time.
    let mut tool_param_types: HashMap<String, HashMap<String, Ty>> = HashMap::new();
    for spec in specs {
        if let NodeKind::Llm { tools, .. } = &spec.kind {
            for tool in tools {
                let mut params = HashMap::new();
                for (param_name, type_name) in &tool.params {
                    if let Some(ty) = parse_type_name(type_name) {
                        params.insert(param_name.clone(), ty);
                    }
                }
                tool_param_types
                    .entry(tool.node.clone())
                    .or_default()
                    .extend(params);
            }
        }
    }

    let mut nodes = Vec::new();
    let mut errors = Vec::new();
    for spec in specs {
        let mut node_ctx = context_types.clone();
        if let Some(params) = tool_param_types.get(&spec.name) {
            node_ctx.extend(params.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        match compile_node(spec, &node_ctx, registry) {
            Ok(node) => nodes.push(node),
            Err(errs) => errors.extend(errs),
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Tool target validation
    let node_names: HashSet<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
    for node in &nodes {
        if let CompiledNodeKind::Llm { tools, .. } = &node.kind {
            for tool in tools {
                if !node_names.contains(tool.node.as_str()) {
                    errors.push(OrchError::new(OrchErrorKind::ToolTargetNotFound {
                        tool: tool.name.clone(),
                        target: tool.node.clone(),
                    }));
                }
            }
        }
    }

    if errors.is_empty() { Ok(nodes) } else { Err(errors) }
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
