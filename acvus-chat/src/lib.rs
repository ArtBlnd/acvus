use std::collections::HashMap;

use acvus_interpreter::{ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value};
use acvus_orchestration::{
    build_cache_request, build_request, output_to_value, parse_cache_response, parse_response,
    CompiledBlock, CompiledMessage, CompiledNode, Fetch, HashMapStorage, Message, ModelResponse,
    NodeKind, Output, ProviderConfig, Storage, StrategyMode, ToolSpec,
};
use futures::future::BoxFuture;

// ---------------------------------------------------------------------------
// Internal: coroutine-driven rendering with demand-driven dependency resolution
// ---------------------------------------------------------------------------

struct RenderCtx<'a, F> {
    nodes: &'a [CompiledNode],
    name_to_idx: &'a HashMap<String, usize>,
    providers: &'a HashMap<String, ProviderConfig>,
    fetch: &'a F,
    extern_fns: &'a ExternFnRegistry,
}

enum BlockDriveResult {
    Done(String),
    NeedContext(NeedContextStepped),
}

fn drive_block(
    coroutine: &mut acvus_interpreter::Coroutine,
    mut key: ResumeKey,
    output: &mut String,
    storage: &HashMapStorage,
    local: &HashMap<String, Value>,
) -> BlockDriveResult {
    loop {
        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => output.push_str(&s),
                    other => panic!("expected String, got {other:?}"),
                }
                key = next_key;
            }
            Stepped::NeedContext(need) => {
                if !need.bindings().is_empty() {
                    return BlockDriveResult::NeedContext(need);
                }
                let name = need.name().to_string();
                if let Some(value) = local.get(&name) {
                    key = need.into_key(value.clone());
                } else if let Some(out) = storage.get(&name) {
                    key = need.into_key(output_to_value(&out));
                } else {
                    return BlockDriveResult::NeedContext(need);
                }
            }
            Stepped::Done => {
                return BlockDriveResult::Done(std::mem::take(output));
            }
        }
    }
}

fn render_with_deps<'a, F>(
    block: &'a CompiledBlock,
    storage: &'a mut HashMapStorage,
    local: HashMap<String, Value>,
    ctx: &'a RenderCtx<'a, F>,
    key_cache: &'a mut HashMap<String, String>,
) -> BoxFuture<'a, String>
where
    F: Fetch,
{
    Box::pin(async move {
        let interp = Interpreter::new(block.module.clone(), ctx.extern_fns.clone());
        let (mut coroutine, key) = interp.execute();
        let mut output = String::new();

        let mut result = drive_block(&mut coroutine, key, &mut output, storage, &local);
        loop {
            match result {
                BlockDriveResult::Done(text) => return text,
                BlockDriveResult::NeedContext(need) => {
                    let name = need.name().to_string();
                    let bindings = need.bindings().clone();
                    if !bindings.is_empty() {
                        if let Some(&idx) = ctx.name_to_idx.get(&name) {
                            resolve_node(idx, storage, ctx, bindings, key_cache).await;
                        }
                        let value = storage
                            .get(&name)
                            .map(|o| output_to_value(&o))
                            .unwrap_or_else(|| panic!("unresolved context: @{name}"));
                        let key = need.into_key(value);
                        result = drive_block(&mut coroutine, key, &mut output, storage, &local);
                    } else {
                        if let Some(&idx) = ctx.name_to_idx.get(&name) {
                            if storage.get(&name).is_none() {
                                resolve_node(idx, storage, ctx, HashMap::new(), key_cache).await;
                            }
                        }
                        let value = storage
                            .get(&name)
                            .map(|o| output_to_value(&o))
                            .unwrap_or_else(|| panic!("unresolved context: @{name}"));
                        let key = need.into_key(value);
                        result = drive_block(&mut coroutine, key, &mut output, storage, &local);
                    }
                }
            }
        }
    })
}

fn resolve_node<'a, F>(
    idx: usize,
    storage: &'a mut HashMapStorage,
    ctx: &'a RenderCtx<'a, F>,
    local: HashMap<String, Value>,
    key_cache: &'a mut HashMap<String, String>,
) -> BoxFuture<'a, ()>
where
    F: Fetch,
{
    Box::pin(async move {
        let node = &ctx.nodes[idx];

        if matches!(node.strategy.mode, StrategyMode::IfModified) {
            if let Some(key_block) = &node.key_module {
                let current_key =
                    render_with_deps(key_block, storage, local.clone(), ctx, key_cache).await;
                if key_cache.get(&node.name).map(|k| k == &current_key).unwrap_or(false) {
                    return;
                }
                key_cache.insert(node.name.clone(), current_key);
            }
        }

        let mut messages = Vec::new();
        for msg in &node.messages {
            let block = match msg {
                CompiledMessage::Block(block) => block,
                CompiledMessage::Iterator { .. } => continue,
            };
            let text = render_with_deps(block, storage, local.clone(), ctx, key_cache).await;
            messages.push(Message::text(&block.role, text));
        }

        let provider_config = ctx
            .providers
            .get(&node.provider)
            .unwrap_or_else(|| panic!("unknown provider: {}", node.provider))
            .clone();

        match node.kind {
            NodeKind::LlmCache { ref ttl, ref cache_config } => {
                let request = build_cache_request(
                    &provider_config, &node.model, &messages, ttl, cache_config,
                );
                let json = ctx
                    .fetch
                    .fetch(&request)
                    .await
                    .unwrap_or_else(|e| panic!("cache creation error for node {}: {e}", node.name));
                let cache_name = parse_cache_response(&provider_config.api, &json)
                    .unwrap_or_else(|e| panic!("cache parse error for node {}: {e}", node.name));
                storage.set(node.name.clone(), Output::Text(cache_name));
            }
            NodeKind::Llm => {
                let tools: Vec<ToolSpec> = node
                    .tools
                    .iter()
                    .map(|t| ToolSpec {
                        name: t.name.clone(),
                        description: String::new(),
                        params: t.params.clone(),
                    })
                    .collect();

                let request = build_request(
                    &provider_config, &node.model, &messages, &tools, &node.generation, None,
                );
                let json = ctx
                    .fetch
                    .fetch(&request)
                    .await
                    .unwrap_or_else(|e| panic!("fetch error for node {}: {e}", node.name));
                let response = parse_response(&provider_config.api, &json)
                    .unwrap_or_else(|e| panic!("parse error for node {}: {e}", node.name));

                match response {
                    ModelResponse::Text(text) => {
                        storage.set(node.name.clone(), Output::Text(text));
                    }
                    ModelResponse::ToolCalls(_) => {
                        panic!("tool calls in dependency node {} not supported", node.name);
                    }
                }
            }
        }
    })
}

fn resolve_index(idx: i64, len: usize) -> usize {
    if idx < 0 {
        (len as i64 + idx).max(0) as usize
    } else {
        (idx as usize).min(len)
    }
}

async fn expand_iterator<F>(
    key: &str,
    block: Option<&CompiledBlock>,
    slice: &Option<Vec<i64>>,
    bind: &Option<String>,
    role_override: &Option<String>,
    storage: &mut HashMapStorage,
    ctx: &RenderCtx<'_, F>,
    key_cache: &mut HashMap<String, String>,
) -> Vec<Message>
where
    F: Fetch,
{
    let all_items = match storage.get(key) {
        Some(Output::Json(serde_json::Value::Array(arr))) => arr,
        _ => return Vec::new(),
    };

    let items: &[serde_json::Value] = if let Some(s) = slice {
        let len = all_items.len();
        match s.as_slice() {
            [start] => &all_items[resolve_index(*start, len)..],
            [start, end] => &all_items[resolve_index(*start, len)..resolve_index(*end, len)],
            _ => &all_items,
        }
    } else {
        &all_items
    };

    let mut messages = Vec::new();
    for item in items {
        let role = role_override
            .as_deref()
            .unwrap_or_else(|| item["type"].as_str().unwrap_or("user"));
        let text = item["text"].as_str().unwrap_or("");

        if let Some(block) = block {
            let local = if let Some(bind_name) = bind {
                HashMap::from([(
                    bind_name.clone(),
                    output_to_value(&Output::Json(item.clone())),
                )])
            } else {
                HashMap::from([
                    ("type".into(), Value::String(role.to_string())),
                    ("text".into(), Value::String(text.to_string())),
                ])
            };
            let rendered = render_with_deps(block, storage, local, ctx, key_cache).await;
            messages.push(Message::text(role, rendered));
        } else {
            messages.push(Message::text(role, text));
        }
    }
    messages
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ChatEngine<F> {
    nodes: Vec<CompiledNode>,
    name_to_idx: HashMap<String, usize>,
    providers: HashMap<String, ProviderConfig>,
    fetch: F,
    extern_fns: ExternFnRegistry,
    storage: HashMapStorage,
    key_cache: HashMap<String, String>,
    static_messages: Vec<Message>,
    first_iter_idx: usize,
    per_turn_keys: Vec<String>,
    iterator_keys: Vec<String>,
    provider_config: ProviderConfig,
    tools: Vec<ToolSpec>,
    output_module: Option<CompiledBlock>,
}

impl<F> ChatEngine<F>
where
    F: Fetch,
{
    pub async fn new(
        nodes: Vec<CompiledNode>,
        providers: HashMap<String, ProviderConfig>,
        fetch: F,
        extern_fns: ExternFnRegistry,
        mut storage: HashMapStorage,
        output_module: Option<CompiledBlock>,
    ) -> Self {
        let name_to_idx: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.clone(), i))
            .collect();

        let node = &nodes[0];

        assert!(
            node.messages.iter().any(|m| matches!(m, CompiledMessage::Iterator { .. })),
            "chat mode requires at least one iterator in messages"
        );

        let provider_config = providers
            .get(&node.provider)
            .unwrap_or_else(|| panic!("unknown provider: {}", node.provider))
            .clone();

        let tools: Vec<ToolSpec> = node
            .tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name.clone(),
                description: String::new(),
                params: t.params.clone(),
            })
            .collect();

        let first_iter_idx = node
            .messages
            .iter()
            .position(|m| matches!(m, CompiledMessage::Iterator { .. }))
            .unwrap();

        // Seed context metadata for the main node
        let ctx_prefix = format!("context.{}", node.name);
        storage.set(
            format!("{ctx_prefix}.model"),
            Output::Text(node.model.clone()),
        );
        storage.set(
            format!("{ctx_prefix}.provider"),
            Output::Text(node.provider.clone()),
        );

        let mut key_cache = HashMap::new();

        let ctx = RenderCtx {
            nodes: &nodes,
            name_to_idx: &name_to_idx,
            providers: &providers,
            fetch: &fetch,
            extern_fns: &extern_fns,
        };

        let mut static_messages = Vec::new();
        for msg in &nodes[0].messages[..first_iter_idx] {
            if let CompiledMessage::Block(block) = msg {
                let text =
                    render_with_deps(block, &mut storage, HashMap::new(), &ctx, &mut key_cache)
                        .await;
                static_messages.push(Message::text(&block.role, text));
            }
        }

        let per_turn_keys: Vec<String> = node
            .all_context_keys
            .iter()
            .filter(|k| !name_to_idx.contains_key(*k) && storage.get(k).is_none())
            .cloned()
            .collect();

        let iterator_keys: Vec<String> = nodes[0].messages[first_iter_idx..]
            .iter()
            .filter_map(|m| match m {
                CompiledMessage::Iterator { key, .. } => Some(key.clone()),
                _ => None,
            })
            .collect();

        Self {
            nodes,
            name_to_idx,
            providers,
            fetch,
            extern_fns,
            storage,
            key_cache,
            static_messages,
            first_iter_idx,
            per_turn_keys,
            iterator_keys,
            provider_config,
            tools,
            output_module,
        }
    }

    pub fn per_turn_keys(&self) -> &[String] {
        &self.per_turn_keys
    }

    pub async fn turn(&mut self, per_turn: HashMap<String, String>) -> String {
        // Remove always-strategy nodes so they re-resolve this turn
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].strategy.mode, StrategyMode::Always) {
                self.storage.remove(&self.nodes[i].name);
            }
        }

        for (key, value) in &per_turn {
            self.storage.set(key.clone(), Output::Text(value.clone()));
        }

        // Split borrows: immutable refs for ctx, mutable refs for storage/key_cache
        let nodes = &self.nodes;
        let name_to_idx = &self.name_to_idx;
        let providers = &self.providers;
        let fetch = &self.fetch;
        let extern_fns = &self.extern_fns;
        let storage = &mut self.storage;
        let key_cache = &mut self.key_cache;

        let ctx = RenderCtx { nodes, name_to_idx, providers, fetch, extern_fns };

        // Resolve cache_key through context system
        let cached_content = if let Some(ref cache_key) = nodes[0].cache_key {
            if storage.get(cache_key).is_none() {
                if let Some(&idx) = name_to_idx.get(cache_key.as_str()) {
                    resolve_node(idx, storage, &ctx, HashMap::new(), key_cache).await;
                }
            }
            match storage.get(cache_key) {
                Some(Output::Text(s)) => Some(s),
                _ => None,
            }
        } else {
            None
        };

        let mut messages = self.static_messages.clone();
        let mut new_messages: Vec<Message> = Vec::new();

        let dynamic_msgs = &nodes[0].messages[self.first_iter_idx..];
        for msg in dynamic_msgs {
            match msg {
                CompiledMessage::Iterator { key, block, slice, bind, role } => {
                    let expanded = expand_iterator(
                        key,
                        block.as_ref(),
                        slice,
                        bind,
                        role,
                        storage,
                        &ctx,
                        key_cache,
                    )
                    .await;
                    messages.extend(expanded);
                }
                CompiledMessage::Block(block) => {
                    let text =
                        render_with_deps(block, storage, HashMap::new(), &ctx, key_cache).await;
                    let message = Message::text(&block.role, text);
                    messages.push(message.clone());
                    new_messages.push(message);
                }
            }
        }

        let request = build_request(
            &self.provider_config,
            &nodes[0].model,
            &messages,
            &self.tools,
            &nodes[0].generation,
            cached_content.as_deref(),
        );
        let json = fetch
            .fetch(&request)
            .await
            .unwrap_or_else(|e| panic!("fetch error: {e}"));
        let response = parse_response(&self.provider_config.api, &json)
            .unwrap_or_else(|e| panic!("parse error: {e}"));

        let response_text = match response {
            ModelResponse::Text(text) => text,
            ModelResponse::ToolCalls(_) => panic!("tool calls not supported in chat mode yet"),
        };

        for iter_key in &self.iterator_keys {
            let mut history = match storage.get(iter_key) {
                Some(Output::Json(serde_json::Value::Array(arr))) => arr,
                _ => Vec::new(),
            };
            for msg in &new_messages {
                history.push(serde_json::json!({"type": msg.role, "text": msg.content}));
            }
            history.push(serde_json::json!({"type": "assistant", "text": response_text}));
            storage.set(
                iter_key.clone(),
                Output::Json(serde_json::Value::Array(history)),
            );
        }

        let final_output = if let Some(output_block) = &self.output_module {
            storage.set(nodes[0].name.clone(), Output::Text(response_text));
            let rendered =
                render_with_deps(output_block, storage, HashMap::new(), &ctx, key_cache).await;
            storage.remove(&nodes[0].name);
            rendered
        } else {
            response_text
        };

        for key in &self.per_turn_keys {
            storage.remove(key);
        }

        final_output
    }
}
