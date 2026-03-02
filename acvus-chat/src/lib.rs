use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value};
use acvus_orchestration::{
    build_cache_request, build_request, parse_cache_response, parse_response, CompiledBlock,
    CompiledMessage, CompiledNode, Fetch, HashMapStorage, Message, ModelResponse, NodeKind,
    ProviderConfig, Storage, StrategyMode, ToolSpec,
};

type Fut<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

// ---------------------------------------------------------------------------
// Internal: coroutine-driven rendering with demand-driven dependency resolution
// ---------------------------------------------------------------------------

struct RenderCtx<'a, F, R> {
    nodes: &'a [CompiledNode],
    name_to_idx: &'a HashMap<String, usize>,
    providers: &'a HashMap<String, ProviderConfig>,
    fetch: &'a F,
    extern_fns: &'a ExternFnRegistry,
    resolver: &'a R,
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
                } else if let Some(arc) = storage.get(&name) {
                    key = need.into_key(Arc::unwrap_or_clone(arc));
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

fn resolve_index(idx: i64, len: usize) -> usize {
    if idx < 0 {
        (len as i64 + idx).max(0) as usize
    } else {
        (idx as usize).min(len)
    }
}

impl<'a, F, R> RenderCtx<'a, F, R>
where
    F: Fetch,
    R: AsyncFn(String) -> Value + Sync,
{
    fn render_with_deps(
        &'a self,
        block: &'a CompiledBlock,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Value>,
        key_cache: &'a mut HashMap<String, String>,
    ) -> Fut<'a, String> {
        Box::pin(async move {
            let interp = Interpreter::new(block.module.clone(), self.extern_fns.clone());
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
                            if let Some(&idx) = self.name_to_idx.get(&name) {
                                self.resolve_node(idx, storage, bindings, key_cache).await;
                            }
                            let value = storage
                                .get(&name)
                                .map(Arc::unwrap_or_clone)
                                .unwrap_or_else(|| panic!("unresolved context: @{name}"));
                            let key = need.into_key(value);
                            result =
                                drive_block(&mut coroutine, key, &mut output, storage, &local);
                        } else {
                            if let Some(&idx) = self.name_to_idx.get(&name) {
                                if storage.get(&name).is_none() {
                                    self.resolve_node(idx, storage, HashMap::new(), key_cache)
                                        .await;
                                }
                            }
                            let value = if let Some(arc) = storage.get(&name) {
                                Arc::unwrap_or_clone(arc)
                            } else {
                                (self.resolver)(name.clone()).await
                            };
                            let key = need.into_key(value);
                            result =
                                drive_block(&mut coroutine, key, &mut output, storage, &local);
                        }
                    }
                }
            }
        })
    }

    fn resolve_node(
        &'a self,
        idx: usize,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Value>,
        key_cache: &'a mut HashMap<String, String>,
    ) -> Fut<'a, ()> {
        Box::pin(async move {
            let node = &self.nodes[idx];

            if matches!(node.strategy.mode, StrategyMode::IfModified) {
                if let Some(key_block) = &node.key_module {
                    let current_key =
                        self.render_with_deps(key_block, storage, local.clone(), key_cache).await;
                    if key_cache
                        .get(&node.name)
                        .map(|k| k == &current_key)
                        .unwrap_or(false)
                    {
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
                let text =
                    self.render_with_deps(block, storage, local.clone(), key_cache).await;
                messages.push(Message::text(&block.role, text));
            }

            let provider_config = self
                .providers
                .get(&node.provider)
                .unwrap_or_else(|| panic!("unknown provider: {}", node.provider))
                .clone();

            match node.kind {
                NodeKind::LlmCache { ref ttl, ref cache_config } => {
                    let request = build_cache_request(
                        &provider_config, &node.model, &messages, ttl, cache_config,
                    );
                    let json = self
                        .fetch
                        .fetch(&request)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("cache creation error for node {}: {e}", node.name)
                        });
                    let cache_name = parse_cache_response(&provider_config.api, &json)
                        .unwrap_or_else(|e| {
                            panic!("cache parse error for node {}: {e}", node.name)
                        });
                    storage.set(node.name.clone(), Value::String(cache_name));
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
                    let json = self
                        .fetch
                        .fetch(&request)
                        .await
                        .unwrap_or_else(|e| panic!("fetch error for node {}: {e}", node.name));
                    let response = parse_response(&provider_config.api, &json)
                        .unwrap_or_else(|e| panic!("parse error for node {}: {e}", node.name));

                    match response {
                        ModelResponse::Text(text) => {
                            storage.set(node.name.clone(), Value::String(text));
                        }
                        ModelResponse::ToolCalls(_) => {
                            panic!("tool calls in dependency node {} not supported", node.name);
                        }
                    }
                }
            }
        })
    }

    async fn expand_iterator(
        &'a self,
        key: &str,
        block: Option<&'a CompiledBlock>,
        slice: &Option<Vec<i64>>,
        bind: &Option<String>,
        role_override: &Option<String>,
        storage: &'a mut HashMapStorage,
        key_cache: &'a mut HashMap<String, String>,
    ) -> Vec<Message> {
        let stored = storage.get(key);
        let all_items = match stored.as_deref() {
            Some(Value::List(items)) => items,
            _ => return Vec::new(),
        };

        let items: &[Value] = if let Some(s) = slice {
            let len = all_items.len();
            match s.as_slice() {
                [start] => &all_items[resolve_index(*start, len)..],
                [start, end] => {
                    &all_items[resolve_index(*start, len)..resolve_index(*end, len)]
                }
                _ => all_items,
            }
        } else {
            all_items
        };

        let mut messages = Vec::new();
        for item in items {
            let (item_type, item_text) = match item {
                Value::Object(obj) => (
                    match obj.get("type") {
                        Some(Value::String(s)) => s.as_str(),
                        _ => "user",
                    },
                    match obj.get("text") {
                        Some(Value::String(s)) => s.as_str(),
                        _ => "",
                    },
                ),
                _ => ("user", ""),
            };
            let role = role_override.as_deref().unwrap_or(item_type);
            let text = item_text;

            if let Some(block) = block {
                let local = if let Some(bind_name) = bind {
                    HashMap::from([(bind_name.clone(), item.clone())])
                } else {
                    HashMap::from([
                        ("type".into(), Value::String(role.to_string())),
                        ("text".into(), Value::String(text.to_string())),
                    ])
                };
                let rendered =
                    self.render_with_deps(block, storage, local, key_cache).await;
                messages.push(Message::text(role, rendered));
            } else {
                messages.push(Message::text(role, text));
            }
        }
        messages
    }
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
            node.messages
                .iter()
                .any(|m| matches!(m, CompiledMessage::Iterator { .. })),
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
            Value::String(node.model.clone()),
        );
        storage.set(
            format!("{ctx_prefix}.provider"),
            Value::String(node.provider.clone()),
        );

        let mut key_cache = HashMap::new();

        let noop = |_: String| async { Value::Unit };
        let ctx = RenderCtx {
            nodes: &nodes,
            name_to_idx: &name_to_idx,
            providers: &providers,
            fetch: &fetch,
            extern_fns: &extern_fns,
            resolver: &noop,
        };

        let mut static_messages = Vec::new();
        for msg in &nodes[0].messages[..first_iter_idx] {
            if let CompiledMessage::Block(block) = msg {
                let text =
                    ctx.render_with_deps(block, &mut storage, HashMap::new(), &mut key_cache)
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

    pub async fn turn<R>(&mut self, resolver: &R) -> String
    where
        R: AsyncFn(String) -> Value + Sync,
    {
        // Remove always-strategy nodes so they re-resolve this turn
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].strategy.mode, StrategyMode::Always) {
                self.storage.remove(&self.nodes[i].name);
            }
        }

        // Split borrows: immutable refs for ctx, mutable refs for storage/key_cache
        let nodes = &self.nodes;
        let name_to_idx = &self.name_to_idx;
        let providers = &self.providers;
        let fetch = &self.fetch;
        let extern_fns = &self.extern_fns;
        let storage = &mut self.storage;
        let key_cache = &mut self.key_cache;

        let ctx = RenderCtx { nodes, name_to_idx, providers, fetch, extern_fns, resolver };

        // Resolve cache_key through context system
        let cached_content = if let Some(ref cache_key) = nodes[0].cache_key {
            if storage.get(cache_key).is_none() {
                if let Some(&idx) = name_to_idx.get(cache_key.as_str()) {
                    ctx.resolve_node(idx, storage, HashMap::new(), key_cache).await;
                }
            }
            storage.get(cache_key).and_then(|arc| match &*arc {
                Value::String(s) => Some(s.clone()),
                _ => None,
            })
        } else {
            None
        };

        let mut messages = self.static_messages.clone();
        let mut new_messages: Vec<Message> = Vec::new();

        let dynamic_msgs = &nodes[0].messages[self.first_iter_idx..];
        for msg in dynamic_msgs {
            match msg {
                CompiledMessage::Iterator { key, block, slice, bind, role } => {
                    let expanded = ctx
                        .expand_iterator(key, block.as_ref(), slice, bind, role, storage, key_cache)
                        .await;
                    messages.extend(expanded);
                }
                CompiledMessage::Block(block) => {
                    let text =
                        ctx.render_with_deps(block, storage, HashMap::new(), key_cache).await;
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
            let mut history = storage
                .get(iter_key)
                .and_then(|arc| match Arc::unwrap_or_clone(arc) {
                    Value::List(items) => Some(items),
                    _ => None,
                })
                .unwrap_or_default();
            for msg in &new_messages {
                history.push(Value::Object(BTreeMap::from([
                    ("type".into(), Value::String(msg.role.clone())),
                    ("text".into(), Value::String(msg.content.clone())),
                ])));
            }
            history.push(Value::Object(BTreeMap::from([
                ("type".into(), Value::String("assistant".into())),
                ("text".into(), Value::String(response_text.clone())),
            ])));
            storage.set(iter_key.clone(), Value::List(history));
        }

        let final_output = if let Some(output_block) = &self.output_module {
            storage.set(nodes[0].name.clone(), Value::String(response_text));
            let rendered =
                ctx.render_with_deps(output_block, storage, HashMap::new(), key_cache).await;
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
