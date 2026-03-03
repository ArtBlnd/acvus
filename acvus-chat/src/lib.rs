mod error;

pub use error::ChatError;

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value};
use acvus_orchestration::{
    build_cache_request, build_request, parse_cache_response, parse_response, CompiledBlock,
    CompiledMessage, CompiledNode, CompiledNodeKind, CompiledToolBinding, Fetch,
    HashMapStorage, Message, ModelResponse, ProviderConfig, Storage, StrategyMode, ToolSpec,
};

use error::value_type_name;

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
    turn_local: &HashMap<String, Value>,
) -> Result<BlockDriveResult, ChatError> {
    loop {
        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => output.push_str(&s),
                    other => return Err(ChatError::EmitType(value_type_name(&other))),
                }
                key = next_key;
            }
            Stepped::NeedContext(need) => {
                if !need.bindings().is_empty() {
                    return Ok(BlockDriveResult::NeedContext(need));
                }
                let name = need.name().to_string();
                if let Some(value) = local.get(&name) {
                    key = need.into_key(value.clone());
                } else if let Some(value) = turn_local.get(&name) {
                    key = need.into_key(value.clone());
                } else if let Some(arc) = storage.get(&name) {
                    key = need.into_key(Arc::unwrap_or_clone(arc));
                } else {
                    return Ok(BlockDriveResult::NeedContext(need));
                }
            }
            Stepped::Done => {
                return Ok(BlockDriveResult::Done(std::mem::take(output)));
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

fn item_fields(item: &Value) -> (&str, &str) {
    match item {
        Value::Object(obj) => {
            let typ = match obj.get("type") {
                Some(Value::String(s)) => s.as_str(),
                _ => "user",
            };
            let text = match obj.get("text") {
                Some(Value::String(s)) => s.as_str(),
                _ => "",
            };
            (typ, text)
        }
        _ => ("user", ""),
    }
}

fn update_history(
    storage: &mut HashMapStorage,
    iterator_keys: &[String],
    new_messages: &[Message],
    response_text: &str,
) {
    for iter_key in iterator_keys {
        let mut history = storage
            .get(iter_key)
            .and_then(|arc| match Arc::unwrap_or_clone(arc) {
                Value::List(items) => Some(items),
                _ => None,
            })
            .unwrap_or_default();
        for msg in new_messages {
            history.push(Value::Object(BTreeMap::from([
                ("type".into(), Value::String(msg.role.clone())),
                ("text".into(), Value::String(msg.content.clone())),
            ])));
        }
        history.push(Value::Object(BTreeMap::from([
            ("type".into(), Value::String("assistant".into())),
            ("text".into(), Value::String(response_text.to_string())),
        ])));
        storage.set(iter_key.clone(), Value::List(history));
    }
}

fn tool_specs(tools: &[CompiledToolBinding]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: String::new(),
            params: t.params.iter().map(|(k, v)| (k.clone(), format!("{v:?}"))).collect(),
        })
        .collect()
}

impl<'a, F, R> RenderCtx<'a, F, R>
where
    F: Fetch,
    R: AsyncFn(String) -> Value + Sync,
{
    async fn resolve_context(
        &self,
        name: &str,
        bindings: HashMap<String, Value>,
        storage: &mut HashMapStorage,
        key_cache: &mut HashMap<String, String>,
        turn_local: &mut HashMap<String, Value>,
    ) -> Result<Value, ChatError> {
        if !bindings.is_empty() {
            if let Some(&idx) = self.name_to_idx.get(name) {
                self.resolve_node(idx, storage, bindings, key_cache, turn_local)
                    .await?;
            }
            return storage
                .get(name)
                .map(Arc::unwrap_or_clone)
                .ok_or_else(|| ChatError::UnresolvedContext(name.to_string()));
        }

        if let Some(&idx) = self.name_to_idx.get(name) {
            if storage.get(name).is_none() {
                self.resolve_node(idx, storage, HashMap::new(), key_cache, turn_local)
                    .await?;
            }
        }
        if let Some(arc) = storage.get(name) {
            Ok(Arc::unwrap_or_clone(arc))
        } else if let Some(cached) = turn_local.get(name) {
            Ok(cached.clone())
        } else {
            let resolved = (self.resolver)(name.to_string()).await;
            turn_local.insert(name.to_string(), resolved.clone());
            Ok(resolved)
        }
    }

    async fn resolve_cached_content(
        &self,
        cache_key: &str,
        storage: &mut HashMapStorage,
        key_cache: &mut HashMap<String, String>,
        turn_local: &mut HashMap<String, Value>,
    ) -> Result<Option<String>, ChatError> {
        if storage.get(cache_key).is_none() {
            if let Some(&idx) = self.name_to_idx.get(cache_key) {
                self.resolve_node(idx, storage, HashMap::new(), key_cache, turn_local)
                    .await?;
            }
        }
        Ok(storage.get(cache_key).and_then(|arc| match &*arc {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }))
    }

    fn render_with_deps(
        &'a self,
        block: &'a CompiledBlock,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Value>,
        key_cache: &'a mut HashMap<String, String>,
        turn_local: &'a mut HashMap<String, Value>,
    ) -> Fut<'a, Result<String, ChatError>> {
        Box::pin(async move {
            let interp = Interpreter::new(block.module.clone(), self.extern_fns.clone());
            let (mut coroutine, key) = interp.execute();
            let mut output = String::new();

            let mut result =
                drive_block(&mut coroutine, key, &mut output, storage, &local, turn_local)?;
            loop {
                match result {
                    BlockDriveResult::Done(text) => return Ok(text),
                    BlockDriveResult::NeedContext(need) => {
                        let name = need.name().to_string();
                        let bindings = need.bindings().clone();
                        let value = self
                            .resolve_context(&name, bindings, storage, key_cache, turn_local)
                            .await?;
                        let key = need.into_key(value);
                        result = drive_block(
                            &mut coroutine, key, &mut output, storage, &local, turn_local,
                        )?;
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
        turn_local: &'a mut HashMap<String, Value>,
    ) -> Fut<'a, Result<(), ChatError>> {
        Box::pin(async move {
            let node = &self.nodes[idx];

            if matches!(node.strategy.mode, StrategyMode::IfModified) {
                if let Some(key_block) = &node.key_module {
                    let current_key = self
                        .render_with_deps(key_block, storage, local.clone(), key_cache, turn_local)
                        .await?;
                    if key_cache
                        .get(&node.name)
                        .map(|k| k == &current_key)
                        .unwrap_or(false)
                    {
                        return Ok(());
                    }
                    key_cache.insert(node.name.clone(), current_key);
                }
            }

            match &node.kind {
                CompiledNodeKind::Plain { block } => {
                    let text = self
                        .render_with_deps(block, storage, local, key_cache, turn_local)
                        .await?;
                    storage.set(node.name.clone(), Value::String(text));
                }
                CompiledNodeKind::LlmCache { provider, model, messages, ttl, cache_config } => {
                    let mut rendered = Vec::new();
                    for msg in messages {
                        let block = match msg {
                            CompiledMessage::Block(block) => block,
                            CompiledMessage::Iterator { .. } => continue,
                        };
                        let text = self
                            .render_with_deps(block, storage, local.clone(), key_cache, turn_local)
                            .await?;
                        rendered.push(Message::text(&block.role, text));
                    }

                    let provider_config = self
                        .providers
                        .get(provider)
                        .ok_or_else(|| ChatError::UnknownProvider(provider.clone()))?
                        .clone();

                    let request = build_cache_request(
                        &provider_config, model, &rendered, ttl, cache_config,
                    );
                    let json = self.fetch.fetch(&request).await.map_err(|e| {
                        ChatError::Fetch { node: node.name.clone(), detail: e }
                    })?;
                    let cache_name =
                        parse_cache_response(&provider_config.api, &json).map_err(|e| {
                            ChatError::Parse { node: node.name.clone(), detail: e }
                        })?;
                    storage.set(node.name.clone(), Value::String(cache_name));
                }
                CompiledNodeKind::Llm { provider, model, messages, tools, generation, cache_key } => {
                    let mut rendered = Vec::new();
                    let mut new_turn_messages = Vec::new();
                    let first_iter = messages
                        .iter()
                        .position(|m| matches!(m, CompiledMessage::Iterator { .. }));

                    let cached_content = if let Some(ck) = cache_key {
                        self.resolve_cached_content(ck, storage, key_cache, turn_local)
                            .await?
                    } else {
                        None
                    };

                    for (i, msg) in messages.iter().enumerate() {
                        match msg {
                            CompiledMessage::Block(block) => {
                                let text = self
                                    .render_with_deps(block, storage, local.clone(), key_cache, turn_local)
                                    .await?;
                                let message = Message::text(&block.role, text);
                                rendered.push(message.clone());
                                if first_iter.map_or(false, |pos| i > pos) {
                                    new_turn_messages.push(message);
                                }
                            }
                            CompiledMessage::Iterator { key, block, slice, bind, role } => {
                                let expanded = self
                                    .expand_iterator(key, block.as_ref(), slice, bind, role, storage, key_cache, turn_local)
                                    .await?;
                                rendered.extend(expanded);
                            }
                        }
                    }

                    let provider_config = self
                        .providers
                        .get(provider)
                        .ok_or_else(|| ChatError::UnknownProvider(provider.clone()))?
                        .clone();

                    let specs = tool_specs(tools);
                    let request = build_request(
                        &provider_config, model, &rendered, &specs, generation, cached_content.as_deref(),
                    );
                    let json = self.fetch.fetch(&request).await.map_err(|e| {
                        ChatError::Fetch { node: node.name.clone(), detail: e }
                    })?;
                    let response =
                        parse_response(&provider_config.api, &json).map_err(|e| {
                            ChatError::Parse { node: node.name.clone(), detail: e }
                        })?;

                    let response_text = match response {
                        ModelResponse::Text(text) => text,
                        ModelResponse::ToolCalls(_) => {
                            return Err(ChatError::UnsupportedToolCalls(node.name.clone()));
                        }
                    };

                    // Update history for iterator keys
                    let iterator_keys: Vec<String> = messages
                        .iter()
                        .filter_map(|m| match m {
                            CompiledMessage::Iterator { key, .. } => Some(key.clone()),
                            _ => None,
                        })
                        .collect();
                    if !iterator_keys.is_empty() {
                        update_history(storage, &iterator_keys, &new_turn_messages, &response_text);
                    }

                    storage.set(node.name.clone(), Value::String(response_text));
                }
            }
            Ok(())
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
        turn_local: &'a mut HashMap<String, Value>,
    ) -> Result<Vec<Message>, ChatError> {
        let stored = storage.get(key);
        let all_items = match stored.as_deref() {
            Some(Value::List(items)) => items,
            _ => return Ok(Vec::new()),
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
            let (item_type, item_text) = item_fields(item);
            let role = role_override.as_deref().unwrap_or(item_type);

            if let Some(block) = block {
                let local = if let Some(bind_name) = bind {
                    HashMap::from([(bind_name.clone(), item.clone())])
                } else {
                    HashMap::from([
                        ("type".into(), Value::String(role.to_string())),
                        ("text".into(), Value::String(item_text.to_string())),
                    ])
                };
                let rendered = self
                    .render_with_deps(block, storage, local, key_cache, turn_local)
                    .await?;
                messages.push(Message::text(role, rendered));
            } else {
                messages.push(Message::text(role, item_text));
            }
        }
        Ok(messages)
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
    entrypoint_idx: usize,
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
        entrypoint: &str,
    ) -> Result<Self, ChatError> {
        let name_to_idx: HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.clone(), i))
            .collect();

        let entrypoint_idx = *name_to_idx
            .get(entrypoint)
            .ok_or_else(|| ChatError::EntrypointNotFound(entrypoint.to_string()))?;

        // Seed context metadata for all Llm nodes
        for node in &nodes {
            if let CompiledNodeKind::Llm { provider, model, .. } = &node.kind {
                let prefix = format!("context.{}", node.name);
                storage.set(format!("{prefix}.model"), Value::String(model.clone()));
                storage.set(format!("{prefix}.provider"), Value::String(provider.clone()));
            }
        }

        Ok(Self {
            nodes,
            name_to_idx,
            providers,
            fetch,
            extern_fns,
            storage,
            key_cache: HashMap::new(),
            entrypoint_idx,
        })
    }

    pub async fn turn<R>(&mut self, resolver: &R) -> Result<String, ChatError>
    where
        R: AsyncFn(String) -> Value + Sync,
    {
        // Remove always-strategy nodes so they re-resolve this turn
        for node in &self.nodes {
            if matches!(node.strategy.mode, StrategyMode::Always) {
                self.storage.remove(&node.name);
            }
        }

        let mut turn_local = HashMap::new();

        let ctx = RenderCtx {
            nodes: &self.nodes,
            name_to_idx: &self.name_to_idx,
            providers: &self.providers,
            fetch: &self.fetch,
            extern_fns: &self.extern_fns,
            resolver,
        };

        ctx.resolve_node(
            self.entrypoint_idx,
            &mut self.storage,
            HashMap::new(),
            &mut self.key_cache,
            &mut turn_local,
        )
        .await?;

        let name = &self.nodes[self.entrypoint_idx].name;
        let result = self
            .storage
            .get(name)
            .map(|v| Arc::unwrap_or_clone(v))
            .ok_or_else(|| ChatError::UnresolvedContext(name.clone()))?;
        match result {
            Value::String(s) => Ok(s),
            other => Err(ChatError::EmitType(value_type_name(&other))),
        }
    }
}
