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
    HashMapStorage, Message, ModelResponse, ProviderConfig, Storage, StrategyMode, ToolCall,
    ToolSpec,
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

fn ty_to_json_schema(ty: &acvus_mir::ty::Ty) -> &'static str {
    use acvus_mir::ty::Ty;
    match ty {
        Ty::String => "string",
        Ty::Int => "integer",
        Ty::Float => "number",
        Ty::Bool => "boolean",
        _ => "string",
    }
}

fn tool_specs(tools: &[CompiledToolBinding]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            params: t.params.iter().map(|(k, v)| (k.clone(), ty_to_json_schema(v).to_string())).collect(),
        })
        .collect()
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            Value::Object(obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

const MAX_TOOL_ROUNDS: usize = 10;

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
            tracing::debug!(context = %name, "resolve via external resolver");
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
            tracing::debug!(node = %node.name, "resolve_node");

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
                        tracing::debug!(node = %node.name, "if_modified skip (key unchanged)");
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
                    tracing::debug!(node = %node.name, body = %request.body, "llm_cache fetch request");
                    let json = self.fetch.fetch(&request).await.map_err(|e| {
                        tracing::warn!(node = %node.name, error = %e, "llm_cache fetch failed");
                        ChatError::Fetch { node: node.name.clone(), detail: e }
                    })?;
                    tracing::debug!(node = %node.name, response = %json, "llm_cache fetch response");
                    let cache_name =
                        parse_cache_response(&provider_config.api, &json).map_err(|e| {
                            tracing::warn!(node = %node.name, response = %json, "llm_cache parse failed");
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
                    tracing::debug!(node = %node.name, body = %request.body, "llm fetch request");
                    let json = self.fetch.fetch(&request).await.map_err(|e| {
                        tracing::warn!(node = %node.name, error = %e, "llm fetch failed");
                        ChatError::Fetch { node: node.name.clone(), detail: e }
                    })?;
                    tracing::debug!(node = %node.name, response = %json, "llm fetch response");
                    let mut response =
                        parse_response(&provider_config.api, &json).map_err(|e| {
                            tracing::warn!(node = %node.name, response = %json, "llm parse failed");
                            ChatError::Parse { node: node.name.clone(), detail: e }
                        })?;

                    let mut tool_rounds = 0usize;
                    loop {
                        match response {
                            ModelResponse::Text(text) => {
                                tracing::debug!(node = %node.name, len = text.len(), "llm text response");
                                let iterator_keys: Vec<String> = messages
                                    .iter()
                                    .filter_map(|m| match m {
                                        CompiledMessage::Iterator { key, .. } => Some(key.clone()),
                                        _ => None,
                                    })
                                    .collect();
                                if !iterator_keys.is_empty() {
                                    update_history(storage, &iterator_keys, &new_turn_messages, &text);
                                }
                                storage.set(node.name.clone(), Value::String(text));
                                break;
                            }
                            ModelResponse::ToolCalls(calls) => {
                                tool_rounds += 1;
                                if tool_rounds > MAX_TOOL_ROUNDS {
                                    return Err(ChatError::ToolCallLimitExceeded(node.name.clone()));
                                }

                                rendered.push(Message {
                                    role: "assistant".into(),
                                    content: String::new(),
                                    tool_calls: calls.clone(),
                                    tool_call_id: None,
                                });

                                for call in &calls {
                                    tracing::debug!(node = %node.name, tool = %call.name, args = %call.arguments, "tool call received");
                                    let result_text = self.execute_tool_call(
                                        call, &node.name, tools, storage, key_cache, turn_local,
                                    ).await?;
                                    tracing::debug!(tool = %call.name, result = %result_text, "tool call result");
                                    rendered.push(Message {
                                        role: "tool".into(),
                                        content: result_text,
                                        tool_calls: Vec::new(),
                                        tool_call_id: Some(call.id.clone()),
                                    });
                                }

                                let request = build_request(
                                    &provider_config, model, &rendered, &specs, generation, cached_content.as_deref(),
                                );
                                tracing::debug!(node = %node.name, body = %request.body, "llm fetch request (tool followup)");
                                let json = self.fetch.fetch(&request).await.map_err(|e| {
                                    tracing::warn!(node = %node.name, error = %e, "llm fetch failed (tool followup)");
                                    ChatError::Fetch { node: node.name.clone(), detail: e }
                                })?;
                                tracing::debug!(node = %node.name, response = %json, "llm fetch response (tool followup)");
                                response = parse_response(&provider_config.api, &json).map_err(|e| {
                                    tracing::warn!(node = %node.name, response = %json, "llm parse failed (tool followup)");
                                    ChatError::Parse { node: node.name.clone(), detail: e }
                                })?;
                            }
                        }
                    }
                }
            }
            Ok(())
        })
    }

    async fn execute_tool_call(
        &'a self,
        call: &ToolCall,
        node_name: &str,
        tools: &[CompiledToolBinding],
        storage: &'a mut HashMapStorage,
        key_cache: &'a mut HashMap<String, String>,
        turn_local: &'a mut HashMap<String, Value>,
    ) -> Result<String, ChatError> {
        let binding = tools.iter().find(|t| t.name == call.name).ok_or_else(|| {
            ChatError::ToolNotFound { node: node_name.to_string(), tool: call.name.clone() }
        })?;
        let target_idx = *self.name_to_idx.get(&binding.node).ok_or_else(|| {
            ChatError::ToolTargetNotFound {
                tool: call.name.clone(),
                target: binding.node.clone(),
            }
        })?;

        let tool_local: HashMap<String, Value> = match &call.arguments {
            serde_json::Value::Object(obj) => {
                obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect()
            }
            _ => HashMap::new(),
        };

        self.resolve_node(target_idx, storage, tool_local, key_cache, turn_local)
            .await?;

        let result = storage
            .get(&binding.node)
            .map(|arc| match &*arc {
                Value::String(s) => s.clone(),
                other => format!("{other:?}"),
            })
            .unwrap_or_default();

        Ok(result)
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
            _ => {
                tracing::debug!(key = %key, "expand_iterator: no history");
                return Ok(Vec::new());
            }
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

        tracing::debug!(key = %key, count = items.len(), "expand_iterator");
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
        let entrypoint = &self.nodes[self.entrypoint_idx].name;
        tracing::debug!(entrypoint = %entrypoint, "turn start");

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::extern_module::ExternRegistry;
    use acvus_orchestration::{
        compile_nodes, ApiKind, GenerationParams, HttpRequest, MessageSpec, NodeKind, NodeSpec,
        Strategy, ToolBinding,
    };

    // -- MockFetch: returns queued JSON responses in order -----------------------

    struct MockFetch {
        responses: Mutex<Vec<serde_json::Value>>,
    }

    impl MockFetch {
        fn new(responses: Vec<serde_json::Value>) -> Self {
            Self { responses: Mutex::new(responses) }
        }
    }

    impl Fetch for MockFetch {
        async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
            let mut q = self.responses.lock().unwrap();
            if q.is_empty() {
                return Err("no more mock responses".into());
            }
            Ok(q.remove(0))
        }
    }

    // -- helpers ----------------------------------------------------------------

    fn openai_text_response(text: &str) -> serde_json::Value {
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": text
                }
            }]
        })
    }

    fn openai_tool_call_response(calls: Vec<(&str, &str, serde_json::Value)>) -> serde_json::Value {
        let tool_calls: Vec<serde_json::Value> = calls
            .into_iter()
            .enumerate()
            .map(|(_, (id, name, args))| {
                serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&args).unwrap()
                    }
                })
            })
            .collect();
        serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls
                }
            }]
        })
    }

    fn default_provider() -> (String, ProviderConfig) {
        (
            "test".into(),
            ProviderConfig {
                api: ApiKind::OpenAI,
                endpoint: "http://mock".into(),
                api_key: String::new(),
            },
        )
    }

    fn noop_resolver() -> impl AsyncFn(String) -> Value + Sync {
        |_: String| async { Value::Unit }
    }

    fn compile_test_nodes(specs: &[NodeSpec]) -> Vec<CompiledNode> {
        compile_nodes(specs, &HashMap::new(), &ExternRegistry::default()).unwrap()
    }

    // -- tests ------------------------------------------------------------------

    #[tokio::test]
    async fn new_valid_entrypoint() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain { source: "hello".into() },
            strategy: Strategy::default(),
        }]);
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn new_invalid_entrypoint() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain { source: "hello".into() },
            strategy: Strategy::default(),
        }]);
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "nonexistent",
        )
        .await;
        assert!(matches!(result, Err(ChatError::EntrypointNotFound(_))));
    }

    #[tokio::test]
    async fn turn_plain_node() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Plain { source: "hello world".into() },
            strategy: Strategy::default(),
        }]);
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            MockFetch::new(vec![]),
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(result, "hello world");
    }

    #[tokio::test]
    async fn turn_llm_text_response() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Llm {
                provider: "test".into(),
                model: "gpt-test".into(),
                messages: vec![MessageSpec::Block {
                    role: "user".into(),
                    source: "hi".into(),
                }],
                tools: vec![],
                generation: GenerationParams::default(),
                cache_key: None,
            },
            strategy: Strategy::default(),
        }]);
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![openai_text_response("hello from LLM")]);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(result, "hello from LLM");
    }

    #[tokio::test]
    async fn turn_tool_call_round_trip() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain { source: "tool result text".into() },
                strategy: Strategy::default(),
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm {
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: "user".into(),
                        source: "use the tool".into(),
                    }],
                    tools: vec![ToolBinding {
                        name: "my_tool".into(),
                        description: String::new(),
                        node: "tool_target".into(),
                        params: HashMap::new(),
                    }],
                    generation: GenerationParams::default(),
                    cache_key: None,
                },
                strategy: Strategy::default(),
            },
        ]);
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![
            openai_tool_call_response(vec![("call_1", "my_tool", serde_json::json!({}))]),
            openai_text_response("final answer"),
        ]);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver()).await.unwrap();
        assert_eq!(result, "final answer");
    }

    #[tokio::test]
    async fn turn_tool_not_found() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain { source: "result".into() },
                strategy: Strategy::default(),
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm {
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: "user".into(),
                        source: "use tool".into(),
                    }],
                    tools: vec![ToolBinding {
                        name: "my_tool".into(),
                        description: String::new(),
                        node: "tool_target".into(),
                        params: HashMap::new(),
                    }],
                    generation: GenerationParams::default(),
                    cache_key: None,
                },
                strategy: Strategy::default(),
            },
        ]);
        let (pname, pconfig) = default_provider();
        // Model calls a tool name that doesn't match any binding
        let mock = MockFetch::new(vec![openai_tool_call_response(vec![(
            "call_1",
            "unknown_tool",
            serde_json::json!({}),
        )])]);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await
        .unwrap();

        let err = engine.turn(&noop_resolver()).await.unwrap_err();
        assert!(matches!(err, ChatError::ToolNotFound { .. }));
    }

    #[tokio::test]
    async fn turn_tool_call_limit() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain { source: "result".into() },
                strategy: Strategy::default(),
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm {
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: "user".into(),
                        source: "loop".into(),
                    }],
                    tools: vec![ToolBinding {
                        name: "my_tool".into(),
                        description: String::new(),
                        node: "tool_target".into(),
                        params: HashMap::new(),
                    }],
                    generation: GenerationParams::default(),
                    cache_key: None,
                },
                strategy: Strategy::default(),
            },
        ]);
        let (pname, pconfig) = default_provider();
        // Return tool calls 11 times → exceeds MAX_TOOL_ROUNDS (10)
        let responses: Vec<_> = (0..=MAX_TOOL_ROUNDS)
            .map(|i| {
                openai_tool_call_response(vec![(
                    &format!("call_{i}"),
                    "my_tool",
                    serde_json::json!({}),
                )])
            })
            .collect();
        let mock = MockFetch::new(responses);
        let mut engine = ChatEngine::new(
            nodes,
            HashMap::from([(pname, pconfig)]),
            mock,
            ExternFnRegistry::new(),
            HashMapStorage::new(),
            "main",
        )
        .await
        .unwrap();

        let err = engine.turn(&noop_resolver()).await.unwrap_err();
        assert!(matches!(err, ChatError::ToolCallLimitExceeded(_)));
    }

    #[test]
    fn tool_specs_json_schema_types() {
        use acvus_mir::ty::Ty;
        let tools = vec![CompiledToolBinding {
            name: "t".into(),
            description: String::new(),
            node: "n".into(),
            params: HashMap::from([
                ("a".into(), Ty::String),
                ("b".into(), Ty::Int),
                ("c".into(), Ty::Float),
                ("d".into(), Ty::Bool),
            ]),
        }];
        let specs = tool_specs(&tools);
        assert_eq!(specs[0].params["a"], "string");
        assert_eq!(specs[0].params["b"], "integer");
        assert_eq!(specs[0].params["c"], "number");
        assert_eq!(specs[0].params["d"], "boolean");
    }

    #[test]
    fn json_to_value_basic() {
        assert!(matches!(json_to_value(&serde_json::json!(null)), Value::Unit));
        assert!(matches!(json_to_value(&serde_json::json!(true)), Value::Bool(true)));
        assert!(matches!(json_to_value(&serde_json::json!(42)), Value::Int(42)));
        assert!(matches!(json_to_value(&serde_json::json!(3.14)), Value::Float(f) if (f - 3.14).abs() < f64::EPSILON));
        assert!(matches!(json_to_value(&serde_json::json!("hello")), Value::String(s) if s == "hello"));
        assert!(matches!(json_to_value(&serde_json::json!([1, 2])), Value::List(v) if v.len() == 2));
        assert!(matches!(json_to_value(&serde_json::json!({"key": "val"})), Value::Object(m) if m.contains_key("key")));
    }
}
