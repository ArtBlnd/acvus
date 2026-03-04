mod error;

pub use error::ChatError;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use acvus_interpreter::{
    ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value,
};
use acvus_orchestration::{
    CompiledBlock, CompiledHistory, CompiledMessage, CompiledNode, CompiledNodeKind,
    CompiledScript, CompiledToolBinding, Fetch, HashMapStorage, LlmModel, Message, ModelResponse,
    ProviderConfig, Storage, StrategyMode, TokenBudget, ToolCall, ToolSpec, build_cache_request,
    create_llm_model, parse_cache_response,
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
    local: &HashMap<String, Arc<Value>>,
    turn_local: &HashMap<String, Arc<Value>>,
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
                if let Some(arc) = local.get(&name) {
                    key = need.into_key(Arc::clone(arc));
                } else if let Some(arc) = turn_local.get(&name) {
                    key = need.into_key(Arc::clone(arc));
                } else if let Some(arc) = storage.get(&name) {
                    key = need.into_key(arc);
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

enum ScriptDriveResult {
    Value(Value),
    NeedContext(NeedContextStepped),
}

fn drive_script(
    coroutine: &mut acvus_interpreter::Coroutine,
    mut key: ResumeKey,
    storage: &HashMapStorage,
    local: &HashMap<String, Arc<Value>>,
    turn_local: &HashMap<String, Arc<Value>>,
) -> Result<ScriptDriveResult, ChatError> {
    loop {
        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, _next_key) = emit.into_parts();
                return Ok(ScriptDriveResult::Value(value));
            }
            Stepped::NeedContext(need) => {
                if !need.bindings().is_empty() {
                    return Ok(ScriptDriveResult::NeedContext(need));
                }
                let name = need.name().to_string();
                if let Some(arc) = local.get(&name) {
                    key = need.into_key(Arc::clone(arc));
                } else if let Some(arc) = turn_local.get(&name) {
                    key = need.into_key(Arc::clone(arc));
                } else if let Some(arc) = storage.get(&name) {
                    key = need.into_key(arc);
                } else {
                    return Ok(ScriptDriveResult::NeedContext(need));
                }
            }
            Stepped::Done => {
                return Ok(ScriptDriveResult::Value(Value::Unit));
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
            let role = match obj.get("role") {
                Some(Value::String(s)) => s.as_str(),
                _ => "user",
            };
            let content = match obj.get("content") {
                Some(Value::String(s)) => s.as_str(),
                _ => "",
            };
            (role, content)
        }
        Value::String(s) => ("user", s.as_str()),
        _ => ("user", ""),
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
            params: t
                .params
                .iter()
                .map(|(k, v)| (k.clone(), ty_to_json_schema(v).to_string()))
                .collect(),
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
        serde_json::Value::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect(),
        ),
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
        bind_cache: &mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &mut HashMap<String, Arc<Value>>,
    ) -> Result<Arc<Value>, ChatError> {
        if !bindings.is_empty() {
            if let Some(&idx) = self.name_to_idx.get(name) {
                let local = bindings
                    .into_iter()
                    .map(|(k, v)| (k, Arc::new(v)))
                    .collect();
                self.resolve_node(idx, storage, local, bind_cache, turn_local)
                    .await?;
            }
            return storage
                .get(name)
                .ok_or_else(|| ChatError::UnresolvedContext(name.to_string()));
        }

        if let Some(&idx) = self.name_to_idx.get(name)
            && storage.get(name).is_none()
        {
            self.resolve_node(idx, storage, HashMap::new(), bind_cache, turn_local)
                .await?;
        }
        if let Some(arc) = storage.get(name) {
            Ok(arc)
        } else if let Some(arc) = turn_local.get(name) {
            Ok(Arc::clone(arc))
        } else {
            tracing::debug!(context = %name, "resolve via external resolver");
            let resolved = Arc::new((self.resolver)(name.to_string()).await);
            turn_local.insert(name.to_string(), Arc::clone(&resolved));
            Ok(resolved)
        }
    }

    async fn eval_script(
        &self,
        expr: &CompiledScript,
        storage: &mut HashMapStorage,
        bind_cache: &mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &mut HashMap<String, Arc<Value>>,
    ) -> Result<Value, ChatError> {
        let interp = Interpreter::new(expr.module.clone(), self.extern_fns);
        let (mut coroutine, key) = interp.execute();
        let mut result = drive_script(&mut coroutine, key, storage, &HashMap::new(), turn_local)?;
        loop {
            match result {
                ScriptDriveResult::Value(v) => return Ok(v),
                ScriptDriveResult::NeedContext(need) => {
                    let name = need.name().to_string();
                    let bindings = need.bindings().clone();
                    let value = self
                        .resolve_context(&name, bindings, storage, bind_cache, turn_local)
                        .await?;
                    let key = need.into_key(value);
                    result =
                        drive_script(&mut coroutine, key, storage, &HashMap::new(), turn_local)?;
                }
            }
        }
    }

    async fn resolve_cached_content(
        &self,
        expr: &CompiledScript,
        storage: &mut HashMapStorage,
        bind_cache: &mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &mut HashMap<String, Arc<Value>>,
    ) -> Result<Option<String>, ChatError> {
        let value = self
            .eval_script(expr, storage, bind_cache, turn_local)
            .await?;
        match value {
            Value::String(s) => Ok(Some(s)),
            Value::Unit => Ok(None),
            _ => Ok(None),
        }
    }

    fn render_with_deps(
        &'a self,
        block: &'a CompiledBlock,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<String, ChatError>> {
        Box::pin(async move {
            let interp = Interpreter::new(block.module.clone(), self.extern_fns);
            let (mut coroutine, key) = interp.execute();
            let mut output = String::new();

            let mut result = drive_block(
                &mut coroutine,
                key,
                &mut output,
                storage,
                &local,
                turn_local,
            )?;
            loop {
                match result {
                    BlockDriveResult::Done(text) => return Ok(text),
                    BlockDriveResult::NeedContext(need) => {
                        let name = need.name().to_string();
                        let bindings = need.bindings().clone();
                        let value = self
                            .resolve_context(&name, bindings, storage, bind_cache, turn_local)
                            .await?;
                        let key = need.into_key(value);
                        result = drive_block(
                            &mut coroutine,
                            key,
                            &mut output,
                            storage,
                            &local,
                            turn_local,
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
        local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Fut<'a, Result<(), ChatError>> {
        Box::pin(self.resolve_node_impl(idx, storage, local, bind_cache, turn_local))
    }

    async fn resolve_node_impl(
        &'a self,
        idx: usize,
        storage: &'a mut HashMapStorage,
        mut local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<(), ChatError> {
        let node = &self.nodes[idx];
        tracing::debug!(node = %node.name, "resolve_node");

        // IfModified: evaluate bind script, check cache
        if matches!(node.strategy.mode, StrategyMode::IfModified)
            && let Some(bind_script) = &node.bind_module
        {
            let bind_value = self
                .eval_script(bind_script, storage, bind_cache, turn_local)
                .await?;
            if let Some(entries) = bind_cache.get(&node.name)
                && let Some((_, cached_output)) = entries.iter().find(|(v, _)| v == &bind_value)
            {
                tracing::debug!(node = %node.name, "skip (bind cached)");
                storage.set(node.name.clone(), Value::clone(cached_output));
                return Ok(());
            }
            local.insert("bind".into(), Arc::new(bind_value));
        }

        match &node.kind {
            CompiledNodeKind::Plain(plain) => {
                let text = self
                    .render_with_deps(&plain.block, storage, local, bind_cache, turn_local)
                    .await?;
                storage.set(node.name.clone(), Value::String(text));
            }
            CompiledNodeKind::LlmCache(_) => {
                self.resolve_llm_cache(node, storage, local, bind_cache, turn_local)
                    .await?;
            }
            CompiledNodeKind::Llm(_) => {
                self.resolve_llm(node, storage, local, bind_cache, turn_local)
                    .await?;
            }
        }
        Ok(())
    }

    async fn resolve_llm_cache(
        &'a self,
        node: &'a CompiledNode,
        storage: &'a mut HashMapStorage,
        local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<(), ChatError> {
        let CompiledNodeKind::LlmCache(llm_cache) = &node.kind else {
            unreachable!()
        };
        let (provider, model, messages, ttl, cache_config) = (
            &llm_cache.provider,
            &llm_cache.model,
            &llm_cache.messages,
            &llm_cache.ttl,
            &llm_cache.cache_config,
        );

        let mut rendered = Vec::new();
        for msg in messages {
            let block = match msg {
                CompiledMessage::Block(block) => block,
                CompiledMessage::Iterator { .. } => continue,
            };
            let text = self
                .render_with_deps(block, storage, local.clone(), bind_cache, turn_local)
                .await?;
            rendered.push(Message::text(&block.role, text));
        }

        let provider_config = self
            .providers
            .get(provider)
            .ok_or_else(|| ChatError::UnknownProvider(provider.clone()))?
            .clone();

        let request = build_cache_request(&provider_config, model, &rendered, ttl, cache_config);
        tracing::debug!(node = %node.name, body = %request.body, "llm_cache fetch request");
        let json = self.fetch.fetch(&request).await.map_err(|e| {
            tracing::warn!(node = %node.name, error = %e, "llm_cache fetch failed");
            ChatError::Fetch {
                node: node.name.clone(),
                detail: e,
            }
        })?;
        tracing::debug!(node = %node.name, response = %json, "llm_cache fetch response");
        let cache_name = parse_cache_response(&provider_config.api, &json).map_err(|e| {
            tracing::warn!(node = %node.name, response = %json, "llm_cache parse failed");
            ChatError::Parse {
                node: node.name.clone(),
                detail: e,
            }
        })?;
        storage.set(node.name.clone(), Value::String(cache_name));
        Ok(())
    }

    async fn resolve_llm(
        &'a self,
        node: &'a CompiledNode,
        storage: &'a mut HashMapStorage,
        mut local: HashMap<String, Arc<Value>>,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<(), ChatError> {
        let CompiledNodeKind::Llm(llm) = &node.kind else {
            unreachable!()
        };
        let (provider, model, messages, tools, generation, cache_key, max_tokens) = (
            &llm.provider,
            &llm.model,
            &llm.messages,
            &llm.tools,
            &llm.generation,
            &llm.cache_key,
            &llm.max_tokens,
        );

        let provider_config = self
            .providers
            .get(provider)
            .ok_or_else(|| ChatError::UnknownProvider(provider.clone()))?
            .clone();
        let llm = create_llm_model(provider_config, model.clone());

        let cached_content = if let Some(expr) = cache_key {
            self.resolve_cached_content(expr, storage, bind_cache, turn_local)
                .await?
        } else {
            None
        };

        let mut segments: Vec<MessageSegment> = Vec::new();
        for msg in messages {
            match msg {
                CompiledMessage::Block(block) => {
                    let text = self
                        .render_with_deps(block, storage, local.clone(), bind_cache, turn_local)
                        .await?;
                    let message = Message::text(&block.role, text);
                    segments.push(MessageSegment::Single(message));
                }
                CompiledMessage::Iterator {
                    expr,
                    slice,
                    role,
                    token_budget,
                    ..
                } => {
                    let expanded = self
                        .expand_iterator(expr, slice, role, storage, bind_cache, turn_local)
                        .await?;
                    segments.push(MessageSegment::Iterator {
                        messages: expanded,
                        budget: token_budget.clone(),
                    });
                }
            }
        }

        self.allocate_token_budgets(&*llm, &node.name, &mut segments, *max_tokens)
            .await?;

        let mut rendered: Vec<Message> = segments
            .into_iter()
            .flat_map(|seg| match seg {
                MessageSegment::Single(m) => vec![m],
                MessageSegment::Iterator { messages, .. } => messages,
            })
            .collect();
        let specs = tool_specs(tools);
        let request = llm.build_request(&rendered, &specs, generation, cached_content.as_deref());
        tracing::debug!(node = %node.name, body = %request.body, "llm fetch request");
        let json = self.fetch.fetch(&request).await.map_err(|e| {
            tracing::warn!(node = %node.name, error = %e, "llm fetch failed");
            ChatError::Fetch {
                node: node.name.clone(),
                detail: e,
            }
        })?;
        tracing::debug!(node = %node.name, response = %json, "llm fetch response");
        let (mut response, _usage) = llm.parse_response(&json).map_err(|e| {
            tracing::warn!(node = %node.name, response = %json, "llm parse failed");
            ChatError::Parse {
                node: node.name.clone(),
                detail: e,
            }
        })?;

        let mut tool_rounds = 0usize;
        loop {
            match response {
                ModelResponse::Text(text) => {
                    tracing::debug!(node = %node.name, len = text.len(), "llm text response");
                    let output = Value::Object(BTreeMap::from([
                        ("role".into(), Value::String("assistant".into())),
                        ("content".into(), Value::String(text.clone())),
                        ("content_type".into(), Value::String("text".into())),
                    ]));
                    // IfModified: cache bind_value → output
                    if matches!(node.strategy.mode, StrategyMode::IfModified)
                        && let Some(bind_val) = local.remove("bind")
                    {
                        bind_cache
                            .entry(node.name.clone())
                            .or_default()
                            .push(((*bind_val).clone(), Arc::new(output.clone())));
                    }
                    storage.set(node.name.clone(), output);
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
                        let result_text = self
                            .execute_tool_call(
                                call, &node.name, tools, storage, bind_cache, turn_local,
                            )
                            .await?;
                        tracing::debug!(tool = %call.name, result = %result_text, "tool call result");
                        rendered.push(Message {
                            role: "tool".into(),
                            content: result_text,
                            tool_calls: Vec::new(),
                            tool_call_id: Some(call.id.clone()),
                        });
                    }

                    let request =
                        llm.build_request(&rendered, &specs, generation, cached_content.as_deref());
                    tracing::debug!(node = %node.name, body = %request.body, "llm fetch request (tool followup)");
                    let json = self.fetch.fetch(&request).await.map_err(|e| {
                        tracing::warn!(node = %node.name, error = %e, "llm fetch failed (tool followup)");
                        ChatError::Fetch { node: node.name.clone(), detail: e }
                    })?;
                    tracing::debug!(node = %node.name, response = %json, "llm fetch response (tool followup)");
                    (response, _) = llm.parse_response(&json).map_err(|e| {
                        tracing::warn!(node = %node.name, response = %json, "llm parse failed (tool followup)");
                        ChatError::Parse { node: node.name.clone(), detail: e }
                    })?;
                }
            }
        }
        Ok(())
    }

    async fn execute_tool_call(
        &'a self,
        call: &ToolCall,
        node_name: &str,
        tools: &[CompiledToolBinding],
        storage: &'a mut HashMapStorage,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<String, ChatError> {
        let binding =
            tools
                .iter()
                .find(|t| t.name == call.name)
                .ok_or_else(|| ChatError::ToolNotFound {
                    node: node_name.to_string(),
                    tool: call.name.clone(),
                })?;
        let target_idx =
            *self
                .name_to_idx
                .get(&binding.node)
                .ok_or_else(|| ChatError::ToolTargetNotFound {
                    tool: call.name.clone(),
                    target: binding.node.clone(),
                })?;

        let tool_local: HashMap<String, Arc<Value>> = match &call.arguments {
            serde_json::Value::Object(obj) => obj
                .iter()
                .map(|(k, v)| (k.clone(), Arc::new(json_to_value(v))))
                .collect(),
            _ => HashMap::new(),
        };

        self.resolve_node(target_idx, storage, tool_local, bind_cache, turn_local)
            .await?;

        let result = storage
            .get(&binding.node)
            .map(|arc| match &*arc {
                Value::String(s) => s.clone(),
                Value::Object(obj) => match obj.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    _ => format!("{obj:?}"),
                },
                other => format!("{other:?}"),
            })
            .unwrap_or_default();

        Ok(result)
    }

    async fn expand_iterator(
        &'a self,
        expr: &CompiledScript,
        slice: &Option<Vec<i64>>,
        role_override: &Option<String>,
        storage: &'a mut HashMapStorage,
        bind_cache: &'a mut HashMap<String, Vec<(Value, Arc<Value>)>>,
        turn_local: &'a mut HashMap<String, Arc<Value>>,
    ) -> Result<Vec<Message>, ChatError> {
        let evaluated = self
            .eval_script(expr, storage, bind_cache, turn_local)
            .await?;
        let all_items = match &evaluated {
            Value::List(items) => items.as_slice(),
            _ => {
                tracing::debug!("expand_iterator: not a list");
                return Ok(Vec::new());
            }
        };

        let items: &[Value] = if let Some(s) = slice {
            let len = all_items.len();
            match s.as_slice() {
                [start] => &all_items[resolve_index(*start, len)..],
                [start, end] => &all_items[resolve_index(*start, len)..resolve_index(*end, len)],
                _ => all_items,
            }
        } else {
            all_items
        };

        tracing::debug!(count = items.len(), "expand_iterator");
        let mut messages = Vec::new();
        for item in items {
            let (item_role, item_text) = item_fields(item);
            let role = role_override.as_deref().unwrap_or(item_role);
            messages.push(Message::text(role, item_text));
        }
        Ok(messages)
    }

    /// Count tokens for messages via the LlmModel. Returns None if unsupported.
    async fn count_tokens(
        &self,
        llm: &dyn LlmModel,
        node_name: &str,
        messages: &[Message],
    ) -> Result<Option<u32>, ChatError> {
        if messages.is_empty() {
            return Ok(Some(0));
        }
        let request = match llm.build_count_tokens_request(messages) {
            Some(r) => r,
            None => return Ok(None),
        };
        let json = self
            .fetch
            .fetch(&request)
            .await
            .map_err(|e| ChatError::TokenCount {
                node: node_name.to_string(),
                detail: e,
            })?;
        let count = llm
            .parse_count_tokens_response(&json)
            .map_err(|e| ChatError::TokenCount {
                node: node_name.to_string(),
                detail: e,
            })?;
        Ok(Some(count))
    }

    /// Allocate token budgets across budgeted iterator segments.
    ///
    /// Algorithm:
    /// 1. Count tokens for each budgeted iterator
    /// 2. Reserve `request` tokens for each that has one
    /// 3. Distribute remaining pool by priority (0 first)
    /// 4. Trim each iterator to its allocated budget
    async fn allocate_token_budgets(
        &self,
        llm: &dyn LlmModel,
        node_name: &str,
        segments: &mut [MessageSegment],
        total_budget: Option<u32>,
    ) -> Result<(), ChatError> {
        // Collect budgeted iterators: (segment index, budget, token count)
        let mut budgeted: Vec<(usize, TokenBudget, u32)> = Vec::new();
        for (i, seg) in segments.iter().enumerate() {
            if let MessageSegment::Iterator {
                messages,
                budget: Some(budget),
            } = seg
            {
                let count = match self.count_tokens(llm, node_name, messages).await? {
                    Some(c) => c,
                    None => {
                        tracing::warn!(node = %node_name, "count tokens not supported, skipping budget allocation");
                        return Ok(());
                    }
                };
                budgeted.push((i, budget.clone(), count));
            }
        }

        if budgeted.is_empty() {
            return Ok(());
        }

        // If no total budget, only apply individual limits
        let Some(total) = total_budget else {
            for (seg_idx, budget, actual) in &budgeted {
                if let Some(limit) = budget.max
                    && *actual > limit
                {
                    trim_segment(&mut segments[*seg_idx], *actual, limit, node_name);
                }
            }
            return Ok(());
        };

        // Reserve pool
        let reserved: u32 = budgeted.iter().filter_map(|(_, b, _)| b.min).sum();
        let mut pool = total.saturating_sub(reserved);

        // Sort by priority ascending (0 = highest priority, fills first)
        budgeted.sort_by_key(|(_, b, _)| b.priority);

        for (seg_idx, budget, actual) in &budgeted {
            let available = pool + budget.min.unwrap_or(0);
            let cap = budget.max.map(|l| available.min(l)).unwrap_or(available);
            let allocated = (*actual).min(cap);
            let consumed_from_pool = allocated.saturating_sub(budget.min.unwrap_or(0));
            pool = pool.saturating_sub(consumed_from_pool);

            if *actual > allocated {
                trim_segment(&mut segments[*seg_idx], *actual, allocated, node_name);
            }
        }

        Ok(())
    }
}

enum MessageSegment {
    Single(Message),
    Iterator {
        messages: Vec<Message>,
        budget: Option<TokenBudget>,
    },
}

fn trim_segment(
    segment: &mut MessageSegment,
    actual_tokens: u32,
    target_tokens: u32,
    node_name: &str,
) {
    let messages = match segment {
        MessageSegment::Iterator { messages, .. } => messages,
        _ => return,
    };
    if messages.is_empty() {
        return;
    }
    let len = messages.len();
    let per_message = actual_tokens / len as u32;
    let keep = if per_message > 0 {
        (target_tokens / per_message) as usize
    } else {
        len
    };
    let keep = keep.max(1).min(len);
    let skip = len - keep;

    tracing::debug!(
        node = %node_name,
        actual_tokens = actual_tokens,
        target_tokens = target_tokens,
        original = len,
        kept = keep,
        "trimmed iterator messages",
    );

    *messages = messages.split_off(skip);
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
    bind_cache: HashMap<String, Vec<(Value, Arc<Value>)>>,
    entrypoint_idx: usize,
    history_nodes: Vec<(String, CompiledHistory)>,
    turn_count: usize,
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

        // Collect history nodes
        let history_nodes: Vec<(String, CompiledHistory)> = nodes
            .iter()
            .filter_map(|n| n.history.as_ref().map(|h| (n.name.clone(), h.clone())))
            .collect();

        // Validate: history + IfModified is not allowed
        for (name, _) in &history_nodes {
            let idx = name_to_idx[name];
            if matches!(nodes[idx].strategy.mode, StrategyMode::IfModified) {
                return Err(ChatError::HistoryNodeIfModified(name.clone()));
            }
        }

        // Validate: history nodes must be reachable from entrypoint via BFS
        // Tool targets are implicitly reachable (called on demand), so we
        // collect all tool target nodes as well.
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(entrypoint_idx);
        reachable.insert(entrypoint_idx);
        while let Some(idx) = queue.pop_front() {
            for key in &nodes[idx].all_context_keys {
                if let Some(&dep_idx) = name_to_idx.get(key)
                    && reachable.insert(dep_idx)
                {
                    queue.push_back(dep_idx);
                }
            }
            // Tool targets are also reachable
            if let CompiledNodeKind::Llm(llm) = &nodes[idx].kind {
                for tool in &llm.tools {
                    if let Some(&dep_idx) = name_to_idx.get(&tool.node)
                        && reachable.insert(dep_idx)
                    {
                        queue.push_back(dep_idx);
                    }
                }
            }
        }
        for (name, _) in &history_nodes {
            if !reachable.contains(&name_to_idx[name]) {
                return Err(ChatError::HistoryNodeUnreachable(name.clone()));
            }
        }

        // Seed context metadata as nested Object: context.{node}.{model,provider}
        let mut context_obj: BTreeMap<String, Value> = BTreeMap::new();
        for node in &nodes {
            if let CompiledNodeKind::Llm(llm) = &node.kind {
                context_obj.insert(
                    node.name.clone(),
                    Value::Object(BTreeMap::from([
                        ("model".into(), Value::String(llm.model.clone())),
                        ("provider".into(), Value::String(llm.provider.clone())),
                    ])),
                );
            }
        }
        if !context_obj.is_empty() {
            storage.set("context".into(), Value::Object(context_obj));
        }

        // Seed history as Object with per-node empty lists
        if !history_nodes.is_empty() {
            let mut history_obj = BTreeMap::new();
            for (name, _) in &history_nodes {
                history_obj.insert(name.clone(), Value::List(Vec::new()));
            }
            storage.set("history".into(), Value::Object(history_obj));
            storage.set("index".into(), Value::Int(0));
        }

        Ok(Self {
            nodes,
            name_to_idx,
            providers,
            fetch,
            extern_fns,
            storage,
            bind_cache: HashMap::new(),
            entrypoint_idx,
            history_nodes,
            turn_count: 0,
        })
    }

    pub async fn turn<R>(&mut self, resolver: &R) -> Result<Value, ChatError>
    where
        R: AsyncFn(String) -> Value + Sync,
    {
        let entrypoint = &self.nodes[self.entrypoint_idx].name;
        tracing::debug!(entrypoint = %entrypoint, "turn start");

        // Update @index to current turn count
        self.storage
            .set("index".into(), Value::Int(self.turn_count as i64));

        // Always-strategy nodes re-resolve every turn
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
            &mut self.bind_cache,
            &mut turn_local,
        )
        .await?;

        // Evaluate store expressions and append to history
        for (name, history) in &self.history_nodes {
            let interp = Interpreter::new(history.store.module.clone(), &self.extern_fns);
            let (mut coroutine, key) = interp.execute();
            let result = drive_script(
                &mut coroutine,
                key,
                &self.storage,
                &HashMap::new(),
                &turn_local,
            )?;
            let value = match result {
                ScriptDriveResult::Value(v) => v,
                ScriptDriveResult::NeedContext(_) => continue,
            };
            if let Some(Value::Object(obj)) = self.storage.get_mut("history")
                && let Some(Value::List(list)) = obj.get_mut(name)
            {
                list.push(value);
            }
        }

        self.turn_count += 1;

        let name = &self.nodes[self.entrypoint_idx].name;
        let result = self
            .storage
            .get(name)
            .ok_or_else(|| ChatError::UnresolvedContext(name.clone()))?;
        Ok(Value::clone(&result))
    }

    pub fn history_len(&self) -> usize {
        self.turn_count
    }

    pub fn history_pop(&mut self) {
        if self.turn_count == 0 {
            return;
        }
        self.turn_count -= 1;
        if let Some(Value::Object(obj)) = self.storage.get_mut("history") {
            for (name, _) in &self.history_nodes {
                if let Some(Value::List(list)) = obj.get_mut(name) {
                    list.pop();
                }
            }
        }
    }

    pub async fn re_execute<R>(&mut self, index: usize, resolver: &R) -> Result<Value, ChatError>
    where
        R: AsyncFn(String) -> Value + Sync,
    {
        assert!(
            index <= self.turn_count,
            "re_execute index out of bounds: {index} > {}",
            self.turn_count,
        );
        // Truncate all history nodes to `index` turns (1 entry per turn)
        if let Some(Value::Object(obj)) = self.storage.get_mut("history") {
            for (name, _) in &self.history_nodes {
                if let Some(Value::List(list)) = obj.get_mut(name) {
                    list.truncate(index);
                }
            }
        }
        self.turn_count = index;
        self.turn(resolver).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::extern_module::ExternRegistry;
    use acvus_orchestration::{
        ApiKind, GenerationParams, HttpRequest, LlmSpec, MessageSpec, NodeKind, NodeSpec,
        PlainSpec, Strategy, ToolBinding, compile_nodes,
    };

    // -- MockFetch: returns queued JSON responses in order -----------------------

    struct MockFetch {
        responses: Mutex<Vec<serde_json::Value>>,
    }

    impl MockFetch {
        fn new(responses: Vec<serde_json::Value>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
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
            kind: NodeKind::Plain(PlainSpec {
                source: "hello".into(),
            }),
            strategy: Strategy::default(),
            history: None,
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
            kind: NodeKind::Plain(PlainSpec {
                source: "hello".into(),
            }),
            strategy: Strategy::default(),
            history: None,
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
            kind: NodeKind::Plain(PlainSpec {
                source: "hello world".into(),
            }),
            strategy: Strategy::default(),
            history: None,
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
        assert_eq!(result, Value::String("hello world".into()));
    }

    #[tokio::test]
    async fn turn_llm_text_response() {
        let nodes = compile_test_nodes(&[NodeSpec {
            name: "main".into(),
            kind: NodeKind::Llm(LlmSpec {
                provider: "test".into(),
                model: "gpt-test".into(),
                messages: vec![MessageSpec::Block {
                    role: "user".into(),
                    source: "hi".into(),
                }],
                tools: vec![],
                generation: GenerationParams::default(),
                cache_key: None,
                max_tokens: None,
            }),
            strategy: Strategy::default(),
            history: None,
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
        let Value::Object(obj) = &result else {
            panic!("expected Object, got {result:?}")
        };
        assert_eq!(obj.get("role"), Some(&Value::String("assistant".into())));
        assert_eq!(
            obj.get("content"),
            Some(&Value::String("hello from LLM".into()))
        );
        assert_eq!(obj.get("content_type"), Some(&Value::String("text".into())));
    }

    #[tokio::test]
    async fn turn_tool_call_round_trip() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain(PlainSpec {
                    source: "tool result text".into(),
                }),
                strategy: Strategy::default(),
                history: None,
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm(LlmSpec {
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
                    max_tokens: None,
                }),
                strategy: Strategy::default(),
                history: None,
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
        let Value::Object(obj) = &result else {
            panic!("expected Object, got {result:?}")
        };
        assert_eq!(
            obj.get("content"),
            Some(&Value::String("final answer".into()))
        );
    }

    #[tokio::test]
    async fn turn_tool_not_found() {
        let nodes = compile_test_nodes(&[
            NodeSpec {
                name: "tool_target".into(),
                kind: NodeKind::Plain(PlainSpec {
                    source: "result".into(),
                }),
                strategy: Strategy::default(),
                history: None,
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm(LlmSpec {
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
                    max_tokens: None,
                }),
                strategy: Strategy::default(),
                history: None,
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
                kind: NodeKind::Plain(PlainSpec {
                    source: "result".into(),
                }),
                strategy: Strategy::default(),
                history: None,
            },
            NodeSpec {
                name: "main".into(),
                kind: NodeKind::Llm(LlmSpec {
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
                    max_tokens: None,
                }),
                strategy: Strategy::default(),
                history: None,
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
        assert!(matches!(
            json_to_value(&serde_json::json!(null)),
            Value::Unit
        ));
        assert!(matches!(
            json_to_value(&serde_json::json!(true)),
            Value::Bool(true)
        ));
        assert!(matches!(
            json_to_value(&serde_json::json!(42)),
            Value::Int(42)
        ));
        assert!(
            matches!(json_to_value(&serde_json::json!(3.14)), Value::Float(f) if (f - 3.14).abs() < f64::EPSILON)
        );
        assert!(
            matches!(json_to_value(&serde_json::json!("hello")), Value::String(s) if s == "hello")
        );
        assert!(
            matches!(json_to_value(&serde_json::json!([1, 2])), Value::List(v) if v.len() == 2)
        );
        assert!(
            matches!(json_to_value(&serde_json::json!({"key": "val"})), Value::Object(m) if m.contains_key("key"))
        );
    }
}
