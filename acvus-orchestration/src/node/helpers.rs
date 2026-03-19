use acvus_interpreter::{Interpreter, LazyValue, PureValue, RuntimeError, Stepped, TypedValue, Value, ValueKind};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Interner, YieldHandle};
use rustc_hash::FxHashMap;

use crate::compile::CompiledScript;
use crate::dsl::TokenBudget;
use crate::http::{Fetch, RequestError};
use crate::spec::CompiledToolBinding;
use crate::message::{Content, Message, ToolSpec, ToolSpecParam};
use crate::dsl::{MSG_ROLE, MSG_CONTENT, MSG_CONTENT_TYPE};

// ── Helpers moved from provider/mod.rs ──────────────────────────────

/// Extract item fields from a message value object.
///
/// All providers use the same message element shape: `{role, content, content_type}`.
pub fn item_fields<'a>(
    interner: &Interner,
    item: &'a Value,
) -> Result<(&'a str, &'a str, &'a str), RequestError> {
    let Value::Lazy(LazyValue::Object(obj)) = item else {
        return Err(RequestError::ResponseParse {
            detail: format!("item_fields: expected Object, got {item:?}"),
        });
    };
    let role_key = interner.intern(MSG_ROLE);
    let content_key = interner.intern(MSG_CONTENT);
    let content_type_key = interner.intern(MSG_CONTENT_TYPE);
    let Some(Value::Pure(PureValue::String(role))) = obj.get(&role_key) else {
        return Err(RequestError::MissingField { field: "role" });
    };
    let Some(Value::Pure(PureValue::String(content))) = obj.get(&content_key) else {
        return Err(RequestError::MissingField { field: "content" });
    };
    let Some(Value::Pure(PureValue::String(content_type))) = obj.get(&content_type_key) else {
        return Err(RequestError::MissingField { field: "content_type" });
    };
    Ok((role.as_str(), content.as_str(), content_type.as_str()))
}

/// Split system messages out of a message list.
///
/// Returns `(system_text, non_system_messages)` where system_text is the
/// concatenation of all system message texts (joined by newline), or `None`
/// if there are no system messages.
///
/// Returns `RequestError` if a system message contains non-text content.
pub fn split_system_messages(messages: &[Message]) -> Result<(Option<String>, Vec<&Message>), RequestError> {
    let mut system_text = String::new();
    let mut rest = Vec::new();

    for m in messages {
        if let Message::Content { role, content } = m
            && role == "system"
        {
            let Content::Text(text) = content else {
                return Err(RequestError::ResponseParse {
                    detail: "system message must be text, got blob".into(),
                });
            };
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(text);
        } else {
            rest.push(m);
        }
    }

    let system = if system_text.is_empty() {
        None
    } else {
        Some(system_text)
    };
    Ok((system, rest))
}

// ── Existing helpers ────────────────────────────────────────────────

pub async fn render_block_in_coroutine(
    interner: &Interner,
    module: &acvus_mir::ir::MirModule,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &YieldHandle<TypedValue>,
) -> Result<String, RuntimeError> {
    let interp = Interpreter::new(interner, module.clone());
    let mut inner = interp.execute();
    let mut output = String::new();
    loop {
        match inner.resume().await {
            Stepped::Emit(value) => {
                let Value::Pure(PureValue::String(s)) = value.value() else {
                    return Err(RuntimeError::unexpected_type(
                        "render_block",
                        &[ValueKind::String],
                        value.value().kind(),
                    ));
                };
                output.push_str(s);
            }
            Stepped::NeedContext(request) => {
                let name = request.name();
                if let Some(val) = local.get(&name) {
                    request.resolve(val.clone());
                } else {
                    let value = handle.request_context(name).await;
                    request.resolve(value);
                }
            }
            Stepped::NeedExternCall(request) => {
                let value = handle
                    .request_extern_call(request.name(), request.args().to_vec())
                    .await;
                request.resolve(value);
            }
            Stepped::Done => return Ok(output),
            Stepped::Error(e) => return Err(e),
        }
    }
}

pub async fn eval_script_in_coroutine(
    interner: &Interner,
    module: &acvus_mir::ir::MirModule,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &YieldHandle<TypedValue>,
) -> Result<TypedValue, RuntimeError> {
    let interp = Interpreter::new(interner, module.clone());
    let mut inner = interp.execute();
    loop {
        match inner.resume().await {
            Stepped::Emit(value) => {
                return Ok(value);
            }
            Stepped::NeedContext(request) => {
                let name = request.name();
                if let Some(val) = local.get(&name) {
                    request.resolve(val.clone());
                } else {
                    let value = handle.request_context(name).await;
                    request.resolve(value);
                }
            }
            Stepped::NeedExternCall(request) => {
                let value = handle
                    .request_extern_call(request.name(), request.args().to_vec())
                    .await;
                request.resolve(value);
            }
            Stepped::Done => return Ok(TypedValue::unit()),
            Stepped::Error(e) => return Err(e),
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

pub async fn expand_iterator_in_coroutine(
    expr: &CompiledScript,
    slice: &Option<Vec<i64>>,
    role_override: &Option<Astr>,
    interner: &Interner,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &YieldHandle<TypedValue>,
) -> Result<Vec<Message>, RuntimeError> {
    let evaluated = eval_script_in_coroutine(interner, &expr.module, local, handle).await?;

    let deque_vec;
    let all_items = match evaluated.value() {
        Value::Lazy(LazyValue::List(items)) => items.as_slice(),
        Value::Lazy(LazyValue::Deque(deque)) => {
            deque_vec = deque.as_slice();
            deque_vec
        }
        other => {
            return Err(RuntimeError::unexpected_type(
                "expand_iterator",
                &[ValueKind::List, ValueKind::Deque],
                other.kind(),
            ));
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

    let role_str = role_override.map(|r| interner.resolve(r).to_string());
    let mut messages = Vec::new();
    for item in items {
        let (part_role, part_text, part_content_type) = item_fields(interner, item)
            .map_err(|e| RuntimeError::fetch(e.to_string()))?;
        let role = role_str.as_deref().unwrap_or(part_role);
        let content = if part_content_type == "text" {
            Content::Text(part_text.to_string())
        } else {
            Content::Blob {
                mime_type: part_content_type.to_string(),
                data: part_text.to_string(),
            }
        };
        messages.push(Message::Content {
            role: role.to_string(),
            content,
        });
    }
    Ok(messages)
}

pub fn content_to_value(interner: &Interner, items: &[crate::message::ContentItem]) -> TypedValue {
    let role_key = interner.intern(MSG_ROLE);
    let content_key = interner.intern(MSG_CONTENT);
    let content_type_key = interner.intern(MSG_CONTENT_TYPE);
    let values: Vec<Value> = items
        .iter()
        .map(|item| {
            let (content_ref, type_ref): (&str, &str) = match &item.content {
                Content::Text(s) => (s.as_str(), "text"),
                Content::Blob { mime_type, data } => (data.as_str(), mime_type.as_str()),
            };
            Value::object(FxHashMap::from_iter([
                (role_key, Value::string(item.role.as_str())),
                (content_key, Value::string(content_ref)),
                (content_type_key, Value::string(type_ref)),
            ]))
        })
        .collect();
    let elem_ty = crate::dsl::message_elem_ty(interner);
    TypedValue::new(Value::list(values), Ty::List(Box::new(elem_ty)))
}

pub fn value_to_tool_result(value: &TypedValue, interner: &Interner) -> String {
    match value.value() {
        Value::Pure(PureValue::String(s)) => s.clone(),
        Value::Pure(PureValue::Int(n)) => n.to_string(),
        Value::Pure(PureValue::Float(f)) => f.to_string(),
        Value::Pure(PureValue::Bool(b)) => b.to_string(),
        Value::Pure(PureValue::Unit) => "null".to_string(),
        _ => {
            // Structured values -> JSON serialization for deterministic output.
            let concrete = value.to_concrete(interner);
            serde_json::to_string(&concrete).unwrap_or_else(|_| format!("{concrete:?}"))
        }
    }
}

pub fn make_tool_specs(tools: &[CompiledToolBinding]) -> Vec<ToolSpec> {
    tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: t.description.clone(),
            params: t
                .params
                .iter()
                .map(|(k, v)| (k.clone(), ToolSpecParam {
                    ty: ty_to_json_schema(&v.ty).to_string(),
                    description: v.description.clone(),
                }))
                .collect(),
        })
        .collect()
}

fn ty_to_json_schema(ty: &acvus_mir::ty::Ty) -> &'static str {
    use acvus_mir::ty::Ty;
    match ty {
        Ty::String => "string",
        Ty::Int => "integer",
        Ty::Float => "number",
        Ty::Bool => "boolean",
        Ty::Object(_) => "object",
        Ty::List(_) | Ty::Deque(..) => "array",
        // Fallback: types that don't map to JSON schema (e.g. Fn, Iterator)
        // are represented as "string" since tool params are ultimately serialized
        // as text for the LLM.
        _ => "string",
    }
}

// ── Shared spawn helpers ──────────────────────────────────────────

/// Render all compiled messages into message segments.
///
/// This is the identical message rendering loop used by all 3 LLM providers
/// (OpenAI, Anthropic, Google). Each `CompiledMessage` is either a template
/// block (rendered via the interpreter) or an iterator expression (expanded
/// into a list of messages with optional token budget).
pub async fn render_messages(
    messages: &[crate::compile::CompiledMessage],
    interner: &Interner,
    local: &FxHashMap<Astr, TypedValue>,
    handle: &YieldHandle<TypedValue>,
) -> Result<Vec<MessageSegment>, RuntimeError> {
    let mut segments = Vec::new();
    for msg in messages {
        match msg {
            crate::compile::CompiledMessage::Block(block) => {
                let text = render_block_in_coroutine(
                    interner,
                    &block.module,
                    local,
                    handle,
                )
                .await?;
                segments.push(MessageSegment::Single(Message::Content {
                    role: interner.resolve(block.role).to_string(),
                    content: Content::Text(text),
                }));
            }
            crate::compile::CompiledMessage::Iterator {
                expr,
                slice,
                role,
                token_budget,
            } => {
                let expanded = expand_iterator_in_coroutine(
                    expr,
                    slice,
                    role,
                    interner,
                    local,
                    handle,
                )
                .await?;
                segments.push(MessageSegment::Iterator {
                    messages: expanded,
                    budget: token_budget.clone(),
                });
            }
        }
    }
    Ok(segments)
}

/// Execute tool calls by dispatching each call to its corresponding tool node.
///
/// This is the identical tool execution loop used by all 3 LLM providers.
/// For each `ToolCall`, it finds the matching `CompiledToolBinding`, constructs
/// the typed argument object, invokes the tool node via `request_extern_call`,
/// and serialises the result. Unknown tools produce a "not found" message.
pub async fn execute_tool_calls(
    calls: &[crate::message::ToolCall],
    tools: &[CompiledToolBinding],
    interner: &Interner,
    handle: &YieldHandle<TypedValue>,
) -> Vec<Message> {
    let mut results = Vec::new();
    for call in calls {
        tracing::debug!(tool = %call.name, id = %call.id, "invoking tool");
        let binding = tools.iter().find(|t| t.name == call.name);
        let result_text = if let Some(binding) = binding {
            let tool_args: FxHashMap<Astr, Value> = match &call.arguments {
                serde_json::Value::Object(obj) => obj
                    .iter()
                    .map(|(k, v)| {
                        (
                            interner.intern(k),
                            crate::convert::json_to_value(interner, v),
                        )
                    })
                    .collect(),
                _ => FxHashMap::default(),
            };
            let tool_obj_ty = Ty::Object(
                binding.params.iter()
                    .map(|(k, p)| (interner.intern(k), p.ty.clone()))
                    .collect(),
            );
            let tool_value = TypedValue::new(Value::object(tool_args), tool_obj_ty);
            let result = handle
                .request_extern_call(
                    interner.intern(&binding.node),
                    vec![tool_value],
                )
                .await;
            value_to_tool_result(&result, interner)
        } else {
            tracing::warn!(tool = %call.name, "tool not found");
            format!("tool '{}' not found", call.name)
        };

        results.push(Message::ToolResult {
            call_id: call.id.clone(),
            content: result_text,
        });
    }
    results
}

pub enum MessageSegment {
    Single(Message),
    Iterator {
        messages: Vec<Message>,
        budget: Option<TokenBudget>,
    },
}

pub fn flatten_segments(segments: Vec<MessageSegment>) -> Vec<Message> {
    segments
        .into_iter()
        .flat_map(|seg| match seg {
            MessageSegment::Single(m) => vec![m],
            MessageSegment::Iterator { messages, .. } => messages,
        })
        .collect()
}

/// Allocate token budgets using provider-specific count_tokens API.
///
/// `provider_kind` determines which count_tokens API to use:
/// - "anthropic" -> Anthropic count_tokens
/// - "google" -> Google countTokens
/// - others -> no-op (count_tokens not supported)
pub async fn allocate_token_budgets<F>(
    endpoint: &str,
    api_key: &str,
    model: &str,
    provider_kind: &str,
    fetch: &F,
    segments: &mut [MessageSegment],
    total_budget: Option<u32>,
) where
    F: Fetch,
{
    let mut budgeted: Vec<(usize, TokenBudget, u32)> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if let MessageSegment::Iterator {
            messages,
            budget: Some(budget),
        } = seg
        {
            let count = match count_tokens(endpoint, api_key, model, provider_kind, fetch, messages).await {
                Some(c) => c,
                None => return,
            };
            budgeted.push((i, budget.clone(), count));
        }
    }

    if budgeted.is_empty() {
        return;
    }

    let Some(total) = total_budget else {
        for (seg_idx, budget, actual) in &budgeted {
            if let Some(limit) = budget.max
                && *actual > limit
            {
                trim_segment(&mut segments[*seg_idx], *actual, limit);
            }
        }
        return;
    };

    let budgeted_indices: std::collections::HashSet<usize> =
        budgeted.iter().map(|(i, _, _)| *i).collect();
    let mut fixed_messages: Vec<Message> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if budgeted_indices.contains(&i) {
            continue;
        }
        match seg {
            MessageSegment::Single(m) => fixed_messages.push(m.clone()),
            MessageSegment::Iterator { messages, .. } => {
                fixed_messages.extend(messages.iter().cloned());
            }
        }
    }

    let fixed_tokens = match count_tokens(endpoint, api_key, model, provider_kind, fetch, &fixed_messages).await {
        Some(c) => c,
        None => return,
    };

    let remaining = total.saturating_sub(fixed_tokens);
    let reserved: u32 = budgeted.iter().filter_map(|(_, b, _)| b.min).sum();
    let mut pool = remaining.saturating_sub(reserved);

    budgeted.sort_by_key(|(_, b, _)| b.priority);

    for (seg_idx, budget, actual) in &budgeted {
        let available = pool + budget.min.unwrap_or(0);
        let cap = budget.max.map(|l| available.min(l)).unwrap_or(available);
        let allocated = (*actual).min(cap);
        let consumed_from_pool = allocated.saturating_sub(budget.min.unwrap_or(0));
        pool = pool.saturating_sub(consumed_from_pool);

        if *actual > allocated {
            trim_segment(&mut segments[*seg_idx], *actual, allocated);
        }
    }
}

async fn count_tokens<F>(
    endpoint: &str,
    api_key: &str,
    model: &str,
    provider_kind: &str,
    fetch: &F,
    messages: &[Message],
) -> Option<u32>
where
    F: Fetch,
{
    if messages.is_empty() {
        return Some(0);
    }
    let request = match provider_kind {
        "anthropic" => {
            super::anthropic::build_count_tokens_request(endpoint, api_key, model, messages).ok()?
        }
        "google" => {
            super::google::build_count_tokens_request(endpoint, api_key, model, messages).ok()?
        }
        _ => return None,
    };
    let json = match fetch.fetch(&request).await {
        Ok(j) => j,
        Err(e) => { tracing::debug!("count_tokens fetch failed: {e}"); return None; }
    };
    let result = match provider_kind {
        "anthropic" => super::anthropic::parse_count_tokens_response(&json),
        "google" => super::google::parse_count_tokens_response(&json),
        _ => return None,
    };
    match result {
        Ok(n) => Some(n),
        Err(e) => { tracing::debug!("count_tokens parse failed: {e}"); None }
    }
}

fn trim_segment(segment: &mut MessageSegment, actual_tokens: u32, target_tokens: u32) {
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
    *messages = messages.split_off(skip);
}
