use std::sync::Arc;

use acvus_interpreter::{Interpreter, RuntimeError, Stepped, Value};
use acvus_utils::{Astr, Interner, YieldHandle};
use rustc_hash::FxHashMap;

use crate::compile::CompiledScript;
use crate::dsl::TokenBudget;
use crate::kind::CompiledToolBinding;
use crate::message::{Content, Message, ToolSpec};
use crate::provider::{ApiKind, Fetch};

pub async fn render_block_in_coroutine(
    interner: &Interner,
    module: &acvus_mir::ir::MirModule,
    local: &FxHashMap<Astr, Arc<Value>>,
    handle: &YieldHandle<Value>,
) -> Result<String, RuntimeError> {
    let interp = Interpreter::new(interner, module.clone());
    let mut inner = interp.execute();
    let mut output = String::new();
    loop {
        match inner.resume().await {
            Stepped::Emit(value) => {
                let Value::String(s) = value else {
                    panic!("render_block: expected String, got {value:?}");
                };
                output.push_str(&s);
            }
            Stepped::NeedContext(request) => {
                let name = request.name();
                if let Some(arc) = local.get(&name) {
                    request.resolve(Arc::clone(arc));
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
    local: &FxHashMap<Astr, Arc<Value>>,
    handle: &YieldHandle<Value>,
) -> Result<Value, RuntimeError> {
    let interp = Interpreter::new(interner, module.clone());
    let mut inner = interp.execute();
    loop {
        match inner.resume().await {
            Stepped::Emit(value) => {
                return Ok(value);
            }
            Stepped::NeedContext(request) => {
                let name = request.name();
                if let Some(arc) = local.get(&name) {
                    request.resolve(Arc::clone(arc));
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
            Stepped::Done => return Ok(Value::Unit),
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
    api: &ApiKind,
    expr: &CompiledScript,
    slice: &Option<Vec<i64>>,
    role_override: &Option<Astr>,
    interner: &Interner,
    local: &FxHashMap<Astr, Arc<Value>>,
    handle: &YieldHandle<Value>,
) -> Result<Vec<Message>, RuntimeError> {
    let evaluated = eval_script_in_coroutine(interner, &expr.module, local, handle).await?;

    let deque_vec;
    let all_items = match &evaluated {
        Value::List(items) => items.as_slice(),
        Value::Deque(deque) => {
            deque_vec = deque.as_slice();
            deque_vec
        }
        _ => return Ok(Vec::new()),
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
        let (part_role, part_text, part_content_type) = api.item_fields(interner, item);
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

pub fn content_to_value(interner: &Interner, items: &[crate::message::ContentItem]) -> Value {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");
    let content_type_key = interner.intern("content_type");
    let values: Vec<Value> = items
        .iter()
        .map(|item| {
            let (content_str, content_type_str) = match &item.content {
                Content::Text(s) => (s.clone(), "text".to_string()),
                Content::Blob { mime_type, data } => (data.clone(), mime_type.clone()),
            };
            Value::Object(FxHashMap::from_iter([
                (role_key, Value::String(item.role.clone())),
                (content_key, Value::String(content_str)),
                (content_type_key, Value::String(content_type_str)),
            ]))
        })
        .collect();
    Value::List(values)
}

pub fn value_to_tool_result(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(obj) => {
            // Try to find "content" key by iterating
            for v in obj.values() {
                if let Value::String(s) = v {
                    return s.clone();
                }
            }
            format!("{obj:?}")
        }
        other => format!("{other:?}"),
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
                .map(|(k, v)| (k.clone(), ty_to_json_schema(v).to_string()))
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
        _ => "string",
    }
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

pub async fn allocate_token_budgets<F>(
    llm: &crate::provider::LlmModelKind,
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
            let count = match count_tokens(llm, fetch, messages).await {
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

    let fixed_tokens = match count_tokens(llm, fetch, &fixed_messages).await {
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
    llm: &crate::provider::LlmModelKind,
    fetch: &F,
    messages: &[Message],
) -> Option<u32>
where
    F: Fetch,
{
    if messages.is_empty() {
        return Some(0);
    }
    let request = llm.build_count_tokens_request(messages)?;
    let json = fetch.fetch(&request).await.ok()?;
    llm.parse_count_tokens_response(&json).ok()
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
