use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use acvus_ast::Literal;
use acvus_interpreter::{
    Coroutine, ExternFnRegistry, Interpreter, NeedContextStepped, ResumeKey, Stepped, Value,
};
use acvus_mir::extern_module::ExternRegistry;
use acvus_mir::ir::MirModule;
use acvus_mir_pass::analysis::reachable_context::reachable_context_keys;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};

use crate::compile::{CompiledMessage, CompiledNode};
use crate::dag::Dag;
use crate::error::{OrchError, OrchErrorKind};
use crate::message::{Message, ModelResponse, Output, ToolCall, ToolResult, ToolSpec};
use crate::provider::{build_request, parse_response, Fetch, ProviderConfig};
use crate::storage::Storage;

pub struct Executor<F, S>
where
    F: Fetch,
    S: Storage,
{
    nodes: Vec<CompiledNode>,
    dag: Dag,
    storage: S,
    fetch: F,
    providers: HashMap<String, ProviderConfig>,
    mir_registry: ExternRegistry,
    fuel_limit: u64,
}

impl<F, S> Executor<F, S>
where
    F: Fetch + 'static,
    S: Storage,
{
    pub fn new(
        nodes: Vec<CompiledNode>,
        dag: Dag,
        storage: S,
        fetch: F,
        providers: HashMap<String, ProviderConfig>,
        mir_registry: ExternRegistry,
        fuel_limit: u64,
    ) -> Self {
        Self { nodes, dag, storage, fetch, providers, mir_registry, fuel_limit }
    }

    /// Run the full DAG with demand-driven rendering.
    ///
    /// Template rendering is sync (coroutine resume loop). When a template
    /// needs a context value that isn't available yet, the node blocks until
    /// the dependency's model call completes. Model calls are the only async
    /// operations, driven via `FuturesUnordered`.
    pub async fn run(self) -> Result<S, OrchError> {
        let Executor { nodes, dag, mut storage, fetch, providers, fuel_limit, .. } = self;
        let fetch = Arc::new(fetch);
        let fuel = Arc::new(AtomicU64::new(0));
        let n = nodes.len();

        let mut launched = vec![false; n];
        let mut completed = vec![false; n];
        let mut node_states: Vec<Option<NodeRunState>> = (0..n).map(|_| None).collect();

        let name_to_idx: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.name.as_str(), i))
            .collect();

        let mut running: FuturesUnordered<BoxFuture<'static, Result<(usize, Output), OrchError>>> =
            FuturesUnordered::new();

        // Launch initially ready nodes (prefetch via static analysis)
        for i in 0..n {
            if is_node_ready(i, &nodes, &dag, &storage, &completed) {
                launched[i] = true;
                node_states[i] = Some(make_node_state(&nodes[i]));
            }
        }

        loop {
            // 1. Drive rendering (sync) — all non-blocked nodes
            loop {
                let mut progress = false;
                let mut to_submit: Vec<usize> = Vec::new();
                let mut to_launch: Vec<usize> = Vec::new();

                for i in 0..n {
                    let state = match &mut node_states[i] {
                        Some(s) if s.waiting_for.is_none() => s,
                        _ => continue,
                    };
                    match drive_rendering(state, &storage) {
                        DriveResult::AllRendered => {
                            to_submit.push(i);
                            progress = true;
                        }
                        DriveResult::Blocked(dep_name) => {
                            if let Some(&dep_idx) = name_to_idx.get(dep_name.as_str()) {
                                if !launched[dep_idx] {
                                    to_launch.push(dep_idx);
                                    progress = true;
                                }
                            }
                        }
                    }
                }

                for i in to_submit {
                    let state = node_states[i].take().unwrap();
                    let provider_config = providers
                        .get(&nodes[i].provider)
                        .ok_or_else(|| {
                            OrchError::new(OrchErrorKind::ModelError(format!(
                                "unknown provider: {}",
                                nodes[i].provider
                            )))
                        })?
                        .clone();
                    running.push(Box::pin(call_model(
                        i,
                        nodes[i].clone(),
                        state.rendered_messages,
                        provider_config,
                        Arc::clone(&fetch),
                        Arc::clone(&fuel),
                        fuel_limit,
                    )));
                }

                for i in to_launch {
                    launched[i] = true;
                    node_states[i] = Some(make_node_state(&nodes[i]));
                }

                if !progress {
                    break;
                }
            }

            // 2. Done check
            if running.is_empty() {
                break;
            }

            // 3. Await model completion
            let result = running.next().await;
            let (idx, output) = match result {
                Some(Ok(r)) => r,
                Some(Err(e)) => return Err(e),
                None => break,
            };
            storage.set(nodes[idx].name.clone(), output);
            completed[idx] = true;

            // 4. Unblock waiting nodes
            let completed_name = &nodes[idx].name;
            for i in 0..n {
                if let Some(state) = &mut node_states[i] {
                    if state.waiting_for.as_deref() == Some(completed_name.as_str()) {
                        let need = state.pending_need.take().unwrap();
                        let output_val = storage.get(completed_name).unwrap();
                        state.resume_key = Some(need.into_key(output_to_value(&output_val)));
                        state.waiting_for = None;
                    }
                }
            }

            // 5. Prefetch — launch nodes that static analysis says are ready
            for i in 0..n {
                if !launched[i] && is_node_ready(i, &nodes, &dag, &storage, &completed) {
                    launched[i] = true;
                    node_states[i] = Some(make_node_state(&nodes[i]));
                }
            }
        }

        Ok(storage)
    }
}

// ---------------------------------------------------------------------------
// NodeRunState — per-node rendering state
// ---------------------------------------------------------------------------

struct NodeRunState {
    blocks: Vec<(String, MirModule)>,
    block_idx: usize,
    coroutine: Option<Coroutine>,
    resume_key: Option<ResumeKey>,
    pending_need: Option<NeedContextStepped>,
    current_output: String,
    rendered_messages: Vec<Message>,
    waiting_for: Option<String>,
}

enum DriveResult {
    AllRendered,
    Blocked(String),
}

fn make_node_state(node: &CompiledNode) -> NodeRunState {
    let blocks: Vec<(String, MirModule)> = node
        .messages
        .iter()
        .filter_map(|msg| match msg {
            CompiledMessage::Block(block) => Some((block.role.clone(), block.module.clone())),
            CompiledMessage::Iterator { .. } => None,
        })
        .collect();

    NodeRunState {
        blocks,
        block_idx: 0,
        coroutine: None,
        resume_key: None,
        pending_need: None,
        current_output: String::new(),
        rendered_messages: Vec::new(),
        waiting_for: None,
    }
}

// ---------------------------------------------------------------------------
// drive_rendering — sync, drives a node's coroutine until blocked or done
// ---------------------------------------------------------------------------

fn drive_rendering<S>(state: &mut NodeRunState, storage: &S) -> DriveResult
where
    S: Storage,
{
    loop {
        // Create coroutine for current block if needed
        if state.coroutine.is_none() {
            if state.block_idx >= state.blocks.len() {
                return DriveResult::AllRendered;
            }
            let (_, module) = &state.blocks[state.block_idx];
            let interp = Interpreter::new(module.clone(), ExternFnRegistry::new());
            let (coroutine, key) = interp.execute();
            state.coroutine = Some(coroutine);
            state.resume_key = Some(key);
        }

        let coroutine = state.coroutine.as_mut().unwrap();
        let key = state.resume_key.take().unwrap();

        match coroutine.resume(key) {
            Stepped::Emit(emit) => {
                let (value, next_key) = emit.into_parts();
                match value {
                    Value::String(s) => state.current_output.push_str(&s),
                    other => panic!("drive_rendering: expected String, got {other:?}"),
                }
                state.resume_key = Some(next_key);
            }
            Stepped::NeedContext(need) => {
                if !need.bindings().is_empty() {
                    panic!("context call with bindings not supported in DAG executor");
                }
                let name = need.name().to_string();
                if let Some(output) = storage.get(&name) {
                    state.resume_key = Some(need.into_key(output_to_value(&output)));
                } else {
                    state.pending_need = Some(need);
                    state.waiting_for = Some(name.clone());
                    return DriveResult::Blocked(name);
                }
            }
            Stepped::Done => {
                let role = state.blocks[state.block_idx].0.clone();
                let output = std::mem::take(&mut state.current_output);
                state.rendered_messages.push(Message::text(&role, output));
                state.coroutine = None;
                state.resume_key = None;
                state.block_idx += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// call_model — async, model call + tool loop only
// ---------------------------------------------------------------------------

async fn call_model<F>(
    idx: usize,
    node: CompiledNode,
    messages: Vec<Message>,
    provider_config: ProviderConfig,
    fetch: Arc<F>,
    fuel: Arc<AtomicU64>,
    fuel_limit: u64,
) -> Result<(usize, Output), OrchError>
where
    F: Fetch,
{
    let tools: Vec<ToolSpec> = node
        .tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name.clone(),
            description: String::new(),
            params: t.params.clone(),
        })
        .collect();

    consume_fuel(&fuel, fuel_limit)?;
    let http_request = build_request(&provider_config, &node.model, &messages, &tools, &node.generation, None);
    let json = fetch
        .fetch(&http_request)
        .await
        .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
    let mut response = parse_response(&provider_config.api, &json)
        .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;

    let mut all_messages = messages;
    while let ModelResponse::ToolCalls(ref calls) = response {
        consume_fuel(&fuel, fuel_limit)?;
        let tool_results = handle_tool_calls(calls);

        all_messages.push(Message {
            role: "assistant".into(),
            content: String::new(),
            tool_calls: calls.clone(),
            tool_call_id: None,
        });

        for result in &tool_results {
            all_messages.push(Message {
                role: "tool".into(),
                content: result.content.clone(),
                tool_calls: Vec::new(),
                tool_call_id: Some(result.call_id.clone()),
            });
        }

        let http_request = build_request(&provider_config, &node.model, &all_messages, &tools, &node.generation, None);
        let json = fetch
            .fetch(&http_request)
            .await
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
        response = parse_response(&provider_config.api, &json)
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
    }

    match response {
        ModelResponse::Text(text) => Ok((idx, Output::Text(text))),
        ModelResponse::ToolCalls(_) => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a node can be launched given the current storage state (for prefetch).
fn is_node_ready<S>(
    idx: usize,
    nodes: &[CompiledNode],
    dag: &Dag,
    storage: &S,
    completed: &[bool],
) -> bool
where
    S: Storage,
{
    let node = &nodes[idx];

    let mut known = HashMap::new();
    for key in &node.all_context_keys {
        if let Some(output) = storage.get(key) {
            if let Some(lit) = output_to_literal(&output) {
                known.insert(key.clone(), lit);
            }
        }
    }

    let mut needed = HashSet::new();
    for msg in &node.messages {
        if let CompiledMessage::Block(block) = msg {
            needed.extend(reachable_context_keys(&block.module, &known, &block.val_def));
        }
    }

    for &dep_idx in &dag.deps[idx] {
        if completed[dep_idx] {
            continue;
        }
        let dep_name = &nodes[dep_idx].name;
        if needed.contains(dep_name) {
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------
// Output → Literal conversion (for reachable_context_keys analysis)
// ---------------------------------------------------------------------------

pub fn output_to_literal(output: &Output) -> Option<Literal> {
    match output {
        Output::Text(s) => Some(Literal::String(s.clone())),
        Output::Json(v) => json_to_literal(v),
        Output::Image(_) => None,
    }
}

fn json_to_literal(v: &serde_json::Value) -> Option<Literal> {
    match v {
        serde_json::Value::String(s) => Some(Literal::String(s.clone())),
        serde_json::Value::Bool(b) => Some(Literal::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Literal::Int(i))
            } else if let Some(f) = n.as_f64() {
                Some(Literal::Float(f))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn consume_fuel(fuel: &AtomicU64, limit: u64) -> Result<(), OrchError> {
    let prev = fuel.fetch_add(1, Ordering::Relaxed);
    if prev + 1 > limit {
        Err(OrchError::new(OrchErrorKind::FuelExhausted))
    } else {
        Ok(())
    }
}

fn handle_tool_calls(calls: &[ToolCall]) -> Vec<ToolResult> {
    calls
        .iter()
        .map(|call| ToolResult {
            call_id: call.id.clone(),
            content: format!("tool '{}' not implemented", call.name),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Output → Value conversion (for interpreter context injection)
// ---------------------------------------------------------------------------

pub fn output_to_value(output: &Output) -> Value {
    match output {
        Output::Text(s) => Value::String(s.clone()),
        Output::Json(v) => json_to_value(v),
        Output::Image(bytes) => Value::List(bytes.iter().map(|&b| Value::Byte(b)).collect()),
    }
}

fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Unit,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Unit
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => Value::List(arr.iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            Value::Object(obj.iter().map(|(k, v)| (k.clone(), json_to_value(v))).collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_to_value_text() {
        let v = output_to_value(&Output::Text("hello".into()));
        assert!(matches!(v, Value::String(ref s) if s == "hello"));
    }

    #[test]
    fn output_to_value_json() {
        let v = output_to_value(&Output::Json(serde_json::json!({"name": "alice", "age": 30})));
        match v {
            Value::Object(obj) => {
                assert!(matches!(obj.get("name"), Some(Value::String(s)) if s == "alice"));
                assert!(matches!(obj.get("age"), Some(Value::Int(30))));
            }
            _ => panic!("expected Object"),
        }
    }

    #[test]
    fn output_to_value_image() {
        let v = output_to_value(&Output::Image(vec![0xff, 0x00]));
        match v {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], Value::Byte(0xff)));
            }
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn output_to_literal_text() {
        let lit = output_to_literal(&Output::Text("hello".into()));
        assert_eq!(lit, Some(Literal::String("hello".into())));
    }

    #[test]
    fn output_to_literal_json_string() {
        let lit = output_to_literal(&Output::Json(serde_json::json!("world")));
        assert_eq!(lit, Some(Literal::String("world".into())));
    }

    #[test]
    fn output_to_literal_json_int() {
        let lit = output_to_literal(&Output::Json(serde_json::json!(42)));
        assert_eq!(lit, Some(Literal::Int(42)));
    }

    #[test]
    fn output_to_literal_json_bool() {
        let lit = output_to_literal(&Output::Json(serde_json::json!(true)));
        assert_eq!(lit, Some(Literal::Bool(true)));
    }

    #[test]
    fn output_to_literal_image_none() {
        let lit = output_to_literal(&Output::Image(vec![0xff]));
        assert!(lit.is_none());
    }

    #[test]
    fn output_to_literal_json_object_none() {
        let lit = output_to_literal(&Output::Json(serde_json::json!({"key": "val"})));
        assert!(lit.is_none());
    }
}
