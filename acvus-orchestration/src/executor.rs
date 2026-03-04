use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
use crate::kind::{CompiledNodeKind, GenerationParams};
use crate::message::{Content, Message, ModelResponse, ToolCall, ToolSpec};
use crate::provider::{Fetch, ProviderConfig, build_request, parse_response};
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
        Self {
            nodes,
            dag,
            storage,
            fetch,
            providers,
            mir_registry,
            fuel_limit,
        }
    }

    /// Run the full DAG with demand-driven rendering.
    ///
    /// Template rendering is sync (coroutine resume loop). When a template
    /// needs a context value that isn't available yet, the node blocks until
    /// the dependency's model call completes. Model calls are the only async
    /// operations, driven via `FuturesUnordered`.
    pub async fn run(self) -> Result<S, OrchError> {
        let Executor {
            nodes,
            dag,
            mut storage,
            fetch,
            providers,
            fuel_limit,
            ..
        } = self;
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

        let mut running: FuturesUnordered<BoxFuture<'static, Result<(usize, Value), OrchError>>> =
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
                            if let Some(&dep_idx) = name_to_idx.get(dep_name.as_str())
                                && !launched[dep_idx]
                            {
                                to_launch.push(dep_idx);
                                progress = true;
                            }
                        }
                    }
                }

                for i in to_submit {
                    let state = node_states[i].take().unwrap();
                    let CompiledNodeKind::Llm(llm) = &nodes[i].kind else {
                        continue;
                    };
                    let tool_specs: Vec<ToolSpec> = llm
                        .tools
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
                        .collect();
                    let (provider, model, tools, generation, max_output_tokens) = (
                        llm.provider.clone(),
                        llm.model.clone(),
                        tool_specs,
                        llm.generation.clone(),
                        llm.max_tokens.output,
                    );
                    let provider_config = providers
                        .get(&provider)
                        .ok_or_else(|| {
                            OrchError::new(OrchErrorKind::ModelError(format!(
                                "unknown provider: {}",
                                provider
                            )))
                        })?
                        .clone();
                    running.push(Box::pin(call_model(
                        i,
                        model,
                        tools,
                        generation,
                        max_output_tokens,
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
                if let Some(state) = &mut node_states[i]
                    && state.waiting_for.as_deref() == Some(completed_name.as_str())
                {
                    let need = state.pending_need.take().unwrap();
                    let arc = storage.get(completed_name).unwrap();
                    state.resume_key = Some(need.into_key(arc));
                    state.waiting_for = None;
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
        .kind
        .messages()
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
            let interp = Interpreter::new(module.clone(), &ExternFnRegistry::new());
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
                if let Some(arc) = storage.get(&name) {
                    state.resume_key = Some(need.into_key(arc));
                } else {
                    state.pending_need = Some(need);
                    state.waiting_for = Some(name.clone());
                    return DriveResult::Blocked(name);
                }
            }
            Stepped::Done => {
                let role = state.blocks[state.block_idx].0.clone();
                let output = std::mem::take(&mut state.current_output);
                state.rendered_messages.push(Message::Content {
                    role,
                    content: Content::Text(output),
                });
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
    model: String,
    tools: Vec<ToolSpec>,
    generation: GenerationParams,
    max_output_tokens: Option<u32>,
    messages: Vec<Message>,
    provider_config: ProviderConfig,
    fetch: Arc<F>,
    fuel: Arc<AtomicU64>,
    fuel_limit: u64,
) -> Result<(usize, Value), OrchError>
where
    F: Fetch,
{
    consume_fuel(&fuel, fuel_limit)?;
    let http_request = build_request(
        &provider_config,
        &model,
        &messages,
        &tools,
        &generation,
        max_output_tokens,
        None,
    );
    let json = fetch
        .fetch(&http_request)
        .await
        .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
    let (mut response, _usage) = parse_response(&provider_config.api, &json)
        .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;

    let mut all_messages = messages;
    while let ModelResponse::ToolCalls(ref calls) = response {
        consume_fuel(&fuel, fuel_limit)?;
        let tool_results = handle_tool_calls(calls);

        all_messages.push(Message::ToolCalls(calls.clone()));
        all_messages.extend(tool_results);

        let http_request = build_request(
            &provider_config,
            &model,
            &all_messages,
            &tools,
            &generation,
            max_output_tokens,
            None,
        );
        let json = fetch
            .fetch(&http_request)
            .await
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
        (response, _) = parse_response(&provider_config.api, &json)
            .map_err(|e| OrchError::new(OrchErrorKind::ModelError(e)))?;
    }

    match response {
        ModelResponse::Content(items) => {
            let mut text = String::new();
            for item in items {
                match item.content {
                    Content::Text(s) => text.push_str(&s),
                    _ => {
                        return Err(OrchError::new(OrchErrorKind::ModelError(
                            "non-text response in DAG executor".into(),
                        )));
                    }
                }
            }
            Ok((idx, Value::String(text)))
        }
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
        if let Some(arc) = storage.get(key)
            && let Some(lit) = value_to_literal(&arc)
        {
            known.insert(key.clone(), lit);
        }
    }

    let mut needed = HashSet::new();
    for msg in node.kind.messages() {
        if let CompiledMessage::Block(block) = msg {
            needed.extend(reachable_context_keys(
                &block.module,
                &known,
                &block.val_def,
            ));
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
// Value → Literal conversion (for reachable_context_keys analysis)
// ---------------------------------------------------------------------------

pub fn value_to_literal(value: &Value) -> Option<Literal> {
    match value {
        Value::String(s) => Some(Literal::String(s.clone())),
        Value::Bool(b) => Some(Literal::Bool(*b)),
        Value::Int(i) => Some(Literal::Int(*i)),
        Value::Float(f) => Some(Literal::Float(*f)),
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

fn handle_tool_calls(calls: &[ToolCall]) -> Vec<Message> {
    calls
        .iter()
        .map(|call| Message::ToolResult {
            call_id: call.id.clone(),
            content: format!("tool '{}' not implemented", call.name),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::extern_module::ExternRegistry;

    use crate::compile::compile_nodes;
    use crate::dag::build_dag;
    use crate::dsl::{MessageSpec, NodeSpec, Strategy};
    use crate::kind::{LlmSpec, MaxTokens};
    use crate::provider::{ApiKind, HttpRequest, ProviderConfig};
    use crate::storage::HashMapStorage;

    #[test]
    fn value_to_literal_string() {
        let lit = value_to_literal(&Value::String("hello".into()));
        assert_eq!(lit, Some(Literal::String("hello".into())));
    }

    #[test]
    fn value_to_literal_int() {
        let lit = value_to_literal(&Value::Int(42));
        assert_eq!(lit, Some(Literal::Int(42)));
    }

    #[test]
    fn value_to_literal_bool() {
        let lit = value_to_literal(&Value::Bool(true));
        assert_eq!(lit, Some(Literal::Bool(true)));
    }

    #[test]
    fn value_to_literal_float() {
        let lit = value_to_literal(&Value::Float(3.14));
        assert_eq!(lit, Some(Literal::Float(3.14)));
    }

    #[test]
    fn value_to_literal_object_none() {
        let lit = value_to_literal(&Value::Object(Default::default()));
        assert!(lit.is_none());
    }

    // -- MockFetch ---------------------------------------------------------------

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

    // -- helpers -----------------------------------------------------------------

    fn compile_test_nodes(specs: &[NodeSpec]) -> Vec<CompiledNode> {
        compile_nodes(specs, &HashMap::new(), &ExternRegistry::default()).unwrap()
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

    fn openai_tool_call_response(
        calls: Vec<(&str, &str, serde_json::Value)>,
    ) -> serde_json::Value {
        let tool_calls: Vec<serde_json::Value> = calls
            .into_iter()
            .map(|(id, name, args)| {
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

    fn llm_node(name: &str, template: &str) -> NodeSpec {
        NodeSpec {
            name: name.into(),
            kind: crate::kind::NodeKind::Llm(LlmSpec {
                api: ApiKind::OpenAI,
                provider: "test".into(),
                model: "m".into(),
                messages: vec![MessageSpec::Block {
                    role: "user".into(),
                    source: template.into(),
                }],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            strategy: Strategy::default(),
            history: None,
        }
    }

    async fn run_executor(
        nodes: Vec<CompiledNode>,
        responses: Vec<serde_json::Value>,
        fuel_limit: u64,
    ) -> Result<HashMapStorage, OrchError> {
        let dag = build_dag(&nodes).unwrap();
        let (pname, pconfig) = default_provider();
        let executor = Executor::new(
            nodes,
            dag,
            HashMapStorage::new(),
            MockFetch::new(responses),
            HashMap::from([(pname, pconfig)]),
            ExternRegistry::default(),
            fuel_limit,
        );
        executor.run().await
    }

    // -- e2e tests ---------------------------------------------------------------

    #[tokio::test]
    async fn single_node_text_response() {
        let nodes = compile_test_nodes(&[llm_node("a", "hello")]);
        let storage = run_executor(nodes, vec![openai_text_response("world")], 10)
            .await
            .unwrap();
        let val = storage.get("a").unwrap();
        assert!(matches!(&*val, Value::String(s) if s == "world"));
    }

    #[tokio::test]
    async fn two_node_dependency() {
        use crate::compile::CompiledBlock;
        use acvus_mir_pass::AnalysisPass;
        use acvus_mir_pass::analysis::val_def::ValDefMapAnalysis;

        // Build node A: simple template "first"
        let nodes_a = compile_test_nodes(&[llm_node("a", "first")]);
        let node_a = nodes_a.into_iter().next().unwrap();

        // Build node B manually with a dependency on "a" via all_context_keys.
        // The template is just "hello" but we mark "a" as a context key to create the dep.
        let ctx = HashMap::from([("a".into(), acvus_mir::ty::Ty::String)]);
        let ast_b = acvus_ast::parse("{{@a}}").unwrap();
        let (module_b, _) = acvus_mir::compile(
            &ast_b,
            ctx,
            &ExternRegistry::default(),
            &acvus_mir::user_type::UserTypeRegistry::new(),
        )
        .unwrap();
        let val_def_b = ValDefMapAnalysis.run(&module_b, ());
        let node_b = CompiledNode {
            name: "b".into(),
            kind: CompiledNodeKind::Llm(crate::kind::CompiledLlm {
                api: ApiKind::OpenAI,
                provider: "test".into(),
                model: "m".into(),
                messages: vec![CompiledMessage::Block(CompiledBlock {
                    role: "user".into(),
                    module: module_b,
                    context_keys: ["a".into()].into(),
                    val_def: val_def_b,
                })],
                tools: vec![],
                generation: Default::default(),
                cache_key: None,
                max_tokens: Default::default(),
            }),
            all_context_keys: ["a".into()].into(),
            strategy: Strategy::default(),
            bind_module: None,
            history: None,
        };

        let storage = run_executor(
            vec![node_a, node_b],
            vec![
                openai_text_response("from-a"),
                openai_text_response("from-b"),
            ],
            10,
        )
        .await
        .unwrap();
        let a = storage.get("a").unwrap();
        assert!(matches!(&*a, Value::String(s) if s == "from-a"));
        let b = storage.get("b").unwrap();
        assert!(matches!(&*b, Value::String(s) if s == "from-b"));
    }

    #[tokio::test]
    async fn tool_call_loop() {
        let nodes = compile_test_nodes(&[llm_node("a", "do something")]);
        let storage = run_executor(
            nodes,
            vec![
                openai_tool_call_response(vec![("c1", "fn1", serde_json::json!({}))]),
                openai_text_response("done"),
            ],
            10,
        )
        .await
        .unwrap();
        let val = storage.get("a").unwrap();
        assert!(matches!(&*val, Value::String(s) if s == "done"));
    }

    #[tokio::test]
    async fn fuel_exhaustion() {
        let nodes = compile_test_nodes(&[llm_node("a", "go")]);
        let err = run_executor(
            nodes,
            vec![
                openai_tool_call_response(vec![("c1", "fn1", serde_json::json!({}))]),
                openai_tool_call_response(vec![("c2", "fn2", serde_json::json!({}))]),
                openai_text_response("never"),
            ],
            1,
        )
        .await
        .unwrap_err();
        assert!(matches!(err.kind, OrchErrorKind::FuelExhausted));
    }

    #[tokio::test]
    async fn malformed_response_error() {
        let nodes = compile_test_nodes(&[llm_node("a", "hello")]);
        let response = serde_json::json!({ "error": "bad request" });
        let err = run_executor(nodes, vec![response], 10).await.unwrap_err();
        assert!(matches!(err.kind, OrchErrorKind::ModelError(_)));
    }

    #[test]
    #[should_panic(expected = "expected Object")]
    fn item_fields_non_object_panics() {
        let api = ApiKind::OpenAI;
        api.item_fields(&Value::String("not an object".into()));
    }
}
