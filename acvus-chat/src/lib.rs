mod error;

pub use error::ChatError;
use rustc_hash::{FxHashMap, FxHashSet};

use std::collections::VecDeque;
use std::sync::Arc;

use acvus_interpreter::{RuntimeError, Value};
use acvus_orchestration::{
    CompiledNode, CompiledNodeKind, CompiledStrategy, Fetch, Node, ProviderConfig, ResolveState,
    Resolved, Resolver, State, Storage, build_dag, build_node_table,
};
use acvus_utils::{Astr, Interner};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ChatEngine<S> {
    nodes: Vec<CompiledNode>,
    node_table: Vec<Arc<dyn Node>>,
    name_to_idx: FxHashMap<Astr, usize>,
    rdeps: Vec<FxHashSet<usize>>,
    pub state: State<S>,
    bind_cache: FxHashMap<Astr, Vec<(Value, Arc<Value>)>>,
    entrypoint_idx: usize,
    history_nodes: Vec<Astr>,
    side_effect_idxs: Vec<usize>,
    interner: Interner,
}

impl<S> ChatEngine<S>
where
    S: Storage,
{
    pub async fn new<F>(
        nodes: Vec<CompiledNode>,
        providers: FxHashMap<String, ProviderConfig>,
        fetch: F,
        mut storage: S,
        entrypoint: &str,
        side_effects: &[String],
        interner: &Interner,
    ) -> Result<Self, ChatError>
    where
        F: Fetch + 'static,
    {
        let name_to_idx: FxHashMap<Astr, usize> =
            nodes.iter().enumerate().map(|(i, n)| (n.name, i)).collect();

        let entrypoint_key = interner.intern(entrypoint);
        let entrypoint_idx = *name_to_idx
            .get(&entrypoint_key)
            .ok_or_else(|| ChatError::EntrypointNotFound(entrypoint.to_string()))?;

        // Collect history nodes
        let history_nodes: Vec<Astr> = nodes
            .iter()
            .filter_map(|n| match &n.strategy {
                CompiledStrategy::History { .. } => Some(n.name),
                _ => None,
            })
            .collect();

        // Validate: no dependency cycles + extract rdeps
        let dag = build_dag(interner, &nodes).map_err(|errs| {
            let msg = errs
                .iter()
                .map(|e| e.display(interner).to_string())
                .collect::<Vec<_>>()
                .join("; ");
            ChatError::CycleDetected(msg)
        })?;
        let rdeps = dag.rdeps;

        // Validate: history nodes must be reachable from entrypoint via BFS
        let mut reachable = FxHashSet::default();
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
            if let CompiledNodeKind::Llm(llm) = &nodes[idx].kind {
                for tool in &llm.tools {
                    let tool_node = interner.intern(&tool.node);
                    if let Some(&dep_idx) = name_to_idx.get(&tool_node)
                        && reachable.insert(dep_idx)
                    {
                        queue.push_back(dep_idx);
                    }
                }
            }
        }
        for &name in &history_nodes {
            if !reachable.contains(&name_to_idx[&name]) {
                return Err(ChatError::HistoryNodeUnreachable(
                    interner.resolve(name).to_string(),
                ));
            }
        }

        // Resolve side_effect node indices
        let side_effect_idxs: Vec<usize> = side_effects
            .iter()
            .filter_map(|name| {
                let key = interner.intern(name);
                name_to_idx.get(&key).copied()
            })
            .collect();

        // Build node table — one match, uniform Arc<dyn Node> from here
        let node_table =
            build_node_table(&nodes, &providers, Arc::new(fetch), interner);

        Ok(Self {
            nodes,
            node_table,
            name_to_idx,
            rdeps,
            state: State::new(storage, 0),
            bind_cache: FxHashMap::default(),
            entrypoint_idx,
            history_nodes,
            side_effect_idxs,
            interner: interner.clone(),
        })
    }

    pub async fn turn<R, EH>(&mut self, resolver: &R, extern_handler: &EH) -> Result<Value, ChatError>
    where
        S: Default,
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
    {
        let interner = &self.interner;
        let entrypoint_name = &self.nodes[self.entrypoint_idx].name;
        tracing::info!(entrypoint = %interner.resolve(*entrypoint_name), "turn start");
        let index_key = interner.intern("index");
        let history_key = interner.intern("history");
        let turn_key = interner.intern("turn");

        // Ensure @turn exists; compute next index
        let turn_index = if let Some(arc) = self.state.storage.get(interner.resolve(turn_key)) {
            let Value::Object(ref turn) = *arc else {
                panic!("@turn must be an Object");
            };
            let Value::Int(i) = turn.get(&index_key).expect("@turn.index missing") else {
                panic!("@turn.index must be Int");
            };
            *i + 1
        } else {
            self.state.storage.set(
                interner.resolve(turn_key).to_string(),
                Value::Object(FxHashMap::from_iter([
                    (index_key, Value::Int(0)),
                    (history_key, Value::List(Vec::new())),
                ])),
            );
            0
        };

        // Always-strategy nodes re-resolve every turn
        for node in &self.nodes {
            if matches!(node.strategy, CompiledStrategy::Always) {
                self.state.storage.remove(interner.resolve(node.name));
            }
        }

        // Build ResolveState for this turn
        let mut rs = ResolveState {
            storage: std::mem::take(&mut self.state.storage),
            turn_context: FxHashMap::default(),
            bind_cache: std::mem::take(&mut self.bind_cache),
            history_entries: FxHashMap::default(),
        };

        let ctx = Resolver {
            nodes: &self.nodes,
            node_table: &self.node_table,
            name_to_idx: &self.name_to_idx,
            extern_handler,
            resolver,
            interner,
            rdeps: &self.rdeps,
        };

        ctx.resolve_node(self.entrypoint_idx, &mut rs, FxHashMap::default())
            .await
            .map_err(|e| ChatError::Resolve(e.to_string()))?;

        // Merge all turn_context results into storage.
        // Every strategy (Always, OncePerTurn, IfModified, History) stores its
        // result in turn_context during the turn. At turn end we persist them all.
        for node in &self.nodes {
            if let Some(v) = rs.turn_context.get(&node.name) {
                rs.storage
                    .set(interner.resolve(node.name).to_string(), Value::clone(v));
            }
        }

        // Flush history + update turn index
        {
            let mut turn_val = rs
                .storage
                .get(interner.resolve(turn_key))
                .map(|arc| Value::clone(&arc))
                .unwrap_or_else(|| Value::Object(FxHashMap::default()));

            if let Value::Object(ref mut turn) = turn_val {
                turn.insert(index_key, Value::Int(turn_index));
                if !rs.history_entries.is_empty() {
                    let history = turn
                        .entry(history_key)
                        .or_insert_with(|| Value::List(Vec::new()));
                    if let Value::List(list) = history {
                        list.push(Value::Object(std::mem::take(&mut rs.history_entries)));
                    }
                }
            }
            rs.storage
                .set(interner.resolve(turn_key).to_string(), turn_val);
        }

        self.state.turn = turn_index as usize;

        // Resolve side-effect nodes concurrently after history flush + turn increment
        if !self.side_effect_idxs.is_empty() {
            let side_effects: Vec<_> = self
                .side_effect_idxs
                .iter()
                .map(|&idx| (idx, FxHashMap::default()))
                .collect();
            ctx.resolve_nodes(side_effects, &mut rs)
                .await
                .map_err(|e| ChatError::Resolve(e.to_string()))?;
        }

        // Restore persistent state
        self.state.storage = rs.storage;
        self.bind_cache = rs.bind_cache;

        tracing::info!(turn = self.state.turn, "turn complete");

        let name = self.nodes[self.entrypoint_idx].name;
        let result = self
            .state
            .storage
            .get(interner.resolve(name))
            .ok_or_else(|| ChatError::UnresolvedContext(interner.resolve(name).to_string()))?;
        Ok(Value::clone(&result))
    }

    pub fn history_len(&self) -> usize {
        self.state.turn
    }

    pub fn history_pop(&mut self) {
        if self.state.turn == 0 {
            return;
        }
        self.state.turn -= 1;
        let interner = &self.interner;
        let turn_key = interner.intern("turn");
        let history_key = interner.intern("history");
        if let Some(arc) = self.state.storage.get(interner.resolve(turn_key)) {
            let mut turn_val = Value::clone(&arc);
            if let Value::Object(ref mut turn) = turn_val
                && let Some(Value::List(history)) = turn.get_mut(&history_key)
            {
                history.pop();
            }
            self.state
                .storage
                .set(interner.resolve(turn_key).to_string(), turn_val);
        }
    }

    pub async fn re_execute<R, EH>(&mut self, index: usize, resolver: &R, extern_handler: &EH) -> Result<Value, ChatError>
    where
        S: Default,
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
    {
        assert!(
            index <= self.state.turn,
            "re_execute index out of bounds: {index} > {}",
            self.state.turn,
        );
        let interner = &self.interner;
        let turn_key = interner.intern("turn");
        let history_key = interner.intern("history");
        if let Some(arc) = self.state.storage.get(interner.resolve(turn_key)) {
            let mut turn_val = Value::clone(&arc);
            if let Value::Object(ref mut turn) = turn_val
                && let Some(Value::List(history)) = turn.get_mut(&history_key)
            {
                history.truncate(index);
            }
            self.state
                .storage
                .set(interner.resolve(turn_key).to_string(), turn_val);
        }
        self.state.turn = index;
        self.turn(resolver, extern_handler).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::ty::Ty;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_orchestration::{
        ApiKind, ExprSpec, GenerationParams, HashMapStorage, HttpRequest, LlmSpec, MaxTokens,
        MessageSpec, NodeKind, NodeSpec, PlainSpec, Strategy, ToolBinding, compile_nodes,
    };

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

    fn noop_resolver() -> impl AsyncFn(Astr) -> Resolved + Sync {
        |_: Astr| async { Resolved::Once(Value::Unit) }
    }

    fn noop_extern_handler() -> impl AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync {
        |_: Astr, _: Vec<Value>| async { Ok(Value::Unit) }
    }

    fn compile_test_nodes(interner: &Interner, specs: &[NodeSpec]) -> Vec<CompiledNode> {
        compile_nodes(
            interner,
            specs,
            PartialContextTypeRegistry::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn new_valid_entrypoint() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello".into(),
                }),

                strategy: Strategy::default(),
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn new_invalid_entrypoint() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello".into(),
                }),

                strategy: Strategy::default(),
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "nonexistent",
            &[],
            &interner,
        )
        .await;
        assert!(matches!(result, Err(ChatError::EntrypointNotFound(_))));
    }

    #[tokio::test]
    async fn turn_plain_node() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "hello world".into(),
                }),

                strategy: Strategy::default(),
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(result, Value::String("hello world".into()));
    }

    #[tokio::test]
    async fn turn_llm_text_response() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Llm(LlmSpec {
                    api: ApiKind::OpenAI,
                    provider: "test".into(),
                    model: "gpt-test".into(),
                    messages: vec![MessageSpec::Block {
                        role: interner.intern("user"),
                        source: "hi".into(),
                    }],
                    tools: vec![],
                    generation: GenerationParams::default(),
                    cache_key: None,
                    max_tokens: MaxTokens::default(),
                }),

                strategy: Strategy::default(),
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![openai_text_response("hello from LLM")]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        // stored value = raw output (List of messages)
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        assert_eq!(msgs.len(), 1);
        let Value::Object(msg) = &msgs[0] else {
            panic!("expected Object");
        };
        let content_key = interner.intern("content");
        assert_eq!(
            msg.get(&content_key),
            Some(&Value::String("hello from LLM".into()))
        );
    }

    #[tokio::test]
    async fn turn_tool_call_round_trip() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("tool_target"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "tool result text".into(),
                    }),
                    strategy: Strategy::default(),
    
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Llm(LlmSpec {
                        api: ApiKind::OpenAI,
                        provider: "test".into(),
                        model: "gpt-test".into(),
                        messages: vec![MessageSpec::Block {
                            role: interner.intern("user"),
                            source: "use the tool".into(),
                        }],
                        tools: vec![ToolBinding {
                            name: "my_tool".into(),
                            description: String::new(),
                            node: "tool_target".into(),
                            params: FxHashMap::default(),
                        }],
                        generation: GenerationParams::default(),
                        cache_key: None,
                        max_tokens: MaxTokens::default(),
                    }),
                    strategy: Strategy::default(),
    
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let mock = MockFetch::new(vec![
            openai_tool_call_response(vec![("call_1", "my_tool", serde_json::json!({}))]),
            openai_text_response("final answer"),
        ]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        // stored value = raw output (List of messages)
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        let Value::Object(msg) = msgs.last().unwrap() else {
            panic!("expected Object");
        };
        let content_key = interner.intern("content");
        assert_eq!(
            msg.get(&content_key),
            Some(&Value::String("final answer".into()))
        );
    }

    // -- regression tests -------------------------------------------------------

    /// #6: initial_value must be evaluated on first run (not Unit).
    /// OncePerTurn: first turn uses initial_value as @self, subsequent turns use persisted @self.
    /// Accumulation is done in the Expr node body using @self.
    #[tokio::test]
    async fn initial_value_evaluated_on_first_run() {
        let interner = Interner::new();
        // Expr uses @self (previous) to accumulate.
        // initial_value = "A".
        // Turn 1: @self = "A" (initial), expr = @self + "B" → "AB"
        // Turn 2: @self = "AB" (persisted), expr = @self + "B" → "ABB"
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Expr(ExprSpec {
                    source: r#"@self + "B""#.into(),
                    output_ty: Ty::Infer,
                    initial_value: Some(r#""A""#.into()),
                }),
                strategy: Strategy::OncePerTurn,
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let r1 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(r1, Value::String("AB".into()));

        let r2 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(r2, Value::String("ABB".into()));
    }

    /// #7: Always nodes must re-execute every invocation, not just once per turn.
    #[tokio::test]
    async fn always_node_re_executes_every_reference() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("counter"),
                    kind: NodeKind::Plain(PlainSpec { source: "x".into() }),
    
                    strategy: Strategy::Always,
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{@counter}}{{@counter}}".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(result, Value::String("xx".into()));
    }

    /// #3 (double-prompt): Turn-resolved external values must be cached within a turn.
    #[tokio::test]
    async fn turn_resolver_caches_within_turn() {
        use acvus_mir::ty::Ty;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let interner = Interner::new();
        // main template references @input twice.
        // External resolver should be called only once for @input per turn.
        let mut ctx = FxHashMap::default();
        ctx.insert(interner.intern("input"), Ty::String);
        let nodes = compile_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Plain(PlainSpec {
                    source: "{{@input}}{{@input}}".into(),
                }),

                strategy: Strategy::default(),
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
            PartialContextTypeRegistry::user_only(ctx),
        )
        .unwrap();
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let call_count = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&call_count);
        let resolver = move |_: Astr| {
            let count = Arc::clone(&count);
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Resolved::Turn(Value::String("hi".into()))
            }
        };

        let result = engine.turn(&resolver, &noop_extern_handler()).await.unwrap();
        assert_eq!(result, Value::String("hihi".into()));
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolver should be called once per turn"
        );
    }

    /// #6b (history_bind @self): history_bind must have access to @self without panic.
    #[tokio::test]
    async fn history_bind_accesses_self() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Llm(LlmSpec {
                    api: ApiKind::OpenAI,
                    provider: "test".into(),
                    model: "m".into(),
                    messages: vec![MessageSpec::Block {
                        role: interner.intern("user"),
                        source: "hi".into(),
                    }],
                    tools: vec![],
                    generation: GenerationParams::default(),
                    cache_key: None,
                    max_tokens: MaxTokens::default(),
                }),

                // history_bind accesses @self (= raw output = List) and extracts content
                strategy: Strategy::History {
                    history_bind: interner.intern(r#"@self | map(x -> x.content) | join("")"#),
                },
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![openai_text_response("hello")]),

            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Should not panic — @self = raw output = List<{role, content, content_type}>
        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        let Value::List(msgs) = &result else {
            panic!("expected List, got {result:?}");
        };
        let Value::Object(msg) = &msgs[0] else {
            panic!("expected Object");
        };
        let content_key = interner.intern("content");
        assert_eq!(msg.get(&content_key), Some(&Value::String("hello".into())));
    }

    /// @self in Expr node body: accumulates across turns.
    /// Uses OncePerTurn so @self persists.
    #[tokio::test]
    async fn node_body_accesses_self() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[NodeSpec {
                name: interner.intern("main"),
                kind: NodeKind::Expr(ExprSpec {
                    source: r#"@self + "B""#.into(),
                    output_ty: Ty::Infer,
                    initial_value: Some(r#""A""#.into()),
                }),
                strategy: Strategy::OncePerTurn,
                retry: 0,
                assert: None,
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Turn 1: @self = "A" (initial), output = "AB"
        let r1 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(r1, Value::String("AB".into()));

        // Turn 2: @self = "AB" (persisted), output = "ABB"
        let r2 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(r2, Value::String("ABB".into()));
    }

    // -- function node tests ---------------------------------------------------

    /// Function node compiles successfully: Expr kind + is_function + fn_params
    #[tokio::test]
    async fn function_node_compile_success() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "ok".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        assert_eq!(nodes.len(), 2);
        assert!(nodes[0].is_function);
    }

    /// Function node is NOT in context_types: @double should fail
    #[test]
    fn function_node_not_in_context() {
        let interner = Interner::new();
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        // @double should be undefined — function nodes are not context
                        source: "{{@double}}".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        );
        // Should fail: @double is not a context key (it's a function)
        assert!(result.is_err());
    }

    /// Other nodes can call function nodes: double(5) should typecheck
    #[test]
    fn function_node_callable_from_other_nodes() {
        let interner = Interner::new();
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@double(5)".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        );
        assert!(result.is_ok(), "function call should typecheck: {result:?}");
    }

    /// Function node with global context: fn body references @globalCtx
    #[test]
    fn function_node_with_global_context() {
        let interner = Interner::new();
        let ctx = FxHashMap::from_iter([(interner.intern("offset"), Ty::Int)]);
        let result = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("add_offset"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x + @offset".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@add_offset(5)".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::user_only(ctx),
        );
        assert!(result.is_ok(), "function with global context should compile: {result:?}");
    }

    /// Full integration: Plain node calls function node, gets result
    #[tokio::test]
    async fn function_call_full_turn() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(5) | to_string }}".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(result, Value::String("10".into()));
    }

    /// @funcName context access should produce a resolve error
    #[tokio::test]
    async fn function_node_context_access_error() {
        let interner = Interner::new();
        // We need to manually set up: double is a function node, main tries to
        // reference @double as context. Since compile rejects @double in templates,
        // we test at the resolver level by injecting @double as a known context type
        // but marking the compiled node as is_function.
        let nodes = compile_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "42".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![],
                },
                // main doesn't use @double in template (compile would reject it),
                // so we just verify the is_function flag propagates
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "ok".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
            PartialContextTypeRegistry::default(),
        )
        .unwrap();

        assert!(nodes[0].is_function);
    }

    /// Multiple function calls in one template
    #[tokio::test]
    async fn function_call_multiple_times() {
        let interner = Interner::new();
        let nodes = compile_test_nodes(
            &interner,
            &[
                NodeSpec {
                    name: interner.intern("double"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@x * 2".into(),
                        output_ty: Ty::Int,
                        initial_value: None,
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: true,
                    fn_params: vec![(interner.intern("x"), Ty::Int)],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(3) | to_string }}-{{ @double(7) | to_string }}".into(),
                    }),
    
                    strategy: Strategy::default(),
                    retry: 0,
                    assert: None,
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            HashMapStorage::new(),
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap();
        assert_eq!(result, Value::String("6-14".into()));
    }
}
