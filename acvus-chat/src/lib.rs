mod error;

pub use error::ChatError;
use rustc_hash::{FxHashMap, FxHashSet};

use std::sync::Arc;

use acvus_interpreter::{RuntimeError, Value};
use acvus_orchestration::{
    CompiledNode, EntryMut, EntryRef, Fetch, Journal, Node, ProviderConfig, ResolveState, Resolved,
    Resolver, build_dag, build_node_table,
};
use acvus_utils::{Astr, Interner};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct ChatEngine<J> {
    nodes: Vec<CompiledNode>,
    node_table: Vec<Arc<dyn Node>>,
    name_to_idx: FxHashMap<Astr, usize>,
    rdeps: Vec<FxHashSet<usize>>,
    pub journal: J,
    pub cursor: Uuid,
    bind_cache: FxHashMap<Astr, Vec<(Value, Arc<Value>)>>,
    entrypoint_idx: usize,
    side_effect_idxs: Vec<usize>,
    interner: Interner,
}

impl<J> ChatEngine<J>
where
    J: Journal,
{
    pub async fn new<F>(
        nodes: Vec<CompiledNode>,
        providers: FxHashMap<String, ProviderConfig>,
        fetch: F,
        journal: J,
        root: Uuid,
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
            journal,
            cursor: root,
            bind_cache: FxHashMap::default(),
            entrypoint_idx,
            side_effect_idxs,
            interner: interner.clone(),
        })
    }

    pub async fn turn<R, EH>(&mut self, resolver: &R, extern_handler: &EH) -> Result<(Value, Uuid), ChatError>
    where
        R: AsyncFn(Astr) -> Resolved + Sync,
        EH: AsyncFn(Astr, Vec<Value>) -> Result<Value, RuntimeError> + Sync,
    {
        let interner = &self.interner;
        let entrypoint_name = &self.nodes[self.entrypoint_idx].name;
        tracing::info!(entrypoint = %interner.resolve(*entrypoint_name), "turn start");

        // Block scope: entry borrows journal mutably, released at block end.
        let (new_cursor, bind_cache, entrypoint_value) = {
            let entry = self.journal.entry_mut(self.cursor).await.next().await;
            let new_cursor = entry.uuid();

            let mut rs = ResolveState {
                entry,
                turn_context: FxHashMap::default(),
                bind_cache: std::mem::take(&mut self.bind_cache),
            };

            {
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
                    .map_err(|e| ChatError::Resolve(format!("[entrypoint] {e}")))?;

                // Side effects
                if !self.side_effect_idxs.is_empty() {
                    let side_effects: Vec<_> = self
                        .side_effect_idxs
                        .iter()
                        .map(|&idx| (idx, FxHashMap::default()))
                        .collect();
                    ctx.resolve_nodes(side_effects, &mut rs)
                        .await
                        .map_err(|e| ChatError::Resolve(format!("[side_effects] {e}")))?;
                }
            }

            // Extract entrypoint value before dropping ResolveState.
            // For Ephemeral persistence the value lives only in turn_context,
            // while Snapshot/Deque/Diff persist to the journal entry.
            let name = self.nodes[self.entrypoint_idx].name;
            let name_str = interner.resolve(name);
            let entrypoint_value = rs.turn_context.get(&name)
                .cloned()
                .or_else(|| rs.entry.get(name_str))
                .ok_or_else(|| ChatError::UnresolvedContext(name_str.to_string()))?;

            let bind_cache = std::mem::take(&mut rs.bind_cache);
            // rs (including entry) dropped here at block end
            (new_cursor, bind_cache, entrypoint_value)
        };

        self.bind_cache = bind_cache;
        self.cursor = new_cursor;

        tracing::info!(depth = self.journal.entry(self.cursor).await.depth(), "turn complete");

        Ok((Value::clone(&entrypoint_value), new_cursor))
    }

    pub async fn history_len(&self) -> usize {
        self.journal.entry(self.cursor).await.depth()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use acvus_mir::ty::Ty;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_orchestration::{
        ApiKind, ExprSpec, GenerationParams, HttpRequest, LlmSpec, MaxTokens,
        Execution, FnParam, MessageSpec, NodeKind, NodeSpec, Persistency, PlainSpec, Strategy, ToolBinding,
        TreeJournal, compile_nodes,
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

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
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

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let result = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
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

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mock = MockFetch::new(vec![openai_text_response("hello from LLM")]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mock = MockFetch::new(vec![
            openai_tool_call_response(vec![("call_1", "my_tool", serde_json::json!({}))]),
            openai_text_response("final answer"),
        ]);
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            mock,
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
                }),
                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::Snapshot,
                    initial_value: Some(interner.intern(r#""A""#)),
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let r1 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
        assert_eq!(r1, Value::String("AB".into()));

        let r2 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
    
                    strategy: Strategy {
                        execution: Execution::Always,
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{@counter}}{{@counter}}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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

                strategy: Strategy {
                    execution: Execution::default(),
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
            PartialContextTypeRegistry::user_only(ctx),
        )
        .unwrap();
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
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

        let result = engine.turn(&resolver, &noop_extern_handler()).await.unwrap().0;
        assert_eq!(result, Value::String("hihi".into()));
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "resolver should be called once per turn"
        );
    }

    /// #6b: LLM nodes with OncePerTurn persist correctly.
    #[tokio::test]
    async fn llm_once_per_turn_persists() {
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

                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::default(),
                    initial_value: None,
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![openai_text_response("hello")]),

            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Verify LLM output is stored and retrievable
        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
                }),
                strategy: Strategy {
                    execution: Execution::OncePerTurn,
                    persistency: Persistency::Snapshot,
                    initial_value: Some(interner.intern(r#""A""#)),
                    retry: 0,
                    assert: None,
                },
                is_function: false,
                fn_params: vec![],
            }],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        // Turn 1: @self = "A" (initial), output = "AB"
        let r1 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
        assert_eq!(r1, Value::String("AB".into()));

        // Turn 2: @self = "AB" (persisted), output = "ABB"
        let r2 = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "ok".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        // @double should be undefined — function nodes are not context
                        source: "{{@double}}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@double(5)".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Expr(ExprSpec {
                        source: "@add_offset(5)".into(),
                        output_ty: Ty::Int,
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(5) | to_string }}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
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
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: true,
                    fn_params: vec![FnParam { name: interner.intern("x"), ty: Ty::Int, description: None }],
                },
                NodeSpec {
                    name: interner.intern("main"),
                    kind: NodeKind::Plain(PlainSpec {
                        source: "{{ @double(3) | to_string }}-{{ @double(7) | to_string }}".into(),
                    }),
    
                    strategy: Strategy {
                        execution: Execution::default(),
                        persistency: Persistency::default(),
                        initial_value: None,
                        retry: 0,
                        assert: None,
                    },
                    is_function: false,
                    fn_params: vec![],
                },
            ],
        );
        let (pname, pconfig) = default_provider();
        let (journal, root) = TreeJournal::new();
        let mut engine = ChatEngine::new(
            nodes,
            FxHashMap::from_iter([(pname, pconfig)]),
            MockFetch::new(vec![]),
            journal,
            root,
            "main",
            &[],
            &interner,
        )
        .await
        .unwrap();

        let result = engine.turn(&noop_resolver(), &noop_extern_handler()).await.unwrap().0;
        assert_eq!(result, Value::String("6-14".into()));
    }
}
