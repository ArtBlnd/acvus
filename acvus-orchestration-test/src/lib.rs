use std::sync::Arc;

use acvus_interpreter::{RuntimeError, TypedValue, ValueKind};
use acvus_mir::graph::Id;
use acvus_orchestration::{
    build_dag, compile_nodes, compute_external_context_env,
    EntryMut, Execution, ExpressionSpec,
    Fetch, HttpRequest, Journal, NodeGraph, NodeKind, NodeSpec, Persistency,
    ResolveError, ResolveState, Resolved, Resolver, Strategy, TreeJournal,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

// ── Dummy Fetch ─────────────────────────────────────────────────────

/// Fetch implementation that panics — Expression nodes never call it.
pub struct DummyFetch;

impl Fetch for DummyFetch {
    async fn fetch(&self, _request: &HttpRequest) -> Result<serde_json::Value, String> {
        panic!("DummyFetch: Expression-only tests must not make HTTP calls")
    }
}

// ── Node builder ────────────────────────────────────────────────────

/// Fluent builder for constructing NodeSpec in tests.
pub struct NodeBuilder {
    interner: Interner,
    specs: Vec<NodeSpec>,
}

impl NodeBuilder {
    pub fn new(interner: Interner) -> Self {
        Self { interner, specs: Vec::new() }
    }

    /// Add an Expression node (ephemeral, no persist).
    pub fn expr(mut self, name: &str, source: &str) -> Self {
        self.specs.push(NodeSpec {
            name: self.interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec {
                source: source.to_string(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        });
        self
    }

    /// Add an Expression node with Patch persistence.
    pub fn patch(mut self, name: &str, body: &str, bind: &str, initial: &str) -> Self {
        let bind_astr = self.interner.intern(bind);
        let init_astr = self.interner.intern(initial);
        self.specs.push(NodeSpec {
            name: self.interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec {
                source: body.to_string(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Patch { bind: bind_astr },
                initial_value: Some(init_astr),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        });
        self
    }

    /// Add an Expression node with Sequence persistence.
    pub fn sequence(mut self, name: &str, body: &str, bind: &str, initial: &str) -> Self {
        let bind_astr = self.interner.intern(bind);
        let init_astr = self.interner.intern(initial);
        self.specs.push(NodeSpec {
            name: self.interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec {
                source: body.to_string(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Sequence { bind: bind_astr },
                initial_value: Some(init_astr),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        });
        self
    }

    /// Add an Expression node with Sequence persistence and explicit raw output_ty.
    /// Simulates LLM nodes that have a fixed raw type (e.g. List<Message>).
    pub fn sequence_with_raw_ty(
        mut self,
        name: &str,
        body: &str,
        bind: &str,
        initial: &str,
        raw_ty: acvus_mir::ty::Ty,
    ) -> Self {
        let bind_astr = self.interner.intern(bind);
        let init_astr = self.interner.intern(initial);
        self.specs.push(NodeSpec {
            name: self.interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec {
                source: body.to_string(),
                output_ty: Some(raw_ty),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Sequence { bind: bind_astr },
                initial_value: Some(init_astr),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        });
        self
    }

    /// Add an Expression node with an assert script.
    pub fn expr_with_assert(mut self, name: &str, source: &str, assert_script: &str) -> Self {
        let assert_astr = self.interner.intern(assert_script);
        self.specs.push(NodeSpec {
            name: self.interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec {
                source: source.to_string(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: Some(assert_astr),
            },
            is_function: false,
            fn_params: vec![],
        });
        self
    }

    pub fn interner(&self) -> &Interner {
        &self.interner
    }

    /// Try to build — returns Err if compilation fails.
    pub fn try_build(self) -> Result<BuiltGraph, Vec<acvus_orchestration::OrchError>> {
        let registry = acvus_mir::context_registry::PartialContextTypeRegistry::system_only(
            FxHashMap::default(),
        );

        // Build context registry via compute_external_context_env (lower + resolve).
        let env = compute_external_context_env(&self.interner, &self.specs, registry.clone())?;
        let context_registry = acvus_mir::context_registry::ContextTypeRegistry::new(
            env.registry.extern_fns().clone(),
            env.registry.system().clone(),
            FxHashMap::default(), // scoped
            env.registry.user().clone(),
        ).expect("registry construction should not conflict");

        // Compile nodes (lower + compile + assemble in one step).
        let fetch = Arc::new(DummyFetch);
        let node_graph = compile_nodes(&self.interner, &self.specs, registry, fetch)?;

        let name_to_id = node_graph.id_table().to_name_to_id();

        Ok(BuiltGraph {
            interner: self.interner,
            graph: node_graph,
            context_registry,
            name_to_id,
        })
    }

    pub fn build(self) -> BuiltGraph {
        self.try_build().expect("build failed")
    }
}

// ── Built graph ─────────────────────────────────────────────────────

pub struct BuiltGraph {
    pub interner: Interner,
    pub graph: NodeGraph,
    /// The context type registry after compilation.
    /// Contains the types that OTHER nodes see when referencing @name.
    pub context_registry: acvus_mir::context_registry::ContextTypeRegistry,
    pub name_to_id: FxHashMap<Astr, Id>,
}

impl BuiltGraph {
    /// Resolve a single node by name, using a fresh TreeJournal.
    /// Returns the value stored/cached for that node after one turn.
    pub async fn resolve_once(&self, name: &str) -> Result<TypedValue, ResolveError> {
        let (mut journal, root) = TreeJournal::new();
        let entry = journal.entry_mut(root).await.unwrap().next().await.unwrap();

        let name_astr = self.interner.intern(name);
        let node_id = self.graph.entrypoint(name_astr).expect("node not found");

        let resolver = Resolver {
            graph: &self.graph,
            resolver: &|_name: Astr| async { Resolved::Once(TypedValue::unit()) },
            extern_handler: &|_name: Astr, _args: Vec<TypedValue>| async {
                Err(RuntimeError::unexpected_type("extern", &[], ValueKind::Unit))
            },
            interner: &self.interner,
            rdeps: &[],
            name_to_id: &self.name_to_id,
        };

        let mut state = ResolveState {
            entry,
            turn_context: FxHashMap::default(),
        };

        resolver.resolve_node(node_id, &mut state, FxHashMap::default(), false).await?;

        // Return the value from turn_context or storage.
        if let Some(v) = state.turn_context.get(&name_astr) {
            return Ok(v.clone());
        }
        let key = self.interner.resolve(name_astr);
        if let Some(v) = state.entry.get(key) {
            return Ok(v);
        }
        panic!("node {name:?} produced no value after resolve");
    }

    /// Resolve a node over multiple turns.
    /// Returns the value after the last turn.
    /// `turns` is the number of turns to execute (must be >= 1).
    pub async fn resolve_turns(
        &self,
        name: &str,
        turns: usize,
    ) -> Result<TypedValue, ResolveError> {
        assert!(turns >= 1, "turns must be >= 1");

        let (mut journal, root) = TreeJournal::new();
        let name_astr = self.interner.intern(name);
        let node_id = self.graph.entrypoint(name_astr).expect("node not found");

        let resolver = Resolver {
            graph: &self.graph,
            resolver: &|_name: Astr| async { Resolved::Once(TypedValue::unit()) },
            extern_handler: &|_name: Astr, _args: Vec<TypedValue>| async {
                Err(RuntimeError::unexpected_type("extern", &[], ValueKind::Unit))
            },
            interner: &self.interner,
            rdeps: &[],
            name_to_id: &self.name_to_id,
        };

        let mut entry = journal.entry_mut(root).await.unwrap().next().await.unwrap();
        let mut last_value = None;

        for _ in 0..turns {
            let mut state = ResolveState {
                entry,
                turn_context: FxHashMap::default(),
                };

            resolver.resolve_node(node_id, &mut state, FxHashMap::default(), false).await?;

            // Capture value.
            if let Some(v) = state.turn_context.get(&name_astr) {
                last_value = Some(v.clone());
            } else {
                let key = self.interner.resolve(name_astr);
                if let Some(v) = state.entry.get(key) {
                    last_value = Some(v);
                }
            }

            // Advance to next turn.
            entry = state.entry.next().await
                .expect("next turn failed");
        }

        last_value.ok_or_else(|| panic!("node {name:?} produced no value after {turns} turns"))
    }

    /// Get the type that OTHER nodes see when referencing @name.
    /// Uses the registry_ty as the canonical output type, since NodeGraph
    /// no longer exposes per-unit output_ty directly.
    pub fn output_ty(&self, name: &str) -> acvus_mir::ty::Ty {
        self.registry_ty(name)
            .unwrap_or_else(|| panic!("node {name:?} not found in context registry"))
    }

    /// Get the type that OTHER nodes see when referencing @name.
    /// This is the type registered in the context registry during compilation.
    /// If this differs from output_ty, downstream nodes will generate wrong casts.
    /// Get the type that OTHER nodes see when referencing @name.
    /// This is the type registered in the context registry during compilation.
    /// If this differs from output_ty, downstream nodes will generate wrong casts.
    pub fn registry_ty(&self, name: &str) -> Option<acvus_mir::ty::Ty> {
        let name_astr = self.interner.intern(name);
        self.context_registry.system().get(&name_astr).cloned()
            .or_else(|| self.context_registry.scoped().get(&name_astr).cloned())
            .or_else(|| self.context_registry.user().get(&name_astr).cloned())
    }
}

