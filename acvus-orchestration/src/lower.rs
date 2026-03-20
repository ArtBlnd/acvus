//! Lowering: NodeSpec[] → CompilationGraph + LoweringManifest.
//!
//! Translates orchestration domain concepts (NodeSpec, Strategy, Persistency)
//! into a flat compilation graph that the MIR graph engine can process.
//! After this point, orchestration domain concepts are gone.

use acvus_mir::context_registry::PartialContextTypeRegistry;
use acvus_mir::graph::{
    CompilationGraph, CompilationUnit, ContextBinding, ContextId, ContextIdTable,
    ContextSource, ExternDecl, ExternalType, Namespace, NamespaceId, Scope, ScopeId,
    SourceKind, TypeTransform, UnitId,
};
use acvus_mir::ty::{Effect, Ty};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::dsl::{
    Execution, MessageSpec, NodeSpec, Persistency,
    KEY_INDEX, KEY_ITEM, KEY_RAW, KEY_SELF, KEY_TURN_INDEX,
};
use crate::spec::NodeKind;

// ── Output ───────────────────────────────────────────────────────────

/// Manifest mapping units back to NodeSpecs — used by Assembly (Step 4).
/// The graph engine ignores this; it's orchestration metadata.
#[derive(Debug)]
pub struct LoweringManifest {
    pub node_units: Vec<NodeUnitMap>,
}

/// Which units were created for a single NodeSpec.
#[derive(Debug, Default)]
pub struct NodeUnitMap {
    pub body: Option<UnitId>,
    pub init: Option<UnitId>,
    pub bind: Option<UnitId>,
    pub assert: Option<UnitId>,
    pub execution: Option<UnitId>,
    pub messages: Vec<UnitId>,
    pub extern_decl: Option<UnitId>,
}

// ── Id allocators ────────────────────────────────────────────────────

struct IdAlloc {
    next_context: u32,
    next_unit: u32,
    next_scope: u32,
    next_namespace: u32,
}

impl IdAlloc {
    fn new() -> Self {
        Self { next_context: 0, next_unit: 0, next_scope: 0, next_namespace: 0 }
    }
    fn context(&mut self) -> ContextId { let id = ContextId(self.next_context); self.next_context += 1; id }
    fn unit(&mut self) -> UnitId { let id = UnitId(self.next_unit); self.next_unit += 1; id }
    fn scope(&mut self) -> ScopeId { let id = ScopeId(self.next_scope); self.next_scope += 1; id }
    fn namespace(&mut self) -> NamespaceId { let id = NamespaceId(self.next_namespace); self.next_namespace += 1; id }
}

// ── Lowering ─────────────────────────────────────────────────────────

pub fn lower(
    interner: &Interner,
    specs: &[NodeSpec],
    registry: &PartialContextTypeRegistry,
) -> (CompilationGraph, LoweringManifest) {
    let mut alloc = IdAlloc::new();
    let mut id_table = ContextIdTable::new();
    let mut units: Vec<CompilationUnit> = Vec::new();
    let mut externs: Vec<ExternDecl> = Vec::new();
    let mut scopes: Vec<Scope> = Vec::new();
    let mut externals: FxHashMap<ContextId, ExternalType> = FxHashMap::default();
    let mut manifest = LoweringManifest { node_units: Vec::new() };

    let global_ns = alloc.namespace();

    // ── Global context ───────────────────────────────────────────

    // @turn_index
    let turn_index_id = alloc.context();
    id_table.insert(turn_index_id, global_ns, interner.intern(KEY_TURN_INDEX));
    externals.insert(turn_index_id, ExternalType::Known(Ty::Int));

    // Extern fns from registry
    let mut global_name_to_id: FxHashMap<Astr, ContextId> = FxHashMap::default();
    global_name_to_id.insert(interner.intern(KEY_TURN_INDEX), turn_index_id);

    for (name, ty) in registry.extern_fns() {
        let id = alloc.context();
        id_table.insert(id, global_ns, *name);
        externals.insert(id, ExternalType::Known(ty.clone()));
        global_name_to_id.insert(*name, id);
    }

    // User params from registry
    for (name, ty) in registry.user() {
        let id = alloc.context();
        id_table.insert(id, global_ns, *name);
        externals.insert(id, ExternalType::Known(ty.clone()));
        global_name_to_id.insert(*name, id);
    }

    // ── Register node names as global context (inter-node refs) ──
    // First pass: allocate ContextIds for all node names so they can
    // be referenced by other nodes.
    let mut node_ctx_ids: Vec<ContextId> = Vec::new();
    for spec in specs {
        let id = alloc.context();
        id_table.insert(id, global_ns, spec.name);
        global_name_to_id.insert(spec.name, id);
        node_ctx_ids.push(id);
    }

    // ── Per-node lowering ────────────────────────────────────────

    for (spec_idx, spec) in specs.iter().enumerate() {
        let node_ns = alloc.namespace();
        let mut node_map = NodeUnitMap::default();

        // Start with global name_to_id, then add node-local names.
        let mut base_name_to_id = global_name_to_id.clone();

        // Add fn_params to base context (function nodes).
        if spec.is_function {
            for p in &spec.fn_params {
                let id = alloc.context();
                id_table.insert(id, node_ns, p.name);
                externals.insert(id, ExternalType::Known(p.ty.clone()));
                base_name_to_id.insert(p.name, id);
            }
        }

        // ── Determine raw_ty and create body unit ────────────────

        let raw_output_ty = spec.kind.raw_output_ty(interner);
        let body_unit_id = alloc.unit();
        let body_unit = match &spec.kind {
            NodeKind::Expression(expr_spec) => {
                Some(CompilationUnit {
                    id: body_unit_id,
                    source: interner.intern(&expr_spec.source),
                    kind: SourceKind::Script,
                    name_to_id: base_name_to_id.clone(),
                    output_binding: None,
                })
            }
            NodeKind::Plain(plain_spec) => {
                Some(CompilationUnit {
                    id: body_unit_id,
                    source: interner.intern(&plain_spec.source),
                    kind: SourceKind::Template,
                    name_to_id: base_name_to_id.clone(),
                    output_binding: None,
                })
            }
            _ => {
                // LLM / Iterator — body is messages/entries, handled separately.
                // We create an ExternDecl for the LLM call instead.
                None
            }
        };

        if let Some(unit) = body_unit {
            units.push(unit);
            node_map.body = Some(body_unit_id);
        }

        // ── LLM message units + ExternDecl ───────────────────────

        let messages = spec.kind.messages();
        if !messages.is_empty() {
            let mut message_unit_ids: Vec<(UnitId, Ty)> = Vec::new();

            for msg in messages {
                let msg_unit_id = alloc.unit();
                match msg {
                    MessageSpec::Block { source, .. } => {
                        units.push(CompilationUnit {
                            id: msg_unit_id,
                            source: interner.intern(source),
                            kind: SourceKind::Template,
                            name_to_id: base_name_to_id.clone(),
                            output_binding: None,
                        });
                        // Template tail_ty is Unit (templates emit text, don't return values).
                        // ExternDecl validates compilation success, not output type matching.
                        message_unit_ids.push((msg_unit_id, Ty::Unit));
                    }
                    MessageSpec::Iterator { key, .. } => {
                        units.push(CompilationUnit {
                            id: msg_unit_id,
                            source: *key,
                            kind: SourceKind::Script,
                            name_to_id: base_name_to_id.clone(),
                            output_binding: None,
                        });
                        // Iterator message: relaxed type check (various iterable types accepted).
                        message_unit_ids.push((msg_unit_id, Ty::error()));
                    }
                }
                node_map.messages.push(msg_unit_id);
            }

            // ExternDecl for the LLM call.
            let extern_id = alloc.unit();
            let output_ty = raw_output_ty.clone().unwrap_or_else(Ty::error);
            externs.push(ExternDecl {
                id: extern_id,
                inputs: message_unit_ids,
                output_ty: output_ty.clone(),
            });
            node_map.extern_decl = Some(extern_id);
            node_map.body = Some(extern_id); // LLM's "body" is the extern
        }

        // ── @raw context id ──────────────────────────────────────

        let raw_ctx_id = alloc.context();
        id_table.insert(raw_ctx_id, node_ns, interner.intern(KEY_RAW));

        // Determine which UnitId produces @raw.
        let raw_producer = node_map.body.unwrap_or(body_unit_id);

        // ── @self context id (if persistent or has initial_value) ─

        let self_ctx_id = if spec.strategy.initial_value.is_some()
            || !matches!(spec.strategy.persistency, Persistency::Ephemeral)
        {
            let id = alloc.context();
            id_table.insert(id, node_ns, interner.intern(KEY_SELF));
            Some(id)
        } else {
            None
        };

        // ── Strategy units: init, bind, assert, execution ────────

        // initial_value
        if let Some(init_src) = spec.strategy.initial_value {
            let init_id = alloc.unit();
            let mut init_name_to_id = base_name_to_id.clone();
            // init can see @self if it's a ScopeLocal (will be added to scope)
            // but doesn't get @self in name_to_id here — it's in the scope bindings.
            // init CAN see other nodes but NOT @self/@raw.
            units.push(CompilationUnit {
                id: init_id,
                source: init_src,
                kind: SourceKind::Script,
                name_to_id: init_name_to_id,
                output_binding: self_ctx_id, // output → @self
            });
            node_map.init = Some(init_id);
        }

        // bind
        if let Persistency::Sequence { bind } | Persistency::Patch { bind } = &spec.strategy.persistency {
            let bind_id = alloc.unit();
            let mut bind_name_to_id = base_name_to_id.clone();
            // bind can see @self and @raw
            if let Some(self_id) = self_ctx_id {
                bind_name_to_id.insert(interner.intern(KEY_SELF), self_id);
            }
            bind_name_to_id.insert(interner.intern(KEY_RAW), raw_ctx_id);
            units.push(CompilationUnit {
                id: bind_id,
                source: *bind,
                kind: SourceKind::Script,
                name_to_id: bind_name_to_id,
                output_binding: self_ctx_id, // output → @self
            });
            node_map.bind = Some(bind_id);
        }

        // assert
        if let Some(assert_src) = spec.strategy.assert {
            let assert_id = alloc.unit();
            let mut assert_name_to_id = base_name_to_id.clone();
            if let Some(self_id) = self_ctx_id {
                assert_name_to_id.insert(interner.intern(KEY_SELF), self_id);
            }
            assert_name_to_id.insert(interner.intern(KEY_RAW), raw_ctx_id);
            units.push(CompilationUnit {
                id: assert_id,
                source: assert_src,
                kind: SourceKind::Script,
                name_to_id: assert_name_to_id,
                output_binding: None,
            });
            node_map.assert = Some(assert_id);
        }

        // execution (IfModified key)
        if let Execution::IfModified { key } = &spec.strategy.execution {
            let exec_id = alloc.unit();
            // IfModified key: no @self, no @raw
            units.push(CompilationUnit {
                id: exec_id,
                source: *key,
                kind: SourceKind::Script,
                name_to_id: base_name_to_id.clone(),
                output_binding: None,
            });
            node_map.execution = Some(exec_id);
        }

        // ── Scope construction ───────────────────────────────────

        // Collect all units for this node's scope.
        let mut scope_units: Vec<UnitId> = Vec::new();
        let mut scope_bindings: Vec<ContextBinding> = Vec::new();

        // Body (if it's a compilation unit, not extern)
        if let Some(body_id) = node_map.body {
            scope_units.push(body_id);
        }
        if let Some(init_id) = node_map.init {
            scope_units.push(init_id);
        }
        if let Some(bind_id) = node_map.bind {
            scope_units.push(bind_id);
        }
        if let Some(assert_id) = node_map.assert {
            scope_units.push(assert_id);
        }
        if let Some(exec_id) = node_map.execution {
            scope_units.push(exec_id);
        }
        // ExternDecl is also in the scope (for DAG edge from inputs).
        if let Some(ext_id) = node_map.extern_decl {
            if !scope_units.contains(&ext_id) {
                scope_units.push(ext_id);
            }
        }

        // @raw binding
        scope_bindings.push(ContextBinding {
            id: raw_ctx_id,
            source: ContextSource::Derived(raw_producer, TypeTransform::Identity),
            constraint: None,
        });

        // @self binding (ScopeLocal with optional constraint)
        if let Some(self_id) = self_ctx_id {
            let constraint = match &spec.strategy.persistency {
                Persistency::Sequence { .. } => {
                    // Sequence enforcement: @self = Sequence<β, O, Pure>
                    // Use Ty::Var as placeholder — instantiate_constraint will replace.
                    use acvus_mir::ty::TySubst;
                    let mut cs = TySubst::new();
                    let beta = cs.fresh_var();
                    let origin = cs.fresh_origin();
                    Some(Ty::Sequence(Box::new(beta), origin, Effect::Pure))
                }
                _ => None,
            };
            scope_bindings.push(ContextBinding {
                id: self_id,
                source: ContextSource::ScopeLocal,
                constraint,
            });
        }

        if !scope_units.is_empty() {
            scopes.push(Scope {
                id: alloc.scope(),
                units: scope_units,
                bindings: scope_bindings,
            });
        }

        // ── Body unit also sees @self if initial_value exists ────
        // Add @self to body's name_to_id (body is already created, mutate in-place).
        if let Some(self_id) = self_ctx_id {
            if let Some(body_id) = node_map.body {
                if let Some(unit) = units.iter_mut().find(|u| u.id == body_id) {
                    unit.name_to_id.insert(interner.intern(KEY_SELF), self_id);
                }
            }
        }

        // ── Register node's visible type ─────────────────────────
        // The node's output ContextId (registered in global pass above) should
        // point to either the body output (ephemeral) or @self (persistent).
        // This is a Derived binding from the appropriate unit.
        // Note: we use node_ctx_ids[spec_idx] as the ContextId for this node.
        // We don't create a scope binding for this — it's a global-level fact.
        // The graph engine will resolve it through unit_outputs.

        manifest.node_units.push(node_map);
    }

    let graph = CompilationGraph {
        units,
        externs,
        scopes,
        externals,
        id_table,
    };

    (graph, manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_mir::context_registry::PartialContextTypeRegistry;
    use acvus_mir::ty::Ty;
    use crate::dsl::*;
    use crate::spec::{NodeKind, ExpressionSpec};

    fn empty_registry() -> PartialContextTypeRegistry {
        PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::default(),
        ).unwrap()
    }

    /// Simple Expression node: body = "1 + 2". Should resolve to Int.
    #[test]
    fn lower_simple_expression() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("counter"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "1 + 2".into(),
                output_ty: None,
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
        };

        let (graph, manifest) = lower(&interner, &[spec], &registry);

        // Should have at least one unit (body).
        assert!(!graph.units.is_empty(), "should have units");
        assert!(manifest.node_units[0].body.is_some(), "should have body unit");

        // Resolve should succeed.
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "resolve errors: {:?}", result.errors);
    }

    /// Expression with string body: "\"hello\"". Should resolve to String.
    #[test]
    fn lower_expression_string() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("greeter"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "\"hello\"".into(),
                output_ty: None,
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
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
    }

    /// Two Expression nodes where one references the other.
    #[test]
    fn lower_inter_node_reference() {
        let interner = Interner::new();
        let registry = empty_registry();
        let specs = vec![
            NodeSpec {
                name: interner.intern("a"),
                kind: NodeKind::Expression(ExpressionSpec {
                    source: "42".into(),
                    output_ty: None,
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
                name: interner.intern("b"),
                kind: NodeKind::Expression(ExpressionSpec {
                    source: "@a + 1".into(),
                    output_ty: None,
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
        ];

        let (graph, _) = lower(&interner, &specs, &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "inter-node reference should work: {:?}", result.errors);
    }

    /// Sequence node: init = "[]", bind = "@self | append(@raw)".
    /// This is the core init/bind/SCC pattern.
    #[test]
    fn lower_sequence_init_bind() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("history"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "\"msg\"".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::Sequence {
                    bind: interner.intern("@self | chain([@raw])"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, manifest) = lower(&interner, &[spec], &registry);

        // Should have body, init, bind units.
        let map = &manifest.node_units[0];
        assert!(map.body.is_some(), "should have body");
        assert!(map.init.is_some(), "should have init");
        assert!(map.bind.is_some(), "should have bind");

        // Should have a scope with ScopeLocal for @self.
        assert!(!graph.scopes.is_empty(), "should have scopes");
        let has_scope_local = graph.scopes.iter().any(|s|
            s.bindings.iter().any(|b| matches!(b.source, ContextSource::ScopeLocal))
        );
        assert!(has_scope_local, "should have ScopeLocal binding for @self");

        // Should have Sequence constraint.
        let has_constraint = graph.scopes.iter().any(|s|
            s.bindings.iter().any(|b| b.constraint.is_some())
        );
        assert!(has_constraint, "should have Sequence constraint");

        // Resolve should succeed — SCC resolves [] elem type via chain([@raw]).
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "SCC resolve should succeed: {:?}", result.errors);
    }

    /// Ephemeral with initial_value: init provides @self type.
    #[test]
    fn lower_ephemeral_with_initial_value() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("counter"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "@self + 1".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::Ephemeral,
                initial_value: Some(interner.intern("0")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "ephemeral with init should work: {:?}", result.errors);
    }

    /// External user param is visible to nodes.
    #[test]
    fn lower_with_user_params() {
        let interner = Interner::new();
        let registry = PartialContextTypeRegistry::new(
            FxHashMap::default(),
            FxHashMap::default(),
            FxHashMap::from_iter([(interner.intern("input"), Ty::String)]),
        ).unwrap();

        let spec = NodeSpec {
            name: interner.intern("echo"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "@input".into(),
                output_ty: None,
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
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "user param should be visible: {:?}", result.errors);
    }

    /// Assert unit is created and put in scope.
    #[test]
    fn lower_with_assert() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("guarded"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "42".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::Ephemeral,
                initial_value: Some(interner.intern("0")),
                retry: 3,
                assert: Some(interner.intern("@self > 0")),
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, manifest) = lower(&interner, &[spec], &registry);
        assert!(manifest.node_units[0].assert.is_some(), "should have assert unit");

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "assert should compile: {:?}", result.errors);
    }

    /// IfModified execution key unit is created.
    #[test]
    fn lower_if_modified() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("cached"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "42".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::IfModified { key: interner.intern("42") },
                persistency: Persistency::Ephemeral,
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, manifest) = lower(&interner, &[spec], &registry);
        assert!(manifest.node_units[0].execution.is_some(), "should have execution unit");

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "if_modified should work: {:?}", result.errors);
    }

    /// Sequence with chain bind: the actual pattern from the real codebase.
    /// body = LLM messages, init = [], bind = @self | chain(@raw | iter)
    /// This exercises: ExternDecl, SCC with @raw from extern, Sequence constraint.
    #[test]
    fn lower_sequence_chain_with_extern() {
        let interner = Interner::new();
        use crate::spec::{OpenAICompatibleSpec, MaxTokens};

        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("chat"),
            kind: NodeKind::OpenAICompatible(OpenAICompatibleSpec {
                endpoint: String::new(),
                api_key: String::new(),
                model: String::new(),
                messages: vec![MessageSpec::Block {
                    role: interner.intern("user"),
                    source: "hello".into(),
                }],
                tools: vec![],
                temperature: None,
                top_p: None,
                cache_key: None,
                max_tokens: MaxTokens::default(),
            }),
            strategy: Strategy {
                execution: Execution::OncePerTurn,
                persistency: Persistency::Sequence {
                    bind: interner.intern("@self | chain(@raw | iter)"),
                },
                initial_value: Some(interner.intern("[]")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, manifest) = lower(&interner, &[spec], &registry);
        assert!(manifest.node_units[0].extern_decl.is_some(), "LLM should have extern");
        assert!(manifest.node_units[0].init.is_some(), "should have init");
        assert!(manifest.node_units[0].bind.is_some(), "should have bind");

        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "LLM sequence should resolve: {:?}", result.errors);
    }

    /// Patch persistency: bind = "@raw" (identity).
    #[test]
    fn lower_patch_identity_bind() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("state"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "{count: 1,}".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::Patch {
                    bind: interner.intern("@raw"),
                },
                initial_value: Some(interner.intern("{count: 0,}")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "patch should resolve: {:?}", result.errors);
    }

    /// Body references @self (body is in the SCC).
    #[test]
    fn lower_body_references_self() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("acc"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "@self + 1".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::Patch {
                    bind: interner.intern("@raw"),
                },
                initial_value: Some(interner.intern("0")),
                retry: 0,
                assert: None,
            },
            is_function: false,
            fn_params: vec![],
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "body referencing @self should work: {:?}", result.errors);
    }

    /// Function node with params.
    #[test]
    fn lower_function_node() {
        let interner = Interner::new();
        let registry = empty_registry();
        let spec = NodeSpec {
            name: interner.intern("double"),
            kind: NodeKind::Expression(ExpressionSpec {
                source: "@x * 2".into(),
                output_ty: None,
            }),
            strategy: Strategy {
                execution: Execution::default(),
                persistency: Persistency::default(),
                initial_value: None,
                retry: 0,
                assert: None,
            },
            is_function: true,
            fn_params: vec![crate::dsl::FnParam {
                name: interner.intern("x"),
                ty: Ty::Int,
                description: None,
            }],
        };

        let (graph, _) = lower(&interner, &[spec], &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "function node should work: {:?}", result.errors);
    }

    /// Multiple nodes: a → b → c chain.
    #[test]
    fn lower_three_node_chain() {
        let interner = Interner::new();
        let registry = empty_registry();
        let specs = vec![
            NodeSpec {
                name: interner.intern("a"),
                kind: NodeKind::Expression(ExpressionSpec { source: "10".into(), output_ty: None }),
                strategy: Strategy { execution: Execution::default(), persistency: Persistency::default(), initial_value: None, retry: 0, assert: None },
                is_function: false, fn_params: vec![],
            },
            NodeSpec {
                name: interner.intern("b"),
                kind: NodeKind::Expression(ExpressionSpec { source: "@a + 1".into(), output_ty: None }),
                strategy: Strategy { execution: Execution::default(), persistency: Persistency::default(), initial_value: None, retry: 0, assert: None },
                is_function: false, fn_params: vec![],
            },
            NodeSpec {
                name: interner.intern("c"),
                kind: NodeKind::Expression(ExpressionSpec { source: "@b + 1".into(), output_ty: None }),
                strategy: Strategy { execution: Execution::default(), persistency: Persistency::default(), initial_value: None, retry: 0, assert: None },
                is_function: false, fn_params: vec![],
            },
        ];

        let (graph, _) = lower(&interner, &specs, &registry);
        let result = graph.resolve(&interner);
        assert!(result.errors.is_empty(), "three-node chain should work: {:?}", result.errors);
    }
}
