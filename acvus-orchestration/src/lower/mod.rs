//! Lowering: NodeSpec[] → LowerOutput.
//!
//! Phase 0: `lower()` — orchestration domain → flat Entity declarations.
//! Phase 1+2: `graph.compile()` — MIR handles type resolution + MIR compilation.
//! Assembly: `assemble()` — CompiledGraph + ExternFactory → runtime Units.

pub(crate) mod assemble;
mod lower_assert;
mod lower_bind;
mod lower_expression;
pub(crate) mod lower_llm;
mod lower_plain;

use std::sync::Arc;

use acvus_mir::graph::{
    CompilationGraph, ContextIdTable, Entity, EntityKind, Constraint,
    Id, LocalDefinition, MembershipId, NamespaceId, SourceKind, TypeTransform, UnitBody,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::dsl::{self, Execution, NodeSpec, Persistency};
use crate::spec::NodeKind;
use crate::unit::Unit;
use crate::unit::anthropic::AnthropicConfig;
use crate::unit::google::GoogleConfig;
use crate::unit::openai::OpenAIConfig;

// ── ExternFactory ───────────────────────────────────────────────────

pub enum ExternFactory {
    Assert { check_id: Id, value_id: Id, retry: u32 },
    Init { storage_read_id: Id, init_value_id: Id },
    OpenAI(Freeze<OpenAIConfig>),
    Anthropic(Freeze<AnthropicConfig>),
    Google(Freeze<GoogleConfig>),
    GoogleCache(Freeze<lower_llm::google_cache::GoogleCacheConfig>),
}

// ── LoweredUnit ─────────────────────────────────────────────────────

enum LoweredUnit {
    Local  { entity: Entity, meta: UnitMeta },
    Extern { entity: Entity, meta: UnitMeta, factory: ExternFactory },
    Context { entity: Entity },
}

// ── TemporalBinding ─────────────────────────────────────────────────

pub struct TemporalBinding {
    pub read: Id,
    pub write: Id,
}

// ── UnitMeta ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UnitPolicy {
    pub execution: Execution,
    pub persistency: Persistency,
}

#[derive(Debug, Clone)]
pub struct UnitMeta {
    pub name: Astr,
    pub policy: UnitPolicy,
}

impl UnitMeta {
    pub fn serialized(&self) -> bool {
        !matches!(self.policy.persistency, Persistency::Ephemeral)
    }
}

// ── LowerCollector ──────────────────────────────────────────────────

struct LowerCollector {
    entities: Vec<Entity>,
    factories: FxHashMap<Id, ExternFactory>,
    metas: FxHashMap<Id, UnitMeta>,
    temporal_bindings: Vec<TemporalBinding>,
}

impl LowerCollector {
    fn new() -> Self {
        Self {
            entities: Vec::new(),
            factories: FxHashMap::default(),
            metas: FxHashMap::default(),
            temporal_bindings: Vec::new(),
        }
    }

    fn push(&mut self, unit: LoweredUnit) {
        match unit {
            LoweredUnit::Local { entity, meta } => {
                let id = entity.id;
                self.entities.push(entity);
                self.metas.insert(id, meta);
            }
            LoweredUnit::Extern { entity, meta, factory } => {
                let id = entity.id;
                self.entities.push(entity);
                self.metas.insert(id, meta);
                self.factories.insert(id, factory);
            }
            LoweredUnit::Context { entity } => {
                self.entities.push(entity);
            }
        }
    }

    fn finish_node(&mut self, temporal: Option<TemporalBinding>) {
        if let Some(t) = temporal {
            self.temporal_bindings.push(t);
        }
    }

    fn into_lower_output(self, id_table: ContextIdTable) -> LowerOutput {
        LowerOutput {
            graph: CompilationGraph {
                entities: self.entities,
                id_table: Freeze::new(id_table),
            },
            extern_factories: self.factories,
            unit_meta: self.metas,
            temporal_bindings: self.temporal_bindings,
        }
    }
}

// ── LowerOutput ─────────────────────────────────────────────────────

pub struct LowerOutput {
    pub graph: CompilationGraph,
    pub extern_factories: FxHashMap<Id, ExternFactory>,
    pub unit_meta: FxHashMap<Id, UnitMeta>,
    pub temporal_bindings: Vec<TemporalBinding>,
}

// ── LowerResult ─────────────────────────────────────────────────────

pub struct LowerResult {
    pub units: Freeze<FxHashMap<Id, (Arc<dyn Unit>, UnitMeta)>>,
    pub entrypoint: Id,
    pub id_table: Freeze<ContextIdTable>,
}

impl LowerResult {
    pub fn unit(&self, id: Id) -> &dyn Unit {
        &*self.units.get(&id)
            .unwrap_or_else(|| panic!("no unit for {id:?}"))
            .0
    }

    pub fn meta(&self, id: Id) -> Option<&UnitMeta> {
        self.units.get(&id).map(|(_, m)| m)
    }

    pub fn has(&self, id: Id) -> bool {
        self.units.contains_key(&id)
    }
}

// ── Phase 0: Lowering ───────────────────────────────────────────────

pub fn lower(
    interner: &Interner,
    specs: &[NodeSpec],
    external_contexts: &[(Astr, Ty)],
) -> LowerOutput {
    let mut id_table = ContextIdTable::new();
    let mut collector = LowerCollector::new();
    let global_ns = NamespaceId::alloc();

    let mut global_names: FxHashMap<Astr, Id> = FxHashMap::default();

    for (name, ty) in external_contexts {
        let id = Id::alloc();
        global_names.insert(*name, id);
        id_table.insert(id, global_ns, *name);
        collector.push(LoweredUnit::Context {
            entity: Entity {
                id,
                kind: EntityKind::External,
                constraint: Constraint::Exact(ty.clone()),
            },
        });
    }

    let node_provides: Vec<Id> = specs.iter().map(|spec| {
        let id = Id::alloc();
        global_names.insert(spec.name, id);
        id_table.insert(id, global_ns, spec.name);
        id
    }).collect();

    let global_names = Freeze::new(global_names);

    for (spec, &provides_id) in specs.iter().zip(&node_provides) {
        lower_node_graph(interner, spec, &global_names, provides_id, &mut id_table, &mut collector);
    }

    collector.into_lower_output(id_table)
}

// ── Per-node lowering ───────────────────────────────────────────────

fn lower_node_graph(
    interner: &Interner,
    spec: &NodeSpec,
    global_names: &Freeze<FxHashMap<Astr, Id>>,
    provides_id: Id,
    id_table: &mut ContextIdTable,
    collector: &mut LowerCollector,
) {
    let node_ns = NamespaceId::alloc();
    let node_name = interner.resolve(spec.name);

    let internal = UnitPolicy {
        execution: Execution::Always,
        persistency: Persistency::Ephemeral,
    };
    let entrypoint_policy = UnitPolicy {
        execution: spec.strategy.execution.clone(),
        persistency: spec.strategy.persistency.clone(),
    };
    let meta = |name: &str, policy: &UnitPolicy| -> UnitMeta {
        UnitMeta {
            name: interner.intern(&format!("{node_name}:{name}")),
            policy: policy.clone(),
        }
    };

    // @raw (type system only — runtime uses body_output_id directly)
    let raw_ctx_id = Id::alloc();
    id_table.insert(raw_ctx_id, node_ns, interner.intern(dsl::KEY_RAW));

    let has_state = matches!(
        spec.strategy.persistency,
        Persistency::Sequence { .. } | Persistency::Patch { .. }
    );
    let init_id = if has_state { Some(Id::alloc()) } else { None };
    let membership = if has_state { Some(MembershipId::alloc()) } else { None };

    // Body name_to_id: @self → init_id (if stateful)
    let body_name_to_id = {
        let mut m = (**global_names).clone();
        if let Some(init) = init_id {
            m.insert(interner.intern(dsl::KEY_SELF), init);
        }
        for p in &spec.fn_params {
            let param_id = Id::alloc();
            id_table.insert(param_id, node_ns, p.name);
            m.insert(p.name, param_id);
        }
        Freeze::new(m)
    };

    // Body entities
    let body_output_id = lower_body(interner, spec, &body_name_to_id, &meta, &internal, collector);

    // @raw context
    collector.push(LoweredUnit::Context {
        entity: Entity {
            id: raw_ctx_id,
            kind: EntityKind::Local { definition: LocalDefinition::Context, membership: None },
            constraint: Constraint::Derived(body_output_id, TypeTransform::Identity),
        },
    });

    // Assert check sees @raw = body_output_id
    let assert_name_to_id = {
        let mut m = (*body_name_to_id).clone();
        m.insert(interner.intern(dsl::KEY_RAW), body_output_id);
        Freeze::new(m)
    };

    match &spec.strategy.persistency {
        Persistency::Ephemeral => {
            // Stateless: body → assert(=provides_id)
            let assert_entries = lower_assert::lower(
                provides_id,
                spec.strategy.assert,
                &assert_name_to_id,
                body_output_id,
            );
            id_table.insert(assert_entries.check_id, node_ns, interner.intern("assert_check"));

            collector.push(LoweredUnit::Local {
                entity: assert_entries.check_entity,
                meta: meta("assert_check", &internal),
            });
            collector.push(LoweredUnit::Extern {
                entity: assert_entries.assert_entity,
                meta: meta("assert", &entrypoint_policy),
                factory: ExternFactory::Assert {
                    check_id: assert_entries.check_id,
                    value_id: body_output_id,
                    retry: spec.strategy.retry,
                },
            });

            collector.finish_node(None);
        }

        Persistency::Sequence { initial_value, bind } |
        Persistency::Patch { initial_value, bind } => {
            // Stateful: body → assert → init → bind(=provides_id)
            let init_id = init_id.expect("has_state");
            let membership = membership.expect("has_state");
            let assert_id = Id::alloc();

            // Assert
            let assert_entries = lower_assert::lower(
                assert_id,
                spec.strategy.assert,
                &assert_name_to_id,
                body_output_id,
            );
            id_table.insert(assert_entries.check_id, node_ns, interner.intern("assert_check"));
            id_table.insert(assert_entries.assert_id, node_ns, interner.intern("assert"));

            collector.push(LoweredUnit::Local {
                entity: assert_entries.check_entity,
                meta: meta("assert_check", &internal),
            });
            collector.push(LoweredUnit::Extern {
                entity: assert_entries.assert_entity,
                meta: meta("assert", &internal),
                factory: ExternFactory::Assert {
                    check_id: assert_entries.check_id,
                    value_id: body_output_id,
                    retry: spec.strategy.retry,
                },
            });

            // storage_read — Context for temporal old value
            let storage_read_id = Id::alloc();
            id_table.insert(storage_read_id, node_ns, interner.intern("self_storage"));
            collector.push(LoweredUnit::Context {
                entity: Entity {
                    id: storage_read_id,
                    kind: EntityKind::Local { definition: LocalDefinition::Context, membership: None },
                    constraint: Constraint::Inferred,
                },
            });

            // init_value — LocalUnit (initial_value expression)
            let init_value_id = Id::alloc();
            id_table.insert(init_value_id, node_ns, interner.intern("init_value"));
            collector.push(LoweredUnit::Local {
                entity: Entity {
                    id: init_value_id,
                    kind: EntityKind::Local {
                        definition: LocalDefinition::LocalUnit(UnitBody {
                            source: *initial_value,
                            kind: SourceKind::Script,
                            name_to_id: global_names.clone(),
                        }),
                        membership: Some(membership),
                    },
                    constraint: Constraint::Inferred,
                },
                meta: meta("init_value", &internal),
            });

            // init — ExternUnit (storage gate)
            id_table.insert(init_id, node_ns, interner.intern("init"));
            collector.push(LoweredUnit::Extern {
                entity: Entity {
                    id: init_id,
                    kind: EntityKind::Local {
                        definition: LocalDefinition::ExternUnit {
                            refs: vec![
                                (storage_read_id, Ty::infer()),
                                (init_value_id, Ty::infer()),
                            ],
                        },
                        membership: Some(membership),
                    },
                    constraint: Constraint::Inferred,
                },
                meta: meta("init", &internal),
                factory: ExternFactory::Init { storage_read_id, init_value_id },
            });

            // bind — LocalUnit (=provides_id)
            let bind_name_to_id = {
                let mut m = (*body_name_to_id).clone();
                m.insert(interner.intern(dsl::KEY_RAW), assert_id);
                Freeze::new(m)
            };
            collector.push(LoweredUnit::Local {
                entity: lower_bind::lower(provides_id, *bind, &bind_name_to_id, membership),
                meta: meta("bind", &entrypoint_policy),
            });

            collector.finish_node(Some(TemporalBinding {
                read: storage_read_id,
                write: provides_id,
            }));
        }
    }
}

// ── Body lowering ───────────────────────────────────────────────────

fn lower_body(
    interner: &Interner,
    spec: &NodeSpec,
    body_name_to_id: &Freeze<FxHashMap<Astr, Id>>,
    meta: &dyn Fn(&str, &UnitPolicy) -> UnitMeta,
    internal: &UnitPolicy,
    collector: &mut LowerCollector,
) -> Id {
    match &spec.kind {
        NodeKind::Expression(expr) => {
            let (entity, id) = lower_expression::lower(interner, expr, body_name_to_id);
            collector.push(LoweredUnit::Local { entity, meta: meta("body", internal) });
            id
        }
        NodeKind::Plain(plain) => {
            let (entity, id) = lower_plain::lower(interner, plain, body_name_to_id);
            collector.push(LoweredUnit::Local { entity, meta: meta("body", internal) });
            id
        }
        NodeKind::OpenAICompatible(s) => {
            let (mut ents, id, config) = lower_llm::openai::lower(interner, s, body_name_to_id);
            let config = Freeze::new(config);
            push_llm_entities(&mut ents, id, &meta, internal, ExternFactory::OpenAI(config), collector);
            id
        }
        NodeKind::Anthropic(s) => {
            let (mut ents, id, config) = lower_llm::anthropic::lower(interner, s, body_name_to_id);
            let config = Freeze::new(config);
            push_llm_entities(&mut ents, id, &meta, internal, ExternFactory::Anthropic(config), collector);
            id
        }
        NodeKind::GoogleAI(s) => {
            let (mut ents, id, config) = lower_llm::google::lower(interner, s, body_name_to_id);
            let config = Freeze::new(config);
            push_llm_entities(&mut ents, id, &meta, internal, ExternFactory::Google(config), collector);
            id
        }
        NodeKind::GoogleAICache(s) => {
            let (mut ents, id, config) = lower_llm::google_cache::lower(interner, s, body_name_to_id);
            let config = Freeze::new(config);
            push_llm_entities(&mut ents, id, &meta, internal, ExternFactory::GoogleCache(config), collector);
            id
        }
    }
}

fn push_llm_entities(
    ents: &mut Vec<Entity>,
    llm_id: Id,
    meta: &dyn Fn(&str, &UnitPolicy) -> UnitMeta,
    internal: &UnitPolicy,
    factory: ExternFactory,
    collector: &mut LowerCollector,
) {
    let llm_idx = ents.iter().position(|e| e.id == llm_id).expect("LLM entity not found");
    let llm_entity = ents.remove(llm_idx);

    for (i, entity) in ents.drain(..).enumerate() {
        collector.push(LoweredUnit::Local { entity, meta: meta(&format!("msg{i}"), internal) });
    }
    collector.push(LoweredUnit::Extern {
        entity: llm_entity,
        meta: meta("llm", internal),
        factory,
    });
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Strategy;
    use crate::spec::ExpressionSpec;

    fn interner() -> Interner { Interner::new() }

    fn default_strategy(interner: &Interner) -> Strategy {
        Strategy {
            execution: Execution::default(),
            persistency: Persistency::default(),
            retry: 0,
            assert: interner.intern("true"),
        }
    }

    fn expr_node(interner: &Interner, name: &str, source: &str) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec { source: source.to_string(), output_ty: None }),
            strategy: default_strategy(interner),
            is_function: false,
            fn_params: vec![],
        }
    }

    fn expr_node_with_strategy(interner: &Interner, name: &str, source: &str, strategy: Strategy) -> NodeSpec {
        NodeSpec {
            name: interner.intern(name),
            kind: NodeKind::Expression(ExpressionSpec { source: source.to_string(), output_ty: None }),
            strategy,
            is_function: false,
            fn_params: vec![],
        }
    }

    fn sequence_strategy(interner: &Interner, init: &str, bind: &str) -> Strategy {
        Strategy {
            execution: Execution::default(),
            persistency: Persistency::Sequence { initial_value: interner.intern(init), bind: interner.intern(bind) },
            retry: 0,
            assert: interner.intern("true"),
        }
    }

    fn local_units(graph: &CompilationGraph) -> Vec<&Entity> {
        graph.entities.iter()
            .filter(|e| matches!(&e.kind, EntityKind::Local { definition: LocalDefinition::LocalUnit(_), .. }))
            .collect()
    }

    fn extern_units(graph: &CompilationGraph) -> Vec<&Entity> {
        graph.entities.iter()
            .filter(|e| matches!(&e.kind, EntityKind::Local { definition: LocalDefinition::ExternUnit { .. }, .. }))
            .collect()
    }

    fn contexts(graph: &CompilationGraph) -> Vec<&Entity> {
        graph.entities.iter()
            .filter(|e| matches!(&e.kind, EntityKind::Local { definition: LocalDefinition::Context, .. }))
            .collect()
    }

    fn externals(graph: &CompilationGraph) -> Vec<&Entity> {
        graph.entities.iter()
            .filter(|e| matches!(&e.kind, EntityKind::External))
            .collect()
    }

    fn membership_of(entity: &Entity) -> Option<MembershipId> {
        match &entity.kind {
            EntityKind::Local { membership, .. } => *membership,
            _ => None,
        }
    }

    fn name_of(graph: &CompilationGraph, id: Id) -> Option<Astr> {
        graph.id_table.get(id).map(|(_, name)| name)
    }

    fn lower_graph(interner: &Interner, specs: &[NodeSpec], ext: &[(Astr, Ty)]) -> CompilationGraph {
        lower(interner, specs, ext).graph
    }

    // ── Ephemeral ──

    #[test]
    fn ephemeral_entity_count() {
        let i = interner();
        let graph = lower_graph(&i, &[expr_node(&i, "counter", "1 + 2")], &[]);
        assert_eq!(local_units(&graph).len(), 2);   // body + assert_check
        assert_eq!(extern_units(&graph).len(), 1);   // assert(=provides)
        assert_eq!(contexts(&graph).len(), 1);        // @raw
        assert_eq!(graph.entities.len(), 4);
    }

    #[test]
    fn ephemeral_no_temporal() {
        let i = interner();
        let output = lower(&i, &[expr_node(&i, "x", "1")], &[]);
        assert!(output.temporal_bindings.is_empty());
    }

    #[test]
    fn ephemeral_provides_is_assert() {
        let i = interner();
        let graph = lower_graph(&i, &[expr_node(&i, "counter", "1 + 2")], &[]);
        let provides = graph.entities.iter()
            .find(|e| name_of(&graph, e.id).map(|n| i.resolve(n)) == Some("counter")).unwrap();
        assert!(matches!(&provides.kind, EntityKind::Local { definition: LocalDefinition::ExternUnit { .. }, .. }));
    }

    // ── Stateful ──

    #[test]
    fn stateful_entity_count() {
        let i = interner();
        let spec = expr_node_with_strategy(&i, "full", "1", Strategy {
            execution: Execution::default(),
            persistency: Persistency::Sequence { initial_value: i.intern("[]"), bind: i.intern("@raw") },
            retry: 0,
            assert: i.intern("len(@raw) > 0"),
        });
        let output = lower(&i, &[spec], &[]);
        assert_eq!(local_units(&output.graph).len(), 4);   // body + assert_check + init_value + bind(=provides)
        assert_eq!(extern_units(&output.graph).len(), 2);   // assert + init
        assert_eq!(contexts(&output.graph).len(), 2);        // @raw + storage_read
    }

    #[test]
    fn stateful_has_temporal() {
        let i = interner();
        let output = lower(&i, &[expr_node_with_strategy(&i, "h", "@self", sequence_strategy(&i, "[]", "@raw"))], &[]);
        assert_eq!(output.temporal_bindings.len(), 1);
    }

    #[test]
    fn stateful_membership() {
        let i = interner();
        let graph = lower_graph(&i, &[expr_node_with_strategy(&i, "h", "@self", sequence_strategy(&i, "[]", "@raw"))], &[]);
        let bind = local_units(&graph).into_iter()
            .find(|e| name_of(&graph, e.id).map(|n| i.resolve(n)) == Some("bind")
                || name_of(&graph, e.id).map(|n| i.resolve(n)) == Some("h")).unwrap();
        let init_value = local_units(&graph).into_iter()
            .find(|e| name_of(&graph, e.id).map(|n| i.resolve(n)) == Some("init_value")).unwrap();
        let mid = membership_of(bind).expect("bind must have membership");
        assert_eq!(membership_of(init_value), Some(mid));
    }

    #[test]
    fn stateful_bind_sees_raw_and_self() {
        let i = interner();
        let graph = lower_graph(&i, &[expr_node_with_strategy(&i, "h", "@self", sequence_strategy(&i, "[]", "@self | chain(@raw | iter)"))], &[]);
        let bind = local_units(&graph).into_iter()
            .find(|e| {
                if let EntityKind::Local { definition: LocalDefinition::LocalUnit(body), .. } = &e.kind {
                    body.name_to_id.contains_key(&i.intern(dsl::KEY_RAW))
                } else { false }
            }).expect("bind should see @raw");
        let body = match &bind.kind {
            EntityKind::Local { definition: LocalDefinition::LocalUnit(body), .. } => body,
            _ => panic!(),
        };
        assert!(body.name_to_id.contains_key(&i.intern(dsl::KEY_SELF)));
    }

    #[test]
    fn stateful_init_factory_exists() {
        let i = interner();
        let output = lower(&i, &[expr_node_with_strategy(&i, "c", "1", sequence_strategy(&i, "0", "@raw"))], &[]);
        assert!(output.extern_factories.values().any(|f| matches!(f, ExternFactory::Init { .. })));
    }

    // ── Policy ──

    #[test]
    fn entrypoint_has_node_execution() {
        let i = interner();
        let spec = expr_node_with_strategy(&i, "p", "1", Strategy {
            execution: Execution::OncePerTurn,
            persistency: Persistency::Patch { initial_value: i.intern("0"), bind: i.intern("@raw") },
            retry: 0,
            assert: i.intern("true"),
        });
        let output = lower(&i, &[spec], &[]);
        let provides = output.graph.entities.iter()
            .find(|e| name_of(&output.graph, e.id).map(|n| i.resolve(n)) == Some("p")).unwrap();
        let meta = &output.unit_meta[&provides.id];
        assert!(matches!(meta.policy.execution, Execution::OncePerTurn));
    }

    // ── Multi-node ──

    #[test]
    fn multi_node_cross_reference() {
        let i = interner();
        let output = lower(&i, &[expr_node(&i, "a", "1"), expr_node(&i, "b", "@a + 1")], &[]);
        let a_key = i.intern("a");
        let a_id = *output.unit_meta.iter()
            .find(|(_, m)| i.resolve(m.name) == "a:assert")
            .map(|(id, _)| id).unwrap();
        let b_body = local_units(&output.graph).into_iter()
            .find(|e| {
                if let EntityKind::Local { definition: LocalDefinition::LocalUnit(body), .. } = &e.kind {
                    i.resolve(body.source).contains("@a")
                } else { false }
            }).unwrap();
        let body = match &b_body.kind {
            EntityKind::Local { definition: LocalDefinition::LocalUnit(body), .. } => body,
            _ => unreachable!(),
        };
        assert_eq!(body.name_to_id.get(&a_key), Some(&a_id));
    }

    // ── External contexts ──

    #[test]
    fn external_contexts() {
        let i = interner();
        let ext = vec![(i.intern("input"), Ty::String)];
        let graph = lower_graph(&i, &[expr_node(&i, "echo", "@input")], &ext);
        assert_eq!(externals(&graph).len(), 1);
        let input_id = externals(&graph)[0].id;
        let body = &local_units(&graph)[0];
        let b = match &body.kind {
            EntityKind::Local { definition: LocalDefinition::LocalUnit(b), .. } => b,
            _ => panic!(),
        };
        assert_eq!(b.name_to_id.get(&i.intern("input")), Some(&input_id));
    }

    // ── Plain ──

    #[test]
    fn plain_node_template() {
        let i = interner();
        let spec = NodeSpec {
            name: i.intern("greeting"),
            kind: NodeKind::Plain(crate::spec::PlainSpec { source: "Hello {{@input}}".to_string() }),
            strategy: default_strategy(&i),
            is_function: false,
            fn_params: vec![],
        };
        let graph = lower_graph(&i, &[spec], &[]);
        let body = &local_units(&graph)[0];
        let b = match &body.kind {
            EntityKind::Local { definition: LocalDefinition::LocalUnit(b), .. } => b,
            _ => panic!(),
        };
        assert_eq!(b.kind, SourceKind::Template);
    }
}
