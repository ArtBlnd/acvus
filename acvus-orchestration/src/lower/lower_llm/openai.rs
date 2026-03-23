use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::dsl::{MessageSpec, message_elem_ty};
use crate::spec::OpenAICompatibleSpec;
use crate::unit::openai::{OpenAIConfig, ToolBinding};

/// Returns (entities, extern_id, config).
pub fn lower(
    interner: &Interner,
    spec: &OpenAICompatibleSpec,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
) -> (Vec<Entity>, Id, OpenAIConfig) {
    let mut entities = Vec::new();
    let mut message_ids: Vec<(Astr, Id)> = Vec::new();

    for msg in &spec.messages {
        match msg {
            MessageSpec::Block { role, source } => {
                let (entity, id, role) = super::lower_message(interner, *role, source, name_to_id);
                entities.push(entity);
                message_ids.push((role, id));
            }
            _ => {}
        }
    }

    let extern_id = Id::alloc();
    let refs: Vec<(Id, Ty)> = message_ids.iter()
        .map(|(_, id)| (*id, Ty::String))
        .collect();
    let output_ty = Ty::List(Box::new(message_elem_ty(interner)));

    entities.push(Entity {
        id: extern_id,
        kind: EntityKind::Local {
            definition: LocalDefinition::ExternUnit { refs },
            membership: None,
        },
        constraint: Constraint::Exact(output_ty),
    });

    let tools = spec.tools.iter().map(|t| {
        let unit_id = *name_to_id.get(&interner.intern(&t.node))
            .unwrap_or_else(|| panic!("tool node '{}' not found in name_to_id", t.node));
        ToolBinding { name: t.name.clone(), description: t.description.clone(), unit_id }
    }).collect();

    let config = OpenAIConfig {
        endpoint: spec.endpoint.clone(),
        api_key: spec.api_key.clone(),
        model: spec.model.clone(),
        message_ids,
        tools,
        temperature: spec.temperature,
        top_p: spec.top_p,
        max_tokens: spec.max_tokens.output,
        max_tool_rounds: 10,
    };

    (entities, extern_id, config)
}
