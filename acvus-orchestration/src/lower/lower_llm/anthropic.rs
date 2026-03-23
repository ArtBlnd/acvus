use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::dsl::{MessageSpec, message_elem_ty};
use crate::spec::AnthropicSpec;
use crate::unit::anthropic::AnthropicConfig;

/// Returns (entities, extern_id, config).
pub fn lower(
    interner: &Interner,
    spec: &AnthropicSpec,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
) -> (Vec<Entity>, Id, AnthropicConfig) {
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

    entities.push(Entity {
        id: extern_id,
        kind: EntityKind::Local {
            definition: LocalDefinition::ExternUnit { refs },
            membership: None,
        },
        constraint: Constraint::Exact(Ty::List(Box::new(message_elem_ty(interner)))),
    });

    let config = AnthropicConfig {
        endpoint: spec.endpoint.clone(),
        api_key: spec.api_key.clone(),
        model: spec.model.clone(),
        message_ids,
        temperature: spec.temperature,
        top_p: spec.top_p,
        top_k: spec.top_k,
        max_tokens: spec.max_tokens.output,
    };

    (entities, extern_id, config)
}
