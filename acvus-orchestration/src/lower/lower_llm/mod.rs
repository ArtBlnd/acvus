pub mod openai;
pub mod anthropic;
pub mod google;
pub mod google_cache;

use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id, SourceKind, UnitBody,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

/// Lower a single message template → Entity (LocalUnit, Template).
/// Common across all LLM providers.
pub fn lower_message(
    interner: &Interner,
    role: Astr,
    source: &str,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
) -> (Entity, Id, Astr) {
    let id = Id::alloc();
    let entity = Entity {
        id,
        kind: EntityKind::Local {
            definition: LocalDefinition::LocalUnit(UnitBody {
                source: interner.intern(source),
                kind: SourceKind::Template,
                name_to_id: name_to_id.clone(),
            }),
            membership: None,
        },
        constraint: Constraint::Inferred,
    };
    (entity, id, role)
}
