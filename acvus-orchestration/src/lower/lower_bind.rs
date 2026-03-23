use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id, MembershipId, SourceKind, UnitBody,
};
use acvus_utils::{Astr, Freeze};
use rustc_hash::FxHashMap;

pub fn lower(
    id: Id,
    bind_source: Astr,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
    membership: MembershipId,
) -> Entity {
    Entity {
        id,
        kind: EntityKind::Local {
            definition: LocalDefinition::LocalUnit(UnitBody {
                source: bind_source,
                kind: SourceKind::Script,
                name_to_id: name_to_id.clone(),
            }),
            membership: Some(membership),
        },
        constraint: Constraint::Inferred,
    }
}
