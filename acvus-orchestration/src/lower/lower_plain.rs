use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id, SourceKind, UnitBody,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::spec::PlainSpec;

pub fn lower(
    interner: &Interner,
    plain: &PlainSpec,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
) -> (Entity, Id) {
    let id = Id::alloc();
    let entity = Entity {
        id,
        kind: EntityKind::Local {
            definition: LocalDefinition::LocalUnit(UnitBody {
                source: interner.intern(&plain.source),
                kind: SourceKind::Template,
                name_to_id: name_to_id.clone(),
            }),
            membership: None,
        },
        constraint: Constraint::Inferred,
    };
    (entity, id)
}
