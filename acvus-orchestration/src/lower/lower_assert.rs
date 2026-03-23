use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id, SourceKind, TypeTransform, UnitBody,
};
use acvus_mir::ty::Ty;
use acvus_utils::{Astr, Freeze};
use rustc_hash::FxHashMap;

pub struct AssertEntries {
    pub check_entity: Entity,
    pub check_id: Id,
    pub assert_entity: Entity,
    pub assert_id: Id,
}

pub fn lower(
    assert_id: Id,
    assert_source: Astr,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
    value_id: Id,
) -> AssertEntries {
    let check_id = Id::alloc();
    let check_entity = Entity {
        id: check_id,
        kind: EntityKind::Local {
            definition: LocalDefinition::LocalUnit(UnitBody {
                source: assert_source,
                kind: SourceKind::Script,
                name_to_id: name_to_id.clone(),
            }),
            membership: None,
        },
        constraint: Constraint::Inferred,
    };

    let assert_entity = Entity {
        id: assert_id,
        kind: EntityKind::Local {
            definition: LocalDefinition::ExternUnit {
                refs: vec![
                    (check_id, Ty::Bool),
                    (value_id, Ty::infer()),
                ],
            },
            membership: None,
        },
        constraint: Constraint::Derived(value_id, TypeTransform::Identity),
    };

    AssertEntries { check_entity, check_id, assert_entity, assert_id }
}
