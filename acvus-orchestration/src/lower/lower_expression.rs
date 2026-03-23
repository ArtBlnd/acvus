use acvus_mir::graph::{
    Entity, EntityKind, Constraint, LocalDefinition,
    Id, SourceKind, UnitBody,
};
use acvus_utils::{Astr, Freeze, Interner};
use rustc_hash::FxHashMap;

use crate::spec::ExpressionSpec;

pub fn lower(
    interner: &Interner,
    expr: &ExpressionSpec,
    name_to_id: &Freeze<FxHashMap<Astr, Id>>,
) -> (Entity, Id) {
    let id = Id::alloc();
    let constraint = match &expr.output_ty {
        Some(ty) => Constraint::Exact(ty.clone()),
        None => Constraint::Inferred,
    };
    let entity = Entity {
        id,
        kind: EntityKind::Local {
            definition: LocalDefinition::LocalUnit(UnitBody {
                source: interner.intern(&expr.source),
                kind: SourceKind::Script,
                name_to_id: name_to_id.clone(),
            }),
            membership: None,
        },
        constraint,
    };
    (entity, id)
}
