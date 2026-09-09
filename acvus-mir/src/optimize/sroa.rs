//! SROA (Scalar Replacement of Aggregates) for Ref projections.
//!
//! Decomposes field-level Ref projections into identity Ref + scalar FieldGet/FieldSet.
//! This enables SSA promotion (mem2reg) which only handles identity Refs.
//!
//! ## Independence
//!
//! This pass is optional. If skipped, field Refs remain in the IR and the
//! interpreter handles them directly. SSA will skip variables that have
//! field Refs (non-promotable), so correctness is preserved either way.

use rustc_hash::FxHashMap;

use acvus_utils::Astr;

use crate::graph::QualifiedRef;
use crate::ir::{Inst, InstKind, MirBody, RefTarget};
use crate::ty::Ty;

/// Collect the whole type for each named storage.
/// Sources: context_types (authoritative), identity Refs in body (for vars/params).
fn build_whole_types(
    body: &MirBody,
    context_types: &FxHashMap<QualifiedRef, Ty>,
) -> FxHashMap<RefTarget, Ty> {
    let mut map: FxHashMap<RefTarget, Ty> = FxHashMap::default();

    // Context types from InferResult (authoritative).
    for (qref, ty) in context_types.iter() {
        map.insert(RefTarget::Context(*qref), ty.clone());
    }

    // Var/Param types from identity Refs in body.
    for inst in &body.insts {
        if let InstKind::Ref { dst, target, path } = &inst.kind
            && path.is_empty()
            && let Some(Ty::Ref(inner)) = body.val_types.get(dst)
        {
            map.entry(target.clone()).or_insert_with(|| *inner.clone());
        }
    }

    map
}

/// Run SROA on a single MirBody.
pub fn run_body(body: &mut MirBody, context_types: &FxHashMap<QualifiedRef, Ty>) {
    let whole_types = build_whole_types(body, context_types);

    let mut new_insts: Vec<Inst> = Vec::with_capacity(body.insts.len());
    let mut i = 0;

    while i < body.insts.len() {
        let is_field_ref =
            matches!(&body.insts[i].kind, InstKind::Ref { path, .. } if !path.is_empty());

        if !is_field_ref || i + 1 >= body.insts.len() {
            new_insts.push(body.insts[i].clone());
            i += 1;
            continue;
        }

        let InstKind::Ref {
            dst: ref_dst,
            target,
            path,
        } = &body.insts[i].kind
        else {
            unreachable!();
        };
        let target = target.clone();
        let field = path[0];
        let rest: Vec<Astr> = path[1..].to_vec();
        let ref_dst = *ref_dst;
        let span = body.insts[i].span;

        let whole_ty = match whole_types.get(&target) {
            Some(ty) => ty.clone(),
            None => {
                // Can't determine whole type - leave as-is.
                new_insts.push(body.insts[i].clone());
                i += 1;
                continue;
            }
        };

        match &body.insts[i + 1].kind {
            InstKind::Load {
                dst: load_dst, src, ..
            } if *src == ref_dst => {
                let load_dst = *load_dst;
                let load_span = body.insts[i + 1].span;

                let identity_ref = body.val_factory.next();
                body.val_types
                    .insert(identity_ref, Ty::Ref(Box::new(whole_ty.clone())));
                new_insts.push(Inst {
                    span,
                    kind: InstKind::Ref {
                        dst: identity_ref,
                        target: target.clone(),
                        path: vec![],
                    },
                });

                let tmp = body.val_factory.next();
                body.val_types.insert(tmp, whole_ty);
                new_insts.push(Inst {
                    span: load_span,
                    kind: InstKind::Load {
                        dst: tmp,
                        src: identity_ref,
                    },
                });

                new_insts.push(Inst {
                    span: load_span,
                    kind: InstKind::FieldGet {
                        dst: load_dst,
                        object: tmp,
                        field,
                        rest: rest.clone(),
                    },
                });

                i += 2;
            }

            InstKind::Store {
                dst: store_dst,
                value,
                ..
            } if *store_dst == ref_dst => {
                let value = *value;
                let store_span = body.insts[i + 1].span;

                let ref_for_load = body.val_factory.next();
                body.val_types
                    .insert(ref_for_load, Ty::Ref(Box::new(whole_ty.clone())));
                new_insts.push(Inst {
                    span,
                    kind: InstKind::Ref {
                        dst: ref_for_load,
                        target: target.clone(),
                        path: vec![],
                    },
                });

                let old = body.val_factory.next();
                body.val_types.insert(old, whole_ty.clone());
                new_insts.push(Inst {
                    span: store_span,
                    kind: InstKind::Load {
                        dst: old,
                        src: ref_for_load,
                    },
                });

                let updated = body.val_factory.next();
                body.val_types.insert(updated, whole_ty.clone());
                new_insts.push(Inst {
                    span: store_span,
                    kind: InstKind::FieldSet {
                        dst: updated,
                        object: old,
                        field,
                        rest: rest.clone(),
                        value,
                    },
                });

                let ref_for_store = body.val_factory.next();
                body.val_types
                    .insert(ref_for_store, Ty::Ref(Box::new(whole_ty)));
                new_insts.push(Inst {
                    span: store_span,
                    kind: InstKind::Ref {
                        dst: ref_for_store,
                        target: target.clone(),
                        path: vec![],
                    },
                });

                new_insts.push(Inst {
                    span: store_span,
                    kind: InstKind::Store {
                        dst: ref_for_store,
                        value: updated,
                    },
                });

                i += 2;
            }

            _ => {
                new_insts.push(body.insts[i].clone());
                i += 1;
            }
        }
    }

    body.insts = new_insts;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, ValueId};
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }
    fn span() -> Span {
        Span { start: 0, end: 0 }
    }
    fn inst(kind: InstKind) -> Inst {
        Inst { span: span(), kind }
    }

    fn make_body(insts: Vec<InstKind>, val_types: FxHashMap<ValueId, Ty>) -> MirBody {
        let max_val = val_types.keys().map(|v| v.to_raw()).max().unwrap_or(0);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=max_val {
            factory.next();
        }
        MirBody {
            insts: insts.into_iter().map(inst).collect(),
            val_types,
            params: vec![],
            captures: vec![],
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        }
    }

    /// SROA decomposes Ref(field) + Load into Ref(identity) + Load + FieldGet.
    #[test]
    fn sroa_field_load() {
        let i = Interner::new();
        let field_name = i.intern("x");
        let obj_ty = Ty::Object(FxHashMap::from_iter([(field_name, Ty::Int)]));
        let ctx = QualifiedRef::root(i.intern("a"));
        let context_types = FxHashMap::from_iter([(ctx, obj_ty.clone())]);

        let mut val_types = FxHashMap::default();
        // v0 = Ref<Int, false> (field ref)
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int)));
        // v1 = Int (load result)
        val_types.insert(v(1), Ty::Int);

        let mut body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Context(ctx),
                    path: vec![field_name],
                },
                InstKind::Load {
                    dst: v(1),
                    src: v(0),
                },
            ],
            val_types,
        );

        run_body(&mut body, &context_types);

        // Should decompose into: Ref(identity) + Load(whole) + FieldGet
        let kinds: Vec<&str> = body
            .insts
            .iter()
            .map(|i| match &i.kind {
                InstKind::Ref { path, .. } if path.is_empty() => "ref_identity",
                InstKind::Ref { path, .. } if !path.is_empty() => "ref_field",
                InstKind::Load { .. } => "load",
                InstKind::FieldGet { .. } => "field_get",
                _ => "other",
            })
            .collect();

        assert_eq!(
            kinds,
            vec!["ref_identity", "load", "field_get"],
            "SROA should decompose field Ref+Load into identity Ref + Load + FieldGet"
        );
    }

    /// SROA decomposes Ref(field) + Store into Ref(identity) + Load + FieldSet + Ref(identity) + Store.
    #[test]
    fn sroa_field_store() {
        let i = Interner::new();
        let field_name = i.intern("x");
        let obj_ty = Ty::Object(FxHashMap::from_iter([(field_name, Ty::Int)]));
        let var_name = v(100); // storage slot ValueId

        // Need an identity Ref in body so SROA can find whole type for Var("a").
        let mut val_types = FxHashMap::default();
        // v0 = identity Ref type
        val_types.insert(v(0), Ty::Ref(Box::new(obj_ty.clone())));
        // v1 = whole object
        val_types.insert(v(1), obj_ty.clone());
        // v2 = field Ref type
        val_types.insert(v(2), Ty::Ref(Box::new(Ty::Int)));
        // v3 = Int (value to store)
        val_types.insert(v(3), Ty::Int);

        let mut body = make_body(
            vec![
                // Identity Ref so SROA can find whole type
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Var(var_name),
                    path: vec![],
                },
                InstKind::Store {
                    dst: v(0),
                    value: v(1),
                },
                // Field store to decompose
                InstKind::Ref {
                    dst: v(2),
                    target: RefTarget::Var(var_name),
                    path: vec![field_name],
                },
                InstKind::Store {
                    dst: v(2),
                    value: v(3),
                },
            ],
            val_types,
        );

        run_body(&mut body, &FxHashMap::default());

        // First two instructions (identity Ref+Store) should pass through unchanged.
        // Field Store should decompose into: Ref(identity) + Load + FieldSet + Ref(identity) + Store.
        let kinds: Vec<&str> = body
            .insts
            .iter()
            .map(|i| match &i.kind {
                InstKind::Ref { path, .. } if path.is_empty() => "ref_identity",
                InstKind::Ref { path, .. } if !path.is_empty() => "ref_field",
                InstKind::Load { .. } => "load",
                InstKind::Store { .. } => "store",
                InstKind::FieldSet { .. } => "field_set",
                InstKind::FieldGet { .. } => "field_get",
                _ => "other",
            })
            .collect();

        assert_eq!(
            kinds,
            vec![
                "ref_identity",
                "store", // original identity store
                "ref_identity",
                "load",
                "field_set",
                "ref_identity",
                "store", // decomposed field store
            ],
            "SROA should decompose field Store: {kinds:?}"
        );
    }

    /// Identity Refs (no field) pass through SROA unchanged.
    #[test]
    fn sroa_identity_passthrough() {
        let var_name = v(100); // storage slot ValueId

        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Ref(Box::new(Ty::Int)));
        val_types.insert(v(1), Ty::Int);

        let mut body = make_body(
            vec![
                InstKind::Ref {
                    dst: v(0),
                    target: RefTarget::Var(var_name),
                    path: vec![],
                },
                InstKind::Load {
                    dst: v(1),
                    src: v(0),
                },
            ],
            val_types,
        );

        let inst_count_before = body.insts.len();
        run_body(&mut body, &FxHashMap::default());

        assert_eq!(
            body.insts.len(),
            inst_count_before,
            "identity Ref+Load should not be modified by SROA"
        );
    }

    /// No field Refs -> SROA is a no-op.
    #[test]
    fn sroa_noop_when_no_field_refs() {
        let mut val_types = FxHashMap::default();
        val_types.insert(v(0), Ty::Int);

        let mut body = make_body(
            vec![
                InstKind::Const {
                    dst: v(0),
                    value: acvus_ast::Literal::Int(42),
                },
                InstKind::Return(v(0)),
            ],
            val_types,
        );

        let inst_count_before = body.insts.len();
        run_body(&mut body, &FxHashMap::default());

        assert_eq!(
            body.insts.len(),
            inst_count_before,
            "no field Refs -> SROA should be no-op"
        );
    }
}
