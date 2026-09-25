//! A reference to the whole of what a reference names is that reference:
//! `Ref { target: Through(r), path: [] }` is `r`.
//!
//! It exists because `match` over a place lowers to one per iteration —
//! `r80 = &prog[pc]; r81 = ref &(*r80)` in the `bf table` bench — and the
//! machine ran the second as an operation of its own.

use rustc_hash::FxHashMap;

use crate::cfg::{CfgBody, Terminator};
use crate::ir::{InstKind, RefTarget, ValueId};
use crate::optimize::ssa_pass::map_uses;
use crate::ty::Ty;
use crate::validate::move_check::is_move_only;

/// A `Ref` that defines no value of its own: `of` and `source` are one
/// reference, to one storage, at one type and mutability, and that type is
/// a word the runtime copies rather than a box the value owns.
struct Reborrow {
    of: ValueId,
    source: ValueId,
}

impl Reborrow {
    fn of_inst(cfg: &CfgBody, kind: &InstKind) -> Option<Self> {
        let InstKind::Ref {
            dst,
            target: RefTarget::Through(source),
            path,
            ..
        } = kind
        else {
            return None;
        };
        let through = cfg.val_types.get(source);
        (path.is_empty()
            && matches!(through, Some(Ty::Ref(..)))
            && through.and_then(is_move_only) == Some(false)
            && through == cfg.val_types.get(dst))
        .then_some(Reborrow {
            of: *dst,
            source: *source,
        })
    }
}

pub fn run(cfg: &mut CfgBody) {
    let sources: FxHashMap<ValueId, ValueId> = cfg
        .blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| Reborrow::of_inst(cfg, &inst.kind))
        .map(|r| (r.of, r.source))
        .collect();
    if sources.is_empty() {
        return;
    }
    let source = |mut v: ValueId| {
        while let Some(&r) = sources.get(&v) {
            v = r;
        }
        v
    };
    for block in &mut cfg.blocks {
        block.insts.retain(
            |inst| !matches!(&inst.kind, InstKind::Ref { dst, .. } if sources.contains_key(dst)),
        );
        for inst in &mut block.insts {
            map_uses(&mut inst.kind, &mut |v| *v = source(*v));
        }
        for used in terminator_uses_mut(&mut block.terminator) {
            *used = source(*used);
        }
    }
    cfg.val_types.retain(|v, _| !sources.contains_key(v));
}

fn terminator_uses_mut(t: &mut Terminator) -> Vec<&mut ValueId> {
    match t {
        Terminator::Return { value, order, .. } => {
            std::iter::once(value).chain(order.as_mut()).collect()
        }
        Terminator::Jump { args, .. } => args.iter_mut().collect(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => std::iter::once(cond)
            .chain(then_args.iter_mut())
            .chain(else_args.iter_mut())
            .collect(),
        Terminator::For {
            source, exit_args, ..
        } => source
            .uses_mut()
            .into_iter()
            .chain(exit_args.iter_mut())
            .collect(),
        Terminator::While {
            cond, exit_args, ..
        } => std::iter::once(cond).chain(exit_args.iter_mut()).collect(),
        Terminator::Switch { tag, arms, default } => std::iter::once(tag)
            .chain(arms.iter_mut().flat_map(|(_, _, args)| args.iter_mut()))
            .chain(default.iter_mut().flat_map(|(_, args)| args.iter_mut()))
            .collect(),
        Terminator::Fallthrough | Terminator::Diverge => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::{DebugInfo, Inst, MirBody, PathSeg};
    use crate::ty::{Mutability, TypeArg};
    use acvus_ast::Span;
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn body(insts: Vec<InstKind>, types: Vec<(ValueId, Ty)>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..16 {
            factory.next();
        }
        promote(MirBody {
            demoted_diamonds: Default::default(),
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: types.into_iter().collect(),
            params: vec![],
            captures: vec![],
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            label_count: 0,
            val_factory: factory,
        })
    }

    fn reference(m: Mutability, inner: Ty) -> Ty {
        Ty::Ref(m, Box::new(TypeArg::uniform(inner)))
    }

    fn refs(cfg: &CfgBody) -> usize {
        cfg.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.kind, InstKind::Ref { .. }))
            .count()
    }

    fn returned(cfg: &CfgBody) -> ValueId {
        match cfg.blocks[0].terminator {
            Terminator::Return { value, .. } => value,
            ref other => panic!("the body returns: {other:?}"),
        }
    }

    fn take(dst: ValueId, target: RefTarget) -> InstKind {
        InstKind::Ref {
            dst,
            target,
            path: vec![],
            mutability: Mutability::Shared,
        }
    }

    #[test]
    fn a_whole_reborrow_is_the_reference_it_reborrows() {
        let mut cfg = body(
            vec![
                take(v(1), RefTarget::Var(v(0))),
                take(v(2), RefTarget::Through(v(1))),
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![
                (v(0), Ty::I64),
                (v(1), reference(Mutability::Shared, Ty::I64)),
                (v(2), reference(Mutability::Shared, Ty::I64)),
            ],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 1);
        assert_eq!(returned(&cfg), v(1));
    }

    #[test]
    fn a_chain_of_reborrows_collapses_to_its_source() {
        let mut cfg = body(
            vec![
                take(v(1), RefTarget::Var(v(0))),
                take(v(2), RefTarget::Through(v(1))),
                take(v(3), RefTarget::Through(v(2))),
                InstKind::Return {
                    value: v(3),
                    order: None,
                },
            ],
            vec![
                (v(0), Ty::I64),
                (v(1), reference(Mutability::Shared, Ty::I64)),
                (v(2), reference(Mutability::Shared, Ty::I64)),
                (v(3), reference(Mutability::Shared, Ty::I64)),
            ],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 1);
        assert_eq!(returned(&cfg), v(1));
    }

    #[test]
    fn a_reborrow_at_a_path_stands() {
        let mut cfg = body(
            vec![
                InstKind::Ref {
                    dst: v(2),
                    target: RefTarget::Through(v(1)),
                    path: vec![PathSeg::Index(0)],
                    mutability: Mutability::Shared,
                },
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![
                (v(1), reference(Mutability::Shared, Ty::I64)),
                (v(2), reference(Mutability::Shared, Ty::I64)),
            ],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 1);
        assert_eq!(returned(&cfg), v(2));
    }

    #[test]
    fn a_shared_reborrow_of_a_mutable_reference_stands() {
        let mut cfg = body(
            vec![
                take(v(2), RefTarget::Through(v(1))),
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![
                (v(1), reference(Mutability::Mut, Ty::I64)),
                (v(2), reference(Mutability::Shared, Ty::I64)),
            ],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 1);
        assert_eq!(returned(&cfg), v(2));
    }

    #[test]
    fn a_reborrow_of_a_slice_reference_folds() {
        let slice = || reference(Mutability::Shared, Ty::Slice(Box::new(Ty::I64)));
        let mut cfg = body(
            vec![
                take(v(2), RefTarget::Through(v(1))),
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![(v(1), slice()), (v(2), slice())],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 0);
        assert_eq!(returned(&cfg), v(1));
    }

    #[test]
    fn a_reference_to_a_value_that_is_not_a_reference_stands() {
        let mut cfg = body(
            vec![
                take(v(2), RefTarget::Through(v(1))),
                InstKind::Return {
                    value: v(2),
                    order: None,
                },
            ],
            vec![
                (v(1), Ty::I64),
                (v(2), reference(Mutability::Shared, Ty::I64)),
            ],
        );
        run(&mut cfg);
        assert_eq!(refs(&cfg), 1);
        assert_eq!(returned(&cfg), v(2));
    }
}
