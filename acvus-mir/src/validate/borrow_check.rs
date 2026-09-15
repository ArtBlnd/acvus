//! The exclusion rule of RFC-0018, checked over a body's CFG.
//!
//! Exclusion: while a `&mut` to a storage is live, no other name of that
//! storage — the storage itself, a `&`, another `&mut` — is read or
//! written; while a `&` is live, the storage is not assigned, moved out of,
//! or mutably referenced. A holder of a loan — a value, or a storage a
//! reference was assigned into — is live from its definition to its last
//! use, counted as liveness counts them (RFC-0029).

use acvus_utils::LocalIdOps;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::loans::{Loan, Loans};
use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::ir::{InstKind, MirBody, MirModule, RefTarget, ValueId};
use crate::ty::{Mutability, Ty};
use crate::validate::move_check::is_move_only;
use crate::validate::type_check::{ValidationError, ValidationErrorKind};

pub fn check_borrows(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    check_body("main", &module.main, &mut errors);
    for (label, closure) in &module.closures {
        check_body(&format!("closure({label:?})"), closure, &mut errors);
    }
    errors
}

fn check_body(scope: &str, body: &MirBody, errors: &mut Vec<ValidationError>) {
    let cfg = promote(body.clone());
    if cfg.blocks.is_empty() {
        return;
    }
    check_exclusion(scope, &cfg, errors);
}

// -- Exclusion ---------------------------------------------------------

/// A `Take` whose result has no type entry is checked as a move: the
/// stricter reading, so a missing type never hides a conflict. A result
/// typed `<error>` is already reported by the type checker and moves
/// nothing here.
const UNTYPED_TAKE_MOVES: bool = true;

fn take_moves(val_types: &FxHashMap<ValueId, Ty>, dst: &ValueId) -> bool {
    match val_types.get(dst) {
        Some(ty) if ty.is_error() => false,
        Some(ty) => is_move_only(ty).unwrap_or(UNTYPED_TAKE_MOVES),
        None => UNTYPED_TAKE_MOVES,
    }
}

/// What an instruction does to a storage.
enum Touch {
    Reference(Mutability),
    /// A move out; `false` when the storage keeps a copy (a primitive).
    Take {
        moves: bool,
    },
    Assign,
}

fn touch(kind: &InstKind, val_types: &FxHashMap<ValueId, Ty>) -> Option<(RefTarget, Touch)> {
    match kind {
        InstKind::Ref {
            target, mutability, ..
        } => Some((target.clone(), Touch::Reference(*mutability))),
        InstKind::Take { dst, target, .. } => Some((
            target.clone(),
            Touch::Take {
                moves: take_moves(val_types, dst),
            },
        )),
        InstKind::Assign { target, .. } => Some((target.clone(), Touch::Assign)),
        _ => None,
    }
}

/// Whether `touch` on a storage conflicts with a live loan of it.
fn conflicts(live: &Loan, touch: &Touch) -> bool {
    match (live.mutability, touch) {
        (Mutability::Mut, _) => true,
        (Mutability::Shared, Touch::Reference(Mutability::Shared)) => false,
        (Mutability::Shared, Touch::Reference(Mutability::Mut)) => true,
        (Mutability::Shared, Touch::Take { moves }) => *moves,
        (Mutability::Shared, Touch::Assign) => true,
    }
}

struct Reached {
    storage: Vec<ValueId>,
    via: Vec<ValueId>,
}

fn reached(target: &RefTarget, loans: &Loans) -> Reached {
    match target {
        RefTarget::Var(s) | RefTarget::Param(s) => Reached {
            storage: vec![*s],
            via: vec![],
        },
        RefTarget::Through(r) => {
            let region = loans.region(*r);
            Reached {
                storage: region.loans.iter().map(|l| l.storage).collect(),
                via: region.via.iter().copied().chain([*r]).collect(),
            }
        }
    }
}

fn check_exclusion(scope: &str, cfg: &CfgBody, errors: &mut Vec<ValidationError>) {
    let loans = Loans::build(cfg);
    let holders: FxHashSet<ValueId> = cfg
        .val_types
        .keys()
        .filter(|v| !loans.region(**v).loans.is_empty())
        .copied()
        .collect();
    if holders.is_empty() {
        return;
    }
    let live = liveness::analyze(cfg);

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // Holders live before each instruction, by a backward walk from
        // the block's live-out set.
        let mut live_before: Vec<FxHashSet<ValueId>> =
            vec![FxHashSet::default(); block.insts.len()];
        let mut current: FxHashSet<ValueId> = holders
            .iter()
            .filter(|v| live.is_live_out(BlockIdx(bi), **v))
            .copied()
            .collect();
        for v in terminator_uses(&block.terminator) {
            if holders.contains(&v) {
                current.insert(v);
            }
        }
        for (ii, inst) in block.insts.iter().enumerate().rev() {
            for d in inst_info::defs(&inst.kind) {
                current.remove(&d);
            }
            for u in loans.uses_with_storage(&inst.kind) {
                if holders.contains(&u) {
                    current.insert(u);
                }
            }
            live_before[ii] = current.clone();
        }

        for (ii, inst) in block.insts.iter().enumerate() {
            let Some((target, touch)) = touch(&inst.kind, &cfg.val_types) else {
                continue;
            };
            let reach = reached(&target, &loans);
            for holder in &live_before[ii] {
                if reach.via.contains(holder) {
                    continue;
                }
                let hit =
                    loans.region(*holder).loans.iter().any(|loan| {
                        reach.storage.contains(&loan.storage) && conflicts(loan, &touch)
                    });
                if hit {
                    errors.push(ValidationError {
                        scope: scope.to_string(),
                        inst_index: ii,
                        span: inst.span,
                        kind: ValidationErrorKind::BorrowConflict {
                            storage: format!("{target:?}"),
                            reference: holder.to_raw() as u32,
                        },
                    });
                }
            }
        }
    }
}

fn terminator_uses(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend(then_args);
            v.extend(else_args);
            v
        }
        Terminator::Return { value, order } => {
            let mut v = vec![*value];
            v.extend(*order);
            v
        }
        Terminator::Fallthrough => Vec::new(),
    }
}

// -- Tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, Label};
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn body(insts: Vec<InstKind>, types: Vec<(ValueId, Ty)>) -> MirBody {
        let max = types.iter().map(|(v, _)| v.to_raw()).max().unwrap_or(0);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=max.max(16) {
            factory.next();
        }
        MirBody {
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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 2,
            order_param: None,
        }
    }

    fn module(main: MirBody) -> MirModule {
        MirModule {
            main,
            closures: FxHashMap::default(),
        }
    }

    fn errors(main: MirBody) -> Vec<ValidationErrorKind> {
        check_borrows(&module(main))
            .into_iter()
            .map(|e| e.kind)
            .collect()
    }

    fn conflict_count(kinds: &[ValidationErrorKind]) -> usize {
        kinds
            .iter()
            .filter(|k| matches!(k, ValidationErrorKind::BorrowConflict { .. }))
            .count()
    }

    fn slot() -> ValueId {
        v(0)
    }

    fn reference(dst: usize, m: Mutability) -> InstKind {
        InstKind::Ref {
            dst: v(dst),
            target: RefTarget::Var(slot()),
            path: vec![],
            mutability: m,
        }
    }

    fn take(dst: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Var(slot()),
            path: vec![],
        }
    }

    fn assign(value: usize) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Var(slot()),
            path: vec![],
            value: v(value),
        }
    }

    fn load(dst: usize, src: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Through(v(src)),
            path: vec![],
        }
    }

    fn ret(value: usize) -> InstKind {
        InstKind::Return {
            value: v(value),
            order: None,
        }
    }

    fn string_slot() -> Vec<(ValueId, Ty)> {
        vec![
            (slot(), Ty::String),
            (v(1), Ty::Ref(Mutability::Shared, Box::new(Ty::String))),
            (v(2), Ty::Ref(Mutability::Mut, Box::new(Ty::String))),
            (v(3), Ty::String),
            (v(4), Ty::Int),
        ]
    }

    // -- exclusion: accepted -------------------------------------------

    #[test]
    fn two_shared_references_may_be_live_together() {
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                reference(5, Mutability::Shared),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(1), v(5)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 0);
    }

    #[test]
    fn a_mutable_reference_whose_last_use_passed_no_longer_excludes() {
        let m = body(
            vec![
                reference(2, Mutability::Mut),
                InstKind::Assign {
                    target: RefTarget::Through(v(2)),
                    path: vec![],
                    value: v(3),
                },
                take(7),
                ret(7),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 0, "the &mut died at the store");
    }

    #[test]
    fn a_primitive_is_read_while_shared_referenced() {
        let mut types = string_slot();
        types[0] = (slot(), Ty::Int);
        types.push((v(7), Ty::Int));
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                take(7),
                load(4, 1),
                ret(7),
            ],
            types,
        );
        assert_eq!(
            conflict_count(&errors(m)),
            0,
            "a word copy does not move the storage"
        );
    }

    // -- exclusion: rejected -------------------------------------------

    #[test]
    fn a_take_while_a_shared_reference_is_live_is_rejected() {
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                take(7),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(1), v(7)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    #[test]
    fn an_assign_while_a_shared_reference_is_live_is_rejected() {
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                assign(3),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(1)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    #[test]
    fn a_second_reference_while_a_mutable_one_is_live_is_rejected() {
        let m = body(
            vec![
                reference(2, Mutability::Mut),
                reference(1, Mutability::Shared),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(1), v(2)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    #[test]
    fn a_mutable_reference_while_a_shared_one_is_live_is_rejected() {
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                reference(2, Mutability::Mut),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(1), v(2)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    #[test]
    fn a_read_of_the_storage_while_a_mutable_reference_is_live_is_rejected() {
        let mut types = string_slot();
        types[0] = (slot(), Ty::Int);
        types.push((v(7), Ty::Int));
        let m = body(
            vec![
                reference(2, Mutability::Mut),
                take(7),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(2), v(7)],
                },
                ret(6),
            ],
            types,
        );
        assert_eq!(
            conflict_count(&errors(m)),
            1,
            "even a word copy reads through the &mut"
        );
    }

    #[test]
    fn a_reference_live_across_a_branch_still_excludes() {
        // ref &mut; jump L1; L1: take; use ref
        let m = body(
            vec![
                reference(2, Mutability::Mut),
                InstKind::Jump {
                    label: Label(1),
                    args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                take(7),
                InstKind::MakeTuple {
                    dst: v(6),
                    elements: vec![v(2), v(7)],
                },
                ret(6),
            ],
            string_slot(),
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    // -- holders in storage and parameters (RFC-0029) ------------------

    fn store(slot: usize, value: usize) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Var(v(slot)),
            path: vec![],
            value: v(value),
        }
    }

    fn take_slot(dst: usize, slot: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Var(v(slot)),
            path: vec![],
        }
    }

    #[test]
    fn a_reference_assigned_into_a_storage_keeps_its_loan_live() {
        let mut types = string_slot();
        types.push((v(7), Ty::Ref(Mutability::Shared, Box::new(Ty::String))));
        types.push((v(8), Ty::Ref(Mutability::Shared, Box::new(Ty::String))));
        let m = body(
            vec![
                reference(1, Mutability::Shared),
                store(7, 1),
                assign(3),
                take_slot(8, 7),
                load(4, 8),
                ret(4),
            ],
            types,
        );
        assert_eq!(conflict_count(&errors(m)), 1);
    }

    #[test]
    fn a_reborrow_of_a_parameter_conflicts_with_a_write_through_it() {
        let param = v(9);
        let mut main = body(
            vec![
                InstKind::Ref {
                    dst: v(1),
                    target: RefTarget::Through(param),
                    path: vec![],
                    mutability: Mutability::Shared,
                },
                InstKind::Assign {
                    target: RefTarget::Through(param),
                    path: vec![],
                    value: v(3),
                },
                load(4, 1),
                ret(4),
            ],
            string_slot(),
        );
        main.params
            .push((acvus_utils::Interner::new().intern("p"), param));
        main.val_types
            .insert(param, Ty::Ref(Mutability::Mut, Box::new(Ty::String)));
        assert_eq!(conflict_count(&errors(main)), 1);
    }
}
