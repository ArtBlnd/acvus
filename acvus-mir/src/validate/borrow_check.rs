//! The two storage rules of RFC-0018, checked over a body's CFG.
//!
//! Exclusion: while a `&mut` to a storage is live, no other name of that
//! storage — the storage itself, a `&`, another `&mut` — is read or
//! written; while a `&` is live, the storage is not assigned, moved out of,
//! or mutably referenced. A reference is live from its `Ref` to its last
//! use, by the body's liveness.
//!
//! Context return: a context a run takes is assigned again on every path
//! before the run ends.

use std::collections::VecDeque;

use acvus_ast::Span;
use acvus_utils::LocalIdOps;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::graph::QualifiedRef;
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
    check_contexts_returned(scope, &cfg, errors);
}

// -- Exclusion ---------------------------------------------------------

/// A reference value: the storage it names and whether it may write.
#[derive(Clone, Copy)]
struct Borrow {
    target: RefTarget,
    mutability: Mutability,
}

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
    Take { moves: bool },
    Assign,
}

fn touch(kind: &InstKind, val_types: &FxHashMap<ValueId, Ty>) -> Option<(RefTarget, Touch)> {
    match kind {
        InstKind::Ref {
            target, mutability, ..
        } => Some((target.clone(), Touch::Reference(*mutability))),
        InstKind::Take { dst, target, .. } => {
            Some((
                target.clone(),
                Touch::Take {
                    moves: take_moves(val_types, dst),
                },
            ))
        }
        InstKind::Assign { target, .. } => Some((target.clone(), Touch::Assign)),
        _ => None,
    }
}

/// Whether `touch` on a storage conflicts with a live borrow of it.
fn conflicts(live: &Borrow, touch: &Touch) -> bool {
    match (live.mutability, touch) {
        (Mutability::Mut, _) => true,
        (Mutability::Shared, Touch::Reference(Mutability::Shared)) => false,
        (Mutability::Shared, Touch::Reference(Mutability::Mut)) => true,
        (Mutability::Shared, Touch::Take { moves }) => *moves,
        (Mutability::Shared, Touch::Assign) => true,
    }
}

fn check_exclusion(scope: &str, cfg: &CfgBody, errors: &mut Vec<ValidationError>) {
    let mut borrows: FxHashMap<ValueId, Borrow> = FxHashMap::default();
    for block in &cfg.blocks {
        for inst in &block.insts {
            if let InstKind::Ref {
                dst,
                target,
                mutability,
                ..
            } = &inst.kind
            {
                borrows.insert(
                    *dst,
                    Borrow {
                        target: target.clone(),
                        mutability: *mutability,
                    },
                );
            }
        }
    }
    if borrows.is_empty() {
        return;
    }
    let live = liveness::analyze(cfg);

    for (bi, block) in cfg.blocks.iter().enumerate() {
        // Borrows live before each instruction, by a backward walk from
        // the block's live-out set.
        let mut live_before: Vec<FxHashSet<ValueId>> = vec![FxHashSet::default(); block.insts.len()];
        let mut current: FxHashSet<ValueId> = borrows
            .keys()
            .filter(|v| live.is_live_out(BlockIdx(bi), **v))
            .copied()
            .collect();
        for v in terminator_uses(&block.terminator) {
            if borrows.contains_key(&v) {
                current.insert(v);
            }
        }
        for (ii, inst) in block.insts.iter().enumerate().rev() {
            for d in inst_info::defs(&inst.kind) {
                current.remove(&d);
            }
            for u in inst_info::uses(&inst.kind) {
                if borrows.contains_key(&u) {
                    current.insert(u);
                }
            }
            live_before[ii] = current.clone();
        }

        for (ii, inst) in block.insts.iter().enumerate() {
            let Some((target, touch)) = touch(&inst.kind, &cfg.val_types) else {
                continue;
            };
            for reference in &live_before[ii] {
                let borrow = borrows[reference];
                if borrow.target == target && conflicts(&borrow, &touch) {
                    errors.push(ValidationError {
                        scope: scope.to_string(),
                        inst_index: ii,
                        span: inst.span,
                        kind: ValidationErrorKind::BorrowConflict {
                            storage: format!("{target:?}"),
                            reference: reference.to_raw() as u32,
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

// -- Contexts returned -------------------------------------------------

/// Forward dataflow: the contexts moved out of and not yet assigned back.
/// A primitive `Take` copies and takes nothing. Taken on any path is taken;
/// each context is reported once per body.
fn check_contexts_returned(scope: &str, cfg: &CfgBody, errors: &mut Vec<ValidationError>) {
    let n = cfg.blocks.len();
    let mut entry: Vec<FxHashSet<QualifiedRef>> = vec![FxHashSet::default(); n];
    let mut worklist: VecDeque<BlockIdx> = VecDeque::from([BlockIdx(0)]);
    let mut visited = vec![false; n];
    let mut reported: FxHashSet<QualifiedRef> = FxHashSet::default();

    while let Some(idx) = worklist.pop_front() {
        visited[idx.0] = true;
        let mut taken = entry[idx.0].clone();
        let block = &cfg.blocks[idx.0];
        for inst in &block.insts {
            match &inst.kind {
                InstKind::Take {
                    dst,
                    target: RefTarget::Context(c),
                    ..
                } => {
                    if take_moves(&cfg.val_types, dst) {
                        taken.insert(*c);
                    }
                }
                InstKind::Assign {
                    target: RefTarget::Context(c),
                    ..
                } => {
                    taken.remove(c);
                }
                _ => {}
            }
        }
        if let Terminator::Return { .. } = block.terminator {
            let mut left: Vec<QualifiedRef> = taken.iter().copied().collect();
            left.sort();
            for c in left {
                if !reported.insert(c) {
                    continue;
                }
                errors.push(ValidationError {
                    scope: scope.to_string(),
                    inst_index: block.insts.len(),
                    span: block
                        .insts
                        .last()
                        .map(|i| i.span)
                        .unwrap_or(Span::ZERO),
                    kind: ValidationErrorKind::ContextLeftTaken {
                        context: format!("{c:?}"),
                    },
                });
            }
        }
        for succ in cfg.successors(idx) {
            let before = entry[succ.0].len();
            entry[succ.0].extend(taken.iter().copied());
            if entry[succ.0].len() != before || !visited[succ.0] {
                worklist.push_back(succ);
            }
        }
    }
}

// -- Tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, Label};
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

    fn left_taken_count(kinds: &[ValidationErrorKind]) -> usize {
        kinds
            .iter()
            .filter(|k| matches!(k, ValidationErrorKind::ContextLeftTaken { .. }))
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
        InstKind::Load {
            dst: v(dst),
            src: v(src),
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
                InstKind::Store {
                    dst: v(2),
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
        assert_eq!(conflict_count(&errors(m)), 0, "a word copy does not move the storage");
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
        assert_eq!(conflict_count(&errors(m)), 1, "even a word copy reads through the &mut");
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

    // -- contexts returned ---------------------------------------------

    fn ctx(i: &Interner) -> QualifiedRef {
        QualifiedRef::root(i.intern("history"))
    }

    fn take_ctx(i: &Interner, dst: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Context(ctx(i)),
            path: vec![],
        }
    }

    fn assign_ctx(i: &Interner, value: usize) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Context(ctx(i)),
            path: vec![],
            value: v(value),
        }
    }

    #[test]
    fn a_context_taken_and_assigned_back_is_accepted() {
        let i = Interner::new();
        let m = body(
            vec![take_ctx(&i, 3), assign_ctx(&i, 3), ret(4)],
            vec![(v(3), Ty::String), (v(4), Ty::Int)],
        );
        assert_eq!(left_taken_count(&errors(m)), 0);
    }

    #[test]
    fn a_primitive_context_read_takes_nothing() {
        let i = Interner::new();
        let m = body(vec![take_ctx(&i, 4), ret(4)], vec![(v(4), Ty::Int)]);
        assert_eq!(left_taken_count(&errors(m)), 0);
    }

    #[test]
    fn a_context_taken_and_not_assigned_is_rejected() {
        let i = Interner::new();
        let m = body(
            vec![take_ctx(&i, 3), ret(3)],
            vec![(v(3), Ty::String)],
        );
        assert_eq!(left_taken_count(&errors(m)), 1);
    }

    #[test]
    fn a_context_assigned_on_one_path_only_is_rejected() {
        let i = Interner::new();
        // take; jump_if c then L1 else L2; L1: assign; ret; L2: ret
        let m = body(
            vec![
                take_ctx(&i, 3),
                InstKind::JumpIf {
                    cond: v(5),
                    then_label: Label(0),
                    then_args: vec![],
                    else_label: Label(1),
                    else_args: vec![],
                },
                InstKind::BlockLabel {
                    label: Label(0),
                    params: vec![],
                    merge_of: None,
                },
                assign_ctx(&i, 3),
                ret(4),
                InstKind::BlockLabel {
                    label: Label(1),
                    params: vec![],
                    merge_of: None,
                },
                ret(4),
            ],
            vec![(v(3), Ty::String), (v(4), Ty::Int), (v(5), Ty::Bool)],
        );
        assert_eq!(left_taken_count(&errors(m)), 1);
    }
}
