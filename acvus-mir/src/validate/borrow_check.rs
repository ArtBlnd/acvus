//! The exclusion rule of RFC-0018, checked over a body's CFG.
//!
//! Exclusion: while a `&mut` to a storage is live, no other name of that
//! storage — the storage itself, a `&`, another `&mut` — is read or
//! written; while a `&` is live, the storage is not assigned, moved out of,
//! or mutably referenced. A holder of a loan — a value, or a storage a
//! reference was assigned into — is live from its definition to its last
//! use, counted as liveness counts them (RFC-0029).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::loans::{Loan, Loans, Summaries, Summary};
use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::ir::{Callee, InstKind, Label as ClosureLabel, MirBody, MirModule, RefTarget, ValueId};
use crate::ty::{Mutability, Ty};
use crate::validate::move_check::is_move_only;
use crate::validate::type_check::{ConflictTouch, ValidationError, ValidationErrorKind};
use acvus_ast::Span;
use acvus_ast::report::Label;

pub struct Checked {
    pub errors: Vec<ValidationError>,
    pub summary: Summary,
}

/// The exclusion rule alone, which holds of a module at any point in the
/// pipeline.
pub fn check_borrows(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    for checking in Bodies::of(module, Summaries::NONE).all() {
        checking.check_exclusion(&mut errors);
    }
    errors
}

/// The exclusion rule and RFC-0064's result rule together, with the summary
/// the module's own body leaves with.
///
/// The result rule is not repeated by `check_borrows` because it cannot be:
/// what a call's result borrows is the callee's summary, so the answer
/// depends on every callee having been checked first, and pass 0 of
/// `graph::optimize` is the only place that order exists. Asking the same
/// question anywhere else would answer it from `Summaries::NONE` and refuse
/// programs pass 0 admitted.
pub fn check_borrows_in_order(module: &MirModule, summaries: Summaries<'_>) -> Checked {
    let mut errors = Vec::new();
    let mut closures: FxHashMap<ClosureLabel, Summary> = FxHashMap::default();
    for label in inner_closures_first(module) {
        let body = &module.closures[&label];
        let Some(checking) = Checking::of(
            format!("closure({label:?})"),
            body,
            summaries.with_closures(&closures),
        ) else {
            continue;
        };
        let summary = checking.check_result(&mut errors);
        checking.check_exclusion(&mut errors);
        closures.insert(label, summary);
    }
    let summary = match Checking::of(
        "main".to_string(),
        &module.main,
        summaries.with_closures(&closures),
    ) {
        Some(main) => {
            let summary = main.check_result(&mut errors);
            main.check_exclusion(&mut errors);
            summary
        }
        None => Summary::default(),
    };
    Checked { errors, summary }
}

/// Every closure of the module, each before the body that makes it.
///
/// A lambda's call summary is read at the call, and a lambda is called in
/// the body that lexically contains it — main, or another lambda.
fn inner_closures_first(module: &MirModule) -> Vec<ClosureLabel> {
    let mut order = Vec::new();
    let mut seen: FxHashSet<ClosureLabel> = FxHashSet::default();
    visit_makers_first(module, &made_by(&module.main), &mut seen, &mut order);
    let mut unreached: Vec<ClosureLabel> = module
        .closures
        .keys()
        .copied()
        .filter(|label| !seen.contains(label))
        .collect();
    unreached.sort_unstable_by_key(|label| label.0);
    order.extend(unreached);
    order
}

fn visit_makers_first(
    module: &MirModule,
    labels: &[ClosureLabel],
    seen: &mut FxHashSet<ClosureLabel>,
    order: &mut Vec<ClosureLabel>,
) {
    for label in labels {
        if !seen.insert(*label) {
            continue;
        }
        let Some(body) = module.closures.get(label) else {
            continue;
        };
        visit_makers_first(module, &made_by(body), seen, order);
        order.push(*label);
    }
}

fn made_by(body: &MirBody) -> Vec<ClosureLabel> {
    body.insts
        .iter()
        .filter_map(|inst| match &inst.kind {
            InstKind::MakeClosure { body, .. } => Some(*body),
            _ => None,
        })
        .collect()
}

struct Bodies {
    main: Option<Checking>,
    closures: Vec<Checking>,
}

impl Bodies {
    fn of(module: &MirModule, summaries: Summaries<'_>) -> Self {
        Self {
            main: Checking::of("main".to_string(), &module.main, summaries),
            closures: module
                .closures
                .iter()
                .filter_map(|(label, closure)| {
                    Checking::of(format!("closure({label:?})"), closure, summaries)
                })
                .collect(),
        }
    }

    fn all(&self) -> impl Iterator<Item = &Checking> {
        self.main.iter().chain(&self.closures)
    }
}

/// One body under the borrow check, with the analyses every rule reads.
struct Checking {
    scope: String,
    cfg: CfgBody,
    loans: Loans,
    sites: HolderSites,
}

impl Checking {
    fn of(scope: String, body: &MirBody, summaries: Summaries<'_>) -> Option<Self> {
        let cfg = promote(body.clone());
        if cfg.blocks.is_empty() {
            return None;
        }
        let loans = Loans::build(&cfg, summaries);
        let sites = HolderSites::of(&cfg, &loans);
        Some(Self {
            scope,
            cfg,
            loans,
            sites,
        })
    }

    /// RFC-0064 Decision 2: what the body's result borrows from its
    /// parameters is the body's summary, and a local's loan in the result is
    /// refused.
    fn check_result(&self, errors: &mut Vec<ValidationError>) -> Summary {
        let mut summary = Summary::default();
        for block in &self.cfg.blocks {
            let Terminator::Return { value, .. } = block.terminator else {
                continue;
            };
            let leaving = self.loans.leaving(value);
            for loan in leaving.summary.loans {
                if !summary.loans.contains(&loan) {
                    summary.loans.push(loan);
                }
            }
            for local in leaving.locals {
                errors.push(ValidationError {
                    scope: self.scope.clone(),
                    inst_index: block.insts.len(),
                    span: self.sites.borrowed_at(value),
                    kind: ValidationErrorKind::ReferenceToLocalLeavesBody {
                        storage: self.cfg.debug.get(local).cloned(),
                    },
                });
            }
        }
        summary
    }
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

impl Touch {
    /// A `&mut` may write through the reference it takes, so it is stated as
    /// a write; a `&` and a copying `Take` only read.
    fn stated(&self) -> ConflictTouch {
        match self {
            Touch::Reference(Mutability::Shared) => ConflictTouch::Read,
            Touch::Reference(Mutability::Mut) | Touch::Assign => ConflictTouch::Written,
            Touch::Take { moves: true } => ConflictTouch::Moved,
            Touch::Take { moves: false } => ConflictTouch::Read,
        }
    }
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
                storage: region.loans.iter().map(|l| l.storage.value()).collect(),
                via: region.via.iter().copied().chain([*r]).collect(),
            }
        }
    }
}

struct HolderUse {
    holder: ValueId,
    span: Span,
    kind: UseKind,
}

/// What a holder's use does with it, where the words for it differ by the
/// holder's form.
///
/// A call is the least of them because `lower.rs` emits the `Ref` of the slot a
/// lambda was stored in with the span of the name alone, at the same offset as
/// the call that reads it. Both uses are then at one place, and the call is the
/// one that says what happened there.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UseKind {
    Call,
    Other,
}

/// The values an indirect call names as the lambda it calls: the callee
/// register, the storage a lambda was assigned into and read back out of, and
/// so on to the storage itself.
fn called_by(loans: &Loans, kind: &InstKind) -> Vec<ValueId> {
    let callee = match kind {
        InstKind::FunctionCall {
            callee: Callee::Indirect(f),
            ..
        }
        | InstKind::Spawn {
            callee: Callee::Indirect(f),
            ..
        } => *f,
        _ => return Vec::new(),
    };
    let mut names = vec![callee];
    let mut at = 0;
    while at < names.len() {
        for loan in &loans.region(names[at]).loans {
            let storage = loan.storage.value();
            if !names.contains(&storage) {
                names.push(storage);
            }
        }
        at += 1;
    }
    names
}

/// Where a holder took the loan it holds, and in which of the two forms
/// RFC-0064 gives a holder: a reference, or a lambda that captured one.
#[derive(Clone, Copy)]
struct Took {
    span: Span,
    form: HolderForm,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HolderForm {
    Reference,
    Lambda,
}

impl HolderForm {
    fn took_it_here(self) -> &'static str {
        match self {
            Self::Reference => "borrowed here",
            Self::Lambda => "captured here",
        }
    }

    fn keeps_it_live_here(self, use_kind: UseKind) -> &'static str {
        match (self, use_kind) {
            (Self::Reference, _) => "the reference is used here",
            (Self::Lambda, UseKind::Call) => "the lambda is called here",
            (Self::Lambda, UseKind::Other) => "the lambda is used here",
        }
    }
}

/// A `Loan` carries no span, and it cannot gain one here: `Region`'s join
/// dedupes loans by equality, so a span in a `Loan` would change what the
/// region analysis converges to. The places a conflict points at are read off
/// the instructions instead.
struct HolderSites {
    borrowed: FxHashMap<ValueId, Took>,
    named: Vec<HolderUse>,
}

impl HolderSites {
    fn of(cfg: &CfgBody, loans: &Loans) -> Self {
        let mut borrowed: FxHashMap<ValueId, Took> = FxHashMap::default();
        let mut named = Vec::new();
        for block in &cfg.blocks {
            for inst in &block.insts {
                match &inst.kind {
                    InstKind::Ref { dst, .. } => {
                        borrowed.insert(
                            *dst,
                            Took {
                                span: inst.span,
                                form: HolderForm::Reference,
                            },
                        );
                    }
                    InstKind::MakeClosure { dst, .. } => {
                        borrowed.insert(
                            *dst,
                            Took {
                                span: inst.span,
                                form: HolderForm::Lambda,
                            },
                        );
                    }
                    // RFC-0029: a storage a reference was assigned into holds
                    // the loan too, and it was borrowed where the reference
                    // it was given was. A slot given a reference twice keeps
                    // the first, since the walk visits blocks in listing
                    // order and cannot say which loan is the live one.
                    InstKind::Assign { target, value, .. } => {
                        if let Some(slot) = inst_info::storage(target)
                            && let Some(took) = borrowed.get(value).copied()
                        {
                            borrowed.entry(slot).or_insert(took);
                        }
                    }
                    _ => {}
                }
                let called = called_by(loans, &inst.kind);
                named.extend(
                    loans
                        .uses_with_storage(&inst.kind)
                        .into_iter()
                        .map(|u| HolderUse {
                            holder: u,
                            span: inst.span,
                            kind: match called.contains(&u) {
                                true => UseKind::Call,
                                false => UseKind::Other,
                            },
                        }),
                );
            }
        }
        Self { borrowed, named }
    }

    fn borrowed_at(&self, holder: ValueId) -> Span {
        self.borrowed
            .get(&holder)
            .map_or(Span::ZERO, |took| took.span)
    }

    /// Where each holder took the loan, and the use of it that reaches past
    /// the conflict. Two holders given the same reference share one `Took`, so
    /// a label already stated is not stated again.
    fn labels(&self, holders: &[ValueId], conflict: Span) -> Vec<Label> {
        let mut labels: Vec<Label> = Vec::new();
        for holder in holders {
            let Some(took) = self
                .borrowed
                .get(holder)
                .filter(|took| took.span != Span::ZERO)
            else {
                continue;
            };
            let later = self.later_use(*holder, conflict);
            let places = std::iter::once(Label::at(took.span, took.form.took_it_here())).chain(
                later.map(|use_| Label::at(use_.span, took.form.keeps_it_live_here(use_.kind))),
            );
            for label in places {
                if !labels.contains(&label) {
                    labels.push(label);
                }
            }
        }
        labels
    }

    fn later_use(&self, holder: ValueId, conflict: Span) -> Option<&HolderUse> {
        self.named
            .iter()
            .filter(|u| u.holder == holder && u.span.start > conflict.start)
            .min_by_key(|u| (u.span.start, u.kind))
    }
}

impl Checking {
    fn check_exclusion(&self, errors: &mut Vec<ValidationError>) {
        let holders: FxHashSet<ValueId> = self
            .cfg
            .val_types
            .keys()
            .filter(|v| !self.loans.region(**v).loans.is_empty())
            .copied()
            .collect();
        if holders.is_empty() {
            return;
        }
        let live = liveness::analyze(&self.cfg);

        for (bi, block) in self.cfg.blocks.iter().enumerate() {
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
                for u in self.loans.uses_with_storage(&inst.kind) {
                    if holders.contains(&u) {
                        current.insert(u);
                    }
                }
                live_before[ii] = current.clone();
            }

            for (ii, inst) in block.insts.iter().enumerate() {
                let Some((target, touch)) = touch(&inst.kind, &self.cfg.val_types) else {
                    continue;
                };
                let reach = reached(&target, &self.loans);
                let mut holders: Vec<ValueId> = live_before[ii]
                    .iter()
                    .copied()
                    .filter(|holder| {
                        !reach.via.contains(holder)
                            && self.loans.region(*holder).loans.iter().any(|loan| {
                                reach.storage.contains(&loan.storage.value())
                                    && conflicts(loan, &touch)
                            })
                    })
                    .collect();
                if holders.is_empty() {
                    continue;
                }
                holders.sort_unstable_by_key(|holder| {
                    (self.sites.borrowed_at(*holder).start, *holder)
                });
                errors.push(ValidationError {
                    scope: self.scope.to_string(),
                    inst_index: ii,
                    span: inst.span,
                    kind: ValidationErrorKind::BorrowConflict {
                        storage: inst_info::storage(&target)
                            .and_then(|slot| self.cfg.debug.get(slot).cloned()),
                        touch: touch.stated(),
                        labels: self.sites.labels(&holders, inst.span),
                    },
                });
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
        }
        | Terminator::Diamond {
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
        Terminator::Return { value, order, .. } => {
            let mut v = vec![*value];
            v.extend(*order);
            v
        }
        Terminator::For {
            source,
            body_args,
            exit_args,
            ..
        } => {
            let mut v = source.uses().to_vec();
            v.extend(body_args);
            v.extend(exit_args);
            v
        }
        Terminator::Switch { tag, arms, default } => {
            let mut v = vec![*tag];
            for (_, _, args) in arms {
                v.extend(args);
            }
            if let Some((_, args)) = default {
                v.extend(args);
            }
            v
        }
        Terminator::Fallthrough | Terminator::Diverge => Vec::new(),
    }
}

// -- Tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DebugInfo, Inst, Label};
    use crate::ty::TypeArg;
    use acvus_ast::Span;
    use acvus_utils::{LocalFactory, LocalIdOps};

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
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 2,
            order_param: None,
            task: crate::ty::Task::Sync,
        }
    }

    fn module(main: MirBody) -> MirModule {
        MirModule {
            main,
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
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
            (
                v(1),
                Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String))),
            ),
            (
                v(2),
                Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::String))),
            ),
            (v(3), Ty::String),
            (v(4), Ty::I64),
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
        types[0] = (slot(), Ty::I64);
        types.push((v(7), Ty::I64));
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
        types[0] = (slot(), Ty::I64);
        types.push((v(7), Ty::I64));
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
        types.push((
            v(7),
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String))),
        ));
        types.push((
            v(8),
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String))),
        ));
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

    // -- The order the summaries are read in ---------------------------

    fn makes(label: u32) -> InstKind {
        InstKind::MakeClosure {
            dst: v(20 + label as usize),
            body: Label(label),
            captures: vec![],
        }
    }

    /// Main makes `L1`, `L1` makes `L2`, and `L3` is reached from neither.
    fn nested_closures() -> MirModule {
        let mut module = module(body(vec![makes(1), ret(0)], string_slot()));
        module
            .closures
            .insert(Label(1), body(vec![makes(2), ret(0)], string_slot()));
        module
            .closures
            .insert(Label(2), body(vec![ret(0)], string_slot()));
        module
            .closures
            .insert(Label(3), body(vec![ret(0)], string_slot()));
        module
    }

    #[test]
    fn every_closure_is_checked_before_the_body_that_makes_it() {
        let module = nested_closures();
        let order = inner_closures_first(&module);
        let at = |label: u32| {
            order
                .iter()
                .position(|l| *l == Label(label))
                .unwrap_or_else(|| panic!("{label} is missing from {order:?}"))
        };
        assert!(at(2) < at(1), "{order:?}");
    }

    #[test]
    fn the_order_names_every_closure_once_reached_or_not() {
        let module = nested_closures();
        let mut order = inner_closures_first(&module);
        assert_eq!(order.len(), module.closures.len(), "{order:?}");
        order.sort_unstable_by_key(|label| label.0);
        order.dedup();
        assert_eq!(order.len(), module.closures.len(), "{order:?}");
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
        main.val_types.insert(
            param,
            Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::String))),
        );
        assert_eq!(conflict_count(&errors(main)), 1);
    }
}
