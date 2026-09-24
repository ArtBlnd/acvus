//! The exclusion rule of RFC-0018, checked over a body's CFG.
//!
//! Exclusion: while a `&mut` to a storage is live, no other name of that
//! storage — the storage itself, a `&`, another `&mut` — is read or
//! written; while a `&` is live, the storage is not assigned, moved out of,
//! or mutably referenced. A holder of a loan — a value, or a storage a
//! reference was assigned into — is live from its definition to its last
//! use, counted as liveness counts them (RFC-0029).

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::loans::{
    Held, HeldInput, Loan, Loans, RegionsAt, Via, held_positions, positions,
};
use crate::analysis::{inst_info, liveness};
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::ir::{Callee, InstKind, Label as ClosureLabel, MirBody, MirModule, RefTarget, ValueId};
use crate::ty::{Alignment, FlowEnd, Flows, Mutability, Ty};
use crate::validate::move_check::is_move_only;
use crate::validate::type_check::{ConflictTouch, ValidationError, ValidationErrorKind};
use acvus_ast::Span;
use acvus_ast::report::Label;

/// The exclusion rule alone, which holds of a module at any point in the
/// pipeline.
pub fn check_borrows(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let promoted = Promoted::of(module);
    for checking in Bodies::of(&promoted).all() {
        checking.check_exclusion(&mut errors);
    }
    errors
}

/// The exclusion rule and the output rule (RFC-0079 rules 5 and 9)
/// together, on the shape the source wrote: every loan a body's result
/// holds, and every loan it writes into what an input names, is a position
/// of an input its function type's flows name.
///
/// The output rule is checked here, in pass 0, and not by `check_borrows`,
/// and that is a decision: the flows are the checker's inference from the
/// source, and a pass that splices a callee into its caller leaves a body
/// whose outputs the inference never saw.
pub fn check_outputs_and_borrows(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let promoted = Promoted::of(module);
    let bodies = Bodies::of(&promoted);
    let made = closure_flows(module);
    for (label, checking) in &bodies.closures {
        for flows in made.get(label).into_iter().flatten() {
            checking.check_outputs(flows, &mut errors);
        }
        checking.check_exclusion(&mut errors);
    }
    if let Some(main) = &bodies.main {
        main.check_outputs(&module.flows, &mut errors);
        main.check_exclusion(&mut errors);
    }
    errors
}

/// The flows of each closure body, off the function type of every
/// `MakeClosure` that makes it. A body nothing makes is never called and has
/// none to be checked against.
fn closure_flows(module: &MirModule) -> FxHashMap<ClosureLabel, Vec<Flows>> {
    let mut made: FxHashMap<ClosureLabel, Vec<Flows>> = FxHashMap::default();
    for body in std::iter::once(&module.main).chain(module.closures.values()) {
        for inst in &body.insts {
            let InstKind::MakeClosure {
                dst, body: label, ..
            } = &inst.kind
            else {
                continue;
            };
            let flows = match body.val_types.get(dst) {
                Some(Ty::Fn { flows, .. }) => flows.get().clone(),
                // A closure the validator finds untyped is refused there;
                // the union is what it is checked against here.
                Some(_) | None => Flows::Every,
            };
            made.entry(*label).or_default().push(flows);
        }
    }
    made
}

/// Each body of a module as a CFG, the ones with no blocks left out: they
/// hold nothing to check.
struct Promoted {
    main: Option<CfgBody>,
    closures: Vec<(ClosureLabel, CfgBody)>,
}

impl Promoted {
    fn of(module: &MirModule) -> Self {
        let promoted = |body: &MirBody| {
            let cfg = promote(body.clone());
            match cfg.blocks.is_empty() {
                true => None,
                false => Some(cfg),
            }
        };
        Self {
            main: promoted(&module.main),
            closures: module
                .closures
                .iter()
                .filter_map(|(label, closure)| Some((*label, promoted(closure)?)))
                .collect(),
        }
    }
}

struct Bodies<'cfg> {
    main: Option<Checking<'cfg>>,
    closures: Vec<(ClosureLabel, Checking<'cfg>)>,
}

impl<'cfg> Bodies<'cfg> {
    fn of(promoted: &'cfg Promoted) -> Self {
        Self {
            main: promoted
                .main
                .as_ref()
                .map(|cfg| Checking::of("main".to_string(), cfg)),
            closures: promoted
                .closures
                .iter()
                .map(|(label, cfg)| (*label, Checking::of(format!("closure({label:?})"), cfg)))
                .collect(),
        }
    }

    fn all(&self) -> impl Iterator<Item = &Checking<'cfg>> {
        self.main
            .iter()
            .chain(self.closures.iter().map(|(_, checking)| checking))
    }
}

/// One body under the borrow check, with the analyses every rule reads.
struct Checking<'cfg> {
    scope: String,
    loans: Loans<'cfg>,
    sites: HolderSites,
}

/// Why an output's loan is refused, each reported once.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Refused {
    Local(ValueId),
    NotStated { to: FlowEnd, from: FlowEnd },
}

/// An output of a body and where it was found to hold a loan.
struct Output {
    end: FlowEnd,
    /// The output's position the loan is at, where the output is the result.
    position: Option<usize>,
    loan: Loan,
    at: ValueId,
}

impl<'cfg> Checking<'cfg> {
    fn of(scope: String, cfg: &'cfg CfgBody) -> Self {
        let loans = Loans::build(cfg);
        let sites = HolderSites::of(&loans);
        Self {
            scope,
            loans,
            sites,
        }
    }

    fn cfg(&self) -> &'cfg CfgBody {
        self.loans.cfg()
    }

    /// Every loan in the body's outputs: its result's positions, and what it
    /// wrote into the storage a parameter or a capture names from outside.
    fn outputs(&self) -> Vec<Output> {
        let mut outputs = Vec::new();
        for block in &self.cfg().blocks {
            let Terminator::Return { value, .. } = block.terminator else {
                continue;
            };
            let regions = self.loans.regions(value);
            let width = self.cfg().val_types.get(&value).map_or(0, positions);
            for position in 0..width {
                for loan in regions.position(position) {
                    outputs.push(Output {
                        end: FlowEnd::Result,
                        position: Some(position),
                        loan: *loan,
                        at: value,
                    });
                }
            }
        }
        let entries = self
            .cfg()
            .params
            .iter()
            .enumerate()
            .map(|(index, (_, value))| (FlowEnd::Param(index), *value))
            .chain(
                self.cfg()
                    .captures
                    .iter()
                    .map(|(_, value)| (FlowEnd::Captures, *value)),
            );
        for (end, entry) in entries {
            for loan in self.loans.written_into(entry) {
                outputs.push(Output {
                    end,
                    position: None,
                    loan,
                    at: entry,
                });
            }
        }
        outputs
    }

    /// RFC-0079 rules 5 and 9: an output holds only loans on the body's
    /// inputs, each one its flows name. A loan on the body's own storage in
    /// an output outlives the storage: `ReferenceToLocalLeavesBody`.
    fn check_outputs(&self, flows: &Flows, errors: &mut Vec<ValidationError>) {
        let result_width = self
            .cfg()
            .blocks
            .iter()
            .find_map(|block| match block.terminator {
                Terminator::Return { value, .. } => {
                    Some(self.cfg().val_types.get(&value).map_or(0, positions))
                }
                _ => None,
            });
        let mut refused: Vec<Refused> = Vec::new();
        for Output {
            end,
            position,
            loan,
            at,
        } in self.outputs()
        {
            let found = match self.loans.held(&loan) {
                Held::Local(local) => Refused::Local(local),
                Held::Input(input) if self.admits(flows, end, position, input, result_width) => {
                    continue;
                }
                Held::Input(input) => Refused::NotStated {
                    to: end,
                    from: input.end,
                },
            };
            if refused.contains(&found) {
                continue;
            }
            refused.push(found);
            let kind = match found {
                Refused::Local(local) => ValidationErrorKind::ReferenceToLocalLeavesBody {
                    storage: self.cfg().debug.get(local).cloned(),
                },
                Refused::NotStated { to, from } => ValidationErrorKind::FlowNotStated { to, from },
            };
            errors.push(ValidationError {
                scope: self.scope.clone(),
                inst_index: 0,
                span: self.sites.borrowed_at(at),
                kind,
            });
        }
    }

    /// Whether `flows` let output `to` hold `input`: an `Any` flow between
    /// the two ends, or an `Aligned` one where the positions agree or the
    /// shapes differ, which a call reads as `Any`.
    fn admits(
        &self,
        flows: &Flows,
        to: FlowEnd,
        position: Option<usize>,
        input: HeldInput,
        result_width: Option<usize>,
    ) -> bool {
        if flows.admits(to, input.end, Alignment::Any) {
            return true;
        }
        if !flows.admits(to, input.end, Alignment::Aligned) {
            return false;
        }
        let input_width = match input.end {
            FlowEnd::Param(index) => self
                .cfg()
                .params
                .get(index)
                .and_then(|(_, value)| self.cfg().val_types.get(value))
                .map(positions),
            FlowEnd::Captures => Some(1),
            FlowEnd::Result => None,
        };
        match position {
            Some(position) => position == input.position || input_width != result_width,
            None => true,
        }
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
#[derive(Clone, Copy)]
enum Touch {
    Reference(Mutability),
    /// A move out; `false` when the storage keeps a copy (a primitive).
    Take {
        moves: bool,
    },
    Assign,
}

impl Touch {
    /// This touch made through a loan of `mutability`.
    fn bounded(&self, mutability: Mutability) -> Touch {
        match (mutability, self) {
            (Mutability::Mut, touch) => *touch,
            (Mutability::Shared, Touch::Reference(_) | Touch::Assign) => {
                Touch::Reference(Mutability::Shared)
            }
            (Mutability::Shared, Touch::Take { .. }) => Touch::Take { moves: false },
        }
    }

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

/// Where a touch lands: a place, or the storages a slot's position names.
enum Touched {
    Place(RefTarget),
    Held { slot: ValueId, position: usize },
}

/// What an instruction does to storages. A loan read out of a storage is
/// the storage's own reference (RFC-0029), taken again: it touches, through
/// the storage, what each position of the storage the value is read from
/// names, as the loan's mutability does.
fn touches(kind: &InstKind, val_types: &FxHashMap<ValueId, Ty>) -> Vec<(Touched, Touch)> {
    match kind {
        InstKind::Ref {
            target, mutability, ..
        } => vec![(
            Touched::Place(target.clone()),
            Touch::Reference(*mutability),
        )],
        InstKind::Take {
            dst, target, path, ..
        } => {
            let mut touches = vec![(
                Touched::Place(target.clone()),
                Touch::Take {
                    moves: take_moves(val_types, dst),
                },
            )];
            if let Some(slot) = inst_info::storage(target)
                && let (Some(slot_ty), Some(taken)) = (val_types.get(&slot), val_types.get(dst))
            {
                touches.extend(held_positions(slot_ty, path, taken).into_iter().map(
                    |(position, mutability)| {
                        (
                            Touched::Held { slot, position },
                            Touch::Reference(mutability),
                        )
                    },
                ));
            }
            touches
        }
        InstKind::Assign { target, .. } => vec![(Touched::Place(target.clone()), Touch::Assign)],
        _ => Vec::new(),
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

/// The storages a touch reaches, each by the loan it is reached through: a
/// touch through a shared loan only reads, as `Loans::storage_effect` bounds
/// it.
struct Reached {
    storage: Vec<Loan>,
    via: Via,
}

fn reached(touched: &Touched, loans: &Loans<'_>, regions: &RegionsAt<'_>) -> Reached {
    match touched {
        Touched::Place(RefTarget::Var(s) | RefTarget::Param(s)) => Reached {
            storage: vec![Loan {
                storage: loans.storage_of(*s),
                mutability: Mutability::Mut,
            }],
            via: Via::new(),
        },
        Touched::Place(RefTarget::Through(r)) => {
            let held = regions.regions(*r);
            Reached {
                storage: held.names().to_vec(),
                via: held.via().with(*r),
            }
        }
        Touched::Held { slot, position } => {
            let held = regions.regions(*slot);
            Reached {
                storage: held.position(*position).to_vec(),
                via: held.via().with(*slot),
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
fn called_by(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
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
        for slot in loans
            .holds(names[at])
            .filter_map(|loan| loan.storage.slot())
        {
            if !names.contains(&slot) {
                names.push(slot);
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
    fn of(loans: &Loans<'_>) -> Self {
        let mut borrowed: FxHashMap<ValueId, Took> = FxHashMap::default();
        let mut named = Vec::new();
        for block in &loans.cfg().blocks {
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

impl Checking<'_> {
    fn check_exclusion(&self, errors: &mut Vec<ValidationError>) {
        let holders: FxHashSet<ValueId> = self
            .cfg()
            .val_types
            .keys()
            .filter(|v| self.loans.regions(**v).holds_any())
            .copied()
            .collect();
        if holders.is_empty() {
            return;
        }
        let live = liveness::analyze_with(&self.loans);

        let mut found: Vec<Conflict> = Vec::new();
        for (bi, block) in self.cfg().blocks.iter().enumerate() {
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

            // A holder conflicts by the loans it holds where the touch is,
            // not by every loan it takes anywhere in the body.
            let mut regions = self.loans.at_entry(BlockIdx(bi));
            for (ii, inst) in block.insts.iter().enumerate() {
                for (target, touch) in touches(&inst.kind, &self.cfg().val_types) {
                    let reach = reached(&target, &self.loans, &regions);
                    let mut holders: Vec<ValueId> = live_before[ii]
                        .iter()
                        .copied()
                        .filter(|holder| {
                            !reach.via.contains(holder)
                                && regions.regions(*holder).holds().any(|loan| {
                                    reach.storage.iter().any(|through| {
                                        through.storage == loan.storage
                                            && conflicts(loan, &touch.bounded(through.mutability))
                                    })
                                })
                        })
                        .collect();
                    if holders.is_empty() {
                        continue;
                    }
                    holders.sort_unstable_by_key(|holder| {
                        (self.sites.borrowed_at(*holder).start, *holder)
                    });
                    found.push(Conflict {
                        made: inst_info::defs(&inst.kind).into_vec(),
                        error: ValidationError {
                            scope: self.scope.to_string(),
                            inst_index: ii,
                            span: inst.span,
                            kind: ValidationErrorKind::BorrowConflict {
                                storage: match &target {
                                    Touched::Place(target) => inst_info::storage(target)
                                        .and_then(|slot| self.cfg().debug.get(slot).cloned()),
                                    Touched::Held { .. } => None,
                                },
                                touch: touch.stated(),
                                labels: self.sites.labels(&holders, inst.span),
                            },
                        },
                        holders,
                    });
                }
                regions.pass();
            }
        }
        errors.extend(self.without_consequences(found));
    }

    /// A reference a refused touch made is that refusal's: a later conflict
    /// whose every holder is built from one is the same conflict again, and
    /// is not reported beside it. Only an earlier refusal answers for a
    /// later one, so the first of any chain is always reported.
    fn without_consequences(&self, found: Vec<Conflict>) -> Vec<ValidationError> {
        let answered = |conflict: &Conflict| {
            conflict.holders.iter().all(|holder| {
                found.iter().any(|earlier| {
                    earlier.error.span.start < conflict.error.span.start
                        && earlier.made.iter().any(|made| {
                            made == holder || self.loans.regions(*holder).via().contains(made)
                        })
                })
            })
        };
        found
            .iter()
            .filter(|conflict| !answered(conflict))
            .map(|conflict| conflict.error.clone())
            .collect()
    }
}

struct Conflict {
    /// What the refused instruction defines.
    made: Vec<ValueId>,
    holders: Vec<ValueId>,
    error: ValidationError,
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
            source, exit_args, ..
        } => {
            let mut v = source.uses().to_vec();
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
            declared_params: main.params.len(),
            main,
            closures: FxHashMap::default(),
            ret: crate::ty::Ty::Unit,
            flows: crate::ty::Flows::Every,
            fetched_first: Vec::new(),
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
            taken_out: false,
        }
    }

    fn assign(value: usize) -> InstKind {
        InstKind::Assign {
            target: RefTarget::Var(slot()),
            path: vec![],
            value: v(value),
            restores: false,
        }
    }

    fn load(dst: usize, src: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Through(v(src)),
            path: vec![],
            taken_out: false,
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
                    restores: false,
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
            restores: false,
        }
    }

    fn take_slot(dst: usize, slot: usize) -> InstKind {
        InstKind::Take {
            dst: v(dst),
            target: RefTarget::Var(v(slot)),
            path: vec![],
            taken_out: false,
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
                    restores: false,
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

    #[test]
    fn a_parameter_stored_and_read_back_is_reborrowed_as_itself() {
        let param = v(9);
        let mutable = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::String)));
        let mut main = body(
            vec![
                store(1, 9),
                take_slot(2, 1),
                InstKind::Ref {
                    dst: v(3),
                    target: RefTarget::Through(v(2)),
                    path: vec![],
                    mutability: Mutability::Shared,
                },
                load(4, 3),
                ret(4),
            ],
            vec![(v(1), mutable.clone()), (v(2), mutable.clone())],
        );
        main.params
            .push((acvus_utils::Interner::new().intern("p"), param));
        main.val_types.insert(param, mutable);
        assert_eq!(conflict_count(&errors(main)), 0);
    }

    #[test]
    fn a_mutable_reference_read_out_of_its_storage_while_another_read_is_live_is_rejected() {
        let mutable = Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::String)));
        let main = body(
            vec![
                reference(1, Mutability::Mut),
                store(2, 1),
                take_slot(3, 2),
                take_slot(4, 2),
                load(5, 3),
                load(6, 4),
                ret(5),
            ],
            vec![
                (v(1), mutable.clone()),
                (v(2), mutable.clone()),
                (v(3), mutable.clone()),
                (v(4), mutable),
                (slot(), Ty::String),
                (v(5), Ty::String),
                (v(6), Ty::String),
            ],
        );
        // The second read is refused where it is made, before either
        // reference is used; the use through the first is the second.
        assert_eq!(conflict_count(&errors(main)), 2);
    }

    fn param_mut_word() -> (ValueId, Ty) {
        (
            v(9),
            Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::I64))),
        )
    }

    fn shared_through(dst: usize, through: usize) -> InstKind {
        InstKind::Ref {
            dst: v(dst),
            target: RefTarget::Through(v(through)),
            path: vec![],
            mutability: Mutability::Shared,
        }
    }

    #[test]
    fn two_shared_reborrows_of_one_mutable_reference_read_side_by_side() {
        let (param, ty) = param_mut_word();
        let mut main = body(
            vec![
                shared_through(1, 9),
                shared_through(2, 9),
                load(3, 1),
                load(4, 2),
                ret(3),
            ],
            vec![(v(3), Ty::I64), (v(4), Ty::I64)],
        );
        main.params
            .push((acvus_utils::Interner::new().intern("p"), param));
        main.val_types.insert(param, ty.clone());
        main.val_types.insert(
            v(1),
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::I64))),
        );
        main.val_types.insert(
            v(2),
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::I64))),
        );
        assert_eq!(conflict_count(&errors(main)), 0);
    }

    #[test]
    fn a_write_through_the_mutable_reference_while_its_shared_reborrow_lives_is_rejected() {
        let (param, ty) = param_mut_word();
        let mut main = body(
            vec![
                shared_through(1, 9),
                InstKind::Assign {
                    target: RefTarget::Through(param),
                    path: vec![],
                    value: v(5),
                    restores: false,
                },
                load(3, 1),
                ret(3),
            ],
            vec![(v(3), Ty::I64), (v(5), Ty::I64)],
        );
        main.params
            .push((acvus_utils::Interner::new().intern("p"), param));
        main.val_types.insert(param, ty);
        main.val_types.insert(
            v(1),
            Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::I64))),
        );
        assert_eq!(conflict_count(&errors(main)), 1);
    }
}
