//! borrowck (ownership §5): while a loan lives, its place is reached only
//! as the loan allows, and nothing that outlives the body holds a loan on
//! its own storage. It reads `Loans` for what each holder holds and each
//! site's accesses, `Liveness` and `ContentLiveness` for which holders are
//! live.
//!
//! A loan lives while a holder of it, or of a loan within it, is live. The
//! content of an outside storage is live throughout the body: its holder is
//! the caller. A value that is only one call's argument holds its loans
//! from that call on.
//!
//! At every access `Loans` records, every live loan whose place overlaps the
//! accessed one is held against it, save the loans the reference the access
//! is made through holds, and those they are within:
//! - a write, a move and an exclusive lend meet every such loan;
//! - a read and a shared lend meet an exclusive one.
//! Loans made at one site meet each other where their places overlap and
//! one is exclusive. A loan an access meets refuses it `LoanConflict`, at the
//! access's span, naming where the loan was made, a parameter's or capture's
//! at its binder.
//!
//! At every `Return`, a loan on the body's own storage, a slot or a
//! by-value parameter's, that the result or an outside storage's content
//! holds refuses it `LoanOutlivesBody`, naming where the loan was made.

use std::collections::BTreeSet;

use acvus_ast::Span;
use rustc_hash::FxHashSet;

use super::Refusal;
use crate::ir::cfg::{BlockId, Body, BodyId, Inst, Module, Op, Site};
use crate::ir::parts::{Path, overlap};
use crate::ir::{RefTarget, ValueId};
use crate::passes::analysis::{AnalysisManager, ContentLiveness, Dominators, Liveness, Loan, LoanSet, Loans, Mode, Storage};
use crate::ty::{Mutability, TyKind, copies};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    for Met { access, loan } in met(module, body, am) {
        out.push(Refusal { kind: super::RefusalKind::LoanConflict { loan: made_at(&loan) }, span: access });
    }
}

/// Where `loan` was made, which `LoanConflict` names.
fn made_at(loan: &Loan) -> Span {
    todo!(
        "where {loan:?} was made: a Loan names its storage, path and mode, but no site it was made at; a reborrow holds \
         its source's loans with the path extended and the mode weakened, and a parameter's loans are made at no site \
         of the body, so the span is not determined by the Loans structure"
    )
}

/// A live loan an access meets: the access's span, and the loan.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Met {
    access: Span,
    loan: Loan,
}

/// What an access does to its place, which decides the loans it meets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    Read,
    Write,
    Move,
    Lend(Mutability),
}

impl Reach {
    fn meets(self, loan: &Loan) -> bool {
        match self {
            Reach::Read | Reach::Lend(Mutability::Shared) => loan.mode == Mode::Exclusive,
            Reach::Write | Reach::Move | Reach::Lend(Mutability::Mut) => true,
        }
    }
}

/// A place an access reaches: a storage and a path from its root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    storage: Storage,
    path: Path,
}

/// An access of a place by an instruction.
struct Access<'i> {
    target: &'i RefTarget,
    path: &'i [crate::ir::PathSeg],
    reach: Reach,
}

/// The access `inst` makes, where it makes one.
///
/// A `Take` moves what does not copy, except a `&mut`, which each use
/// reborrows and its place keeps, and a take-out for a call, which moves
/// whatever it takes (`Op::Take`).
fn access<'i>(body: &Body, module: &Module, inst: &'i Inst) -> Option<Access<'i>> {
    let types = module.types();
    let (target, path, reach) = match inst.op() {
        Op::Ref { target, path, mutability } => (target, path, Reach::Lend(*mutability)),
        Op::Assign { target, path, .. } => (target, path, Reach::Write),
        Op::Take { target, path, taken_out } => {
            let ty = body.ty_of(inst.result().expect("a take yields a value"));
            let reborrowed = matches!(types.kind(ty), TyKind::Ref(Mutability::Mut, _));
            let moves = *taken_out || !(copies(types, ty) || reborrowed);
            (target, path, if moves { Reach::Move } else { Reach::Read })
        }
        _ => return None,
    };
    Some(Access { target, path, reach })
}

/// The places an access of `target` at `path` reaches: by name, the
/// storage and the path; through a reference, each place a loan at its own
/// position names, the path extended by `path`.
fn accessed(loans: &Loans, target: &RefTarget, path: &[crate::ir::PathSeg]) -> BTreeSet<Place> {
    match target {
        RefTarget::Var(slot) => BTreeSet::from([Place { storage: Storage::Slot(*slot), path: path.to_vec() }]),
        RefTarget::Param(param) => BTreeSet::from([Place { storage: Storage::Param(*param), path: path.to_vec() }]),
        RefTarget::Through(reference) => loans
            .of(*reference, 0)
            .iter()
            .map(|loan| Place { storage: loan.storage, path: [&loan.path[..], path].concat() })
            .collect(),
    }
}

/// Every live loan each access of `body` meets, in the body's order.
fn met(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>) -> Vec<Met> {
    let loans = am.get::<Loans>(module, body_id);
    let liveness = am.get::<Liveness>(module, body_id);
    let contents = am.get::<ContentLiveness>(module, body_id);
    let reachable = am.get::<Dominators>(module, body_id);
    let body = module.body(body_id);
    let slots: FxHashSet<ValueId> = body.slots().iter().copied().collect();

    let mut met = Vec::new();
    for (id, block) in body.blocks().filter(|(id, _)| reachable.reachable(*id)) {
        let insts: Vec<&Inst> = block.insts().collect();
        let live_values = live_before_each(&liveness, id, block.terminator().operands(), &insts);
        for (inst, live) in insts.iter().zip(&live_values) {
            let Some(Access { target, path, reach }) = access(body, module, inst) else {
                continue;
            };
            let places = accessed(&loans, target, path);
            let own = match target {
                RefTarget::Through(reference) => loans.all_of(*reference),
                RefTarget::Var(_) | RefTarget::Param(_) => LoanSet::new(),
            };

            let held_by_values = live.iter().filter(|value| !slots.contains(value)).flat_map(|value| loans.all_of(*value));
            let site = Site::Inst(inst.id());
            let held_by_contents = contents.live_before(inst.id()).iter().flat_map(|place| {
                let storage = if slots.contains(place) { Storage::Slot(*place) } else { Storage::Param(*place) };
                loans.content_before(site, storage).iter().flatten().cloned().collect::<Vec<Loan>>()
            });
            let live_loans: BTreeSet<Loan> = held_by_values.chain(held_by_contents).collect();

            for loan in live_loans {
                let overlaps = places.iter().any(|place| loan.storage == place.storage && overlap(&loan.path, &place.path));
                if overlaps && reach.meets(&loan) && !own.contains(&loan) {
                    met.push(Met { access: inst.span(), loan });
                }
            }
        }
    }
    met
}

/// The values live just before each of `insts`, `block`'s instructions: from
/// what `Liveness` has live after its terminator, which reads `terminator`,
/// each instruction back to front ends what it defines and reads its
/// operands.
fn live_before_each(
    liveness: &Liveness,
    block: BlockId,
    terminator: impl IntoIterator<Item = ValueId>,
    insts: &[&Inst],
) -> Vec<FxHashSet<ValueId>> {
    let mut live: FxHashSet<ValueId> = liveness.live_at_end(block).clone();
    live.extend(terminator);
    let mut before = vec![FxHashSet::default(); insts.len()];
    for (at, inst) in insts.iter().enumerate().rev() {
        if let Some(result) = inst.result() {
            live.remove(&result);
        }
        if let Some(token) = inst.order().and_then(|order| order.yielded()) {
            live.remove(&token);
        }
        live.extend(inst.operands());
        before[at] = live.clone();
    }
    before
}

#[cfg(test)]
mod tests;
