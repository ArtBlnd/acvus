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

use acvus_ast::Span;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{Refusal, RefusalKind};
use crate::ir::ValueId;
use crate::ir::cfg::{BlockId, Body, BodyId, Inst, InstId, Module, Site, Terminator};
use crate::passes::analysis::{
    Access, AnalysisManager, ContentLiveness, Dominators, Liveness, Loan, LoanSet, Loans, Made, Mode, Reach, Storage,
};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let body = module.body(body_id);
    let conflicts = met(module, body_id, am).into_iter().map(|Met { access, loan }| Refusal {
        kind: RefusalKind::LoanConflict { loan: made_at(body, &loan) },
        span: access,
    });
    let outliving = outlived(module, body_id, am).into_iter().map(|Outlived { at, loan }| Refusal {
        kind: RefusalKind::LoanOutlivesBody { loan: made_at(body, &loan) },
        span: at,
    });
    for refusal in conflicts.chain(outliving) {
        if !out.contains(&refusal) {
            out.push(refusal);
        }
    }
}

/// Where `loan` was made, which a refusal names: a site's span, and a
/// parameter's or capture's binder.
fn made_at(body: &Body, loan: &Loan) -> Span {
    match loan.made {
        Made::At(site) => span_of(body, site),
        Made::Outside { of } => body.binder_span(of),
    }
}

fn span_of(body: &Body, site: Site) -> Span {
    match site {
        Site::Inst(inst) => body.inst(inst).span(),
        Site::Term(block) => body.block(block).terminator_span(),
    }
}

/// A live loan an access meets: the access's span, and the loan.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Met {
    access: Span,
    loan: Loan,
}

/// A loan on the body's own storage that outlives it: the `Return`'s span,
/// and the loan.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Outlived {
    at: Span,
    loan: Loan,
}

/// Whether `access` meets `loan`, a live loan it is not made through: a
/// write, a move and an exclusive lend meet every loan whose place overlaps
/// the accessed one, a read and a shared lend an exclusive one.
fn meets(access: &Access, loan: &Loan) -> bool {
    match access.reach {
        Reach::Read => loan.mode == Mode::Exclusive && access.places.iter().any(|place| place.meets(loan)),
        Reach::Write | Reach::Move => access.places.iter().any(|place| place.meets(loan)),
        Reach::Lend => access
            .made
            .iter()
            .any(|made| (made.mode == Mode::Exclusive || loan.mode == Mode::Exclusive) && made.place().meets(loan)),
    }
}

/// Every live loan each access of `body` meets, and every two loans made
/// at one site that meet each other, in the body's order.
fn met(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>) -> Vec<Met> {
    let loans = am.get::<Loans>(module, body_id);
    let liveness = am.get::<Liveness>(module, body_id);
    let contents = am.get::<ContentLiveness>(module, body_id);
    let reachable = am.get::<Dominators>(module, body_id);
    let body = module.body(body_id);
    let slots: FxHashSet<ValueId> = body.slots().iter().copied().collect();
    let order = SiteOrder::of(body);
    let live_loans = |site: Site, values: &FxHashSet<ValueId>, places: &FxHashSet<ValueId>| -> LoanSet {
        let mut held = LoanSet::new();
        for value in values.iter().filter(|value| !slots.contains(value)) {
            let from = loans.holds_from(*value);
            let not_yet_made = |loan: &Loan| from.is_some_and(|call| loan.made == Made::At(call) && !order.after(site, call));
            held.extend(loans.all_of(*value).into_iter().filter(|loan| !not_yet_made(loan)));
        }
        for (storage, content) in loans.contents_before(site) {
            let live = match storage {
                Storage::Slot(place) | Storage::Param(place) => places.contains(place),
                Storage::Outside { .. } => true,
            };
            if live {
                held.extend(content.iter().flatten().cloned());
            }
        }
        let ancestors = loans.ancestors(&held);
        held.extend(ancestors);
        held
    };

    let mut met = Vec::new();
    for (id, block) in body.blocks().filter(|(id, _)| reachable.reachable(*id)) {
        let insts: Vec<&Inst> = block.insts().collect();
        let terminator = block.terminator();
        let live_values = live_before_each(&liveness, id, terminator, &insts);
        for (inst, values) in insts.iter().zip(&live_values) {
            let site = Site::Inst(inst.id());
            let live = live_loans(site, values, contents.live_before(inst.id()));
            met_at(&loans, site, inst.span(), &live, &mut met);
        }
        let site = Site::Term(id);
        let values = live_before_terminator(&liveness, id, terminator);
        let live = live_loans(site, &values, contents.live_before_terminator(id));
        met_at(&loans, site, block.terminator_span(), &live, &mut met);
    }
    met
}

/// What the accesses of `site`, at `span`, meet among `live`, and each two
/// of its lends meet of each other's loans, pushed to `met`.
fn met_at(loans: &Loans, site: Site, span: Span, live: &LoanSet, met: &mut Vec<Met>) {
    let accesses = loans.accesses(site);
    for access in accesses {
        let mut excluded = access.through.clone();
        excluded.extend(loans.ancestors(&access.through));
        let meeting: LoanSet = live.iter().filter(|loan| !excluded.contains(*loan) && meets(access, loan)).cloned().collect();
        for loan in outermost(loans, &meeting) {
            push_once(met, Met { access: span, loan });
        }
    }
    let lends: Vec<&Access> = accesses.iter().filter(|access| access.reach == Reach::Lend).collect();
    for (at, first) in lends.iter().enumerate() {
        for second in &lends[at + 1..] {
            let one_run = match (first.on_edge, second.on_edge) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };
            if !one_run {
                continue;
            }
            for made in &first.made {
                let meets_made = |other: &Loan| {
                    (made.mode == Mode::Exclusive || other.mode == Mode::Exclusive)
                        && made.place().meets(other)
                        && !related(loans, made, other)
                };
                if second.made.iter().any(meets_made) {
                    push_once(met, Met { access: span, loan: made.clone() });
                }
            }
        }
    }
}

/// The loans of `meeting` not made through another of them. An access that
/// meets a loan meets every live loan it is made through, which holds its
/// place at no weaker a mode: that is one conflict, named once, at the
/// outermost loan.
fn outermost(loans: &Loans, meeting: &LoanSet) -> Vec<Loan> {
    meeting
        .iter()
        .filter(|loan| loans.ancestors(&LoanSet::from([(*loan).clone()])).is_disjoint(meeting))
        .cloned()
        .collect()
}

/// Whether one of the two loans is made through the other.
fn related(loans: &Loans, a: &Loan, b: &Loan) -> bool {
    loans.ancestors(&LoanSet::from([a.clone()])).contains(b) || loans.ancestors(&LoanSet::from([b.clone()])).contains(a)
}

fn push_once(met: &mut Vec<Met>, found: Met) {
    if !met.contains(&found) {
        met.push(found);
    }
}

/// At every `Return`, each loan on the body's own storage that the result
/// or an outside storage's content holds.
fn outlived(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>) -> Vec<Outlived> {
    let loans = am.get::<Loans>(module, body_id);
    let reachable = am.get::<Dominators>(module, body_id);
    let body = module.body(body_id);
    let mut outlived = Vec::new();
    for (id, block) in body.blocks().filter(|(id, _)| reachable.reachable(*id)) {
        let Terminator::Return { value, span, .. } = block.terminator() else {
            continue;
        };
        let mut held = loans.all_of(*value);
        for (storage, content) in loans.contents_before(Site::Term(id)) {
            if let Storage::Outside { .. } = storage {
                held.extend(content.iter().flatten().cloned());
            }
        }
        for loan in held.into_iter().filter(|loan| matches!(loan.storage, Storage::Slot(_) | Storage::Param(_))) {
            let found = Outlived { at: *span, loan };
            if !outlived.contains(&found) {
                outlived.push(found);
            }
        }
    }
    outlived
}

/// Where each instruction stands, and which blocks a run reaches after
/// leaving each block, so that a site is known to follow a call.
struct SiteOrder {
    position: FxHashMap<InstId, (BlockId, usize)>,
    successors: FxHashMap<BlockId, Vec<BlockId>>,
}

impl SiteOrder {
    fn of(body: &Body) -> SiteOrder {
        let mut position = FxHashMap::default();
        let mut successors = FxHashMap::default();
        for (id, block) in body.blocks() {
            position.extend(block.insts().enumerate().map(|(index, inst)| (inst.id(), (id, index))));
            successors.insert(id, block.terminator().edges().iter().map(|edge| edge.to).collect());
        }
        SiteOrder { position, successors }
    }

    /// Whether some run reaches `site` after `call` has started. Cost:
    /// O(blocks + edges).
    fn after(&self, site: Site, call: Site) -> bool {
        let Site::Inst(call) = call else {
            panic!("a call is an instruction");
        };
        let (block, index) = self.position[&call];
        let (site_block, later_in_block) = match site {
            Site::Inst(inst) => {
                let (site_block, site_index) = self.position[&inst];
                (site_block, site_index > index)
            }
            Site::Term(term) => (term, true),
        };
        if site_block == block && later_in_block {
            return true;
        }
        let mut seen: FxHashSet<BlockId> = FxHashSet::default();
        let mut pending: Vec<BlockId> = self.successors[&block].clone();
        while let Some(next) = pending.pop() {
            if !seen.insert(next) {
                continue;
            }
            if next == site_block {
                return true;
            }
            pending.extend(self.successors[&next].iter().copied());
        }
        false
    }
}

/// The values live just before each of `insts`, `block`'s instructions: from
/// what is live just before its terminator, each instruction back to front
/// ends what it defines and reads its operands.
fn live_before_each(liveness: &Liveness, block: BlockId, terminator: &Terminator, insts: &[&Inst]) -> Vec<FxHashSet<ValueId>> {
    let mut live = live_before_terminator(liveness, block, terminator);
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

/// The values live just before `block`'s terminator: those live after it,
/// and those it reads.
fn live_before_terminator(liveness: &Liveness, block: BlockId, terminator: &Terminator) -> FxHashSet<ValueId> {
    let mut live = liveness.live_at_end(block).clone();
    live.extend(terminator.operands());
    live
}

#[cfg(test)]
mod tests;
