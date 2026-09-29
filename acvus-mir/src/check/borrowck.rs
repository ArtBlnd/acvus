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
//!
//! Loans made at one site meet each other where their places overlap and
//! one is exclusive. Each pair of loans that meet, each loan taken up to the
//! outermost of the loans it is made through that the other is not, is one
//! `LoanConflict` (refusals: one fact, one path): at where the later loan was
//! made, naming where the earlier one was, a parameter's or capture's at its
//! binder, and every other access that met the two while both lived. An
//! access by name that meets a loan is one of its own, at the access.
//!
//! At every `Return`, a loan on the body's own storage, a slot or a
//! by-value parameter's, that the result or an outside storage's content
//! holds refuses it `LoanOutlivesBody`, naming where the loan was made. In a
//! lambda's body, every loan an output end holds from another input end is
//! covered by a flow its claimed function type states, a lambda's flows
//! being what its body does; a breach is the compiler's, and panics.

use acvus_ast::Span;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{Refusal, RefusalKind};
use crate::ir::ValueId;
use crate::ir::cfg::{BlockId, Body, BodyId, Inst, InstId, Module, Op, Site, Terminator};
use crate::passes::analysis::{
    Access, AnalysisManager, ContentLiveness, Dominators, Liveness, Loan, LoanSet, Loans, Made, Mode, Reach, Storage,
};
use crate::ty::{FlowEnd, Ty, TyKind};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let body = module.body(body_id);
    let conflicts = conflicts(body, &met(module, body_id, am)).into_iter().map(|Conflict { at, first, later_uses }| Refusal {
        kind: RefusalKind::LoanConflict { first, later_uses },
        span: at,
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
    hold_lambda_flows(module, body_id, am);
}

/// A lambda's flows are what its body does (ownership §5): at every
/// `Return`, each loan an output end holds that came from another input end
/// is covered by a flow its claimed function type states from that input
/// into that output.
///
/// # Panics
/// A loan no claimed flow covers: the user wrote no flows, so the claim
/// typeck made is the compiler's, and a breach is its defect, not a
/// refusal.
fn hold_lambda_flows(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>) {
    let Some(signature) = claimed_signature(module, body_id) else {
        return;
    };
    let TyKind::Fn { params, flows, .. } = module.types().kind(signature) else {
        panic!("a lambda's signature is a function type")
    };
    let (flows, arity) = (flows.get(), params.len());
    let loans = am.get::<Loans>(module, body_id);
    let reachable = am.get::<Dominators>(module, body_id);
    let body = module.body(body_id);
    assert_eq!(body.params().len(), arity, "{body_id:?} has a parameter for each of its signature's");
    let end_of = |value: ValueId| match body.params().iter().position(|(_, param)| *param == value) {
        Some(index) => FlowEnd::Param(index),
        None if body.captures().iter().any(|(_, capture)| *capture == value) => FlowEnd::Captures,
        None => panic!("{value:?}, an outside storage's owner, is a parameter or a capture of {body_id:?}"),
    };
    for (id, block) in body.blocks().filter(|(id, _)| reachable.reachable(*id)) {
        let Terminator::Return { value, .. } = block.terminator() else {
            continue;
        };
        let mut outputs: Vec<(FlowEnd, LoanSet)> = vec![(FlowEnd::Result, loans.all_of(*value))];
        for (storage, content) in loans.contents_before(Site::Term(id)) {
            if let Storage::Outside { of, .. } = storage {
                outputs.push((end_of(*of), content.iter().flatten().cloned().collect()));
            }
        }
        for (to, held) in outputs {
            for loan in held {
                let Storage::Outside { of, .. } = loan.storage else { continue };
                let from = end_of(of);
                let covered = from == to || flows.into_end(to, arity).iter().any(|source| source.from == from);
                assert!(
                    covered,
                    "lambda body {body_id:?} breaks its claimed flows: at the return of {id:?}, {to:?} holds {loan:?} from \
                     {from:?}, and {flows:?} of {signature:?} state no flow from {from:?} into {to:?}"
                );
            }
        }
    }
}

/// The function type the `MakeClosure` that names `body_id` pairs it with,
/// where one does: its `Fn` type, or the signature of its `Closures` type at
/// the body's place among its bodies.
fn claimed_signature(module: &Module, body_id: BodyId) -> Option<Ty> {
    for owner in module.body_ids() {
        let body = module.body(owner);
        for (_, block) in body.blocks() {
            for inst in block.insts() {
                let Op::MakeClosure { bodies, .. } = inst.op() else { continue };
                let Some(at) = bodies.iter().position(|named| *named == body_id) else { continue };
                let closure = body.ty_of(inst.result().expect("a closure is a value"));
                return Some(match module.types().kind(closure) {
                    TyKind::Fn { .. } => closure,
                    TyKind::Closures(signatures) => signatures[at],
                    other => panic!("a `MakeClosure` yields a function value, not {other:?}"),
                });
            }
        }
    }
    None
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

/// A live loan an access meets: the access's span; the loan the access
/// makes or is made through, none where it is by name; whether the access
/// makes it; and the met loan. Each of the two stands for the pair as
/// `side` takes it, and the met loan of an access by name for its outermost.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Met {
    access: Span,
    by: Option<Loan>,
    makes: bool,
    loan: Loan,
}

/// One refusal of conflicting loans: where the later loan was made, where
/// the earlier one was, and every other access that met the two while both
/// lived.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Conflict {
    at: Span,
    first: Span,
    later_uses: Vec<Span>,
}

/// What a conflict is one of: a pair of loans, or an access by name and
/// the loan it met, which is no pair.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Fact {
    Pair(Loan, Loan),
    ByName(Span, Loan),
}

impl Fact {
    fn of(met: &Met) -> Fact {
        match &met.by {
            Some(by) => Fact::Pair(by.clone().min(met.loan.clone()), by.clone().max(met.loan.clone())),
            None => Fact::ByName(met.access, met.loan.clone()),
        }
    }
}

/// Each fact `met` holds, refused once, in the order the body first meets
/// it (refusals: one fact, one path). A pair stands where its later loan
/// was made, the one an access made while the other lived; where no access
/// made either while the other lived, at the first access that met them.
fn conflicts(body: &Body, met: &[Met]) -> Vec<Conflict> {
    let mut facts: Vec<(Fact, Vec<&Met>)> = Vec::new();
    for each in met {
        let fact = Fact::of(each);
        match facts.iter_mut().find(|(known, _)| *known == fact) {
            Some((_, events)) => events.push(each),
            None => facts.push((fact, vec![each])),
        }
    }
    let mut conflicts = Vec::new();
    for (_, events) in facts {
        let maker = events.iter().find(|event| event.makes);
        let (at, earlier, later) = match maker {
            Some(Met { by: Some(later), loan, .. }) => (made_at(body, later), loan, Some(later)),
            Some(Met { by: None, .. }) => panic!("an access that makes a loan is made with it"),
            None => (events[0].access, &events[0].loan, None),
        };
        let mut later_uses: Vec<Span> = Vec::new();
        for event in &events {
            let making = event.makes && later.is_some() && event.by.as_ref() == later;
            if !making && event.access != at && !later_uses.contains(&event.access) {
                later_uses.push(event.access);
            }
        }
        conflicts.push(Conflict { at, first: made_at(body, earlier), later_uses });
    }
    conflicts
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
        let (made_by, makes) = match access.reach {
            Reach::Lend => (&access.made, true),
            Reach::Read | Reach::Write | Reach::Move => (&access.through, false),
        };
        for loan in &meeting {
            if made_by.is_empty() {
                for outer in outermost(loans, &with_ancestors(loans, loan)) {
                    push_once(met, Met { access: span, by: None, makes, loan: outer });
                }
            }
            for by in made_by {
                let (ours, theirs) = (side(loans, by, loan), side(loans, loan, by));
                for (by, loan) in ours.iter().flat_map(|ours| theirs.iter().map(move |theirs| (ours, theirs))) {
                    push_once(met, Met { access: span, by: Some(by.clone()), makes, loan: loan.clone() });
                }
            }
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
                let meets_made = |other: &&Loan| {
                    (made.mode == Mode::Exclusive || other.mode == Mode::Exclusive)
                        && made.place().meets(other)
                        && !related(loans, made, other)
                };
                for other in second.made.iter().filter(meets_made) {
                    push_once(met, Met { access: span, by: Some(other.clone()), makes: true, loan: made.clone() });
                }
            }
        }
    }
}

/// `loan` and every loan it is made through.
fn with_ancestors(loans: &Loans, loan: &Loan) -> LoanSet {
    let mut chain = loans.ancestors(&LoanSet::from([loan.clone()]));
    chain.insert(loan.clone());
    chain
}

/// What `by` stands for in a pair with `other` (one fact, one refusal): the
/// outermost of `by` and the loans it is made through that `other` is not
/// made through, or where `other` is made through all of them, the
/// outermost of them.
fn side(loans: &Loans, by: &Loan, other: &Loan) -> Vec<Loan> {
    let own = with_ancestors(loans, by);
    let theirs = with_ancestors(loans, other);
    let apart: LoanSet = own.difference(&theirs).cloned().collect();
    match apart.is_empty() {
        true => outermost(loans, &own),
        false => outermost(loans, &apart),
    }
}

/// The loans of `chain` not made through another of them.
fn outermost(loans: &Loans, chain: &LoanSet) -> Vec<Loan> {
    chain
        .iter()
        .filter(|loan| loans.ancestors(&LoanSet::from([(*loan).clone()])).is_disjoint(chain))
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
