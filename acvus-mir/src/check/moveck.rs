//! moveck (ownership §2, §6): a value that does not copy is not used after
//! a use moved it, whole or in the part moved; a value a part of which moved
//! out is not used whole again. It reads `Initialization`: at each `Take` or
//! `Ref` whose target is a slot or a by-value parameter's storage, a unit its
//! path overlaps that some path moved out refuses the read `UsedAfterMove`
//! at its span, naming the move. A read one of whose units is only never
//! written is `initck`'s.
//!
//! Storage outside the body is whole whenever the body returns (ownership
//! §4): at each `Return`, a unit of one that some path moved out refuses
//! the return `NotRefilled`, naming the move. A context is given back to
//! what is outside the run by its commit (context §1, §3), the whole `Take`
//! of its slot and the `Commit` of what it took (`Body::committed_slot`):
//! there, a unit of the slot that some path moved out refuses the commit
//! `NotRefilled`, naming the move, and the commit's `Take` is judged as the
//! commit and not as a use. A move through a reference is refused
//! `MoveThroughShared` where a loan of the reference it is made through is
//! shared (ownership §5, what a shared reference lends is never taken
//! from), and `MoveThroughSeveral` where it reaches more than one place, or
//! a place that is not one place (`Loans::is_one`: not exact, or naming a
//! sequence's element, which stands for every element) (ownership §4).

use std::collections::BTreeSet;

use acvus_ast::Span;

use super::{Refusal, RefusalKind};
use crate::ir::cfg::{BodyId, Module, Op, Site, Terminator};
use crate::passes::analysis::{AnalysisManager, Initialization, Loans, Mode, Reach, Storage};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let initialization = am.get::<Initialization>(module, body_id);
    let loans = am.get::<Loans>(module, body_id);
    let body = module.body(body_id);
    for (id, block) in body.blocks() {
        for inst in block.insts() {
            moves_through(&loans, Site::Inst(inst.id()), inst.span(), out);
            if let Some((slot, take)) = body.committed_slot(inst.id()) {
                let taken = taken_before(&initialization, Site::Inst(take), [Storage::Slot(slot)].into_iter());
                out.extend(taken.into_iter().map(|taken| Refusal { kind: RefusalKind::NotRefilled { taken }, span: inst.span() }));
            }
            let (Op::Take { target, path, .. } | Op::Ref { target, path, .. }) = inst.op() else {
                continue;
            };
            let Some(place) = Storage::by_name(target) else {
                continue;
            };
            if body.is_commit_take(inst.id()) {
                continue;
            }
            let read = initialization.before(Site::Inst(inst.id()), place, path);
            let Some(moved) = read.iter().find_map(|(_, fact)| fact.moved.first()) else {
                continue;
            };
            out.push(Refusal { kind: RefusalKind::UsedAfterMove { moved: *moved }, span: inst.span() });
        }
        moves_through(&loans, Site::Term(id), block.terminator_span(), out);
        if let Terminator::Return { span, .. } = block.terminator() {
            let outside = initialization.places().filter(|place| matches!(place, Storage::Outside { .. }));
            let taken = taken_before(&initialization, Site::Term(id), outside);
            out.extend(taken.into_iter().map(|taken| Refusal { kind: RefusalKind::NotRefilled { taken }, span: *span }));
        }
    }
}

/// Each move `site` makes through a reference whose loans include a shared
/// one, and each that reaches several places or one that is not one place.
fn moves_through(loans: &Loans, site: Site, span: Span, out: &mut Vec<Refusal>) {
    for access in loans.accesses(site) {
        if access.reach != Reach::Move || access.through.is_empty() {
            continue;
        }
        if access.through.iter().any(|loan| loan.mode == Mode::Shared) {
            out.push(Refusal { kind: RefusalKind::MoveThroughShared, span });
        }
        if access.places.len() > 1 || !access.places.iter().all(|place| loans.is_one(place)) {
            out.push(Refusal { kind: RefusalKind::MoveThroughSeveral, span });
        }
    }
}

/// Where some path took out a part of one of `places` that `site` finds not
/// written back.
fn taken_before(initialization: &Initialization, site: Site, places: impl Iterator<Item = Storage>) -> BTreeSet<Span> {
    places
        .flat_map(|place| initialization.before(site, place, &[]).into_iter().flat_map(|(_, fact)| fact.moved.iter().copied()).collect::<Vec<_>>())
        .collect()
}
