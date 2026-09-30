//! moveck (ownership §2, §6): a value that does not copy is not used after
//! a use moved it, whole or in the part moved; a value a part of which moved
//! out is not used whole again. It reads `Initialization`: at each `Take` or
//! `Ref` whose target is a place by name, a slot, a by-value parameter's
//! storage or a context, a unit its path overlaps that some path moved out
//! refuses the read `UsedAfterMove` at its span, naming the move. A read one of whose units is only never
//! written is `initck`'s.
//!
//! Storage outside the body is whole whenever the body returns (ownership
//! §4): at each `Return`, a unit of one that some path moved out refuses
//! the return `NotRefilled`, naming the move. A context is whole wherever
//! the page is seen beyond the program (context §1): at each site `Loans`
//! records seeing it, a call that may touch it or a `Return`, a unit of it
//! that some path moved out refuses the site `NotRefilled`, naming the
//! move. A move through a reference is refused
//! `MoveThroughShared` where a loan of the reference it is made through is
//! shared (ownership §5, what a shared reference lends is never taken
//! from), and `MoveThroughSeveral` where it reaches more than one place, or
//! a place that is not one place (`Loans::is_one`: not exact, or naming a
//! sequence's element, which stands for every element) (ownership §4).
//!
//! A store whose old content the body ends at the store
//! (`lower::required::OverwriteEnd::AtStore`: a context, storage outside the
//! body, or places a reference may reach), of a value that does not copy,
//! must be able to end it (ownership §6). It reads `ContentLiveness`: a
//! store through a reference that may reach several places, where what it
//! replaces may be absent at one of them on some path (R3), and a store
//! into one place, a part of which is present on some paths only and last
//! known present on an edge where no root names that place
//! (`lower::required::rooted_place`, R2), is refused
//! `StoreOverContentMaybeMoved` at its span, naming the move; the program
//! makes the paths agree. Where no path moved the content, it is only never
//! written, which `initck` judges where the place is lent.

use std::collections::BTreeSet;

use acvus_ast::Span;

use super::{Refusal, RefusalKind};
use crate::ir::cfg::{Body, BodyId, Inst, Module, Op, Site, Terminator};
use crate::ir::parts::{self, Path};
use crate::lower::required::{OverwriteEnd, absent_among, edge_drops, overwrite_end};
use crate::passes::analysis::{AnalysisManager, ContentLiveness, Initialization, Loans, Mode, Reach, Storage, Stored};
use crate::ty::{Concrete, Types, copies};

/// The judgment over `body`, its refusals pushed to `out`.
pub(super) fn check(module: &Module, body_id: BodyId, am: &mut AnalysisManager<'_>, out: &mut Vec<Refusal>) {
    let initialization = am.get::<Initialization>(module, body_id);
    let loans = am.get::<Loans>(module, body_id);
    let content = am.get::<ContentLiveness>(module, body_id);
    let body = module.body(body_id);
    for (id, block) in body.blocks() {
        for inst in block.insts() {
            let site = Site::Inst(inst.id());
            moves_through(&loans, site, inst.span(), out);
            let taken = taken_before(&initialization, site, loans.page_seen(site).map(Storage::Context));
            out.extend(taken.into_iter().map(|taken| Refusal { kind: RefusalKind::NotRefilled { taken }, span: inst.span() }));
            if let Some(moved) = stored_over_content_maybe_moved(module.types(), body, &initialization, &loans, &content, inst) {
                out.push(Refusal { kind: RefusalKind::StoreOverContentMaybeMoved { moved }, span: inst.span() });
            }
            let (Op::Take { target, path, .. } | Op::Ref { target, path, .. }) = inst.op() else {
                continue;
            };
            let Some(place) = Storage::by_name(target) else {
                continue;
            };
            let read = initialization.before(Site::Inst(inst.id()), place, path);
            let Some(moved) = read.iter().find_map(|(_, fact)| fact.moved.first()) else {
                continue;
            };
            out.push(Refusal { kind: RefusalKind::UsedAfterMove { moved: *moved }, span: inst.span() });
        }
        let site = Site::Term(id);
        moves_through(&loans, site, block.terminator_span(), out);
        if let Terminator::Return { span, .. } = block.terminator() {
            let outside = initialization.places().filter(|place| matches!(place, Storage::Outside { .. }));
            let seen = loans.page_seen(site).map(Storage::Context);
            let taken = taken_before(&initialization, site, outside.chain(seen));
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

/// Where a path moved out content that `inst`, a store that drops what it
/// overwrites, of a value that does not copy, replaces where it cannot end
/// it: among several places a reference may reach, one where it may be
/// absent (R3); in one place, a part present on some paths only on an edge
/// where no root names it (R2).
fn stored_over_content_maybe_moved(
    types: &Types<Concrete>,
    body: &Body,
    initialization: &Initialization,
    loans: &Loans,
    content: &ContentLiveness,
    inst: &Inst,
) -> Option<Span> {
    let Some(OverwriteEnd::AtStore(stored)) = overwrite_end(loans, inst) else {
        return None;
    };
    let Op::Assign { value, .. } = inst.op() else {
        unreachable!("an overwrite is a store");
    };
    let ty = body.ty_of(*value);
    if copies(types, ty) {
        return None;
    }
    let unended: Vec<(Storage, Path)> = match stored {
        Stored::One { place, path } => edge_drops(body, types, loans, content, inst, (place, path), ty)
            .into_iter()
            .filter(|drop| drop.name.is_none())
            .map(|drop| (place, drop.unit))
            .collect(),
        Stored::Several(places) => absent_among(content, inst, &places).map(|(place, path)| (place, path.to_vec())).collect(),
    };
    let site = Site::Inst(inst.id());
    unended.into_iter().find_map(|(place, path)| {
        let before = initialization.before(site, place, parts::owned_region(&path));
        before.iter().find_map(|(_, fact)| fact.moved.first().copied())
    })
}

/// Where some path took out a part of one of `places` that `site` finds not
/// written back.
fn taken_before(initialization: &Initialization, site: Site, places: impl Iterator<Item = Storage>) -> BTreeSet<Span> {
    places
        .flat_map(|place| initialization.before(site, place, &[]).into_iter().flat_map(|(_, fact)| fact.moved.iter().copied()).collect::<Vec<_>>())
        .collect()
}
