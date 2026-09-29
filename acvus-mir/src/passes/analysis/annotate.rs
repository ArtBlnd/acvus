//! Annotations of the IR printer (`ir::cfg::print::Annotate`) from the
//! analyses: each says, in notation, what its analysis holds at a point.
//! - `LiveAnnotations`, key `live`: the values `Liveness` holds live at a
//!   block's start, `{v1, v2}`.
//! - `InitAnnotations`, key `init`: before each instruction, each unit of
//!   each place `Initialization` tracks, a storage outside the body as
//!   `*p0#n` (`LoanAnnotations`'), `slot0.a=W` written on every path
//!   that reaches, `N` left unwritten by some path, `M@start..end` moved out
//!   at that span by some path, several joined by `|`.
//! - `LoanAnnotations`, key `loans`: each value defined by the instruction
//!   and holding loans, `v2=[slot0.b mut @25..33]`, a loan by its place, its
//!   mode and where it was made, `~` after the path where it is not exact.
//!
//! Every set prints sorted, values in the order their names print; a value
//! and a block by the print's `Names`, a path and a span as the printer
//! writes them.

use acvus_utils::Interner;

use crate::ir::ValueId;
use crate::ir::cfg::print::{self, Annotate, Names};
use crate::ir::cfg::{BlockId, Body, InstId, Site};
use crate::passes::analysis::{Initialization, Liveness, Loan, Loans, Made, Mode, Storage, Written};

/// `live`: the values live at a block's start.
pub struct LiveAnnotations<'a> {
    liveness: &'a Liveness,
}

impl<'a> LiveAnnotations<'a> {
    pub fn new(liveness: &'a Liveness) -> Self {
        LiveAnnotations { liveness }
    }
}

impl Annotate for LiveAnnotations<'_> {
    fn block(&self, names: &Names<'_>, block: BlockId) -> Vec<(&'static str, String)> {
        let mut live: Vec<ValueId> = self.liveness.live_at_start(block).iter().copied().collect();
        names.sort_values(&mut live);
        let live: Vec<String> = live.iter().map(|value| names.value(*value)).collect();
        vec![("live", format!("{{{}}}", live.join(", ")))]
    }

    fn before(&self, _: &Names<'_>, _: InstId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn edge(&self, _: &Names<'_>, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// `init`: what each unit of each place holds before an instruction.
pub struct InitAnnotations<'a> {
    interner: &'a Interner,
    initialization: &'a Initialization,
}

impl<'a> InitAnnotations<'a> {
    pub fn new(interner: &'a Interner, initialization: &'a Initialization) -> Self {
        InitAnnotations { interner, initialization }
    }

    /// Every place the analysis tracks, in the order their names print,
    /// each outside storage after its parameter's or capture's own.
    fn places(&self, names: &Names<'_>) -> Vec<Storage> {
        let mut places: Vec<Storage> = self.initialization.places().collect();
        let mut roots: Vec<ValueId> = places.iter().map(|place| root(*place)).collect();
        names.sort_values(&mut roots);
        let order = |place: &Storage| roots.iter().position(|value| *value == root(*place));
        places.sort_by_key(|place| (order(place), matches!(place, Storage::Outside { .. }), outside_position(*place)));
        places
    }
}

/// The value a place is the storage of, or lent by.
fn root(place: Storage) -> ValueId {
    match place {
        Storage::Slot(value) | Storage::Param(value) | Storage::Outside { of: value, .. } => value,
    }
}

fn outside_position(place: Storage) -> usize {
    match place {
        Storage::Outside { position, .. } => position,
        Storage::Slot(_) | Storage::Param(_) => 0,
    }
}

/// `W`, or `N` and each `M@start..end`, joined by `|`.
fn written(written: &Written) -> String {
    let mut parts: Vec<String> = Vec::new();
    if written.never {
        parts.push("N".to_string());
    }
    parts.extend(written.moved.iter().map(|span| format!("M{}", print::span(*span))));
    if parts.is_empty() { "W".to_string() } else { parts.join("|") }
}

impl Annotate for InitAnnotations<'_> {
    fn block(&self, _: &Names<'_>, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn before(&self, names: &Names<'_>, inst: InstId) -> Vec<(&'static str, String)> {
        let mut units: Vec<String> = Vec::new();
        for place in self.places(names) {
            let root = storage(names, place);
            let mut held: Vec<String> = self
                .initialization
                .before(Site::Inst(inst), place, &[])
                .into_iter()
                .map(|(unit, holds)| format!("{root}{}={}", print::path(self.interner, unit), written(holds)))
                .collect();
            held.sort();
            units.extend(held);
        }
        if units.is_empty() {
            return Vec::new();
        }
        vec![("init", units.join(" "))]
    }

    fn edge(&self, _: &Names<'_>, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// `loans`: the loans each value an instruction defines holds.
pub struct LoanAnnotations<'a> {
    body: &'a Body,
    interner: &'a Interner,
    loans: &'a Loans,
}

impl<'a> LoanAnnotations<'a> {
    pub fn new(body: &'a Body, interner: &'a Interner, loans: &'a Loans) -> Self {
        LoanAnnotations { body, interner, loans }
    }

    /// `slot0.b~ mut @25..33`.
    fn loan(&self, names: &Names<'_>, loan: &Loan) -> String {
        let mut text = storage(names, loan.storage);
        text.push_str(&print::path(self.interner, &loan.path));
        if !loan.exact {
            text.push('~');
        }
        if loan.mode == Mode::Exclusive {
            text.push_str(" mut");
        }
        let made = match loan.made {
            Made::At(Site::Inst(inst)) => self.body.inst(inst).span(),
            Made::At(Site::Term(block)) => self.body.block(block).terminator_span(),
            Made::Outside { of } => self.body.binder_span(of),
        };
        text.push(' ');
        text.push_str(&print::span(made));
        text
    }

}

/// A slot or a parameter's own storage by its value, `slot0`; what position
/// `n` of a parameter or capture names outside the body, `*p0#n`.
fn storage(names: &Names<'_>, storage: Storage) -> String {
    match storage {
        Storage::Slot(value) | Storage::Param(value) => names.value(value),
        Storage::Outside { of, position } => format!("*{}#{position}", names.value(of)),
    }
}

impl Annotate for LoanAnnotations<'_> {
    fn block(&self, _: &Names<'_>, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn before(&self, names: &Names<'_>, inst: InstId) -> Vec<(&'static str, String)> {
        let Some(result) = self.body.inst(inst).result() else {
            return Vec::new();
        };
        let held = self.loans.all_of(result);
        if held.is_empty() {
            return Vec::new();
        }
        let mut loans: Vec<String> = held.iter().map(|loan| self.loan(names, loan)).collect();
        loans.sort();
        vec![("loans", format!("{}=[{}]", names.value(result), loans.join(", ")))]
    }

    fn edge(&self, _: &Names<'_>, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}
