//! Annotations of the IR printer (`ir::cfg::print::Annotate`) from the
//! analyses: each says, in notation, what its analysis holds at a point.
//! - `LiveAnnotations`, key `live`: the values `Liveness` holds live at a
//!   block's start, `{v1, v2}`.
//! - `InitAnnotations`, key `init`: before each instruction, each unit of
//!   each place `Initialization` tracks, `slot0.a=W` written on every path
//!   that reaches, `N` left unwritten by some path, `M@start..end` moved out
//!   at that span by some path, several joined by `|`.
//! - `LoanAnnotations`, key `loans`: each value defined by the instruction
//!   and holding loans, `v2=[slot0.b mut @25..33]`, a loan by its place, its
//!   mode and where it was made, `~` after the path where it is not exact.
//!
//! Every set prints sorted; a value, a place and a span as the printer
//! prints them.

use acvus_utils::Interner;

use crate::ir::ValueId;
use crate::ir::cfg::print::{self, Annotate};
use crate::ir::cfg::{BlockId, Body, Definition, InstId, Site};
use crate::passes::analysis::{Initialization, Liveness, Loan, Loans, Made, Mode, Storage, Written};

/// Where a value sorts among the values printed together: parameters,
/// captures, slots, tokens, then every other value, each kind by its
/// number.
fn value_rank(body: &Body, value: ValueId) -> (u8, usize) {
    use acvus_utils::LocalIdOps;
    match body.definition(value) {
        Definition::Param(index) => (0, index),
        Definition::Capture(index) => (1, index),
        Definition::Slot(index) => (2, index),
        Definition::OrderParam | Definition::BlockParam(..) | Definition::Inst(_) => {
            (if body.is_token(value) { 3 } else { 4 }, value.to_raw())
        }
    }
}

/// `live`: the values live at a block's start.
pub struct LiveAnnotations<'a> {
    body: &'a Body,
    liveness: &'a Liveness,
}

impl<'a> LiveAnnotations<'a> {
    pub fn new(body: &'a Body, liveness: &'a Liveness) -> Self {
        LiveAnnotations { body, liveness }
    }
}

impl Annotate for LiveAnnotations<'_> {
    fn block(&self, block: BlockId) -> Vec<(&'static str, String)> {
        let mut live: Vec<ValueId> = self.liveness.live_at_start(block).iter().copied().collect();
        live.sort_by_key(|value| value_rank(self.body, *value));
        let live: Vec<String> = live.iter().map(|value| print::value(self.body, *value)).collect();
        vec![("live", format!("{{{}}}", live.join(", ")))]
    }

    fn before(&self, _: InstId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn edge(&self, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// `init`: what each unit of each place holds before an instruction.
pub struct InitAnnotations<'a> {
    body: &'a Body,
    names: &'a Interner,
    initialization: &'a Initialization,
}

impl<'a> InitAnnotations<'a> {
    pub fn new(body: &'a Body, names: &'a Interner, initialization: &'a Initialization) -> Self {
        InitAnnotations { body, names, initialization }
    }

    /// Every place the analysis tracks, sorted.
    fn places(&self) -> Vec<ValueId> {
        let mut places: Vec<ValueId> = self
            .body
            .params()
            .iter()
            .chain(self.body.captures())
            .map(|(_, value)| *value)
            .chain(self.body.slots().iter().copied())
            .filter(|place| self.initialization.units(*place).is_some())
            .collect();
        places.sort_by_key(|place| value_rank(self.body, *place));
        places
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
    fn block(&self, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn before(&self, inst: InstId) -> Vec<(&'static str, String)> {
        let mut units: Vec<String> = Vec::new();
        for place in self.places() {
            let root = print::value(self.body, place);
            let mut held: Vec<String> = self
                .initialization
                .before(inst, place, &[])
                .into_iter()
                .map(|(unit, holds)| format!("{root}{}={}", print::path(self.names, unit), written(holds)))
                .collect();
            held.sort();
            units.extend(held);
        }
        if units.is_empty() {
            return Vec::new();
        }
        vec![("init", units.join(" "))]
    }

    fn edge(&self, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// `loans`: the loans each value an instruction defines holds.
pub struct LoanAnnotations<'a> {
    body: &'a Body,
    names: &'a Interner,
    loans: &'a Loans,
}

impl<'a> LoanAnnotations<'a> {
    pub fn new(body: &'a Body, names: &'a Interner, loans: &'a Loans) -> Self {
        LoanAnnotations { body, names, loans }
    }

    /// `slot0.b~ mut @25..33`.
    fn loan(&self, loan: &Loan) -> String {
        let mut text = self.storage(loan.storage);
        text.push_str(&print::path(self.names, &loan.path));
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

    /// A slot or a parameter's own storage by its value, `slot0`; what
    /// position `n` of a parameter or capture names outside the body,
    /// `*p0#n`.
    fn storage(&self, storage: Storage) -> String {
        match storage {
            Storage::Slot(value) | Storage::Param(value) => print::value(self.body, value),
            Storage::Outside { of, position } => format!("*{}#{position}", print::value(self.body, of)),
        }
    }
}

impl Annotate for LoanAnnotations<'_> {
    fn block(&self, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    fn before(&self, inst: InstId) -> Vec<(&'static str, String)> {
        let Some(result) = self.body.inst(inst).result() else {
            return Vec::new();
        };
        let held = self.loans.all_of(result);
        if held.is_empty() {
            return Vec::new();
        }
        let mut loans: Vec<String> = held.iter().map(|loan| self.loan(loan)).collect();
        loans.sort();
        vec![("loans", format!("{}=[{}]", print::value(self.body, result), loans.join(", ")))]
    }

    fn edge(&self, _: BlockId, _: BlockId) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}
