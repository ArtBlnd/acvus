//! The storage a `for` changes (RFC-0089 rule 2), which `analysis::loop_deps`
//! reads as tokens.

use rustc_hash::FxHashSet;

use crate::analysis::inst_info;
use crate::analysis::loans::{Loans, StorageEffect};
use crate::cfg::{BlockIdx, CfgBody};
use crate::ir::{ForSource, InstKind, RefTarget, ValueId};
use crate::ty::Mutability;

/// What a write to a slot live at the header changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    Storage(ValueId),
    /// The element of a `&mut` source, whose writes land at the counter's
    /// slot.
    Element,
}

pub struct TargetSlots {
    element: Vec<ValueId>,
    live: FxHashSet<ValueId>,
}

impl TargetSlots {
    /// A slot is live at the header when an instruction outside the loop's
    /// blocks touches it or it is the body's parameter or capture. That is
    /// the decision this makes in place of a liveness query: a slot the body
    /// defines and releases within an iteration is touched only inside.
    pub fn of(loans: &Loans<'_>, source: Option<ForSource>, loop_blocks: &[BlockIdx]) -> Self {
        let cfg = loans.cfg();
        let mut live: FxHashSet<ValueId> = cfg
            .params
            .iter()
            .chain(&cfg.captures)
            .map(|(_, slot)| *slot)
            .collect();
        for (at, block) in cfg.blocks.iter().enumerate() {
            if loop_blocks.contains(&BlockIdx(at)) {
                continue;
            }
            for inst in &block.insts {
                live.extend(touched_slots(loans, &inst.kind));
            }
        }
        let element = match source {
            Some(ForSource::SliceMut(slice)) => loans
                .names(slice)
                .iter()
                .filter(|loan| loan.mutability == Mutability::Mut)
                .filter_map(|loan| loan.storage.slot())
                .collect(),
            Some(ForSource::Slice(_) | ForSource::Array(_) | ForSource::Range { .. }) | None => {
                Vec::new()
            }
        };
        Self { element, live }
    }

    /// The target a write to `slot` changes, or `None` for a slot the body
    /// defines and releases within one iteration.
    pub fn target_of(&self, slot: ValueId) -> Option<Written> {
        if self.element.contains(&slot) {
            return Some(Written::Element);
        }
        self.live.contains(&slot).then_some(Written::Storage(slot))
    }
}

pub fn touched_slots(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
    let named = match kind {
        InstKind::Ref { target, .. }
        | InstKind::Take { target, .. }
        | InstKind::Assign { target, .. } => inst_info::storage(target),
        _ => None,
    };
    let effect = loans.storage_effect(kind);
    named
        .into_iter()
        .chain(effect.reads)
        .chain(effect.writes)
        .collect()
}

/// What an instruction does to storage. A `Take` that copies a word out of
/// its place reads it: `Loans::storage_effect` counts every take as a write,
/// since a take may empty its slot, and a word's leaves it holding the word.
pub fn effect(loans: &Loans<'_>, kind: &InstKind) -> StorageEffect {
    let mut effect = loans.storage_effect(kind);
    if copies_a_word(loans.cfg(), kind) {
        let written = std::mem::take(&mut effect.writes);
        effect.reads.extend(written);
    }
    effect
}

fn copies_a_word(cfg: &CfgBody, kind: &InstKind) -> bool {
    let InstKind::Take {
        dst,
        taken_out: false,
        ..
    } = kind
    else {
        return false;
    };
    cfg.val_types.get(dst).and_then(|ty| ty.is_word()) == Some(true)
}

pub fn slots_lent_mutably(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
    let InstKind::Ref {
        target,
        mutability: Mutability::Mut,
        ..
    } = kind
    else {
        return Vec::new();
    };
    match target {
        RefTarget::Var(slot) | RefTarget::Param(slot) => vec![*slot],
        RefTarget::Through(reference) => loans
            .holds(*reference)
            .filter_map(|loan| loan.storage.slot())
            .collect(),
    }
}
