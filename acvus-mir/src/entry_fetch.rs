//! The entry fetches of a body (RFC-0025 rule 2): a body does not fetch a
//! context that every path assigns whole before touching it. A call whose
//! summary names the context touches it, through the `Commit` that brackets
//! the call, and a callee's assignment is not the body's. Decided as
//! definite assignment over the body's CFG.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{BlockIdx, CfgBody, promote};
use crate::graph::QualifiedRef;
use crate::ir::{InstKind, MirBody, RefTarget, ValueId};
use crate::ty::Mutability;

struct EntryFetch {
    context: QualifiedRef,
    slot: ValueId,
    fetched: ValueId,
}

/// The page operations lowering emitted for one body.
#[derive(Default)]
pub(crate) struct PageOps {
    entry: Vec<EntryFetch>,
    /// The registers the `Fetch` after a call writes. That store re-reads the
    /// page; the source did not write it, so it assigns nothing here.
    refetched: FxHashSet<ValueId>,
    /// The registers a place taken out for a shared lend is put back from:
    /// the value it held, so the store writes nothing new.
    handed_back: FxHashSet<ValueId>,
}

impl PageOps {
    pub(crate) fn entry(&mut self, context: QualifiedRef, slot: ValueId, fetched: ValueId) {
        self.entry.push(EntryFetch {
            context,
            slot,
            fetched,
        });
    }

    pub(crate) fn refetched(&mut self, fetched: ValueId) {
        self.refetched.insert(fetched);
    }

    pub(crate) fn handed_back(&mut self, restored: ValueId) {
        self.handed_back.insert(restored);
    }
}

/// Set each `Commit`'s `wrote`: whether the body may have written its
/// variable since the variable's last fetch (RFC-0025 rules 2, 4). A write is
/// an assignment the source wrote, a store into a part, or a `&mut` lend; a
/// `Fetch` into the variable, at entry or after a call, starts it over. A
/// call that writes the context is bracketed, so no call between a fetch and
/// the commit after it writes the variable. Decided as a forward may-analysis
/// over the body's CFG, read at each commit.
pub(crate) fn decide_writes(body: &mut MirBody, ops: &PageOps) {
    if ops.entry.is_empty() {
        return;
    }
    let at_of_slot: FxHashMap<ValueId, usize> = ops
        .entry
        .iter()
        .enumerate()
        .map(|(at, entry)| (entry.slot, at))
        .collect();
    let at_of_context: FxHashMap<QualifiedRef, usize> = ops
        .entry
        .iter()
        .enumerate()
        .map(|(at, entry)| (entry.context, at))
        .collect();
    let fetched: FxHashSet<ValueId> = ops.entry.iter().map(|entry| entry.fetched).collect();
    let cfg = promote(body.clone());
    let n = ops.entry.len();

    // The state after `inst`, from the state before it; each commit's
    // `wrote` goes to `commits`.
    let step = |inst: &crate::ir::Inst, written: &mut [bool], commits: &mut FxHashMap<ValueId, bool>| {
        match &inst.kind {
            InstKind::Assign {
                target: RefTarget::Var(slot),
                path,
                value,
                ..
            } => {
                let Some(&at) = at_of_slot.get(slot) else { return };
                if !path.is_empty() {
                    written[at] = true;
                } else if fetched.contains(value) || ops.refetched.contains(value) {
                    written[at] = false;
                } else if !ops.handed_back.contains(value) {
                    written[at] = true;
                }
            }
            InstKind::Ref {
                target: RefTarget::Var(slot),
                mutability: Mutability::Mut,
                ..
            } => {
                if let Some(&at) = at_of_slot.get(slot) {
                    written[at] = true;
                }
            }
            InstKind::Commit { context, value, .. } => {
                let wrote = at_of_context.get(context).is_none_or(|&at| written[at]);
                commits.insert(*value, wrote);
            }
            _ => {}
        }
    };

    let mut commits: FxHashMap<ValueId, bool> = FxHashMap::default();
    forward_or(&cfg, vec![false; n], |at, written| {
        for inst in &cfg.blocks[at.0].insts {
            step(inst, written, &mut commits);
        }
    });
    // A commit no path reaches never runs; a store there loses no write.
    for inst in &mut body.insts {
        if let InstKind::Commit { value, wrote, .. } = &mut inst.kind {
            *wrote = commits.get(value).copied().unwrap_or(true);
        }
    }
}

/// Remove the entry fetch of every context `body` assigns whole on every
/// path before touching it.
pub(crate) fn skip_assigned_fetches(body: &mut MirBody, ops: PageOps) {
    if ops.entry.is_empty() {
        return;
    }
    let touched = touched_unset(&promote(body.clone()), &ops);
    let skipped: FxHashSet<ValueId> = ops
        .entry
        .iter()
        .zip(touched)
        .filter(|(_, touched)| !touched)
        .map(|(entry, _)| entry.fetched)
        .collect();
    body.insts.retain(|inst| match &inst.kind {
        InstKind::Fetch { dst, .. } => !skipped.contains(dst),
        InstKind::Assign {
            target: RefTarget::Var(_),
            path,
            value,
            ..
        } => !(path.is_empty() && skipped.contains(value)),
        _ => true,
    });
    for fetched in &skipped {
        body.val_types.remove(fetched);
    }
}

/// Per entry fetch, in order: whether the body touches the variable where it
/// may still be unassigned. A `Take` touches it, whether it reads the value,
/// moves it, or commits it at an exit or around a call; so do a `Ref` and a
/// store into a part. A block no path reaches touches nothing.
fn touched_unset(cfg: &CfgBody, ops: &PageOps) -> Vec<bool> {
    let entry_of_slot: FxHashMap<ValueId, usize> = ops
        .entry
        .iter()
        .enumerate()
        .map(|(at, entry)| (entry.slot, at))
        .collect();
    let entry_fetched: FxHashSet<ValueId> = ops.entry.iter().map(|e| e.fetched).collect();
    let tracked = |target: &RefTarget| match target {
        RefTarget::Var(slot) => entry_of_slot.get(slot).copied(),
        RefTarget::Param(_) | RefTarget::Through(_) => None,
    };
    let n = ops.entry.len();
    let mut touched = vec![false; n];

    forward_or(cfg, vec![true; n], |at, maybe_unset| {
    for inst in &cfg.blocks[at.0].insts {
        match &inst.kind {
            InstKind::Assign {
                target,
                path,
                value,
                ..
            } => {
                let Some(i) = tracked(target) else { continue };
                let written_by_source =
                    !entry_fetched.contains(value) && !ops.refetched.contains(value);
                if !path.is_empty() {
                    touched[i] |= maybe_unset[i];
                } else if written_by_source {
                    maybe_unset[i] = false;
                }
            }
            InstKind::Take { target, .. } | InstKind::Ref { target, .. } => {
                let Some(i) = tracked(target) else { continue };
                touched[i] |= maybe_unset[i];
            }
            _ => {}
        }
    }
    });
    touched
}

/// A forward may-analysis over `cfg` whose state is one flag per tracked
/// variable, joined by `or`: `transfer` turns the state at a block's entry
/// into the state at its exit, and runs again on a block whenever its entry
/// state grows. `entry` is the state at the body's first block.
fn forward_or<F>(cfg: &CfgBody, entry: Vec<bool>, mut transfer: F)
where
    F: FnMut(BlockIdx, &mut Vec<bool>),
{
    let bottom = vec![false; entry.len()];
    let mut block_in: Vec<Vec<bool>> = vec![bottom; cfg.blocks.len()];
    let mut reached = vec![false; cfg.blocks.len()];
    block_in[0] = entry;
    reached[0] = true;
    let mut worklist = vec![BlockIdx(0)];
    while let Some(at) = worklist.pop() {
        let mut state = block_in[at.0].clone();
        transfer(at, &mut state);
        for succ in cfg.successors(at) {
            let mut changed = !std::mem::replace(&mut reached[succ.0], true);
            for (into, from) in block_in[succ.0].iter_mut().zip(&state) {
                if *from && !*into {
                    *into = true;
                    changed = true;
                }
            }
            if changed {
                worklist.push(succ);
            }
        }
    }
}
