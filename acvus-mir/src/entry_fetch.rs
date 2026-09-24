//! The entry fetches of a body (RFC-0025 rule 2): a body does not fetch a
//! context that every path assigns whole before touching it. A call whose
//! summary names the context touches it, through the `Commit` that brackets
//! the call, and a callee's assignment is not the body's. Decided as
//! definite assignment over the body's CFG.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::cfg::{BlockIdx, CfgBody, promote};
use crate::graph::QualifiedRef;
use crate::ir::{InstKind, MirBody, RefTarget, ValueId};

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
}

/// Remove the entry fetch of every context `body` assigns whole on every
/// path before touching it; the contexts it still fetches, in the order
/// lowering fetched them.
pub(crate) fn skip_assigned_fetches(body: &mut MirBody, ops: PageOps) -> Vec<QualifiedRef> {
    if ops.entry.is_empty() {
        return Vec::new();
    }
    let touched = touched_unset(&promote(body.clone()), &ops);

    let mut still_fetched = Vec::new();
    let mut skipped: FxHashSet<ValueId> = FxHashSet::default();
    for (entry, touched) in ops.entry.iter().zip(touched) {
        if touched {
            still_fetched.push(entry.context);
        } else {
            skipped.insert(entry.fetched);
        }
    }
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
    still_fetched
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

    let mut block_in: Vec<Option<Vec<bool>>> = vec![None; cfg.blocks.len()];
    block_in[0] = Some(vec![true; n]);
    let mut worklist = vec![BlockIdx(0)];
    while let Some(at) = worklist.pop() {
        let mut maybe_unset = block_in[at.0]
            .clone()
            .expect("a block on the worklist has an entry state");
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
        for succ in cfg.successors(at) {
            let changed = match &mut block_in[succ.0] {
                unreached @ None => {
                    *unreached = Some(maybe_unset.clone());
                    true
                }
                Some(into) => {
                    let mut changed = false;
                    for (into, from) in into.iter_mut().zip(&maybe_unset) {
                        if *from && !*into {
                            *into = true;
                            changed = true;
                        }
                    }
                    changed
                }
            };
            if changed {
                worklist.push(succ);
            }
        }
    }
    touched
}
