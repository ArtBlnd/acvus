//! Loans: which storage each reference names, and what each instruction does
//! to storage through the references it uses (RFC-0015, RFC-0018).
//!
//! A `Ref` lends a storage slot to its result; a call that takes the
//! reference reads or writes that slot for the call's duration, and a
//! spawned call holds it until its `Eval`. Every pass that orders, moves,
//! removes, or allocates around storage asks here rather than reading the
//! instruction on its own.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::inst_info;
use crate::cfg::CfgBody;
use crate::ir::{Inst, InstKind, RefTarget, ValueId};
use crate::ty::Mutability;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    pub storage: ValueId,
    pub mutability: Mutability,
}

#[derive(Default, Debug)]
pub struct StorageEffect {
    pub reads: SmallVec<[ValueId; 2]>,
    pub writes: SmallVec<[ValueId; 2]>,
}

impl StorageEffect {
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }

    pub fn conflicts(&self, other: &StorageEffect) -> bool {
        let hits = |a: &[ValueId], b: &[ValueId]| a.iter().any(|s| b.contains(s));
        hits(&self.writes, &other.reads)
            || hits(&self.writes, &other.writes)
            || hits(&self.reads, &other.writes)
    }

    fn add(&mut self, loan: Loan) {
        match loan.mutability {
            Mutability::Shared => self.reads.push(loan.storage),
            Mutability::Mut => self.writes.push(loan.storage),
        }
    }
}

pub struct Loans {
    roots: FxHashMap<ValueId, Loan>,
    held_by_spawn: FxHashMap<ValueId, Vec<Loan>>,
}

impl Loans {
    pub fn build(cfg: &CfgBody) -> Self {
        Self::build_from(|| cfg.blocks.iter().flat_map(|b| b.insts.iter()))
    }

    /// Loans of one block's instructions, for a pass that works block-local.
    pub fn build_from_insts(insts: &[Inst]) -> Self {
        Self::build_from(|| insts.iter())
    }

    fn build_from<'a, I>(insts: impl Fn() -> I) -> Self
    where
        I: Iterator<Item = &'a Inst>,
    {
        let raw: FxHashMap<ValueId, (&RefTarget, Mutability)> = insts()
            .filter_map(|inst| match &inst.kind {
                InstKind::Ref {
                    dst,
                    target,
                    mutability,
                    ..
                } => Some((*dst, (target, *mutability))),
                _ => None,
            })
            .collect();
        let roots = raw
            .iter()
            .filter_map(|(&r, &(target, mutability))| {
                storage_of(target, &raw).map(|storage| {
                    (
                        r,
                        Loan {
                            storage,
                            mutability,
                        },
                    )
                })
            })
            .collect();
        let mut loans = Self {
            roots,
            held_by_spawn: FxHashMap::default(),
        };
        loans.held_by_spawn = insts()
            .filter_map(|inst| match &inst.kind {
                InstKind::Spawn { dst, args, .. } => {
                    Some((*dst, args.iter().filter_map(|a| loans.root(*a)).collect()))
                }
                _ => None,
            })
            .collect();
        loans
    }

    /// The storage `reference` names, when it is a reference lent in this
    /// body. A reference that came in as a parameter names no local storage.
    pub fn root(&self, reference: ValueId) -> Option<Loan> {
        self.roots.get(&reference).copied()
    }

    pub fn storage_effect(&self, kind: &InstKind) -> StorageEffect {
        let mut effect = StorageEffect::default();
        match kind {
            InstKind::Ref { target, .. } | InstKind::Take { target, .. } => {
                self.touch(&mut effect, target, Mutability::Shared);
            }
            InstKind::Assign { target, path, .. } => {
                if !path.is_empty() {
                    self.touch(&mut effect, target, Mutability::Shared);
                }
                self.touch(&mut effect, target, Mutability::Mut);
            }
            InstKind::FunctionCall { args, .. } | InstKind::Spawn { args, .. } => {
                args.iter()
                    .filter_map(|a| self.root(*a))
                    .for_each(|loan| effect.add(loan));
            }
            InstKind::Eval { src, .. } => {
                self.held(*src).for_each(|loan| effect.add(loan));
            }
            _ => {}
        }
        effect
    }

    /// The values an instruction uses, plus the storage each used reference
    /// keeps alive: what liveness and register allocation count.
    pub fn uses_with_storage(&self, kind: &InstKind) -> SmallVec<[ValueId; 4]> {
        let uses = inst_info::uses(kind);
        let mut all: SmallVec<[ValueId; 4]> = uses.iter().copied().collect();
        all.extend(uses.iter().filter_map(|u| self.root(*u).map(|loan| loan.storage)));
        let effect = self.storage_effect(kind);
        all.extend(effect.reads.iter().chain(&effect.writes).copied());
        if let InstKind::Assign { target, path, .. } = kind
            && path.is_empty()
        {
            all.retain(|v| Some(*v) != inst_info::storage(target));
        }
        all.sort_unstable();
        all.dedup();
        all
    }

    fn held(&self, handle: ValueId) -> impl Iterator<Item = Loan> + '_ {
        self.held_by_spawn.get(&handle).into_iter().flatten().copied()
    }

    fn touch(&self, effect: &mut StorageEffect, target: &RefTarget, mutability: Mutability) {
        let storage = match target {
            RefTarget::Var(s) | RefTarget::Param(s) => Some(*s),
            RefTarget::Through(r) => self.root(*r).map(|loan| loan.storage),
        };
        if let Some(storage) = storage {
            effect.add(Loan {
                storage,
                mutability,
            });
        }
    }
}

/// The slot a target names once every `Through` is followed to a reference
/// lent in this body; `None` when the chain reaches a reference lent
/// elsewhere (a parameter).
fn storage_of(target: &RefTarget, raw: &FxHashMap<ValueId, (&RefTarget, Mutability)>) -> Option<ValueId> {
    let mut current = target;
    let mut followed = 0;
    loop {
        match current {
            RefTarget::Var(s) | RefTarget::Param(s) => return Some(*s),
            RefTarget::Through(r) => {
                let (next, _) = raw.get(r)?;
                followed += 1;
                assert!(followed <= raw.len(), "a reference is lent through itself: {r:?}");
                current = next;
            }
        }
    }
}
