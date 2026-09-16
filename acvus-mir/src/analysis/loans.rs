//! Regions: the storage a value may name, as a trivial lifetime (RFC-0015,
//! RFC-0018). A region is a set of loans; join is union; bottom names
//! nothing. A `Ref` starts a region at its slot, a parameter or capture of
//! reference type starts one at itself (RFC-0029), and a value whose type
//! contains a reference takes the join of the regions it is built from:
//! a call's result from its arguments, a closure from its captures, a
//! block parameter from the jump arguments that reach it, a slot from
//! what is assigned into it, a spawn's handle from its arguments whatever
//! its type says. A call that takes a value reads or writes that value's
//! region for the call's duration, and a spawned call holds it until its
//! `Eval`. Every pass that orders, moves, removes, or allocates around
//! storage asks here rather than reading the instruction on its own.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::dataflow::{DataflowAnalysis, DataflowState, forward_analysis};
use crate::analysis::domain::SemiLattice;
use crate::analysis::inst_info;
use crate::cfg::CfgBody;
use crate::ir::{Inst, InstKind, RefTarget, ValueId};
use crate::ty::{Mutability, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    pub storage: ValueId,
    pub mutability: Mutability,
}

/// The loans a value holds, and the references it was built through: a
/// touch through one of those is the value's own access.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub loans: Vec<Loan>,
    pub via: Vec<ValueId>,
}

impl SemiLattice for Region {
    fn bottom() -> Self {
        Self::default()
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for loan in &other.loans {
            if !self.loans.contains(loan) {
                self.loans.push(*loan);
                changed = true;
            }
        }
        for v in &other.via {
            if !self.via.contains(v) {
                self.via.push(*v);
                changed = true;
            }
        }
        changed
    }
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

// -- The dataflow ---------------------------------------------------

struct RegionAnalysis<'a> {
    val_types: &'a FxHashMap<ValueId, Ty>,
}

/// A value with no type entry is taken to carry a reference: the stricter
/// reading, so a missing type never hides a loan. The pipeline types every
/// defined value; only a hand-built body lacks one.
const UNTYPED_CARRIES_REFERENCE: bool = true;

impl RegionAnalysis<'_> {
    fn carries_ref(&self, v: ValueId) -> bool {
        self.val_types
            .get(&v)
            .map_or(UNTYPED_CARRIES_REFERENCE, contains_ref)
    }

    /// `to ⊒ from`, when `to` can hold a reference or `always` says so.
    fn flow(
        &self,
        state: &mut DataflowState<ValueId, Region>,
        from: ValueId,
        to: ValueId,
        always: bool,
    ) {
        if !always && !self.carries_ref(to) {
            return;
        }
        let source = state.get(from);
        let mut target = state.get(to);
        if target.join_mut(&source) {
            state.set(to, target);
        }
    }
}

impl DataflowAnalysis for RegionAnalysis<'_> {
    type Key = ValueId;
    type Domain = Region;

    fn transfer_inst(&self, inst: &Inst, state: &mut DataflowState<ValueId, Region>) {
        match &inst.kind {
            InstKind::Ref {
                dst,
                target,
                mutability,
                ..
            } => match target {
                RefTarget::Var(s) | RefTarget::Param(s) => state.set(
                    *dst,
                    Region {
                        loans: vec![Loan {
                            storage: *s,
                            mutability: *mutability,
                        }],
                        via: vec![],
                    },
                ),
                RefTarget::Through(r) => {
                    let mut region = state.get(*r);
                    region.via.push(*r);
                    state.set(*dst, region);
                }
            },
            // A reference read out of a storage is the storage's own
            // reference, not a second holder (RFC-0029).
            InstKind::Take { dst, target, .. } => {
                if let Some(slot) = inst_info::storage(target)
                    && self.carries_ref(*dst)
                {
                    let mut region = state.get(slot);
                    region.via.push(slot);
                    let mut target = state.get(*dst);
                    if target.join_mut(&region) {
                        state.set(*dst, target);
                    }
                }
            }
            InstKind::Assign { target, value, .. } => {
                if let Some(slot) = inst_info::storage(target) {
                    self.flow(state, *value, slot, false);
                }
            }
            InstKind::Spawn { dst, args, .. } => {
                for a in args {
                    self.flow(state, *a, *dst, true);
                }
            }
            kind => {
                for dst in inst_info::defs(kind) {
                    for u in inst_info::uses(kind) {
                        self.flow(state, u, dst, false);
                    }
                }
            }
        }
    }

    fn propagate_forward(
        &self,
        source_exit: &DataflowState<ValueId, Region>,
        params: &[ValueId],
        args: &[ValueId],
        target_entry: &mut DataflowState<ValueId, Region>,
    ) -> bool {
        let mut changed = target_entry.join_from(source_exit);
        for (param, arg) in params.iter().zip(args) {
            if !self.carries_ref(*param) {
                continue;
            }
            let mut region = target_entry.get(*param);
            if region.join_mut(&source_exit.get(*arg)) {
                target_entry.set(*param, region);
                changed = true;
            }
        }
        changed
    }

    fn propagate_backward(
        &self,
        _: &DataflowState<ValueId, Region>,
        _: &[ValueId],
        _: &[ValueId],
        _: &mut DataflowState<ValueId, Region>,
    ) {
        unreachable!("regions flow forward")
    }
}

// -- The result -----------------------------------------------------

pub struct Loans {
    region: FxHashMap<ValueId, Region>,
}

static NOTHING: Region = Region {
    loans: Vec::new(),
    via: Vec::new(),
};

impl Loans {
    pub fn build(cfg: &CfgBody) -> Self {
        let analysis = RegionAnalysis {
            val_types: &cfg.val_types,
        };
        let mut entry = DataflowState::new();
        for storage in cfg.entry_defs() {
            if let Some(Ty::Ref(mutability, _)) = cfg.val_types.get(&storage) {
                entry.set(
                    storage,
                    Region {
                        loans: vec![Loan {
                            storage,
                            mutability: *mutability,
                        }],
                        via: vec![],
                    },
                );
            }
        }
        let result = forward_analysis(cfg, &analysis, entry);
        let mut region: FxHashMap<ValueId, Region> = FxHashMap::default();
        for exit in &result.block_exit {
            for (v, r) in &exit.values {
                region.entry(*v).or_default().join_mut(r);
            }
        }
        Self { region }
    }

    /// The region of a value; a value that names no storage has the empty
    /// region.
    pub fn region(&self, value: ValueId) -> &Region {
        self.region.get(&value).unwrap_or(&NOTHING)
    }

    pub fn storage_effect(&self, kind: &InstKind) -> StorageEffect {
        let mut effect = StorageEffect::default();
        match kind {
            InstKind::Ref { target, .. } => {
                self.touch(&mut effect, target, Mutability::Shared);
            }
            // A take empties the slot of a heap value: a write, whatever
            // the type.
            InstKind::Take { target, .. } => {
                self.touch(&mut effect, target, Mutability::Mut);
            }
            InstKind::Assign { target, path, .. } => {
                if !path.is_empty() {
                    self.touch(&mut effect, target, Mutability::Shared);
                }
                self.touch(&mut effect, target, Mutability::Mut);
            }
            InstKind::FunctionCall { args, .. } | InstKind::Spawn { args, .. } => {
                for a in args {
                    self.region(*a).loans.iter().for_each(|l| effect.add(*l));
                }
            }
            InstKind::Eval { src, .. } => {
                self.region(*src).loans.iter().for_each(|l| effect.add(*l));
            }
            _ => {}
        }
        effect
    }

    /// The values an instruction uses, plus the storage each used value
    /// keeps alive: what liveness and register allocation count.
    pub fn uses_with_storage(&self, kind: &InstKind) -> SmallVec<[ValueId; 4]> {
        let uses = inst_info::uses(kind);
        let mut all: SmallVec<[ValueId; 4]> = uses.iter().copied().collect();
        all.extend(
            uses.iter()
                .flat_map(|u| self.region(*u).loans.iter().map(|l| l.storage)),
        );
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

    fn touch(&self, effect: &mut StorageEffect, target: &RefTarget, mutability: Mutability) {
        match target {
            RefTarget::Var(s) | RefTarget::Param(s) => effect.add(Loan {
                storage: *s,
                mutability,
            }),
            RefTarget::Through(r) => {
                for loan in &self.region(*r).loans {
                    effect.add(Loan {
                        storage: loan.storage,
                        mutability,
                    });
                }
            }
        }
    }
}

pub fn contains_ref(ty: &Ty) -> bool {
    match ty {
        Ty::Ref(..) => true,
        Ty::Array(inner, _) | Ty::Option(inner) | Ty::Handle(inner) => contains_ref(inner),
        Ty::Result(ok, err) => contains_ref(ok) || contains_ref(err),
        Ty::Object(fields) => fields.values().any(contains_ref),
        Ty::Tuple(items) => items.iter().any(contains_ref),
        Ty::Fn { captures, ret, .. } => captures.iter().any(contains_ref) || contains_ref(ret),
        // An extension type with an identity is tied to what built it
        // (docs/identity-type-system.md) and may hold its references; one
        // without holds only what its type arguments show.
        Ty::UserDefined {
            type_args,
            identity_args,
            ..
        } => !identity_args.is_empty() || type_args.iter().any(contains_ref),
        Ty::Enum { variants, .. } => variants.values().flatten().any(|t| contains_ref(t)),
        _ => false,
    }
}
