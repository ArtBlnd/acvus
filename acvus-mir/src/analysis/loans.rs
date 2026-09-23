//! Regions: the storage a value may name, as a trivial lifetime (RFC-0018,
//! RFC-0064). A region is a set of loans; join is union; bottom names
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
use crate::cfg::{CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{Callee, ForSource, Inst, InstKind, Label, RefTarget, ValueId};
use crate::ty::{Mutability, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoanStorage {
    Local(ValueId),
    Param { index: usize, value: ValueId },
}

impl LoanStorage {
    pub fn value(self) -> ValueId {
        match self {
            Self::Local(value) | Self::Param { value, .. } => value,
        }
    }

    pub fn param(self) -> Option<usize> {
        match self {
            Self::Local(_) => None,
            Self::Param { index, .. } => Some(index),
        }
    }
}

/// The parameters of one body, and so the only entry definitions whose loan a
/// summary can name.
///
/// `EntryStorage::loan` is the only place a `LoanStorage` is built, which is
/// how step 1 of RFC-0064 settled the question of which parameter a loan
/// names: derive the form from the body, never from the call site.
///
/// A closure's capture register is deliberately not among them.
/// `machine::bind_captures` points the register at the word the closure owns,
/// so a reference derived from a capture of an owned value names the closure's
/// own storage and dies with the closure: `LoanStorage::Local`, which the
/// result rule already refuses. A capture of a reference is read one level
/// through the register instead, and what the body then holds is the caller's
/// reference, carrying no loan of this body at all.
struct EntryStorage {
    params: FxHashMap<ValueId, usize>,
}

impl EntryStorage {
    fn of(cfg: &CfgBody) -> Self {
        Self {
            params: cfg
                .params
                .iter()
                .enumerate()
                .map(|(index, (_, value))| (*value, index))
                .collect(),
        }
    }

    fn loan(&self, value: ValueId, mutability: Mutability) -> Loan {
        let storage = match self.params.get(&value) {
            Some(&index) => LoanStorage::Param { index, value },
            None => LoanStorage::Local(value),
        };
        Loan {
            storage,
            mutability,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    pub storage: LoanStorage,
    pub mutability: Mutability,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamLoan {
    pub index: usize,
    pub mutability: Mutability,
}

/// What a body's result borrows from the body's parameters: the object
/// RFC-0064 rule 2 calls a body's summary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub loans: Vec<ParamLoan>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Leaving {
    pub summary: Summary,
    pub locals: Vec<ValueId>,
}

/// The summaries of the callees a body calls: the named functions and the
/// closure bodies of the module being checked.
///
/// Most consumers pass `NONE`, and that is a decision rather than an
/// omission. Without a summary a call's result takes the union of every
/// argument's region, which is a superset of what substitution yields, so
/// the only cost is refusing a program a summary would have admitted — no
/// pass can be made unsound by it. Threading a table through every
/// optimization pass to buy precision no pass spends would be the whole
/// pipeline's signature for nothing.
#[derive(Clone, Copy)]
pub struct Summaries<'a> {
    named: Option<&'a FxHashMap<QualifiedRef, Summary>>,
    closures: Option<&'a FxHashMap<Label, Summary>>,
}

impl<'a> Summaries<'a> {
    pub const NONE: Self = Self {
        named: None,
        closures: None,
    };

    pub fn of(table: &'a FxHashMap<QualifiedRef, Summary>) -> Self {
        Self {
            named: Some(table),
            closures: None,
        }
    }

    pub fn with_closures(self, closures: &'a FxHashMap<Label, Summary>) -> Self {
        Self {
            closures: Some(closures),
            ..self
        }
    }
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

    fn add(&mut self, storage: ValueId, mutability: Mutability) {
        match mutability {
            Mutability::Shared => self.reads.push(storage),
            Mutability::Mut => self.writes.push(storage),
        }
    }
}

// -- Which closure a value is ---------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct MadeBy {
    body: Label,
    closure: ValueId,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Made {
    By(MadeBy),
    ByMoreThanOne,
}

impl Made {
    fn joined(self, other: Made) -> Option<Made> {
        match self == other {
            true => None,
            false => Some(Made::ByMoreThanOne),
        }
    }
}

struct Carried {
    dst: ValueId,
    src: ValueId,
}

/// Which closure body each value of a body is.
///
/// A lambda bound by `let` and then called reaches this pass as a
/// `MakeClosure`, an `Assign` into the binding's slot and a `Ref` of that
/// slot at the call, which is the shape `lower.rs` emits and the shape this
/// walk follows. A value no `MakeClosure` reaches has no entry, and its
/// calls take the conservative union of the arguments rather than a
/// summary.
struct ClosureOrigins(FxHashMap<ValueId, Made>);

impl ClosureOrigins {
    fn of(cfg: &CfgBody) -> Self {
        let mut origins: FxHashMap<ValueId, Made> = FxHashMap::default();
        for block in &cfg.blocks {
            for inst in &block.insts {
                if let InstKind::MakeClosure { dst, body, .. } = &inst.kind {
                    origins.insert(
                        *dst,
                        Made::By(MadeBy {
                            body: *body,
                            closure: *dst,
                        }),
                    );
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for block in &cfg.blocks {
                for inst in &block.insts {
                    let Some(Carried { dst, src }) = carried(&inst.kind) else {
                        continue;
                    };
                    let Some(made) = origins.get(&src).copied() else {
                        continue;
                    };
                    let next = match origins.get(&dst) {
                        None => made,
                        Some(held) => match held.joined(made) {
                            Some(next) => next,
                            None => continue,
                        },
                    };
                    origins.insert(dst, next);
                    changed = true;
                }
            }
        }
        Self(origins)
    }

    fn made_by(&self, value: ValueId) -> Option<MadeBy> {
        match self.0.get(&value)? {
            Made::By(made) => Some(*made),
            Made::ByMoreThanOne => None,
        }
    }
}

fn carried(kind: &InstKind) -> Option<Carried> {
    match kind {
        InstKind::Assign { target, value, .. } => Some(Carried {
            dst: inst_info::storage(target)?,
            src: *value,
        }),
        InstKind::Take { dst, target, .. } | InstKind::Ref { dst, target, .. } => match target {
            RefTarget::Var(s) | RefTarget::Param(s) | RefTarget::Through(s) => {
                Some(Carried { dst: *dst, src: *s })
            }
        },
        _ => None,
    }
}

// -- The dataflow ---------------------------------------------------

struct RegionAnalysis<'a> {
    val_types: &'a FxHashMap<ValueId, Ty>,
    cfg: &'a CfgBody,
    entry: EntryStorage,
    closures: ClosureOrigins,
    summaries: Summaries<'a>,
}

/// A value with no type entry is taken to carry a reference: the stricter
/// reading, so a missing type never hides a loan. The pipeline types every
/// defined value; only a hand-built body lacks one.
const UNTYPED_CARRIES_REFERENCE: bool = true;

impl RegionAnalysis<'_> {
    fn loan(&self, storage: ValueId, mutability: Mutability) -> Loan {
        self.entry.loan(storage, mutability)
    }

    /// An extern declares its summary in its signature, and reading it is
    /// RFC-0064 rule 6; until then an extern call takes the union of its
    /// arguments like any callee with no summary.
    fn summary_of(&self, callee: &Callee) -> Option<&Summary> {
        match callee {
            Callee::Direct(id) => self.summaries.named?.get(id),
            Callee::Indirect(f) => self
                .summaries
                .closures?
                .get(&self.closures.made_by(*f)?.body),
            Callee::Extern { .. } => None,
        }
    }

    /// RFC-0064 rule 3: a call substitutes each `Param(i)` of the
    /// callee's summary with argument `i`'s region, and a lambda's call adds
    /// the region of the closure value itself. That addition is what a
    /// captured reference travels on: inside the closure the value read out
    /// of a capture register carries no loan, so the loan the result names
    /// reaches the caller here and nowhere else.
    fn substituted(
        &self,
        state: &DataflowState<ValueId, Region>,
        callee: &Callee,
        args: &[ValueId],
        dst: ValueId,
    ) -> Option<Region> {
        let summary = self.summary_of(callee)?;
        let mut region = Region::default();
        for loan in &summary.loans {
            let Some(arg) = args.get(loan.index) else {
                return None;
            };
            let mut borrowed = state.get(*arg);
            if loan.mutability == Mutability::Mut {
                for held in &mut borrowed.loans {
                    held.mutability = Mutability::Mut;
                }
            }
            region.join_mut(&borrowed);
        }
        if let Callee::Indirect(f) = callee
            && self.carries_ref(dst)
            && let Some(made) = self.closures.made_by(*f)
        {
            region.join_mut(&state.get(made.closure));
        }
        Some(region)
    }

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
                        loans: vec![self.loan(*s, *mutability)],
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
            // RFC-0064 rule 5: a lambda that captures a reference is a
            // holder of that loan, so its region is the join of what it
            // captured whatever its type says about the captures.
            InstKind::MakeClosure { dst, captures, .. } => {
                for c in captures {
                    self.flow(state, *c, *dst, true);
                }
            }
            InstKind::FunctionCall {
                dst, callee, args, ..
            } => {
                let Some(region) = self.substituted(state, callee, args, *dst) else {
                    for a in args {
                        self.flow(state, *a, *dst, false);
                    }
                    if let Callee::Indirect(f) = callee {
                        self.flow(state, *f, *dst, false);
                    }
                    return;
                };
                state.set(*dst, region);
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

    /// A `For` over a slice hands the body a reference into it, so the
    /// element holds the source's loan for as long as the loop runs, which
    /// is the terminator's own extent (RFC-0057 rule 2). An array's
    /// element is moved out and a range's is a number: neither is a loan.
    fn terminator_uses(&self, term: &Terminator, state: &mut DataflowState<ValueId, Region>) {
        let Terminator::For { source, body, .. } = term else {
            return;
        };
        let (ForSource::Slice(slice) | ForSource::SliceMut(slice)) = source else {
            return;
        };
        let Some(&target) = self.cfg.label_to_block.get(body) else {
            return;
        };
        let Some(&element) = self.cfg.blocks[target.0].params.first() else {
            return;
        };
        let mut region = state.get(*slice);
        region.via.push(*slice);
        state.set(element, region);
    }

    fn propagate_forward(
        &self,
        source_exit: &DataflowState<ValueId, Region>,
        params: &[ValueId],
        first: usize,
        args: &[ValueId],
        target_entry: &mut DataflowState<ValueId, Region>,
    ) -> bool {
        let mut changed = target_entry.join_from(source_exit);
        for (param, arg) in params.iter().skip(first).zip(args) {
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
        _: usize,
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
    pub fn build(cfg: &CfgBody, summaries: Summaries<'_>) -> Self {
        let analysis = RegionAnalysis {
            val_types: &cfg.val_types,
            cfg,
            entry: EntryStorage::of(cfg),
            closures: ClosureOrigins::of(cfg),
            summaries,
        };
        let mut entry = DataflowState::new();
        for storage in cfg.entry_defs() {
            if let Some(mutability) = cfg.val_types.get(&storage).and_then(entry_loan) {
                entry.set(
                    storage,
                    Region {
                        loans: vec![analysis.loan(storage, mutability)],
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

    /// RFC-0064 rules 2 and 5: a body's summary is the region of its result
    /// over `Param` loans, and a local loan in the result is refused.
    pub fn leaving(&self, value: ValueId) -> Leaving {
        let mut leaving = Leaving::default();
        for loan in &self.region(value).loans {
            match loan.storage {
                LoanStorage::Local(local) => leaving.locals.push(local),
                LoanStorage::Param { index, .. } => leaving.summary.loans.push(ParamLoan {
                    index,
                    mutability: loan.mutability,
                }),
            }
        }
        leaving
    }

    fn add_loans(&self, effect: &mut StorageEffect, value: ValueId) {
        for loan in &self.region(value).loans {
            effect.add(loan.storage.value(), loan.mutability);
        }
    }

    pub fn storage_effect(&self, kind: &InstKind) -> StorageEffect {
        let mut effect = StorageEffect::default();
        match kind {
            InstKind::Ref { target, .. } => {
                self.touch(&mut effect, target, Mutability::Shared);
            }
            // A take out of a storage may empty its slot, whatever the type;
            // `touch` bounds that by the reference when it goes through one.
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
                    self.add_loans(&mut effect, *a);
                }
            }
            InstKind::Eval { src, .. } => self.add_loans(&mut effect, *src),
            // A slice is a borrow of its container taken with the slice's
            // own mutability; indexing touches the run that borrow names
            // (RFC-0047).
            InstKind::AsSlice {
                container,
                mutability,
                ..
            } => self.touch_region(&mut effect, *container, *mutability),
            InstKind::Index { slice, .. } => {
                self.touch_region(&mut effect, *slice, Mutability::Shared)
            }
            InstKind::IndexSet { slice, .. } => {
                self.touch_region(&mut effect, *slice, Mutability::Mut)
            }
            InstKind::StringAppend { target, .. } => {
                self.touch_region(&mut effect, *target, Mutability::Mut)
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
        let mut storage: SmallVec<[ValueId; 4]> = SmallVec::new();
        for u in &uses {
            self.reachable_storage(*u, &mut storage);
        }
        all.extend(storage);
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

    /// The storage `values` keep alive, which a use of a reference reaches
    /// through its loans: the half of [`Self::uses_with_storage`] a reader
    /// that has its own use list needs.
    pub fn storage_behind(&self, values: &[ValueId]) -> SmallVec<[ValueId; 4]> {
        let mut storage: SmallVec<[ValueId; 4]> = SmallVec::new();
        for value in values {
            self.reachable_storage(*value, &mut storage);
        }
        storage
    }

    fn reachable_storage(&self, value: ValueId, out: &mut SmallVec<[ValueId; 4]>) {
        for loan in &self.region(value).loans {
            let storage = loan.storage.value();
            if !out.contains(&storage) {
                out.push(storage);
                self.reachable_storage(storage, out);
            }
        }
    }

    /// An access through a reference is bounded by that reference's own
    /// mutability. The interpreter is where that bound is kept:
    /// `ops::storage::take_through` reads the referent through a `&Value`
    /// and copies the word out of it, while `take_var` reaches its slot by
    /// `&mut` and may empty it.
    fn touch(&self, effect: &mut StorageEffect, target: &RefTarget, mutability: Mutability) {
        match target {
            RefTarget::Var(s) | RefTarget::Param(s) => effect.add(*s, mutability),
            RefTarget::Through(r) => self.touch_region(effect, *r, mutability),
        }
    }

    /// As `touch`, for a storage reached only through `reference`.
    fn touch_region(&self, effect: &mut StorageEffect, reference: ValueId, mutability: Mutability) {
        for loan in &self.region(reference).loans {
            let bounded = match (mutability, loan.mutability) {
                (Mutability::Mut, Mutability::Mut) => Mutability::Mut,
                (Mutability::Shared, _) | (_, Mutability::Shared) => Mutability::Shared,
            };
            effect.add(loan.storage.value(), bounded);
        }
    }
}

/// The loan a body's entry definition starts, and its mutability.
///
/// A parameter or capture of reference type names storage outside the body,
/// and inside the body that storage is the entry itself (RFC-0029). A
/// parameter that is a lambda holding a loan names storage outside the body
/// the same way (RFC-0064 rule 1), so it starts a loan too.
fn entry_loan(ty: &Ty) -> Option<Mutability> {
    match ty {
        Ty::Ref(mutability, _) => Some(*mutability),
        Ty::Fn { captures, .. } => {
            let held: Vec<Mutability> = captures.iter().filter_map(entry_loan).collect();
            match held.contains(&Mutability::Mut) {
                true => Some(Mutability::Mut),
                false => held.first().copied(),
            }
        }
        _ => None,
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
        } => !identity_args.is_empty() || type_args.iter().any(|a| contains_ref(&a.ty)),
        Ty::Enum { variants, .. } => variants.values().flatten().any(|t| contains_ref(t)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::{DebugInfo, MirBody};
    use crate::ty::{Task, TypeArg};
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory};

    fn shared_string() -> Ty {
        Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::String)))
    }

    /// Two reference parameters, and a reborrow of the second returned.
    fn two_parameters() -> CfgBody {
        let i = Interner::new();
        let mut factory = LocalFactory::<ValueId>::new();
        let (p0, p1, dst) = (factory.next(), factory.next(), factory.next());
        promote(MirBody {
            insts: vec![
                Inst {
                    span: Span::ZERO,
                    kind: InstKind::Ref {
                        dst,
                        target: RefTarget::Through(p1),
                        path: vec![],
                        mutability: Mutability::Shared,
                    },
                },
                Inst {
                    span: Span::ZERO,
                    kind: InstKind::Return {
                        value: dst,
                        order: None,
                    },
                },
            ],
            val_types: [
                (p0, shared_string()),
                (p1, shared_string()),
                (dst, shared_string()),
            ]
            .into_iter()
            .collect(),
            params: vec![(i.intern("a"), p0), (i.intern("b"), p1)],
            captures: vec![],
            order_param: None,
            task: Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 2,
            demoted_diamonds: Default::default(),
        })
    }

    #[test]
    fn a_param_loan_names_the_parameter_at_its_index() {
        let cfg = two_parameters();
        let loans = Loans::build(&cfg, Summaries::NONE);
        let mut seen = 0;
        for value in cfg.val_types.keys() {
            for loan in &loans.region(*value).loans {
                let Some(index) = loan.storage.param() else {
                    panic!("a body of only parameters holds {:?}", loan.storage);
                };
                assert_eq!(cfg.params[index].1, loan.storage.value());
                seen += 1;
            }
        }
        assert!(seen >= cfg.params.len(), "every parameter starts a region");
    }

    #[test]
    fn the_summary_names_the_parameter_the_result_reborrows() {
        let cfg = two_parameters();
        let loans = Loans::build(&cfg, Summaries::NONE);
        let returned = cfg.params[1].1;
        let dst = *cfg
            .val_types
            .keys()
            .find(|v| **v != cfg.params[0].1 && **v != returned)
            .expect("the reborrow");
        let leaving = loans.leaving(dst);
        assert_eq!(leaving.locals, []);
        assert_eq!(
            leaving.summary.loans,
            [ParamLoan {
                index: 1,
                mutability: Mutability::Shared,
            }]
        );
    }
}
