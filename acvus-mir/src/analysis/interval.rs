//! The interval domain of RFC-0047 rule 7: which `Index` and `IndexSet`
//! can run without comparing their index to the slice's length.
//!
//! Each integer value carries `[lo, hi]`, whose endpoints are a constant, one
//! other SSA value plus a constant, or a length plus a constant. A slice's
//! length is fixed for as long as the slice value lives (RFC-0047 rule 2), so
//! a fact about it holds wherever the value is in scope. A container's length
//! is the one a storage holds at the moment a `len` read it through a
//! reference, and it holds only until an instruction that may write any
//! storage runs: `Facts::storage_written` forgets every such fact there. An
//! `AsSlice` of the same storage while the fact holds makes it a fact about
//! the slice.
//!
//! A fact that names a value dies where that value is defined again, which
//! a loop does on every iteration.

use std::collections::BTreeSet;

use rustc_hash::{FxHashMap, FxHashSet};

use acvus_ast::Literal;

use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{BinOp, Callee, ForSource, InstKind, Overflow, PathSeg, RefTarget, UnaryOp, ValueId};
use crate::laws::{LawTable, PostTerm, Postcondition, Relation, Subject};
use crate::ty::{IntTy, LenTerm, Mutability, Ty};

/// How many endpoints a proof follows from the index to a length. A chain
/// longer than this is left checked; it is a bound on the work, and a cycle
/// of endpoints would otherwise not end.
const CHASE_LIMIT: usize = 16;

/// A header's entry is joined with what its back edges bring this many times
/// before every endpoint that still moves is widened away.
const HEADER_VISITS_BEFORE_WIDENING: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    Const(i128),
    Value { of: ValueId, plus: i128 },
    /// The length of the slice value `of`, plus `plus`.
    SliceLen { of: ValueId, plus: i128 },
    /// The element count of the container in the storage `of` stands for
    /// (`Domain::storage`), as it stands now, plus `plus`.
    ContainerLen { of: ValueId, plus: i128 },
}

impl Endpoint {
    fn plus(self, k: i128) -> Option<Endpoint> {
        Some(match self {
            Endpoint::Const(c) => Endpoint::Const(c.checked_add(k)?),
            Endpoint::Value { of, plus } => Endpoint::Value {
                of,
                plus: plus.checked_add(k)?,
            },
            Endpoint::SliceLen { of, plus } => Endpoint::SliceLen {
                of,
                plus: plus.checked_add(k)?,
            },
            Endpoint::ContainerLen { of, plus } => Endpoint::ContainerLen {
                of,
                plus: plus.checked_add(k)?,
            },
        })
    }

    fn atom(self) -> Option<ValueId> {
        match self {
            Endpoint::Const(_) => None,
            Endpoint::Value { of, .. }
            | Endpoint::SliceLen { of, .. }
            | Endpoint::ContainerLen { of, .. } => Some(of),
        }
    }

    fn names(self, value: ValueId) -> bool {
        self.atom() == Some(value)
    }

    /// The greater of two upper bounds, where one term states it.
    fn join_hi(a: Endpoint, b: Endpoint) -> Option<Endpoint> {
        Self::same_atom(a, b, i128::max)
    }

    /// The lesser of two lower bounds, where one term states it.
    fn join_lo(a: Endpoint, b: Endpoint) -> Option<Endpoint> {
        Self::same_atom(a, b, i128::min)
    }

    fn same_atom(a: Endpoint, b: Endpoint, pick: fn(i128, i128) -> i128) -> Option<Endpoint> {
        match (a, b) {
            (Endpoint::Const(x), Endpoint::Const(y)) => Some(Endpoint::Const(pick(x, y))),
            (Endpoint::Value { of: x, plus: p }, Endpoint::Value { of: y, plus: q }) if x == y => {
                Some(Endpoint::Value {
                    of: x,
                    plus: pick(p, q),
                })
            }
            (Endpoint::SliceLen { of: x, plus: p }, Endpoint::SliceLen { of: y, plus: q })
                if x == y =>
            {
                Some(Endpoint::SliceLen {
                    of: x,
                    plus: pick(p, q),
                })
            }
            (
                Endpoint::ContainerLen { of: x, plus: p },
                Endpoint::ContainerLen { of: y, plus: q },
            ) if x == y => Some(Endpoint::ContainerLen {
                of: x,
                plus: pick(p, q),
            }),
            _ => None,
        }
    }
}

/// `None` at either end is unbounded there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Interval {
    pub lo: Option<Endpoint>,
    pub hi: Option<Endpoint>,
}

impl Interval {
    fn exactly(at: Endpoint) -> Self {
        Interval {
            lo: Some(at),
            hi: Some(at),
        }
    }

    fn forget(&mut self, value: ValueId) {
        self.forget_where(|e| e.names(value));
    }

    fn forget_where(&mut self, stale: impl Fn(Endpoint) -> bool) {
        if self.lo.is_some_and(&stale) {
            self.lo = None;
        }
        if self.hi.is_some_and(&stale) {
            self.hi = None;
        }
    }

    /// Where the container `container` was just sliced as `slice`, its
    /// current length is the slice's.
    fn sliced(&mut self, container: ValueId, slice: ValueId) {
        let move_to_slice = |end: Option<Endpoint>| match end {
            Some(Endpoint::ContainerLen { of, plus }) if of == container => {
                Some(Endpoint::SliceLen { of: slice, plus })
            }
            other => other,
        };
        self.lo = move_to_slice(self.lo);
        self.hi = move_to_slice(self.hi);
    }

    fn join(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.zip(other.lo).and_then(|(a, b)| Endpoint::join_lo(a, b)),
            hi: self.hi.zip(other.hi).and_then(|(a, b)| Endpoint::join_hi(a, b)),
        }
    }

    /// Each end that `next` does not leave where it was is unbounded.
    fn widen(&self, next: &Interval) -> Interval {
        Interval {
            lo: self.lo.filter(|_| self.lo == next.lo),
            hi: self.hi.filter(|_| self.hi == next.hi),
        }
    }

    fn constant(&self) -> Option<i128> {
        match (self.lo, self.hi) {
            (Some(Endpoint::Const(a)), Some(Endpoint::Const(b))) if a == b => Some(a),
            _ => None,
        }
    }

    fn is_top(&self) -> bool {
        self.lo.is_none() && self.hi.is_none()
    }
}

/// `lesser < greater` where `strict`, else `lesser ≤ greater`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Comparison {
    lesser: ValueId,
    greater: ValueId,
    strict: bool,
}

impl Comparison {
    fn negated(self) -> Comparison {
        Comparison {
            lesser: self.greater,
            greater: self.lesser,
            strict: !self.strict,
        }
    }

    fn names(self, value: ValueId) -> bool {
        self.lesser == value || self.greater == value
    }
}

/// What a boolean value says of two integers on each edge it decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Test {
    when_true: Comparison,
    when_false: Comparison,
}

impl Test {
    fn of(comparison: Comparison) -> Test {
        Test {
            when_true: comparison,
            when_false: comparison.negated(),
        }
    }

    fn inverted(self) -> Test {
        Test {
            when_true: self.when_false,
            when_false: self.when_true,
        }
    }

    fn names(self, value: ValueId) -> bool {
        self.when_true.names(value)
    }
}

/// What holds at one point of a body. A value absent from `intervals` is
/// unbounded.
#[derive(Debug, Clone, PartialEq, Default)]
struct Facts {
    intervals: FxHashMap<ValueId, Interval>,
    tests: FxHashMap<ValueId, Test>,
    /// A storage (`Domain::storage`) and the slice an `AsSlice` took of the
    /// container it holds, while no storage has been written since: the
    /// container's element count is the slice's length.
    slice_of: FxHashMap<ValueId, ValueId>,
    /// Every value an endpoint or a test names, so that defining a value no
    /// fact names costs nothing.
    named: FxHashSet<ValueId>,
}

impl Facts {
    fn interval(&self, value: ValueId) -> Interval {
        self.intervals.get(&value).copied().unwrap_or_default()
    }

    fn set(&mut self, value: ValueId, interval: Interval) {
        if interval.is_top() {
            self.intervals.remove(&value);
            return;
        }
        let named = [interval.lo, interval.hi].into_iter().flatten();
        self.named.extend(named.filter_map(Endpoint::atom));
        self.intervals.insert(value, interval);
    }

    fn set_slice_of(&mut self, container: ValueId, slice: ValueId) {
        self.named.insert(container);
        self.named.insert(slice);
        self.slice_of.insert(container, slice);
        for interval in self.intervals.values_mut() {
            interval.sliced(container, slice);
        }
        let named = self.intervals.values().flat_map(|i| [i.lo, i.hi]).flatten();
        self.named.extend(named.filter_map(Endpoint::atom).collect::<Vec<_>>());
    }

    /// An instruction that may write a storage ran: no container's element
    /// count is known any more.
    fn storage_written(&mut self) {
        for interval in self.intervals.values_mut() {
            interval.forget_where(|e| matches!(e, Endpoint::ContainerLen { .. }));
        }
        self.intervals.retain(|_, interval| !interval.is_top());
        self.slice_of.clear();
    }

    fn set_test(&mut self, value: ValueId, test: Test) {
        self.named.insert(test.when_true.lesser);
        self.named.insert(test.when_true.greater);
        self.tests.insert(value, test);
    }

    /// `value` is defined again: what was known of it and every fact that
    /// names its old value no longer hold.
    fn redefine(&mut self, value: ValueId) {
        self.intervals.remove(&value);
        self.tests.remove(&value);
        if !self.named.remove(&value) {
            return;
        }
        for interval in self.intervals.values_mut() {
            interval.forget(value);
        }
        self.intervals.retain(|_, interval| !interval.is_top());
        self.tests.retain(|_, test| !test.names(value));
        self.slice_of
            .retain(|container, slice| *container != value && *slice != value);
    }

    fn assume(&mut self, comparison: Comparison) {
        let bound = Endpoint::Value {
            of: comparison.greater,
            plus: match comparison.strict {
                true => -1,
                false => 0,
            },
        };
        let mut lesser = self.interval(comparison.lesser);
        lesser.hi = Some(match lesser.hi {
            Some(held) => Endpoint::join_lo(held, bound).unwrap_or(bound),
            None => bound,
        });
        self.set(comparison.lesser, lesser);
    }

    fn join(&self, other: &Facts) -> Facts {
        let mut joined = Facts::default();
        for (value, interval) in &self.intervals {
            if let Some(theirs) = other.intervals.get(value) {
                joined.set(*value, interval.join(theirs));
            }
        }
        for (value, test) in &self.tests {
            if other.tests.get(value) == Some(test) {
                joined.set_test(*value, *test);
            }
        }
        joined.keep_slices_both_hold(self, other);
        joined
    }

    fn keep_slices_both_hold(&mut self, a: &Facts, b: &Facts) {
        for (container, slice) in &a.slice_of {
            if b.slice_of.get(container) == Some(slice) {
                self.named.insert(*container);
                self.named.insert(*slice);
                self.slice_of.insert(*container, *slice);
            }
        }
    }

    /// A header's next entry: only what `next` still states as `self` did,
    /// so that a header's facts only ever shrink.
    fn widen(&self, next: &Facts) -> Facts {
        let mut widened = Facts::default();
        for (value, interval) in &self.intervals {
            if let Some(theirs) = next.intervals.get(value) {
                widened.set(*value, interval.widen(theirs));
            }
        }
        for (value, test) in &self.tests {
            if next.tests.get(value) == Some(test) {
                widened.set_test(*value, *test);
            }
        }
        widened.keep_slices_both_hold(self, next);
        widened
    }

    /// Whether `index` is below the length of `slice`: its upper bound,
    /// followed through the upper bounds of the values it names, reaches
    /// `len(slice) − k` for some `k ≥ 1`.
    fn below_len(&self, index: ValueId, slice: ValueId) -> bool {
        let Some(mut bound) = self.interval(index).hi else {
            return false;
        };
        for _ in 0..CHASE_LIMIT {
            match bound {
                Endpoint::SliceLen { of, plus } => return of == slice && plus < 0,
                Endpoint::Const(_) | Endpoint::ContainerLen { .. } => return false,
                Endpoint::Value { of, plus } => {
                    let Some(next) = self.interval(of).hi.and_then(|hi| hi.plus(plus)) else {
                        return false;
                    };
                    bound = next;
                }
            }
        }
        false
    }
}

impl Facts {
    fn excludes(&self, value: ValueId, n: i128) -> bool {
        self.constant_lower_bound(value).is_some_and(|lo| lo > n)
            || self.constant_upper_bound(value).is_some_and(|hi| hi < n)
    }

    fn constant_lower_bound(&self, value: ValueId) -> Option<i128> {
        let mut bound = self.interval(value).lo?;
        for _ in 0..CHASE_LIMIT {
            match bound {
                Endpoint::Const(c) => return Some(c),
                Endpoint::SliceLen { plus, .. } | Endpoint::ContainerLen { plus, .. } => {
                    return Some(plus);
                }
                Endpoint::Value { of, plus } => bound = self.interval(of).lo?.plus(plus)?,
            }
        }
        None
    }

    fn constant_upper_bound(&self, value: ValueId) -> Option<i128> {
        let mut bound = self.interval(value).hi?;
        for _ in 0..CHASE_LIMIT {
            match bound {
                Endpoint::Const(c) => return Some(c),
                Endpoint::SliceLen { .. } | Endpoint::ContainerLen { .. } => return None,
                Endpoint::Value { of, plus } => bound = self.interval(of).hi?.plus(plus)?,
            }
        }
        None
    }
}

/// One `Index` or `IndexSet` of a body, by where it stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstAt {
    pub block: BlockIdx,
    pub at: usize,
}

/// The `Index` and `IndexSet` instructions of `cfg` whose index the
/// interval domain puts below their slice's length.
pub fn bounded_indices(cfg: &CfgBody, laws: &LawTable) -> FxHashSet<InstAt> {
    let mut bounded = FxHashSet::default();
    let domain = Domain::new(cfg, laws);
    domain.visit_with_facts_before(|at, kind, facts| {
        if domain.index_below_len(kind, facts) {
            bounded.insert(at);
        }
    });
    bounded
}

/// The instructions of `cfg` whose trap the interval domain rules out
/// (RFC-0037 rule 2, RFC-0047 rule 7).
pub fn untrapping(cfg: &CfgBody, laws: &LawTable) -> FxHashSet<InstAt> {
    let mut untrapping = FxHashSet::default();
    let domain = Domain::new(cfg, laws);
    domain.visit_with_facts_before(|at, kind, facts| {
        if domain.index_below_len(kind, facts) || division_defined(cfg, kind, facts) {
            untrapping.insert(at);
        }
    });
    untrapping
}

/// The constant bounds the interval domain proves of a value, `None` at an
/// end it leaves unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConstantBounds {
    pub lo: Option<i128>,
    pub hi: Option<i128>,
}

impl ConstantBounds {
    pub fn excludes(self, n: i128) -> bool {
        self.lo.is_some_and(|lo| lo > n) || self.hi.is_some_and(|hi| hi < n)
    }
}

/// The constant bounds the interval domain proves of each of `values` where
/// `block` begins; `block` unreached proves none. A value a `Const` defines
/// is its literal there whatever a test assumed of it.
pub fn constant_bounds_on_entry(
    cfg: &CfgBody,
    laws: &LawTable,
    block: BlockIdx,
    values: &[ValueId],
) -> Vec<ConstantBounds> {
    let domain = Domain::new(cfg, laws);
    let entries = domain.fixpoint();
    let Some(mut facts) = entries[block.0].clone() else {
        return vec![ConstantBounds::default(); values.len()];
    };
    for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
        if let InstKind::Const { dst, value } = &inst.kind
            && let Some(constant) = domain.int_constant(*dst, value)
        {
            facts.set(*dst, Interval::exactly(Endpoint::Const(constant)));
        }
    }
    values
        .iter()
        .map(|value| ConstantBounds {
            lo: facts.constant_lower_bound(*value),
            hi: facts.constant_upper_bound(*value),
        })
        .collect()
}

fn division_defined(cfg: &CfgBody, kind: &InstKind, facts: &Facts) -> bool {
    let InstKind::BinOp {
        op: BinOp::Div | BinOp::Mod,
        left,
        right,
        ..
    } = kind
    else {
        return false;
    };
    let Some(Ty::Int(width)) = cfg.val_types.get(left) else {
        return false;
    };
    let no_overflow =
        !width.signed() || facts.excludes(*right, -1) || facts.excludes(*left, width.min());
    facts.excludes(*right, 0) && no_overflow
}

fn array_len(container: &Ty) -> Option<u64> {
    let array = match container {
        Ty::Ref(_, referent) => referent.ty().into_owned(),
        other => other.clone(),
    };
    match array {
        Ty::Array(_, LenTerm::Known(len)) => u64::try_from(len).ok(),
        _ => None,
    }
}

struct Domain<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    /// Each reference a `Ref` made of a variable or a parameter, to the
    /// first reference made of the same target and path: two references of
    /// one storage name one container. A `Ref` through another reference is
    /// left out, since that reference may name another storage each time
    /// its definition runs.
    same_storage: FxHashMap<ValueId, ValueId>,
    array_slice_lens: FxHashMap<ValueId, u64>,
}

/// What a `Ref` names.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Storage {
    target: RefTarget,
    path: Vec<PathSeg>,
}

/// The `k`th edge a block's terminator leaves by, or the body's entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EdgeId {
    Entry,
    Leaving { from: BlockIdx, k: usize },
}

/// What a terminator hands one successor.
struct Edge<'a> {
    target: BlockIdx,
    args: &'a [ValueId],
    supplied: Supplied,
    decided_by: Option<Decision>,
}

/// The edge a two-way terminator takes where `cond` is `taken`.
#[derive(Clone, Copy)]
struct Decision {
    cond: ValueId,
    taken: bool,
}

/// The target's leading parameters the terminator fills itself.
enum Supplied {
    Nothing,
    /// A range `for`'s counter, which the body sees in `[at, hi − 1]`.
    Counter { at: ValueId, hi: ValueId },
    /// Parameters the domain knows nothing of: a slice's element and index,
    /// a trip count.
    Unknown(usize),
}

impl<'a> Domain<'a> {
    fn new(cfg: &'a CfgBody, laws: &'a LawTable) -> Self {
        let mut first: FxHashMap<Storage, ValueId> = FxHashMap::default();
        let mut same_storage = FxHashMap::default();
        for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
            if let InstKind::Ref {
                dst,
                target: target @ (RefTarget::Var(_) | RefTarget::Param(_)),
                path,
                ..
            } = &inst.kind
            {
                let storage = Storage {
                    target: *target,
                    path: path.clone(),
                };
                let named = *first.entry(storage).or_insert(*dst);
                same_storage.insert(*dst, named);
            }
        }
        let array_slice_lens = cfg
            .blocks
            .iter()
            .flat_map(|block| &block.insts)
            .filter_map(|inst| match &inst.kind {
                InstKind::AsSlice { dst, container, .. } => {
                    Some((*dst, array_len(cfg.val_types.get(container)?)?))
                }
                _ => None,
            })
            .collect();
        Domain {
            cfg,
            laws,
            same_storage,
            array_slice_lens,
        }
    }

    fn index_below_len(&self, kind: &InstKind, facts: &Facts) -> bool {
        let (InstKind::Index { slice, index, .. } | InstKind::IndexSet { slice, index, .. }) =
            kind
        else {
            return false;
        };
        let below_array_len = || {
            let len = self.array_slice_lens.get(slice)?;
            Some(facts.constant_upper_bound(*index)? < i128::from(*len))
        };
        facts.below_len(*index, *slice) || below_array_len() == Some(true)
    }

    /// The value that stands for the storage `reference` names: the first
    /// reference a `Ref` made of it, or `reference` itself where no `Ref`
    /// made it (a parameter, a call's result).
    fn storage(&self, reference: ValueId) -> ValueId {
        match self.same_storage.get(&reference) {
            Some(named) => *named,
            None => reference,
        }
    }

    /// The integer a `Const` of an integer type defines `dst` as.
    fn int_constant(&self, dst: ValueId, value: &Literal) -> Option<i128> {
        let constant = match value {
            Literal::Int(v) => *v,
            Literal::IntOf(suffixed) => suffixed.value,
            _ => return None,
        };
        self.int_ty(dst).map(|_| constant)
    }

    fn int_ty(&self, value: ValueId) -> Option<IntTy> {
        match self.cfg.val_types.get(&value) {
            Some(Ty::Int(t)) => Some(*t),
            _ => None,
        }
    }

    fn reference(&self, value: ValueId) -> Option<Referent> {
        let Some(Ty::Ref(_, inner)) = self.cfg.val_types.get(&value) else {
            return None;
        };
        match &*inner.ty() {
            Ty::Slice(_) => Some(Referent::Slice),
            _ => Some(Referent::Container),
        }
    }

    /// Whether `kind` may write a storage, which moves a container's
    /// element count. A pure extern call whose arguments are words and
    /// shared references writes none (RFC-0080 rule 3); a `Drop` ends a
    /// value, not a storage a reference names; the rest that only compute
    /// values write none.
    fn may_write_storage(&self, kind: &InstKind) -> bool {
        match kind {
            InstKind::FunctionCall {
                callee: Callee::Extern { .. },
                args,
                order: None,
                ..
            } => !args.iter().all(|arg| self.only_reads(*arg)),
            InstKind::FunctionCall { .. }
            | InstKind::Spawn { .. }
            | InstKind::Eval { .. }
            | InstKind::Merge { .. }
            | InstKind::Assign { .. }
            | InstKind::Take { .. }
            | InstKind::StringAppend { .. }
            | InstKind::IndexSet { .. }
            | InstKind::Fetch { .. }
            | InstKind::Commit { .. }
            | InstKind::Poison { .. } => true,
            InstKind::Const { .. }
            | InstKind::ConstStr { .. }
            | InstKind::StringConcat { .. }
            | InstKind::StringEq { .. }
            | InstKind::StringClone { .. }
            | InstKind::StructuralEq { .. }
            | InstKind::StructuralClone { .. }
            | InstKind::Ref { .. }
            | InstKind::AsSlice { .. }
            | InstKind::Index { .. }
            | InstKind::FieldGet { .. }
            | InstKind::FieldSet { .. }
            | InstKind::BinOp { .. }
            | InstKind::UnaryOp { .. }
            | InstKind::Check { .. }
            | InstKind::CheckSteps { .. }
            | InstKind::Cast { .. }
            | InstKind::LoadFunction { .. }
            | InstKind::ArrayBegin { .. }
            | InstKind::ArrayPush { .. }
            | InstKind::MakeObject { .. }
            | InstKind::MakeTuple { .. }
            | InstKind::TupleIndex { .. }
            | InstKind::TestLiteral { .. }
            | InstKind::TestObjectKey { .. }
            | InstKind::ArrayIndex { .. }
            | InstKind::ObjectGet { .. }
            | InstKind::MakeClosure { .. }
            | InstKind::MakeVariant { .. }
            | InstKind::TestVariant { .. }
            | InstKind::UnwrapVariant { .. }
            | InstKind::Undef { .. }
            | InstKind::Nop
            | InstKind::Drop { .. } => false,
            InstKind::BlockLabel { .. }
            | InstKind::Jump { .. }
            | InstKind::JumpIf { .. }
            | InstKind::Diamond { .. }
            | InstKind::Switch { .. }
            | InstKind::For { .. }
            | InstKind::Return { .. }
            | InstKind::Diverge => {
                unreachable!("a block's instructions hold no control flow: {kind:?}")
            }
        }
    }

    /// A word, or a shared reference: an argument a callee can read and
    /// cannot write through.
    fn only_reads(&self, value: ValueId) -> bool {
        matches!(
            self.cfg.val_types.get(&value),
            Some(
                Ty::Int(_)
                    | Ty::Float
                    | Ty::Bool
                    | Ty::Char
                    | Ty::Unit
                    | Ty::Ref(Mutability::Shared, _)
            )
        )
    }

    fn visit_with_facts_before(&self, mut visit: impl FnMut(InstAt, &InstKind, &Facts)) {
        for (b, entry) in self.fixpoint().into_iter().enumerate() {
            let Some(mut facts) = entry else {
                continue;
            };
            for (at, inst) in self.cfg.blocks[b].insts.iter().enumerate() {
                let here = InstAt {
                    block: BlockIdx(b),
                    at,
                };
                visit(here, &inst.kind, &facts);
                self.transfer(&inst.kind, &mut facts);
            }
        }
    }

    fn fixpoint(&self) -> Vec<Option<Facts>> {
        let n = self.cfg.blocks.len();
        let order = reverse_postorder(self.cfg);
        let rank: FxHashMap<BlockIdx, usize> =
            order.iter().enumerate().map(|(r, b)| (*b, r)).collect();
        let headers = back_edge_targets(self.cfg, &rank);
        let mut entries: Vec<Option<Facts>> = vec![None; n];
        let mut incoming: Vec<FxHashMap<EdgeId, Facts>> = vec![FxHashMap::default(); n];
        let mut visits = vec![0usize; n];
        entries[0] = Some(Facts::default());
        incoming[0].insert(EdgeId::Entry, Facts::default());
        let mut pending: BTreeSet<usize> = BTreeSet::new();
        pending.insert(0);
        while let Some(r) = pending.pop_first() {
            let b = order[r];
            let Some(mut facts) = entries[b.0].clone() else {
                continue;
            };
            for inst in &self.cfg.blocks[b.0].insts {
                self.transfer(&inst.kind, &mut facts);
            }
            for (k, edge) in self.edges(b).into_iter().enumerate() {
                let handed = self.hand(&facts, &edge);
                let t = edge.target;
                let id = EdgeId::Leaving { from: b, k };
                if incoming[t.0].get(&id) == Some(&handed) {
                    continue;
                }
                incoming[t.0].insert(id, handed);
                let mut arriving = incoming[t.0].values();
                let Some(first) = arriving.next() else {
                    unreachable!("the edge just recorded arrives at its target");
                };
                let joined = arriving.fold(first.clone(), |acc, facts| acc.join(facts));
                let next = match &entries[t.0] {
                    Some(old)
                        if headers.contains(&t)
                            && visits[t.0] >= HEADER_VISITS_BEFORE_WIDENING =>
                    {
                        old.widen(&joined)
                    }
                    _ => joined,
                };
                if entries[t.0].as_ref() != Some(&next) {
                    visits[t.0] += 1;
                    entries[t.0] = Some(next);
                    pending.insert(rank[&t]);
                }
            }
        }
        entries
    }

    /// The facts `edge` brings its target: the test that decides it, then
    /// the target's parameters bound to what the edge passes.
    fn hand(&self, facts: &Facts, edge: &Edge<'_>) -> Facts {
        let mut handed = facts.clone();
        if let Some(Decision { cond, taken }) = edge.decided_by
            && let Some(test) = facts.tests.get(&cond)
        {
            handed.assume(match taken {
                true => test.when_true,
                false => test.when_false,
            });
        }
        let params = &self.cfg.blocks[edge.target.0].params;
        let (leading, carried) = match edge.supplied {
            Supplied::Nothing => (Vec::new(), params.as_slice()),
            Supplied::Unknown(n) => (vec![Interval::default(); n], &params[n..]),
            Supplied::Counter { at, hi } => {
                let counter = Interval {
                    lo: handed.interval(at).lo,
                    hi: Some(Endpoint::Value { of: hi, plus: -1 }),
                };
                (vec![counter], &params[1..])
            }
        };
        let passed: Vec<Interval> = edge.args.iter().map(|a| handed.interval(*a)).collect();
        let tests: Vec<Option<Test>> = edge.args.iter().map(|a| handed.tests.get(a).copied()).collect();
        for param in params {
            handed.redefine(*param);
        }
        let bound = leading.into_iter().chain(passed).zip(params.iter());
        for (mut interval, param) in bound {
            for p in params {
                interval.forget(*p);
            }
            handed.set(*param, interval);
        }
        for (test, param) in tests.into_iter().zip(carried) {
            if let Some(test) = test
                && !params.iter().any(|p| test.names(*p))
            {
                handed.set_test(*param, test);
            }
        }
        handed
    }

    fn transfer(&self, kind: &InstKind, facts: &mut Facts) {
        if self.may_write_storage(kind) {
            facts.storage_written();
        }
        for def in inst_info::defs(kind) {
            facts.redefine(def);
        }
        match kind {
            InstKind::AsSlice { dst, container, .. } => {
                facts.set_slice_of(self.storage(*container), *dst);
            }
            InstKind::Const { dst, value } => {
                if let Some(v) = self.int_constant(*dst, value) {
                    facts.set(*dst, Interval::exactly(Endpoint::Const(v)));
                }
            }
            InstKind::BinOp {
                dst,
                op,
                left,
                right,
            } => self.binop(*dst, *op, *left, *right, facts),
            InstKind::UnaryOp {
                dst,
                op: UnaryOp::Not,
                operand,
            } => {
                if let Some(test) = facts.tests.get(operand).copied() {
                    facts.set_test(*dst, test.inverted());
                }
            }
            InstKind::FunctionCall {
                dst, callee, args, ..
            } => {
                if self.int_ty(*dst).is_none() {
                    return;
                }
                let mut result = Interval::default();
                for postcondition in self.laws.postconditions_of(callee) {
                    self.read(postcondition, args, facts, &mut result);
                }
                facts.set(*dst, result);
            }
            _ => {}
        }
    }

    fn binop(&self, dst: ValueId, op: BinOp, left: ValueId, right: ValueId, facts: &mut Facts) {
        let compared = self.int_ty(left).is_some() && self.int_ty(right).is_some();
        let comparison = |lesser, greater, strict| Comparison {
            lesser,
            greater,
            strict,
        };
        let test = match op {
            BinOp::Lt if compared => Some(comparison(left, right, true)),
            BinOp::Lte if compared => Some(comparison(left, right, false)),
            BinOp::Gt if compared => Some(comparison(right, left, true)),
            BinOp::Gte if compared => Some(comparison(right, left, false)),
            _ => None,
        };
        if let Some(comparison) = test {
            facts.set_test(dst, Test::of(comparison));
            return;
        }
        let Some(ty) = self.int_ty(dst) else {
            return;
        };
        let (l, r) = (facts.interval(left), facts.interval(right));
        let by = |x: Interval, k: i128, overflow: Overflow| match overflow {
            Overflow::Trap => Self::shift_exactly(x, k, ty),
            Overflow::Wrap => self.shift(x, k, ty),
        };
        let shifted = match op {
            BinOp::Add(overflow) => match (l.constant(), r.constant()) {
                (_, Some(k)) => by(l, k, overflow),
                (Some(k), None) => by(r, k, overflow),
                (None, None) => None,
            },
            BinOp::Sub(overflow) => r.constant().and_then(|k| by(l, k.checked_neg()?, overflow)),
            _ => None,
        };
        if let Some(interval) = shifted {
            facts.set(dst, interval);
        }
    }

    /// `x + k` at width `ty` for a trapping `+` or `-`: the exact interval
    /// met with the width's range. A run whose result leaves the width
    /// ended at the operation (RFC-0037 rule 3), so every value the result
    /// holds where it is read is `x + k` over the integers, and within the
    /// width. An end the domain cannot state stays unbounded.
    fn shift_exactly(x: Interval, k: i128, ty: IntTy) -> Option<Interval> {
        let met = |end: Endpoint| match end {
            Endpoint::Const(c) => Endpoint::Const(c.clamp(ty.min(), ty.max())),
            symbolic => symbolic,
        };
        Some(Interval {
            lo: x.lo.and_then(|lo| lo.plus(k)).map(met),
            hi: x.hi.and_then(|hi| hi.plus(k)).map(met),
        })
    }

    /// `x + k` at width `ty` for a wrapping `+` or `-`, which a pass wrote
    /// (RFC-0037 rule 3): `None` where the domain cannot show that no value
    /// of `x` wraps.
    fn shift(&self, x: Interval, k: i128, ty: IntTy) -> Option<Interval> {
        if k >= 0 {
            let hi = self.at_most_max(x.hi?, k, ty)?;
            Some(Interval {
                lo: x.lo.and_then(|lo| lo.plus(k)),
                hi: Some(hi),
            })
        } else {
            let lo = self.at_least_min(x.lo?, k, ty)?;
            Some(Interval {
                lo: Some(lo),
                hi: x.hi.and_then(|hi| hi.plus(k)),
            })
        }
    }

    /// `hi + k` for `k ≥ 0`, where it stays at or below `ty`'s maximum: a
    /// constant that does, or a value of `ty` or a length plus a constant no
    /// greater than zero, which is at most that value.
    fn at_most_max(&self, hi: Endpoint, k: i128, ty: IntTy) -> Option<Endpoint> {
        let moved = hi.plus(k)?;
        let fits = match moved {
            Endpoint::Const(c) => c <= ty.max(),
            Endpoint::Value { of, plus } => plus <= 0 && self.int_ty(of) == Some(ty),
            Endpoint::SliceLen { plus, .. } | Endpoint::ContainerLen { plus, .. } => {
                plus <= 0 && ty == IntTy::U64
            }
        };
        fits.then_some(moved)
    }

    /// `lo + k` for `k < 0`, where it stays at or above `ty`'s minimum.
    fn at_least_min(&self, lo: Endpoint, k: i128, ty: IntTy) -> Option<Endpoint> {
        let moved = lo.plus(k)?;
        let fits = match moved {
            Endpoint::Const(c) => c >= ty.min(),
            Endpoint::Value { of, plus } => {
                plus >= 0 && self.int_ty(of) == Some(ty)
            }
            Endpoint::SliceLen { plus, .. } | Endpoint::ContainerLen { plus, .. } => {
                plus >= 0 && ty == IntTy::U64
            }
        };
        fits.then_some(moved)
    }

    /// Narrows `result` by what one postcondition says of `ret`, where the
    /// other side is a term the domain holds as an endpoint.
    fn read(
        &self,
        postcondition: &Postcondition,
        args: &[ValueId],
        facts: &Facts,
        result: &mut Interval,
    ) {
        let Postcondition {
            left,
            relation,
            right,
        } = postcondition;
        let (term, ret_side) = match (left, right) {
            (PostTerm::Ret, other) => (other, RetSide::Left),
            (other, PostTerm::Ret) => (other, RetSide::Right),
            _ => return,
        };
        let Some(at) = self.endpoint_of(term, args, facts) else {
            return;
        };
        let lower = |e: Endpoint, strict: bool| if strict { e.plus(1) } else { Some(e) };
        let upper = |e: Endpoint, strict: bool| if strict { e.plus(-1) } else { Some(e) };
        match (relation, ret_side) {
            (Relation::Eq, _) => *result = Interval::exactly(at),
            (Relation::Le, RetSide::Left) => result.hi = upper(at, false),
            (Relation::Lt, RetSide::Left) => result.hi = upper(at, true),
            (Relation::Le, RetSide::Right) => result.lo = lower(at, false),
            (Relation::Lt, RetSide::Right) => result.lo = lower(at, true),
        }
    }

    /// A constant, an integer argument, or the length of a slice argument or
    /// of the container a shared reference argument names, plus a constant.
    fn endpoint_of(&self, term: &PostTerm, args: &[ValueId], facts: &Facts) -> Option<Endpoint> {
        match term {
            PostTerm::Const(c) => Some(Endpoint::Const(*c)),
            PostTerm::Param(k) => {
                let of = *args.get(*k)?;
                self.int_ty(of)?;
                Some(Endpoint::Value { of, plus: 0 })
            }
            PostTerm::Len(Subject::Param(k)) => {
                let of = *args.get(*k)?;
                match self.reference(of)? {
                    Referent::Slice => Some(Endpoint::SliceLen { of, plus: 0 }),
                    Referent::Container => {
                        let storage = self.storage(of);
                        Some(match facts.slice_of.get(&storage) {
                            Some(slice) => Endpoint::SliceLen {
                                of: *slice,
                                plus: 0,
                            },
                            None => Endpoint::ContainerLen {
                                of: storage,
                                plus: 0,
                            },
                        })
                    }
                }
            }
            PostTerm::Add(a, b) => match (&**a, &**b) {
                (term, PostTerm::Const(k)) | (PostTerm::Const(k), term) => {
                    self.endpoint_of(term, args, facts)?.plus(*k)
                }
                _ => None,
            },
            PostTerm::Sub(a, b) => match &**b {
                PostTerm::Const(k) => self.endpoint_of(a, args, facts)?.plus(k.checked_neg()?),
                _ => None,
            },
            PostTerm::Ret
            | PostTerm::Len(Subject::Ret)
            | PostTerm::Mul(..)
            | PostTerm::Max(..)
            | PostTerm::Old(..) => None,
        }
    }

    fn edges(&self, b: BlockIdx) -> Vec<Edge<'a>> {
        let cfg: &'a CfgBody = self.cfg;
        let mut edges = Vec::new();
        let mut push = |target: Option<BlockIdx>, args: &'a [ValueId], supplied, decided_by| {
            if let Some(target) = target {
                edges.push(Edge {
                    target,
                    args,
                    supplied,
                    decided_by,
                });
            }
        };
        let block_of = |label| cfg.label_to_block.get(label).copied();
        match &cfg.blocks[b.0].terminator {
            Terminator::Jump { label, args } => push(block_of(label), args, Supplied::Nothing, None),
            Terminator::JumpIf {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
            }
            | Terminator::Diamond {
                cond,
                then_label,
                then_args,
                else_label,
                else_args,
                ..
            } => {
                let decided = |taken| {
                    Some(Decision {
                        cond: *cond,
                        taken,
                    })
                };
                push(block_of(then_label), then_args, Supplied::Nothing, decided(true));
                push(block_of(else_label), else_args, Supplied::Nothing, decided(false));
            }
            Terminator::Switch { arms, default, .. } => {
                for (_, label, args) in arms {
                    push(block_of(label), args, Supplied::Nothing, None);
                }
                if let Some((label, args)) = default {
                    push(block_of(label), args, Supplied::Nothing, None);
                }
            }
            Terminator::For {
                source,
                stages,
                exit,
                exit_trip,
                exit_args,
            } => {
                let supplied = match *source {
                    ForSource::Range { at, hi } => Supplied::Counter { at, hi },
                    ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => {
                        Supplied::Unknown(source.supplied_params())
                    }
                };
                let body = stages.body();
                push(block_of(&body), &[], supplied, None);
                let trip = Supplied::Unknown(exit_trip.supplied_params());
                push(block_of(exit), exit_args, trip, None);
            }
            Terminator::Fallthrough => {
                let next = BlockIdx(b.0 + 1);
                let target = (next.0 < cfg.blocks.len()).then_some(next);
                push(target, &[], Supplied::Nothing, None);
            }
            Terminator::Return { .. } | Terminator::Diverge => {}
        }
        edges
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Referent {
    Slice,
    /// A reference to anything but a slice: a container `len` reads.
    Container,
}

#[derive(Clone, Copy)]
enum RetSide {
    Left,
    Right,
}

fn reverse_postorder(cfg: &CfgBody) -> Vec<BlockIdx> {
    let mut seen = vec![false; cfg.blocks.len()];
    let mut post = Vec::new();
    let mut stack: Vec<(BlockIdx, usize)> = vec![(BlockIdx(0), 0)];
    seen[0] = true;
    while let Some(top) = stack.last_mut() {
        let block = top.0;
        let succs = cfg.successors(block);
        match succs.get(top.1) {
            Some(&next) => {
                top.1 += 1;
                if !seen[next.0] {
                    seen[next.0] = true;
                    stack.push((next, 0));
                }
            }
            None => {
                post.push(block);
                stack.pop();
            }
        }
    }
    post.reverse();
    post
}

/// A block an edge enters from a block no earlier in reverse postorder: a
/// loop header, where joining alone need not settle.
fn back_edge_targets(cfg: &CfgBody, rank: &FxHashMap<BlockIdx, usize>) -> FxHashSet<BlockIdx> {
    let mut headers = FxHashSet::default();
    for (from, r) in rank {
        for to in cfg.successors(*from) {
            if rank.get(&to).is_some_and(|t| t <= r) {
                headers.insert(to);
            }
        }
    }
    headers
}
