//! Regions: the storage a value may name, as a trivial lifetime (RFC-0018,
//! RFC-0064, RFC-0079). A region is a set of loans; join is union; bottom
//! names nothing. A value holds one region per position of its type
//! (`positions`), and the storage holds the positions: a read through a
//! reference reads what the storage it names holds now. Every pass that
//! orders, moves, removes, or allocates around storage asks here rather than
//! reading the instruction on its own, through two readers: `names`, the
//! storage a reference points at, and `holds`, every loan in any position.

use std::ops::Range;
use std::sync::Arc;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::analysis::dataflow::{DataflowAnalysis, DataflowState, forward_analysis};
use crate::analysis::domain::SemiLattice;
use crate::analysis::inst_info;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{
    Callee, ForSource, IndexMode, Inst, InstKind, PathSeg, RefTarget, ValueId,
};
use crate::decl::{Declarations, Segment, segments};
use crate::flows::segment_positions;
use crate::ty::{
    Alignment, Concrete, FlowEnd, FlowTerm, Flows, Holds, Labelled, Mutability, PositionKind, Source, Ty,
    TyKind, Types, position_kinds, positions,
};

// -- Positions (ownership §5) ---------------------------------------

/// A field of an object or a payload of an enum's variant.
struct NamedPart {
    name: Astr,
    ty: Ty,
}

/// A part of a value: the position it starts at among the whole's, and its
/// type.
struct Part {
    at: usize,
    ty: Ty,
}

/// The positions of a module's types: their count (`ty::positions`), their
/// kinds, and where a path names a part among them.
#[derive(Clone, Copy)]
pub struct Layout<'a> {
    pub types: &'a Types<Concrete>,
    pub decls: &'a Declarations,
}

impl<'a> Layout<'a> {
    pub fn new(types: &'a Types<Concrete>, decls: &'a Declarations) -> Self {
        Layout { types, decls }
    }

    /// How many positions a value of `ty` has.
    pub fn positions(self, ty: Ty) -> usize {
        positions(self.types, self.decls, ty)
    }

    /// Whether a value of `ty` reads, through one of its positions, what a
    /// storage holds when it is read: a reference whose pointee has
    /// positions, or a function value's captures.
    pub fn reads_through(self, ty: Ty) -> bool {
        self.kinds(ty).into_iter().any(|kind| match kind {
            PositionKind::Ref(_, width) => width > 0,
            PositionKind::Captures(_) => true,
            PositionKind::Lifetime | PositionKind::InFlight => false,
        })
    }

    /// Whether a callee handed a value of `ty` can write through it: a `&mut`
    /// position, or a function value that captured mutably.
    pub fn writes_through(self, ty: Ty) -> bool {
        self.kinds(ty).into_iter().any(|kind| {
            matches!(
                kind,
                PositionKind::Ref(Mutability::Mut, _) | PositionKind::Captures(Holds::Mut)
            )
        })
    }

    /// The kind of each of `ty`'s positions, in order (`ty::position_kinds`),
    /// bounded by the access a holder has: behind a `&T`, a `&mut` reaches
    /// its target only as a shared reborrow (RFC-0029 rule 5), and a
    /// function value's mutable captures are read-only.
    pub fn kinds(self, ty: Ty) -> Vec<PositionKind> {
        let mut kinds = position_kinds(self.types, self.decls, ty);
        let mut shared_until = 0;
        for k in 0..kinds.len() {
            let behind_shared = k < shared_until;
            match &mut kinds[k] {
                PositionKind::Ref(mutability, width) => {
                    if behind_shared {
                        *mutability = Mutability::Shared;
                    }
                    if *mutability == Mutability::Shared {
                        shared_until = shared_until.max(k + 1 + *width);
                    }
                }
                PositionKind::Captures(holds) => {
                    if behind_shared && *holds == Holds::Mut {
                        *holds = Holds::Shared;
                    }
                }
                PositionKind::Lifetime | PositionKind::InFlight => {}
            }
        }
        kinds
    }

    /// An object's fields or an enum's payloads, in the order their
    /// positions are laid out, which is the order the type holds them in.
    fn parts_in_order(self, ty: Ty) -> Vec<NamedPart> {
        match self.types.kind(ty) {
            TyKind::Object(object) => object
                .fields()
                .iter()
                .map(|(name, ty)| NamedPart { name: *name, ty: *ty })
                .collect(),
            TyKind::Enum { variants, .. } => variants
                .iter()
                .filter_map(|(name, payload)| Some(NamedPart { name: *name, ty: (*payload)? }))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The parts a path step names in a value of `ty`. A payload of a
    /// result or an enum is any one of its variants'.
    fn steps(self, ty: Ty, seg: &PathSeg) -> Vec<Part> {
        let named = |name: Astr| {
            let mut at = 0;
            for part in self.parts_in_order(ty) {
                if part.name == name {
                    return vec![Part { at, ty: part.ty }];
                }
                at += self.positions(part.ty);
            }
            Vec::new()
        };
        match (self.types.kind(ty), seg) {
            (TyKind::Object(_) | TyKind::Enum { .. }, PathSeg::Field(name)) => named(*name),
            (TyKind::Tuple(items, _), PathSeg::Index(index)) => {
                let at = items.iter().take(*index).map(|item| self.positions(*item)).sum();
                items.get(*index).map(|item| Part { at, ty: *item }).into_iter().collect()
            }
            (TyKind::Array(inner, _, _) | TyKind::Slice(inner), PathSeg::Index(_))
            | (TyKind::Option(inner, _), PathSeg::Payload) => vec![Part { at: 0, ty: *inner }],
            (TyKind::Result(ok, err, _), PathSeg::Payload) => vec![
                Part { at: 0, ty: *ok },
                Part {
                    at: self.positions(*ok),
                    ty: *err,
                },
            ],
            (TyKind::Enum { .. }, PathSeg::Payload) => {
                let mut at = 0;
                let mut out = Vec::new();
                for part in self.parts_in_order(ty) {
                    let width = self.positions(part.ty);
                    out.push(Part { at, ty: part.ty });
                    at += width;
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// Where each part `path` may name starts among `ty`'s positions, for
    /// the parts `width` positions wide.
    fn offsets(self, ty: Ty, path: &[PathSeg], width: usize) -> Vec<usize> {
        let mut parts: Vec<Part> = vec![Part { at: 0, ty }];
        for seg in path {
            parts = parts
                .into_iter()
                .flat_map(|outer| {
                    self.steps(outer.ty, seg).into_iter().map(move |inner| Part {
                        at: outer.at + inner.at,
                        ty: inner.ty,
                    })
                })
                .collect();
        }
        parts
            .into_iter()
            .filter(|part| self.positions(part.ty) == width)
            .map(|part| part.at)
            .collect()
    }

    /// The positions of a storage of type `slot` at which a value of type
    /// `taken` read out of it at `path` holds a loan, with the loan's
    /// mutability: where the value has a reference or a closure's captures.
    /// Every position of the storage stands for the value when its place in
    /// it is not known.
    pub fn held_positions(self, slot: Ty, path: &[PathSeg], taken: Ty) -> Vec<(usize, Mutability)> {
        let held: Vec<(usize, Mutability)> = self
            .kinds(taken)
            .into_iter()
            .enumerate()
            .filter_map(|(k, kind)| match kind {
                PositionKind::Ref(mutability, _) => Some((k, mutability)),
                PositionKind::Captures(holds) => captured(holds).map(|mutability| (k, mutability)),
                PositionKind::Lifetime | PositionKind::InFlight => None,
            })
            .collect();
        if held.is_empty() {
            return held;
        }
        let at = self.offsets(slot, path, self.positions(taken));
        match at.is_empty() {
            true => {
                let mutability = match held.iter().any(|(_, m)| *m == Mutability::Mut) {
                    true => Mutability::Mut,
                    false => Mutability::Shared,
                };
                (0..self.positions(slot)).map(|k| (k, mutability)).collect()
            }
            false => at
                .into_iter()
                .flat_map(|offset| held.iter().map(move |(k, m)| (offset + k, *m)))
                .collect(),
        }
    }

    /// How many positions the target of the reference at position `k` of
    /// `ty` has, when position `k` is a reference; its positions follow it.
    fn target_width(self, ty: Ty, k: usize) -> Option<usize> {
        match position_kinds(self.types, self.decls, ty).get(k) {
            Some(PositionKind::Ref(_, width)) => Some(*width),
            _ => None,
        }
    }
}

// -- Loans ----------------------------------------------------------

/// Storage outside the body that position `position` of an entry
/// definition names (RFC-0079 rule 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Outside {
    pub entry: ValueId,
    pub position: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoanStorage {
    Local(ValueId),
    Param { index: usize, value: ValueId },
    Outside(Outside),
}

impl LoanStorage {
    /// The slot of this body the storage is, if it is one.
    pub fn slot(self) -> Option<ValueId> {
        match self {
            Self::Local(value) | Self::Param { value, .. } => Some(value),
            Self::Outside(_) => None,
        }
    }

    pub fn param(self) -> Option<usize> {
        match self {
            Self::Param { index, .. } => Some(index),
            Self::Local(_) | Self::Outside(_) => None,
        }
    }

    /// The value whose positions hold what this storage holds.
    fn holder(self) -> ValueId {
        match self {
            Self::Local(value) | Self::Param { value, .. } => value,
            Self::Outside(outside) => outside.entry,
        }
    }
}

/// The parameters of one body, and so the only entry definitions whose loan
/// an output of the body can hold as a parameter's (`Loans::held`).
///
/// `EntryStorage::loan` is the only place a `Local` or `Param` storage is
/// built, which is how step 1 of RFC-0064 settled the question of which
/// parameter a loan names: derive the form from the body, never from the call
/// site.
///
/// A closure's capture register is deliberately not among them.
/// `machine::bind_captures` points the register at the word the closure owns,
/// so a reference derived from a capture of an owned value names the closure's
/// own storage and dies with the closure: `LoanStorage::Local`, which the
/// output rule refuses. A capture of a reference is read one level through
/// the register instead, and what the body then holds is the caller's
/// reference, whose loans are the register's `Outside` positions: the
/// closure's `Captures` input.
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

    fn storage(&self, value: ValueId) -> LoanStorage {
        match self.params.get(&value) {
            Some(&index) => LoanStorage::Param { index, value },
            None => LoanStorage::Local(value),
        }
    }

    fn loan(&self, value: ValueId, mutability: Mutability) -> Loan {
        Loan {
            storage: self.storage(value),
            mutability,
            taken: mutability,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loan {
    pub storage: LoanStorage,
    /// What the holder may do through the loan: a `&T` position holds it
    /// `Shared` whatever it was taken as (`read_only_shared`).
    pub mutability: Mutability,
    /// How the loan was taken at its storage. A shared reborrow of a `&mut`,
    /// and a `&T` a call returns from one, hold the exclusive loan read-only:
    /// its holder reads through it, and the place itself is still not read or
    /// written while it lives (RFC-0018 rule 8).
    pub taken: Mutability,
}

/// One position's loans.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub loans: Vec<Loan>,
}

impl Region {
    fn join_mut(&mut self, other: &Region) -> bool {
        let mut changed = false;
        for loan in &other.loans {
            if !self.loans.contains(loan) {
                self.loans.push(*loan);
                changed = true;
            }
        }
        changed
    }

    fn without(mut self, storage: LoanStorage) -> Self {
        self.loans.retain(|loan| loan.storage != storage);
        self
    }

    fn shared(mut self) -> Self {
        for loan in &mut self.loans {
            loan.mutability = Mutability::Shared;
        }
        self
    }
}

/// The regions a value holds, one per position of its type, and the
/// references it was built through: a touch through one of those is the
/// value's own access.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct Regions {
    positions: Vec<Region>,
    via: Via,
}

/// The references a value was built through.
///
/// The set is a persistent balanced tree, and a value built from another
/// shares that value's nodes. Each `ArrayPush` of an array literal is built
/// from the one before it and carries every earlier push in its set; the
/// sorted `Vec` this replaces was copied at each push, which made the
/// analysis of a literal quadratic in its length. Shared, a push costs one
/// insertion.
#[derive(Default, Clone)]
pub struct Via(Tree);

type Tree = Option<Arc<ViaNode>>;

struct ViaNode {
    value: ValueId,
    height: u8,
    len: usize,
    left: Tree,
    right: Tree,
}

fn height(tree: &Tree) -> u8 {
    tree.as_deref().map_or(0, |node| node.height)
}

fn tree_len(tree: &Tree) -> usize {
    tree.as_deref().map_or(0, |node| node.len)
}

fn node(value: ValueId, left: Tree, right: Tree) -> Tree {
    Some(Arc::new(ViaNode {
        value,
        height: 1 + height(&left).max(height(&right)),
        len: 1 + tree_len(&left) + tree_len(&right),
        left,
        right,
    }))
}

/// An AVL rebalance of `node(value, left, right)`, whose sides differ in
/// height by at most two.
fn balanced(value: ValueId, left: Tree, right: Tree) -> Tree {
    let (hl, hr) = (height(&left), height(&right));
    if let Some(l) = left.as_deref().filter(|_| hl > hr + 1) {
        return match l.right.as_deref().filter(|_| height(&l.right) > height(&l.left)) {
            Some(lr) => node(
                lr.value,
                node(l.value, l.left.clone(), lr.left.clone()),
                node(value, lr.right.clone(), right),
            ),
            None => node(l.value, l.left.clone(), node(value, l.right.clone(), right)),
        };
    }
    if let Some(r) = right.as_deref().filter(|_| hr > hl + 1) {
        return match r.left.as_deref().filter(|_| height(&r.left) > height(&r.right)) {
            Some(rl) => node(
                rl.value,
                node(value, left, rl.left.clone()),
                node(r.value, rl.right.clone(), r.right.clone()),
            ),
            None => node(r.value, node(value, left, r.left.clone()), r.right.clone()),
        };
    }
    node(value, left, right)
}

/// `tree` with `value` added, or `None` when it already holds it.
fn inserted(tree: &Tree, value: ValueId) -> Option<Tree> {
    let Some(at) = tree.as_deref() else {
        return Some(node(value, None, None));
    };
    match value.cmp(&at.value) {
        std::cmp::Ordering::Less => {
            inserted(&at.left, value).map(|left| balanced(at.value, left, at.right.clone()))
        }
        std::cmp::Ordering::Greater => {
            inserted(&at.right, value).map(|right| balanced(at.value, at.left.clone(), right))
        }
        std::cmp::Ordering::Equal => None,
    }
}

/// The members of a set in ascending order.
struct ViaIter<'a> {
    stack: Vec<&'a ViaNode>,
}

impl<'a> ViaIter<'a> {
    fn new(tree: &'a Tree) -> Self {
        let mut iter = ViaIter { stack: Vec::new() };
        iter.descend(tree);
        iter
    }

    fn descend(&mut self, mut tree: &'a Tree) {
        while let Some(at) = tree.as_deref() {
            self.stack.push(at);
            tree = &at.left;
        }
    }
}

impl<'a> Iterator for ViaIter<'a> {
    type Item = &'a ValueId;

    fn next(&mut self) -> Option<&'a ValueId> {
        let at = self.stack.pop()?;
        self.descend(&at.right);
        Some(&at.value)
    }
}

impl Via {
    pub const fn new() -> Self {
        Self(None)
    }

    pub fn contains(&self, value: &ValueId) -> bool {
        let mut tree = &self.0;
        while let Some(at) = tree.as_deref() {
            tree = match value.cmp(&at.value) {
                std::cmp::Ordering::Less => &at.left,
                std::cmp::Ordering::Greater => &at.right,
                std::cmp::Ordering::Equal => return true,
            };
        }
        false
    }

    /// This set with `value` added.
    pub fn with(&self, value: ValueId) -> Self {
        let mut out = self.clone();
        out.insert(value);
        out
    }

    fn len(&self) -> usize {
        tree_len(&self.0)
    }

    fn same(&self, other: &Via) -> bool {
        match (&self.0, &other.0) {
            (Some(mine), Some(theirs)) => Arc::ptr_eq(mine, theirs),
            (None, None) => true,
            _ => false,
        }
    }

    fn iter(&self) -> impl Iterator<Item = &ValueId> {
        ViaIter::new(&self.0)
    }

    /// Adds `value`; true when it was new.
    fn insert(&mut self, value: ValueId) -> bool {
        match inserted(&self.0, value) {
            Some(tree) => {
                self.0 = tree;
                true
            }
            None => false,
        }
    }

    /// Adds every member of `other`; true when one was new. The smaller set
    /// is inserted into the larger.
    fn join_mut(&mut self, other: &Via) -> bool {
        if self.same(other) || other.len() == 0 {
            return false;
        }
        let before = self.len();
        let (mut into, from) = match other.len() > before {
            true => (other.clone(), &*self),
            false => (self.clone(), other),
        };
        for value in from.iter() {
            into.insert(*value);
        }
        let added = into.len() > before;
        *self = into;
        added
    }
}

impl PartialEq for Via {
    fn eq(&self, other: &Via) -> bool {
        self.same(other) || (self.len() == other.len() && self.iter().eq(other.iter()))
    }
}

impl Eq for Via {}

impl std::fmt::Debug for Via {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl FromIterator<ValueId> for Via {
    fn from_iter<I>(values: I) -> Self
    where
        I: IntoIterator<Item = ValueId>,
    {
        let mut via = Via::new();
        for value in values {
            via.insert(value);
        }
        via
    }
}

impl Regions {
    /// What a reference points at: its first position.
    pub fn names(&self) -> &[Loan] {
        self.positions.first().map_or(&[], |region| &region.loans)
    }

    /// Every loan in any position (RFC-0079 rule 10).
    pub fn holds(&self) -> impl Iterator<Item = &Loan> {
        self.positions.iter().flat_map(|region| &region.loans)
    }

    pub fn holds_any(&self) -> bool {
        self.positions.iter().any(|region| !region.loans.is_empty())
    }

    pub fn via(&self) -> &Via {
        &self.via
    }

    pub fn position(&self, k: usize) -> &[Loan] {
        self.positions.get(k).map_or(&[], |region| &region.loans)
    }

    fn folded(&self) -> Region {
        let mut all = Region::default();
        for region in &self.positions {
            all.join_mut(region);
        }
        all
    }

    /// The region of a value built from `from`: `from`'s loans, held
    /// through `from`. Every flow of a loan is one, so a value made from a
    /// holder — read out of it, stored into it, passed on, projected — is
    /// that holder's own reference, not a second holder beside it.
    fn through(mut self, from: ValueId) -> Self {
        self.via.insert(from);
        self
    }

    fn with_via(positions: Vec<Region>, via: &Via) -> Self {
        Self {
            positions,
            via: via.clone(),
        }
    }
}

impl SemiLattice for Regions {
    fn bottom() -> Self {
        Self::default()
    }

    fn join_mut(&mut self, other: &Self) -> bool {
        let mut changed = false;
        if self.positions.len() < other.positions.len() {
            self.positions.resize_with(other.positions.len(), Region::default);
        }
        for (mine, theirs) in self.positions.iter_mut().zip(&other.positions) {
            changed |= mine.join_mut(theirs);
        }
        changed |= self.via.join_mut(&other.via);
        changed
    }
}

fn fill(width: usize, region: Region) -> Vec<Region> {
    vec![region; width]
}

fn fold(positions: &[Region]) -> Region {
    let mut all = Region::default();
    for region in positions {
        all.join_mut(region);
    }
    all
}

/// `from` read as a value of `width` positions: the same positions when the
/// widths agree, since a part as wide as the whole starts at its first
/// position, and every loan in every position otherwise.
fn reshape(from: &[Region], width: usize) -> Vec<Region> {
    match from.len() == width {
        true => from.to_vec(),
        false => fill(width, fold(from)),
    }
}

fn padded(mut positions: Vec<Region>, width: usize) -> Vec<Region> {
    positions.resize_with(width, Region::default);
    positions
}

#[derive(Default, Debug)]
pub struct StorageEffect {
    pub reads: SmallVec<[ValueId; 2]>,
    pub writes: SmallVec<[ValueId; 2]>,
    /// The storages among `reads` reached through a loan taken `&mut` and
    /// held read-only: a shared reborrow of a `&mut`, or a `&T` a call
    /// returned from one.
    pub held: SmallVec<[ValueId; 2]>,
    /// The storages among `reads` and `writes` touched at the place itself,
    /// not through a reference.
    pub placed: SmallVec<[ValueId; 2]>,
}

impl StorageEffect {
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }

    /// Whether the two keep their order. Two reads commute, except a touch
    /// of a place and a read through a loan taken `&mut` on it: moved past
    /// one another, the place is touched while that loan lives, which
    /// RFC-0018 rule 8 excludes however the loan is held.
    pub fn conflicts(&self, other: &StorageEffect) -> bool {
        let hits = |a: &[ValueId], b: &[ValueId]| a.iter().any(|s| b.contains(s));
        hits(&self.writes, &other.reads)
            || hits(&self.writes, &other.writes)
            || hits(&self.reads, &other.writes)
            || hits(&self.held, &other.placed)
            || hits(&self.placed, &other.held)
    }

    fn add(&mut self, loan: Loan) {
        let Some(slot) = loan.storage.slot() else {
            return;
        };
        match loan.mutability {
            Mutability::Shared => self.reads.push(slot),
            Mutability::Mut => self.writes.push(slot),
        }
        if loan.mutability == Mutability::Shared && loan.taken == Mutability::Mut {
            self.held.push(slot);
        }
    }

    fn add_placed(&mut self, storage: LoanStorage, mutability: Mutability) {
        let Some(slot) = storage.slot() else {
            return;
        };
        match mutability {
            Mutability::Shared => self.reads.push(slot),
            Mutability::Mut => self.writes.push(slot),
        }
        self.placed.push(slot);
    }
}

// -- The dataflow ---------------------------------------------------

type State = DataflowState<ValueId, Regions>;

struct RegionAnalysis<'a> {
    layout: Layout<'a>,
    val_types: &'a FxHashMap<ValueId, Ty>,
    cfg: &'a CfgBody,
    entry: EntryStorage,
}

/// Where the loans a storage holds live among its holder's positions.
enum Contents {
    /// The positions of the value the storage holds.
    Positions(Range<usize>),
    /// One position standing for everything behind it: a region parameter, a
    /// closure's captures or a call in flight, whose parts have no type here.
    Summarized(usize),
}

/// A part of a value written into storage: the offsets it may start at in
/// the pointee, which is `pointee_width` positions wide, and its positions.
/// No offset is a part whose place is not known.
struct WrittenPart<'v> {
    pointee_width: usize,
    at: &'v [usize],
    values: &'v [Region],
}

/// A value to place into a built value, at the part `seg` names.
struct Placed {
    seg: PathSeg,
    value: ValueId,
}

/// What a callee can reach from the values a call hands it.
#[derive(Default)]
struct Reach {
    loans: Region,
    via: Via,
    storages: Vec<LoanStorage>,
}

impl Reach {
    fn take(&mut self, region: &Region) {
        for loan in &region.loans {
            if !self.loans.loans.contains(loan) {
                self.loans.loans.push(*loan);
            }
            if !self.storages.contains(&loan.storage) {
                self.storages.push(loan.storage);
            }
        }
    }
}

/// The values a transfer gave new regions.
type Changed = Vec<ValueId>;

fn add_via(via: &mut Via, held: &Regions, from: ValueId) {
    via.join_mut(&held.via);
    via.insert(from);
}

impl RegionAnalysis<'_> {
    /// # Panics
    /// `v` has no recorded type: every IR value has one.
    fn recorded(&self, v: ValueId) -> Ty {
        crate::ir::recorded(self.val_types, v)
    }

    fn width(&self, v: ValueId) -> usize {
        self.layout.positions(self.recorded(v))
    }

    /// The regions `v` holds in `state`, one per position of its type. A
    /// value the dataflow has not reached is the lattice's bottom
    /// (`DataflowState::get`, `analysis/dataflow.rs:31`), which at `v`'s
    /// width is every position holding no loan.
    ///
    /// # Panics
    /// A holder's regions are not its type's width: `put`, `join_at` and the
    /// entry regions keep every holder at its width.
    fn held(&self, state: &State, v: ValueId) -> Regions {
        let width = self.width(v);
        let mut regions = state.get(v);
        if regions.positions.is_empty() {
            regions.positions = fill(width, Region::default());
        }
        assert_eq!(regions.positions.len(), width, "a holder's regions are its type's width");
        regions
    }

    /// # Panics
    /// `reference` is not of reference type: every value an instruction
    /// reads through is.
    fn pointee(&self, reference: ValueId) -> Ty {
        match self.layout.types.kind(self.recorded(reference)) {
            TyKind::Ref(_, inner) => inner.ty(),
            _ => panic!("a value read through is of reference type"),
        }
    }

    fn pointee_width(&self, reference: ValueId) -> usize {
        self.layout.positions(self.pointee(reference))
    }

    /// The width of what a `Handle` carries.
    ///
    /// # Panics
    /// `handle` is not of `Handle` type: a `Spawn` defines one and a
    /// `TakeEffect` reads one.
    fn handled_width(&self, handle: ValueId) -> usize {
        match self.layout.types.kind(self.recorded(handle)) {
            TyKind::Handle(inner) => self.layout.positions(*inner),
            _ => panic!("a spawned call's value is of Handle type"),
        }
    }

    /// `regions` given to `v`: a position `regions` lacks holds nothing, and
    /// more positions than `v` has put every loan in each of `v`'s.
    fn put(&self, state: &mut State, v: ValueId, regions: Regions, changed: &mut Changed) {
        let width = self.width(v);
        let positions = match regions.positions.len() > width {
            true => fill(width, fold(&regions.positions)),
            false => padded(regions.positions, width),
        };
        let mut regions = Regions {
            positions,
            via: regions.via,
        };
        self.read_only_shared(v, &mut regions.positions);
        state.set(v, regions);
        changed.push(v);
    }

    fn join_at(&self, state: &mut State, v: ValueId, at: usize, region: &Region, changed: &mut Changed) {
        let mut regions = self.held(state, v);
        if regions.positions[at].join_mut(region) {
            self.read_only_shared(v, &mut regions.positions);
            state.set(v, regions);
            changed.push(v);
        }
    }

    /// A loan at a `&T` position of `v` is shared, whatever it was where it
    /// came from: nothing writes through a `&T`.
    fn read_only_shared(&self, v: ValueId, positions: &mut [Region]) {
        let holds_mut = positions
            .iter()
            .any(|region| region.loans.iter().any(|loan| loan.mutability == Mutability::Mut));
        if !holds_mut {
            return;
        }
        for (region, kind) in positions.iter_mut().zip(self.layout.kinds(self.recorded(v))) {
            if let PositionKind::Ref(Mutability::Shared, _) = kind {
                let mut shared = Region::default();
                shared.join_mut(&std::mem::take(region).shared());
                *region = shared;
            }
        }
    }

    /// Where what `storage` holds lives among its holder's positions. A
    /// storage of reference type is what the reference points at, since `&r`
    /// of a reference is a reborrow (RFC-0029 rule 3) and never names the
    /// slot holding it.
    fn contents(&self, storage: LoanStorage) -> Contents {
        match storage {
            LoanStorage::Local(value) | LoanStorage::Param { value, .. } => {
                let ty = self.recorded(value);
                match self.layout.types.kind(ty) {
                    TyKind::Ref(_, inner) => Contents::Positions(1..1 + self.layout.positions(inner.ty())),
                    _ => Contents::Positions(0..self.layout.positions(ty)),
                }
            }
            LoanStorage::Outside(Outside { entry, position }) => {
                match self.layout.target_width(self.recorded(entry), position) {
                    Some(width) => Contents::Positions(position + 1..position + 1 + width),
                    None => Contents::Summarized(position),
                }
            }
        }
    }

    /// RFC-0079 rule 3: what a reference naming `storage` reads, as a value
    /// of `width` positions.
    fn storage_view(&self, state: &State, storage: LoanStorage, width: usize) -> Vec<Region> {
        let held = self.held(state, storage.holder());
        let at = |k: usize| held.positions[k].clone();
        match self.contents(storage) {
            Contents::Positions(range) => reshape(&range.map(at).collect::<Vec<_>>(), width),
            Contents::Summarized(position) => fill(width, at(position)),
        }
    }

    fn content_range(&self, storage: LoanStorage) -> Range<usize> {
        match self.contents(storage) {
            Contents::Positions(range) => range,
            Contents::Summarized(position) => position..position + 1,
        }
    }

    /// RFC-0079 rule 4: a written part joined into what `storage` holds, or
    /// into every position it holds when the part's place is not known. A
    /// storage never takes a loan on itself.
    fn storage_write(&self, state: &mut State, storage: LoanStorage, part: &WrittenPart, changed: &mut Changed) {
        let holder = storage.holder();
        let range = self.content_range(storage);
        match range.len() == part.pointee_width && !part.at.is_empty() {
            true => {
                for offset in part.at {
                    for (k, region) in part.values.iter().enumerate() {
                        let region = region.clone().without(storage);
                        self.join_at(state, holder, range.start + offset + k, &region, changed);
                    }
                }
            }
            false => {
                let all = fold(part.values).without(storage);
                for k in range {
                    self.join_at(state, holder, k, &all, changed);
                }
            }
        }
    }

    /// What `reference` points at now: its own tail, and what every storage
    /// it names holds (RFC-0079 rule 3).
    fn deref(&self, state: &State, reference: ValueId) -> Vec<Region> {
        let width = self.pointee_width(reference);
        let held = self.held(state, reference);
        let mut out: Vec<Region> = (1..1 + width)
            .map(|k| held.positions[k].clone())
            .collect();
        for loan in held.names() {
            for (mine, theirs) in out.iter_mut().zip(self.storage_view(state, loan.storage, width)) {
                mine.join_mut(&theirs);
            }
        }
        out
    }

    /// The part of `from`, a value of type `ty`, that `path` names, as a value
    /// of `width` positions; every loan of `from` when the part is not known.
    fn project(&self, from: &[Region], ty: Ty, path: &[PathSeg], width: usize) -> Vec<Region> {
        if path.is_empty() {
            return reshape(from, width);
        }
        let at = self.layout.offsets(ty, path, width);
        if at.is_empty() {
            return fill(width, fold(from));
        }
        let mut out = fill(width, Region::default());
        for offset in at {
            for (k, mine) in out.iter_mut().enumerate() {
                if let Some(region) = from.get(offset + k) {
                    mine.join_mut(region);
                }
            }
        }
        out
    }

    /// Every position of `dst` holds every loan any of `uses` holds.
    fn gather(&self, state: &mut State, dst: ValueId, uses: &[ValueId], changed: &mut Changed) {
        let mut all = Region::default();
        let mut via = Via::new();
        for u in uses {
            let held = self.held(state, *u);
            all.join_mut(&held.folded());
            add_via(&mut via, &held, *u);
        }
        let regions = Regions::with_via(fill(self.width(dst), all), &via);
        self.put(state, dst, regions, changed);
    }

    /// A value built from `parts`, each placed at every offset its path step
    /// may name among `dst`'s positions.
    fn build_from(&self, state: &mut State, dst: ValueId, parts: &[Placed], changed: &mut Changed) {
        let ty = self.recorded(dst);
        let mut out = fill(self.width(dst), Region::default());
        let mut via = Via::new();
        for Placed { seg, value } in parts {
            let held = self.held(state, *value);
            add_via(&mut via, &held, *value);
            let part_width = self.width(*value);
            let at = self.layout.offsets(ty, std::slice::from_ref(seg), part_width);
            join_part(&mut out, &at, &held.positions);
        }
        self.put(state, dst, Regions::with_via(out, &via), changed);
    }

    /// `v`'s positions with what each of its references points at read from
    /// the storage it names now (RFC-0079 rule 3).
    fn current(&self, state: &State, v: ValueId) -> Vec<Region> {
        let mut held = self.held(state, v).positions;
        for (k, kind) in self.layout.kinds(self.recorded(v)).into_iter().enumerate() {
            let PositionKind::Ref(_, width) = kind else {
                continue;
            };
            for loan in held[k].loans.clone() {
                let view = self.storage_view(state, loan.storage, width);
                for (mine, theirs) in held[k + 1..k + 1 + width].iter_mut().zip(&view) {
                    mine.join_mut(theirs);
                }
            }
        }
        held
    }

    /// Every loan `values` hold, and every loan the storages those name hold
    /// now, to any depth.
    fn reach(&self, state: &State, values: &[ValueId]) -> Reach {
        let mut reach = Reach::default();
        for v in values {
            let held = self.held(state, *v);
            add_via(&mut reach.via, &held, *v);
            for region in &held.positions {
                reach.take(region);
            }
        }
        self.deepen(state, &mut reach);
        reach
    }

    /// What a closure reaches through its captures: every loan at its
    /// captures' position, and what the storages those name hold now, to any
    /// depth. A closure called through a reference to the storage holding
    /// it reaches its captures as they are held there, not the storage.
    fn captures_reach(&self, state: &State, callee: ValueId) -> Reach {
        let mut reach = Reach::default();
        let held = self.current(state, callee);
        add_via(&mut reach.via, &self.held(state, callee), callee);
        for (kind, region) in self.layout.kinds(self.recorded(callee)).into_iter().zip(&held) {
            if let PositionKind::Captures(_) = kind {
                reach.take(region);
            }
        }
        self.deepen(state, &mut reach);
        reach
    }

    /// `reach` taken on through what every storage it names holds now.
    fn deepen(&self, state: &State, reach: &mut Reach) {
        let mut at = 0;
        while at < reach.storages.len() {
            let storage = reach.storages[at];
            let holder = self.held(state, storage.holder());
            for k in self.content_range(storage) {
                reach.take(&holder.positions[k]);
            }
            at += 1;
        }
    }

    /// A call's result and writes (RFC-0079 rule 5), and every loan the call
    /// hands its callee. Every callee's flows are the ones its function type
    /// states: a direct call, an extern's (read off its Rust signature, rule
    /// 6) and a call through a function value alike.
    fn call(&self, state: &mut State, call: &Call, width: usize, changed: &mut Changed) -> (Regions, Reach) {
        let values: Vec<ValueId> = call.args.iter().chain(callee_value(call.callee)).copied().collect();
        let reach = self.reach(state, &values);
        let result = self.flowed(state, call, self.callee_fn(call.callee_ty).flows, width, changed);
        (result, reach)
    }

    /// What input `from` of a call holds now: an argument's positions, or
    /// everything the callee reaches through its captures in one position.
    fn call_input(&self, state: &State, call: &Call, from: FlowEnd) -> CallInput {
        match from {
            FlowEnd::Param(index) => match call.args.get(index) {
                Some(arg) => CallInput {
                    positions: self.current(state, *arg),
                    via: Via::from_iter([*arg]),
                },
                // A validated call has an argument for every parameter; a
                // flow past them stands for every argument.
                None => CallInput {
                    positions: vec![fold(
                        &call
                            .args
                            .iter()
                            .flat_map(|arg| self.current(state, *arg))
                            .collect::<Vec<_>>(),
                    )],
                    via: call.args.iter().copied().collect(),
                },
            },
            FlowEnd::Captures => match callee_value(call.callee) {
                Some(callee) => {
                    let reach = self.captures_reach(state, *callee);
                    CallInput {
                        positions: vec![reach.loans],
                        via: reach.via,
                    }
                }
                None => CallInput {
                    positions: Vec::new(),
                    via: Via::new(),
                },
            },
            FlowEnd::Result => unreachable!("a call's result is no input"),
        }
    }

    /// RFC-0079 rule 5: each output of the call is the join of the inputs
    /// its callee's flows name, as the call's arguments hold them now. The
    /// result is read before anything is written.
    fn flowed(&self, state: &mut State, call: &Call, flows: &Flows, width: usize, changed: &mut Changed) -> Regions {
        let arity = call.args.len();
        let mut out = fill(width, Region::default());
        let mut via = Via::new();
        for Source { from, alignment } in flows.into_end(FlowEnd::Result, arity) {
            let input = self.call_input(state, call, from);
            for v in input.via.iter() {
                add_via(&mut via, &self.held(state, *v), *v);
            }
            match &alignment {
                Alignment::Aligned if input.positions.len() == width => {
                    join_part(&mut out, &[0], &input.positions)
                }
                Alignment::Labelled(map) => {
                    let ends = self.labelled_ends(call.callee_ty, from, map);
                    join_labelled(&mut out, map, &ends, &input.positions)
                }
                Alignment::Aligned | Alignment::Any => join_part(&mut out, &[], &input.positions),
            }
        }
        let written: Vec<Written> = (0..arity)
            .map(FlowEnd::Param)
            .chain(callee_value(call.callee).map(|_| FlowEnd::Captures))
            .filter_map(|to| {
                let sources = flows.into_end(to, arity);
                if sources.is_empty() {
                    return None;
                }
                let mut region = Region::default();
                for Source { from, .. } in sources {
                    region.join_mut(&fold(&self.call_input(state, call, from).positions));
                }
                Some(Written { to, region })
            })
            .collect();
        for Written { to, region } in written {
            let into = match to {
                FlowEnd::Param(index) => call.args[index],
                FlowEnd::Captures => *callee_value(call.callee).expect("only a callee value captures"),
                FlowEnd::Result => unreachable!("the result is not written into"),
            };
            self.write_into(state, into, &region, changed);
        }
        Regions::with_via(out, &via)
    }

    /// `region` written through every `&mut` position of `v` and into what
    /// its captures name mutably, and joined into the pointee positions `v`
    /// itself carries.
    fn write_into(&self, state: &mut State, v: ValueId, region: &Region, changed: &mut Changed) {
        let unknown_place = WrittenPart {
            pointee_width: 0,
            at: &[],
            values: std::slice::from_ref(region),
        };
        let held = self.current(state, v);
        for (k, kind) in self.layout.kinds(self.recorded(v)).into_iter().enumerate() {
            let writable = match kind {
                PositionKind::Ref(Mutability::Mut, _) | PositionKind::Captures(_) => true,
                PositionKind::Ref(Mutability::Shared, _) | PositionKind::Lifetime | PositionKind::InFlight => false,
            };
            if !writable {
                continue;
            }
            let named: Vec<LoanStorage> = held[k]
                .loans
                .iter()
                .filter(|loan| loan.mutability == Mutability::Mut)
                .map(|loan| loan.storage)
                .collect();
            for storage in named {
                self.storage_write(state, storage, &unknown_place, changed);
            }
            if let PositionKind::Ref(Mutability::Mut, width) = kind {
                for position in k + 1..k + 1 + width {
                    self.join_at(state, v, position, region, changed);
                }
            }
        }
    }

    /// A reference: what it names, then the positions of what it points at.
    fn reference(&self, names: Region, pointee: Vec<Region>, via: &Via) -> Regions {
        Regions::with_via(std::iter::once(names).chain(pointee).collect(), via)
    }

    /// A `For` over a slice hands the body a reference into it, so the
    /// element holds the source's loan for as long as the loop runs, which
    /// is the terminator's own extent (RFC-0057 rule 2). An array's element
    /// is moved out of it, and holds the array's element positions.
    fn terminator_step(&self, term: &Terminator, state: &mut State, changed: &mut Changed) {
        let Terminator::For { source, stages, .. } = term else {
            return;
        };
        let Some(&target) = self.cfg.label_to_block.get(&stages.body()) else {
            return;
        };
        let Some(&element) = self.cfg.blocks[target.0].params.first() else {
            return;
        };
        let regions = match source {
            ForSource::Slice(slice) | ForSource::SliceMut(slice) => {
                let held = self.held(state, *slice).through(*slice);
                let names = Region {
                    loans: held.names().to_vec(),
                };
                let items = self.deref(state, *slice);
                let positions = std::iter::once(names)
                    .chain(reshape(&items, self.pointee_width(element)))
                    .collect();
                Regions::with_via(positions, &held.via)
            }
            ForSource::Array(array) => {
                let held = self.held(state, *array).through(*array);
                let positions = reshape(&held.positions, self.width(element));
                Regions::with_via(positions, &held.via)
            }
            ForSource::Range { .. } => return,
        };
        self.put(state, element, regions, changed);
    }

    fn step(&self, inst: &Inst, state: &mut State, changed: &mut Changed) {
        match &inst.kind {
            InstKind::Ref {
                dst,
                target,
                path,
                mutability,
            } => {
                let width = self.pointee_width(*dst);
                let regions = match target {
                    RefTarget::Var(s) | RefTarget::Param(s) => {
                        let storage = self.entry.storage(*s);
                        let names = Region {
                            loans: vec![Loan {
                                storage,
                                mutability: *mutability,
                                taken: *mutability,
                            }],
                        };
                        let held = self.held(state, *s).through(*s);
                        let contents: Vec<Region> = self
                            .content_range(storage)
                            .map(|k| held.positions[k].clone())
                            .collect();
                        let slot_ty = self.recorded(*s);
                        let contents_ty = match self.layout.types.kind(slot_ty) {
                            TyKind::Ref(_, inner) => inner.ty(),
                            _ => slot_ty,
                        };
                        let pointee = self.project(&contents, contents_ty, path, width);
                        self.reference(names, pointee, &held.via)
                    }
                    // A shared reborrow holds what it reborrows as shared
                    // (RFC-0029 rule 3).
                    RefTarget::Through(r) => {
                        let held = self.held(state, *r).through(*r);
                        let names = Region {
                            loans: held.names().to_vec(),
                        };
                        let names = match mutability {
                            Mutability::Shared => names.shared(),
                            Mutability::Mut => names,
                        };
                        let whole = self.deref(state, *r);
                        let pointee = self.project(&whole, self.pointee(*r), path, width);
                        self.reference(names, pointee, &held.via)
                    }
                };
                self.put(state, *dst, regions, changed);
            }
            InstKind::Take {
                dst, target, path, ..
            } => {
                let width = self.width(*dst);
                let regions = match target {
                    RefTarget::Var(s) | RefTarget::Param(s) => {
                        let held = self.held(state, *s).through(*s);
                        let part = self.project(&held.positions, self.recorded(*s), path, width);
                        Regions::with_via(part, &held.via)
                    }
                    RefTarget::Through(r) => {
                        let held = self.held(state, *r).through(*r);
                        let whole = self.deref(state, *r);
                        let part = self.project(&whole, self.pointee(*r), path, width);
                        Regions::with_via(part, &held.via)
                    }
                };
                self.put(state, *dst, regions, changed);
            }
            InstKind::Assign {
                target,
                path,
                value,
                ..
            } => {
                let held = self.held(state, *value).through(*value);
                let value_width = self.width(*value);
                let values = held.positions.clone();
                match target {
                    // RFC-0079 rule 4: nothing names a slot while it is
                    // assigned whole (RFC-0029), so the assign replaces.
                    RefTarget::Var(s) | RefTarget::Param(s) if path.is_empty() => {
                        let positions = reshape(&values, self.width(*s));
                        self.put(state, *s, Regions::with_via(positions, &held.via), changed);
                    }
                    RefTarget::Var(s) | RefTarget::Param(s) => {
                        let at = self.layout.offsets(self.recorded(*s), path, value_width);
                        let part = WrittenPart {
                            pointee_width: self.width(*s),
                            at: &at,
                            values: &values,
                        };
                        self.write_part(state, *s, self.entry.storage(*s), &part, changed);
                    }
                    RefTarget::Through(m) => {
                        let at = self.layout.offsets(self.pointee(*m), path, value_width);
                        let part = WrittenPart {
                            pointee_width: self.pointee_width(*m),
                            at: &at,
                            values: &values,
                        };
                        self.write_through(state, *m, &part, changed);
                    }
                }
            }
            InstKind::Spawn {
                dst,
                callee,
                callee_ty,
                args,
                ..
            } => {
                let call = Call {
                    callee,
                    callee_ty: *callee_ty,
                    args: args.clone(),
                };
                let (result, reach) = self.call(state, &call, self.handled_width(*dst), changed);
                let positions = std::iter::once(reach.loans).chain(result.positions).collect();
                self.put(state, *dst, Regions::with_via(positions, &reach.via), changed);
            }
            InstKind::FunctionCall {
                dst,
                callee,
                callee_ty,
                args,
                ..
            } => {
                let call = Call {
                    callee,
                    callee_ty: *callee_ty,
                    args: args.clone(),
                };
                let (result, _) = self.call(state, &call, self.width(*dst), changed);
                self.put(state, *dst, result, changed);
            }
            // RFC-0064 rule 5: a lambda that captures a reference is a
            // holder of that loan, so its one position is the join of what
            // it captured whatever its type says about the captures.
            InstKind::MakeClosure { dst, captures, .. } => {
                self.gather(state, *dst, captures, changed);
            }
            InstKind::TakeEffect { dst, src, .. } => {
                let held = self.held(state, *src).through(*src);
                let carried: Vec<Region> = (1..1 + self.handled_width(*src))
                    .map(|k| held.positions[k].clone())
                    .collect();
                let positions = reshape(&carried, self.width(*dst));
                self.put(state, *dst, Regions::with_via(positions, &held.via), changed);
            }
            InstKind::FieldGet {
                dst,
                object,
                field,
                rest,
            } => {
                let path: Vec<PathSeg> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(PathSeg::Field)
                    .collect();
                self.read_part(state, *dst, *object, &path, changed);
            }
            InstKind::ObjectGet { dst, object, key } => {
                self.read_part(state, *dst, *object, &[PathSeg::Field(*key)], changed);
            }
            InstKind::TupleIndex { dst, tuple, index } => {
                self.read_part(state, *dst, *tuple, &[PathSeg::Index(*index)], changed);
            }
            InstKind::ArrayIndex { dst, array, index } => {
                self.read_part(state, *dst, *array, &[PathSeg::Index(*index)], changed);
            }
            InstKind::UnwrapVariant { dst, src } => {
                self.read_part(state, *dst, *src, &[PathSeg::Payload], changed);
            }
            InstKind::FieldSet {
                dst,
                object,
                field,
                rest,
                value,
            } => {
                let path: Vec<PathSeg> = std::iter::once(*field)
                    .chain(rest.iter().copied())
                    .map(PathSeg::Field)
                    .collect();
                let object_held = self.held(state, *object).through(*object);
                let value_held = self.held(state, *value).through(*value);
                let value_width = self.width(*value);
                let mut positions = reshape(&object_held.positions, self.width(*dst));
                let at = self.layout.offsets(self.recorded(*dst), &path, value_width);
                join_part(&mut positions, &at, &value_held.positions);
                let mut regions = Regions::with_via(positions, &object_held.via);
                regions.join_mut(&Regions::with_via(Vec::new(), &value_held.via));
                self.put(state, *dst, regions, changed);
            }
            InstKind::MakeVariant { dst, tag, payload } => {
                let Some(payload) = payload else {
                    self.put(state, *dst, Regions::default(), changed);
                    return;
                };
                let seg = match self.layout.types.kind(self.recorded(*dst)) {
                    TyKind::Enum { .. } => PathSeg::Field(*tag),
                    _ => PathSeg::Payload,
                };
                let parts = [Placed {
                    seg,
                    value: *payload,
                }];
                self.build_from(state, *dst, &parts, changed);
            }
            InstKind::ArrayBegin { dst, .. } => self.build_from(state, *dst, &[], changed),
            InstKind::ArrayPush { dst, array, value } => {
                let array_held = self.held(state, *array).through(*array);
                let value_held = self.held(state, *value).through(*value);
                let value_width = self.width(*value);
                let mut positions = reshape(&array_held.positions, self.width(*dst));
                let at = self
                    .layout
                    .offsets(self.recorded(*dst), &[PathSeg::Index(0)], value_width);
                join_part(&mut positions, &at, &value_held.positions);
                let mut regions = Regions::with_via(positions, &array_held.via);
                regions.join_mut(&Regions::with_via(Vec::new(), &value_held.via));
                self.put(state, *dst, regions, changed);
            }
            InstKind::MakeTuple { dst, elements } => {
                let parts: Vec<Placed> = elements
                    .iter()
                    .enumerate()
                    .map(|(index, value)| Placed {
                        seg: PathSeg::Index(index),
                        value: *value,
                    })
                    .collect();
                self.build_from(state, *dst, &parts, changed);
            }
            InstKind::MakeObject { dst, fields } => {
                let parts: Vec<Placed> = fields
                    .iter()
                    .map(|(name, value)| Placed {
                        seg: PathSeg::Field(*name),
                        value: *value,
                    })
                    .collect();
                self.build_from(state, *dst, &parts, changed);
            }
            // A slice is a borrow of what its container names, with the
            // container's elements as its own (RFC-0047).
            InstKind::AsSlice { dst, container, .. } => {
                let held = self.held(state, *container).through(*container);
                let names = Region {
                    loans: held.names().to_vec(),
                };
                let elements = reshape(&self.deref(state, *container), self.pointee_width(*dst));
                self.put(state, *dst, self.reference(names, elements, &held.via), changed);
            }
            InstKind::Index {
                dst, slice, mode, ..
            } => {
                let held = self.held(state, *slice).through(*slice);
                let element = self.deref(state, *slice);
                let regions = match mode {
                    IndexMode::Ref => {
                        let names = Region {
                            loans: held.names().to_vec(),
                        };
                        let pointee = reshape(&element, self.pointee_width(*dst));
                        self.reference(names, pointee, &held.via)
                    }
                    IndexMode::Copy => {
                        Regions::with_via(reshape(&element, self.width(*dst)), &held.via)
                    }
                };
                self.put(state, *dst, regions, changed);
            }
            InstKind::IndexSet { slice, value, .. } => {
                let values = self.held(state, *value).positions;
                let part = WrittenPart {
                    pointee_width: self.pointee_width(*slice),
                    at: &[0],
                    values: &values,
                };
                self.write_through(state, *slice, &part, changed);
            }
            InstKind::StructuralClone { dst, src, .. } => {
                let held = self.held(state, *src).through(*src);
                let from = match self.layout.types.kind(self.recorded(*src)) {
                    TyKind::Ref(..) => self.deref(state, *src),
                    _ => held.positions.clone(),
                };
                let positions = reshape(&from, self.width(*dst));
                self.put(state, *dst, Regions::with_via(positions, &held.via), changed);
            }
            InstKind::Const { .. }
            | InstKind::ConstStr { .. }
            | InstKind::LoadFunction { .. }
            | InstKind::Fetch { .. }
            | InstKind::Commit { .. }
            | InstKind::Undef { .. }
            | InstKind::Poison { .. }
            | InstKind::Merge { .. }
            | InstKind::Drop { .. }
            | InstKind::Nop
            | InstKind::StringAppend { .. } => {}
            kind => {
                let uses = inst_info::uses(kind);
                for dst in inst_info::defs(kind) {
                    self.gather(state, dst, &uses, changed);
                }
            }
        }
    }

    fn read_part(&self, state: &mut State, dst: ValueId, src: ValueId, path: &[PathSeg], changed: &mut Changed) {
        let held = self.held(state, src).through(src);
        let part = self.project(&held.positions, self.recorded(src), path, self.width(dst));
        self.put(state, dst, Regions::with_via(part, &held.via), changed);
    }

    /// `*m = v` (RFC-0079 rule 4): the part joined into every storage `m`
    /// names, and into `m`'s own tail.
    fn write_through(&self, state: &mut State, m: ValueId, part: &WrittenPart, changed: &mut Changed) {
        for loan in self.held(state, m).names().to_vec() {
            self.storage_write(state, loan.storage, part, changed);
        }
        let tail: Vec<usize> = part.at.iter().map(|offset| offset + 1).collect();
        let own = WrittenPart {
            pointee_width: 1 + part.pointee_width,
            at: &tail,
            values: part.values,
        };
        self.write_part(state, m, LoanStorage::Local(m), &own, changed);
    }

    /// The part joined into `v`'s own positions, without a loan on `storage`.
    fn write_part(&self, state: &mut State, v: ValueId, storage: LoanStorage, part: &WrittenPart, changed: &mut Changed) {
        let before = self.held(state, v);
        let mut positions = before.positions.clone();
        let values: Vec<Region> = part.values.iter().map(|r| r.clone().without(storage)).collect();
        join_part(&mut positions, part.at, &values);
        self.read_only_shared(v, &mut positions);
        let after = Regions {
            positions,
            via: before.via.clone(),
        };
        if after != before {
            state.set(v, after);
            changed.push(v);
        }
    }
}

/// A call as the analysis reads it.
struct Call<'i> {
    callee: &'i Callee,
    callee_ty: Ty,
    args: Vec<ValueId>,
}

/// An input of a call: its positions, and the values it was read from.
struct CallInput {
    positions: Vec<Region>,
    via: Via,
}

/// An output of a call and the join of the inputs it takes.
struct Written {
    to: FlowEnd,
    region: Region,
}

/// The parts of a callee's function type a call reads.
struct CalleeFn<'t> {
    params: &'t [Ty],
    ret: Ty,
    flows: &'t Flows,
}

impl RegionAnalysis<'_> {
    /// The function type a call reads its flows and its ends' types off. A
    /// callee value reached through a reference is called as the function
    /// it points at.
    ///
    /// # Panics
    /// `callee_ty` is neither a function type nor a reference to one: the
    /// checker types every callee so.
    fn callee_fn(&self, callee_ty: Ty) -> CalleeFn<'_> {
        let types = self.layout.types;
        let fn_ty = match types.kind(callee_ty) {
            TyKind::Ref(_, inner) => inner.ty(),
            _ => callee_ty,
        };
        let TyKind::Fn { params, ret, flows, .. } = types.kind(fn_ty) else {
            panic!("a call's callee type is a function type or a reference to one")
        };
        CalleeFn {
            params,
            ret: *ret,
            flows: match flows {
                FlowTerm::Known(flows) => flows,
                FlowTerm::Var(var) => match *var {},
            },
        }
    }

    /// The positions each segment of a `Labelled` flow's two ends covers at
    /// a call: the output's (the result) and the input's, each read off the
    /// end's declared type beside the concrete type the callee's function
    /// type gives it.
    ///
    /// # Panics
    /// The flow's input is not a parameter, or its map names a segment its
    /// ends do not have: a `Labelled` flow is read off an extern's
    /// signature, whose ends are its parameters and its result.
    fn labelled_ends(&self, callee_ty: Ty, from: FlowEnd, map: &Labelled) -> LabelledEnds {
        let Layout { types, decls } = self.layout;
        let CalleeFn { params, ret, .. } = self.callee_fn(callee_ty);
        let FlowEnd::Param(index) = from else {
            panic!("a Labelled flow's input is a parameter of the extern it is read off")
        };
        let from_ty = *params
            .get(index)
            .expect("a Labelled flow's input is a parameter of its function type");
        let poly = decls.types();
        let to_decl = map.declared.end_ty(decls, FlowEnd::Result);
        let from_decl = map.declared.end_ty(decls, from);
        let ends = LabelledEnds {
            to: segment_positions(poly, to_decl, types, ret, decls),
            from: segment_positions(poly, from_decl, types, from_ty, decls),
            to_segments: segments(poly, decls, to_decl),
            from_segments: segments(poly, decls, from_decl),
        };
        assert!(
            map.map.len() == ends.to.len() && map.map.iter().flatten().all(|segment| *segment < ends.from.len()),
            "a Labelled flow's map names the segments of its ends"
        );
        ends
    }
}

/// Where each segment of both ends of a `Labelled` flow lies at a call, and
/// what each segment is.
struct LabelledEnds {
    to: Vec<Range<usize>>,
    from: Vec<Range<usize>>,
    to_segments: Vec<Segment>,
    from_segments: Vec<Segment>,
}

/// Each output segment joins exactly the input segments its map names,
/// position by position where both are one type variable's
/// (`Segment::aligned_with`).
fn join_labelled(out: &mut [Region], map: &Labelled, ends: &LabelledEnds, input: &[Region]) {
    for (at, takes) in map.map.iter().enumerate() {
        let to = &ends.to[at];
        for &segment in takes.iter() {
            let from = &ends.from[segment];
            let values = &input[from.clone()];
            match ends.to_segments[at].aligned_with(ends.from_segments[segment]) {
                true => {
                    for (mine, region) in out[to.clone()].iter_mut().zip(values) {
                        mine.join_mut(region);
                    }
                }
                false => {
                    let all = fold(values);
                    for mine in &mut out[to.clone()] {
                        mine.join_mut(&all);
                    }
                }
            }
        }
    }
}

fn callee_value(callee: &Callee) -> Option<&ValueId> {
    match callee {
        Callee::Indirect(f) => Some(f),
        Callee::Direct(_) | Callee::Extern { .. } => None,
    }
}

fn join_part(positions: &mut [Region], at: &[usize], values: &[Region]) {
    match at.is_empty() {
        true => {
            let all = fold(values);
            positions.iter_mut().for_each(|mine| {
                mine.join_mut(&all);
            });
        }
        false => {
            for offset in at {
                for (k, region) in values.iter().enumerate() {
                    if let Some(mine) = positions.get_mut(offset + k) {
                        mine.join_mut(region);
                    }
                }
            }
        }
    }
}

impl DataflowAnalysis for RegionAnalysis<'_> {
    type Key = ValueId;
    type Domain = Regions;

    fn transfer_inst(&self, inst: &Inst, state: &mut State) {
        self.step(inst, state, &mut Changed::new());
    }

    fn terminator_uses(&self, term: &Terminator, state: &mut State) {
        self.terminator_step(term, state, &mut Changed::new());
    }

    fn propagate_forward(
        &self,
        source_exit: &State,
        params: &[ValueId],
        first: usize,
        args: &[ValueId],
        target_entry: &mut State,
    ) -> bool {
        let mut changed = target_entry.join_from(source_exit);
        for (param, arg) in params.iter().skip(first).zip(args) {
            let width = self.width(*param);
            if width == 0 {
                continue;
            }
            let from = self.held(source_exit, *arg).through(*arg);
            let incoming = Regions {
                positions: reshape(&from.positions, width),
                via: from.via,
            };
            let mut regions = target_entry.get(*param);
            if regions.join_mut(&incoming) {
                self.read_only_shared(*param, &mut regions.positions);
                target_entry.set(*param, regions);
                changed = true;
            }
        }
        changed
    }

    fn propagate_backward(&self, _: &State, _: &[ValueId], _: usize, _: &[ValueId], _: &mut State) {
        unreachable!("regions flow forward")
    }
}

// -- The result -----------------------------------------------------

/// The regions of the values of the CFG `cfg`.
pub struct Loans<'cfg> {
    cfg: &'cfg CfgBody,
    layout: Layout<'cfg>,
    regions: FxHashMap<ValueId, Regions>,
    entry: Vec<State>,
    given: Vec<Vec<Given>>,
    storage: EntryStorage,
}

/// The regions an instruction gave a value it wrote: one it defines, or a
/// storage written through a reference or by a call. A block's entry state
/// and these give the regions before each of its instructions.
struct Given {
    at: usize,
    value: ValueId,
    regions: Regions,
}

/// What a loan an output holds is, from inside the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    Local(ValueId),
    Input(HeldInput),
}

/// A position of one of the body's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldInput {
    pub end: FlowEnd,
    pub position: usize,
}

static NOTHING: Regions = Regions {
    positions: Vec::new(),
    via: Via::new(),
};

impl<'cfg> Loans<'cfg> {
    pub fn build(cfg: &'cfg CfgBody, types: &'cfg Types<Concrete>, decls: &'cfg Declarations) -> Self {
        let layout = Layout::new(types, decls);
        let analysis = RegionAnalysis {
            layout,
            val_types: &cfg.val_types,
            cfg,
            entry: EntryStorage::of(cfg),
        };
        let mut entry = DataflowState::new();
        for value in cfg.entry_defs() {
            entry.set(value, entry_regions(layout, &analysis.entry, value, analysis.recorded(value)));
        }
        let result = forward_analysis(cfg, &analysis, entry);
        // The fixpoint's block exits are not joined here. The walk below
        // steps each block from its entry again and ends at its exit, so an
        // exit adds nothing; its regions are equal to the walk's but built
        // separately, and joining them cost each value the size of its
        // `Via`, quadratic over an array literal's pushes.
        let mut regions: FxHashMap<ValueId, Regions> = FxHashMap::default();
        for state in &result.block_entry {
            for (v, r) in &state.values {
                regions.entry(*v).or_default().join_mut(r);
            }
        }
        let mut given: Vec<Vec<Given>> = Vec::with_capacity(cfg.blocks.len());
        for (block, entry) in cfg.blocks.iter().zip(&result.block_entry) {
            let mut state = entry.clone();
            let mut block_given = Vec::new();
            for (at, inst) in block.insts.iter().enumerate() {
                let mut changed = Changed::new();
                analysis.step(inst, &mut state, &mut changed);
                changed.sort_unstable();
                changed.dedup();
                for value in changed {
                    let now = state.get(value);
                    regions.entry(value).or_default().join_mut(&now);
                    block_given.push(Given {
                        at,
                        value,
                        regions: now,
                    });
                }
            }
            let mut changed = Changed::new();
            analysis.terminator_step(&block.terminator, &mut state, &mut changed);
            for value in changed {
                regions.entry(value).or_default().join_mut(&state.get(value));
            }
            given.push(block_given);
        }
        Self {
            cfg,
            layout,
            regions,
            entry: result.block_entry,
            given,
            storage: analysis.entry,
        }
    }

    /// The CFG these regions are of.
    pub fn cfg(&self) -> &'cfg CfgBody {
        self.cfg
    }

    /// # Panics
    /// `v` has no recorded type: every IR value has one.
    pub fn recorded(&self, v: ValueId) -> Ty {
        self.cfg.ty_of(v)
    }

    /// The positions of the CFG's types.
    pub fn layout(&self) -> Layout<'cfg> {
        self.layout
    }

    /// The regions at the entry of `block`, to be walked through its
    /// instructions in order.
    pub fn at_entry(&self, block: BlockIdx) -> RegionsAt<'_> {
        RegionsAt {
            state: self.entry[block.0].clone(),
            given: &self.given[block.0],
            next: 0,
            at: 0,
        }
    }

    /// Every region a value holds anywhere in the body. A value no step of
    /// the dataflow gave regions is the lattice's bottom, as
    /// `DataflowState::get` reads an unset value (`analysis/dataflow.rs:31`):
    /// it holds no loan.
    pub fn regions(&self, value: ValueId) -> &Regions {
        self.regions.get(&value).unwrap_or(&NOTHING)
    }

    /// The storage a reference points at, anywhere in the body.
    pub fn names(&self, value: ValueId) -> &[Loan] {
        self.regions(value).names()
    }

    /// Every loan a value holds in any position, anywhere in the body.
    pub fn holds(&self, value: ValueId) -> impl Iterator<Item = &Loan> {
        self.regions(value).holds()
    }

    /// The storage `&slot` names.
    pub fn storage_of(&self, slot: ValueId) -> LoanStorage {
        self.storage.storage(slot)
    }

    /// What a loan an output of the body holds is, as the body's inputs
    /// name it (RFC-0079 rule 9). A loan on a parameter's own slot, which a
    /// by-value parameter's reference names, is the body's storage: the
    /// slot is gone once the body returns.
    pub fn held(&self, loan: &Loan) -> Held {
        match loan.storage {
            LoanStorage::Local(value) => Held::Local(value),
            LoanStorage::Param { index, value } => match self.layout.types.kind(self.recorded(value)) {
                TyKind::Ref(..) | TyKind::Fn { .. } => Held::Input(HeldInput {
                    end: FlowEnd::Param(index),
                    position: 0,
                }),
                _ => Held::Local(value),
            },
            LoanStorage::Outside(Outside { entry, position }) => match self.storage.params.get(&entry) {
                Some(index) => Held::Input(HeldInput {
                    end: FlowEnd::Param(*index),
                    position,
                }),
                None => Held::Input(HeldInput {
                    end: FlowEnd::Captures,
                    position: 0,
                }),
            },
        }
    }

    /// The loans the body wrote into what an entry definition names from
    /// outside the body: every loan its positions hold anywhere in the body
    /// that it did not start with (RFC-0079 rule 4).
    pub fn written_into(&self, entry: ValueId) -> Vec<Loan> {
        let own = self.storage.storage(entry);
        let mut written: Vec<Loan> = Vec::new();
        for loan in self.holds(entry) {
            let started = loan.storage == own
                || matches!(loan.storage, LoanStorage::Outside(Outside { entry: at, .. }) if at == entry);
            if !started && !written.contains(loan) {
                written.push(*loan);
            }
        }
        written
    }

    fn add_holds(&self, effect: &mut StorageEffect, value: ValueId) {
        for loan in self.holds(value) {
            effect.add(*loan);
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
            // A callee may read or write through any reference an argument
            // holds, in any position.
            InstKind::FunctionCall { args, callee, .. } | InstKind::Spawn { args, callee, .. } => {
                for a in args {
                    self.add_holds(&mut effect, *a);
                }
                // A lambda called uses what it captured at the captures'
                // own mutability, as an argument would: the callee is the
                // lambda, or a reference to the storage holding it.
                if let Callee::Indirect(f) = callee {
                    self.add_holds(&mut effect, *f);
                    for loan in self.holds(*f) {
                        if let Some(slot) = loan.storage.slot() {
                            self.add_holds(&mut effect, slot);
                        }
                    }
                }
            }
            InstKind::TakeEffect { src, .. } => self.add_holds(&mut effect, *src),
            // A slice is a borrow of its container taken with the slice's
            // own mutability; indexing touches the run that borrow names
            // (RFC-0047).
            InstKind::AsSlice {
                container,
                mutability,
                ..
            } => self.touch_named(&mut effect, *container, *mutability),
            InstKind::Index { slice, .. } => {
                self.touch_named(&mut effect, *slice, Mutability::Shared)
            }
            InstKind::IndexSet { slice, .. } => {
                self.touch_named(&mut effect, *slice, Mutability::Mut)
            }
            InstKind::StringAppend { target, .. } => {
                self.touch_named(&mut effect, *target, Mutability::Mut)
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

    /// The storage `values` keep alive, which a use of a holder reaches
    /// through the loans it holds: the half of [`Self::uses_with_storage`] a
    /// reader that has its own use list needs.
    pub fn storage_behind(&self, values: &[ValueId]) -> SmallVec<[ValueId; 4]> {
        let mut storage: SmallVec<[ValueId; 4]> = SmallVec::new();
        for value in values {
            self.reachable_storage(*value, &mut storage);
        }
        storage
    }

    fn reachable_storage(&self, value: ValueId, out: &mut SmallVec<[ValueId; 4]>) {
        let mut work: Vec<ValueId> = vec![value];
        let mut seen: Vec<ValueId> = vec![value];
        while let Some(at) = work.pop() {
            for loan in self.holds(at) {
                let holder = loan.storage.holder();
                if let Some(slot) = loan.storage.slot()
                    && !out.contains(&slot)
                {
                    out.push(slot);
                }
                if !seen.contains(&holder) {
                    seen.push(holder);
                    work.push(holder);
                }
            }
        }
    }

    /// An access through a reference is bounded by that reference's own
    /// mutability. The interpreter is where that bound is kept:
    /// `ops::storage::take_through` reads the referent through a `&Value`
    /// and copies the word out of it, while `take_var` reaches its slot by
    /// `&mut` and may empty it.
    ///
    /// A `Var`, and a parameter held by value, is the place itself; a
    /// reference parameter's slot is what it lends, reached through it.
    fn touch(&self, effect: &mut StorageEffect, target: &RefTarget, mutability: Mutability) {
        match target {
            RefTarget::Var(s) => effect.add_placed(self.storage_of(*s), mutability),
            RefTarget::Param(s) => match self.layout.types.kind(self.recorded(*s)) {
                TyKind::Ref(..) | TyKind::Fn { .. } => effect.add(Loan {
                    storage: self.storage_of(*s),
                    mutability,
                    taken: mutability,
                }),
                _ => effect.add_placed(self.storage_of(*s), mutability),
            },
            RefTarget::Through(r) => self.touch_named(effect, *r, mutability),
        }
    }

    /// As `touch`, for the storage `reference` names.
    fn touch_named(&self, effect: &mut StorageEffect, reference: ValueId, mutability: Mutability) {
        for loan in self.names(reference) {
            let bounded = match (mutability, loan.mutability) {
                (Mutability::Mut, Mutability::Mut) => Mutability::Mut,
                (Mutability::Shared, _) | (_, Mutability::Shared) => Mutability::Shared,
            };
            effect.add(Loan {
                mutability: bounded,
                ..*loan
            });
        }
    }
}

/// An entry definition's positions (RFC-0079 rule 3): a parameter or capture
/// of reference or function type names itself first, as RFC-0064 rule 1
/// states, and every other position names one outside storage.
fn entry_regions(layout: Layout<'_>, storage: &EntryStorage, value: ValueId, ty: Ty) -> Regions {
    let names_itself = matches!(layout.types.kind(ty), TyKind::Ref(..) | TyKind::Fn { .. });
    let positions = layout
        .kinds(ty)
        .into_iter()
        .enumerate()
        .map(|(position, kind)| {
            let mutability = match kind {
                PositionKind::Ref(mutability, _) => mutability,
                PositionKind::Captures(holds) => match captured(holds) {
                    Some(mutability) => mutability,
                    None => return Region::default(),
                },
                PositionKind::Lifetime | PositionKind::InFlight => Mutability::Shared,
            };
            let loan = match position == 0 && names_itself {
                true => storage.loan(value, mutability),
                false => Loan {
                    storage: LoanStorage::Outside(Outside {
                        entry: value,
                        position,
                    }),
                    mutability,
                    taken: mutability,
                },
            };
            Region { loans: vec![loan] }
        })
        .collect();
    Regions {
        positions,
        via: Via::new(),
    }
}

/// The regions a body's values hold before one instruction of a block: what
/// a holder lends where it is used, where `Loans::regions` is what it lends
/// anywhere in the body. A storage is given loans by the assignments that
/// reach it, so before the first of them it lends only what it held
/// (RFC-0064: a reference's extent is per instruction).
pub struct RegionsAt<'a> {
    state: State,
    given: &'a [Given],
    next: usize,
    at: usize,
}

impl RegionsAt<'_> {
    pub fn regions(&self, value: ValueId) -> Regions {
        self.state.get(value)
    }

    /// Past the instruction the walk is before.
    pub fn pass(&mut self) {
        while let Some(given) = self
            .given
            .get(self.next)
            .filter(|given| given.at == self.at)
        {
            self.state.set(given.value, given.regions.clone());
            self.next += 1;
        }
        self.at += 1;
    }
}

/// The loan a value of `ty` holds on what it names, and its mutability: what
/// a body's entry definition starts, and what a read of the value out of a
/// storage takes again.
///
/// A parameter or capture of reference type names storage outside the body,
/// and inside the body that storage is the entry itself (RFC-0029). A
/// parameter that is a lambda holding a loan names storage outside the body
/// the same way (RFC-0064 rule 1), so it starts a loan too.
pub fn held_loan(types: &Types<Concrete>, ty: Ty) -> Option<Mutability> {
    match types.kind(ty) {
        TyKind::Ref(mutability, _) => Some(*mutability),
        TyKind::Fn { holds, .. } => captured(*holds),
        _ => None,
    }
}

/// The loan a function value's captures hold, as its `holds` states it.
fn captured(holds: Holds) -> Option<Mutability> {
    match holds {
        Holds::Nothing => None,
        Holds::Shared => Some(Mutability::Shared),
        Holds::Mut => Some(Mutability::Mut),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::decl::{Core, LoopMarkers, RawDeclarations};
    use crate::ir::{DebugInfo, MirBody};
    use crate::ty::TypeArg;
    use acvus_ast::Span;
    use acvus_utils::{Interner, LocalFactory};

    fn empty_declarations(interner: &Interner) -> Declarations {
        Declarations::verify(RawDeclarations {
            lang_impls: Vec::new(),
            types: Types::new(),
            functions: Vec::new(),
            user_types: Vec::new(),
            casts: Vec::new(),
            conversions: Vec::new(),
            contexts: Vec::new(),
            inputs: Vec::new(),
            loop_markers: LoopMarkers::default(),
            core: Core::of(interner),
        })
        .unwrap()
    }

    fn shared_string(types: &mut Types<Concrete>) -> Ty {
        let string_ty = types.intern(TyKind::String);
        types.intern(TyKind::Ref(Mutability::Shared, TypeArg::Uniform(string_ty)))
    }

    /// Two reference parameters, and a reborrow of the second returned.
    fn two_parameters(types: &mut Types<Concrete>) -> CfgBody {
        let i = Interner::new();
        let mut factory = LocalFactory::<ValueId>::new();
        let (p0, p1, dst) = (factory.next(), factory.next(), factory.next());
        let ref_ty = shared_string(types);
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
            val_types: [(p0, ref_ty), (p1, ref_ty), (dst, ref_ty)]
                .into_iter()
                .collect(),
            params: vec![(i.intern("a"), p0), (i.intern("b"), p1)],
            captures: vec![],
            order_param: None,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 2,
            demoted_diamonds: Default::default(),
        })
    }

    #[test]
    fn a_param_loan_names_the_parameter_at_its_index() {
        let interner = Interner::new();
        let mut types = Types::new();
        let decls = empty_declarations(&interner);
        let cfg = two_parameters(&mut types);
        let loans = Loans::build(&cfg, &types, &decls);
        let mut seen = 0;
        for value in cfg.val_types.keys() {
            for loan in loans.holds(*value) {
                let Some(index) = loan.storage.param() else {
                    panic!("a body of only parameters holds {:?}", loan.storage);
                };
                assert_eq!(Some(cfg.params[index].1), loan.storage.slot());
                seen += 1;
            }
        }
        assert!(seen >= cfg.params.len(), "every parameter starts a region");
    }

    #[test]
    fn the_result_holds_the_parameter_it_reborrows() {
        let interner = Interner::new();
        let mut types = Types::new();
        let decls = empty_declarations(&interner);
        let cfg = two_parameters(&mut types);
        let loans = Loans::build(&cfg, &types, &decls);
        let returned = cfg.params[1].1;
        let dst = *cfg
            .val_types
            .keys()
            .find(|v| **v != cfg.params[0].1 && **v != returned)
            .expect("the reborrow");
        let held: Vec<Held> = loans.holds(dst).map(|loan| loans.held(loan)).collect();
        assert_eq!(
            held,
            [Held::Input(HeldInput {
                end: FlowEnd::Param(1),
                position: 0,
            })]
        );
    }
}
