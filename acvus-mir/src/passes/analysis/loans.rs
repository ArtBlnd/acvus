//! Loans (ownership §5): which storage each value may name at each position
//! of its type, and what each storage's content holds, at every point of a
//! body: the dense engine, forward. The borrow check runs before promotion,
//! when every binding is a slot, so the content of a slot is tracked per
//! point: an `Assign` of a whole place replaces what the place held.
//!
//! - A storage is a slot, a by-value parameter's own storage (what
//!   `RefTarget::Param` names), or one outside storage per position of each
//!   parameter's and capture's type. An outside storage is distinct from
//!   every other storage, since its callers were checked.
//! - A `Ref` names its place's storage, with its mutability; a reborrow
//!   names what its source names. `Assign` and `Take` move loans between a
//!   value and a storage's content. A call's result and written arguments
//!   take loans only as its function type's flows state (extern-fn §4). An
//!   edge joins its arguments into its target's parameters.
//! - A call whose callee's impl declares its access (extern-fn §3) touches
//!   only the places that access states, in their modes: it reads and
//!   writes those alone, and writes loans into no other. A call without one
//!   touches all its arguments reach, and writes as its flows state.
//! - A reference whose pointee has as many positions as the storage it names
//!   covers that storage whole, and reads and writes through it keep
//!   positions aligned. One with fewer names a part the loan does not
//!   locate, so a read through it takes every loan of the storage's content
//!   at each position, and a write joins into every position.
//!
//! Cost: a block's state only grows, and the lattice is finite: a set of
//! loans per position of each value and each storage, with loans over the
//! body's storages. So the engine runs O(h · (blocks + edges)) block
//! transfers, h being the number of positions times the number of loans; a
//! transfer costs O(its instructions' operands · positions · loans). The
//! widening is the join. A block's state holds only the values whose type
//! has a position, so its clones and joins cost those values' positions and
//! the storages', not the body's values.
//!
//! passes-analyses.md.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use rustc_hash::FxHashMap;

use crate::decl::declared::{Access, Element, Touch};
use crate::decl::{Declarations, segments};
use crate::flows::segment_positions;
use crate::ir::cfg::{Aspect, Block, BlockId, Body, BodyId, Definition, Edge, Inst, Module, Op, Site, Terminator};
use crate::ir::{Callee, ForSource, IndexMode, PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, Dominators, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::ty::{
    Alignment, Concrete, FlowEnd, Holds, Labelled, Mutability, PositionKind, Source, Ty, TyKind, Types, copies,
    position_kinds, positions,
};

/// A storage a loan names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Storage {
    Slot(ValueId),
    /// A by-value parameter's own storage.
    Param(ValueId),
    /// What position `position` of the parameter or capture `of` names,
    /// outside the body.
    Outside { of: ValueId, position: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mode {
    Shared,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Loan {
    pub storage: Storage,
    pub mode: Mode,
}

/// The loans one position may hold.
pub type LoanSet = BTreeSet<Loan>;

/// A storage whole, or one element of it named by a value, which a call
/// with a declared access touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TouchedPlace {
    pub storage: Storage,
    pub element: Option<ElementAt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementAt {
    /// The entry at this key.
    Key(ValueId),
    /// The element at this `u64` index.
    Index(ValueId),
}

impl TouchedPlace {
    pub fn whole(storage: Storage) -> TouchedPlace {
        TouchedPlace { storage, element: None }
    }

    /// Whether the two may be one place, or one hold the other, on some
    /// run of `body`. Two entries may be one place whatever their keys:
    /// whether two keys name one place is what the key type's `eq` says
    /// (extern-fn §3), and nothing here evaluates it. Two elements are
    /// distinct where their indices are two different integer constants.
    pub fn may_overlap(&self, other: &TouchedPlace, body: &Body) -> bool {
        if self.storage != other.storage {
            return false;
        }
        match (self.element, other.element) {
            (Some(ElementAt::Index(a)), Some(ElementAt::Index(b))) => {
                a == b || !matches!((constant_index(body, a), constant_index(body, b)), (Some(a), Some(b)) if a != b)
            }
            (None, _) | (_, None) | (Some(ElementAt::Key(_)), _) | (_, Some(ElementAt::Key(_))) => true,
        }
    }
}

/// The integer constant that defines `value`, where one does.
fn constant_index(body: &Body, value: ValueId) -> Option<i128> {
    let Definition::Inst(made) = body.definition(value) else {
        return None;
    };
    match body.inst(made).op() {
        Op::Const { value } => match value.desugared() {
            acvus_ast::Literal::Int(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

pub struct Loans {
    values: FxHashMap<ValueId, Vec<LoanSet>>,
    touched: FxHashMap<Site, Touched>,
    before: FxHashMap<Site, BTreeMap<Storage, Vec<LoanSet>>>,
}

/// The storages one site reads and writes, and the places within them.
/// A site without a declared access touches its storages whole.
#[derive(Debug, Clone, Default)]
struct Touched {
    reads: BTreeSet<Storage>,
    writes: BTreeSet<Storage>,
    read_places: BTreeSet<TouchedPlace>,
    written_places: BTreeSet<TouchedPlace>,
    by_access: bool,
}

impl Touched {
    fn whole(reads: BTreeSet<Storage>, writes: BTreeSet<Storage>) -> Touched {
        Touched {
            read_places: reads.iter().copied().map(TouchedPlace::whole).collect(),
            written_places: writes.iter().copied().map(TouchedPlace::whole).collect(),
            reads,
            writes,
            by_access: false,
        }
    }
}

impl Loans {
    /// The loans position `position` of `value` may hold, over every point
    /// of the body. A value no run defines holds none. Cost: O(1).
    ///
    /// # Panics
    /// `value` is a slot or not a value of the body, or `position` is not a
    /// position of its type.
    pub fn of(&self, value: ValueId, position: usize) -> &LoanSet {
        let positions = self
            .values
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is a value of the body the loans cover, and not a slot"));
        positions
            .get(position)
            .unwrap_or_else(|| panic!("{value:?} has {} positions, not {position}", positions.len()))
    }

    /// Every loan any position of `value` may hold. Cost: O(its positions ·
    /// loans).
    pub fn all_of(&self, value: ValueId) -> LoanSet {
        (0..self.positions_of(value)).flat_map(|position| self.of(value, position).iter().copied()).collect()
    }

    /// How many positions `value`'s type has. Cost: O(1).
    pub fn positions_of(&self, value: ValueId) -> usize {
        self.values
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is a value of the body the loans cover, and not a slot"))
            .len()
    }

    /// The storages the instruction or terminator at `site` reads. Cost:
    /// O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn reads(&self, site: Site) -> &BTreeSet<Storage> {
        &self.touched(site).reads
    }

    /// The storages the instruction or terminator at `site` writes. Cost:
    /// O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn writes(&self, site: Site) -> &BTreeSet<Storage> {
        &self.touched(site).writes
    }

    /// The places the instruction or terminator at `site` reads: those its
    /// callee's declared access states, else its storages whole. Cost:
    /// O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn read_places(&self, site: Site) -> &BTreeSet<TouchedPlace> {
        &self.touched(site).read_places
    }

    /// The places the instruction or terminator at `site` writes, as
    /// `read_places`. Cost: O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn written_places(&self, site: Site) -> &BTreeSet<TouchedPlace> {
        &self.touched(site).written_places
    }

    /// Whether what `site` touches is its callee's declared access, so that
    /// it touches no other place through its arguments. Cost: O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn by_access(&self, site: Site) -> bool {
        self.touched(site).by_access
    }

    /// What `storage`'s content may hold, at each position of its type, just
    /// before `site` runs. A storage a run has not reached holds none. Cost:
    /// O(log storages).
    ///
    /// # Panics
    /// `site` is not a site of the body that a run reaches, or `storage` is
    /// not a storage of it.
    pub fn content_before(&self, site: Site, storage: Storage) -> &[LoanSet] {
        self.before
            .get(&site)
            .unwrap_or_else(|| panic!("{site:?} is a reached site of the body the loans cover"))
            .get(&storage)
            .unwrap_or_else(|| panic!("{storage:?} is a storage of the body the loans cover"))
    }

    fn touched(&self, site: Site) -> &Touched {
        self.touched.get(&site).unwrap_or_else(|| panic!("{site:?} is a site of the body the loans cover"))
    }
}

impl Analysis for Loans {
    type Facts = Loans;

    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Loans {
        let reachable = am.get::<Dominators>(module, body);
        let decls = am.decls();
        let body = module.body(body);
        let forward = Forward { body, types: module.types(), decls };
        let facts = solve_dense(&forward, body);

        let mut loans = Loans { values: FxHashMap::default(), touched: FxHashMap::default(), before: FxHashMap::default() };
        for (id, block) in body.blocks() {
            if !reachable.reachable(id) {
                forward.unreached(id, block, &mut loans);
                continue;
            }
            let mut state = facts.at(id).clone();
            for inst in block.insts() {
                let site = Site::Inst(inst.id());
                loans.before.insert(site, state.storages.clone());
                loans.touched.insert(site, forward.touched(inst, &state));
                forward.transfer(inst, &mut state);
            }
            let site = Site::Term(id);
            loans.before.insert(site, state.storages.clone());
            loans.touched.insert(site, forward.touched_by_terminator(block.terminator(), &state));
            forward.transfer_terminator(block.terminator(), &mut state);
            for (value, positions) in state.values {
                let kept = loans.values.entry(value).or_insert_with(|| empty(positions.len()));
                join_positions(kept, &positions);
            }
            for value in forward.defined_in(block) {
                if forward.holds_no_loan(value) {
                    loans.values.entry(value).or_default();
                }
            }
        }
        for Held { value, positions } in forward.defined_before_entry() {
            loans.values.entry(value).or_insert(positions);
        }
        loans
    }
}

/// Every storage's content, and the positions of every defined value whose
/// type has any. A value of a type with no position holds no loan wherever
/// it stands, so the state does not carry it: it would add an entry to every
/// block's state, and every clone and join of one, and say nothing.
#[derive(Debug, Clone, PartialEq)]
struct State {
    storages: BTreeMap<Storage, Vec<LoanSet>>,
    values: BTreeMap<ValueId, Vec<LoanSet>>,
}

impl JoinSemiLattice for State {
    fn bottom() -> Self {
        State { storages: BTreeMap::new(), values: BTreeMap::new() }
    }

    fn join(&mut self, other: &Self) -> bool {
        join_maps(&mut self.storages, &other.storages) | join_maps(&mut self.values, &other.values)
    }

    fn le(&self, other: &Self) -> bool {
        le_maps(&self.storages, &other.storages) && le_maps(&self.values, &other.values)
    }
}

/// The lattice is finite over a body, so its widening is its join.
impl Widen for State {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

fn join_maps<K>(mine: &mut BTreeMap<K, Vec<LoanSet>>, other: &BTreeMap<K, Vec<LoanSet>>) -> bool
where
    K: Ord + Copy,
{
    let mut grew = false;
    for (key, positions) in other {
        match mine.get_mut(key) {
            Some(kept) => grew |= join_positions(kept, positions),
            None => {
                mine.insert(*key, positions.clone());
                grew = true;
            }
        }
    }
    grew
}

fn le_maps<K>(mine: &BTreeMap<K, Vec<LoanSet>>, other: &BTreeMap<K, Vec<LoanSet>>) -> bool
where
    K: Ord,
{
    mine.iter().all(|(key, positions)| {
        other.get(key).is_some_and(|theirs| positions.iter().zip(theirs).all(|(a, b)| a.is_subset(b)))
    })
}

/// Joins `other` into `kept` position by position.
///
/// # Panics
/// The two have different widths: one value or storage has one type.
fn join_positions(kept: &mut [LoanSet], other: &[LoanSet]) -> bool {
    assert_eq!(kept.len(), other.len(), "one value or storage has one width");
    let mut grew = false;
    for (mine, theirs) in kept.iter_mut().zip(other) {
        let before = mine.len();
        mine.extend(theirs.iter().copied());
        grew |= mine.len() != before;
    }
    grew
}

fn empty(width: usize) -> Vec<LoanSet> {
    vec![LoanSet::new(); width]
}

fn fold(positions: &[LoanSet]) -> LoanSet {
    positions.iter().flatten().copied().collect()
}

fn mode(mutability: Mutability) -> Mode {
    match mutability {
        Mutability::Shared => Mode::Shared,
        Mutability::Mut => Mode::Exclusive,
    }
}

/// A value and the loans each of its positions holds.
struct Held {
    value: ValueId,
    positions: Vec<LoanSet>,
}

/// What a call writes through one of its arguments or its callee value.
struct Written {
    into: ValueId,
    region: LoanSet,
}

/// The one part of one storage a write replaces whole.
struct Replaced {
    storage: Storage,
    range: Range<usize>,
}

/// A part of a type: where its positions lie among the whole's, and its
/// type.
struct Part {
    range: Range<usize>,
    ty: Ty,
}

/// Where a path leads within a type: every part it may name, and whether it
/// names exactly one part exactly, which an array index or an ambiguous
/// payload does not, since the loan does not tell which element or variant.
struct Reached {
    parts: Vec<Part>,
    exact: bool,
}

/// The storages a place lies in, the type of what a whole storage holds as
/// the place's root sees it, and its path.
struct Place<'p> {
    storages: BTreeSet<Storage>,
    root: Ty,
    path: &'p [PathSeg],
}

struct Forward<'b> {
    body: &'b Body,
    types: &'b Types<Concrete>,
    decls: &'b Declarations,
}

impl Forward<'_> {
    /// The places a call's callee declares it touches, where its impl
    /// declares an access.
    fn access(&self, callee: &Callee) -> Option<&[Touch]> {
        let Callee::Extern { id, r#impl, .. } = callee else {
            return None;
        };
        let facts = self.decls.impl_facts(*id, *r#impl)?;
        facts.access.iter().find_map(|form| match form {
            Access::Touches(touches) => Some(&touches[..]),
            Access::Equivalence => None,
        })
    }

    /// The places of `access` over a call's arguments, those it reads and
    /// those it writes.
    fn touched_by_access(&self, state: &State, access: &[Touch], args: &[ValueId]) -> Touched {
        let mut touched = Touched { by_access: true, ..Touched::default() };
        for Touch { place, mode } in access {
            let element = place.element.map(|element| match element {
                Element::Key(key) => ElementAt::Key(args[key]),
                Element::Index(index) => ElementAt::Index(args[index]),
            });
            for storage in self.named(state, args[place.param]) {
                let at = TouchedPlace { storage, element };
                if mode.reads() {
                    touched.reads.insert(storage);
                    touched.read_places.insert(at);
                }
                if mode.writes() {
                    touched.writes.insert(storage);
                    touched.written_places.insert(at);
                }
            }
        }
        touched
    }

    fn width(&self, ty: Ty) -> usize {
        positions(self.types, self.decls, ty)
    }

    fn kinds(&self, ty: Ty) -> Vec<PositionKind> {
        position_kinds(self.types, self.decls, ty)
    }

    /// Whether `value`'s type has no position, so that it holds no loan.
    fn holds_no_loan(&self, value: ValueId) -> bool {
        self.width(self.body.ty_of(value)) == 0
    }

    /// The values `block` defines: its parameters, and each instruction's
    /// result and the token it yields.
    fn defined_in<'b>(&self, block: Block<'b>) -> impl Iterator<Item = ValueId> + 'b {
        let defined_by_insts =
            block.insts().flat_map(|inst| inst.result().into_iter().chain(inst.order().and_then(|order| order.yielded())));
        block.params().iter().copied().chain(defined_by_insts)
    }

    fn value(&self, state: &State, value: ValueId) -> Vec<LoanSet> {
        if self.holds_no_loan(value) {
            return Vec::new();
        }
        state
            .values
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is defined before it is read: its definition dominates its uses"))
            .clone()
    }

    fn content<'s>(&self, state: &'s State, storage: Storage) -> &'s [LoanSet] {
        state
            .storages
            .get(&storage)
            .unwrap_or_else(|| panic!("{storage:?} is a storage of the body, present in every reached state"))
    }

    /// The type a reference lends.
    ///
    /// # Panics
    /// `ty` is not a reference.
    fn lent(&self, ty: Ty) -> Ty {
        match self.types.kind(ty) {
            TyKind::Ref(_, lent) => lent.ty(),
            other => panic!("a reference is read through, not {other:?}"),
        }
    }

    /// The loans of each parameter and capture, each position naming its
    /// own outside storage; and each order parameter's none.
    fn defined_before_entry(&self) -> Vec<Held> {
        let outside = self.body.params().iter().chain(self.body.captures()).map(|(_, value)| *value);
        let mut defined: Vec<Held> =
            outside.map(|value| Held { value, positions: self.outside_loans(value) }).collect();
        defined.extend(self.body.order_param().map(|token| Held { value: token, positions: Vec::new() }));
        defined
    }

    fn outside_loans(&self, of: ValueId) -> Vec<LoanSet> {
        self.kinds(self.body.ty_of(of))
            .into_iter()
            .enumerate()
            .map(|(position, kind)| {
                let mode = match kind {
                    PositionKind::Ref(mutability, _) => mode(mutability),
                    PositionKind::Captures(Holds::Nothing) => return LoanSet::new(),
                    PositionKind::Captures(Holds::Shared) => Mode::Shared,
                    PositionKind::Captures(Holds::Mut) => Mode::Exclusive,
                    PositionKind::Lifetime => Mode::Exclusive,
                    PositionKind::InFlight => panic!("a parameter or capture of a body holds no started call"),
                };
                LoanSet::from([Loan { storage: Storage::Outside { of, position }, mode }])
            })
            .collect()
    }

    /// Values no run defines hold no loan.
    fn unreached(&self, id: BlockId, block: Block<'_>, loans: &mut Loans) {
        for value in self.defined_in(block) {
            loans.values.insert(value, empty(self.width(self.body.ty_of(value))));
        }
        for inst in block.insts() {
            loans.touched.insert(Site::Inst(inst.id()), Touched::default());
        }
        loans.touched.insert(Site::Term(id), Touched::default());
    }

    /// Where `path` leads within a value of `root`. Cost: O(the path · the
    /// parts it passes).
    fn reach(&self, root: Ty, path: &[PathSeg]) -> Reached {
        let mut parts = vec![Part { range: 0..self.width(root), ty: root }];
        let mut exact = true;
        for seg in path {
            let mut next = Vec::new();
            for part in parts {
                let start = part.range.start;
                let laid = |items: &mut dyn Iterator<Item = Ty>| {
                    let mut at = start;
                    items
                        .map(|ty| {
                            let width = self.width(ty);
                            let item = Part { range: at..at + width, ty };
                            at += width;
                            item
                        })
                        .collect::<Vec<Part>>()
                };
                match (seg, self.types.kind(part.ty)) {
                    (PathSeg::Field(field), TyKind::Object(object)) => {
                        let fields = laid(&mut object.fields().iter().map(|(_, ty)| *ty));
                        let at = object.fields().iter().position(|(name, _)| name == field);
                        next.extend(at.map(|at| fields.into_iter().nth(at).expect("one part per field")));
                    }
                    (PathSeg::Index(index), TyKind::Tuple(items)) => {
                        next.extend(laid(&mut items.iter().copied()).into_iter().nth(*index));
                    }
                    (PathSeg::Index(_), TyKind::Array(element, _)) => {
                        exact = false;
                        next.push(Part { range: part.range.clone(), ty: *element });
                    }
                    (PathSeg::Payload, TyKind::Option(payload)) => {
                        next.push(Part { range: part.range.clone(), ty: *payload });
                    }
                    (PathSeg::Payload, TyKind::Result(ok, err)) => {
                        exact = false;
                        next.extend(laid(&mut [*ok, *err].into_iter()));
                    }
                    (PathSeg::Payload, TyKind::Enum { variants, .. }) => {
                        let payloads: Vec<Ty> = variants.iter().filter_map(|(_, payload)| *payload).collect();
                        exact &= payloads.len() == 1;
                        next.extend(laid(&mut payloads.into_iter()));
                    }
                    (seg, other) => panic!("{seg:?} steps into {other:?}; the path agrees with its type"),
                }
            }
            parts = next;
        }
        Reached { parts, exact }
    }

    /// The parts of `reached` of type `ty`.
    fn parts_of(&self, reached: &Reached, ty: Ty) -> Vec<Range<usize>> {
        let ranges: Vec<Range<usize>> =
            reached.parts.iter().filter(|part| part.ty == ty).map(|part| part.range.clone()).collect();
        assert!(!ranges.is_empty(), "a place's path reaches its type, {:?}", self.types.kind(ty));
        ranges
    }

    fn place<'p>(&self, state: &State, target: &RefTarget, path: &'p [PathSeg]) -> Place<'p> {
        match target {
            RefTarget::Var(slot) => {
                Place { storages: BTreeSet::from([Storage::Slot(*slot)]), root: self.body.ty_of(*slot), path }
            }
            RefTarget::Param(param) => {
                Place { storages: BTreeSet::from([Storage::Param(*param)]), root: self.body.ty_of(*param), path }
            }
            RefTarget::Through(reference) => Place {
                storages: self.named(state, *reference),
                root: self.lent(self.body.ty_of(*reference)),
                path,
            },
        }
    }

    /// The storages a reference names, at its own position.
    fn named(&self, state: &State, reference: ValueId) -> BTreeSet<Storage> {
        let positions = self.value(state, reference);
        let own = positions.first().expect("a reference has its own position");
        own.iter().map(|loan| loan.storage).collect()
    }

    /// What the part of `ty` at `place` holds: aligned where a storage is
    /// covered whole, else every loan of its content at each position.
    fn read(&self, state: &State, place: &Place<'_>, ty: Ty) -> Vec<LoanSet> {
        let width = self.width(ty);
        let mut out = empty(width);
        let reached = self.reach(place.root, place.path);
        let ranges = self.parts_of(&reached, ty);
        for storage in &place.storages {
            let content = self.content(state, *storage);
            if content.len() == self.width(place.root) {
                for range in &ranges {
                    join_positions(&mut out, &content[range.clone()]);
                }
            } else {
                let all = fold(content);
                for position in &mut out {
                    position.extend(all.iter().copied());
                }
            }
        }
        out
    }

    /// Whether a write to `place` replaces what the one part it names held.
    fn strong(&self, state: &State, place: &Place<'_>, ty: Ty) -> Option<Replaced> {
        let [storage] = place.storages.iter().copied().collect::<Vec<_>>()[..] else {
            return None;
        };
        if self.content(state, storage).len() != self.width(place.root) {
            return None;
        }
        let reached = self.reach(place.root, place.path);
        let ranges = self.parts_of(&reached, ty);
        match (reached.exact, &ranges[..]) {
            (true, [range]) => Some(Replaced { storage, range: range.clone() }),
            _ => None,
        }
    }

    fn write(&self, state: &mut State, place: &Place<'_>, ty: Ty, value: &[LoanSet]) {
        if let Some(Replaced { storage, range }) = self.strong(state, place, ty) {
            let content = state.storages.get_mut(&storage).expect("a written storage is present");
            content[range].clone_from_slice(value);
            return;
        }
        let reached = self.reach(place.root, place.path);
        let ranges = self.parts_of(&reached, ty);
        let root_width = self.width(place.root);
        let all = fold(value);
        for storage in &place.storages {
            let content = state.storages.get_mut(storage).expect("a written storage is present");
            if content.len() == root_width {
                for range in &ranges {
                    join_positions(&mut content[range.clone()], value);
                }
            } else {
                for position in content.iter_mut() {
                    position.extend(all.iter().copied());
                }
            }
        }
    }

    /// A value's positions with what its references and captures reach read
    /// from the storages they name now, outer positions first so that an
    /// inner one sees what the outer read brought.
    fn deepened(&self, state: &State, value: ValueId) -> Vec<LoanSet> {
        let mut positions = self.value(state, value);
        for (at, kind) in self.kinds(self.body.ty_of(value)).into_iter().enumerate() {
            let named: Vec<Storage> = positions[at].iter().map(|loan| loan.storage).collect();
            match kind {
                PositionKind::Ref(_, width) => {
                    for storage in named {
                        let content = self.content(state, storage);
                        let pointee = &mut positions[at + 1..at + 1 + width];
                        if content.len() == width {
                            join_positions(pointee, content);
                        } else {
                            let all = fold(content);
                            pointee.iter_mut().for_each(|position| position.extend(all.iter().copied()));
                        }
                    }
                }
                PositionKind::Captures(_) => {
                    let reached: LoanSet = named.iter().flat_map(|storage| fold(self.content(state, *storage))).collect();
                    positions[at].extend(reached);
                }
                PositionKind::Lifetime | PositionKind::InFlight => {}
            }
        }
        positions
    }

    /// A reborrow of the whole of what `reference` lends, of mode `mode`.
    fn reborrow(&self, state: &State, reference: ValueId, mode: Mode) -> Vec<LoanSet> {
        let ty = self.body.ty_of(reference);
        let place = Place { storages: self.named(state, reference), root: self.lent(ty), path: &[] };
        let own: LoanSet = place.storages.iter().map(|storage| Loan { storage: *storage, mode }).collect();
        let mut positions = vec![own];
        positions.extend(self.read(state, &place, self.lent(ty)));
        join_positions(&mut positions[1..], &self.value(state, reference)[1..]);
        positions
    }

    fn define(&self, state: &mut State, value: ValueId, positions: Vec<LoanSet>) {
        if self.holds_no_loan(value) {
            return;
        }
        match state.values.get_mut(&value) {
            Some(kept) => {
                join_positions(kept, &positions);
            }
            None => {
                state.values.insert(value, positions);
            }
        }
    }

    /// The loans each position of an instruction's result holds, and the
    /// writes it makes.
    fn result(&self, inst: &Inst, state: &mut State) -> Option<Vec<LoanSet>> {
        let result_ty = inst.result().map(|result| self.body.ty_of(result));
        let none = || result_ty.map(|ty| empty(self.width(ty)));
        let ty = || result_ty.expect("the operation yields a value");
        Some(match inst.op() {
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
            | Op::StringEq { .. }
            | Op::StringClone { .. }
            | Op::StructuralEq { .. }
            | Op::Fetch { .. }
            | Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Cast { .. }
            | Op::LoadFunction { .. }
            | Op::Merge { .. }
            | Op::ArrayBegin { .. }
            | Op::TestLiteral { .. }
            | Op::TestObjectKey { .. }
            | Op::TestVariant { .. }
            | Op::SliceLen { .. }
            | Op::Undef => return none(),
            Op::Commit { .. } | Op::Check { .. } | Op::CheckSteps { .. } | Op::Drop { .. } => return None,
            Op::StringAppend { .. } => return None,
            Op::Ref { target, path, mutability } => {
                let place = self.place(state, target, path);
                let own: LoanSet =
                    place.storages.iter().map(|storage| Loan { storage: *storage, mode: mode(*mutability) }).collect();
                let mut positions = vec![own];
                positions.extend(self.read(state, &place, self.lent(ty())));
                positions
            }
            Op::Take { target, path, .. } => {
                let place = self.place(state, target, path);
                let taken = self.read(state, &place, ty());
                if !copies(self.types, ty())
                    && let Some(Replaced { storage, range }) = self.strong(state, &place, ty())
                {
                    let content = state.storages.get_mut(&storage).expect("a taken storage is present");
                    content[range].iter_mut().for_each(LoanSet::clear);
                }
                taken
            }
            Op::Assign { target, path, value, .. } => {
                let place = self.place(state, target, path);
                let written = self.value(state, *value);
                self.write(state, &place, self.body.ty_of(*value), &written);
                return None;
            }
            Op::AsSlice { container: lent, mutability, .. } | Op::ArrayView { src: lent, mutability } => {
                self.reborrow(state, *lent, mode(*mutability))
            }
            Op::Index { slice, mode: IndexMode::Ref, .. } => {
                let held = match self.types.kind(self.body.ty_of(*slice)) {
                    TyKind::Ref(held, _) => mode(*held),
                    other => panic!("an index reads a slice reference, not {other:?}"),
                };
                self.reborrow(state, *slice, held)
            }
            Op::Index { slice, mode: IndexMode::Copy, .. } => self.reborrow(state, *slice, Mode::Shared)[1..].to_vec(),
            Op::IndexSet { slice, value, .. } => {
                let storages = self.named(state, *slice);
                let written = self.value(state, *value);
                self.write_elements(state, &storages, &written);
                return None;
            }
            Op::StructuralClone { src, .. } => self.reborrow(state, *src, Mode::Shared)[1..].to_vec(),
            Op::FieldGet { object, field, rest } => {
                let path: Vec<PathSeg> = std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect();
                self.project(state, *object, &path, ty())
            }
            Op::ObjectGet { object, key } => self.project(state, *object, &[PathSeg::Field(*key)], ty()),
            Op::TupleIndex { tuple, index } => self.project(state, *tuple, &[PathSeg::Index(*index)], ty()),
            Op::ArrayIndex { array, .. } => self.project(state, *array, &[PathSeg::Index(0)], ty()),
            Op::UnwrapVariant { src } => self.project(state, *src, &[PathSeg::Payload], ty()),
            Op::FieldSet { object, field, rest, value } => {
                let path: Vec<PathSeg> = std::iter::once(field).chain(rest).map(|name| PathSeg::Field(*name)).collect();
                let mut positions = self.value(state, *object);
                let reached = self.reach(self.body.ty_of(*object), &path);
                let [range] = &self.parts_of(&reached, self.body.ty_of(*value))[..] else {
                    panic!("a field path names one field")
                };
                positions[range.clone()].clone_from_slice(&self.value(state, *value));
                positions
            }
            Op::ArrayPush { array, value } => {
                let mut positions = self.value(state, *array);
                join_positions(&mut positions, &self.value(state, *value));
                positions
            }
            Op::MakeTuple { elements } => elements.iter().flat_map(|element| self.value(state, *element)).collect(),
            Op::MakeObject { fields } => {
                let TyKind::Object(object) = self.types.kind(ty()) else {
                    panic!("an object is made at an object type")
                };
                object
                    .fields()
                    .iter()
                    .flat_map(|(name, _)| {
                        let (_, value) = fields
                            .iter()
                            .find(|(given, _)| given == name)
                            .expect("an object is made with every field of its type");
                        self.value(state, *value)
                    })
                    .collect()
            }
            Op::MakeVariant { tag, payload } => {
                let mut positions = empty(self.width(ty()));
                if let Some(payload) = payload {
                    let held = self.value(state, *payload);
                    for range in self.variant_ranges(ty(), *tag, self.body.ty_of(*payload)) {
                        join_positions(&mut positions[range], &held);
                    }
                }
                positions
            }
            Op::MakeClosure { captures, .. } => {
                vec![captures.iter().flat_map(|capture| fold(&self.value(state, *capture))).collect()]
            }
            Op::FunctionCall { callee, callee_ty, args } => self.call(state, callee, *callee_ty, args),
            Op::Spawn { callee, callee_ty, args } => {
                let held: LoanSet = args
                    .iter()
                    .chain(indirect(callee))
                    .flat_map(|value| fold(&self.deepened(state, *value)))
                    .collect();
                let result = self.call(state, callee, *callee_ty, args);
                std::iter::once(held).chain(result).collect()
            }
            Op::TakeEffect { src } => self.value(state, *src)[1..].to_vec(),
        })
    }

    /// Joins `written`, an element's positions, into every element of
    /// `storages`: the elements of a sequence share one set of positions.
    fn write_elements(&self, state: &mut State, storages: &BTreeSet<Storage>, written: &[LoanSet]) {
        let element_width = written.len();
        let all = fold(written);
        for storage in storages {
            let content = state.storages.get_mut(storage).expect("a written storage is present");
            if content.len() == element_width {
                join_positions(content, written);
            } else {
                content.iter_mut().for_each(|position| position.extend(all.iter().copied()));
            }
        }
    }

    /// The part of `value` at `path` of type `ty`, joined over every part
    /// the path may name.
    fn project(&self, state: &State, value: ValueId, path: &[PathSeg], ty: Ty) -> Vec<LoanSet> {
        let positions = self.value(state, value);
        let reached = self.reach(self.body.ty_of(value), path);
        let mut out = empty(self.width(ty));
        for range in self.parts_of(&reached, ty) {
            join_positions(&mut out, &positions[range]);
        }
        out
    }

    /// The ranges of the variant `tag` of `ty` whose payload is `payload`:
    /// an `Option`'s one, a `Result`'s of that type, an enum's by its tag.
    fn variant_ranges(&self, ty: Ty, tag: acvus_utils::Astr, payload: Ty) -> Vec<Range<usize>> {
        match self.types.kind(ty) {
            TyKind::Option(_) | TyKind::Result(..) => self.parts_of(&self.reach(ty, &[PathSeg::Payload]), payload),
            TyKind::Enum { variants, .. } => {
                let mut at = 0;
                for (name, declared) in variants.iter() {
                    let Some(declared) = declared else { continue };
                    let width = self.width(*declared);
                    if *name == tag {
                        return vec![at..at + width];
                    }
                    at += width;
                }
                panic!("a variant is made with a tag of its enum")
            }
            other => panic!("a variant is made at a variant type, not {other:?}"),
        }
    }

    /// A call's result, and its writes through the arguments its flows name
    /// as outputs (ownership §5, extern-fn §4).
    fn call(&self, state: &mut State, callee: &Callee, callee_ty: Ty, args: &[ValueId]) -> Vec<LoanSet> {
        let TyKind::Fn { params, ret, flows, .. } = self.types.kind(callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        let flows = flows.get();
        let arity = args.len();
        let input = |from: FlowEnd| -> Vec<LoanSet> {
            match from {
                FlowEnd::Param(index) => {
                    let arg = args.get(index).expect("a flow names a parameter the call passes");
                    self.deepened(state, *arg)
                }
                FlowEnd::Captures => match callee {
                    Callee::Indirect(function) => vec![fold(&self.deepened(state, *function))],
                    Callee::Direct(_) | Callee::Extern { .. } => Vec::new(),
                },
                FlowEnd::Result => panic!("a call's result is an output only"),
            }
        };

        let mut result = empty(self.width(*ret));
        for Source { from, alignment } in flows.into_end(FlowEnd::Result, arity) {
            let taken = input(from);
            match &alignment {
                Alignment::Aligned => {
                    join_positions(&mut result, &taken);
                }
                Alignment::Any => {
                    let all = fold(&taken);
                    result.iter_mut().for_each(|position| position.extend(all.iter().copied()));
                }
                Alignment::Labelled(labelled) => {
                    let FlowEnd::Param(index) = from else {
                        panic!("a Labelled flow's input is a parameter of the extern it is read off")
                    };
                    self.join_labelled(&mut result, labelled, params[index], *ret, from, &taken);
                }
            }
        }

        let access = self.access(callee);
        let written_by_access = |to: &FlowEnd| match (access, to) {
            (None, _) => true,
            (Some(touches), FlowEnd::Param(index)) => {
                touches.iter().any(|touch| touch.place.param == *index && touch.mode.writes())
            }
            (Some(_), FlowEnd::Captures | FlowEnd::Result) => false,
        };
        let outputs = (0..arity).map(FlowEnd::Param).chain(indirect(callee).map(|_| FlowEnd::Captures));
        let written: Vec<Written> = outputs
            .filter(written_by_access)
            .filter_map(|to| {
                let sources = flows.into_end(to, arity);
                if sources.is_empty() {
                    return None;
                }
                let region: LoanSet = sources.into_iter().flat_map(|Source { from, .. }| fold(&input(from))).collect();
                let into = match to {
                    FlowEnd::Param(index) => args[index],
                    FlowEnd::Captures => *indirect(callee).next().expect("only a callee value captures"),
                    FlowEnd::Result => panic!("the result is not written into"),
                };
                Some(Written { into, region })
            })
            .collect();
        for Written { into, region } in written {
            self.write_into(state, into, &region);
        }
        result
    }

    /// `region` written through every `&mut` position of `value` and into
    /// what its mutable captures name, and joined into the pointee positions
    /// of its `&mut` positions.
    fn write_into(&self, state: &mut State, value: ValueId, region: &LoanSet) {
        let held = self.deepened(state, value);
        let mut own = self.value(state, value);
        for (at, kind) in self.kinds(self.body.ty_of(value)).into_iter().enumerate() {
            let pointee = match kind {
                PositionKind::Ref(Mutability::Mut, width) => Some(width),
                PositionKind::Captures(Holds::Mut) => None,
                PositionKind::Ref(Mutability::Shared, _)
                | PositionKind::Captures(Holds::Shared | Holds::Nothing)
                | PositionKind::Lifetime
                | PositionKind::InFlight => continue,
            };
            for loan in held[at].iter().filter(|loan| loan.mode == Mode::Exclusive) {
                let content = state.storages.get_mut(&loan.storage).expect("a named storage is present");
                content.iter_mut().for_each(|position| position.extend(region.iter().copied()));
            }
            if let Some(width) = pointee {
                own[at + 1..at + 1 + width].iter_mut().for_each(|position| position.extend(region.iter().copied()));
            }
        }
        self.define(state, value, own);
    }

    /// Each output segment joins exactly the input segments a `Labelled`
    /// flow's map names, position by position where both are one type
    /// variable's part.
    fn join_labelled(
        &self,
        out: &mut [LoanSet],
        labelled: &Labelled,
        from_ty: Ty,
        to_ty: Ty,
        from: FlowEnd,
        input: &[LoanSet],
    ) {
        let poly = self.decls.types();
        let to_decl = labelled.declared.end_ty(self.decls, FlowEnd::Result);
        let from_decl = labelled.declared.end_ty(self.decls, from);
        let to_ranges = segment_positions(poly, to_decl, self.types, to_ty, self.decls);
        let from_ranges = segment_positions(poly, from_decl, self.types, from_ty, self.decls);
        let to_segments = segments(poly, self.decls, to_decl);
        let from_segments = segments(poly, self.decls, from_decl);
        assert!(
            labelled.map.len() == to_ranges.len() && labelled.map.iter().flatten().all(|segment| *segment < from_ranges.len()),
            "a Labelled flow's map names the segments of its ends"
        );
        for (at, takes) in labelled.map.iter().enumerate() {
            let to = to_ranges[at].clone();
            for segment in takes.iter().copied() {
                let values = &input[from_ranges[segment].clone()];
                if to_segments[at].aligned_with(from_segments[segment]) {
                    join_positions(&mut out[to.clone()], values);
                } else {
                    let all = fold(values);
                    out[to.clone()].iter_mut().for_each(|position| position.extend(all.iter().copied()));
                }
            }
        }
    }

    /// The storages an instruction reads and writes, in `state`, just
    /// before it runs.
    fn touched(&self, inst: &Inst, state: &State) -> Touched {
        if let Op::FunctionCall { callee, args, .. } | Op::Spawn { callee, args, .. } = inst.op()
            && let Some(access) = self.access(callee)
        {
            return self.touched_by_access(state, access, args);
        }
        let mut touched = Touched::default();
        let named_if_reference = |value: ValueId| match self.types.kind(self.body.ty_of(value)) {
            TyKind::Ref(..) => self.named(state, value),
            _ => BTreeSet::new(),
        };
        match inst.op() {
            Op::Take { target, path, .. } => touched.reads = self.place(state, target, path).storages,
            Op::Assign { target, path, .. } => touched.writes = self.place(state, target, path).storages,
            Op::Index { slice, .. } => touched.reads = self.named(state, *slice),
            Op::IndexSet { slice, .. } => touched.writes = self.named(state, *slice),
            Op::StringAppend { target, part } => {
                touched.writes = self.named(state, *target);
                touched.reads = named_if_reference(*part);
            }
            Op::StringEq { a, b } | Op::StructuralEq { a, b, .. } => {
                touched.reads = named_if_reference(*a);
                touched.reads.extend(named_if_reference(*b));
            }
            Op::StringClone { src } | Op::StructuralClone { src, .. } => touched.reads = named_if_reference(*src),
            Op::FunctionCall { callee, args, .. } | Op::Spawn { callee, args, .. } => {
                for value in args.iter().chain(indirect(callee)) {
                    let held = self.deepened(state, *value);
                    touched.reads.extend(held.iter().flatten().map(|loan| loan.storage));
                    for (at, kind) in self.kinds(self.body.ty_of(*value)).into_iter().enumerate() {
                        if matches!(kind, PositionKind::Ref(Mutability::Mut, _) | PositionKind::Captures(Holds::Mut)) {
                            let written = held[at].iter().filter(|loan| loan.mode == Mode::Exclusive);
                            touched.writes.extend(written.map(|loan| loan.storage));
                        }
                    }
                }
            }
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
            | Op::Ref { .. }
            | Op::AsSlice { .. }
            | Op::ArrayView { .. }
            | Op::SliceLen { .. }
            | Op::Fetch { .. }
            | Op::Commit { .. }
            | Op::FieldGet { .. }
            | Op::FieldSet { .. }
            | Op::BinOp { .. }
            | Op::UnaryOp { .. }
            | Op::Check { .. }
            | Op::CheckSteps { .. }
            | Op::Cast { .. }
            | Op::LoadFunction { .. }
            | Op::TakeEffect { .. }
            | Op::Merge { .. }
            | Op::ArrayBegin { .. }
            | Op::ArrayPush { .. }
            | Op::MakeObject { .. }
            | Op::MakeTuple { .. }
            | Op::TupleIndex { .. }
            | Op::TestLiteral { .. }
            | Op::TestObjectKey { .. }
            | Op::ArrayIndex { .. }
            | Op::ObjectGet { .. }
            | Op::MakeClosure { .. }
            | Op::MakeVariant { .. }
            | Op::TestVariant { .. }
            | Op::UnwrapVariant { .. }
            | Op::Undef
            | Op::Drop { .. } => {}
        }
        Touched::whole(touched.reads, touched.writes)
    }

    /// A `For` over a slice reads its source's storage; no other terminator
    /// touches storage.
    fn touched_by_terminator(&self, terminator: &Terminator, state: &State) -> Touched {
        let reads = match terminator {
            Terminator::For { source: ForSource::Slice(slice) | ForSource::SliceMut(slice), .. } => {
                self.named(state, *slice)
            }
            _ => BTreeSet::new(),
        };
        Touched::whole(reads, BTreeSet::new())
    }
}

fn indirect(callee: &Callee) -> impl Iterator<Item = &ValueId> {
    match callee {
        Callee::Indirect(function) => Some(function),
        Callee::Direct(_) | Callee::Extern { .. } => None,
    }
    .into_iter()
}

impl DenseAnalysis for Forward<'_> {
    type State = State;

    const DIRECTION: Direction = Direction::Forward;

    /// Every slot empty; every parameter and capture holding its outside
    /// storages, each by-value parameter's own storage holding the
    /// parameter, and each outside storage of a reference position holding
    /// what the parameter's positions under it name.
    fn entry(&self, body: &Body) -> State {
        let mut state = State::bottom();
        for slot in body.slots() {
            state.storages.insert(Storage::Slot(*slot), empty(self.width(body.ty_of(*slot))));
        }
        for Held { value, positions } in self.defined_before_entry() {
            if body.params().iter().any(|(_, param)| *param == value) {
                state.storages.insert(Storage::Param(value), positions.clone());
            }
            if !body.is_token(value) {
                for (position, kind) in self.kinds(body.ty_of(value)).into_iter().enumerate() {
                    let content = match kind {
                        PositionKind::Ref(_, width) => positions[position + 1..position + 1 + width].to_vec(),
                        PositionKind::Captures(_) | PositionKind::Lifetime | PositionKind::InFlight => Vec::new(),
                    };
                    state.storages.insert(Storage::Outside { of: value, position }, content);
                }
            }
            if !self.holds_no_loan(value) {
                state.values.insert(value, positions);
            }
        }
        state
    }

    fn transfer(&self, inst: &Inst, state: &mut State) {
        if let Some(positions) = self.result(inst, state) {
            let result = inst.result().expect("an operation with loans yields a value");
            self.define(state, result, positions);
        }
        if let Some(token) = inst.order().and_then(|order| order.yielded()) {
            self.define(state, token, Vec::new());
        }
    }

    /// A `For` fills its body's leading parameters: a slice's element is a
    /// reborrow of what the source lends, an array's is moved out of it,
    /// and the counter holds none; an exit's trip count holds none.
    fn transfer_terminator(&self, terminator: &Terminator, state: &mut State) {
        let Terminator::For { source, stages, exit, exit_trip, .. } = terminator else {
            return;
        };
        let params = self.body.block(stages.body()).params();
        let element = match source {
            ForSource::Slice(slice) => Some(self.reborrow(state, *slice, Mode::Shared)),
            ForSource::SliceMut(slice) => Some(self.reborrow(state, *slice, Mode::Exclusive)),
            ForSource::Array(array) => Some(self.value(state, *array)),
            ForSource::Range { .. } => None,
        };
        let counter = params[source.counter_param()];
        self.define(state, counter, empty(self.width(self.body.ty_of(counter))));
        if let Some(element) = element {
            let element_param = params[..source.supplied_params()]
                .iter()
                .copied()
                .find(|param| *param != counter)
                .expect("a traversal supplies its element beside its counter");
            self.define(state, element_param, element);
        }
        if let Some(trip) = exit_trip.trip_param(self.body.block(*exit).params()) {
            self.define(state, trip, empty(self.width(self.body.ty_of(trip))));
        }
    }

    fn transfer_edge(&self, edge: &Edge<'_>, target_params: &[ValueId], state: &mut State) {
        for (param, arg) in target_params[edge.supplied..].iter().zip(edge.args) {
            let positions = self.value(state, *arg);
            self.define(state, *param, positions);
        }
    }
}
