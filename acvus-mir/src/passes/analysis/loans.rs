//! Loans (ownership §5): which place each value may name at each position
//! of its type, and what each storage's content holds, at every point of a
//! body: the dense engine, forward, before promotion, when every binding is
//! a slot. A storage's content changes along the paths through the body,
//! so it is tracked per point; a value's loans are fixed where it is
//! defined, and joined at a block's parameters.
//!
//! Its dimensions:
//! - where: a loan names a place, a storage and a path in it (`ir::parts`),
//!   so two loans on parts that do not overlap never meet (ownership §5: a
//!   reference to a transparent structured value is the structure of
//!   references to its parts, a sum's including its variant's place);
//! - mode: shared or exclusive;
//! - who holds it: a position of a value, or of a storage's content;
//! - when: while its holder is live, which `check::borrowck` reads from
//!   `Liveness` and `ContentLiveness`; this analysis does not decide it.
//!
//! A storage is a slot, a by-value parameter's own storage (what
//! `RefTarget::Param` names), or one outside storage per position of each
//! parameter's and capture's type, distinct from every other since its
//! callers were checked.
//!
//! A loan is made at one site, on one place, at one mode: two loans are one
//! only where one site made them on one place. A loan made through another
//! is within it, and lives no longer than it (ownership §5). A loan's path
//! is exact, or, where a call's result borrows from an argument, some part
//! under it the flows do not tell (extern-fn §4). A loan a reference holds
//! is at a part of its target: the whole for a plain reference, part `i`
//! for the reference `Gather` builds of its parts' references.
//!
//! How loans are made and move:
//! 1. Lend, `Ref` by name of a place whose type is not a reference: one loan
//!    made there on that place, exact, at the reference's mode.
//! 2. Reborrow: a `Ref` through `r`; a `Ref` by name of a place whose type
//!    is a reference, which reborrows its target (types §4); and a use the
//!    takings (`ir::cfg::takings`) take `Read` of a `&mut` at a by-value
//!    position, which reborrows it implicitly. For each loan `l` of `r`'s
//!    own, one loan made there on `l`'s place extended by the path, exact
//!    where `l` is, at the weaker of `l`'s mode and the reborrow's, within
//!    `l`. The result holds these as its own. A `Take` through `r` of a
//!    part whose type is a reference uses the reference stored there: for
//!    each of its loans, one made there within it and within the loans of
//!    `r` that reach the part, what a path of references reaches being held
//!    by its first, exclusive only where all of them are.
//! 3. Move between a value and a storage: a `Take` gives its value what the
//!    content held at the positions its path reaches; an `Assign` gives the
//!    content at its path's positions what the value holds. A move leaves
//!    the content as it was: whether it is read again is `ContentLiveness`'
//!    and `check::moveck`'s.
//! 4. A part, both ways: reading a part gives what its positions hold, and
//!    building a value of parts (`MakeTuple`, `MakeObject`, `MakeVariant`,
//!    `ArrayPush`, `FieldSet`) puts each part's at its place. A closure's
//!    positions are its captures'; `Specialize` holds its closure's. All
//!    elements of a sequence are one part, which an index, a slice, a
//!    `For` and `ArrayMap` reach (ownership §5): every index of a place in
//!    a sequence names it, so any two overlap. `ArrayMap` applies its
//!    function's flows once, to that part of each array, as a call.
//! 5. Gather: the reference built holds each part reference's own loans at
//!    that part; no loan is made.
//! 6. A call: an argument's lend or reborrow is made where the call starts
//!    (ownership §5, effects §6), so a value that is only one call's argument
//!    holds its loans from the call on. The result and the written
//!    arguments take loans only as the function type's flows state; one the
//!    result takes from an argument is made at the call, within that
//!    argument's loan, not exact. A call with a declared access touches
//!    only the places that access states, in their modes.
//! 7. A join: a block's parameter holds what any argument into it holds.
//!
//! A context is a storage by its name (context §3), whole at entry, whose
//! content holds no loan, since what it holds outlives the run. It is seen
//! beyond the program, so accessed whole by name, at each call whose
//! effect may read it (a read) or write it (a write), a call whose effect
//! is opaque touching every context the body names, and read whole at
//! every `Return` (context §1). A call is where its effect enters: a
//! `FunctionCall`, `ArrayMap`'s call of its function, and a split call's
//! taking (effects §6). These accesses are recorded as any other.
//!
//! Each site's accesses are recorded as `accesses`: what the takings read,
//! write and move (`Moved`, `PartMoved`), a `Take` through a reference
//! moving out of the places it lends as one by name does, and each loan
//! made there, a lend at its mode. An access through a reference reaches,
//! for each of its loans at a part the access's path meets, that loan's
//! place extended by the rest of the path, or everything under it where
//! the loan is not exact. A place is one place (`is_one`) where it is
//! exact and names no sequence's element: an index path, normalized to
//! the one element part, names every element.
//!
//! passes-analyses.md.


use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::decl::declared::{Access as DeclaredAccess, Element, Touch};
use crate::decl::{Declarations, segments};
use crate::flows::segment_positions;
use crate::ir::cfg::takings::{Taken, by_value, place_taking};
use crate::ir::cfg::{Aspect, Block, BlockId, Body, BodyId, Definition, Edge, Inst, Module, Op, Site, Terminator};
use crate::ir::parts::{Path, covers, overlap};
use acvus_utils::QualifiedRef;

use crate::ir::{Callee, ForSource, IndexMode, Intrinsic, PathSeg, RefTarget, ValueId, fn_effect};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, Dominators, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::ty::structural::{PartKey, structure};
use crate::ty::{
    Alignment, Concrete, FlowEnd, Flows, Holds, Labelled, Mutability, PositionKind, Reissue, Source, Ty, TyKind,
    Types, lent_types, position_kinds, positions,
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
    /// A context, by its name (context §3). Whole at the body's entry. It is
    /// seen beyond the program, so accessed whole, at each call whose effect
    /// may read it (a read) or write it (a write), an opaque call touching
    /// every context, and read whole at every return: these accesses are
    /// recorded at those sites as any other, so every reader of accesses
    /// meets them with no rule of its own (context §1, §3).
    Context(QualifiedRef),
}

impl Storage {
    /// The storage `target` names by name; `None` through a reference.
    pub fn by_name(target: &RefTarget) -> Option<Storage> {
        match target {
            RefTarget::Var(slot) => Some(Storage::Slot(*slot)),
            RefTarget::Param(param) => Some(Storage::Param(*param)),
            RefTarget::Context(context) => Some(Storage::Context(*context)),
            RefTarget::Through(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mode {
    Shared,
    Exclusive,
}

/// A loan (ownership §5): the place it names, a storage and a path in it
/// (`ir::parts`), and its mode; the part of its reference's target it is
/// at, and where it was made.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Loan {
    pub storage: Storage,
    pub path: Path,
    /// false: some part under `path` the flows do not tell (a call's result borrowing from an argument).
    pub exact: bool,
    pub mode: Mode,
    /// The part of a reference's target this loan is at: [] for a plain reference, [i] for part i of a Gather.
    pub at: Path,
    pub made: Made,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Made {
    At(Site),
    /// A parameter's or capture's loan, made outside the body; named at its binder.
    Outside { of: ValueId },
}

/// The loans one position may hold.
pub type LoanSet = BTreeSet<Loan>;

/// A place an access reaches: a storage and a path from its root, and
/// whether it is that place exactly or some part under it the flows do not
/// tell.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Place {
    pub storage: Storage,
    pub path: Path,
    pub exact: bool,
}

impl Place {
    /// Whether `loan`'s place and this one share anything.
    pub fn meets(&self, loan: &Loan) -> bool {
        self.storage == loan.storage && overlap(&self.path, &loan.path)
    }
}

/// What an access does to the places it reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    Read,
    Write,
    Move,
    /// The access makes the loans `Access::made`, each at its mode.
    Lend,
}

/// One access a site makes (ownership §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Access {
    pub reach: Reach,
    pub places: BTreeSet<Place>,
    /// The loans of the reference the access is made through; none for an
    /// access by name.
    pub through: LoanSet,
    /// A lend's loans; none for any other access.
    pub made: LoanSet,
    /// The edge of the site's terminator the access is made on, where it
    /// is an edge's argument: a run takes one edge.
    pub on_edge: Option<usize>,
}

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
    accesses: FxHashMap<Site, Vec<Access>>,
    within: BTreeMap<Loan, LoanSet>,
    holds_from: FxHashMap<ValueId, Site>,
    /// Every place an access reaches whose path steps into a sequence's
    /// elements.
    elements: FxHashSet<Place>,
    /// The contexts each site sees beyond the program, where it sees any.
    seen: FxHashMap<Site, BTreeSet<QualifiedRef>>,
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
    /// The page `page` sees, touched whole as well.
    fn seeing(mut self, page: &Page) -> Touched {
        for context in &page.reads {
            self.reads.insert(Storage::Context(*context));
            self.read_places.insert(TouchedPlace::whole(Storage::Context(*context)));
        }
        for context in &page.writes {
            self.writes.insert(Storage::Context(*context));
            self.written_places.insert(TouchedPlace::whole(Storage::Context(*context)));
        }
        self
    }

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
        (0..self.positions_of(value)).flat_map(|position| self.of(value, position).iter().cloned()).collect()
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
        self.contents_before(site)
            .get(&storage)
            .unwrap_or_else(|| panic!("{storage:?} is a storage of the body the loans cover"))
    }

    /// Every storage's content just before `site` runs. Cost: O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body that a run reaches.
    pub fn contents_before(&self, site: Site) -> &BTreeMap<Storage, Vec<LoanSet>> {
        self.before.get(&site).unwrap_or_else(|| panic!("{site:?} is a reached site of the body the loans cover"))
    }

    /// The accesses `site` makes; none where no run reaches it. Cost: O(1).
    pub fn accesses(&self, site: Site) -> &[Access] {
        self.accesses.get(&site).map_or(&[], Vec::as_slice)
    }

    /// The one place an access of `reach` that `site` makes through a
    /// reference reaches, where it reaches one place (`is_one`) and no
    /// other. Cost: O(the site's accesses).
    pub fn through_one(&self, site: Site, reach: Reach) -> Option<&Place> {
        self.accesses(site).iter().filter(|access| access.reach == reach && !access.through.is_empty()).find_map(|access| {
            let mut places = access.places.iter();
            match (places.next(), places.next()) {
                (Some(place), None) if self.is_one(place) => Some(place),
                _ => None,
            }
        })
    }

    /// Whether `place`, which an access of the body reaches, is one place:
    /// exact, and naming no sequence's element, which stands for every
    /// element (ownership §4, §5). Cost: O(1).
    pub fn is_one(&self, place: &Place) -> bool {
        place.exact && !self.elements.contains(place)
    }

    /// Every loan some loan of `loans` is made through, directly or through
    /// another. Cost: O(the loans reached · log loans).
    pub fn ancestors(&self, loans: &LoanSet) -> LoanSet {
        let mut reached = LoanSet::new();
        let mut pending: Vec<&Loan> = loans.iter().collect();
        while let Some(loan) = pending.pop() {
            for parent in self.within.get(loan).into_iter().flatten() {
                if reached.insert(parent.clone()) {
                    pending.push(parent);
                }
            }
        }
        reached
    }

    /// The contexts `site` sees beyond the program, which it accesses whole
    /// (context §1): those a call's effect may read or write, and every
    /// context the body names at a `Return`. Cost: O(1).
    pub fn page_seen(&self, site: Site) -> impl Iterator<Item = QualifiedRef> + '_ {
        self.seen.get(&site).into_iter().flatten().copied()
    }

    /// The call `value` is only an argument of, from which on it holds the
    /// loans made there (ownership §5, effects §6). Cost: O(1).
    pub fn holds_from(&self, value: ValueId) -> Option<Site> {
        self.holds_from.get(&value).copied()
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
        let forward = Forward::new(body, module.types(), decls);
        let facts = solve_dense(&forward, body);

        let mut loans = Loans {
            values: FxHashMap::default(),
            touched: FxHashMap::default(),
            before: FxHashMap::default(),
            accesses: FxHashMap::default(),
            within: BTreeMap::new(),
            holds_from: forward.sole_call.clone(),
            elements: FxHashSet::default(),
            seen: FxHashMap::default(),
        };
        let mut record = Record::default();
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
                let page = forward.page_of_inst(inst);
                if !page.is_empty() {
                    loans.seen.insert(site, page.seen());
                }
                forward.step(inst, &mut state, &mut record);
            }
            let site = Site::Term(id);
            loans.before.insert(site, state.storages.clone());
            loans.touched.insert(site, forward.touched_by_terminator(block.terminator(), &state));
            let page = forward.page_of_terminator(block.terminator());
            if !page.is_empty() {
                loans.seen.insert(site, page.seen());
            }
            forward.step_terminator(id, block.terminator(), &mut state, &mut record);
            for (index, edge) in block.terminator().edges().iter().enumerate() {
                let mut along = state.clone();
                forward.step_edge(id, Some(index), edge, body.block(edge.to).params(), &mut along, &mut record);
            }
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
        loans.elements = record
            .accesses
            .values()
            .flatten()
            .flat_map(|access| &access.places)
            .filter(|place| forward.names_an_element(place))
            .cloned()
            .collect();
        loans.accesses = record.accesses;
        loans.within = record.within;
        loans
    }
}

/// What the steps of the final walk record: each site's accesses, and each
/// loan made through another. The fixed point's steps record into one
/// thrown away.
#[derive(Default)]
struct Record {
    accesses: FxHashMap<Site, Vec<Access>>,
    within: BTreeMap<Loan, LoanSet>,
}

impl Record {
    fn access(&mut self, site: Site, access: Access) {
        self.accesses.entry(site).or_default().push(access);
    }

    fn through(&mut self, site: Site, reach: Reach, located: &[Located], through: LoanSet) {
        let places = located.iter().map(Located::place).collect();
        self.access(site, Access { reach, places, through, made: LoanSet::new(), on_edge: None });
    }

    /// `made`, made at `site` through `through`, a lend at each loan's mode.
    fn lend(&mut self, site: Site, on_edge: Option<usize>, made: LoanSet, through: LoanSet) {
        if made.is_empty() {
            return;
        }
        let places = made.iter().map(Loan::place).collect();
        self.access(site, Access { reach: Reach::Lend, places, through, made, on_edge });
    }

    fn made_within(&mut self, loan: &Loan, parent: &Loan) {
        self.within.entry(loan.clone()).or_default().insert(parent.clone());
    }
}

/// The contexts a site sees beyond the program: those it reads whole, and
/// those it writes whole (context §1).
#[derive(Debug, Default)]
struct Page {
    reads: BTreeSet<QualifiedRef>,
    writes: BTreeSet<QualifiedRef>,
}

impl Page {
    fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }

    fn seen(&self) -> BTreeSet<QualifiedRef> {
        self.reads.union(&self.writes).copied().collect()
    }

    /// The accesses of the page at `site`: a whole read of each context it
    /// reads, then a whole write of each it writes, by name.
    fn record(&self, site: Site, record: &mut Record) {
        for (reach, contexts) in [(Reach::Read, &self.reads), (Reach::Write, &self.writes)] {
            if contexts.is_empty() {
                continue;
            }
            let places = contexts
                .iter()
                .map(|context| Place { storage: Storage::Context(*context), path: Vec::new(), exact: true })
                .collect();
            record.access(site, Access { reach, places, through: LoanSet::new(), made: LoanSet::new(), on_edge: None });
        }
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
        mine.extend(theirs.iter().cloned());
        grew |= mine.len() != before;
    }
    grew
}

/// `held` joined into `into`: position by position where the two are as
/// wide, else every loan of `held` into every position of `into`.
fn join_aligned_or_spread(into: &mut [LoanSet], held: &[LoanSet]) {
    match into.len() == held.len() {
        true => {
            join_positions(into, held);
        }
        false => spread(into, &fold(held)),
    }
}

fn empty(width: usize) -> Vec<LoanSet> {
    vec![LoanSet::new(); width]
}

fn fold(positions: &[LoanSet]) -> LoanSet {
    positions.iter().flatten().cloned().collect()
}

fn storages_of(place: &[Located]) -> BTreeSet<Storage> {
    place.iter().map(|located| located.storage).collect()
}

/// `loans` joined into every one of `positions`.
fn spread(positions: &mut [LoanSet], loans: &LoanSet) {
    for position in positions {
        position.extend(loans.iter().cloned());
    }
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

/// A part of a type: where its positions lie among the whole's, and its
/// type.
struct Part {
    range: Range<usize>,
    ty: Ty,
}

/// Where a path leads within a type: every part it may name, and whether it
/// names exactly one part exactly, which an array index or an ambiguous
/// payload does not, since the path does not tell which element or variant.
struct Reached {
    parts: Vec<Part>,
    exact: bool,
}

/// The positions of a storage's content a path names: the range of each
/// part it may name, and whether it names exactly one part exactly.
struct Region {
    ranges: Vec<Range<usize>>,
    exact: bool,
}

/// One place an access or a read reaches: a storage and a path from its
/// root, exact or everything under it; and the part of the value read or
/// written that this place is, `[]` for the whole.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Located {
    storage: Storage,
    path: Path,
    exact: bool,
    supplies: Path,
}

impl Located {
    fn by_name(storage: Storage, path: &[PathSeg]) -> Located {
        Located { storage, path: path.to_vec(), exact: true, supplies: Vec::new() }
    }

    fn place(&self) -> Place {
        Place { storage: self.storage, path: self.path.clone(), exact: self.exact }
    }
}

impl Loan {
    pub fn place(&self) -> Place {
        Place { storage: self.storage, path: self.path.clone(), exact: self.exact }
    }

    /// Where an access of `path` in its reference's target reaches through
    /// this loan, where the loan's part meets `path`: the loan's place
    /// extended by the rest of `path` past its part, or the loan's place
    /// where it is not exact; and which part of what is accessed that place
    /// is, where the loan's part lies under `path`.
    fn resolve(&self, path: &[PathSeg]) -> Option<Located> {
        if !overlap(&self.at, path) {
            return None;
        }
        let (rest, supplies) = match covers(&self.at, path) {
            true => (&path[self.at.len()..], &[][..]),
            false => (&[][..], &self.at[path.len()..]),
        };
        let path = match self.exact {
            true => [&self.path[..], rest].concat(),
            false => self.path.clone(),
        };
        Some(Located { storage: self.storage, path, exact: self.exact, supplies: supplies.to_vec() })
    }

    /// The loan a reborrow of `path` at `mode` through a reference holding
    /// this one makes at `made`, where this loan's part meets `path`: on the
    /// place it reaches there, at the weaker of the two modes, at the part
    /// of the new reference's target this loan's part is (ownership §5).
    fn reborrowed(&self, path: &[PathSeg], mode: Mode, made: Site) -> Option<Loan> {
        let Located { storage, path, exact, supplies } = self.resolve(path)?;
        Some(Loan { storage, path, exact, mode: self.mode.min(mode), at: supplies, made: Made::At(made) })
    }

    /// This loan, held by a part reference `Gather` puts at `part` of the
    /// reference it builds.
    fn at_part(&self, part: PathSeg) -> Loan {
        let at = std::iter::once(part).chain(self.at.iter().copied()).collect();
        Loan { at, ..self.clone() }
    }

    /// The loan a call's result takes from an argument holding this one,
    /// made at the call on some part under this loan's place, at no weaker
    /// a mode than the result's position holds (extern-fn §4).
    fn taken_by_result(&self, held: Option<Mode>, made: Site) -> Loan {
        let mode = held.map_or(self.mode, |held| self.mode.min(held));
        Loan { storage: self.storage, path: self.path.clone(), exact: false, mode, at: Vec::new(), made: Made::At(made) }
    }
}

/// The step from a sequence to its elements, all of which are its one part
/// (ownership §5): every index of a sequence names it.
const ELEMENT: PathSeg = PathSeg::Index(0);

/// The step from a structure to its part `key`.
fn part_seg(key: PartKey) -> PathSeg {
    match key {
        PartKey::Index(index) => PathSeg::Index(index),
        PartKey::Field(field) => PathSeg::Field(field),
        PartKey::Variant(_) | PartKey::Some | PartKey::Ok | PartKey::Err => PathSeg::Payload,
        PartKey::Element => ELEMENT,
    }
}

struct Forward<'b> {
    body: &'b Body,
    types: &'b Types<Concrete>,
    decls: &'b Declarations,
    /// Each value that is only one call's argument, and that call.
    sole_call: FxHashMap<ValueId, Site>,
}

impl<'b> Forward<'b> {
    fn new(body: &'b Body, types: &'b Types<Concrete>, decls: &'b Declarations) -> Self {
        Forward { body, types, decls, sole_call: sole_calls(body) }
    }
}

/// Each value that is only one call's argument, never its callee, and the
/// call: the loans such a value makes are made where the call starts
/// (ownership §5, effects §6).
fn sole_calls(body: &Body) -> FxHashMap<ValueId, Site> {
    let mut sole = FxHashMap::default();
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            let (Op::FunctionCall { callee, args, .. } | Op::Spawn { callee, args, .. }) = inst.op() else {
                continue;
            };
            let site = Site::Inst(inst.id());
            for arg in args {
                let only_an_argument = indirect(callee).all(|function| function != arg) && body.users(*arg).all(|user| user == site);
                if only_an_argument {
                    sole.insert(*arg, site);
                }
            }
        }
    }
    sole
}

impl Forward<'_> {
    /// The places a call's callee declares it touches, where the facts it
    /// reaches declare an access.
    fn access(&self, callee: &Callee) -> Option<&[Touch]> {
        let Callee::Extern { id, r#impl, .. } = callee else {
            return None;
        };
        let facts = self.decls.call_facts(*id, *r#impl);
        facts.access.iter().find_map(|form| match form {
            DeclaredAccess::Touches(touches) => Some(&touches[..]),
            DeclaredAccess::Equivalence => None,
        })
    }

    /// The contexts the instruction `inst` sees beyond the program: those
    /// the effect of the call it is may read or write, of the contexts the
    /// body names, every one where the effect is opaque (context §1,
    /// extern-fn §2). A call's effect enters where it is taken (effects §6):
    /// at a `FunctionCall`, at `ArrayMap`, which calls its function, and at
    /// a split call's taking, not its start.
    fn page_of_inst(&self, inst: &Inst) -> Page {
        let callee_ty = match inst.op() {
            Op::FunctionCall { callee_ty, .. } => *callee_ty,
            Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args } => {
                let [.., function] = &args[..] else { panic!("`ArrayMap` takes `[a_1, …, a_k, f]`") };
                self.body.ty_of(*function)
            }
            Op::TakeEffect { src } => {
                let Definition::Inst(start) = self.body.definition(*src) else {
                    panic!("a taking's handle is a start's result (effects §6)")
                };
                match self.body.inst(start).op() {
                    Op::Spawn { callee_ty, .. } => *callee_ty,
                    other => panic!("a taking's handle is a start's result (effects §6), not {other:?}'s"),
                }
            }
            _ => return Page::default(),
        };
        let effect = fn_effect(self.types, callee_ty).expect("a call's callee type is a function type");
        let named = self.body.contexts().iter().map(|named| named.context);
        match effect.reissue() {
            Reissue::Opaque => Page { reads: named.clone().collect(), writes: named.collect() },
            Reissue::Pure | Reissue::Idempotent => Page {
                reads: named.clone().filter(|context| effect.reads().contains(context)).collect(),
                writes: named.filter(|context| effect.writes().contains(context)).collect(),
            },
        }
    }

    /// The contexts `terminator` sees beyond the program: every context the
    /// body names, read whole, at a `Return` (context §1).
    fn page_of_terminator(&self, terminator: &Terminator) -> Page {
        match terminator {
            Terminator::Return { .. } => {
                Page { reads: self.body.contexts().iter().map(|named| named.context).collect(), writes: BTreeSet::new() }
            }
            _ => Page::default(),
        }
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

    fn is_mut_ref(&self, ty: Ty) -> bool {
        matches!(self.types.kind(ty), TyKind::Ref(Mutability::Mut, _))
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
                let storage = Storage::Outside { of, position };
                LoanSet::from([Loan { storage, path: Vec::new(), exact: true, mode, at: Vec::new(), made: Made::Outside { of } }])
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

    /// Where `path` leads within a value of `root`, or `None` where it
    /// steps into a part `root` does not have: a loan a call's flows or a
    /// join moved to a reference of another target keeps the path of the
    /// place it was made on, and a path extended from it need not agree
    /// with any type. Cost: O(the path · the parts it passes).
    fn reach(&self, root: Ty, path: &[PathSeg]) -> Option<Reached> {
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
                        let at = object.fields().iter().position(|(name, _)| name == field)?;
                        next.extend(fields.into_iter().nth(at));
                    }
                    (PathSeg::Index(index), TyKind::Tuple(items, _)) => {
                        let item = laid(&mut items.iter().copied()).into_iter().nth(*index)?;
                        next.push(item);
                    }
                    (PathSeg::Index(_), TyKind::Array(element, _, _)) => {
                        exact = false;
                        next.push(Part { range: part.range.clone(), ty: *element });
                    }
                    (PathSeg::Payload, TyKind::Option(payload, _)) => {
                        next.push(Part { range: part.range.clone(), ty: *payload });
                    }
                    (PathSeg::Payload, TyKind::Result(ok, err, _)) => {
                        exact = false;
                        next.extend(laid(&mut [*ok, *err].into_iter()));
                    }
                    (PathSeg::Payload, TyKind::Enum { variants, .. }) => {
                        let payloads: Vec<Ty> = variants.iter().filter_map(|(_, payload)| *payload).collect();
                        exact &= payloads.len() == 1;
                        next.extend(laid(&mut payloads.into_iter()));
                    }
                    _ => return None,
                }
            }
            parts = next;
        }
        Some(Reached { parts, exact })
    }

    /// Where `path` leads within a value of `root`, a part of which the body
    /// reads by its type.
    ///
    /// # Panics
    /// `path` steps into a part `root` does not have.
    fn reach_value(&self, root: Ty, path: &[PathSeg]) -> Reached {
        self.reach(root, path).unwrap_or_else(|| panic!("{path:?} steps into {:?}; a part read agrees with its type", self.types.kind(root)))
    }

    /// The parts of `reached` of type `ty`.
    fn parts_of(&self, reached: &Reached, ty: Ty) -> Vec<Range<usize>> {
        let ranges: Vec<Range<usize>> =
            reached.parts.iter().filter(|part| part.ty == ty).map(|part| part.range.clone()).collect();
        assert!(!ranges.is_empty(), "a value's path reaches its part's type, {:?}", self.types.kind(ty));
        ranges
    }

    /// The parts of `storage`'s content, `width` positions long, that
    /// `path` names; `None` where the content's layout does not locate
    /// them. An outside storage's content is its reference's target, whose
    /// type the storage does not keep, so only its whole is located.
    fn region(&self, storage: Storage, path: &[PathSeg], width: usize) -> Option<Region> {
        match storage {
            Storage::Slot(value) | Storage::Param(value) => {
                let Reached { parts, exact } = self.reach(self.body.ty_of(value), path)?;
                Some(Region { ranges: parts.into_iter().map(|part| part.range).collect(), exact })
            }
            Storage::Context(context) => {
                let Reached { parts, exact } = self.reach(self.body.context_ty(context), path)?;
                Some(Region { ranges: parts.into_iter().map(|part| part.range).collect(), exact })
            }
            Storage::Outside { .. } if path.is_empty() => Some(Region { ranges: vec![0..width], exact: true }),
            Storage::Outside { .. } => None,
        }
    }

    /// The ranges of a value `width` positions long, of type `ty` where it
    /// is known, that the part `supplies` of it lies at; `None` where the
    /// type does not locate it.
    fn supplied(&self, ty: Option<Ty>, width: usize, supplies: &[PathSeg]) -> Option<Vec<Range<usize>>> {
        if supplies.is_empty() {
            return Some(vec![0..width]);
        }
        let reached = self.reach(ty?, supplies)?;
        Some(reached.parts.into_iter().map(|part| part.range).collect())
    }

    /// What a reference names at its own position.
    fn own(&self, state: &State, reference: ValueId) -> LoanSet {
        let positions = self.value(state, reference);
        positions.into_iter().next().expect("a reference has its own position")
    }

    /// The places `target` and `path` name: by name, the storage and the
    /// path; through a reference, the place each loan it holds reaches.
    fn place(&self, state: &State, target: &RefTarget, path: &[PathSeg]) -> Vec<Located> {
        match target {
            RefTarget::Var(slot) => vec![Located::by_name(Storage::Slot(*slot), &self.canonical(self.body.ty_of(*slot), path))],
            RefTarget::Param(param) => vec![Located::by_name(Storage::Param(*param), &self.canonical(self.body.ty_of(*param), path))],
            RefTarget::Context(context) => {
                vec![Located::by_name(Storage::Context(*context), &self.canonical(self.body.context_ty(*context), path))]
            }
            RefTarget::Through(reference) => self.place_through(state, *reference, path),
        }
    }

    fn place_through(&self, state: &State, reference: ValueId, path: &[PathSeg]) -> Vec<Located> {
        through_loans(&self.own(state, reference), &self.canonical_through(reference, path))
    }

    /// `path` in a value of `ty`, every step into a sequence's elements
    /// written `ELEMENT`, so that a place names a sequence's one element
    /// part whatever index reached it. A step past a part whose type the
    /// path does not tell, the payload of one of several variants, stays as
    /// it is.
    fn canonical(&self, ty: Ty, path: &[PathSeg]) -> Path {
        let mut at = vec![ty];
        let mut canonical = Vec::with_capacity(path.len());
        for seg in path {
            let sequence = !at.is_empty() && at.iter().all(|ty| matches!(self.types.kind(*ty), TyKind::Array(..) | TyKind::Slice(_)));
            canonical.push(match (seg, sequence) {
                (PathSeg::Index(_), true) => ELEMENT,
                _ => *seg,
            });
            at = at.iter().flat_map(|ty| self.part_types(*ty, seg)).collect();
        }
        canonical
    }

    /// The types `seg` may step to from a value of `ty`.
    fn part_types(&self, ty: Ty, seg: &PathSeg) -> Vec<Ty> {
        match (seg, self.types.kind(ty)) {
            (PathSeg::Field(field), TyKind::Object(object)) => {
                object.fields().iter().filter(|(name, _)| name == field).map(|(_, ty)| *ty).collect()
            }
            (PathSeg::Index(index), TyKind::Tuple(items, _)) => items.get(*index).copied().into_iter().collect(),
            (PathSeg::Index(_), TyKind::Array(element, _, _) | TyKind::Slice(element)) => vec![*element],
            (PathSeg::Payload, TyKind::Option(payload, _)) => vec![*payload],
            (PathSeg::Payload, TyKind::Result(ok, err, _)) => vec![*ok, *err],
            (PathSeg::Payload, TyKind::Enum { variants, .. }) => variants.iter().filter_map(|(_, payload)| *payload).collect(),
            _ => Vec::new(),
        }
    }

    /// Whether `place`'s path steps into a sequence's elements, all of which
    /// are its one part. What a function value's or a lifetime's position
    /// lends has no type and no part.
    fn names_an_element(&self, place: &Place) -> bool {
        let root = match place.storage {
            Storage::Slot(value) | Storage::Param(value) => Some(self.body.ty_of(value)),
            Storage::Context(context) => Some(self.body.context_ty(context)),
            Storage::Outside { of, position } => lent_types(self.types, self.decls, self.body.ty_of(of))[position],
        };
        let mut at: Vec<Ty> = root.into_iter().collect();
        for seg in &place.path {
            let sequence = |ty: &Ty| matches!(self.types.kind(*ty), TyKind::Array(..) | TyKind::Slice(_));
            if matches!(seg, PathSeg::Index(_)) && at.iter().any(sequence) {
                return true;
            }
            at = at.iter().flat_map(|ty| self.part_types(*ty, seg)).collect();
        }
        false
    }

    /// `path` in the target of `reference`, canonical.
    fn canonical_through(&self, reference: ValueId, path: &[PathSeg]) -> Path {
        self.canonical(self.lent(self.body.ty_of(reference)), path)
    }

    /// The loans of the reference `target` is made through; none by name.
    fn through(&self, state: &State, target: &RefTarget) -> LoanSet {
        match target {
            RefTarget::Through(reference) => self.own(state, *reference),
            RefTarget::Var(_) | RefTarget::Param(_) | RefTarget::Context(_) => LoanSet::new(),
        }
    }

    /// The storages a reference names, at its own position.
    fn named(&self, state: &State, reference: ValueId) -> BTreeSet<Storage> {
        self.own(state, reference).iter().map(|loan| loan.storage).collect()
    }

    /// What a value of `width` positions, of type `ty` where it is known,
    /// read out of `place` holds: aligned where a place is exact and names a
    /// part as wide as the part of the value it supplies, else every loan
    /// under the place at each of them; every loan of the content where the
    /// place is not located.
    fn read(&self, state: &State, place: &[Located], width: usize, ty: Option<Ty>) -> Vec<LoanSet> {
        let mut out = empty(width);
        for located in place {
            let content = self.content(state, located.storage);
            let into = self.supplied(ty, width, &located.supplies).unwrap_or_else(|| vec![0..width]);
            let region = self.region(located.storage, &located.path, content.len());
            for into in into {
                match &region {
                    Some(region) => {
                        for range in &region.ranges {
                            let held = &content[range.clone()];
                            match located.exact {
                                true => join_aligned_or_spread(&mut out[into.clone()], held),
                                false => spread(&mut out[into.clone()], &fold(held)),
                            }
                        }
                    }
                    None => spread(&mut out[into.clone()], &fold(content)),
                }
            }
        }
        out
    }

    /// `value`, of type `ty` where it is known, written into `place`: it
    /// replaces what the one part of the one storage `place` names exactly
    /// held, and joins into every part it may name otherwise.
    fn write(&self, state: &mut State, place: &[Located], ty: Option<Ty>, value: &[LoanSet]) {
        if let [located] = place
            && located.exact
            && located.supplies.is_empty()
        {
            let width = self.content(state, located.storage).len();
            if let Some(region) = self.region(located.storage, &located.path, width)
                && region.exact
                && let [range] = &region.ranges[..]
                && range.len() == value.len()
            {
                let content = state.storages.get_mut(&located.storage).expect("a written storage is present");
                content[range.clone()].clone_from_slice(value);
                return;
            }
        }
        self.join_into(state, place, ty, value);
    }

    /// `value`, of type `ty` where it is known, joined into every part
    /// `place` may name: the part of the value each place supplies, aligned
    /// where the place is exact and the two have as many positions, and into
    /// each position under the place otherwise; into every position of the
    /// content where the place is not located.
    fn join_into(&self, state: &mut State, place: &[Located], ty: Option<Ty>, value: &[LoanSet]) {
        for located in place {
            let supplied: Vec<LoanSet> = match self.supplied(ty, value.len(), &located.supplies) {
                Some(ranges) if ranges.len() == 1 => value[ranges[0].clone()].to_vec(),
                Some(ranges) => vec![ranges.into_iter().flat_map(|range| fold(&value[range])).collect()],
                None => vec![fold(value)],
            };
            let width = self.content(state, located.storage).len();
            let region = self.region(located.storage, &located.path, width);
            let content = state.storages.get_mut(&located.storage).expect("a written storage is present");
            match region {
                Some(region) => {
                    for range in region.ranges {
                        match located.exact {
                            true => join_aligned_or_spread(&mut content[range], &supplied),
                            false => spread(&mut content[range], &fold(&supplied)),
                        }
                    }
                }
                None => spread(content, &fold(&supplied)),
            }
        }
    }

    /// `positions`, a value of type `ty`'s, with what its references and
    /// captures reach read from the places they name now, outer positions
    /// first so that an inner one sees what the outer read brought.
    fn deepened(&self, state: &State, ty: Ty, mut positions: Vec<LoanSet>) -> Vec<LoanSet> {
        for (at, kind) in self.kinds(ty).into_iter().enumerate() {
            let named = through_loans(&positions[at], &[]);
            match kind {
                PositionKind::Ref(_, width) => {
                    let reached = self.read(state, &named, width, None);
                    join_positions(&mut positions[at + 1..at + 1 + width], &reached);
                }
                PositionKind::Captures(_) => {
                    let [reached] = &self.read(state, &named, 1, None)[..] else { unreachable!("one position was read") };
                    positions[at].extend(reached.iter().cloned());
                }
                PositionKind::Lifetime | PositionKind::InFlight => {}
            }
        }
        positions
    }

    /// The loans a reborrow of `path` at `mode` through a reference holding
    /// `source` makes at `made`, each within the loan of `source` it is
    /// made through (ownership §5).
    fn reborrow_loans(&self, source: &LoanSet, path: &[PathSeg], mode: Mode, made: Site, record: &mut Record) -> LoanSet {
        let mut own = LoanSet::new();
        for loan in source {
            let Some(reborrowed) = loan.reborrowed(path, mode, made) else { continue };
            record.made_within(&reborrowed, loan);
            own.insert(reborrowed);
        }
        own
    }

    /// A use of a reference `stored` in a place a path of references reaches,
    /// the first of which holds `first`: a reborrow of it made at `made`, of
    /// mode `mode`, each loan within the stored one and within `first`, and
    /// exclusive only where all of them are (ownership §5); a lend there.
    fn reborrow_stored(&self, stored: &LoanSet, first: &LoanSet, mode: Mode, made: Site, record: &mut Record) -> LoanSet {
        let mode = first.iter().map(|loan| loan.mode).fold(mode, Mode::min);
        let mut own = LoanSet::new();
        for loan in stored {
            let reborrowed = loan.reborrowed(&[], mode, made).expect("a reborrow of the whole meets every part");
            record.made_within(&reborrowed, loan);
            for outer in first {
                record.made_within(&reborrowed, outer);
            }
            own.insert(reborrowed);
        }
        let mut through = stored.clone();
        through.extend(first.iter().cloned());
        record.lend(made, None, own.clone(), through);
        own
    }

    /// A reborrow of the whole of what `reference` lends, of mode `mode`,
    /// made at `made`, a lend there: its own loans, and what its target
    /// holds.
    fn reborrow(&self, state: &State, reference: ValueId, mode: Mode, made: Site, record: &mut Record) -> Vec<LoanSet> {
        let source = self.own(state, reference);
        let own = self.reborrow_loans(&source, &[], mode, made, record);
        record.lend(made, None, own.clone(), source);
        let mut positions = vec![own];
        positions.extend(self.target(state, reference));
        positions
    }

    /// What `reference`'s target holds: what it holds at its target's
    /// positions, and what the places it names hold now.
    fn target(&self, state: &State, reference: ValueId) -> Vec<LoanSet> {
        let held = self.value(state, reference);
        let target_ty = self.lent(self.body.ty_of(reference));
        let mut target = self.read(state, &self.place_through(state, reference, &[]), held.len() - 1, Some(target_ty));
        join_positions(&mut target, &held[1..]);
        target
    }

    /// `value` taken at a by-value position of `site`: a `&mut` the
    /// takings read there is reborrowed, a loan made there within each of
    /// its own, and a lend (ownership §5); any other value is as it is.
    fn passed(&self, state: &State, value: ValueId, site: Site, on_edge: Option<usize>, record: &mut Record) -> Vec<LoanSet> {
        let mut positions = self.value(state, value);
        let ty = self.body.ty_of(value);
        if by_value(self.types, ty) == Taken::Read && self.is_mut_ref(ty) {
            let source = positions[0].clone();
            let own = self.reborrow_loans(&source, &[], Mode::Exclusive, site, record);
            record.lend(site, on_edge, own.clone(), source);
            positions[0] = own;
        }
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

    fn step(&self, inst: &Inst, state: &mut State, record: &mut Record) {
        self.page_of_inst(inst).record(Site::Inst(inst.id()), record);
        if let Some(positions) = self.result(inst, state, record) {
            let result = inst.result().expect("an operation with loans yields a value");
            self.define(state, result, positions);
        }
        if let Some(token) = inst.order().and_then(|order| order.yielded()) {
            self.define(state, token, Vec::new());
        }
    }

    /// The loans each position of an instruction's result holds, the writes
    /// it makes, and the accesses it records.
    fn result(&self, inst: &Inst, state: &mut State, record: &mut Record) -> Option<Vec<LoanSet>> {
        let site = Site::Inst(inst.id());
        let result_ty = inst.result().map(|result| self.body.ty_of(result));
        let none = || result_ty.map(|ty| empty(self.width(ty)));
        let ty = || result_ty.expect("the operation yields a value");
        let made = || self.made_at(inst);
        Some(match inst.op() {
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
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
            Op::StringEq { a, b } => {
                for value in [*a, *b] {
                    self.read_through_if_reference(state, value, site, record);
                }
                return none();
            }
            Op::StringClone { src } => {
                self.read_through_if_reference(state, *src, site, record);
                return none();
            }
            Op::Check { .. } | Op::CheckSteps { .. } | Op::Drop { .. } => return None,
            Op::StringAppend { target, .. } => {
                let place = self.place_through(state, *target, &[]);
                record.through(site, Reach::Write, &place, self.own(state, *target));
                return None;
            }
            Op::Ref { .. } => self.lend(state, inst, record),
            Op::Take { target, path, .. } => {
                let place = self.place(state, target, path);
                let reach = match place_taking(self.body, self.types, inst).expect("a Take takes its place") {
                    Taken::Read => Reach::Read,
                    Taken::Moved | Taken::PartMoved => Reach::Move,
                };
                record.through(site, reach, &place, self.through(state, target));
                let mut positions = self.read(state, &place, self.width(ty()), Some(ty()));
                if let RefTarget::Through(reference) = target
                    && let TyKind::Ref(mutability, _) = self.types.kind(ty())
                {
                    let path = self.canonical_through(*reference, path);
                    let first: LoanSet = self.own(state, *reference).into_iter().filter(|loan| loan.resolve(&path).is_some()).collect();
                    positions[0] = self.reborrow_stored(&positions[0], &first, mode(*mutability), site, record);
                }
                positions
            }
            Op::Assign { target, path, value, .. } => {
                let place = self.place(state, target, path);
                record.through(site, Reach::Write, &place, self.through(state, target));
                let written = self.passed(state, *value, site, None, record);
                self.write(state, &place, Some(self.body.ty_of(*value)), &written);
                return None;
            }
            Op::AsSlice { container: lent, mutability, .. } | Op::ArrayView { src: lent, mutability } => {
                self.reborrow(state, *lent, mode(*mutability), made(), record)
            }
            Op::Index { slice, mode: IndexMode::Ref, .. } => {
                let held = match self.types.kind(self.body.ty_of(*slice)) {
                    TyKind::Ref(held, _) => mode(*held),
                    other => panic!("an index reads a slice reference, not {other:?}"),
                };
                self.reborrow(state, *slice, held, made(), record)
            }
            Op::Index { slice, mode: IndexMode::Copy, .. } => {
                let place = self.place_through(state, *slice, &[]);
                record.through(site, Reach::Read, &place, self.own(state, *slice));
                self.target(state, *slice)
            }
            Op::IndexSet { slice, value, .. } => {
                let place = self.place_through(state, *slice, &[]);
                record.through(site, Reach::Write, &place, self.own(state, *slice));
                let written = self.passed(state, *value, site, None, record);
                self.join_into(state, &place, None, &written);
                return None;
            }
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
                let reached = self.reach_value(self.body.ty_of(*object), &path);
                let [range] = &self.parts_of(&reached, self.body.ty_of(*value))[..] else {
                    panic!("a field path names one field")
                };
                positions[range.clone()].clone_from_slice(&self.passed(state, *value, site, None, record));
                positions
            }
            Op::ArrayPush { array, value } => {
                let mut positions = self.passed(state, *array, site, None, record);
                join_positions(&mut positions, &self.passed(state, *value, site, None, record));
                positions
            }
            Op::MakeTuple { elements } => {
                elements.iter().flat_map(|element| self.passed(state, *element, site, None, record)).collect()
            }
            Op::Intrinsic { intrinsic: Intrinsic::Gather, args } => {
                let [structure] = &args[..] else { panic!("`Gather` takes `[structure]`") };
                let held = self.passed(state, *structure, site, None, record);
                self.gather(self.body.ty_of(*structure), &held, ty())
            }
            Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args } => self.array_map(state, args, site, record),
            Op::Intrinsic { intrinsic: Intrinsic::StringClone, .. } => {
                panic!("`StringClone` is no `Op::Intrinsic`: `Intrinsic` states no arguments for it")
            }
            Op::MakeObject { fields } => {
                let TyKind::Object(object) = self.types.kind(ty()) else {
                    panic!("an object is made at an object type")
                };
                let mut passed: BTreeMap<acvus_utils::Astr, Vec<LoanSet>> = BTreeMap::new();
                for (name, value) in fields {
                    passed.insert(*name, self.passed(state, *value, site, None, record));
                }
                object
                    .fields()
                    .iter()
                    .flat_map(|(name, _)| passed.remove(name).expect("an object is made with every field of its type"))
                    .collect()
            }
            Op::MakeVariant { tag, payload } => {
                let mut positions = empty(self.width(ty()));
                if let Some(payload) = payload {
                    let held = self.passed(state, *payload, site, None, record);
                    for range in self.variant_ranges(ty(), *tag, self.body.ty_of(*payload)) {
                        join_positions(&mut positions[range], &held);
                    }
                }
                positions
            }
            Op::MakeClosure { captures, .. } => {
                vec![captures.iter().flat_map(|capture| fold(&self.passed(state, *capture, site, None, record))).collect()]
            }
            // The same environment at fewer signatures: its one position
            // holds what the closure's does.
            Op::Specialize { closure, .. } => vec![fold(&self.value(state, *closure))],
            Op::FunctionCall { args, .. } => {
                let passed = self.arguments(state, args, site, record);
                self.call(state, inst, &passed, record)
            }
            // A started call holds what its arguments and its callee value
            // hold until it is taken.
            Op::Spawn { callee, args, .. } => {
                let passed = self.arguments(state, args, site, record);
                let mut held: LoanSet = passed.iter().flat_map(|positions| fold(positions)).collect();
                for function in indirect(callee) {
                    held.extend(fold(&self.deepened(state, self.body.ty_of(*function), self.value(state, *function))));
                }
                let result = self.call(state, inst, &passed, record);
                std::iter::once(held).chain(result).collect()
            }
            Op::TakeEffect { src } => self.value(state, *src)[1..].to_vec(),
        })
    }

    /// A read of what `value` names, where it is a reference.
    fn read_through_if_reference(&self, state: &State, value: ValueId, site: Site, record: &mut Record) {
        if let TyKind::Ref(..) = self.types.kind(self.body.ty_of(value)) {
            record.through(site, Reach::Read, &self.place_through(state, value, &[]), self.own(state, value));
        }
    }

    /// Where the loans `inst` makes its result with are made: at the call
    /// the result is only an argument of, else at `inst` (ownership §5,
    /// effects §6).
    fn made_at(&self, inst: &Inst) -> Site {
        inst.result().and_then(|result| self.sole_call.get(&result).copied()).unwrap_or(Site::Inst(inst.id()))
    }

    /// The reference a `Ref` yields, its loans made where `made_at` says
    /// (ownership §5): through a reference, or by name of a place whose type
    /// is a reference, a reborrow; by name of any other place, a lend of it.
    /// Its lend is the one access it makes: it reaches its place by lending
    /// it, and reads no content.
    fn lend(&self, state: &State, inst: &Inst, record: &mut Record) -> Vec<LoanSet> {
        let Op::Ref { target, path, mutability } = inst.op() else {
            panic!("a lend is a `Ref`, not {:?}", inst.op());
        };
        let (made, mode) = (self.made_at(inst), mode(*mutability));
        let lent = self.lent(self.body.ty_of(inst.result().expect("a `Ref` yields a reference")));
        let lent_width = self.width(lent);
        let (storage, storage_ty) = match target {
            RefTarget::Through(reference) => {
                let source = self.own(state, *reference);
                let own = self.reborrow_loans(&source, &self.canonical_through(*reference, path), mode, made, record);
                record.lend(made, None, own.clone(), source);
                let place = self.place_through(state, *reference, path);
                let mut positions = vec![own];
                positions.extend(self.read(state, &place, lent_width, Some(lent)));
                return positions;
            }
            RefTarget::Var(slot) => (Storage::Slot(*slot), self.body.ty_of(*slot)),
            RefTarget::Param(param) => (Storage::Param(*param), self.body.ty_of(*param)),
            RefTarget::Context(context) => (Storage::Context(*context), self.body.context_ty(*context)),
        };
        let path = &self.canonical(storage_ty, path);
        let place = [Located::by_name(storage, path)];
        let named = self.reach_value(storage_ty, path);
        let reference = named.parts.iter().find(|part| matches!(self.types.kind(part.ty), TyKind::Ref(..)));
        match reference {
            Some(reference) => {
                let held = self.read(state, &place, self.width(reference.ty), Some(reference.ty));
                let source = held[0].clone();
                let own = self.reborrow_loans(&source, &[], mode, made, record);
                record.lend(made, None, own.clone(), source);
                let mut positions = vec![own];
                positions.extend(held[1..].iter().cloned());
                positions
            }
            None => {
                let loan = Loan { storage, path: path.to_vec(), exact: true, mode, at: Vec::new(), made: Made::At(made) };
                let own = LoanSet::from([loan]);
                record.lend(made, None, own.clone(), LoanSet::new());
                let mut positions = vec![own];
                positions.extend(self.read(state, &place, lent_width, Some(lent)));
                positions
            }
        }
    }

    /// The reference of type `result` `Gather` builds of a structure of
    /// type `structure` holding `held`: each part reference's own loans at
    /// its part, and what its target holds at that part's positions
    /// (ownership §5). No loan is made.
    fn gather(&self, structure_ty: Ty, held: &[LoanSet], result: Ty) -> Vec<LoanSet> {
        let target = self.lent(result);
        let parts_of = |ty: Ty| {
            structure(self.types, ty)
                .unwrap_or_else(|| panic!("`Gather` gathers a transparent structured type, not {:?}", self.types.kind(ty)))
                .parts()
        };
        let (gathered, whole) = (parts_of(structure_ty), parts_of(target));
        let mut own = LoanSet::new();
        let mut targets = empty(self.width(target));
        let (mut from, mut into) = (0, 0);
        for (part, of) in gathered.iter().zip(whole.iter()) {
            let (part_width, target_width) = (self.width(part.ty), self.width(of.ty));
            let reference = &held[from..from + part_width];
            let seg = part_seg(of.key);
            own.extend(reference[0].iter().map(|loan| loan.at_part(seg)));
            join_positions(&mut targets[into..into + target_width], &reference[1..]);
            from += part_width;
            into += target_width;
        }
        std::iter::once(own).chain(targets).collect()
    }

    /// The part of `value` at `path` of type `ty`, joined over every part
    /// the path may name.
    fn project(&self, state: &State, value: ValueId, path: &[PathSeg], ty: Ty) -> Vec<LoanSet> {
        let positions = self.value(state, value);
        let reached = self.reach_value(self.body.ty_of(value), path);
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
            TyKind::Option(.., _) | TyKind::Result(..) => self.parts_of(&self.reach_value(ty, &[PathSeg::Payload]), payload),
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

    /// Each argument of the call at `site` as the call takes it: passed by
    /// value, so that a `&mut` is reborrowed there, and deepened through the
    /// references it holds.
    fn arguments(&self, state: &State, args: &[ValueId], site: Site, record: &mut Record) -> Vec<Vec<LoanSet>> {
        args.iter()
            .map(|arg| {
                let positions = self.passed(state, *arg, site, None, record);
                self.deepened(state, self.body.ty_of(*arg), positions)
            })
            .collect()
    }

    /// A call's result, and its writes through the arguments its flows name
    /// as outputs (ownership §5, extern-fn §4), its arguments taken as
    /// `passed`. Each loan the result takes from an argument is made at the
    /// call, within that argument's loan.
    fn call(&self, state: &mut State, inst: &Inst, passed: &[Vec<LoanSet>], record: &mut Record) -> Vec<LoanSet> {
        let (Op::FunctionCall { callee, callee_ty, args } | Op::Spawn { callee, callee_ty, args }) = inst.op() else {
            panic!("a call is a `FunctionCall` or a `Spawn`, not {:?}", inst.op());
        };
        let site = Site::Inst(inst.id());
        let TyKind::Fn { params, ret, flows, .. } = self.types.kind(*callee_ty) else {
            panic!("a call's callee type is a function type")
        };
        let flows = flows.get();
        let captured: Vec<LoanSet> = indirect(callee)
            .map(|function| fold(&self.deepened(state, self.body.ty_of(*function), self.value(state, *function))))
            .collect();
        let result = self.flowed_result(flows, params, *ret, passed, &captured);
        let result = self.taken_by_result(result, *ret, site, record);
        let callee_value = indirect(callee).next().copied();
        for (to, region) in self.flowed_writes(flows, self.access(callee), passed, &captured, callee_value.is_some()) {
            let into = match to {
                FlowEnd::Param(index) => args[index],
                FlowEnd::Captures => callee_value.expect("only a callee value captures"),
                FlowEnd::Result => panic!("the result is not written into"),
            };
            self.write_into(state, into, &region);
        }
        result
    }

    /// `ArrayMap`'s result: `f` applied through its flows once to the one
    /// element part of each array (ownership §5), what it writes written
    /// through each element and into its captures, and each loan the result
    /// takes made here.
    fn array_map(&self, state: &mut State, args: &[ValueId], site: Site, record: &mut Record) -> Vec<LoanSet> {
        let [arrays @ .., function] = args else { panic!("`ArrayMap` takes `[a_1, …, a_k, f]`") };
        let function_ty = self.body.ty_of(*function);
        let TyKind::Fn { params, ret, flows, .. } = self.types.kind(function_ty) else {
            panic!("`ArrayMap`'s `f` is a function value")
        };
        let flows = flows.get();
        let elements: Vec<Vec<LoanSet>> =
            arrays.iter().zip(params.iter()).map(|(array, param)| self.element(state, *array, *param, site, record)).collect();
        let passed: Vec<Vec<LoanSet>> =
            elements.iter().zip(params.iter()).map(|(element, param)| self.deepened(state, *param, element.clone())).collect();
        let captured = vec![fold(&self.deepened(state, function_ty, self.value(state, *function)))];
        let result = self.flowed_result(flows, params, *ret, &passed, &captured);
        let result = self.taken_by_result(result, *ret, site, record);
        for (to, region) in self.flowed_writes(flows, None, &passed, &captured, true) {
            match to {
                FlowEnd::Param(index) => {
                    self.write_through(state, params[index], &elements[index], &region);
                }
                FlowEnd::Captures => self.write_into(state, *function, &region),
                FlowEnd::Result => panic!("the result is not written into"),
            }
        }
        result
    }

    /// What `ArrayMap` passes `f` of `array`'s one element part, as a value
    /// of `param`: an array's element as it holds it; through a reference,
    /// the element lent at the reference's mutability, or where the element
    /// is itself a reference, that reference reborrowed (`Intrinsic`,
    /// ownership §5).
    fn element(&self, state: &State, array: ValueId, param: Ty, site: Site, record: &mut Record) -> Vec<LoanSet> {
        let passed = self.passed(state, array, site, None, record);
        let TyKind::Ref(mutability, lent) = self.types.kind(self.body.ty_of(array)) else {
            return passed;
        };
        let element_ty = match self.types.kind(lent.ty()) {
            TyKind::Array(element, _, _) | TyKind::Slice(element) => *element,
            other => panic!("`ArrayMap` maps arrays or references to them, not {other:?}"),
        };
        let first = passed[0].clone();
        let place = through_loans(&first, &[ELEMENT]);
        let element = match self.types.kind(element_ty) {
            TyKind::Ref(..) => {
                let mut stored = self.read(state, &place, self.width(element_ty), Some(element_ty));
                stored[0] = self.reborrow_stored(&stored[0], &first, mode(*mutability), site, record);
                stored
            }
            _ => {
                let own = self.reborrow_loans(&first, &[ELEMENT], mode(*mutability), site, record);
                record.lend(site, None, own.clone(), first);
                std::iter::once(own).chain(self.read(state, &place, self.width(element_ty), Some(element_ty))).collect()
            }
        };
        assert_eq!(element.len(), self.width(param), "`f` takes the element as `ArrayMap` passes it");
        element
    }

    /// What a callee's result of type `ret` takes by `flows` from its
    /// arguments, `passed` as parameters of `params`, and its captures.
    fn flowed_result(&self, flows: &Flows, params: &[Ty], ret: Ty, passed: &[Vec<LoanSet>], captured: &[LoanSet]) -> Vec<LoanSet> {
        let mut result = empty(self.width(ret));
        for Source { from, alignment } in flows.into_end(FlowEnd::Result, passed.len()) {
            let taken = flowed_input(from, passed, captured);
            match &alignment {
                Alignment::Aligned => {
                    join_positions(&mut result, &taken);
                }
                Alignment::Any => spread(&mut result, &fold(&taken)),
                Alignment::Labelled(labelled) => {
                    let FlowEnd::Param(index) = from else {
                        panic!("a Labelled flow's input is a parameter of the extern it is read off")
                    };
                    self.join_labelled(&mut result, labelled, params[index], ret, from, &taken);
                }
            }
        }
        result
    }

    /// What a callee writes by `flows` through each argument and into its
    /// captures, where it has any, as `access` allows where one is declared.
    fn flowed_writes(
        &self,
        flows: &Flows,
        access: Option<&[Touch]>,
        passed: &[Vec<LoanSet>],
        captured: &[LoanSet],
        captures: bool,
    ) -> Vec<(FlowEnd, LoanSet)> {
        let arity = passed.len();
        let written_by_access = |to: &FlowEnd| match (access, to) {
            (None, _) => true,
            (Some(touches), FlowEnd::Param(index)) => {
                touches.iter().any(|touch| touch.place.param == *index && touch.mode.writes())
            }
            (Some(_), FlowEnd::Captures | FlowEnd::Result) => false,
        };
        let outputs = (0..arity).map(FlowEnd::Param).chain(captures.then_some(FlowEnd::Captures));
        outputs
            .filter(written_by_access)
            .filter_map(|to| {
                let sources = flows.into_end(to, arity);
                if sources.is_empty() {
                    return None;
                }
                let region = sources.into_iter().flat_map(|Source { from, .. }| fold(&flowed_input(from, passed, captured))).collect();
                Some((to, region))
            })
            .collect()
    }

    /// The loans a call's result of type `ty` takes, `taken` at each
    /// position, each made at the call within the argument's loan it is
    /// taken from, not exact; a lend there.
    fn taken_by_result(&self, taken: Vec<LoanSet>, ty: Ty, site: Site, record: &mut Record) -> Vec<LoanSet> {
        let mut sources = LoanSet::new();
        let mut made = LoanSet::new();
        let mut result = Vec::with_capacity(taken.len());
        for (loans, kind) in taken.into_iter().zip(self.kinds(ty)) {
            let held = match kind {
                PositionKind::Ref(mutability, _) => Some(mode(mutability)),
                PositionKind::Captures(Holds::Shared) => Some(Mode::Shared),
                PositionKind::Captures(Holds::Mut | Holds::Nothing) | PositionKind::Lifetime | PositionKind::InFlight => None,
            };
            let mut position = LoanSet::new();
            for loan in loans {
                let new = loan.taken_by_result(held, site);
                record.made_within(&new, &loan);
                made.insert(new.clone());
                position.insert(new);
                sources.insert(loan);
            }
            result.push(position);
        }
        record.lend(site, None, made, sources);
        result
    }

    /// `region` written through `value` as `write_through` writes it, and
    /// `value` holding it at the pointee positions of its `&mut` positions.
    fn write_into(&self, state: &mut State, value: ValueId, region: &LoanSet) {
        let own = self.write_through(state, self.body.ty_of(value), &self.value(state, value), region);
        self.define(state, value, own);
    }

    /// `region` written through every `&mut` position of a value of `ty`
    /// holding `positions` into the places its loans name, and into those
    /// its mutable captures name; `positions` with `region` joined into the
    /// pointee positions of its `&mut` positions.
    fn write_through(&self, state: &mut State, ty: Ty, positions: &[LoanSet], region: &LoanSet) -> Vec<LoanSet> {
        let held = self.deepened(state, ty, positions.to_vec());
        let mut own = positions.to_vec();
        for (at, kind) in self.kinds(ty).into_iter().enumerate() {
            let pointee = match kind {
                PositionKind::Ref(Mutability::Mut, width) => Some(width),
                PositionKind::Captures(Holds::Mut) => None,
                PositionKind::Ref(Mutability::Shared, _)
                | PositionKind::Captures(Holds::Shared | Holds::Nothing)
                | PositionKind::Lifetime
                | PositionKind::InFlight => continue,
            };
            let exclusive: LoanSet = held[at].iter().filter(|loan| loan.mode == Mode::Exclusive).cloned().collect();
            let written = through_loans(&exclusive, &[]);
            self.join_into(state, &written, None, std::slice::from_ref(region));
            if let Some(width) = pointee {
                own[at + 1..at + 1 + width].iter_mut().for_each(|position| position.extend(region.iter().cloned()));
            }
        }
        own
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
                    out[to.clone()].iter_mut().for_each(|position| position.extend(all.iter().cloned()));
                }
            }
        }
    }

    /// The storages an instruction reads and writes, in `state`, just
    /// before it runs.
    fn touched(&self, inst: &Inst, state: &State) -> Touched {
        self.touched_by_operation(inst, state).seeing(&self.page_of_inst(inst))
    }

    /// What `inst` touches by its operation and its operands, apart from
    /// the page.
    fn touched_by_operation(&self, inst: &Inst, state: &State) -> Touched {
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
        let mut by_arguments = |values: &mut dyn Iterator<Item = ValueId>| {
            for value in values {
                let ty = self.body.ty_of(value);
                let held = self.deepened(state, ty, self.value(state, value));
                touched.reads.extend(held.iter().flatten().map(|loan| loan.storage));
                for (at, kind) in self.kinds(ty).into_iter().enumerate() {
                    if matches!(kind, PositionKind::Ref(Mutability::Mut, _) | PositionKind::Captures(Holds::Mut)) {
                        let written = held[at].iter().filter(|loan| loan.mode == Mode::Exclusive);
                        touched.writes.extend(written.map(|loan| loan.storage));
                    }
                }
            }
        };
        match inst.op() {
            // `ArrayMap` calls `f` on the elements at each index: it touches
            // what its arrays and `f` hold as a call of `f` with them would.
            Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, args } => by_arguments(&mut args.iter().copied()),
            Op::FunctionCall { callee, args, .. } | Op::Spawn { callee, args, .. } => {
                by_arguments(&mut args.iter().chain(indirect(callee)).copied())
            }
            _ => {}
        }
        match inst.op() {
            Op::Take { target, path, .. } => touched.reads = storages_of(&self.place(state, target, path)),
            Op::Assign { target, path, .. } => touched.writes = storages_of(&self.place(state, target, path)),
            Op::Index { slice, .. } => touched.reads = self.named(state, *slice),
            Op::IndexSet { slice, .. } => touched.writes = self.named(state, *slice),
            Op::StringAppend { target, part } => {
                touched.writes = self.named(state, *target);
                touched.reads = named_if_reference(*part);
            }
            Op::StringEq { a, b } => {
                touched.reads = named_if_reference(*a);
                touched.reads.extend(named_if_reference(*b));
            }
            Op::StringClone { src } => touched.reads = named_if_reference(*src),
            Op::Intrinsic { intrinsic: Intrinsic::ArrayMap, .. } | Op::FunctionCall { .. } | Op::Spawn { .. } => {}
            Op::Const { .. }
            | Op::ConstStr { .. }
            | Op::StringConcat { .. }
            | Op::Ref { .. }
            | Op::AsSlice { .. }
            | Op::ArrayView { .. }
            | Op::SliceLen { .. }
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
            | Op::Intrinsic { .. }
            | Op::TupleIndex { .. }
            | Op::TestLiteral { .. }
            | Op::TestObjectKey { .. }
            | Op::ArrayIndex { .. }
            | Op::ObjectGet { .. }
            | Op::MakeClosure { .. }
            | Op::Specialize { .. }
            | Op::MakeVariant { .. }
            | Op::TestVariant { .. }
            | Op::UnwrapVariant { .. }
            | Op::Undef
            | Op::Drop { .. } => {}
        }
        Touched::whole(touched.reads, touched.writes)
    }

    /// A `For` over a slice reads its source's storage, and a `Return` the
    /// page; no other terminator touches storage.
    fn touched_by_terminator(&self, terminator: &Terminator, state: &State) -> Touched {
        let reads = match terminator {
            Terminator::For { source: ForSource::Slice(slice) | ForSource::SliceMut(slice), .. } => {
                self.named(state, *slice)
            }
            _ => BTreeSet::new(),
        };
        Touched::whole(reads, BTreeSet::new()).seeing(&self.page_of_terminator(terminator))
    }

    /// A `For` fills its body's leading parameters: a slice's element is a
    /// reborrow of what the source lends, made at the `For`, an array's is
    /// moved out of it, and the counter holds none; an exit's trip count
    /// holds none. A `Return` passes its value out by value.
    fn step_terminator(&self, block: BlockId, terminator: &Terminator, state: &mut State, record: &mut Record) {
        let site = Site::Term(block);
        let (source, stages, exit, exit_trip) = match terminator {
            Terminator::For { source, stages, exit, exit_trip, .. } => (source, stages, exit, exit_trip),
            Terminator::Return { value, .. } => {
                self.passed(state, *value, site, None, record);
                self.page_of_terminator(terminator).record(site, record);
                return;
            }
            _ => return,
        };
        let params = self.body.block(stages.body()).params();
        let element = match source {
            ForSource::Slice(slice) => Some(self.reborrow(state, *slice, Mode::Shared, site, record)),
            ForSource::SliceMut(slice) => Some(self.reborrow(state, *slice, Mode::Exclusive, site, record)),
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

    /// Each argument of `edge`, the `on_edge`th out of `from`, passed by
    /// value into the parameter it fills.
    fn step_edge(
        &self,
        from: BlockId,
        on_edge: Option<usize>,
        edge: &Edge<'_>,
        target_params: &[ValueId],
        state: &mut State,
        record: &mut Record,
    ) {
        for (param, arg) in target_params[edge.supplied..].iter().zip(edge.args) {
            let positions = self.passed(state, *arg, Site::Term(from), on_edge, record);
            self.define(state, *param, positions);
        }
    }
}

/// The places an access of `path` in a reference's target reaches through
/// each of `loans` whose part meets it.
fn through_loans(loans: &LoanSet, path: &[PathSeg]) -> Vec<Located> {
    let located: BTreeSet<Located> = loans.iter().filter_map(|loan| loan.resolve(path)).collect();
    located.into_iter().collect()
}

/// What the input end `from` of a callee's flows holds: an argument
/// `passed`, or its captures.
fn flowed_input(from: FlowEnd, passed: &[Vec<LoanSet>], captured: &[LoanSet]) -> Vec<LoanSet> {
    match from {
        FlowEnd::Param(index) => passed.get(index).expect("a flow names a parameter the call passes").clone(),
        FlowEnd::Captures => captured.to_vec(),
        FlowEnd::Result => panic!("a call's result is an output only"),
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

    /// Every slot empty and every context holding no loan; every parameter
    /// and capture holding its outside storages, each by-value parameter's
    /// own storage holding the parameter, and each outside storage of a
    /// reference position holding what the parameter's positions under it
    /// name.
    fn entry(&self, body: &Body) -> State {
        let mut state = State::bottom();
        for slot in body.slots() {
            state.storages.insert(Storage::Slot(*slot), empty(self.width(body.ty_of(*slot))));
        }
        for named in body.contexts() {
            state.storages.insert(Storage::Context(named.context), empty(self.width(named.ty)));
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
        self.step(inst, state, &mut Record::default());
    }

    fn transfer_terminator(&self, block: BlockId, terminator: &Terminator, state: &mut State) {
        self.step_terminator(block, terminator, state, &mut Record::default());
    }

    fn transfer_edge(&self, from: BlockId, edge: &Edge<'_>, target_params: &[ValueId], state: &mut State) {
        self.step_edge(from, None, edge, target_params, state, &mut Record::default());
    }
}
