//! For each place the body tracks, a slot, a by-value parameter's own
//! storage, a context, or what a position of a parameter's or a capture's
//! type lends outside the body (`Storage::Outside`), whether its current
//! content may be read before a whole store replaces it or the body ends:
//! the dense engine, backward, over sets of places, each named by its
//! `Storage`. A site gens every place `Loans` says it reads, directly or
//! through a reference; a `Return` reads every context (`Loans`) and every
//! storage outside the body, which its caller holds (ownership §4). A
//! store into a whole place kills it, by name or through a reference that
//! reaches that place alone (`Loans::through_one`), as does any other
//! write `Loans` records by name of a whole place, such as a call's write
//! of a context it sees beyond the program (context §1). A store into part
//! of a place kills nothing. A store through a reference that may reach
//! several places reads each of them, since what it replaces there is
//! dropped at the store, and kills none: after it each still holds a
//! content that ends at its own end. The body's end reads none of its own
//! places. `stored` is what a store touches, for every reader.
//!
//! `Liveness` answers for values; this answers for what a place holds, which
//! a slot's or a parameter's own `ValueId` does not follow, since every
//! access names it.
//!
//! The same answer is kept per part (ownership §6), for each place whose
//! type does not copy: its units (`ir::parts`), cut where the body moves a
//! part that does not copy out of it, or stores one into part of it, by
//! name or through a reference to it alone. A unit is read where a `Take`
//! of an overlapping path by name reads it, and wherever `Loans` says the
//! place is read otherwise, since a read through a reference does not
//! locate its part, and a store through a reference that may reach several
//! places reads each unit overlapping what it may reach; a store into a
//! path covering the unit, by name or through a reference to the place
//! alone, kills it. Beside it, forward, each unit's presence: a store into
//! a path covering it, by name or through a reference to the place alone,
//! fills it, a store that may reach several places leaves each as it was,
//! and a `Take` that
//! moves a covering path out of it, by name or through a reference to it
//! alone, empties it; any other write by name of a whole place fills it. A
//! slot starts empty, and a by-value parameter's storage, a context and a
//! storage outside the body full.
//!
//! Cost: `Loans`', and then a block's state only grows, at most once per
//! place or unit, and each growth re-runs the transfer of the block, so
//! O((places + units) · (blocks + edges) · the largest block's sites · the
//! storages a site reads) for liveness and O(units · (blocks + edges) · the
//! largest block's sites) for presence, a unit's presence rising at most
//! three times. Finding the units is O(instructions) cuts and
//! `ir::parts::units` per place. The per-point sets take O(points ·
//! (places + units)) space. The widening is the join.
//!
//! passes-analyses.md.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Edge, Inst, InstId, Module, Op, Site, Terminator};
use crate::ir::parts::{self, Path, covers, overlap};
use crate::ir::{PathSeg, RefTarget, ValueId};
use crate::decl::Declarations;
use crate::ty::{Ty, lent_types};
use crate::passes::analysis::{Analysis, DenseAnalysis, Direction, Loans, Queries, Reach, Storage, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::ty::{Concrete, Types, copies};

pub struct ContentLiveness {
    /// Each instruction's block and place in it, as the body stood when the
    /// facts were computed; the facts are dropped when an instruction
    /// changes, so the places stay true while they are kept.
    position: FxHashMap<InstId, (BlockId, usize)>,
    /// Per block, the places live before each instruction, and last the
    /// places live before its terminator.
    before: FxHashMap<BlockId, Vec<FxHashSet<Storage>>>,
    at_end: FxHashMap<BlockId, FxHashSet<Storage>>,
    units: FxHashMap<Storage, Vec<Path>>,
    /// Per block, the units live before each instruction and before its
    /// terminator, and after it.
    parts_before: FxHashMap<BlockId, Vec<FxHashSet<UnitId>>>,
    parts_at_end: FxHashMap<BlockId, FxHashSet<UnitId>>,
    /// Per block, each unit's presence before each instruction and before
    /// its terminator.
    presence_before: FxHashMap<BlockId, Vec<PresenceOf>>,
}

/// A unit of a place: the place, and the unit's index among its units.
type UnitId = (Storage, usize);

/// Whether a unit of a place holds its content at a point, over the paths
/// that reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Unreached,
    Absent,
    Present,
    Maybe,
}

impl Presence {
    fn join(self, other: Presence) -> Presence {
        match (self, other) {
            (Presence::Unreached, other) | (other, Presence::Unreached) => other,
            (a, b) if a == b => a,
            _ => Presence::Maybe,
        }
    }
}

impl ContentLiveness {
    /// The places whose content may be read from just after the instruction
    /// at `site` runs, before a whole store replaces it or the body ends;
    /// for a terminator, from just after it, on the way into a successor.
    /// Cost: O(1).
    ///
    /// # Panics
    /// `site` is not a site of the body.
    pub fn live_after(&self, site: Site) -> &FxHashSet<Storage> {
        match site {
            Site::Inst(inst) => {
                let (block, index) = self.position_of(inst);
                &self.points(block)[index + 1]
            }
            Site::Term(block) => self.live_at_end(block),
        }
    }

    /// # Panics
    /// `inst` is not an instruction of the body the facts cover.
    fn position_of(&self, inst: InstId) -> (BlockId, usize) {
        *self.position.get(&inst).unwrap_or_else(|| panic!("{inst:?} is an instruction of the body the facts cover"))
    }

    /// The places whose content may be read from `block`'s start. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_start(&self, block: BlockId) -> &FxHashSet<Storage> {
        &self.points(block)[0]
    }

    /// The places whose content may be read from just before `inst`. Cost:
    /// O(1).
    ///
    /// # Panics
    /// `inst` is not an instruction of the body.
    pub fn live_before(&self, inst: InstId) -> &FxHashSet<Storage> {
        let (block, index) = self.position_of(inst);
        &self.points(block)[index]
    }

    /// The places whose content may be read from just before `block`'s
    /// terminator. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_before_terminator(&self, block: BlockId) -> &FxHashSet<Storage> {
        self.points(block).last().expect("a block's points end before its terminator")
    }

    /// The places whose content may be read after `block`'s terminator, on
    /// its way into a successor. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_end(&self, block: BlockId) -> &FxHashSet<Storage> {
        self.at_end.get(&block).expect("content liveness covers every block of its body")
    }

    fn points(&self, block: BlockId) -> &[FxHashSet<Storage>] {
        self.before.get(&block).expect("content liveness covers every block of its body")
    }

    /// The units of `place`, a slot, a by-value parameter's own storage, a
    /// context or a storage outside the body, whose type does not copy, in
    /// the order the body cuts them; `[[]]` where nothing cuts it. `None`
    /// for any other place. Cost: O(1).
    pub fn units(&self, place: Storage) -> Option<&[Path]> {
        self.units.get(&place).map(Vec::as_slice)
    }

    /// Whether a unit of `place` overlapping `path` may be read from just
    /// before `inst`, before a store replaces it or the body ends. Cost:
    /// O(units of `place`).
    ///
    /// # Panics
    /// `place` has no units, or `inst` is not an instruction of the body.
    pub fn part_live_before(&self, inst: InstId, place: Storage, path: &[PathSeg]) -> bool {
        let (block, index) = self.position_of(inst);
        self.part_live_at(block, index, place, path)
    }

    fn part_live_at(&self, block: BlockId, index: usize, place: Storage, path: &[PathSeg]) -> bool {
        let points = self.parts_before.get(&block).expect("content liveness covers every block of its body");
        let live = points.get(index).unwrap_or_else(|| panic!("{block:?} has no position {index}"));
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// As `part_live_before`, from just after the instruction or terminator
    /// at `site`. Cost: O(units of `place`).
    pub fn part_live_after(&self, site: Site, place: Storage, path: &[PathSeg]) -> bool {
        match site {
            Site::Inst(inst) => {
                let (block, index) = self.position_of(inst);
                self.part_live_at(block, index + 1, place, path)
            }
            Site::Term(block) => self.part_live_at_end(block, place, path),
        }
    }

    /// As `part_live_before`, from just before `block`'s terminator. Cost:
    /// O(units of `place`).
    pub fn part_live_before_terminator(&self, block: BlockId, place: Storage, path: &[PathSeg]) -> bool {
        let points = self.parts_before.get(&block).expect("content liveness covers every block of its body");
        let live = points.last().expect("a block's points end before its terminator");
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// As `part_live_before`, from `block`'s start. Cost: O(units of `place`).
    pub fn part_live_at_start(&self, block: BlockId, place: Storage, path: &[PathSeg]) -> bool {
        self.part_live_at(block, 0, place, path)
    }

    /// As `part_live_before`, after `block`'s terminator, on its way into a
    /// successor. Cost: O(units of `place`).
    pub fn part_live_at_end(&self, block: BlockId, place: Storage, path: &[PathSeg]) -> bool {
        let live = self.parts_at_end.get(&block).expect("content liveness covers every block of its body");
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// The presence of the unit `unit` of `place` just before `inst`. Cost:
    /// O(units of `place`).
    ///
    /// # Panics
    /// `unit` is not one of `place`'s units, or `inst` is not an
    /// instruction of the body.
    pub fn presence_before(&self, inst: InstId, place: Storage, unit: &[PathSeg]) -> Presence {
        let (block, index) = self.position_of(inst);
        let points = self.presence_before.get(&block).expect("content presence covers every block of its body");
        let state = points.get(index).unwrap_or_else(|| panic!("{block:?} has no position {index}"));
        state.of((place, self.unit_index(place, unit)))
    }

    /// The presence of `unit` of `place` after `block`'s terminator, which
    /// changes no presence. Cost: O(units of `place`).
    pub fn presence_at_end(&self, block: BlockId, place: Storage, unit: &[PathSeg]) -> Presence {
        let points = self.presence_before.get(&block).expect("content presence covers every block of its body");
        let state = points.last().expect("a block's points end before its terminator");
        state.of((place, self.unit_index(place, unit)))
    }

    fn unit_index(&self, place: Storage, unit: &[PathSeg]) -> usize {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?} has no units"));
        units
            .iter()
            .position(|known| known.as_slice() == unit)
            .unwrap_or_else(|| panic!("{unit:?} is not one of {place:?}'s units {units:?}"))
    }

    fn overlapping<'a>(&'a self, place: Storage, path: &'a [PathSeg]) -> impl Iterator<Item = UnitId> + 'a {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?} has no units"));
        units.iter().enumerate().filter(move |(_, unit)| overlap(unit, path)).map(move |(index, _)| (place, index))
    }
}

impl Analysis for ContentLiveness {
    type Facts = ContentLiveness;

    /// `Loans`' aspects: what a site reads is read from it.
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage, Aspect::Types];

    fn compute(queries: &mut Queries<'_, '_>, module: &Module, body: BodyId) -> ContentLiveness {
        let loans = queries.get::<Loans>(module, body);
        let body = module.body(body);
        let outside = outside_storages(body, module.types(), queries.decls());
        let units = units_of(body, module.types(), &outside, &loans);
        let returned: Vec<Storage> = outside.keys().copied().collect();
        let backward = Backward { loans: &loans, units: &units, returned: &returned };
        let facts = solve_dense(&backward, body);
        let mut position = FxHashMap::default();
        for (id, block) in body.blocks() {
            position.extend(block.insts().enumerate().map(|(index, inst)| (inst.id(), (id, index))));
        }
        let mut before = FxHashMap::default();
        let mut at_end = FxHashMap::default();
        let mut parts_before = FxHashMap::default();
        let mut parts_at_end = FxHashMap::default();
        for (id, block) in body.blocks() {
            let end = facts.at(id).clone();
            let mut state = end.clone();
            backward.transfer_terminator(id, block.terminator(), &mut state);
            let mut points = vec![state.clone()];
            for inst in block.insts().rev() {
                backward.transfer(inst, &mut state);
                points.push(state.clone());
            }
            points.reverse();
            let (places, parts): (Vec<_>, Vec<_>) = points.into_iter().map(|live| (live.places, live.parts)).unzip();
            before.insert(id, places);
            parts_before.insert(id, parts);
            at_end.insert(id, end.places);
            parts_at_end.insert(id, end.parts);
        }
        let forward = Forward { units: &units, types: module.types(), body, loans: &loans };
        let presence = solve_dense(&forward, body);
        let mut presence_before = FxHashMap::default();
        for (id, block) in body.blocks() {
            let mut state = presence.at(id).clone();
            let mut points = vec![state.clone()];
            for inst in block.insts() {
                forward.transfer(inst, &mut state);
                points.push(state.clone());
            }
            presence_before.insert(id, points);
        }
        ContentLiveness { position, before, at_end, units, parts_before, parts_at_end, presence_before }
    }
}

/// Every storage outside the body, one per position of each parameter's
/// and capture's type, with the type that position lends where it has one:
/// a function value's or a lifetime's position lends none.
fn outside_storages(body: &Body, types: &Types<Concrete>, decls: &Declarations) -> FxHashMap<Storage, Option<Ty>> {
    let holders = body.params().iter().chain(body.captures()).map(|(_, value)| *value).filter(|value| !body.is_token(*value));
    let mut outside = FxHashMap::default();
    for of in holders {
        for (position, lent) in lent_types(types, decls, body.ty_of(of)).into_iter().enumerate() {
            outside.insert(Storage::Outside { of, position }, lent);
        }
    }
    outside
}

/// The places whose type does not copy, each with its units: a slot, a
/// parameter the body names as a storage, a context the body names, and a
/// storage outside the body whose position lends a type.
fn units_of(
    body: &Body,
    types: &Types<Concrete>,
    outside: &FxHashMap<Storage, Option<Ty>>,
    loans: &Loans,
) -> FxHashMap<Storage, Vec<Path>> {
    let ty_of = |place: Storage| match place {
        Storage::Outside { .. } => outside.get(&place).copied().flatten(),
        Storage::Slot(_) | Storage::Param(_) | Storage::Context(_) => Some(place_ty(body, place)),
    };
    let mut cuts: FxHashMap<Storage, Vec<Path>> = FxHashMap::default();
    let slots = body.slots().iter().map(|slot| Storage::Slot(*slot));
    let contexts = body.contexts().iter().map(|named| Storage::Context(named.context));
    for place in slots.chain(contexts).chain(outside.keys().copied()) {
        if ty_of(place).is_some_and(|ty| !copies(types, ty)) {
            cuts.entry(place).or_default();
        }
    }
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            let (place, path, moves) = match (inst.op(), by_name(inst.op())) {
                (Op::Take { .. }, Some((place, path))) => {
                    let taken = inst.result().expect("a Take yields a value");
                    (place, parts::owned_region(path), !copies(types, body.ty_of(taken)))
                }
                (Op::Assign { value, .. }, _) => {
                    let Some(Stored::One { place, path }) = stored(loans, inst) else {
                        continue;
                    };
                    (place, parts::owned_region(path), !copies(types, body.ty_of(*value)))
                }
                (Op::Ref { .. }, Some((place, path))) => (place, path, false),
                (Op::Take { target: RefTarget::Through(_), .. }, None) => {
                    let Some((place, path)) = moved_through(loans, inst) else {
                        continue;
                    };
                    (place, path, true)
                }
                _ => continue,
            };
            let cut = moves && !path.is_empty();
            if ty_of(place).is_none_or(|ty| copies(types, ty)) {
                continue;
            }
            let place_cuts = cuts.entry(place).or_default();
            if cut && !place_cuts.iter().any(|known| known.as_slice() == path) {
                place_cuts.push(path.to_vec());
            }
        }
    }
    cuts.into_iter()
        .map(|(place, cuts)| {
            let ty = ty_of(place).expect("a place with units has a type");
            (place, parts::units(types, ty, &cuts))
        })
        .collect()
}

/// The place a `Take`, an `Assign` or a `Ref` names by name, and its path.
fn by_name(op: &Op) -> Option<(Storage, &[PathSeg])> {
    match op {
        Op::Take { target, path, .. } | Op::Assign { target, path, .. } | Op::Ref { target, path, .. } => {
            Storage::by_name(target).map(|place| (place, path.as_slice()))
        }
        _ => None,
    }
}

/// The type of `place`, a place the body names by name.
///
/// # Panics
/// `place` is storage outside the body, which no path by name reaches.
fn place_ty(body: &Body, place: Storage) -> Ty {
    match place {
        Storage::Slot(value) | Storage::Param(value) => body.ty_of(value),
        Storage::Context(context) => body.context_ty(context),
        Storage::Outside { .. } => panic!("{place:?} is outside the body, and no place by name"),
    }
}

/// Every write `Loans` records at `site` by name of a whole place.
fn whole_writes(loans: &Loans, site: Site) -> impl Iterator<Item = Storage> + '_ {
    let written = loans.accesses(site).iter().filter(|access| access.reach == Reach::Write && access.through.is_empty());
    written.flat_map(|access| &access.places).filter(|place| place.exact && place.path.is_empty()).map(|place| place.storage)
}

/// A set of places and a set of units; finite over a body, so its
/// widening is its join.
#[derive(Debug, Clone, PartialEq)]
struct Live {
    places: FxHashSet<Storage>,
    parts: FxHashSet<UnitId>,
}

impl JoinSemiLattice for Live {
    fn bottom() -> Self {
        Live { places: FxHashSet::default(), parts: FxHashSet::default() }
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = (self.places.len(), self.parts.len());
        self.places.extend(other.places.iter().copied());
        self.parts.extend(other.parts.iter().copied());
        (self.places.len(), self.parts.len()) != before
    }

    fn le(&self, other: &Self) -> bool {
        self.places.is_subset(&other.places) && self.parts.is_subset(&other.parts)
    }
}

impl Widen for Live {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

struct Backward<'f> {
    loans: &'f Loans,
    units: &'f FxHashMap<Storage, Vec<Path>>,
    /// The storages outside the body, which its caller reads once it
    /// returns.
    returned: &'f [Storage],
}

impl Backward<'_> {
    /// A store by name into `path` of `place`: the whole place where the
    /// path is empty, and each unit the path covers.
    fn kill(&self, place: Storage, path: &[PathSeg], state: &mut Live) {
        if path.is_empty() {
            state.places.remove(&place);
        }
        if let Some(units) = self.units.get(&place) {
            for (index, unit) in units.iter().enumerate() {
                if covers(parts::owned_region(path), unit) {
                    state.parts.remove(&(place, index));
                }
            }
        }
    }

    /// What `site` reads: the places `Loans` names, and of each, the units
    /// overlapping the path a `Take` names it by, or all of them.
    fn gen_reads(&self, site: Site, by_name: Option<(Storage, &[PathSeg])>, state: &mut Live) {
        for place in self.loans.reads(site).iter().copied() {
            let read: &[PathSeg] = match by_name {
                Some((named, path)) if named == place => path,
                _ => &[],
            };
            self.gen_read(place, read, state);
        }
    }

    /// A read of `path` of `place`: the place, and each unit overlapping
    /// the path.
    fn gen_read(&self, place: Storage, path: &[PathSeg], state: &mut Live) {
        state.places.insert(place);
        let Some(units) = self.units.get(&place) else {
            return;
        };
        let touched = units.iter().enumerate().filter(|(_, unit)| overlap(unit, path));
        state.parts.extend(touched.map(|(index, _)| (place, index)));
    }
}

impl DenseAnalysis for Backward<'_> {
    type State = Live;

    const DIRECTION: Direction = Direction::Backward;

    fn entry(&self, _: &Body) -> Live {
        Live::bottom()
    }

    fn transfer(&self, inst: &Inst, state: &mut Live) {
        let site = Site::Inst(inst.id());
        let stored = stored(self.loans, inst);
        let written = match &stored {
            Some(Stored::One { place, path }) => Some((*place, *path)),
            Some(Stored::Several(_)) | None => None,
        };
        let whole = whole_writes(self.loans, site).map(|place| (place, &[][..]));
        for (place, path) in written.into_iter().chain(whole) {
            self.kill(place, path, state);
        }
        let read_by_name = match (inst.op(), by_name(inst.op())) {
            (Op::Take { .. }, Some(read)) => Some(read),
            _ => None,
        };
        self.gen_reads(site, read_by_name, state);
        if let Some(Stored::Several(places)) = &stored {
            for (place, path) in places {
                self.gen_read(*place, path, state);
            }
        }
    }

    fn transfer_terminator(&self, block: BlockId, terminator: &Terminator, state: &mut Live) {
        let site = Site::Term(block);
        for place in whole_writes(self.loans, site) {
            self.kill(place, &[], state);
        }
        self.gen_reads(site, None, state);
        if matches!(terminator, Terminator::Return { .. }) {
            for place in self.returned {
                state.places.insert(*place);
                let units = self.units.get(place).map_or(0, Vec::len);
                state.parts.extend((0..units).map(|index| (*place, index)));
            }
        }
    }

    fn transfer_edge(&self, _: BlockId, _: &Edge<'_>, _: &[ValueId], _: &mut Live) {}
}

/// Each unit's presence; a unit not in the map is unreached.
#[derive(Debug, Clone, PartialEq, Default)]
struct PresenceOf(FxHashMap<UnitId, Presence>);

impl PresenceOf {
    fn of(&self, unit: UnitId) -> Presence {
        self.0.get(&unit).copied().unwrap_or(Presence::Unreached)
    }
}

impl JoinSemiLattice for PresenceOf {
    fn bottom() -> Self {
        PresenceOf::default()
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut rose = false;
        for (unit, presence) in &other.0 {
            let joined = self.of(*unit).join(*presence);
            if joined != self.of(*unit) {
                self.0.insert(*unit, joined);
                rose = true;
            }
        }
        rose
    }

    fn le(&self, other: &Self) -> bool {
        self.0.iter().all(|(unit, presence)| other.of(*unit).join(*presence) == other.of(*unit))
    }
}

/// Finite: four states per unit.
impl Widen for PresenceOf {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// Forward over the units: a slot's start empty, and a parameter's, a
/// context's and a storage outside the body's full; a store into one place
/// (`Stored::One`) fills what it covers, as does any other write by name of
/// a whole place, a store that may reach several places changes nothing,
/// and a `Take` by name of a path that does not copy, or one that moves
/// through a reference to one place, empties what it covers.
struct Forward<'f> {
    units: &'f FxHashMap<Storage, Vec<Path>>,
    types: &'f Types<Concrete>,
    body: &'f Body,
    loans: &'f Loans,
}

/// The place, and the region in it, that a `Take` through a reference
/// moves out of, where it reaches that place alone.
fn moved_through<'l>(loans: &'l Loans, inst: &Inst) -> Option<(Storage, &'l [PathSeg])> {
    let place = loans.through_one(Site::Inst(inst.id()), Reach::Move)?;
    Some((place.storage, parts::owned_region(&place.path)))
}

/// What a store writes.
#[derive(Debug, Clone)]
pub enum Stored<'a> {
    /// One place and the path in it: named, or reached through a reference
    /// that reaches that place alone (`Loans::through_one`).
    One { place: Storage, path: &'a [PathSeg] },
    /// The places a reference may reach, each a storage and a path from
    /// its root, where it reaches more than one place, or one that is not
    /// one place (`Loans::is_one`).
    Several(Vec<(Storage, &'a [PathSeg])>),
}

/// What `inst` writes where it is an `Assign`.
pub fn stored<'a>(loans: &'a Loans, inst: &'a Inst) -> Option<Stored<'a>> {
    let Op::Assign { target, path, .. } = inst.op() else {
        return None;
    };
    if let Some(place) = Storage::by_name(target) {
        return Some(Stored::One { place, path });
    }
    let site = Site::Inst(inst.id());
    if let Some(place) = loans.through_one(site, Reach::Write) {
        return Some(Stored::One { place: place.storage, path: &place.path });
    }
    let reached = loans.accesses(site).iter().filter(|access| access.reach == Reach::Write && !access.through.is_empty());
    Some(Stored::Several(reached.flat_map(|access| &access.places).map(|place| (place.storage, place.path.as_slice())).collect()))
}

impl Forward<'_> {
    fn set(&self, place: Storage, path: &[PathSeg], to: Presence, state: &mut PresenceOf) {
        let Some(units) = self.units.get(&place) else {
            return;
        };
        for (index, unit) in units.iter().enumerate() {
            if covers(path, unit) {
                state.0.insert((place, index), to);
            }
        }
    }
}

impl DenseAnalysis for Forward<'_> {
    type State = PresenceOf;

    const DIRECTION: Direction = Direction::Forward;

    fn entry(&self, _: &Body) -> PresenceOf {
        let mut state = PresenceOf::default();
        for (place, units) in self.units {
            let start = match place {
                Storage::Param(_) | Storage::Context(_) | Storage::Outside { .. } => Presence::Present,
                Storage::Slot(_) => Presence::Absent,
            };
            state.0.extend((0..units.len()).map(|index| ((*place, index), start)));
        }
        state
    }

    fn transfer(&self, inst: &Inst, state: &mut PresenceOf) {
        for place in whole_writes(self.loans, Site::Inst(inst.id())) {
            self.set(place, &[], Presence::Present, state);
        }
        match (inst.op(), by_name(inst.op())) {
            (Op::Assign { .. }, _) => {
                if let Some(Stored::One { place, path }) = stored(self.loans, inst) {
                    self.set(place, parts::owned_region(path), Presence::Present, state);
                }
            }
            (Op::Take { .. }, Some((place, path)))
                if self.units.contains_key(&place)
                    && !copies(self.types, self.body.ty_of(inst.result().expect("a Take yields a value"))) =>
            {
                self.set(place, parts::owned_region(path), Presence::Absent, state);
            }
            (Op::Take { target: RefTarget::Through(_), .. }, None) => {
                if let Some((place, region)) = moved_through(self.loans, inst) {
                    self.set(place, region, Presence::Absent, state);
                }
            }
            _ => {}
        }
    }

    fn transfer_terminator(&self, block: BlockId, _: &Terminator, state: &mut PresenceOf) {
        for place in whole_writes(self.loans, Site::Term(block)) {
            self.set(place, &[], Presence::Present, state);
        }
    }

    fn transfer_edge(&self, _: BlockId, _: &Edge<'_>, _: &[ValueId], _: &mut PresenceOf) {}
}
