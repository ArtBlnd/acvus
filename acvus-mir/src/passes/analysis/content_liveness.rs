//! For each place the body owns, a slot or a by-value parameter's own
//! storage, whether its current content may be read before a whole store
//! replaces it or the body ends: the dense engine, backward, over sets of
//! places, each named by its slot or its parameter. A site gens every such
//! place `Loans` says it reads, directly or through a reference; a store
//! into a whole place by name kills it. A store into part of a place, or
//! into a place through a reference, kills nothing, so a place is live
//! wherever those leave it in doubt. The body's end reads none of its own
//! places.
//!
//! `Liveness` answers for values; this answers for what a place holds, which
//! a place's own `ValueId` does not follow, since every access names it.
//!
//! The same answer is kept per part (ownership §6), for each place whose
//! type does not copy: its units (`ir::parts`), cut where the body moves a
//! part that does not copy out of it or stores one into part of it. A unit
//! is read where a `Take` of an overlapping path by name reads it, and
//! wherever `Loans` says the place is read otherwise, since a read through
//! a reference does not locate its part; a store by name into a path
//! covering the unit kills it. Beside it, forward, each unit's presence:
//! a store by name into a path covering it fills it, and a `Take` by name
//! that moves a covering path empties it. A slot starts empty and a
//! by-value parameter's storage full.
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

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Definition, Edge, Inst, InstId, Module, Op, Site, Terminator};
use crate::ir::parts::{self, Path, covers, overlap};
use crate::ir::{PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, Loans, Storage, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::ty::{Concrete, Types, copies};

pub struct ContentLiveness {
    /// Each instruction's block and place in it, as the body stood when the
    /// facts were computed; the facts are dropped when an instruction
    /// changes, so the places stay true while they are kept.
    position: FxHashMap<InstId, (BlockId, usize)>,
    /// Per block, the places live before each instruction, and last the
    /// places live before its terminator.
    before: FxHashMap<BlockId, Vec<FxHashSet<ValueId>>>,
    at_end: FxHashMap<BlockId, FxHashSet<ValueId>>,
    units: FxHashMap<ValueId, Vec<Path>>,
    /// Per block, the units live before each instruction and before its
    /// terminator, and after it.
    parts_before: FxHashMap<BlockId, Vec<FxHashSet<UnitId>>>,
    parts_at_end: FxHashMap<BlockId, FxHashSet<UnitId>>,
    /// Per block, each unit's presence before each instruction and before
    /// its terminator.
    presence_before: FxHashMap<BlockId, Vec<PresenceOf>>,
}

/// A unit of a place: the place, and the unit's index among its units.
type UnitId = (ValueId, usize);

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
    pub fn live_after(&self, site: Site) -> &FxHashSet<ValueId> {
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
    pub fn live_at_start(&self, block: BlockId) -> &FxHashSet<ValueId> {
        &self.points(block)[0]
    }

    /// The places whose content may be read from just before `inst`. Cost:
    /// O(1).
    ///
    /// # Panics
    /// `inst` is not an instruction of the body.
    pub fn live_before(&self, inst: InstId) -> &FxHashSet<ValueId> {
        let (block, index) = self.position_of(inst);
        &self.points(block)[index]
    }

    /// The places whose content may be read from just before `block`'s
    /// terminator. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_before_terminator(&self, block: BlockId) -> &FxHashSet<ValueId> {
        self.points(block).last().expect("a block's points end before its terminator")
    }

    /// The places whose content may be read after `block`'s terminator, on
    /// its way into a successor. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_end(&self, block: BlockId) -> &FxHashSet<ValueId> {
        self.at_end.get(&block).expect("content liveness covers every block of its body")
    }

    fn points(&self, block: BlockId) -> &[FxHashSet<ValueId>] {
        self.before.get(&block).expect("content liveness covers every block of its body")
    }

    /// The units of `place`, a slot or a by-value parameter's own storage
    /// whose type does not copy, in the order the body cuts them; `[[]]`
    /// where nothing cuts it. `None` for any other value. Cost: O(1).
    pub fn units(&self, place: ValueId) -> Option<&[Path]> {
        self.units.get(&place).map(Vec::as_slice)
    }

    /// Whether a unit of `place` overlapping `path` may be read from just
    /// before `inst`, before a store replaces it or the body ends. Cost:
    /// O(units of `place`).
    ///
    /// # Panics
    /// `place` has no units, or `inst` is not an instruction of the body.
    pub fn part_live_before(&self, inst: InstId, place: ValueId, path: &[PathSeg]) -> bool {
        let (block, index) = self.position_of(inst);
        self.part_live_at(block, index, place, path)
    }

    fn part_live_at(&self, block: BlockId, index: usize, place: ValueId, path: &[PathSeg]) -> bool {
        let points = self.parts_before.get(&block).expect("content liveness covers every block of its body");
        let live = points.get(index).unwrap_or_else(|| panic!("{block:?} has no position {index}"));
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// As `part_live_before`, from just after the instruction or terminator
    /// at `site`. Cost: O(units of `place`).
    pub fn part_live_after(&self, site: Site, place: ValueId, path: &[PathSeg]) -> bool {
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
    pub fn part_live_before_terminator(&self, block: BlockId, place: ValueId, path: &[PathSeg]) -> bool {
        let points = self.parts_before.get(&block).expect("content liveness covers every block of its body");
        let live = points.last().expect("a block's points end before its terminator");
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// As `part_live_before`, from `block`'s start. Cost: O(units of `place`).
    pub fn part_live_at_start(&self, block: BlockId, place: ValueId, path: &[PathSeg]) -> bool {
        self.part_live_at(block, 0, place, path)
    }

    /// As `part_live_before`, after `block`'s terminator, on its way into a
    /// successor. Cost: O(units of `place`).
    pub fn part_live_at_end(&self, block: BlockId, place: ValueId, path: &[PathSeg]) -> bool {
        let live = self.parts_at_end.get(&block).expect("content liveness covers every block of its body");
        self.overlapping(place, path).any(|unit| live.contains(&unit))
    }

    /// The presence of the unit `unit` of `place` just before `inst`. Cost:
    /// O(units of `place`).
    ///
    /// # Panics
    /// `unit` is not one of `place`'s units, or `inst` is not an
    /// instruction of the body.
    pub fn presence_before(&self, inst: InstId, place: ValueId, unit: &[PathSeg]) -> Presence {
        let (block, index) = self.position_of(inst);
        let points = self.presence_before.get(&block).expect("content presence covers every block of its body");
        let state = points.get(index).unwrap_or_else(|| panic!("{block:?} has no position {index}"));
        state.of((place, self.unit_index(place, unit)))
    }

    /// The presence of `unit` of `place` after `block`'s terminator, which
    /// changes no presence. Cost: O(units of `place`).
    pub fn presence_at_end(&self, block: BlockId, place: ValueId, unit: &[PathSeg]) -> Presence {
        let points = self.presence_before.get(&block).expect("content presence covers every block of its body");
        let state = points.last().expect("a block's points end before its terminator");
        state.of((place, self.unit_index(place, unit)))
    }

    fn unit_index(&self, place: ValueId, unit: &[PathSeg]) -> usize {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?} has no units"));
        units
            .iter()
            .position(|known| known.as_slice() == unit)
            .unwrap_or_else(|| panic!("{unit:?} is not one of {place:?}'s units {units:?}"))
    }

    fn overlapping<'a>(&'a self, place: ValueId, path: &'a [PathSeg]) -> impl Iterator<Item = UnitId> + 'a {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?} has no units"));
        units.iter().enumerate().filter(move |(_, unit)| overlap(unit, path)).map(move |(index, _)| (place, index))
    }
}

impl Analysis for ContentLiveness {
    type Facts = ContentLiveness;

    /// `Loans`' aspects: what a site reads is read from it.
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> ContentLiveness {
        let loans = am.get::<Loans>(module, body);
        let body = module.body(body);
        let units = units_of(body, module.types());
        let backward = Backward::new(&loans, &units);
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
        let forward = Forward { units: &units, types: module.types(), body };
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

/// The places whose type does not copy, each with its units: a slot, or a
/// parameter the body names as a storage.
fn units_of(body: &Body, types: &Types<Concrete>) -> FxHashMap<ValueId, Vec<Path>> {
    let mut cuts: FxHashMap<ValueId, Vec<Path>> = FxHashMap::default();
    for slot in body.slots() {
        if !copies(types, body.ty_of(*slot)) {
            cuts.entry(*slot).or_default();
        }
    }
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            let (place, path, moves) = match inst.op() {
                Op::Take { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } => {
                    let taken = inst.result().expect("a Take yields a value");
                    (*place, parts::owned_region(path), !copies(types, body.ty_of(taken)))
                }
                Op::Assign { target: RefTarget::Var(place) | RefTarget::Param(place), path, value, .. } => {
                    (*place, parts::owned_region(path), !copies(types, body.ty_of(*value)))
                }
                Op::Ref { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } => (*place, path.as_slice(), false),
                _ => continue,
            };
            let cut = moves && !path.is_empty();
            if copies(types, body.ty_of(place)) {
                continue;
            }
            let place_cuts = cuts.entry(place).or_default();
            if cut && !place_cuts.iter().any(|known| known.as_slice() == path) {
                place_cuts.push(path.to_vec());
            }
        }
    }
    cuts.into_iter()
        .filter(|(place, _)| matches!(body.definition(*place), Definition::Slot(_) | Definition::Param(_)))
        .map(|(place, cuts)| (place, parts::units(types, body.ty_of(place), &cuts)))
        .collect()
}

/// A set of places and a set of units; finite over a body, so its
/// widening is its join.
#[derive(Debug, Clone, PartialEq)]
struct Live {
    places: FxHashSet<ValueId>,
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
    units: &'f FxHashMap<ValueId, Vec<Path>>,
}

impl<'f> Backward<'f> {
    fn new(loans: &'f Loans, units: &'f FxHashMap<ValueId, Vec<Path>>) -> Self {
        Backward { loans, units }
    }

    /// What `site` reads: the places `Loans` names, and of each, the units
    /// overlapping the path a `Take` names it by, or all of them.
    fn gen_reads(&self, site: Site, by_name: Option<(ValueId, &[PathSeg])>, state: &mut Live) {
        let read_places = self.loans.reads(site).iter().filter_map(|storage| match storage {
            Storage::Slot(place) | Storage::Param(place) => Some(*place),
            Storage::Outside { .. } => None,
        });
        for place in read_places {
            state.places.insert(place);
            let Some(units) = self.units.get(&place) else {
                continue;
            };
            let read: &[PathSeg] = match by_name {
                Some((named, path)) if named == place => path,
                _ => &[],
            };
            let touched = units.iter().enumerate().filter(|(_, unit)| overlap(unit, read));
            state.parts.extend(touched.map(|(index, _)| (place, index)));
        }
    }
}

impl DenseAnalysis for Backward<'_> {
    type State = Live;

    const DIRECTION: Direction = Direction::Backward;

    fn entry(&self, _: &Body) -> Live {
        Live::bottom()
    }

    fn transfer(&self, inst: &Inst, state: &mut Live) {
        if let Op::Assign { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } = inst.op() {
            if path.is_empty() {
                state.places.remove(place);
            }
            if let Some(units) = self.units.get(place) {
                for (index, unit) in units.iter().enumerate() {
                    if covers(parts::owned_region(path), unit) {
                        state.parts.remove(&(*place, index));
                    }
                }
            }
        }
        let by_name = match inst.op() {
            Op::Take { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } => Some((*place, path.as_slice())),
            _ => None,
        };
        self.gen_reads(Site::Inst(inst.id()), by_name, state);
    }

    fn transfer_terminator(&self, block: BlockId, _: &Terminator, state: &mut Live) {
        self.gen_reads(Site::Term(block), None, state);
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

/// Forward over the units: a slot's start empty and a parameter's full; a
/// store by name fills what it covers, and a `Take` by name of a path that
/// does not copy empties what it covers.
struct Forward<'f> {
    units: &'f FxHashMap<ValueId, Vec<Path>>,
    types: &'f Types<Concrete>,
    body: &'f Body,
}

impl Forward<'_> {
    fn set(&self, place: ValueId, path: &[PathSeg], to: Presence, state: &mut PresenceOf) {
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

    fn entry(&self, body: &Body) -> PresenceOf {
        let mut state = PresenceOf::default();
        for (place, units) in self.units {
            let start = match body.definition(*place) {
                Definition::Param(_) => Presence::Present,
                _ => Presence::Absent,
            };
            state.0.extend((0..units.len()).map(|index| ((*place, index), start)));
        }
        state
    }

    fn transfer(&self, inst: &Inst, state: &mut PresenceOf) {
        match inst.op() {
            Op::Assign { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } => {
                self.set(*place, parts::owned_region(path), Presence::Present, state);
            }
            Op::Take { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. }
                if self.units.contains_key(place)
                    && !copies(self.types, self.body.ty_of(inst.result().expect("a Take yields a value"))) =>
            {
                self.set(*place, parts::owned_region(path), Presence::Absent, state);
            }
            _ => {}
        }
    }

    fn transfer_terminator(&self, _: BlockId, _: &Terminator, _: &mut PresenceOf) {}

    fn transfer_edge(&self, _: BlockId, _: &Edge<'_>, _: &[ValueId], _: &mut PresenceOf) {}
}
