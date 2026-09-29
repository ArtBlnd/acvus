//! Initialization (ownership §2, §4, §6): whether each part of each place
//! the body holds is written, never written, or moved out, at every point:
//! the dense engine, forward, before promotion, when every binding is a
//! slot. `check::initck` and `check::moveck` read it.
//!
//! - A place is a slot or a by-value parameter's own storage, whatever its
//!   type: a binding is built field by field whether its fields copy or
//!   not (ownership §4).
//! - A place's units are its type cut at every path the body names in it,
//!   by name or through a reference, a store's, a read's or a move's, by
//!   `ir::parts::units`, so they cover the whole type. What a function
//!   value's or a lifetime's position lends has no type, and is one unit.
//! - Entry: a slot's units are never written; a by-value parameter's are
//!   written.
//! - An `Assign` into a path by name writes every unit that path covers.
//! - A `Take` by name that `ir::cfg::takings` takes its place `Moved` or
//!   `PartMoved` moves every unit its path covers, at its span (ownership
//!   §2, §6). One it takes `Read` moves nothing: a value that copies, and a
//!   `&mut`, which each use reborrows (ownership §1, §5).
//! - Storage outside the body, what each position of a parameter's or a
//!   capture's type lends (`Loans`' `Storage::Outside`), is a place too,
//!   its units the lent target's type cut as a slot's are, written at entry
//!   (ownership §4: it is whole whenever the body returns).
//! - A `Take` or an `Assign` through a reference reaches the places `Loans`
//!   records for its site (`Loans::accesses`, a `Move` or a `Write`). Where
//!   it reaches one place (`Loans::is_one`: exact, naming no sequence's
//!   element), the move or the write is of that place's units there, as by
//!   name. Where it reaches several, or one that is not one place, nothing
//!   is recorded: `check::moveck` refuses such a move, and such a write
//!   writes no whole unit.
//! - A join unions what each path may leave.
//!
//! Cost: a unit's fact only grows, and it holds a flag and the spans of the
//! body's moves, so each unit rises at most O(moves + 1) times; the engine
//! runs O(units · (moves + 1) · (blocks + edges)) block transfers,
//! each O(its instructions · the units they cover).

use std::collections::BTreeSet;

use acvus_ast::Span;
use rustc_hash::FxHashMap;

use crate::decl::Declarations;
use crate::ir::cfg::takings::{Taken, inst_takings};
use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Edge, Inst, Module, Op, Site, Terminator};
use crate::ir::parts::{self, Path, covers, is_cut_of, overlap};
use crate::ir::{PathSeg, RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, Loans, Reach, Storage, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};
use crate::ty::{Concrete, Ty, Types, lent_types};

/// What the paths reaching a point may leave in one unit: never written, or
/// moved out at these spans. Neither is what a path that wrote the unit
/// leaves, and what a point no path reaches holds: a read there has every
/// path to it writing what it reads (ownership §4).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Written {
    pub never: bool,
    pub moved: BTreeSet<Span>,
}

impl JoinSemiLattice for Written {
    fn bottom() -> Self {
        Written::default()
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = self.clone();
        self.never |= other.never;
        self.moved.extend(other.moved.iter().copied());
        *self != before
    }

    fn le(&self, other: &Self) -> bool {
        (!self.never || other.never) && self.moved.is_subset(&other.moved)
    }
}

/// Finite: a flag and the body's move spans. The widening is the join.
impl Widen for Written {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// The facts: each place's units, and each unit's `Written` just before
/// every instruction and every terminator.
pub struct Initialization {
    units: FxHashMap<Storage, Vec<Path>>,
    before: FxHashMap<Site, FxHashMap<(Storage, usize), Written>>,
}

/// A unit no path reaches, or one every path wrote.
static UNREACHED: Written = Written { never: false, moved: BTreeSet::new() };

impl Initialization {
    /// Every place of the body, in no stated order.
    pub fn places(&self) -> impl Iterator<Item = Storage> + '_ {
        self.units.keys().copied()
    }

    /// The units of `place`, in one order; `None` for a storage that is not
    /// a place of the body.
    pub fn units(&self, place: Storage) -> Option<&[Path]> {
        self.units.get(&place).map(Vec::as_slice)
    }

    /// What each unit of `place` overlapping `path` holds just before
    /// `site`, with the unit. Cost: O(units of `place`).
    ///
    /// # Panics
    /// `place` is not a place of the body, or `site` is not a site of it.
    pub fn before(&self, site: Site, place: Storage, path: &[PathSeg]) -> Vec<(&Path, &Written)> {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?} is not a place of the body"));
        let state = self.before.get(&site).unwrap_or_else(|| panic!("{site:?} is not a site of the body"));
        units
            .iter()
            .enumerate()
            .filter(|(_, unit)| overlap(unit, path))
            .map(|(index, unit)| (unit, state.get(&(place, index)).unwrap_or(&UNREACHED)))
            .collect()
    }
}

impl Analysis for Initialization {
    type Facts = Initialization;

    const READS: &'static [Aspect] = &[Aspect::Instructions, Aspect::CfgShape, Aspect::Storage, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> Self::Facts {
        let loans = am.get::<Loans>(module, body);
        let body = module.body(body);
        let units = units_of(body, module.types(), am.decls(), &loans);
        let forward = Forward { units: &units, types: module.types(), body, loans: &loans };
        let facts = solve_dense(&forward, body);
        let mut before = FxHashMap::default();
        for (id, block) in body.blocks() {
            let mut state = facts.at(id).clone();
            for inst in block.insts() {
                before.insert(Site::Inst(inst.id()), state.0.clone());
                forward.transfer(inst, &mut state);
            }
            before.insert(Site::Term(id), state.0);
        }
        Initialization { units, before }
    }
}

/// The place a `Take`, `Ref` or `Assign` names by name, and the region its
/// path reaches in it.
fn named_place(op: &Op) -> Option<(Storage, &[PathSeg])> {
    match op {
        Op::Take { target, path, .. } | Op::Ref { target, path, .. } | Op::Assign { target, path, .. } => {
            Storage::by_name(target).map(|place| (place, parts::owned_region(path)))
        }
        _ => None,
    }
}

/// Whether `op` is a `Take`, a `Ref` or an `Assign` through a reference.
fn is_through(op: &Op) -> bool {
    matches!(
        op,
        Op::Take { target: RefTarget::Through(_), .. }
            | Op::Ref { target: RefTarget::Through(_), .. }
            | Op::Assign { target: RefTarget::Through(_), .. }
    )
}

/// A place's type, where it has one, and the regions the body cuts it at.
/// What a function value's or a lifetime's position lends has no type.
struct Holder {
    ty: Option<Ty>,
    cuts: Vec<Path>,
}

impl Holder {
    fn typed(ty: Ty) -> Holder {
        Holder { ty: Some(ty), cuts: Vec::new() }
    }

    fn cut(&mut self, region: &[PathSeg]) {
        if !region.is_empty() && !self.cuts.iter().any(|known| known.as_slice() == region) {
            self.cuts.push(region.to_vec());
        }
    }
}

/// Every slot, every parameter the body names as a storage, and every
/// storage outside the body; each with its units, cut at every region a
/// path by name, or through a reference to one place, reaches in it.
fn units_of(body: &Body, types: &Types<Concrete>, decls: &Declarations, loans: &Loans) -> FxHashMap<Storage, Vec<Path>> {
    let mut holders: FxHashMap<Storage, Holder> =
        body.slots().iter().map(|slot| (Storage::Slot(*slot), Holder::typed(body.ty_of(*slot)))).collect();
    let outside = body.params().iter().chain(body.captures()).map(|(_, value)| *value).filter(|value| !body.is_token(*value));
    for of in outside {
        for (position, lent) in lent_types(types, decls, body.ty_of(of)).into_iter().enumerate() {
            holders.insert(Storage::Outside { of, position }, Holder { ty: lent, cuts: Vec::new() });
        }
    }
    for (_, block) in body.blocks() {
        for inst in block.insts() {
            if let Some((place, region)) = named_place(inst.op()) {
                let named = match place {
                    Storage::Slot(value) | Storage::Param(value) => value,
                    Storage::Outside { .. } => unreachable!("a place by name is a slot or a parameter"),
                };
                holders.entry(place).or_insert_with(|| Holder::typed(body.ty_of(named))).cut(region);
                continue;
            }
            if !is_through(inst.op()) {
                continue;
            }
            let reached = loans.accesses(Site::Inst(inst.id())).iter().flat_map(|access| &access.places);
            for place in reached.filter(|place| loans.is_one(place)) {
                let holder = holders.entry(place.storage).or_insert_with(|| match place.storage {
                    Storage::Slot(value) | Storage::Param(value) => Holder::typed(body.ty_of(value)),
                    Storage::Outside { .. } => panic!("{:?} is an outside storage of a parameter or a capture", place.storage),
                });
                let region = parts::owned_region(&place.path);
                if holder.ty.is_some_and(|ty| is_cut_of(types, ty, region)) {
                    holder.cut(region);
                }
            }
        }
    }
    holders
        .into_iter()
        .map(|(place, Holder { ty, cuts })| {
            let units = match ty {
                Some(ty) => parts::units(types, ty, &cuts),
                None => vec![Vec::new()],
            };
            (place, units)
        })
        .collect()
}

/// Each unit's fact; a unit not in the map holds none, as where no path
/// reaches.
#[derive(Debug, Clone, PartialEq, Default)]
struct Units(FxHashMap<(Storage, usize), Written>);

impl Units {
    fn of(&self, unit: (Storage, usize)) -> &Written {
        self.0.get(&unit).unwrap_or(&UNREACHED)
    }
}

impl JoinSemiLattice for Units {
    fn bottom() -> Self {
        Units::default()
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut rose = false;
        for (unit, fact) in &other.0 {
            rose |= self.0.entry(*unit).or_default().join(fact);
        }
        rose
    }

    fn le(&self, other: &Self) -> bool {
        self.0.iter().all(|(unit, fact)| fact.le(other.of(*unit)))
    }
}

/// Finite: each unit's `Written` is. The widening is the join.
impl Widen for Units {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// Forward over the units, by the module doc's transfer.
struct Forward<'f> {
    units: &'f FxHashMap<Storage, Vec<Path>>,
    types: &'f Types<Concrete>,
    body: &'f Body,
    loans: &'f Loans,
}

impl Forward<'_> {
    /// Every unit of `place` that `region` covers now holds `fact` alone.
    fn set(&self, place: Storage, region: &[PathSeg], fact: &Written, state: &mut Units) {
        let units = self.units.get(&place).unwrap_or_else(|| panic!("{place:?}, reached by the body, is a place of it"));
        for (index, unit) in units.iter().enumerate() {
            if covers(region, unit) {
                state.0.insert((place, index), fact.clone());
            }
        }
    }

    /// A `Take` or an `Assign` through a reference: the move or the write
    /// `Loans` records at it that reaches one place, of that place's region.
    fn transfer_through(&self, inst: &Inst, state: &mut Units) {
        let site = Site::Inst(inst.id());
        if let Some(place) = self.loans.through_one(site, Reach::Move) {
            let moved = Written { moved: BTreeSet::from([inst.span()]), ..Written::default() };
            self.set(place.storage, parts::owned_region(&place.path), &moved, state);
        }
        if let Some(place) = self.loans.through_one(site, Reach::Write) {
            self.set(place.storage, parts::owned_region(&place.path), &Written::default(), state);
        }
    }
}

impl DenseAnalysis for Forward<'_> {
    type State = Units;

    const DIRECTION: Direction = Direction::Forward;

    fn entry(&self, _: &Body) -> Units {
        let mut state = Units::default();
        for (place, units) in self.units {
            let start = match place {
                Storage::Slot(_) => Written { never: true, ..Written::default() },
                Storage::Param(_) | Storage::Outside { .. } => Written::default(),
            };
            state.0.extend((0..units.len()).map(|index| ((*place, index), start.clone())));
        }
        state
    }

    fn transfer(&self, inst: &Inst, state: &mut Units) {
        match inst.op() {
            Op::Assign { target: RefTarget::Var(_) | RefTarget::Param(_), .. } => {
                let (place, region) = named_place(inst.op()).expect("matched as a store by name");
                self.set(place, region, &Written::default(), state);
            }
            Op::Take { target: RefTarget::Var(named) | RefTarget::Param(named), .. } => {
                let (place, region) = named_place(inst.op()).expect("matched as a take by name");
                let taken = inst_takings(self.body, self.types, inst)
                    .into_iter()
                    .find(|operand| operand.names_place && operand.value == *named)
                    .expect("a Take by name names its place among its operands")
                    .taken;
                if matches!(taken, Taken::Moved | Taken::PartMoved) {
                    let moved = Written { moved: BTreeSet::from([inst.span()]), ..Written::default() };
                    self.set(place, region, &moved, state);
                }
            }
            Op::Take { target: RefTarget::Through(_), .. } | Op::Assign { target: RefTarget::Through(_), .. } => {
                self.transfer_through(inst, state);
            }
            _ => {}
        }
    }

    fn transfer_terminator(&self, _: BlockId, _: &Terminator, _: &mut Units) {}

    fn transfer_edge(&self, _: BlockId, _: &Edge<'_>, _: &[ValueId], _: &mut Units) {}
}
