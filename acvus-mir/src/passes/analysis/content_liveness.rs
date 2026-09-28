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
//! Cost: `Loans`', and then a block's state only grows, at most once per
//! place, and each growth re-runs the transfer of the block, so O(places ·
//! (blocks + edges) · the largest block's sites · the storages a site
//! reads). The per-point sets take O(points · places) space. The height is
//! the body's places; the widening is the join.
//!
//! passes-analyses.md.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::ir::cfg::{Aspect, BlockId, Body, BodyId, Edge, Inst, Module, Op, Site, Terminator};
use crate::ir::{RefTarget, ValueId};
use crate::passes::analysis::{Analysis, AnalysisManager, DenseAnalysis, Direction, Loans, Storage, solve_dense};
use crate::passes::domain::{JoinSemiLattice, Widen};

pub struct ContentLiveness {
    /// Per block, the places live before each instruction, and last the
    /// places live before its terminator.
    before: FxHashMap<BlockId, Vec<FxHashSet<ValueId>>>,
    at_end: FxHashMap<BlockId, FxHashSet<ValueId>>,
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
            Site::Inst(block, index) => self.points(block).get(index + 1).unwrap_or_else(|| panic!("{site:?} is a site of the body")),
            Site::Term(block) => self.live_at_end(block),
        }
    }

    /// The places whose content may be read from `block`'s start. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body.
    pub fn live_at_start(&self, block: BlockId) -> &FxHashSet<ValueId> {
        &self.points(block)[0]
    }

    /// The places whose content may be read from just before the
    /// instruction at `index` of `block`, or before its terminator where
    /// `index` is the block's length. Cost: O(1).
    ///
    /// # Panics
    /// `block` is not a block of the body, or `index` is past its length.
    pub fn live_before(&self, block: BlockId, index: usize) -> &FxHashSet<ValueId> {
        self.points(block).get(index).unwrap_or_else(|| panic!("{block:?} has no position {index}"))
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
}

impl Analysis for ContentLiveness {
    type Facts = ContentLiveness;

    /// `Loans`' aspects: what a site reads is read from it.
    const READS: &'static [Aspect] = &[Aspect::CfgShape, Aspect::Instructions, Aspect::Storage, Aspect::Types];

    fn compute(am: &mut AnalysisManager<'_>, module: &Module, body: BodyId) -> ContentLiveness {
        let loans = am.get::<Loans>(module, body);
        let body = module.body(body);
        let backward = Backward::new(body, &loans);
        let facts = solve_dense(&backward, body);
        let mut before = FxHashMap::default();
        let mut at_end = FxHashMap::default();
        for (id, block) in body.blocks() {
            let end = facts.at(id).clone();
            let mut state = end.clone();
            backward.transfer_terminator(block.terminator(), &mut state);
            let mut points = vec![state.0.clone()];
            for inst in block.insts().iter().rev() {
                backward.transfer(inst, &mut state);
                points.push(state.0.clone());
            }
            points.reverse();
            before.insert(id, points);
            at_end.insert(id, end.0);
        }
        ContentLiveness { before, at_end }
    }
}

/// A set of places; finite over a body, so its widening is its join.
#[derive(Debug, Clone, PartialEq)]
struct Live(FxHashSet<ValueId>);

impl JoinSemiLattice for Live {
    fn bottom() -> Self {
        Live(FxHashSet::default())
    }

    fn join(&mut self, other: &Self) -> bool {
        let before = self.0.len();
        self.0.extend(other.0.iter().copied());
        self.0.len() != before
    }

    fn le(&self, other: &Self) -> bool {
        self.0.is_subset(&other.0)
    }
}

impl Widen for Live {
    fn widen(&mut self, next: &Self) -> bool {
        self.join(next)
    }
}

/// The engine hands a transfer the instruction or terminator alone; the
/// site `Loans` keys its reads by is found by the record's address in the
/// body, which the engine borrows unchanged for the whole solve.
struct Backward<'f> {
    loans: &'f Loans,
    inst_sites: FxHashMap<*const Inst, Site>,
    term_sites: FxHashMap<*const Terminator, Site>,
}

impl<'f> Backward<'f> {
    /// Cost: O(the body's instructions).
    fn new(body: &'f Body, loans: &'f Loans) -> Self {
        let mut inst_sites = FxHashMap::default();
        let mut term_sites = FxHashMap::default();
        for (block, record) in body.blocks() {
            for (index, inst) in record.insts().iter().enumerate() {
                inst_sites.insert(std::ptr::from_ref(inst), Site::Inst(block, index));
            }
            term_sites.insert(std::ptr::from_ref(record.terminator()), Site::Term(block));
        }
        Backward { loans, inst_sites, term_sites }
    }

    fn gen_reads(&self, site: Site, state: &mut Live) {
        let read_places = self.loans.reads(site).iter().filter_map(|storage| match storage {
            Storage::Slot(place) | Storage::Param(place) => Some(*place),
            Storage::Outside { .. } => None,
        });
        state.0.extend(read_places);
    }
}

impl DenseAnalysis for Backward<'_> {
    type State = Live;

    const DIRECTION: Direction = Direction::Backward;

    fn entry(&self, _: &Body) -> Live {
        Live::bottom()
    }

    fn transfer(&self, inst: &Inst, state: &mut Live) {
        if let Op::Assign { target: RefTarget::Var(place) | RefTarget::Param(place), path, .. } = inst.op()
            && path.is_empty()
        {
            state.0.remove(place);
        }
        let site = self.inst_sites.get(&std::ptr::from_ref(inst)).expect("the instruction transferred is one of the body's");
        self.gen_reads(*site, state);
    }

    fn transfer_terminator(&self, terminator: &Terminator, state: &mut Live) {
        let site = self.term_sites.get(&std::ptr::from_ref(terminator)).expect("the terminator transferred is one of the body's");
        self.gen_reads(*site, state);
    }

    fn transfer_edge(&self, _: &Edge<'_>, _: &[ValueId], _: &mut Live) {}
}
