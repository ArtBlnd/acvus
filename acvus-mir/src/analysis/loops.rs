//! The natural loops of a body, and what "loop-invariant" means.
//!
//! There is no loop analysis below this file: a loop is a back edge of the
//! dominator tree and nothing else. An edge `tail -> head` whose head
//! dominates its tail names one, and its body is the head together with
//! every block that reaches the tail without passing the head.
//!
//! `optimize::code_motion`, which asks how deep a block sits and which block
//! enters a loop, `optimize::lsr`, which asks for the header, the latches,
//! the entering block and the body, and `analysis::affine` and
//! `analysis::carried` read the definition from here, so none of them can
//! drift from the others on what a loop is or on which values one
//! iteration shares with the next.
//!
//! A retreating edge whose target does not dominate its source is
//! irreducible control flow, which has no natural loop. The lowering emits
//! `while`, `while let` and `for` and nothing else, so [`back_edges`] aborts
//! rather than guess a shape the front end cannot produce.
//!
//! A `for` loop has a back edge like the other two -- its latch jumps to its
//! header -- so it is a natural loop here without a word added. What its
//! terminator gives on top of that is the header without the search and the
//! induction variable without a pattern match: [`for_headers`].
//!
//! # The nest
//!
//! [`LoopNest`] is these loops with what a reader of a loop asks next: the
//! loop that contains it, the loops it contains, whether a terminator
//! states its traversal, and how many times its body runs. The parent is
//! the smallest other loop that contains the header; natural loops of a
//! reducible body are nested or disjoint, so that loop contains the whole
//! body. The trip count is a [`Term`] over what one entry to the loop
//! fixes: `max(hi − at, 0)` for a range and `len(source)` for a slice or
//! an array, both read off the terminator (RFC-0057 rule 3). A `while` has
//! no such statement, and its count is [`Trip::Unknown`] rather than one
//! derived from its recurrence (RFC-0066). A loop is rectangular in its
//! parent when its trip count is a term every atom of which is invariant
//! in the parent, so every entry to it runs the same number of times.
//!
//! A term denotes an integer, not a value at a width: a trip count is a
//! count and does not wrap. `analysis::affine` builds its base and step
//! from the same atoms and the same operations, and there the integer is
//! taken modulo `2^width`, which is the value itself on every run that
//! reaches it: a program's operation whose result leaves the width ends the
//! run (RFC-0037 rule 3).

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::targets;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, ExitTrip, ForSource, InstKind, Label, ValueId};
use crate::ty::{Mutability, Ty};
use acvus_ast::Literal;
use rustc_hash::{FxHashMap, FxHashSet};

pub struct BackEdge {
    pub tail: BlockIdx,
    pub head: BlockIdx,
}

struct DfsFrame {
    block: BlockIdx,
    next_succ: usize,
}

/// # Panics
/// On irreducible control flow: a retreating edge whose head does not
/// dominate its tail.
pub fn back_edges(cfg: &CfgBody, domtree: &DomTree) -> Vec<BackEdge> {
    let n = cfg.blocks.len();
    if n == 0 {
        return Vec::new();
    }

    let mut edges = Vec::new();
    let mut visited = vec![false; n];
    let mut on_path = vec![false; n];
    let mut stack = vec![DfsFrame {
        block: BlockIdx(0),
        next_succ: 0,
    }];
    visited[0] = true;
    on_path[0] = true;

    while let Some(frame) = stack.last_mut() {
        let block = frame.block;
        let succs = cfg.successors(block);
        let Some(&succ) = succs.get(frame.next_succ) else {
            on_path[block.0] = false;
            stack.pop();
            continue;
        };
        frame.next_succ += 1;

        if on_path[succ.0] {
            assert!(
                domtree.dominates(succ, block),
                "irreducible control flow: the edge from block {} back to block {} \
                 has a head that does not dominate its tail, and the lowering emits \
                 only structured loops",
                block.0,
                succ.0
            );
            edges.push(BackEdge {
                tail: block,
                head: succ,
            });
            continue;
        }

        if !visited[succ.0] {
            visited[succ.0] = true;
            on_path[succ.0] = true;
            stack.push(DfsFrame {
                block: succ,
                next_succ: 0,
            });
        }
    }

    edges
}

pub struct NaturalLoop {
    pub header: BlockIdx,
    pub latches: Vec<BlockIdx>,
    /// The header's predecessors outside the body: the blocks every entry
    /// to the loop leaves from.
    pub entering: Vec<BlockIdx>,
    body: Vec<bool>,
}

impl NaturalLoop {
    /// The one value every back edge sends header parameter `param`, or
    /// `None` when two back edges send different values or a latch reaches
    /// the header by more than one edge.
    pub fn back_arg(&self, cfg: &CfgBody, param: usize) -> Option<ValueId> {
        self.sole_arg(cfg, &self.latches, param)
    }

    /// The one value every entering edge sends header parameter `param`,
    /// as [`Self::back_arg`] is for the back edges.
    pub fn entry_arg(&self, cfg: &CfgBody, param: usize) -> Option<ValueId> {
        self.sole_arg(cfg, &self.entering, param)
    }

    fn sole_arg(&self, cfg: &CfgBody, from: &[BlockIdx], param: usize) -> Option<ValueId> {
        let label = cfg.blocks[self.header.0].label;
        let mut found: Option<ValueId> = None;
        for block in from {
            let arg = edge_args(&cfg.blocks[block.0].terminator, label)?[param];
            match found {
                Some(seen) if seen != arg => return None,
                _ => found = Some(arg),
            }
        }
        found
    }

    /// No pass assumes a `while` ends, even one with no effect: the
    /// analysis that finds "no effect" is the one that errs (RFC-0088,
    /// Rejected).
    pub fn is_while(&self, cfg: &CfgBody) -> bool {
        !matches!(cfg.blocks[self.header.0].terminator, Terminator::For { .. })
    }

    pub fn contains(&self, block: BlockIdx) -> bool {
        self.body[block.0]
    }

    pub fn block_count(&self) -> usize {
        self.body.iter().filter(|inside| **inside).count()
    }

    pub fn blocks(&self) -> impl Iterator<Item = BlockIdx> + '_ {
        self.body
            .iter()
            .enumerate()
            .filter(|(_, inside)| **inside)
            .map(|(b, _)| BlockIdx(b))
    }
}

/// Why a value is the same on every iteration of a loop, and what it takes
/// to read it above the header.
///
/// `code_motion` lifts an instruction only to a block of no greater loop
/// depth, so what its hoist leaves above a header is [`Invariant::Outside`]
/// and nothing else. A word `Const` is the one thing the hoist cannot lift
/// out of a loop and that is invariant all the same: control equivalence
/// holds it inside the body, because a loop's body does not post-dominate
/// its preheader, while the instruction reads nothing and writes the same
/// word every time it runs. `lsr` needs that case — a literal step `1` and
/// a literal factor `3` both stand inside the body — so the term carries
/// it, with the price named: the constant is re-emitted above the header
/// rather than moved. Which literals are words and which build a heap value
/// is the same split `code_motion::hoistable` makes on `Const`.
///
/// Each case carries what a reader above the header reads: the value
/// itself, or the word to write again.
#[derive(Clone, Debug, PartialEq)]
pub enum Invariant {
    Outside(ValueId),
    Word(Literal),
}

/// Why a value is the same on every iteration of a loop (RFC-0066 rule 3):
/// it can be read above the header ([`Invariant`]), or it is an operation
/// of invariant operands inside the loop that reads nothing the loop
/// writes, left where it stands.
///
/// A standing operation is not moved: one that can end the run keeps its
/// trap on the paths it ran on (RFC-0037 rule 3), and a loop that runs no
/// iteration never computes it. So no reader above the header can read
/// it, and a reader that writes a value there asks [`Invariance::above`],
/// which a standing value answers with nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum Invariance {
    Above(Invariant),
    Standing(ValueId),
}

impl Invariance {
    /// What a reader above the header reads, where it can read one.
    pub fn above(&self) -> Option<&Invariant> {
        match self {
            Invariance::Above(invariant) => Some(invariant),
            Invariance::Standing(_) => None,
        }
    }
}

/// The one answer to "is this value the same on every iteration", built
/// once per body and asked per loop.
pub struct Invariants {
    def_block: FxHashMap<ValueId, BlockIdx>,
    words: FxHashMap<ValueId, Literal>,
    /// Per loop header, the operations standing in that loop.
    standing: FxHashMap<BlockIdx, FxHashSet<ValueId>>,
}

impl Invariants {
    pub fn of(cfg: &CfgBody) -> Self {
        let words = cfg
            .blocks
            .iter()
            .flat_map(|block| block.insts.iter())
            .filter_map(|inst| match &inst.kind {
                InstKind::Const { dst, value } => match value {
                    Literal::String(_) | Literal::List(_) => None,
                    word => Some((*dst, word.clone())),
                },
                _ => None,
            })
            .collect();
        let mut invariants = Self {
            def_block: def_blocks(cfg),
            words,
            standing: FxHashMap::default(),
        };
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        if loops.is_empty() {
            return invariants;
        }
        let loans = Loans::build(cfg);
        for loop_ in &loops {
            let standing = invariants.standing_in(cfg, &loans, loop_);
            invariants.standing.insert(loop_.header, standing);
        }
        invariants
    }

    /// The operations standing in `loop_` (RFC-0066 rule 3): each defines
    /// one word, has no effect, reads only values invariant in the loop,
    /// and touches no storage the loop writes. Found to a fixpoint, since
    /// an operand may itself be a standing operation.
    fn standing_in(&self, cfg: &CfgBody, loans: &Loans<'_>, loop_: &NaturalLoop) -> FxHashSet<ValueId> {
        let written = written_in(cfg, loans, loop_);
        let candidates: Vec<(ValueId, &InstKind)> = loop_
            .blocks()
            .flat_map(|block| cfg.blocks[block.0].insts.iter())
            .filter_map(|inst| {
                let dst = standing_operation(&inst.kind)?;
                let is_word = cfg.val_types.get(&dst).and_then(Ty::is_word) == Some(true);
                (is_word && touches_none_of(loans, &inst.kind, &written)).then_some((dst, &inst.kind))
            })
            .collect();
        let mut standing: FxHashSet<ValueId> = FxHashSet::default();
        loop {
            let before = standing.len();
            for (dst, kind) in &candidates {
                if standing.contains(dst) {
                    continue;
                }
                let invariant_operand = |value: &ValueId| {
                    !loop_.contains(self.def_block(*value))
                        || self.words.contains_key(value)
                        || standing.contains(value)
                };
                if inst_info::uses(kind).iter().all(invariant_operand) {
                    standing.insert(*dst);
                }
            }
            if standing.len() == before {
                return standing;
            }
        }
    }

    /// # Panics
    /// If `value` has no definition in this body.
    pub fn def_block(&self, value: ValueId) -> BlockIdx {
        *self
            .def_block
            .get(&value)
            .unwrap_or_else(|| panic!("{value:?} is used but never defined"))
    }

    pub fn word(&self, value: ValueId) -> Option<&Literal> {
        self.words.get(&value)
    }

    /// `value` as a reader above `loop_`'s header reads it, where it can:
    /// defined outside the loop, or a word to write again.
    pub fn above(&self, loop_: &NaturalLoop, value: ValueId) -> Option<Invariant> {
        if !loop_.contains(self.def_block(value)) {
            return Some(Invariant::Outside(value));
        }
        self.words.get(&value).cloned().map(Invariant::Word)
    }

    /// Whether `value` is invariant in `loop_` (RFC-0066 rule 3), and why.
    pub fn in_loop(&self, loop_: &NaturalLoop, value: ValueId) -> Option<Invariance> {
        if let Some(invariant) = self.above(loop_, value) {
            return Some(Invariance::Above(invariant));
        }
        self.standing
            .get(&loop_.header)
            .is_some_and(|standing| standing.contains(&value))
            .then_some(Invariance::Standing(value))
    }
}

/// The value an instruction defines when it is an operation that can stand
/// in a loop: arithmetic and a cast, a shared reference to a storage and
/// its slice, an element read, and a call that carries no `Order`, which
/// has no effect (RFC-0013). A `Take` may empty its slot, a `Fetch` reads a
/// context, and a constructor builds a heap value each time; none of them
/// stands.
fn standing_operation(kind: &InstKind) -> Option<ValueId> {
    match kind {
        InstKind::BinOp { dst, .. }
        | InstKind::UnaryOp { dst, .. }
        | InstKind::Cast { dst, .. }
        | InstKind::Index { dst, .. }
        | InstKind::Ref {
            dst,
            mutability: Mutability::Shared,
            ..
        }
        | InstKind::AsSlice {
            dst,
            mutability: Mutability::Shared,
            ..
        } => Some(*dst),
        InstKind::FunctionCall {
            dst,
            callee: Callee::Direct(_) | Callee::Extern { .. },
            order: None,
            ..
        } => Some(*dst),
        _ => None,
    }
}

/// The storage slots `loop_` writes: what an instruction writes or lends
/// exclusively, and the source a `for` inside it writes through.
fn written_in(cfg: &CfgBody, loans: &Loans<'_>, loop_: &NaturalLoop) -> FxHashSet<ValueId> {
    let mut written: FxHashSet<ValueId> = FxHashSet::default();
    for block in loop_.blocks() {
        let held = &cfg.blocks[block.0];
        for inst in &held.insts {
            written.extend(targets::effect(loans, &inst.kind).writes);
            written.extend(targets::slots_lent_mutably(loans, &inst.kind));
        }
        if let Terminator::For {
            source: ForSource::SliceMut(source),
            ..
        } = &held.terminator
        {
            written.extend(loans.holds(*source).filter_map(|loan| loan.storage.slot()));
        }
    }
    written
}

/// Whether `kind` touches no storage in `written`. Every loan it reaches
/// must name a slot of this body: a storage outside it is one the loop's
/// writes are not counted against, so an operation that reaches one does
/// not stand.
fn touches_none_of(loans: &Loans<'_>, kind: &InstKind, written: &FxHashSet<ValueId>) -> bool {
    let through_slots = inst_info::uses(kind)
        .iter()
        .all(|value| loans.holds(*value).all(|loan| loan.storage.slot().is_some()));
    let effect = loans.storage_effect(kind);
    through_slots
        && effect
            .reads
            .iter()
            .chain(&effect.writes)
            .all(|slot| !written.contains(slot))
}

/// The arguments the one edge of `term` to `label` carries, or `None` when
/// `term` has no such edge or more than one. A `For`'s body edge is not
/// among them: it carries only the parameters after the ones the
/// terminator fills, so its arguments do not line up with its target's
/// parameters (RFC-0057). Nor is its exit edge where that edge defines the
/// trip count (rule 9); where it does not, it carries its target's whole
/// list.
pub fn edge_args(term: &Terminator, label: Label) -> Option<&[ValueId]> {
    let mut edges: Vec<&[ValueId]> = Vec::new();
    match term {
        Terminator::Jump { label: l, args } => {
            if *l == label {
                edges.push(args);
            }
        }
        Terminator::JumpIf {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        }
        | Terminator::Diamond {
            then_label,
            then_args,
            else_label,
            else_args,
            ..
        } => {
            if *then_label == label {
                edges.push(then_args);
            }
            if *else_label == label {
                edges.push(else_args);
            }
        }
        Terminator::Switch { arms, default, .. } => {
            for (_, l, args) in arms {
                if *l == label {
                    edges.push(args);
                }
            }
            if let Some((l, args)) = default {
                if *l == label {
                    edges.push(args);
                }
            }
        }
        Terminator::For {
            exit,
            exit_trip,
            exit_args,
            ..
        } => {
            if *exit_trip == ExitTrip::Absent && *exit == label {
                edges.push(exit_args);
            }
        }
        Terminator::While {
            stages,
            exit,
            exit_args,
            ..
        } => {
            if stages.body() == label {
                edges.push(&[]);
            }
            if *exit == label {
                edges.push(exit_args);
            }
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
    }
    match edges[..] {
        [args] => Some(args),
        _ => None,
    }
}

/// A symbolic integer over what one entry to a loop fixes (RFC-0066 rule
/// 3): a constant, a value invariant in the loop, the length of a `for`'s
/// source, a storage's length on the loop's entry, and `+`, `−`, `×` and
/// `max` of those. It is written as the analysis found it and not
/// simplified; a reader evaluates it where it knows the atoms.
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
    Const(Literal),
    Value(ValueId),
    /// The element count of a `Slice`, `SliceMut` or `Array` source, which
    /// the loop cannot change: the source's borrow or move holds it for
    /// the loop's extent (RFC-0057 rules 2 and 5).
    Len(ValueId),
    /// The element count storage slot `s` holds when the loop is entered
    /// (RFC-0066 rule 4's `len(s) on entry`).
    LenOnEntry(ValueId),
    Add(Box<Term>, Box<Term>),
    Sub(Box<Term>, Box<Term>),
    Mul(Box<Term>, Box<Term>),
    Max(Box<Term>, Box<Term>),
}

impl Term {
    pub fn int(value: i128) -> Term {
        Term::Const(Literal::Int(value))
    }

    pub fn add(self, other: Term) -> Term {
        Term::Add(Box::new(self), Box::new(other))
    }

    pub fn sub(self, other: Term) -> Term {
        Term::Sub(Box::new(self), Box::new(other))
    }

    pub fn mul(self, other: Term) -> Term {
        Term::Mul(Box::new(self), Box::new(other))
    }

    pub fn max(self, other: Term) -> Term {
        Term::Max(Box::new(self), Box::new(other))
    }

    /// The values the term reads, `Len`'s source and `LenOnEntry`'s slot
    /// among them.
    pub fn atoms(&self) -> Vec<ValueId> {
        match self {
            Term::Const(_) => Vec::new(),
            Term::Value(value) | Term::Len(value) | Term::LenOnEntry(value) => vec![*value],
            Term::Add(a, b) | Term::Sub(a, b) | Term::Mul(a, b) | Term::Max(a, b) => {
                a.atoms().into_iter().chain(b.atoms()).collect()
            }
        }
    }
}

impl From<Invariant> for Term {
    fn from(invariant: Invariant) -> Term {
        match invariant {
            Invariant::Outside(value) => Term::Value(value),
            Invariant::Word(literal) => Term::Const(literal),
        }
    }
}

impl From<Invariance> for Term {
    fn from(invariance: Invariance) -> Term {
        match invariance {
            Invariance::Above(invariant) => Term::from(invariant),
            Invariance::Standing(value) => Term::Value(value),
        }
    }
}

/// The loop headers a terminator names, with the traversal each one is
/// (RFC-0057 rule 3). A `for` header is a loop header by what ends it, so
/// a reader that wants the loop and its induction variable asks the
/// terminator rather than searching for a back edge.
pub fn for_headers(cfg: &CfgBody) -> FxHashMap<BlockIdx, ForSource> {
    cfg.blocks
        .iter()
        .enumerate()
        .filter_map(|(bi, block)| match &block.terminator {
            Terminator::For { source, .. } => Some((BlockIdx(bi), *source)),
            _ => None,
        })
        .collect()
}

pub fn natural_loops_innermost_first(cfg: &CfgBody, domtree: &DomTree) -> Vec<NaturalLoop> {
    let n = cfg.blocks.len();
    let preds = cfg.predecessors();
    let mut by_header: FxHashMap<usize, NaturalLoop> = FxHashMap::default();

    for edge in back_edges(cfg, domtree) {
        let loop_ = by_header.entry(edge.head.0).or_insert_with(|| {
            let mut body = vec![false; n];
            body[edge.head.0] = true;
            NaturalLoop {
                header: edge.head,
                latches: Vec::new(),
                entering: Vec::new(),
                body,
            }
        });
        loop_.latches.push(edge.tail);

        let mut stack = Vec::new();
        if !loop_.body[edge.tail.0] {
            loop_.body[edge.tail.0] = true;
            stack.push(edge.tail);
        }
        while let Some(b) = stack.pop() {
            for &p in preds.get(&b).into_iter().flatten() {
                if !loop_.body[p.0] {
                    loop_.body[p.0] = true;
                    stack.push(p);
                }
            }
        }
    }

    let mut loops: Vec<NaturalLoop> = by_header.into_values().collect();
    for loop_ in &mut loops {
        loop_.entering = preds
            .get(&loop_.header)
            .into_iter()
            .flatten()
            .copied()
            .filter(|p| !loop_.body[p.0])
            .collect();
    }
    loops.sort_by_key(|l| (l.block_count(), l.header.0));
    loops
}

// -- The nest -------------------------------------------------------

/// A loop's index in its [`LoopNest`], innermost first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopId(pub usize);

/// What ends the header: a `for` terminator, which states the traversal,
/// or any other, which is a `while` or a `while let`.
#[derive(Clone, Copy, Debug)]
pub enum LoopKind {
    For { source: ForSource },
    While,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Trip {
    Known(Term),
    /// A `while`: nothing states its exit.
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nesting {
    Outermost,
    Inside { parent: LoopId, rectangular: bool },
}

pub struct Loop {
    pub natural: NaturalLoop,
    pub kind: LoopKind,
    pub trip: Trip,
    pub nesting: Nesting,
    pub children: Vec<LoopId>,
}

pub struct LoopNest {
    loops: Vec<Loop>,
}

impl LoopNest {
    pub fn of(cfg: &CfgBody, domtree: &DomTree, invariants: &Invariants) -> Self {
        let headers = for_headers(cfg);
        let naturals = natural_loops_innermost_first(cfg, domtree);
        let parents: Vec<Option<LoopId>> = naturals
            .iter()
            .enumerate()
            .map(|(i, inner)| {
                naturals
                    .iter()
                    .enumerate()
                    .skip(i + 1)
                    .find(|(_, outer)| outer.contains(inner.header))
                    .map(|(j, _)| LoopId(j))
            })
            .collect();

        let mut loops: Vec<Loop> = naturals
            .into_iter()
            .map(|natural| {
                let kind = match headers.get(&natural.header) {
                    Some(source) => LoopKind::For { source: *source },
                    None => LoopKind::While,
                };
                let trip = trip_count(&natural, kind, invariants);
                Loop {
                    natural,
                    kind,
                    trip,
                    nesting: Nesting::Outermost,
                    children: Vec::new(),
                }
            })
            .collect();

        for (i, parent) in parents.iter().enumerate() {
            let Some(parent) = *parent else {
                continue;
            };
            let rectangular = match &loops[i].trip {
                Trip::Known(term) => term
                    .atoms()
                    .into_iter()
                    .all(|atom| invariants.in_loop(&loops[parent.0].natural, atom).is_some()),
                Trip::Unknown => false,
            };
            loops[i].nesting = Nesting::Inside {
                parent,
                rectangular,
            };
            loops[parent.0].children.push(LoopId(i));
        }
        Self { loops }
    }

    pub fn get(&self, id: LoopId) -> &Loop {
        &self.loops[id.0]
    }

    /// Every loop with its id, innermost first.
    pub fn iter(&self) -> impl Iterator<Item = (LoopId, &Loop)> + '_ {
        self.loops.iter().enumerate().map(|(i, l)| (LoopId(i), l))
    }

    pub fn by_header(&self, header: BlockIdx) -> Option<LoopId> {
        self.loops
            .iter()
            .position(|l| l.natural.header == header)
            .map(LoopId)
    }
}

/// How many times one entry to the loop runs its body: what the `for`
/// terminator states.
///
/// # Panics
/// If a `for` source reads a value defined inside its own loop: every
/// source is settled before the header runs (`ir::ForSource`).
fn trip_count(natural: &NaturalLoop, kind: LoopKind, invariants: &Invariants) -> Trip {
    let LoopKind::For { source } = kind else {
        return Trip::Unknown;
    };
    let settled = |value: ValueId| {
        Term::from(invariants.above(natural, value).unwrap_or_else(|| {
            panic!(
                "the `for` headed at block {} reads {value:?}, which its own body defines",
                natural.header.0
            )
        }))
    };
    Trip::Known(match source {
        ForSource::Range { at, hi } => settled(hi).sub(settled(at)).max(Term::int(0)),
        ForSource::Slice(source) | ForSource::SliceMut(source) | ForSource::Array(source) => {
            settled(source);
            Term::Len(source)
        }
    })
}

pub struct LoopDepth {
    per_block: Vec<usize>,
}

impl LoopDepth {
    pub fn of(cfg: &CfgBody, domtree: &DomTree) -> Self {
        let mut per_block = vec![0usize; cfg.blocks.len()];
        for loop_ in natural_loops_innermost_first(cfg, domtree) {
            for b in loop_.blocks() {
                per_block[b.0] += 1;
            }
        }
        Self { per_block }
    }

    pub fn at(&self, block: BlockIdx) -> usize {
        self.per_block[block.0]
    }
}

pub fn def_blocks(cfg: &CfgBody) -> FxHashMap<ValueId, BlockIdx> {
    let mut def_block = FxHashMap::default();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        let idx = BlockIdx(bi);
        for &p in &block.params {
            def_block.insert(p, idx);
        }
        for inst in &block.insts {
            for d in crate::analysis::inst_info::defs(&inst.kind) {
                def_block.insert(d, idx);
            }
        }
    }
    for v in cfg.entry_defs() {
        def_block.entry(v).or_insert(BlockIdx(0));
    }
    def_block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::promote;
    use crate::ir::*;
    use acvus_utils::{LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    fn label(n: u32) -> InstKind {
        InstKind::BlockLabel {
            label: Label(n),
            params: vec![],
        }
    }

    fn cfg_of(insts: Vec<InstKind>) -> CfgBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..32 {
            factory.next();
        }
        promote(MirBody {
            demoted_diamonds: Default::default(),
            insts: insts
                .into_iter()
                .map(|kind| Inst {
                    span: acvus_ast::Span::ZERO,
                    kind,
                })
                .collect(),
            val_types: FxHashMap::default(),
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: crate::ty::Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 0,
        })
    }

    const OUTER_HEADER: u32 = 0;
    const INNER_HEADER: u32 = 1;
    const OUTER_LATCH: u32 = 2;
    const AFTER: u32 = 3;

    fn nested_loops() -> CfgBody {
        cfg_of(vec![
            InstKind::Const {
                dst: v(0),
                value: acvus_ast::Literal::Bool(true),
            },
            InstKind::Jump {
                label: Label(OUTER_HEADER),
                args: vec![],
            },
            label(OUTER_HEADER),
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(INNER_HEADER),
                then_args: vec![],
                else_label: Label(AFTER),
                else_args: vec![],
            },
            label(INNER_HEADER),
            InstKind::JumpIf {
                cond: v(0),
                then_label: Label(INNER_HEADER),
                then_args: vec![],
                else_label: Label(OUTER_LATCH),
                else_args: vec![],
            },
            label(OUTER_LATCH),
            InstKind::Jump {
                label: Label(OUTER_HEADER),
                args: vec![],
            },
            label(AFTER),
            InstKind::Return {
                value: v(0),
                order: None,
            },
        ])
    }

    #[test]
    fn the_header_is_in_its_own_body() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let loops = natural_loops_innermost_first(&cfg, &domtree);
        assert_eq!(loops.len(), 2);
        for loop_ in &loops {
            assert!(
                loop_.contains(loop_.header),
                "the loop headed at block {} does not run its own header",
                loop_.header.0
            );
        }
    }

    #[test]
    fn an_inner_loop_comes_before_the_outer_one() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let loops = natural_loops_innermost_first(&cfg, &domtree);
        let [inner, outer] = &loops[..] else {
            panic!("two nested loops, got {}", loops.len());
        };
        assert!(
            outer.contains(inner.header) && !inner.contains(outer.header),
            "the first loop is the one the second contains"
        );
    }

    #[test]
    fn a_value_defined_above_the_header_is_invariant() {
        let cfg = nested_loops();
        let domtree = DomTree::build(&cfg);
        let invariants = Invariants::of(&cfg);
        for loop_ in natural_loops_innermost_first(&cfg, &domtree) {
            assert_eq!(invariants.above(&loop_, v(0)), Some(Invariant::Outside(v(0))));
        }
    }
}
