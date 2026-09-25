//! IV canonicalization: a `for` computes each induction variable that
//! anything besides its own step reads from its own counter (RFC-0066
//! rule 7).
//!
//! A header parameter `p` that `analysis::affine` derives as carried is
//! `base + k·step` over the iteration number `k`. Carried from one iteration
//! to the next, it makes each iteration wait for the one before: it is a
//! token, and its cycle orders the iterations (RFC-0089 rule 2). The pass writes
//! `base + k·step` at the head of the body, with `k` read off the counter the
//! terminator advances: `counter − at` for a range and the index for a slice
//! or an array. Where `p` is read after the loop, it writes `base + trip·step`
//! at the head of the exit block, from the trip count the `for` terminator
//! defines on its exit edge (RFC-0057 rule 9). A rewritten `p` is carried no
//! longer, so it is no token and orders nothing.
//!
//! The choice is per variable. It reads the parameter's derivation and its
//! readers, and nothing the stage pass decides, which runs after this one
//! (`graph/optimize.rs`). A parameter nothing reads but its own step is left
//! to `dce`, which sweeps it with its step.
//!
//! Every operand is converted to `p`'s width before the arithmetic. An
//! integer `Cast` yields the value congruent to its operand modulo `2^w`,
//! and the pass's `+`, `−` and `*` wrap (RFC-0037 rule 3), which is
//! arithmetic modulo `2^w`, so `(counter as w) − (at as w)` is `k` exactly
//! at `p`'s width. Subtracting at the range's own width first would not be:
//! at a narrower width the difference is `k` only modulo that width. The
//! trip count is a `u64`, which holds `max(hi − at, 0)` exactly at every
//! width up to 64.
//!
//! The written operations wrap because they are the pass's and not the
//! program's, and their operands are right only modulo `2^w`: a `u8`
//! counter at 150 is `-106` as an `i8`, so an `i8` variable that starts at
//! `-100` and steps by one holds `50` there while `-100 + (-106)` leaves
//! the width. A trapping `+` would end a run the program defines.
//!
//! `p`'s own step, the program's trapping `+`, is what ends a run whose
//! variable leaves the width. Once `p` is rewritten the step's value reaches
//! no later iteration, and `dce` would sweep it with its trap, so the pass
//! writes a `Check` of the step at its place, over the rewritten `p`
//! (RFC-0037 rule 3). A step that cannot trap needs none: the range's own
//! counter, and a variable whose entry, step and trip count are words that
//! keep it inside the width.
//!
//! A read of `p` outside the loop takes the value it has there. A block the
//! body block dominates is reached from inside an iteration, by a `break` or
//! a `return`, and reads that iteration's `base + k·step`. A block the exit
//! block dominates, where only the header enters the exit block, reads
//! `base + trip·step`. A variable read anywhere else, such as a block both
//! the exit and a `break` reach, would need a value per incoming edge, which
//! the header's exit edge cannot state for a count the exit block defines
//! only when the header alone enters it; that variable is declined and stays
//! carried. So is one the header itself reads: the header runs before the
//! body's head.
//!
//! A `while` is declined. Nothing states its trip count, so an `Iv` read
//! after it has no exit value to take (RFC-0066 rule 2). `while_to_for`
//! makes a `for` of a `while` where that is exact, and this pass then
//! applies to the result unchanged.
//!
//! It adds only `BinOp`, `Cast` and `Const` instructions for an `Iv`.
//! Strength reduction runs after the stages, and reduces only what an
//! `InOrder` stage reads (RFC-0056).
//!
//! A header parameter every back edge sends `f(e)`, `e` the element read at
//! the counter of a slice source and `f` pure work on `e` and invariants,
//! is `f` of the element at `k − 1` and its entry value at `k = 0`
//! (RFC-0066 rule 7). The pass reads it so with a branch on `k = 0` at the
//! body's head and `f` run again on `source[k − 1]`, and the parameter
//! carries nothing. The source is a shared slice, which no iteration
//! writes, so the element read again is the one read before. An array
//! source is declined.

use acvus_ast::{Literal, Span};
use rustc_hash::FxHashMap;

use crate::analysis::affine::{AffineValues, Derivation, for_body};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info::{self, Reads};
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopNest, edge_args};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{
    BinOp, Checked, ExitTrip, ForSource, IndexBound, IndexMode, Inst, InstKind, Label, Overflow,
    RefTarget, UnaryOp, ValOrigin, ValueId,
};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::{CastTy, IntTy, LenTerm, Ty};

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let domtree = DomTree::build(cfg);
    let nest = LoopNest::of(cfg, &domtree, &Invariants::of(cfg));
    for (_, loop_) in nest.iter() {
        let Some(shape) = Shape::of(cfg, loop_) else {
            continue;
        };
        let affine = AffineValues::of(cfg, loop_, &Invariants::of(cfg), laws);
        let reads = Reads::in_body(cfg);
        let ivs: Vec<Iv> = cfg.blocks[shape.header.0]
            .params
            .clone()
            .into_iter()
            .enumerate()
            .filter_map(|(header_index, param)| {
                Iv::of(cfg, loop_, &shape, &domtree, &affine, &reads, header_index, param)
            })
            .collect();
        if ivs.is_empty() {
            continue;
        }
        let invariants = Invariants::of(cfg);
        let can_trap: Vec<&Iv> = ivs
            .iter()
            .filter(|iv| {
                !iv.is_the_counter(cfg, &shape, &invariants)
                    && !iv.fits_on_every_iteration(cfg, &shape, &invariants)
            })
            .collect();
        for iv in can_trap {
            keep_step_trap(cfg, loop_, iv);
        }
        let replacements = Replacements {
            inside: rewrite_inside(cfg, &shape, &ivs),
            after: rewrite_after(cfg, &shape, &ivs),
        };
        substitute(cfg, loop_, &shape, &domtree, &replacements);
        drop_header_params(cfg, &shape, &ivs);
    }
    rewrite_neighbours(cfg);
}

// -- The loop's blocks ------------------------------------------------

struct Shape {
    source: ForSource,
    header: BlockIdx,
    header_label: Label,
    body: BlockIdx,
    /// The exit block, where the header alone enters it: only then may its
    /// edge define the trip count (RFC-0057 rule 9).
    exit: Option<BlockIdx>,
}

impl Shape {
    fn of(cfg: &CfgBody, loop_: &Loop) -> Option<Shape> {
        let header = loop_.natural.header;
        let Terminator::For { source, exit, .. } = &cfg.blocks[header.0].terminator else {
            return None;
        };
        let source = *source;
        let body = for_body(cfg, header);
        let exit = *cfg.label_to_block.get(exit)?;
        let preds = cfg.predecessors();
        let entered_by_header_alone =
            |block: BlockIdx| preds.get(&block).is_some_and(|from| from[..] == [header]);
        entered_by_header_alone(body).then(|| Shape {
            source,
            header,
            header_label: cfg.blocks[header.0].label,
            body,
            exit: entered_by_header_alone(exit).then_some(exit),
        })
    }
}

// -- One induction variable -----------------------------------------

struct Iv {
    header_index: usize,
    param: ValueId,
    next: ValueId,
    width: IntTy,
    init: ValueId,
    step: Invariant,
    read_after: bool,
}

impl Iv {
    /// `Some` for a carried integer `Iv` that something besides its own
    /// step reads, and whose every read outside the loop is one the rewrite
    /// can state (the module's text).
    #[allow(clippy::too_many_arguments)]
    fn of(
        cfg: &CfgBody,
        loop_: &Loop,
        shape: &Shape,
        domtree: &DomTree,
        affine: &AffineValues,
        reads: &Reads,
        header_index: usize,
        param: ValueId,
    ) -> Option<Iv> {
        let Derivation::Carried { init, step } = &affine.get(param)?.derivation else {
            return None;
        };
        // `base + k·step` is written at the head of the body and after the
        // loop, where a standing step is not yet computed, or never is.
        let step = step.invariance.above()?;
        let Ty::Int(width) = cfg.val_types[&param] else {
            return None;
        };
        let header = &cfg.blocks[shape.header.0];
        if reads_value(header, param) {
            return None;
        }
        let next = loop_.natural.back_arg(cfg, header_index)?;
        let sent_back = loop_
            .natural
            .latches
            .iter()
            .filter_map(|latch| edge_args(&cfg.blocks[latch.0].terminator, shape.header_label))
            .flatten()
            .filter(|arg| **arg == next)
            .count();
        let only_its_step_reads = reads.count(param) == 1 && reads.count(next) == sent_back;
        if only_its_step_reads {
            return None;
        }
        let mut read_after = false;
        let outside = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| (BlockIdx(at), block))
            .filter(|(at, _)| !loop_.natural.contains(*at))
            .filter(|(_, block)| reads_value(block, param));
        for (at, _) in outside {
            match Reading::of(shape, domtree, at) {
                Some(Reading::Inside) => {}
                Some(Reading::After) => read_after = true,
                None => return None,
            }
        }
        Some(Iv {
            header_index,
            param,
            next,
            width,
            init: *init,
            step: step.clone(),
            read_after,
        })
    }
}

impl Iv {
    /// Whether the variable is the range's own counter, entered with `at`
    /// and stepped by one at the range's width: then its step is at most
    /// `hi`, which the width holds, and cannot trap (RFC-0081 rule 4).
    fn is_the_counter(&self, cfg: &CfgBody, shape: &Shape, invariants: &Invariants) -> bool {
        let ForSource::Range { at, .. } = shape.source else {
            return false;
        };
        let one = match &self.step {
            Invariant::Word(literal) => Some(literal),
            Invariant::Outside(value) => invariants.word(*value),
        };
        let entered_at = self.init == at
            || invariants
                .word(self.init)
                .zip(invariants.word(at))
                .is_some_and(|(init, at)| init.desugared() == at.desugared());
        entered_at
            && cfg.val_types[&at] == Ty::Int(self.width)
            && one.is_some_and(|literal| literal.desugared() == Literal::Int(1))
    }

    /// Whether the entry, the step and the trip count are words and every
    /// value the step computes, `base + (k + 1)·step` for `k` below the
    /// count, fits the width: the step cannot trap.
    fn fits_on_every_iteration(
        &self,
        cfg: &CfgBody,
        shape: &Shape,
        invariants: &Invariants,
    ) -> bool {
        let int = |value: ValueId| match (invariants.word(value), &cfg.val_types[&value]) {
            (Some(literal), Ty::Int(width)) => match literal.desugared() {
                Literal::Int(n) => Some(width.read(n as u64)),
                _ => None,
            },
            _ => None,
        };
        let step = match &self.step {
            Invariant::Outside(value) => int(*value),
            Invariant::Word(literal) => match literal.desugared() {
                Literal::Int(n) => Some(self.width.read(n as u64)),
                _ => None,
            },
        };
        let trip = match shape.source {
            ForSource::Range { at, hi } => int(at).zip(int(hi)).map(|(at, hi)| (hi - at).max(0)),
            ForSource::Array(array) => match &cfg.val_types[&array] {
                Ty::Array(_, LenTerm::Known(len)) => i128::try_from(*len).ok(),
                _ => None,
            },
            ForSource::Slice(_) | ForSource::SliceMut(_) => None,
        };
        let (Some(base), Some(step), Some(trip)) = (int(self.init), step, trip) else {
            return false;
        };
        trip == 0
            || [base + step, base + trip * step]
                .into_iter()
                .all(|value| self.width.holds(value))
    }
}

/// Which value a parameter's read in a block holds there.
enum Reading {
    /// The iteration's `base + k·step`.
    Inside,
    /// `base + trip·step`.
    After,
}

impl Reading {
    /// For a block outside the loop; `None` where neither holds on every
    /// path to it.
    fn of(shape: &Shape, domtree: &DomTree, block: BlockIdx) -> Option<Reading> {
        if domtree.dominates(shape.body, block) {
            return Some(Reading::Inside);
        }
        let exit = shape.exit?;
        domtree.dominates(exit, block).then_some(Reading::After)
    }
}

// -- The rewrite -----------------------------------------------------

fn keep_step_trap(cfg: &mut CfgBody, loop_: &Loop, iv: &Iv) {
    let (block, at) = loop_
        .natural
        .blocks()
        .find_map(|block| {
            cfg.blocks[block.0]
                .insts
                .iter()
                .position(|inst| inst_info::defs(&inst.kind).contains(&iv.next))
                .map(|at| (block, at))
        })
        .unwrap_or_else(|| {
            panic!(
                "{:?} is an `Iv`'s step, which `analysis::affine` read off an instruction of \
                 the loop",
                iv.next
            )
        });
    let step = &cfg.blocks[block.0].insts[at];
    let InstKind::BinOp {
        op, left, right, ..
    } = step.kind
    else {
        panic!("an `Iv`'s step is the `BinOp` `analysis::affine` read, not {step:?}")
    };
    let Some(op) = Checked::of_trapping(op) else {
        return;
    };
    let check = Inst {
        span: step.span,
        kind: InstKind::Check { op, left, right },
    };
    cfg.blocks[block.0].insts.insert(at, check);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Converted {
    value: ValueId,
    width: IntTy,
}

struct BlockHead<'a> {
    cfg: &'a mut CfgBody,
    span: Span,
    insts: Vec<Inst>,
    conversions: FxHashMap<Converted, ValueId>,
}

impl<'a> BlockHead<'a> {
    fn of(cfg: &'a mut CfgBody, block: BlockIdx) -> Self {
        let span = cfg.blocks[block.0]
            .insts
            .first()
            .map_or(Span::ZERO, |inst| inst.span);
        Self {
            cfg,
            span,
            insts: Vec::new(),
            conversions: FxHashMap::default(),
        }
    }

    fn fresh(&mut self, width: IntTy) -> ValueId {
        let value = self.cfg.val_factory.next();
        let previous = self.cfg.val_types.insert(value, Ty::Int(width));
        assert!(
            previous.is_none(),
            "{value:?} is fresh from the factory and already carried a type"
        );
        self.cfg.debug.set(value, ValOrigin::Expr);
        value
    }

    fn push(&mut self, kind: InstKind) {
        self.insts.push(Inst {
            span: self.span,
            kind,
        });
    }

    fn at_width(&mut self, value: ValueId, width: IntTy) -> ValueId {
        if self.cfg.val_types[&value] == Ty::Int(width) {
            return value;
        }
        let key = Converted { value, width };
        if let Some(converted) = self.conversions.get(&key) {
            return *converted;
        }
        let dst = self.fresh(width);
        self.push(InstKind::Cast {
            dst,
            src: value,
            to: CastTy::Int(width),
        });
        self.conversions.insert(key, dst);
        dst
    }

    fn arith(&mut self, op: BinOp, left: ValueId, right: ValueId, width: IntTy) -> ValueId {
        let dst = self.fresh(width);
        self.push(InstKind::BinOp {
            dst,
            op,
            left,
            right,
        });
        dst
    }

    fn read(&mut self, invariant: &Invariant, width: IntTy) -> ValueId {
        match invariant {
            Invariant::Outside(value) => *value,
            Invariant::Word(literal) => {
                let dst = self.fresh(width);
                self.push(InstKind::Const {
                    dst,
                    value: literal.clone(),
                });
                dst
            }
        }
    }

    fn advanced_by(&mut self, iv: &Iv, count: ValueId) -> ValueId {
        let step = self.read(&iv.step, iv.width);
        let advanced = self.arith(BinOp::Mul(Overflow::Wrap), count, step, iv.width);
        self.arith(BinOp::Add(Overflow::Wrap), iv.init, advanced, iv.width)
    }

    fn prepend_to(self, block: BlockIdx) {
        let insts = &mut self.cfg.blocks[block.0].insts;
        let held = std::mem::replace(insts, self.insts);
        insts.extend(held);
    }
}

fn rewrite_inside(cfg: &mut CfgBody, shape: &Shape, ivs: &[Iv]) -> FxHashMap<ValueId, ValueId> {
    let counter = cfg.blocks[shape.body.0].params[shape.source.counter_param()];
    let mut head = BlockHead::of(cfg, shape.body);
    let mut iteration: FxHashMap<IntTy, ValueId> = FxHashMap::default();
    let mut inside = FxHashMap::default();
    for iv in ivs {
        let k = match iteration.get(&iv.width) {
            Some(k) => *k,
            None => {
                let k = match shape.source {
                    ForSource::Range { at, .. } => {
                        let counter = head.at_width(counter, iv.width);
                        let at = head.at_width(at, iv.width);
                        head.arith(BinOp::Sub(Overflow::Wrap), counter, at, iv.width)
                    }
                    ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => {
                        head.at_width(counter, iv.width)
                    }
                };
                iteration.insert(iv.width, k);
                k
            }
        };
        inside.insert(iv.param, head.advanced_by(iv, k));
    }
    head.prepend_to(shape.body);
    inside
}

fn rewrite_after(cfg: &mut CfgBody, shape: &Shape, ivs: &[Iv]) -> FxHashMap<ValueId, ValueId> {
    let read_after: Vec<&Iv> = ivs.iter().filter(|iv| iv.read_after).collect();
    let mut after = FxHashMap::default();
    let Some(exit) = shape.exit.filter(|_| !read_after.is_empty()) else {
        return after;
    };
    let trip = trip_count_on_exit(cfg, shape.header, exit);
    let mut head = BlockHead::of(cfg, exit);
    for iv in read_after {
        let trip = head.at_width(trip, iv.width);
        after.insert(iv.param, head.advanced_by(iv, trip));
    }
    head.prepend_to(exit);
    after
}

/// Each map replaces a parameter, its key, with the value it holds there.
struct Replacements {
    inside: FxHashMap<ValueId, ValueId>,
    after: FxHashMap<ValueId, ValueId>,
}

/// Every block of the loop but the header, and every block outside it the
/// body block dominates, reads the iteration's value; every other block
/// that reads a rewritten parameter is one the exit block dominates
/// (`Iv::of`).
fn substitute(
    cfg: &mut CfgBody,
    loop_: &Loop,
    shape: &Shape,
    domtree: &DomTree,
    replacements: &Replacements,
) {
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let block_idx = BlockIdx(bi);
        if block_idx == shape.header {
            continue;
        }
        let inside =
            loop_.natural.contains(block_idx) || domtree.dominates(shape.body, block_idx);
        let subst = match inside {
            true => &replacements.inside,
            false => &replacements.after,
        };
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, subst);
        }
        apply_subst_terminator(&mut block.terminator, subst);
    }
}

fn reads_value(block: &crate::cfg::Block, value: ValueId) -> bool {
    block
        .insts
        .iter()
        .flat_map(|inst| inst_info::uses(&inst.kind))
        .chain(inst_info::terminator_uses(&block.terminator))
        .any(|used| used == value)
}

fn trip_count_on_exit(cfg: &mut CfgBody, header: BlockIdx, exit: BlockIdx) -> ValueId {
    let Terminator::For { exit_trip, .. } = &mut cfg.blocks[header.0].terminator else {
        panic!("block {} is a `for` header and does not end in `For`", header.0)
    };
    match exit_trip {
        ExitTrip::Defined => ExitTrip::Defined
            .trip_param(&cfg.blocks[exit.0].params)
            .expect("an exit edge that defines the trip count enters a block with a parameter"),
        ExitTrip::Absent => {
            *exit_trip = ExitTrip::Defined;
            let trip = cfg.val_factory.next();
            let previous = cfg.val_types.insert(trip, Ty::U64);
            assert!(
                previous.is_none(),
                "{trip:?} is fresh from the factory and already carried a type"
            );
            cfg.debug.set(trip, ValOrigin::Expr);
            cfg.blocks[exit.0].params.insert(0, trip);
            trip
        }
    }
}

fn drop_header_params(cfg: &mut CfgBody, shape: &Shape, ivs: &[Iv]) {
    let header_label = cfg.blocks[shape.header.0].label;
    let mut dropped: Vec<usize> = ivs.iter().map(|iv| iv.header_index).collect();
    dropped.sort_unstable_by(|a, b| b.cmp(a));
    for block in &mut cfg.blocks {
        for edge in edges_into(&mut block.terminator, header_label) {
            for index in &dropped {
                let at = index.checked_sub(edge.supplied_params).unwrap_or_else(|| {
                    panic!(
                        "header parameter {index} is one an edge's terminator fills, \
                         and an `Iv` is a parameter every edge sends"
                    )
                });
                edge.args.remove(at);
            }
        }
    }
    let params = &mut cfg.blocks[shape.header.0].params;
    for index in dropped {
        params.remove(index);
    }
}

struct EdgeInto<'a> {
    supplied_params: usize,
    args: &'a mut Vec<ValueId>,
}

/// The edges of `term` into `label`. A `For`'s body edge passes nothing, so
/// only its exit edge is one (RFC-0089 rule 1).
fn edges_into(term: &mut Terminator, label: Label) -> Vec<EdgeInto<'_>> {
    let carried = |args| EdgeInto {
        supplied_params: 0,
        args,
    };
    match term {
        Terminator::Jump { label: to, args } => {
            (*to == label).then(|| carried(args)).into_iter().collect()
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
            let mut edges = Vec::new();
            if *then_label == label {
                edges.push(carried(then_args));
            }
            if *else_label == label {
                edges.push(carried(else_args));
            }
            edges
        }
        Terminator::Switch { arms, default, .. } => arms
            .iter_mut()
            .map(|(_, to, args)| (to, args))
            .chain(default.iter_mut().map(|(to, args)| (to, args)))
            .filter(|(to, _)| **to == label)
            .map(|(_, args)| carried(args))
            .collect(),
        Terminator::For {
            exit,
            exit_trip,
            exit_args,
            ..
        } => (*exit == label)
            .then(|| EdgeInto {
                supplied_params: exit_trip.supplied_params(),
                args: exit_args,
            })
            .into_iter()
            .collect(),
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

// -- A header parameter carrying work on the previous element ----------

/// A header parameter every back edge sends `f(e)`, `e` the element the
/// iteration read at the counter of a slice source and `f` pure work on
/// `e` and invariants: from the second iteration on it holds `f` of the
/// element at `k − 1`, and on the first its entry value (RFC-0066 rule 7).
struct Neighbour {
    header_index: usize,
    param: ValueId,
    init: ValueId,
    source: ValueId,
    element: ValueId,
    work: Vec<Work>,
    sent: Sent,
}

/// What the back edges send: the element itself, or the value the last
/// step of `f` defines.
#[derive(Clone, Copy)]
enum Sent {
    Element,
    Computed(ValueId),
}

/// One step of `f`, in an order that defines each operand before its
/// reader.
#[derive(Clone)]
struct Work {
    at: InstAt,
    kind: InstKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct InstAt {
    block: BlockIdx,
    at: usize,
}

/// Rewrites one such parameter at a time, since each rewrite adds blocks
/// to its loop and the nest is read again after it.
fn rewrite_neighbours(cfg: &mut CfgBody) {
    loop {
        let domtree = DomTree::build(cfg);
        let invariants = Invariants::of(cfg);
        let nest = LoopNest::of(cfg, &domtree, &invariants);
        let found = nest.iter().find_map(|(_, loop_)| {
            let shape = Shape::of(cfg, loop_)?;
            let neighbour = Neighbour::find(cfg, loop_, &shape, &invariants)?;
            Some((shape, neighbour))
        });
        let Some((shape, neighbour)) = found else {
            return;
        };
        neighbour.rewrite(cfg, &shape);
    }
}

impl Neighbour {
    fn find(cfg: &CfgBody, loop_: &Loop, shape: &Shape, invariants: &Invariants) -> Option<Self> {
        let ForSource::Slice(source) = shape.source else {
            return None;
        };
        let [element_ref, counter] = cfg.blocks[shape.body.0].params[..] else {
            return None;
        };
        let header = &cfg.blocks[shape.header.0];
        header
            .params
            .iter()
            .enumerate()
            .find_map(|(header_index, &param)| {
                let init = loop_.natural.entry_arg(cfg, header_index)?;
                let next = loop_.natural.back_arg(cfg, header_index)?;
                let read_outside = cfg
                    .blocks
                    .iter()
                    .enumerate()
                    .filter(|(at, _)| {
                        !loop_.natural.contains(BlockIdx(*at)) || BlockIdx(*at) == shape.header
                    })
                    .any(|(_, block)| reads_value(block, param));
                if next == param || read_outside {
                    return None;
                }
                let mut reading = PreviousWork {
                    cfg,
                    loop_,
                    invariants,
                    source,
                    element_ref,
                    counter,
                    element: None,
                    work: Vec::new(),
                };
                reading.read(next)?;
                let element = reading.element?;
                let sent = match next == element {
                    true => Sent::Element,
                    false => Sent::Computed(next),
                };
                Some(Neighbour {
                    header_index,
                    param,
                    init,
                    source,
                    element,
                    work: reading.work,
                    sent,
                })
            })
    }

    /// The body block `B(e, k)` becomes `B(e, k): if k == 0 -> J(init) else
    /// R`, `R: J(f(source[k − 1]))`, and `J(p)` holds what `B` held, with
    /// `p` for the parameter. `f` runs again in `R` with its arithmetic
    /// wrapping; it cannot trap there, since the same `f` of the same
    /// element ran at the iteration before and a trap would have ended the
    /// run there. Each trapping step of `f` left in the body, which `dce`
    /// sweeps once nothing reads it, keeps its trap by a `Check` at its
    /// place (RFC-0037 rule 3).
    fn rewrite(self, cfg: &mut CfgBody, shape: &Shape) {
        let mut checks: Vec<(InstAt, InstKind)> = self
            .work
            .iter()
            .filter_map(|step| match step.kind {
                InstKind::BinOp {
                    op, left, right, ..
                } => Checked::of_trapping(op)
                    .map(|op| (step.at, InstKind::Check { op, left, right })),
                _ => None,
            })
            .collect();
        checks.sort_by_key(|(at, _)| std::cmp::Reverse((at.block, at.at)));
        for (at, kind) in checks {
            let span = cfg.blocks[at.block.0].insts[at.at].span;
            cfg.blocks[at.block.0].insts.insert(at.at, Inst { span, kind });
        }

        let body = shape.body;
        let counter = cfg.blocks[body.0].params[shape.source.counter_param()];
        let span = cfg.blocks[body.0]
            .insts
            .first()
            .map_or(Span::ZERO, |inst| inst.span);
        let mut labels = cfg
            .blocks
            .iter()
            .map(|block| block.label)
            .filter(|label| *label != crate::cfg::ENTRY_LABEL)
            .map(|label| label.0 + 1)
            .max()
            .unwrap_or(0);
        let mut fresh_label = || {
            let label = Label(labels);
            labels += 1;
            label
        };
        let rest_label = fresh_label();
        let join_label = fresh_label();
        let fresh = |cfg: &mut CfgBody, ty: Ty| {
            let value = cfg.val_factory.next();
            let previous = cfg.val_types.insert(value, ty);
            assert!(
                previous.is_none(),
                "{value:?} is fresh from the factory and already carried a type"
            );
            cfg.debug.set(value, ValOrigin::Expr);
            value
        };
        let at = |kind: InstKind| Inst { span, kind };

        let held = cfg.val_types[&self.param].clone();
        let joined = fresh(cfg, held);
        let zero = fresh(cfg, Ty::U64);
        let first = fresh(cfg, Ty::Bool);
        let one = fresh(cfg, Ty::U64);
        let before = fresh(cfg, Ty::U64);
        let element_ty = cfg.val_types[&self.element].clone();
        let previous = fresh(cfg, element_ty);

        let mut rest = vec![
            at(InstKind::Const {
                dst: one,
                value: Literal::Int(1),
            }),
            at(InstKind::BinOp {
                dst: before,
                op: BinOp::Sub(Overflow::Wrap),
                left: counter,
                right: one,
            }),
            at(InstKind::Index {
                dst: previous,
                slice: self.source,
                index: before,
                mode: IndexMode::Copy,
                bound: IndexBound::Checked,
            }),
        ];
        let mut renamed: FxHashMap<ValueId, ValueId> = FxHashMap::from_iter([(self.element, previous)]);
        for step in &self.work {
            let mut kind = step.kind.clone();
            for def in inst_info::defs(&kind) {
                let ty = cfg.val_types[&def].clone();
                renamed.insert(def, fresh(cfg, ty));
            }
            apply_subst(&mut kind, &renamed);
            rename_defs(&mut kind, &renamed);
            if let InstKind::BinOp { op, .. } = &mut kind
                && let Some(checked) = Checked::of_trapping(*op)
            {
                *op = checked.wrapping_op();
            }
            rest.push(at(kind));
        }
        let sent_back = match self.sent {
            Sent::Element => previous,
            Sent::Computed(value) => renamed[&value],
        };

        let block = &mut cfg.blocks[body.0];
        let insts = std::mem::replace(
            &mut block.insts,
            vec![
                at(InstKind::Const {
                    dst: zero,
                    value: Literal::Int(0),
                }),
                at(InstKind::BinOp {
                    dst: first,
                    op: BinOp::Eq,
                    left: counter,
                    right: zero,
                }),
            ],
        );
        let terminator = std::mem::replace(
            &mut block.terminator,
            Terminator::JumpIf {
                cond: first,
                then_label: join_label,
                then_args: vec![self.init],
                else_label: rest_label,
                else_args: Vec::new(),
            },
        );
        for (label, params, insts, terminator) in [
            (
                rest_label,
                Vec::new(),
                rest,
                Terminator::Jump {
                    label: join_label,
                    args: vec![sent_back],
                },
            ),
            (join_label, vec![joined], insts, terminator),
        ] {
            cfg.label_to_block.insert(label, BlockIdx(cfg.blocks.len()));
            cfg.blocks.push(crate::cfg::Block {
                label,
                params,
                insts,
                terminator,
            });
        }

        let body_label = cfg.blocks[body.0].label;
        if cfg.demoted_diamonds.remove(&body_label) {
            cfg.demoted_diamonds.insert(join_label);
        }
        let subst = FxHashMap::from_iter([(self.param, joined)]);
        for (at, block) in cfg.blocks.iter_mut().enumerate() {
            if BlockIdx(at) == shape.header {
                continue;
            }
            for inst in &mut block.insts {
                apply_subst(&mut inst.kind, &subst);
            }
            apply_subst_terminator(&mut block.terminator, &subst);
        }
        drop_header_param(cfg, shape, self.header_index);
        remove_unread_element_take(cfg, self.element);
    }
}

/// `dce` keeps every `Take`, since one may empty the storage it names. A
/// word taken through the element's shared reference is a copy that
/// empties nothing, so the element read the back edges alone read goes
/// with them.
fn remove_unread_element_take(cfg: &mut CfgBody, element: ValueId) {
    if Reads::in_body(cfg).count(element) != 0 {
        return;
    }
    for block in &mut cfg.blocks {
        block.insts.retain(|inst| {
            !matches!(&inst.kind, InstKind::Take {
                dst,
                target: RefTarget::Through(_),
                ..
            } if *dst == element)
        });
    }
}

/// Reads `f` back from the value the back edges send, as far as it is
/// pure work on the one element read at the counter and on invariants.
struct PreviousWork<'a> {
    cfg: &'a CfgBody,
    loop_: &'a Loop,
    invariants: &'a Invariants,
    source: ValueId,
    element_ref: ValueId,
    counter: ValueId,
    element: Option<ValueId>,
    work: Vec<Work>,
}

impl PreviousWork<'_> {
    fn read(&mut self, value: ValueId) -> Option<()> {
        if self.element == Some(value) || self.work.iter().any(|step| inst_info::defs(&step.kind).contains(&value)) {
            return Some(());
        }
        if let Some(invariant) = self.invariants.above(&self.loop_.natural, value) {
            return match invariant {
                Invariant::Outside(_) => Some(()),
                Invariant::Word(_) => {
                    let at = self.def_of(value)?;
                    self.work.push(Work {
                        at,
                        kind: self.cfg.blocks[at.block.0].insts[at.at].kind.clone(),
                    });
                    Some(())
                }
            };
        }
        let at = self.def_of(value)?;
        let kind = &self.cfg.blocks[at.block.0].insts[at.at].kind;
        match kind {
            InstKind::Take {
                target: RefTarget::Through(reference),
                path,
                taken_out: false,
                ..
            } if *reference == self.element_ref && path.is_empty() => self.read_element(value),
            InstKind::Index {
                slice,
                index,
                mode: IndexMode::Copy,
                ..
            } if *slice == self.source && *index == self.counter => self.read_element(value),
            InstKind::BinOp {
                op, left, right, ..
            } => {
                let traps_uncheckably =
                    op.can_trap_on_integers() && Checked::of_trapping(*op).is_none();
                if traps_uncheckably {
                    return None;
                }
                let (left, right) = (*left, *right);
                self.read(left)?;
                self.read(right)?;
                self.work.push(Work {
                    at,
                    kind: kind.clone(),
                });
                Some(())
            }
            InstKind::UnaryOp {
                op: UnaryOp::Not | UnaryOp::Neg(Overflow::Wrap),
                operand,
                ..
            } => {
                let operand = *operand;
                self.read(operand)?;
                self.work.push(Work {
                    at,
                    kind: kind.clone(),
                });
                Some(())
            }
            _ => None,
        }
    }

    /// The element is read once, and is a word read again at `k − 1`.
    fn read_element(&mut self, value: ValueId) -> Option<()> {
        if self.element.is_some_and(|read| read != value) {
            return None;
        }
        self.cfg.val_types[&value].is_word().filter(|word| *word)?;
        self.element = Some(value);
        Some(())
    }

    fn def_of(&self, value: ValueId) -> Option<InstAt> {
        self.loop_.natural.blocks().find_map(|block| {
            self.cfg.blocks[block.0]
                .insts
                .iter()
                .position(|inst| inst_info::defs(&inst.kind).contains(&value))
                .map(|at| InstAt { block, at })
        })
    }
}

fn rename_defs(kind: &mut InstKind, renamed: &FxHashMap<ValueId, ValueId>) {
    match kind {
        InstKind::BinOp { dst, .. } | InstKind::UnaryOp { dst, .. } | InstKind::Const { dst, .. } => {
            *dst = renamed[dst];
        }
        other => panic!("`f` holds only `BinOp`, `UnaryOp` and `Const`, not {other:?}"),
    }
}

fn drop_header_param(cfg: &mut CfgBody, shape: &Shape, index: usize) {
    let header_label = cfg.blocks[shape.header.0].label;
    for block in &mut cfg.blocks {
        for edge in edges_into(&mut block.terminator, header_label) {
            let at = index.checked_sub(edge.supplied_params).unwrap_or_else(|| {
                panic!(
                    "header parameter {index} is one an edge's terminator fills, and a \
                     parameter the back edges send is one every edge sends"
                )
            });
            edge.args.remove(at);
        }
    }
    cfg.blocks[shape.header.0].params.remove(index);
}
