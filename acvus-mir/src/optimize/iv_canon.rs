//! IV canonicalization: a `for` computes each induction variable that
//! anything besides its own step reads from its own counter (RFC-0066
//! rule 7).
//!
//! A header parameter `p` that `analysis::affine` derives as carried is
//! `base + k·step` over the iteration number `k`. Carried from one iteration
//! to the next, it makes each iteration wait for the one before, and it is a
//! target the stages would order (RFC-0089 rule 2). The pass writes
//! `base + k·step` at the head of the body, with `k` read off the counter the
//! terminator advances: `counter − at` for a range and the index for a slice
//! or an array. Where `p` is read after the loop, it writes `base + trip·step`
//! at the head of the exit block, from the trip count the `for` terminator
//! defines on its exit edge (RFC-0057 rule 9). A rewritten `p` is carried no
//! longer, so it is no target and orders nothing.
//!
//! The choice is per variable. It reads the parameter's derivation and its
//! readers, and nothing the stage pass decides, which runs after this one
//! (`graph/optimize.rs`). A parameter nothing reads but its own step is left
//! to `dce`, which sweeps it with its step.
//!
//! Every operand is converted to `p`'s width before the arithmetic. An
//! integer `Cast` yields the value congruent to its operand modulo `2^w`,
//! and wrapping `+`, `−` and `*` are arithmetic modulo `2^w` (RFC-0037), so
//! `(counter as w) − (at as w)` is `k` exactly at `p`'s width. Subtracting at
//! the range's own width first would not be: at a narrower width the
//! difference is `k` only modulo that width. The trip count is a `u64`,
//! which holds `max(hi − at, 0)` exactly at every width up to 64.
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
//! It adds only `BinOp`, `Cast` and `Const` instructions. Strength reduction
//! runs after the stages, and reduces only what an `InOrder` join reads
//! (RFC-0056).

use acvus_ast::Span;
use rustc_hash::FxHashMap;

use crate::analysis::affine::{AffineValues, Derivation, for_body};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopNest, edge_args};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{BinOp, ExitTrip, ForSource, Inst, InstKind, Label, ValOrigin, ValueId};
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::{CastTy, IntTy, Ty};

pub fn run(cfg: &mut CfgBody) {
    let domtree = DomTree::build(cfg);
    let nest = LoopNest::of(cfg, &domtree, &Invariants::of(cfg));
    for (_, loop_) in nest.iter() {
        let Some(shape) = Shape::of(cfg, loop_) else {
            continue;
        };
        let affine = AffineValues::of(cfg, loop_, &Invariants::of(cfg));
        let reads = Reads::of(cfg);
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
        let replacements = Replacements {
            inside: rewrite_inside(cfg, &shape, &ivs),
            after: rewrite_after(cfg, &shape, &ivs),
        };
        substitute(cfg, loop_, &shape, &domtree, &replacements);
        drop_header_params(cfg, &shape, &ivs);
    }
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

/// How many times each value is read, by instructions and terminators, in
/// the whole body.
struct Reads {
    by_value: FxHashMap<ValueId, usize>,
}

impl Reads {
    fn of(cfg: &CfgBody) -> Self {
        let mut by_value: FxHashMap<ValueId, usize> = FxHashMap::default();
        for block in &cfg.blocks {
            let insts = block
                .insts
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind));
            for value in insts.chain(inst_info::terminator_uses(&block.terminator)) {
                *by_value.entry(value).or_default() += 1;
            }
        }
        Self { by_value }
    }

    fn count(&self, value: ValueId) -> usize {
        self.by_value.get(&value).copied().unwrap_or(0)
    }
}

// -- One induction variable -----------------------------------------

struct Iv {
    header_index: usize,
    param: ValueId,
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
            width,
            init: *init,
            step: step.invariant.clone(),
            read_after,
        })
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
        let advanced = self.arith(BinOp::Mul, count, step, iv.width);
        self.arith(BinOp::Add, iv.init, advanced, iv.width)
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
                        head.arith(BinOp::Sub, counter, at, iv.width)
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
