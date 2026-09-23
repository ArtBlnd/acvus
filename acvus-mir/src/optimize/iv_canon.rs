//! IV canonicalization: a weak `for` computes each induction variable from
//! its own counter (RFC-0066 rule 7).
//!
//! A header parameter `p` that `analysis::carried` classifies as an `Iv` is
//! `base + k·step` over the iteration number `k`. Carried from one iteration
//! to the next, it makes each iteration wait for the one before, which a weak
//! loop's iterations need not do. The pass writes `base + k·step` at the head
//! of the body, with `k` read off the counter the terminator advances:
//! `counter − at` for a range and the index for a slice or an array. Where
//! `p` is read after the loop, it writes `base + trip·step` at the head of
//! the exit block, from the trip count the `for` terminator defines on its
//! exit edge (RFC-0057 rule 9).
//!
//! Every operand is converted to `p`'s width before the arithmetic. An
//! integer `Cast` yields the value congruent to its operand modulo `2^w`,
//! and wrapping `+`, `−` and `*` are arithmetic modulo `2^w` (RFC-0037), so
//! `(counter as w) − (at as w)` is `k` exactly at `p`'s width. Subtracting at
//! the range's own width first would not be: at a narrower width the
//! difference is `k` only modulo that width. The trip count is a `u64`,
//! which holds `max(hi − at, 0)` exactly at every width up to 64.
//!
//! A `while` is declined. Nothing states its trip count, so an `Iv` read
//! after it has no exit value to take (RFC-0066 rule 2). A later pass is to
//! make a `for` of a `while` where that is exact, and this pass then applies
//! to the result unchanged.
//!
//! `optimize::lsr` reduces strong loops and declines weak ones, and this
//! pass rewrites weak loops and declines strong ones. It adds only `BinOp`,
//! `Cast` and `Const` instructions, which change no loop's strength, and it
//! runs before `lsr` (`graph/optimize.rs`), so no loop is rewritten by both.

use acvus_ast::{BinOp, Span};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::{AffineValues, Derivation, for_body};
use crate::analysis::carried::{Carried, CarriedParam, CarriedState, Strength};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::{Loans, Summaries};
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopNest};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ExitTrip, ForSource, Inst, InstKind, Label, ValOrigin, ValueId};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator};
use crate::ty::{CastTy, IntTy, Ty};

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let domtree = DomTree::build(cfg);
    let nest = LoopNest::of(cfg, &domtree, &Invariants::of(cfg));
    for (_, loop_) in nest.iter() {
        let Some(shape) = Shape::of(cfg, loop_) else {
            continue;
        };
        let affine = AffineValues::of(cfg, loop_, &Invariants::of(cfg));
        let loans = Loans::build(cfg, Summaries::NONE);
        let state = CarriedState::of(cfg, loop_, &affine, &loans, laws);
        if state.strength() != Strength::Weak {
            continue;
        }
        let ivs: Vec<Iv> = state
            .params
            .iter()
            .enumerate()
            .filter(|(_, carried)| carried.carried == Carried::Iv)
            .filter_map(|(header_index, carried)| {
                Iv::of(cfg, &shape, &affine, header_index, carried)
            })
            .collect();
        if ivs.is_empty() {
            continue;
        }
        let replacements = Replacements {
            inside: rewrite_inside(cfg, &shape, &ivs),
            after: rewrite_after(cfg, loop_, &shape, &ivs),
        };
        substitute(cfg, loop_, &shape, &replacements);
        let sent = drop_header_params(cfg, &shape, &ivs);
        sweep(cfg, loop_, sent);
    }
}

// -- The loop's three blocks ----------------------------------------

struct Shape {
    source: ForSource,
    header: BlockIdx,
    body: BlockIdx,
    exit: BlockIdx,
}

impl Shape {
    fn of(cfg: &CfgBody, loop_: &Loop) -> Option<Shape> {
        let header = loop_.natural.header;
        let Terminator::For { source, exit, .. } = &cfg.blocks[header.0].terminator else {
            return None;
        };
        let body = for_body(cfg, header);
        let exit = *cfg.label_to_block.get(exit)?;
        let preds = cfg.predecessors();
        let entered_by_header_alone =
            |block: BlockIdx| preds.get(&block).is_some_and(|from| from[..] == [header]);
        (entered_by_header_alone(body) && entered_by_header_alone(exit)).then_some(Shape {
            source: *source,
            header,
            body,
            exit,
        })
    }
}

// -- One induction variable -----------------------------------------

struct Iv {
    header_index: usize,
    param: ValueId,
    width: IntTy,
    init: ValueId,
    step: Invariant,
}

impl Iv {
    /// `None` also where the header reads the parameter: the header runs
    /// before the body's head, and its exit edge would carry the parameter
    /// rather than its value after the loop.
    fn of(
        cfg: &CfgBody,
        shape: &Shape,
        affine: &AffineValues,
        header_index: usize,
        carried: &CarriedParam,
    ) -> Option<Iv> {
        let param = carried.param;
        let Derivation::Carried { init, step } = &affine.get(param)?.derivation else {
            return None;
        };
        let Ty::Int(width) = cfg.val_types[&param] else {
            return None;
        };
        let header = &cfg.blocks[shape.header.0];
        let read_in_header = header
            .insts
            .iter()
            .flat_map(|inst| inst_info::uses(&inst.kind))
            .chain(inst_info::terminator_uses(&header.terminator))
            .any(|used| used == param);
        (!read_in_header).then(|| Iv {
            header_index,
            param,
            width,
            init: *init,
            step: step.invariant.clone(),
        })
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

fn rewrite_after(
    cfg: &mut CfgBody,
    loop_: &Loop,
    shape: &Shape,
    ivs: &[Iv],
) -> FxHashMap<ValueId, ValueId> {
    let read_after: Vec<&Iv> = ivs
        .iter()
        .filter(|iv| read_outside(cfg, loop_, iv.param))
        .collect();
    let mut after = FxHashMap::default();
    if read_after.is_empty() {
        return after;
    }
    let trip = trip_count_on_exit(cfg, shape);
    let mut head = BlockHead::of(cfg, shape.exit);
    for iv in read_after {
        let trip = head.at_width(trip, iv.width);
        after.insert(iv.param, head.advanced_by(iv, trip));
    }
    head.prepend_to(shape.exit);
    after
}

/// Each map replaces a parameter, its key, with the value it holds there.
struct Replacements {
    inside: FxHashMap<ValueId, ValueId>,
    after: FxHashMap<ValueId, ValueId>,
}

/// `analysis::carried` classifies a loop left anywhere but its header as
/// strong, so a weak loop is left by the header's exit edge alone, and a
/// block outside the loop that reads a parameter is one the exit block
/// dominates.
fn substitute(cfg: &mut CfgBody, loop_: &Loop, shape: &Shape, replacements: &Replacements) {
    for (bi, block) in cfg.blocks.iter_mut().enumerate() {
        let block_idx = BlockIdx(bi);
        if block_idx == shape.header {
            continue;
        }
        let subst = match loop_.natural.contains(block_idx) {
            true => &replacements.inside,
            false => &replacements.after,
        };
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, subst);
        }
        apply_subst_terminator(&mut block.terminator, subst);
    }
}

fn read_outside(cfg: &CfgBody, loop_: &Loop, value: ValueId) -> bool {
    cfg.blocks
        .iter()
        .enumerate()
        .filter(|(bi, _)| !loop_.natural.contains(BlockIdx(*bi)))
        .any(|(_, block)| reads(block, value))
}

fn reads(block: &crate::cfg::Block, value: ValueId) -> bool {
    block
        .insts
        .iter()
        .flat_map(|inst| inst_info::uses(&inst.kind))
        .chain(inst_info::terminator_uses(&block.terminator))
        .any(|used| used == value)
}

fn trip_count_on_exit(cfg: &mut CfgBody, shape: &Shape) -> ValueId {
    let Terminator::For { exit_trip, .. } = &mut cfg.blocks[shape.header.0].terminator else {
        panic!(
            "block {} is a `for` header and does not end in `For`",
            shape.header.0
        )
    };
    match exit_trip {
        ExitTrip::Defined => ExitTrip::Defined
            .trip_param(&cfg.blocks[shape.exit.0].params)
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
            cfg.blocks[shape.exit.0].params.insert(0, trip);
            trip
        }
    }
}

struct SentArgs(Vec<ValueId>);

fn drop_header_params(cfg: &mut CfgBody, shape: &Shape, ivs: &[Iv]) -> SentArgs {
    let header_label = cfg.blocks[shape.header.0].label;
    let mut dropped: Vec<usize> = ivs.iter().map(|iv| iv.header_index).collect();
    dropped.sort_unstable_by(|a, b| b.cmp(a));
    let mut sent = Vec::new();
    for block in &mut cfg.blocks {
        for edge in edges_into(&mut block.terminator, header_label) {
            for index in &dropped {
                let at = index.checked_sub(edge.supplied_params).unwrap_or_else(|| {
                    panic!(
                        "header parameter {index} is one an edge's terminator fills, \
                         and an `Iv` is a parameter every edge sends"
                    )
                });
                sent.push(edge.args.remove(at));
            }
        }
    }
    let params = &mut cfg.blocks[shape.header.0].params;
    for index in dropped {
        params.remove(index);
    }
    SentArgs(sent)
}

struct EdgeInto<'a> {
    supplied_params: usize,
    args: &'a mut Vec<ValueId>,
}

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
            source,
            body,
            body_args,
            exit,
            exit_trip,
            exit_args,
        } => {
            let mut edges = Vec::new();
            if *body == label {
                edges.push(EdgeInto {
                    supplied_params: source.supplied_params(),
                    args: body_args,
                });
            }
            if *exit == label {
                edges.push(EdgeInto {
                    supplied_params: exit_trip.supplied_params(),
                    args: exit_args,
                });
            }
            edges
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

/// `dce` runs before this pass and not after it, so what the removed
/// arguments leave unread inside the loop is removed here.
fn sweep(cfg: &mut CfgBody, loop_: &Loop, sent: SentArgs) {
    let SentArgs(mut unread) = sent;
    let mut swept: FxHashSet<ValueId> = FxHashSet::default();
    while let Some(value) = unread.pop() {
        if swept.contains(&value) || cfg.blocks.iter().any(|block| reads(block, value)) {
            continue;
        }
        let defined = loop_.natural.blocks().find_map(|block| {
            cfg.blocks[block.0]
                .insts
                .iter()
                .position(|inst| inst_info::defs(&inst.kind).contains(&value))
                .map(|inst| InstAt { block, inst })
        });
        let Some(InstAt { block, inst }) = defined else {
            continue;
        };
        let kind = &mut cfg.blocks[block.0].insts[inst].kind;
        let operands: Vec<ValueId> = match kind {
            InstKind::BinOp { left, right, .. } => vec![*left, *right],
            InstKind::Cast { src, .. } => vec![*src],
            InstKind::Const { .. } => Vec::new(),
            _ => continue,
        };
        *kind = InstKind::Nop;
        swept.insert(value);
        unread.extend(operands);
    }
}

struct InstAt {
    block: BlockIdx,
    inst: usize,
}
