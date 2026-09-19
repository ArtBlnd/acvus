//! Loop strength reduction: a loop multiplies once.
//!
//! An induction variable is a header block parameter `i` whose latch
//! argument is `i + c` with `c` loop-invariant (`analysis::loops`, which
//! `code_motion` reads the same term from). An expression `i * k + x` with
//! `k` and `x` loop-invariant is then itself an induction variable: it
//! starts at `i0 * k + x` and advances by `c * k`. The pass gives it its own
//! header parameter, computes the start and the step in the preheader, adds
//! the step in the latch, and replaces the body's multiplication and sum
//! with the parameter. `i` itself stays: the loop condition reads it.
//!
//! # The count is the rule
//!
//! A reduction must lower the body's operation count. `i * k + x` sheds two
//! operations — the multiplication and the sum — and gains one, the latch's
//! addition: 4 operations to 3, counting the counter's own increment. A bare
//! `i * k`, with no invariant sum reading it, sheds one and gains one: 3 to
//! 3, plus a header parameter to carry. A change with no gain is not a
//! reduction, so the bare form is declined and its multiplication stays
//! where the program wrote it.
//!
//! The count, not the shape, is what excludes it. A bare product read by two
//! consumers is still the bare form: SSA computes it once however many
//! values read it, so reducing it still sheds one operation and gains one.
//! A product read by *two* invariant sums is a different body — three
//! operations become two derived counters and two latch additions, and the
//! count does fall — but that body is outside this pattern, which matches
//! one product and the one sum that reads it, and it is declined for that
//! reason rather than for the count.
//!
//! It runs after `code_motion` and before `reorder` (`graph/optimize.rs`).
//! After the hoist, because the hoist is what puts `k` and `x` above the
//! header and what leaves the preheader a block of its own; before the
//! reorder, because the reorder schedules within a block and this pass adds
//! instructions to three of them.
//!
//! # Why this is not a rounding change
//!
//! `(i0 + n*c) * k` and `i0*k + n*(c*k)` are the same integer: `+` and `*`
//! wrap at the operand's width (RFC-0037), and wrapping arithmetic is
//! arithmetic modulo `2^width`, where multiplication distributes over
//! addition exactly. Neither `*` nor `+` can raise — only `/` and `%` can —
//! so no trap moves and none is introduced.
//!
//! In floating point the two are *not* the same number: the accumulated
//! form carries the rounding error of every earlier step. The measurement
//! that settles it is in RFC-0056 — on mandelbrot's 200x100x200 grid the
//! accumulated `cx` differs from the recomputed one by up to 6.4e-15 and 34
//! of the 20000 pixels change escape count — so `exact_under_wrapping`
//! admits integers and refuses floats.
//!
//! # What it does not look at
//!
//! Nothing deeper than "a header parameter plus an invariant" and "a
//! multiplication by an invariant, optionally plus an invariant". There is
//! no scalar evolution here: a derived variable of a derived variable, a
//! step that is itself an induction variable, and a loop whose counter is
//! rewritten through memory are all outside the pattern and stay as they
//! are.

use acvus_ast::{BinOp, Literal, Span};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loops::{Invariant, Invariants, NaturalLoop, natural_loops_innermost_first};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, ValOrigin, ValueId};
use crate::optimize::ssa_pass::apply_subst;
use crate::ty::Ty;

pub fn run(cfg: &mut CfgBody) {
    let domtree = DomTree::build(cfg);
    for loop_ in natural_loops_innermost_first(cfg, &domtree) {
        let Some(frame) = Frame::of(cfg, &loop_) else {
            continue;
        };
        let invariants = Invariants::of(cfg);
        let ivs = induction_variables(cfg, &loop_, &frame, &invariants);
        let uses = use_blocks(cfg);
        let reductions = Scope {
            cfg,
            loop_: &loop_,
            frame: &frame,
            ivs: &ivs,
            invariants: &invariants,
            uses: &uses,
            domtree: &domtree,
        }
        .candidates();
        for reduction in reductions {
            apply(cfg, &frame, &reduction);
        }
    }
}

// -- The loop's three blocks ----------------------------------------

struct Frame {
    header: BlockIdx,
    header_label: Label,
    preheader: BlockIdx,
    latch: BlockIdx,
}

impl Frame {
    fn of(cfg: &CfgBody, loop_: &NaturalLoop) -> Option<Frame> {
        let [latch] = loop_.latches[..] else {
            return None;
        };
        let header_label = cfg.blocks[loop_.header.0].label;
        let preds = cfg.predecessors();
        let entering: Vec<BlockIdx> = preds
            .get(&loop_.header)
            .into_iter()
            .flatten()
            .copied()
            .filter(|b| !loop_.contains(*b))
            .collect();
        let [preheader] = entering[..] else {
            return None;
        };
        match cfg.blocks[preheader.0].terminator {
            Terminator::Jump { label, .. } if label == header_label => {}
            _ => return None,
        }
        sole_edge_args(&cfg.blocks[latch.0].terminator, header_label)?;
        Some(Frame {
            header: loop_.header,
            header_label,
            preheader,
            latch,
        })
    }
}

fn sole_edge_args(term: &Terminator, label: Label) -> Option<&Vec<ValueId>> {
    let mut edges: Vec<&Vec<ValueId>> = Vec::new();
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
        // A `For`'s exit edge carries its target's whole parameter list. Its
        // body edge carries only the parameters after the ones the
        // terminator fills, so it is not an edge whose arguments a reader
        // can line up with that block's parameters (RFC-0057).
        Terminator::For {
            exit, exit_args, ..
        } => {
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

fn sole_edge_args_mut(term: &mut Terminator, label: Label) -> &mut Vec<ValueId> {
    let mut edges: Vec<&mut Vec<ValueId>> = Vec::new();
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
            for (_, l, args) in arms.iter_mut() {
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
            exit, exit_args, ..
        } => {
            if *exit == label {
                edges.push(exit_args);
            }
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => {}
    }
    let count = edges.len();
    match <[&mut Vec<ValueId>; 1]>::try_from(edges) {
        Ok([args]) => args,
        Err(_) => panic!("{label:?} is {count} edges of this terminator, not one"),
    }
}

// -- Induction variables --------------------------------------------

/// A value the preheader's arithmetic reads: one that already reaches the
/// preheader, or a word to re-emit there because the hoist could not lift
/// it out of the body (`analysis::loops::Invariant`).
#[derive(Clone)]
enum Source {
    Ready(ValueId),
    Word(Literal),
}

impl Source {
    fn of(invariant: Invariant, value: ValueId) -> Source {
        match invariant {
            Invariant::Outside => Source::Ready(value),
            Invariant::Word(literal) => Source::Word(literal),
        }
    }
}

struct Iv {
    var: ValueId,
    init: ValueId,
    step: Source,
}

fn induction_variables(
    cfg: &CfgBody,
    loop_: &NaturalLoop,
    frame: &Frame,
    invariants: &Invariants,
) -> Vec<Iv> {
    let header = &cfg.blocks[frame.header.0];
    let init_args = sole_edge_args(
        &cfg.blocks[frame.preheader.0].terminator,
        frame.header_label,
    )
    .expect("the preheader ends in a Jump to the header");
    let next_args = sole_edge_args(&cfg.blocks[frame.latch.0].terminator, frame.header_label)
        .expect("the latch has one edge to the header");
    assert_eq!(
        init_args.len(),
        header.params.len(),
        "the preheader sends the header a different number of arguments than it has parameters"
    );
    assert_eq!(
        next_args.len(),
        header.params.len(),
        "the latch sends the header a different number of arguments than it has parameters"
    );

    let defining = defining_insts(cfg);
    header
        .params
        .iter()
        .enumerate()
        .filter_map(|(p, &var)| {
            let Some(InstKind::BinOp {
                op: BinOp::Add,
                left,
                right,
                ..
            }) = defining.get(&next_args[p])
            else {
                return None;
            };
            let step = match (*left == var, *right == var) {
                (true, false) => *right,
                (false, true) => *left,
                _ => return None,
            };
            invariants.at(loop_, step).map(|invariant| Iv {
                var,
                init: init_args[p],
                step: Source::of(invariant, step),
            })
        })
        .collect()
}

fn defining_insts(cfg: &CfgBody) -> FxHashMap<ValueId, InstKind> {
    cfg.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .flat_map(|inst| {
            inst_info::defs(&inst.kind)
                .into_iter()
                .map(move |d| (d, inst.kind.clone()))
        })
        .collect()
}

fn use_blocks(cfg: &CfgBody) -> FxHashMap<ValueId, Vec<BlockIdx>> {
    let mut uses: FxHashMap<ValueId, Vec<BlockIdx>> = FxHashMap::default();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for inst in &block.insts {
            for u in inst_info::uses(&inst.kind) {
                uses.entry(u).or_default().push(BlockIdx(bi));
            }
        }
        for u in terminator_uses(&block.terminator) {
            uses.entry(u).or_default().push(BlockIdx(bi));
        }
    }
    uses
}

fn terminator_uses(term: &Terminator) -> Vec<ValueId> {
    match term {
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => std::iter::once(*cond)
            .chain(then_args.iter().copied())
            .chain(else_args.iter().copied())
            .collect(),
        Terminator::For {
            source,
            body_args,
            exit_args,
            ..
        } => source
            .uses()
            .into_iter()
            .chain(body_args.iter().copied())
            .chain(exit_args.iter().copied())
            .collect(),
        Terminator::Switch { tag, arms, default } => std::iter::once(*tag)
            .chain(arms.iter().flat_map(|(_, _, args)| args.iter().copied()))
            .chain(default.iter().flat_map(|(_, args)| args.iter().copied()))
            .collect(),
        Terminator::Return { value, order } => std::iter::once(*value).chain(*order).collect(),
        Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

// -- Candidates ------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct Site {
    block: BlockIdx,
    inst: usize,
}

struct Sum {
    site: Site,
    dst: ValueId,
    offset: ValueId,
    invariant: Invariant,
}

#[derive(PartialEq, Eq, Hash)]
struct DerivedKey {
    iv: ValueId,
    factor: ValueId,
    offset: ValueId,
}

struct IvFactor<'a> {
    iv: &'a Iv,
    factor: ValueId,
}

struct Reduction {
    span: Span,
    dst: ValueId,
    ty: Ty,
    iv_init: ValueId,
    iv_step: Source,
    factor: Source,
    offset: Source,
    product: Site,
    sum: Site,
}

fn exact_under_wrapping(ty: &Ty) -> bool {
    match ty {
        Ty::Int(_) => true,
        Ty::Float => false,
        _ => false,
    }
}

struct Scope<'a> {
    cfg: &'a CfgBody,
    loop_: &'a NaturalLoop,
    frame: &'a Frame,
    ivs: &'a [Iv],
    invariants: &'a Invariants,
    uses: &'a FxHashMap<ValueId, Vec<BlockIdx>>,
    domtree: &'a DomTree,
}

impl Scope<'_> {
    fn runs_every_iteration(&self, block: BlockIdx) -> bool {
        self.domtree.dominates(block, self.frame.latch)
    }

    fn used_only_inside(&self, value: ValueId) -> bool {
        self.uses
            .get(&value)
            .into_iter()
            .flatten()
            .all(|b| self.loop_.contains(*b))
    }

    fn use_count(&self, value: ValueId) -> usize {
        self.uses.get(&value).map_or(0, Vec::len)
    }

    fn iv_factor(&self, left: ValueId, right: ValueId) -> Option<IvFactor<'_>> {
        self.ivs
            .iter()
            .find_map(|iv| match (left == iv.var, right == iv.var) {
                (true, false) => Some(IvFactor { iv, factor: right }),
                (false, true) => Some(IvFactor { iv, factor: left }),
                _ => None,
            })
    }

    fn candidates(&self) -> Vec<Reduction> {
        let mut found: Vec<Reduction> = Vec::new();
        let mut taken: FxHashSet<DerivedKey> = FxHashSet::default();

        for block in self.loop_.blocks() {
            if !self.runs_every_iteration(block) {
                continue;
            }
            for (inst, item) in self.cfg.blocks[block.0].insts.iter().enumerate() {
                let InstKind::BinOp {
                    dst,
                    op: BinOp::Mul,
                    left,
                    right,
                } = &item.kind
                else {
                    continue;
                };
                let Some(IvFactor { iv, factor }) = self.iv_factor(*left, *right) else {
                    continue;
                };
                let Some(factor_invariant) = self.invariants.at(self.loop_, factor) else {
                    continue;
                };
                let ty = &self.cfg.val_types[dst];
                if !exact_under_wrapping(ty) {
                    continue;
                }

                let Some(sum) = self.sole_invariant_sum(*dst) else {
                    continue;
                };
                if !self.used_only_inside(sum.dst) {
                    continue;
                }
                let key = DerivedKey {
                    iv: iv.var,
                    factor,
                    offset: sum.offset,
                };
                if !taken.insert(key) {
                    continue;
                }
                found.push(Reduction {
                    span: item.span,
                    dst: sum.dst,
                    ty: ty.clone(),
                    iv_init: iv.init,
                    iv_step: iv.step.clone(),
                    factor: Source::of(factor_invariant, factor),
                    offset: Source::of(sum.invariant, sum.offset),
                    product: Site { block, inst },
                    sum: sum.site,
                });
            }
        }
        found
    }

    fn sole_invariant_sum(&self, product: ValueId) -> Option<Sum> {
        if self.use_count(product) != 1 {
            return None;
        }
        for block in self.loop_.blocks() {
            if !self.runs_every_iteration(block) {
                continue;
            }
            for (inst, item) in self.cfg.blocks[block.0].insts.iter().enumerate() {
                let InstKind::BinOp {
                    dst,
                    op: BinOp::Add,
                    left,
                    right,
                } = &item.kind
                else {
                    continue;
                };
                let offset = match (*left == product, *right == product) {
                    (true, false) => *right,
                    (false, true) => *left,
                    _ => continue,
                };
                return self.invariants.at(self.loop_, offset).map(|invariant| Sum {
                    site: Site { block, inst },
                    dst: *dst,
                    offset,
                    invariant,
                });
            }
        }
        None
    }
}

// -- Applying one reduction ------------------------------------------

/// The fresh values one reduction adds, and the instructions that give a
/// re-emitted word a register above the header.
struct Emit<'a> {
    cfg: &'a mut CfgBody,
    ty: Ty,
    span: Span,
    preamble: Vec<Inst>,
}

impl Emit<'_> {
    fn fresh(&mut self) -> ValueId {
        let value = self.cfg.val_factory.next();
        let previous = self.cfg.val_types.insert(value, self.ty.clone());
        assert!(
            previous.is_none(),
            "{value:?} is fresh from the factory and already carried a type"
        );
        self.cfg.debug.set(value, ValOrigin::Expr);
        value
    }

    fn read(&mut self, source: &Source) -> ValueId {
        match source {
            Source::Ready(value) => *value,
            Source::Word(literal) => {
                let dst = self.fresh();
                let value = literal.clone();
                let span = self.span;
                self.preamble.push(Inst {
                    span,
                    kind: InstKind::Const { dst, value },
                });
                dst
            }
        }
    }

    fn arith(&mut self, op: BinOp, left: ValueId, right: ValueId) -> (ValueId, Inst) {
        let dst = self.fresh();
        (
            dst,
            Inst {
                span: self.span,
                kind: InstKind::BinOp {
                    dst,
                    op,
                    left,
                    right,
                },
            },
        )
    }
}

fn apply(cfg: &mut CfgBody, frame: &Frame, reduction: &Reduction) {
    let mut emit = Emit {
        cfg,
        ty: reduction.ty.clone(),
        span: reduction.span,
        preamble: Vec::new(),
    };
    let derived = emit.fresh();
    let iv_step = emit.read(&reduction.iv_step);
    let factor = emit.read(&reduction.factor);
    let offset = emit.read(&reduction.offset);

    let (step, step_inst) = emit.arith(BinOp::Mul, iv_step, factor);
    let (start_product, start_product_inst) = emit.arith(BinOp::Mul, reduction.iv_init, factor);
    let (start, start_inst) = emit.arith(BinOp::Add, start_product, offset);
    let mut preheader_insts = std::mem::take(&mut emit.preamble);
    preheader_insts.push(step_inst);
    preheader_insts.push(start_product_inst);
    preheader_insts.push(start_inst);
    let (advanced, advanced_inst) = emit.arith(BinOp::Add, derived, step);

    let preheader = &mut cfg.blocks[frame.preheader.0];
    preheader.insts.extend(preheader_insts);
    sole_edge_args_mut(&mut preheader.terminator, frame.header_label).push(start);

    cfg.blocks[frame.header.0].params.push(derived);

    let latch = &mut cfg.blocks[frame.latch.0];
    latch.insts.push(advanced_inst);
    sole_edge_args_mut(&mut latch.terminator, frame.header_label).push(advanced);

    for site in [reduction.product, reduction.sum] {
        cfg.blocks[site.block.0].insts[site.inst].kind = InstKind::Nop;
    }

    let subst = FxHashMap::from_iter([(reduction.dst, derived)]);
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, &subst);
        }
        subst_terminator(&mut block.terminator, &subst);
    }
}

fn subst_terminator(term: &mut Terminator, subst: &FxHashMap<ValueId, ValueId>) {
    let mut one = |v: &mut ValueId| {
        if let Some(&to) = subst.get(v) {
            *v = to;
        }
    };
    match term {
        Terminator::Jump { args, .. } => args.iter_mut().for_each(&mut one),
        Terminator::JumpIf {
            cond,
            then_args,
            else_args,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_args,
            else_args,
            ..
        } => {
            one(cond);
            then_args.iter_mut().for_each(&mut one);
            else_args.iter_mut().for_each(&mut one);
        }
        Terminator::For {
            source,
            body_args,
            exit_args,
            ..
        } => {
            source.for_each_use(&mut one);
            body_args.iter_mut().for_each(&mut one);
            exit_args.iter_mut().for_each(&mut one);
        }
        Terminator::Switch { tag, arms, default } => {
            one(tag);
            for (_, _, args) in arms.iter_mut() {
                args.iter_mut().for_each(&mut one);
            }
            if let Some((_, args)) = default {
                args.iter_mut().for_each(&mut one);
            }
        }
        Terminator::Return { value, order } => {
            one(value);
            if let Some(order) = order {
                one(order);
            }
        }
        Terminator::Diverge | Terminator::Fallthrough => {}
    }
}
