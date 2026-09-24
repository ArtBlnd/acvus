//! Loop strength reduction: a loop multiplies once.
//!
//! An induction variable is a header block parameter `i` whose latch
//! argument is `i + c` with `c` loop-invariant, and `i * k + x` with `k` and
//! `x` loop-invariant is then affine too: it starts at `i0 * k + x` and
//! advances by `c * k`. Both are `analysis::affine`'s, and the pass reads
//! the rule that made each value affine rather than matching the
//! instructions itself. The pass gives `i * k + x` its own header parameter,
//! computes the start and the step in the preheader, adds the step in the
//! latch, and replaces the body's multiplication and sum with the
//! parameter. `i` itself stays: the loop condition reads it.
//!
//! # Strong loops only
//!
//! The pass transforms a loop that `analysis::carried` classifies strong and
//! leaves a weak one exactly as it is (RFC-0056). A weak loop's iterations
//! may run in any order, and its counters are IV canonicalization's to
//! normalize (RFC-0066), which keeps each iteration computing `i * k + x`
//! from its own `i`; a derived counter would make it wait for the previous
//! iteration's value. A strong loop runs in order anyway, so the
//! accumulated form costs it no independence.
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
//! of the 20000 pixels change escape count — so
//! `analysis::affine::exact_under_wrapping` admits integers and refuses
//! floats.
//!
//! # What it does not look at
//!
//! Nothing deeper than "a header parameter plus an invariant" and "a
//! multiplication by an invariant, optionally plus an invariant". There is
//! no scalar evolution here: a derived variable of a derived variable, a
//! step that is itself an induction variable, and a loop whose counter is
//! rewritten through memory are all outside the pattern and stay as they
//! are. A `for`'s counter is affine from its terminator, and a product of
//! it is not reduced: the pattern RFC-0056 states multiplies a carried
//! header parameter.

use acvus_ast::Span;

use crate::ir::BinOp;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::{Affine, AffineValues, Derivation, Operand};
use crate::analysis::carried::{CarriedState, Strength};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariant, Invariants, LoopNest, NaturalLoop, edge_args};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Inst, InstKind, Label, ValOrigin, ValueId};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::apply_subst;
use crate::ty::Ty;

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let domtree = DomTree::build(cfg);
    let nest = LoopNest::of(cfg, &domtree, &Invariants::of(cfg));
    let loans = Loans::build(cfg);
    for (_, loop_) in nest.iter() {
        let Some(frame) = Frame::of(cfg, &loop_.natural) else {
            continue;
        };
        let invariants = Invariants::of(cfg);
        let affine = AffineValues::of(cfg, loop_, &invariants);
        if CarriedState::of(cfg, loop_, &affine, &loans, laws).strength() == Strength::Weak {
            continue;
        }
        let uses = use_blocks(cfg);
        let reductions = Scope {
            cfg,
            loop_: &loop_.natural,
            frame: &frame,
            affine: &affine,
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
        let [preheader] = loop_.entering[..] else {
            return None;
        };
        let header_label = cfg.blocks[loop_.header.0].label;
        match cfg.blocks[preheader.0].terminator {
            Terminator::Jump { label, .. } if label == header_label => {}
            _ => return None,
        }
        edge_args(&cfg.blocks[latch.0].terminator, header_label)?;
        Some(Frame {
            header: loop_.header,
            header_label,
            preheader,
            latch,
        })
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

// -- Uses ------------------------------------------------------------

fn use_blocks(cfg: &CfgBody) -> FxHashMap<ValueId, Vec<BlockIdx>> {
    let mut uses: FxHashMap<ValueId, Vec<BlockIdx>> = FxHashMap::default();
    for (bi, block) in cfg.blocks.iter().enumerate() {
        for inst in &block.insts {
            for u in inst_info::uses(&inst.kind) {
                uses.entry(u).or_default().push(BlockIdx(bi));
            }
        }
        for u in inst_info::terminator_uses(&block.terminator) {
            uses.entry(u).or_default().push(BlockIdx(bi));
        }
    }
    uses
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
    offset: Operand,
}

#[derive(PartialEq, Eq, Hash)]
struct DerivedKey {
    iv: ValueId,
    factor: ValueId,
    offset: ValueId,
}

struct Reduction {
    span: Span,
    dst: ValueId,
    ty: Ty,
    iv_init: ValueId,
    iv_step: Invariant,
    factor: Invariant,
    offset: Invariant,
    product: Site,
    sum: Site,
}

struct Scope<'a> {
    cfg: &'a CfgBody,
    loop_: &'a NaturalLoop,
    frame: &'a Frame,
    affine: &'a AffineValues,
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

    fn derivation(&self, value: ValueId) -> Option<&Derivation> {
        self.affine
            .get(value)
            .map(|Affine { derivation, .. }| derivation)
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
                    ..
                } = &item.kind
                else {
                    continue;
                };
                let Some(Derivation::Scaled { of: iv, factor }) = self.derivation(*dst) else {
                    continue;
                };
                let Some(Derivation::Carried { init, step }) = self.derivation(*iv) else {
                    continue;
                };
                let Some(sum) = self.sole_invariant_sum(*dst) else {
                    continue;
                };
                if !self.used_only_inside(sum.dst) {
                    continue;
                }
                let key = DerivedKey {
                    iv: *iv,
                    factor: factor.value,
                    offset: sum.offset.value,
                };
                if !taken.insert(key) {
                    continue;
                }
                found.push(Reduction {
                    span: item.span,
                    dst: sum.dst,
                    ty: self.cfg.val_types[dst].clone(),
                    iv_init: *init,
                    iv_step: step.invariant.clone(),
                    factor: factor.invariant.clone(),
                    offset: sum.offset.invariant,
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
                    ..
                } = &item.kind
                else {
                    continue;
                };
                let Some(Derivation::Offset { of, offset }) = self.derivation(*dst) else {
                    continue;
                };
                if *of != product {
                    continue;
                }
                return Some(Sum {
                    site: Site { block, inst },
                    dst: *dst,
                    offset: offset.clone(),
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

    fn read(&mut self, invariant: &Invariant) -> ValueId {
        match invariant {
            Invariant::Outside(value) => *value,
            Invariant::Word(literal) => {
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
        Terminator::Return { value, order, .. } => {
            one(value);
            if let Some(order) = order {
                one(order);
            }
        }
        Terminator::Diverge | Terminator::Fallthrough => {}
    }
}
