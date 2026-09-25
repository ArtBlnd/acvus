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
//! # Only inside one `InOrder` stage
//!
//! The pass runs after the stages are written (RFC-0089 rule 6), and it
//! reduces a counter expression only when every reader of it sits in one
//! stage whose every cycle `analysis::loop_deps` judges `InOrder` with no
//! law (RFC-0056, RFC-0066 rule 7). The derived counter is a carried value:
//! its advance is placed at the end of that stage, where its own cycle then
//! lies. Such a stage runs in order already, so the reduction costs it
//! nothing. Anywhere else each iteration computes `i * k + x` from its own
//! `i`, and a derived counter would make it wait for the previous
//! iteration's value. A stage with a law combines a chunk before it joins
//! the partial (RFC-0092), which a counter carried from the chunk
//! before would serialize; that stage is declined. A `while` has no stages
//! and is declined.
//!
//! `i` is a `for`'s counter, which the terminator advances by one, or a
//! carried header parameter IV canonicalization left (RFC-0066 rule 7). For
//! the counter the step is `k` itself and the start is `at * k + x` for a
//! range and `x` for a slice or an array, whose counter starts at zero.
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
//! It runs after the stage pass (`graph/optimize.rs`), which follows the
//! hoist that puts `k` and `x` above the header. It needs the preheader a
//! block of its own that jumps to the header, and one latch.
//!
//! # Only a pass's wrapping arithmetic
//!
//! `(i0 + n*c) * k` and `i0*k + n*(c*k)` are the same integer modulo
//! `2^width`, where multiplication distributes over addition exactly. The
//! start, the step and the latch's advance are operations this pass writes,
//! and they wrap: the start is computed before the first iteration and the
//! advance after the last one, on values the program never computed, so a
//! trapping one could end a run the program defines.
//!
//! The program's `i * k + x` is reduced only where the counter's bounds, `k`
//! and `x` are words that put both operations inside the width on every
//! iteration, a decision and not an omission. Its `*` and `+` trap where
//! they overflow, and a rewrite keeps those traps on exactly the runs and
//! iterations where they trapped (RFC-0037 rule 3). A check of each at its
//! place costs what the reduction sheds, so the body's count does not fall
//! and the count rule above declines it; where neither can trap there is
//! nothing to keep. A pass's own wrapping `i * k + x` is reduced as well.
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
//! are.

use acvus_ast::{Literal, Span};

use crate::ir::{BinOp, Overflow};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::{Affine, AffineValues, Derivation, Operand};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loops::{Invariant, Invariants, LoopNest, NaturalLoop, edge_args};
use crate::analysis::loop_deps::{LoopDeps, Order, StageMembership};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{ForSource, Inst, InstKind, Label, ValOrigin, ValueId};
use crate::laws::LawTable;
use crate::optimize::ssa_pass::apply_subst;
use crate::ty::{LenTerm, Ty};

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    let domtree = DomTree::build(cfg);
    let nest = LoopNest::of(cfg, &domtree, &Invariants::of(cfg));
    for (_, loop_) in nest.iter() {
        let Some(frame) = Frame::of(cfg, &loop_.natural) else {
            continue;
        };
        let Some(chain) = Chain::of(cfg, &frame, laws) else {
            continue;
        };
        let invariants = Invariants::of(cfg);
        let affine = AffineValues::of(cfg, loop_, &invariants, laws);
        let uses = use_blocks(cfg);
        let reductions = Scope {
            cfg,
            loop_: &loop_.natural,
            frame: &frame,
            chain: &chain,
            affine: &affine,
            uses: &uses,
            domtree: &domtree,
        }
        .candidates();
        for reduction in reductions {
            apply(cfg, &invariants, &frame, &chain, &reduction);
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

// -- The stages ------------------------------------------------------

struct Chain {
    source: ForSource,
    membership: StageMembership,
    ends: Vec<BlockIdx>,
    reducible: Vec<bool>,
}

impl Chain {
    fn of(cfg: &CfgBody, frame: &Frame, laws: &LawTable) -> Option<Chain> {
        let Terminator::For { source, .. } = &cfg.blocks[frame.header.0].terminator else {
            return None;
        };
        let deps = LoopDeps::of(cfg, laws, frame.header).ok()?;
        let ends: Vec<BlockIdx> = deps
            .membership
            .stages()
            .iter()
            .map(|stage| stage.sole_end())
            .collect::<Option<_>>()?;
        if ends.last() != Some(&frame.latch) {
            return None;
        }
        let judged = deps.judge(cfg, laws);
        let reducible = (0..ends.len())
            .map(|stage| {
                let mut held = deps
                    .cycles
                    .iter()
                    .zip(&judged)
                    .filter(|(cycle, _)| cycle.stage() == Some(stage))
                    .peekable();
                held.peek().is_some()
                    && held.all(|(_, judged)| {
                        judged.order == Order::InOrder && judged.law.is_none()
                    })
            })
            .collect();
        Some(Chain {
            source: *source,
            membership: deps.membership,
            ends,
            reducible,
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
    kind: Overflow,
    offset: Operand,
}

#[derive(PartialEq, Eq, Hash)]
struct DerivedKey {
    iv: ValueId,
    factor: ValueId,
    offset: ValueId,
}

/// Where `i` starts and how it advances.
enum Counted {
    /// A carried header parameter entered as `init` and advanced by `step`.
    Carried { init: ValueId, step: Invariant },
    /// A range's counter, from `at` by one.
    Range { at: ValueId },
    /// A slice's or an array's index, from zero by one.
    Index,
}

struct Reduction {
    span: Span,
    dst: ValueId,
    ty: Ty,
    counted: Counted,
    factor: Invariant,
    offset: Invariant,
    product: Site,
    sum: Site,
    /// The stage that reads the reduced value, whose every cycle is
    /// `InOrder` with no law; the derived counter's cycle lies there too.
    join: usize,
}

struct Scope<'a> {
    cfg: &'a CfgBody,
    loop_: &'a NaturalLoop,
    frame: &'a Frame,
    chain: &'a Chain,
    affine: &'a AffineValues,
    uses: &'a FxHashMap<ValueId, Vec<BlockIdx>>,
    domtree: &'a DomTree,
}

impl Scope<'_> {
    fn runs_every_iteration(&self, block: BlockIdx) -> bool {
        self.domtree.dominates(block, self.frame.latch)
    }

    /// The one reducible stage every reader of `value` sits in.
    fn read_in_one_join(&self, value: ValueId) -> Option<usize> {
        let mut stages = self
            .uses
            .get(&value)
            .into_iter()
            .flatten()
            .map(|block| self.chain.membership.stage_of(*block));
        let join = stages.next()??;
        let one = stages.all(|stage| stage == Some(join));
        (one && self.chain.reducible[join]).then_some(join)
    }

    fn counted(&self, iv: ValueId) -> Option<Counted> {
        match self.derivation(iv)? {
            // A standing step is not read above the header, where the
            // derived counter starts.
            Derivation::Carried { init, step } => Some(Counted::Carried {
                init: *init,
                step: step.invariance.above()?.clone(),
            }),
            Derivation::Counter => match self.chain.source {
                ForSource::Range { at, .. } => Some(Counted::Range { at }),
                ForSource::Slice(_) | ForSource::SliceMut(_) | ForSource::Array(_) => {
                    Some(Counted::Index)
                }
            },
            Derivation::Scaled { .. }
            | Derivation::Offset { .. }
            | Derivation::Lowered { .. }
            | Derivation::Reflected { .. }
            | Derivation::Length { .. } => None,
        }
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
                    op: BinOp::Mul(product_kind),
                    ..
                } = &item.kind
                else {
                    continue;
                };
                let Some(Derivation::Scaled { of: iv, factor }) = self.derivation(*dst) else {
                    continue;
                };
                let Some(counted) = self.counted(*iv) else {
                    continue;
                };
                let Some(sum) = self.sole_invariant_sum(*dst) else {
                    continue;
                };
                let Some(join) = self.read_in_one_join(sum.dst) else {
                    continue;
                };
                // The derived counter starts above the header, where no
                // standing operand is read (RFC-0066 rule 3).
                let (Some(factor_above), Some(offset_above)) =
                    (factor.invariance.above(), sum.offset.invariance.above())
                else {
                    continue;
                };
                let traps = *product_kind == Overflow::Trap || sum.kind == Overflow::Trap;
                if traps
                    && !self.fits_on_every_iteration(
                        &counted,
                        factor_above,
                        offset_above,
                        *dst,
                    )
                {
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
                    counted,
                    factor: factor_above.clone(),
                    offset: offset_above.clone(),
                    product: Site { block, inst },
                    sum: sum.site,
                    join,
                });
            }
        }
        found
    }

    /// Whether the counter's every value `i` is a word range and `i * k` and
    /// `i * k + x` fit the width over all of it, so neither of the program's
    /// operations can trap and the reduction keeps their traps, which are
    /// none. Both are monotone in `i`, so the two ends decide.
    fn fits_on_every_iteration(
        &self,
        counted: &Counted,
        factor: &Invariant,
        offset: &Invariant,
        product: ValueId,
    ) -> bool {
        let Ty::Int(width) = self.cfg.val_types[&product] else {
            return false;
        };
        let invariants = Invariants::of(self.cfg);
        let int = |invariant: &Invariant| {
            let literal = match invariant {
                Invariant::Outside(value) => invariants.word(*value)?.desugared(),
                Invariant::Word(literal) => literal.desugared(),
            };
            match literal {
                Literal::Int(n) => Some(width.read(n as u64)),
                _ => None,
            }
        };
        let ends = match (counted, self.chain.source) {
            (Counted::Range { at }, ForSource::Range { hi, .. }) => int(&Invariant::Outside(*at))
                .zip(int(&Invariant::Outside(hi)))
                .map(|(at, hi)| (at, hi - 1)),
            (Counted::Index, ForSource::Array(array)) => match &self.cfg.val_types[&array] {
                Ty::Array(_, LenTerm::Known(len)) => {
                    i128::try_from(*len).ok().map(|len| (0, len - 1))
                }
                _ => None,
            },
            _ => None,
        };
        let (Some((first, last)), Some(k), Some(x)) = (ends, int(factor), int(offset)) else {
            return false;
        };
        last < first
            || [first, last]
                .into_iter()
                .all(|i| width.holds(i * k) && width.holds(i * k + x))
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
                    op: BinOp::Add(kind),
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
                    kind: *kind,
                    offset: offset.clone(),
                });
            }
        }
        None
    }
}

// -- Applying one reduction ------------------------------------------

#[derive(Clone)]
enum SetupOperand {
    Value(ValueId),
    Int {
        value: i128,
        literal: Literal,
        held: Option<ValueId>,
    },
}

impl SetupOperand {
    fn is(&self, int: i128) -> bool {
        matches!(self, SetupOperand::Int { value, .. } if *value == int)
    }
}

/// No value numbering runs after this pass (RFC-0056), so the start and the
/// step are written folded where an operand is the integer 0 or 1.
struct Emit<'a> {
    cfg: &'a mut CfgBody,
    invariants: &'a Invariants,
    ty: Ty,
    span: Span,
    words: Vec<Inst>,
    setup: Vec<Inst>,
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

    fn held(&self, value: ValueId) -> SetupOperand {
        match self
            .invariants
            .word(value)
            .map(|literal| (literal.desugared(), literal))
        {
            Some((Literal::Int(int), literal)) => SetupOperand::Int {
                value: int,
                literal: literal.clone(),
                held: Some(value),
            },
            _ => SetupOperand::Value(value),
        }
    }

    fn term(&self, invariant: &Invariant) -> SetupOperand {
        match invariant {
            Invariant::Outside(value) => self.held(*value),
            Invariant::Word(literal) => match literal.desugared() {
                Literal::Int(int) => SetupOperand::Int {
                    value: int,
                    literal: literal.clone(),
                    held: None,
                },
                _ => panic!(
                    "a reduced counter's operand is an integer word (`exact_under_wrapping`), \
                     not {literal:?}"
                ),
            },
        }
    }

    fn value(&mut self, term: SetupOperand) -> ValueId {
        match term {
            SetupOperand::Value(value)
            | SetupOperand::Int {
                held: Some(value), ..
            } => value,
            SetupOperand::Int {
                literal,
                held: None,
                ..
            } => {
                let dst = self.fresh();
                let span = self.span;
                self.words.push(Inst {
                    span,
                    kind: InstKind::Const {
                        dst,
                        value: literal,
                    },
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

    fn setup(&mut self, op: BinOp, left: SetupOperand, right: SetupOperand) -> SetupOperand {
        let left = self.value(left);
        let right = self.value(right);
        let (dst, inst) = self.arith(op, left, right);
        self.setup.push(inst);
        SetupOperand::Value(dst)
    }

    fn mul(&mut self, left: SetupOperand, right: SetupOperand) -> SetupOperand {
        match (left, right) {
            (zero, _) | (_, zero) if zero.is(0) => zero,
            (one, other) | (other, one) if one.is(1) => other,
            (left, right) => self.setup(BinOp::Mul(Overflow::Wrap), left, right),
        }
    }

    fn add(&mut self, left: SetupOperand, right: SetupOperand) -> SetupOperand {
        match (left, right) {
            (zero, other) | (other, zero) if zero.is(0) => other,
            (left, right) => self.setup(BinOp::Add(Overflow::Wrap), left, right),
        }
    }
}

fn apply(
    cfg: &mut CfgBody,
    invariants: &Invariants,
    frame: &Frame,
    chain: &Chain,
    reduction: &Reduction,
) {
    let mut emit = Emit {
        cfg,
        invariants,
        ty: reduction.ty.clone(),
        span: reduction.span,
        words: Vec::new(),
        setup: Vec::new(),
    };
    let derived = emit.fresh();
    let factor = emit.term(&reduction.factor);
    let offset = emit.term(&reduction.offset);
    let (step, start) = match &reduction.counted {
        Counted::Carried { init, step } => {
            let iv_step = emit.term(step);
            let step = emit.mul(iv_step, factor.clone());
            let init = emit.held(*init);
            let start_product = emit.mul(init, factor);
            (step, emit.add(start_product, offset))
        }
        Counted::Range { at } => {
            let at = emit.held(*at);
            let start_product = emit.mul(at, factor.clone());
            (factor, emit.add(start_product, offset))
        }
        Counted::Index => (factor, offset),
    };
    let step = emit.value(step);
    let start = emit.value(start);
    let mut preheader_insts = std::mem::take(&mut emit.words);
    preheader_insts.append(&mut emit.setup);
    let (advanced, advanced_inst) = emit.arith(BinOp::Add(Overflow::Wrap), derived, step);

    let preheader = &mut cfg.blocks[frame.preheader.0];
    preheader.insts.extend(preheader_insts);
    sole_edge_args_mut(&mut preheader.terminator, frame.header_label).push(start);

    cfg.blocks[frame.header.0].params.push(derived);
    let ends_join = chain.ends[reduction.join];
    cfg.blocks[ends_join.0].insts.push(advanced_inst);
    let latch = &mut cfg.blocks[frame.latch.0];
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
            source, exit_args, ..
        } => {
            source.for_each_use(&mut one);
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
