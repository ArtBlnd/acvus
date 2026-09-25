//! A `while` that is a counted traversal becomes a `for` over a range
//! (RFC-0081).
//!
//! In `let i = b; while i < n { …; i = i + 1; }` the header's `jump_if`
//! becomes `For { source: Range { at: b, hi: n }, … }` and the body block
//! gains the counter that terminator fills. The pass does not touch `i` and
//! does not compute its value after the loop: the exit edge carries it as
//! before, and normalizing it is IV canonicalization's (RFC-0066 rule 7).
//!
//! The rewrite is one half of a contract whose other half is the
//! interpreter. `acvus_interpreter::ops::control::Range` runs the body while
//! `T::read(at) < T::read(hi)` and then advances with `wrapping_add` at `T`;
//! the `while` compares with `ops::arith::word::lt` at the same `T` and its
//! `i + 1` is the program's trapping `+` at `T` (RFC-0037 rule 3). Both
//! advance only from a counter that compared below `n`, so neither leaves
//! the width: the range never wraps and the `+` never traps. Both start at
//! `b` on every entry, so both loops see `b + k` on their `k`-th header
//! visit. If either operation changes, the corpus under
//! `acvus-interpreter-test/tests/soundness/while-to-for` is what disagrees
//! across optimization levels.
//!
//! The machine reads a range's bounds on the edge that enters the loop,
//! before the header runs, and a `for` header holds no instruction. A bound
//! the header computes therefore moves to the end of the entering block.
//!
//! A [`Step`] is deterministic: a word operation, or a call of an extern
//! declared `pure` whose reference arguments are shared and defined
//! outside the loop, which the borrow check keeps unwritten while the
//! header reads them (RFC-0064). Over operands defined outside the loop it
//! gives the same value, and raises the same trap, on every header visit.
//! The header runs on every entry before any body block, so one evaluation
//! on the entering edge is that first visit moved ahead of the header's
//! other instructions, onto exactly the paths it ran on, and every later
//! visit repeated it. A trapping `+`,
//! `-` or `*`, a `/`, a `%` and a call can trap, so a bound holding one is
//! promoted only when no header instruction before its last such step,
//! other than a step of the bound, can trap: then the entry raises exactly
//! the trap the first visit raised, and in the header's order. An effect
//! before it does not decline: a trap is not ordered with effects
//! (RFC-0048 rule 8).

//!
//! RFC-0094 adds four spellings of the same traversal. `i <= n` is the
//! range `b..n + 1` where the interval domain puts `n` below the width's
//! maximum, the `+ 1` the pass's own wrapping one. `i > n` whose back edges
//! send `i − 1` is `Range { at: n, hi: b }`. A word-constant step `s ≠ 1`
//! compared in the direction it moves is a range over a trip count written
//! above the header: `⌈d / |s|⌉` of the distance `d` read unsigned at the
//! width, as `(d − 1) / |s| + 1` where the counter has not passed the bound
//! and zero where it has, so nothing there wraps or traps. `i + c₀ < n` is
//! `b + c₀ .. n`, the start the header's first `i + c₀` moved to the entry;
//! the `i + c₀` itself moves to the head of the body, where it can no
//! longer trap, since its value there is at most `n`. A `Ref` the header
//! makes of a slot no instruction of the loop writes, lends `&mut`, or lends
//! to a call that reaches it is a deterministic step. In every form `i`
//! stays a header parameter advanced by its own step, which keeps its trap,
//! and IV canonicalization reads it from the counter (RFC-0066 rule 7).
//!
//! RFC-0094 rule 7 admits an edge out of the body to a block from which
//! every path reaches the loop's exit with no `return` and no `!`: the
//! `k`-th visit holds `b + k` in both loops, so an edge the body takes on
//! that visit leaves both with the same values. The `For` keeps the edge,
//! and its count is then a bound. Rule 6, in [`leave`], moves a back edge whose constant decides the
//! header's test to the exit, which makes such an edge.

use acvus_ast::{Literal, SuffixedInt};
use acvus_utils::Interner;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::analysis::affine::{AffineValues, Derivation, Operand as AffineOperand};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::interval::constant_bounds_on_entry;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariant, Invariants, Loop, LoopKind, LoopNest};
use crate::analysis::targets::{effect, slots_lent_mutably, touched_slots};
use crate::cfg::{Block, BlockIdx, CfgBody, ENTRY_LABEL, Terminator};
use crate::ir::{
    BinOp, Callee, Checked, ExitTrip, ForSource, Inst, InstKind, Label, Overflow, RefTarget,
    Stages, ValOrigin, ValueId,
};
use crate::laws::{LawTable, Reaches};
use crate::optimize::forward::edges_mut;
use crate::optimize::ssa_pass::{apply_subst, apply_subst_terminator, map_value_defs};
use crate::ty::{CastTy, IntTy, Mutability, Ty};

mod leave;

pub fn run(interner: &Interner, cfg: &mut CfgBody, laws: &LawTable) {
    leave::run(cfg, laws);
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let preds = cfg.predecessors();
    let literals = const_literals(cfg);
    let loans = Loans::build(cfg);
    let mut counted: Vec<Counted> = Vec::new();
    let mut pulls: Vec<Pull> = Vec::new();
    for (_, loop_) in nest.iter() {
        let recognizer = Recognizer {
            cfg,
            loop_,
            invariants: &invariants,
            laws,
            loans: &loans,
            preds: &preds,
            literals: &literals,
        };
        if let Some(found) = recognizer.counted() {
            counted.push(found);
        } else if let Some(found) = recognizer.pull(interner) {
            pulls.push(found);
        }
    }
    let mut labels = LabelFactory::of(cfg);
    for loop_ in counted {
        loop_.apply(cfg, &mut labels);
    }
    for loop_ in pulls {
        loop_.apply(cfg);
    }
}

/// Whether rules 1 to 5 convert the loop headed at `header`.
fn converts(cfg: &CfgBody, laws: &LawTable, header: Label) -> bool {
    let domtree = DomTree::build(cfg);
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &domtree, &invariants);
    let preds = cfg.predecessors();
    let literals = const_literals(cfg);
    let loans = Loans::build(cfg);
    nest.iter()
        .filter(|(_, loop_)| cfg.blocks[loop_.natural.header.0].label == header)
        .any(|(_, loop_)| {
            Recognizer {
                cfg,
                loop_,
                invariants: &invariants,
                laws,
                loans: &loans,
                preds: &preds,
                literals: &literals,
            }
            .counted()
            .is_some()
        })
}

/// A pull loop (RFC-0089 rule 1): the header lends one storage `&mut` to
/// an extern `next` (D10), tests the `Option` it returns, and branches on
/// that test to the body or out of the loop, which leaves nowhere else.
/// No instruction outside the header touches the storage or reads through
/// a loan of it, so no cycle through the pull reaches the body.
struct Pull {
    header: Label,
    cond: ValueId,
    body: Label,
    body_args: Vec<ValueId>,
    exit: Label,
    exit_args: Vec<ValueId>,
}

enum Bound {
    Outside(ValueId),
    Computed { value: ValueId },
}

impl Bound {
    fn value(&self) -> ValueId {
        match self {
            Self::Outside(value) | Self::Computed { value } => *value,
        }
    }
}

struct Step {
    header_position: usize,
    dst: ValueId,
    kind: StepKind,
    emit: Emit,
}

/// How a step reaches the entering block: the header's instruction moved
/// there, or, for RFC-0094 rule 4's start, a copy of the header's `i + c₀`
/// reading the entry value `b` for `i`, the original moving to the body.
#[derive(Clone, Copy)]
enum Emit {
    Move,
    Start,
}

enum StepKind {
    Word,
    Borrow,
    Arith(Arith),
    Call,
}

impl StepKind {
    /// Whether evaluating the step may end the run. A `pure` extern
    /// declares that the call may be reissued, not that it returns:
    /// `unwrap` is `pure` and panics.
    fn traps(&self) -> bool {
        match self {
            Self::Word | Self::Borrow => false,
            Self::Arith(op) => op.traps(),
            Self::Call => true,
        }
    }
}

#[derive(Clone, Copy)]
enum Operand {
    Outside(ValueId),
    Step(ValueId),
}

/// The integer operations a bound may hold. Each is a function of its two
/// words at the width (RFC-0037): a trapping `+`, `-` or `*` traps where its
/// exact result leaves the width, a wrapping one wraps, and `/` and `%`
/// panic on a zero divisor and on the signed minimum divided by `-1`.
#[derive(Clone, Copy)]
enum Arith {
    Add(Overflow),
    Sub(Overflow),
    Mul(Overflow),
    Div,
    Rem,
}

impl Arith {
    fn of(op: BinOp) -> Option<Self> {
        match op {
            BinOp::Add(overflow) => Some(Self::Add(overflow)),
            BinOp::Sub(overflow) => Some(Self::Sub(overflow)),
            BinOp::Mul(overflow) => Some(Self::Mul(overflow)),
            BinOp::Div => Some(Self::Div),
            BinOp::Mod => Some(Self::Rem),
            _ => None,
        }
    }

    fn op(self) -> BinOp {
        match self {
            Self::Add(overflow) => BinOp::Add(overflow),
            Self::Sub(overflow) => BinOp::Sub(overflow),
            Self::Mul(overflow) => BinOp::Mul(overflow),
            Self::Div => BinOp::Div,
            Self::Rem => BinOp::Mod,
        }
    }

    fn traps(self) -> bool {
        self.op().can_trap_on_integers()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Compare {
    Below,
    AtMost,
    Above,
}

#[derive(Clone, Copy)]
struct Condition {
    compare: Compare,
    compared: ValueId,
    bound: ValueId,
}

/// RFC-0094 rule 4's `i + c₀`, which the header computes and compares.
#[derive(Clone, Copy)]
struct Sum {
    value: ValueId,
    header_position: usize,
    offset: ValueId,
}

struct HeaderSum {
    position: usize,
    left: ValueId,
    right: ValueId,
}

/// The header parameter the compared value counts with.
struct Compared {
    counter: ValueId,
    sum: Option<Sum>,
}

struct Counting {
    init: ValueId,
    form: Form,
}

struct SplitSum {
    counter: ValueId,
    offset: ValueId,
}

/// Which spelling of a counted loop the header is, and what its range is.
#[derive(Clone, Copy)]
enum Form {
    /// RFC-0081: `i < n` stepping by one, the range `b..n`.
    UpTo,
    /// RFC-0094 rule 1: `i <= n` stepping by one, `b..n + 1`.
    Through,
    /// RFC-0094 rule 2: `i > n` stepping by `− 1`, `Range { at: n, hi: b }`.
    DownTo,
    /// RFC-0094 rule 3: a word-constant step compared in the direction it
    /// moves, a range over the trip count.
    Stepped { ascending: bool, stride: u64 },
    /// RFC-0094 rule 4: `i + c₀ < n` stepping by one, `b + c₀ .. n`.
    Offset(Sum),
}

struct Recognized {
    form: Form,
    counter: ValueId,
    init: ValueId,
    bound: ValueId,
    width: IntTy,
    step_may_trap: bool,
}

struct Counted {
    header: Label,
    entering: Label,
    body: Label,
    body_args: Vec<ValueId>,
    exit: Label,
    exit_args: Vec<ValueId>,
    init: ValueId,
    counter: ValueId,
    width: IntTy,
    hi: Bound,
    steps: Vec<Step>,
    form: Form,
    /// The value every back edge sends the counter, where the program's
    /// own step can leave the width on the last iteration: the compare that
    /// kept it live is gone, so a `Check` at its place keeps its trap
    /// (RFC-0037 rule 3).
    kept_step: Option<ValueId>,
    own_exit: Option<OwnExit>,
}

/// Where the exit block is entered from the body too (RFC-0094 rule 7), the
/// `For` leaves through a block of its own that jumps to it, so the exit
/// block of the range is entered from the header alone (RFC-0081 rule 1)
/// and IV canonicalization can hand it the trip count. The block goes right
/// after the last latch, where `acvus-interpreter`'s `prepare` reads a
/// `for` region's exit label.
struct OwnExit {
    after: Label,
}

struct Recognizer<'a> {
    cfg: &'a CfgBody,
    loop_: &'a Loop,
    invariants: &'a Invariants,
    laws: &'a LawTable,
    loans: &'a Loans<'a>,
    preds: &'a FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
    literals: &'a FxHashMap<ValueId, Literal>,
}

impl Recognizer<'_> {
    fn counted(&self) -> Option<Counted> {
        let LoopKind::While = self.loop_.kind else {
            return None;
        };
        let natural = &self.loop_.natural;
        let header = natural.header;
        let [entering] = natural.entering[..] else {
            return None;
        };
        let Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } = &self.cfg.blocks[header.0].terminator
        else {
            return None;
        };
        let body_block = self.cfg.label_to_block[then_label];
        let exit_block = self.cfg.label_to_block[else_label];
        if body_block == header
            || !natural.contains(body_block)
            || natural.contains(exit_block)
            || !self.only_from_header(body_block)
            || !self.leaves_by_breaks(exit_block)
        {
            return None;
        }
        let own_exit = match self.only_from_header(exit_block) {
            true => None,
            false => Some(OwnExit {
                after: self.cfg.blocks[natural.latches.iter().max()?.0].label,
            }),
        };

        let affine = AffineValues::of(self.cfg, self.loop_, self.invariants, self.laws);
        let recognized = self
            .conditions(header, *cond)
            .into_iter()
            .find_map(|condition| self.form(&affine, condition, body_block))?;
        let ty = Ty::Int(recognized.width);
        if self.cfg.val_types[&recognized.bound] != ty
            || self.cfg.val_types[&recognized.init] != ty
        {
            return None;
        }

        let mut steps: Vec<Step> = Vec::new();
        let hi = match self.evaluate(recognized.bound, &mut steps)? {
            Operand::Outside(value) => Bound::Outside(value),
            Operand::Step(value) => Bound::Computed { value },
        };
        if let Form::Offset(sum) = recognized.form {
            self.evaluate(sum.offset, &mut steps)?;
            if !self.read_by_test_and_body_alone(sum.value, *cond) {
                return None;
            }
            steps.push(Step {
                header_position: sum.header_position,
                dst: sum.value,
                kind: StepKind::Arith(Arith::of(self.op_at(header, sum.header_position)?)?),
                emit: Emit::Start,
            });
        }
        if !self.moves_ahead_unobserved(&steps) {
            return None;
        }
        let writes_above = !steps.is_empty()
            || matches!(
                recognized.form,
                Form::Through | Form::Stepped { .. } | Form::Offset(_)
            );
        if writes_above && !self.only_enters_the_header(entering) {
            return None;
        }
        steps.sort_by_key(|step| step.header_position);
        let counter_index = self.cfg.blocks[header.0]
            .params
            .iter()
            .position(|param| *param == recognized.counter)?;
        let kept_step = match recognized.step_may_trap {
            true => Some(natural.back_arg(self.cfg, counter_index)?),
            false => None,
        };

        Some(Counted {
            header: self.cfg.blocks[header.0].label,
            entering: self.cfg.blocks[entering.0].label,
            body: *then_label,
            body_args: then_args.clone(),
            exit: *else_label,
            exit_args: else_args.clone(),
            init: recognized.init,
            counter: recognized.counter,
            width: recognized.width,
            hi,
            steps,
            form: recognized.form,
            kept_step,
            own_exit,
        })
    }

    fn conditions(&self, header: BlockIdx, cond: ValueId) -> Vec<Condition> {
        let test = self.cfg.blocks[header.0]
            .insts
            .iter()
            .find(|inst| inst_info::defs(&inst.kind).contains(&cond));
        let Some(Inst {
            kind: InstKind::BinOp {
                op, left, right, ..
            },
            ..
        }) = test
        else {
            return Vec::new();
        };
        let reading = |compare, compared: &ValueId, bound: &ValueId| Condition {
            compare,
            compared: *compared,
            bound: *bound,
        };
        match op {
            BinOp::Lt => vec![
                reading(Compare::Below, left, right),
                reading(Compare::Above, right, left),
            ],
            BinOp::Gt => vec![
                reading(Compare::Above, left, right),
                reading(Compare::Below, right, left),
            ],
            BinOp::Lte => vec![reading(Compare::AtMost, left, right)],
            BinOp::Gte => vec![reading(Compare::AtMost, right, left)],
            _ => Vec::new(),
        }
    }

    fn form(
        &self,
        affine: &AffineValues,
        condition: Condition,
        body_block: BlockIdx,
    ) -> Option<Recognized> {
        let Compared { counter, sum } = self.compared(condition.compared)?;
        let Ty::Int(width) = self.cfg.val_types[&counter] else {
            return None;
        };
        let Counting { init, form } = match (&affine.get(counter)?.derivation, sum) {
            (Derivation::Carried { init, step }, Some(sum))
                if condition.compare == Compare::Below && self.is_one(step) =>
            {
                Counting {
                    init: *init,
                    form: Form::Offset(sum),
                }
            }
            (Derivation::Carried { init, step }, None) if self.is_one(step) => Counting {
                init: *init,
                form: match condition.compare {
                    Compare::Below => Form::UpTo,
                    Compare::AtMost if self.below_the_maximum(condition.bound, width, body_block) => {
                        Form::Through
                    }
                    Compare::AtMost | Compare::Above => return None,
                },
            },
            (Derivation::Carried { init, step }, None) => {
                let step = width.read(self.step_word(step)? as u64);
                let ascending = match condition.compare {
                    Compare::Below if step > 1 => true,
                    Compare::Above if step < 0 => false,
                    Compare::Below | Compare::Above | Compare::AtMost => return None,
                };
                Counting {
                    init: *init,
                    form: Form::Stepped {
                        ascending,
                        stride: u64::try_from(step.unsigned_abs()).ok()?,
                    },
                }
            }
            (Derivation::CountsDown { init, .. }, None) if condition.compare == Compare::Above => {
                Counting {
                    init: *init,
                    form: Form::DownTo,
                }
            }
            _ => return None,
        };
        let step_may_trap = match form {
            Form::UpTo | Form::Through | Form::DownTo => false,
            Form::Stepped { .. } => true,
            Form::Offset(sum) => {
                let offset = self.word_of(sum.offset)?.desugared();
                let Literal::Int(offset) = offset else {
                    return None;
                };
                width.read(offset as u64) < 0
            }
        };
        Some(Recognized {
            form,
            counter,
            init,
            bound: condition.bound,
            width,
            step_may_trap,
        })
    }

    /// The compared value is a header parameter, or the header's integer
    /// `+` of one and a word constant (RFC-0094 rule 4).
    fn compared(&self, value: ValueId) -> Option<Compared> {
        let is_param = |value: ValueId| {
            self.cfg.blocks[self.loop_.natural.header.0]
                .params
                .contains(&value)
        };
        if is_param(value) {
            return Some(Compared {
                counter: value,
                sum: None,
            });
        }
        let HeaderSum {
            position,
            left,
            right,
        } = self.header_sum(value)?;
        let SplitSum { counter, offset } = match (is_param(left), is_param(right)) {
            (true, false) => SplitSum {
                counter: left,
                offset: right,
            },
            (false, true) => SplitSum {
                counter: right,
                offset: left,
            },
            (true, true) | (false, false) => return None,
        };
        self.word_of(offset)?;
        Some(Compared {
            counter,
            sum: Some(Sum {
                value,
                header_position: position,
                offset,
            }),
        })
    }

    fn header_sum(&self, value: ValueId) -> Option<HeaderSum> {
        let header = self.loop_.natural.header;
        self.cfg.blocks[header.0]
            .insts
            .iter()
            .enumerate()
            .find_map(|(position, inst)| match inst.kind {
                InstKind::BinOp {
                    dst,
                    op: BinOp::Add(_),
                    left,
                    right,
                } if dst == value => Some(HeaderSum {
                    position,
                    left,
                    right,
                }),
                _ => None,
            })
    }

    fn op_at(&self, block: BlockIdx, position: usize) -> Option<BinOp> {
        match self.cfg.blocks[block.0].insts[position].kind {
            InstKind::BinOp { op, .. } => Some(op),
            _ => None,
        }
    }

    fn read_by_test_and_body_alone(&self, value: ValueId, cond: ValueId) -> bool {
        let natural = &self.loop_.natural;
        let header = natural.header;
        self.cfg.blocks.iter().enumerate().all(|(at, block)| {
            let at = BlockIdx(at);
            let reads = block
                .insts
                .iter()
                .filter(|inst| inst_info::uses(&inst.kind).contains(&value))
                .map(|inst| inst_info::defs(&inst.kind).into_vec())
                .collect::<Vec<_>>();
            let terminator_reads = inst_info::terminator_uses(&block.terminator).contains(&value);
            match (at == header, natural.contains(at)) {
                (true, _) => reads.iter().all(|defs| defs[..] == [cond]) && !terminator_reads,
                (false, true) => true,
                (false, false) => reads.is_empty() && !terminator_reads,
            }
        })
    }

    fn below_the_maximum(&self, bound: ValueId, width: IntTy, at: BlockIdx) -> bool {
        let [bounds] = constant_bounds_on_entry(self.cfg, self.laws, at, &[bound])[..] else {
            return false;
        };
        bounds.hi.is_some_and(|hi| hi < width.max())
    }

    fn evaluate(&self, value: ValueId, steps: &mut Vec<Step>) -> Option<Operand> {
        let natural = &self.loop_.natural;
        if let Some(Invariant::Outside(value)) = self.invariants.above(natural, value) {
            return Some(Operand::Outside(value));
        }
        if steps.iter().any(|step| step.dst == value) {
            return Some(Operand::Step(value));
        }
        let (header_position, inst) = self.cfg.blocks[natural.header.0]
            .insts
            .iter()
            .enumerate()
            .find(|(_, inst)| inst_info::defs(&inst.kind).contains(&value))?;
        let kind = match &inst.kind {
            InstKind::Const { .. } => match self.invariants.above(natural, value)? {
                Invariant::Word(_) => StepKind::Word,
                Invariant::Outside(_) => return None,
            },
            InstKind::Ref {
                dst,
                target,
                mutability: Mutability::Shared,
                ..
            } if self.left_alone(target, *dst) => StepKind::Borrow,
            InstKind::BinOp {
                op, left, right, ..
            } if matches!(self.cfg.val_types[&value], Ty::Int(_)) => {
                let op = Arith::of(*op)?;
                self.evaluate(*left, steps)?;
                self.evaluate(*right, steps)?;
                StepKind::Arith(op)
            }
            InstKind::FunctionCall {
                callee: Callee::Extern { .. },
                callee_ty,
                args,
                ..
            } if callee_ty.effect().is_some_and(|effect| effect.is_empty()) => {
                for &arg in args {
                    self.argument(arg, steps)?;
                }
                StepKind::Call
            }
            _ => return None,
        };
        steps.push(Step {
            header_position,
            dst: value,
            kind,
            emit: Emit::Move,
        });
        Some(Operand::Step(value))
    }

    fn argument(&self, arg: ValueId, steps: &mut Vec<Step>) -> Option<Operand> {
        match &self.cfg.val_types[&arg] {
            Ty::Ref(mutability, _) => {
                if *mutability == Mutability::Mut {
                    return None;
                }
                match self.evaluate(arg, steps)? {
                    outside @ Operand::Outside(_) => Some(outside),
                    Operand::Step(step) => steps
                        .iter()
                        .any(|found| found.dst == step && matches!(found.kind, StepKind::Borrow))
                        .then_some(Operand::Step(step)),
                }
            }
            ty if ty.is_scalar() => self.evaluate(arg, steps),
            _ => None,
        }
    }

    /// RFC-0094 rule 5: the slot a header `Ref` names is one no instruction
    /// of the loop writes, lends `&mut`, or lends to a call that reaches it
    /// (RFC-0082 rule 7), and no `for` of the loop traverses `&mut`. A call
    /// the borrow itself is lent to reads through a shared reference and is
    /// a step of the bound, which rule 3 of RFC-0081 decides.
    fn left_alone(&self, target: &RefTarget, borrow: ValueId) -> bool {
        let Some(slot) = inst_info::storage(target) else {
            return false;
        };
        let lends = |value: &ValueId| {
            *value != borrow
                && self
                    .loans
                    .holds(*value)
                    .any(|loan| loan.storage.slot() == Some(slot))
        };
        self.loop_.natural.blocks().all(|block| {
            let held = &self.cfg.blocks[block.0];
            let insts_leave_it = held.insts.iter().all(|inst| {
                let kind = &inst.kind;
                !effect(self.loans, kind).writes.contains(&slot)
                    && !slots_lent_mutably(self.loans, kind).contains(&slot)
                    && !self.call_reaches(kind, &lends)
            });
            let traversal_leaves_it = match &held.terminator {
                Terminator::For {
                    source: ForSource::SliceMut(source),
                    ..
                } => !lends(source),
                _ => true,
            };
            insts_leave_it && traversal_leaves_it
        })
    }

    fn call_reaches(&self, kind: &InstKind, lends: &impl Fn(&ValueId) -> bool) -> bool {
        let (InstKind::FunctionCall { callee, args, .. } | InstKind::Spawn { callee, args, .. }) =
            kind
        else {
            return false;
        };
        let reached = |param: usize| match callee {
            Callee::Extern { .. } => match self.laws.reaches_of(callee) {
                Reaches::Lent => true,
                Reaches::Places(declared) => declared.iter().any(|place| place.param == param),
            },
            Callee::Direct(_) | Callee::Indirect(_) => true,
        };
        args.iter()
            .enumerate()
            .any(|(param, arg)| lends(arg) && reached(param))
    }

    fn moves_ahead_unobserved(&self, steps: &[Step]) -> bool {
        let Some(last_trap) = steps
            .iter()
            .filter(|step| step.kind.traps())
            .map(|step| step.header_position)
            .max()
        else {
            return true;
        };
        self.cfg.blocks[self.loop_.natural.header.0].insts[..last_trap]
            .iter()
            .enumerate()
            .filter(|(position, _)| !steps.iter().any(|step| step.header_position == *position))
            .all(|(_, inst)| inst_info::cannot_end_run(&inst.kind, &self.cfg.val_types))
    }

    fn only_enters_the_header(&self, block: BlockIdx) -> bool {
        let header = self.cfg.blocks[self.loop_.natural.header.0].label;
        matches!(
            &self.cfg.blocks[block.0].terminator,
            Terminator::Jump { label, .. } if *label == header
        )
    }

    fn only_from_header(&self, block: BlockIdx) -> bool {
        match self.preds.get(&block) {
            Some(preds) => preds[..] == [self.loop_.natural.header],
            None => false,
        }
    }

    /// RFC-0094 rule 7: every edge out of a block of the loop other than
    /// the header goes to a block outside it from which every path reaches
    /// `exit` with no `return` and no `!` on the way, and `exit` is entered
    /// from nowhere else.
    fn leaves_by_breaks(&self, exit: BlockIdx) -> bool {
        let natural = &self.loop_.natural;
        let mut broken: FxHashSet<BlockIdx> = FxHashSet::default();
        let mut work: Vec<BlockIdx> = natural
            .blocks()
            .filter(|&block| block != natural.header)
            .flat_map(|block| self.cfg.successors(block))
            .filter(|&succ| !natural.contains(succ))
            .collect();
        while let Some(block) = work.pop() {
            if block == exit || !broken.insert(block) {
                continue;
            }
            if matches!(
                self.cfg.blocks[block.0].terminator,
                Terminator::Return { .. } | Terminator::Diverge
            ) {
                return false;
            }
            for succ in self.cfg.successors(block) {
                if natural.contains(succ) {
                    return false;
                }
                work.push(succ);
            }
        }
        self.preds.get(&exit).is_some_and(|preds| {
            preds
                .iter()
                .all(|pred| natural.contains(*pred) || broken.contains(pred))
        })
    }

    fn leaves_only_from_header(&self) -> bool {
        let natural = &self.loop_.natural;
        natural
            .blocks()
            .filter(|&block| block != natural.header)
            .all(|block| {
                !matches!(
                    self.cfg.blocks[block.0].terminator,
                    Terminator::Return { .. } | Terminator::Diverge
                ) && self
                    .cfg
                    .successors(block)
                    .iter()
                    .all(|&succ| natural.contains(succ))
            })
    }

    fn word_of(&self, value: ValueId) -> Option<&Literal> {
        match self.invariants.above(&self.loop_.natural, value)? {
            Invariant::Word(_) | Invariant::Outside(_) => self.literals.get(&value),
        }
    }

    fn step_word(&self, step: &AffineOperand) -> Option<i128> {
        let literal = match step.invariance.above()? {
            Invariant::Word(literal) => literal.clone(),
            Invariant::Outside(value) => self.literals.get(value)?.clone(),
        };
        match literal.desugared() {
            Literal::Int(value) => Some(value),
            _ => None,
        }
    }

    fn is_one(&self, step: &AffineOperand) -> bool {
        let literal = match step.invariance.above() {
            Some(Invariant::Word(literal)) => Some(literal),
            Some(Invariant::Outside(value)) => self.literals.get(value),
            None => None,
        };
        matches!(
            literal,
            Some(Literal::Int(1) | Literal::IntOf(SuffixedInt { value: 1, .. }))
        )
    }
}

impl Recognizer<'_> {
    fn pull(&self, interner: &Interner) -> Option<Pull> {
        let LoopKind::While = self.loop_.kind else {
            return None;
        };
        let natural = &self.loop_.natural;
        let header = natural.header;
        let Terminator::JumpIf {
            cond,
            then_label,
            then_args,
            else_label,
            else_args,
        } = &self.cfg.blocks[header.0].terminator
        else {
            return None;
        };
        let body_block = self.cfg.label_to_block[then_label];
        let exit_block = self.cfg.label_to_block[else_label];
        if body_block == header
            || !natural.contains(body_block)
            || natural.contains(exit_block)
            || !self.only_from_header(body_block)
            || !self.leaves_only_from_header()
        {
            return None;
        }
        let [lend, pull, test] = &self.cfg.blocks[header.0].insts[..] else {
            return None;
        };
        let InstKind::Ref {
            dst: lent,
            target,
            mutability: Mutability::Mut,
            ..
        } = &lend.kind
        else {
            return None;
        };
        let InstKind::FunctionCall {
            dst: pulled,
            callee: Callee::Extern { id, .. },
            args,
            ..
        } = &pull.kind
        else {
            return None;
        };
        let InstKind::TestVariant { dst, src, tag } = &test.kind else {
            return None;
        };
        let storage = inst_info::storage(target)?;
        let pulls = interner.resolve(id.name) == "next"
            && args[..] == [*lent]
            && matches!(self.cfg.val_types.get(pulled), Some(Ty::Option(_)))
            && src == pulled
            && interner.resolve(*tag) == "Some"
            && dst == cond;
        let lends_the_storage_alone = self
            .loans
            .names(*lent)
            .iter()
            .all(|loan| loan.storage.slot() == Some(storage));
        if !pulls || !lends_the_storage_alone || self.touched_past_header(storage) {
            return None;
        }
        Some(Pull {
            header: self.cfg.blocks[header.0].label,
            cond: *cond,
            body: *then_label,
            body_args: then_args.clone(),
            exit: *else_label,
            exit_args: else_args.clone(),
        })
    }

    fn touched_past_header(&self, storage: ValueId) -> bool {
        let natural = &self.loop_.natural;
        let holds = |value: ValueId| {
            self.loans
                .holds(value)
                .any(|loan| loan.storage.slot() == Some(storage))
        };
        natural
            .blocks()
            .filter(|block| *block != natural.header)
            .any(|block| {
                let held = &self.cfg.blocks[block.0];
                held.insts.iter().any(|inst| {
                    touched_slots(self.loans, &inst.kind).contains(&storage)
                        || inst_info::uses(&inst.kind).into_iter().any(holds)
                }) || inst_info::terminator_uses(&held.terminator)
                    .into_iter()
                    .any(holds)
            })
    }
}

impl Pull {
    fn apply(self, cfg: &mut CfgBody) {
        substitute_body_params(cfg, self.body, self.body_args);
        let header = cfg.label_to_block[&self.header];
        cfg.blocks[header.0].terminator = Terminator::While {
            cond: self.cond,
            stages: Stages::lowered(self.body),
            exit: self.exit,
            exit_args: self.exit_args,
        };
    }
}

impl Counted {
    fn apply(self, cfg: &mut CfgBody, labels: &mut LabelFactory) {
        let header = cfg.label_to_block[&self.header];
        let entering = cfg.label_to_block[&self.entering];
        let ty = Ty::Int(self.width);
        let mut moved: Vec<Inst> = Vec::with_capacity(self.steps.len());
        let mut start: Option<ValueId> = None;
        for step in &self.steps {
            let mut copied = cfg.blocks[header.0].insts[step.header_position].clone();
            if let Emit::Start = step.emit {
                let entry_start = fresh(cfg, &ty);
                apply_subst(
                    &mut copied.kind,
                    &FxHashMap::from_iter([(self.counter, self.init)]),
                );
                let InstKind::BinOp { dst, .. } = &mut copied.kind else {
                    panic!("rule 4's start copies the header's `i + c₀`, a `BinOp`")
                };
                *dst = entry_start;
                start = Some(entry_start);
            }
            moved.push(copied);
        }
        let mut into_body: Vec<Inst> = Vec::new();
        let mut into_exit: Vec<Inst> = Vec::new();
        let header_insts = std::mem::take(&mut cfg.blocks[header.0].insts);
        for (position, inst) in header_insts.into_iter().enumerate() {
            let emit = self
                .steps
                .iter()
                .find(|step| step.header_position == position)
                .map(|step| step.emit);
            match emit {
                Some(Emit::Move) => {}
                Some(Emit::Start) => into_body.push(inst),
                None => {
                    into_exit.push(inst.clone());
                    into_body.push(inst);
                }
            }
        }
        cfg.blocks[entering.0].insts.extend(moved);

        let hi = self.hi.value();
        let Traversal { source, counter } = match self.form {
            Form::UpTo => Traversal {
                source: ForSource::Range {
                    at: self.init,
                    hi,
                },
                counter: ty,
            },
            Form::Through => {
                let one = constant(cfg, entering, &ty, 1);
                let past = fresh(cfg, &ty);
                push(
                    cfg,
                    entering,
                    InstKind::BinOp {
                        dst: past,
                        op: BinOp::Add(Overflow::Wrap),
                        left: hi,
                        right: one,
                    },
                );
                Traversal {
                    source: ForSource::Range {
                        at: self.init,
                        hi: past,
                    },
                    counter: ty,
                }
            }
            Form::DownTo => Traversal {
                source: ForSource::Range {
                    at: hi,
                    hi: self.init,
                },
                counter: ty,
            },
            Form::Offset(_) => {
                let Some(at) = start else {
                    panic!("rule 4's range starts at the entry's copy of its `i + c₀`")
                };
                Traversal {
                    source: ForSource::Range { at, hi },
                    counter: ty,
                }
            }
            Form::Stepped { ascending, stride } => {
                let trip = TripCount {
                    header: self.header,
                    entering: self.entering,
                    width: self.width,
                    init: self.init,
                    bound: hi,
                    ascending,
                    stride,
                }
                .write(cfg, labels);
                Traversal {
                    source: ForSource::Range {
                        at: trip.zero,
                        hi: trip.count,
                    },
                    counter: Ty::Int(unsigned(self.width)),
                }
            }
        };

        if let Some(next) = self.kept_step {
            keep_step_trap(cfg, next);
        }
        substitute_body_params(cfg, self.body, self.body_args);
        let body_block = cfg.label_to_block[&self.body];
        let head = std::mem::take(&mut cfg.blocks[body_block.0].insts);
        cfg.blocks[body_block.0].insts = into_body.into_iter().chain(head).collect();
        let counter = fresh(cfg, &counter);
        cfg.blocks[body_block.0].params.push(counter);
        let (exit, joins) = match self.own_exit {
            Some(own) => (
                ExitEdge {
                    label: own.write(cfg, labels, self.exit, self.exit_args),
                    args: Vec::new(),
                },
                Some(self.exit),
            ),
            None => (
                ExitEdge {
                    label: self.exit,
                    args: self.exit_args,
                },
                None,
            ),
        };
        let exit_label = exit.label;
        let header = cfg.label_to_block[&self.header];
        cfg.blocks[header.0].terminator = Terminator::For {
            source,
            stages: Stages::lowered(self.body),
            exit: exit.label,
            exit_args: exit.args,
            exit_trip: ExitTrip::Absent,
        };
        HeaderRest {
            header,
            body: cfg.label_to_block[&self.body],
            exit: cfg.label_to_block[&exit_label],
            own_exit_target: joins.map(|label| cfg.label_to_block[&label]),
            insts: into_exit,
        }
        .run_at_exit(cfg);
    }
}

/// RFC-0081 rule 2: the header's instructions that are no step of the
/// bound, which the body block's head runs under the header's names and
/// the exit block's head under fresh ones.
struct HeaderRest {
    header: BlockIdx,
    body: BlockIdx,
    exit: BlockIdx,
    own_exit_target: Option<BlockIdx>,
    insts: Vec<Inst>,
}

impl HeaderRest {
    fn run_at_exit(self, cfg: &mut CfgBody) {
        let mut renamed: FxHashMap<ValueId, ValueId> = FxHashMap::default();
        let mut copies: Vec<Inst> = Vec::with_capacity(self.insts.len());
        for mut inst in self.insts {
            apply_subst(&mut inst.kind, &renamed);
            map_value_defs(&mut inst.kind, &mut |dst| {
                let ty = cfg.val_types[dst].clone();
                let copy = fresh(cfg, &ty);
                renamed.insert(*dst, copy);
                *dst = copy;
            });
            copies.push(inst);
        }
        if copies.is_empty() {
            return;
        }
        let exit_insts = std::mem::take(&mut cfg.blocks[self.exit.0].insts);
        cfg.blocks[self.exit.0].insts = copies.into_iter().chain(exit_insts).collect();

        let mut at_exit = renamed.clone();
        let Terminator::For { exit_args, .. } = &mut cfg.blocks[self.header.0].terminator else {
            panic!("the header was just given its `For`")
        };
        let sent: Vec<usize> = (0..exit_args.len())
            .filter(|at| renamed.contains_key(&exit_args[*at]))
            .collect();
        let values: Vec<ValueId> = sent.iter().rev().map(|at| exit_args.remove(*at)).collect();
        for (at, value) in sent.iter().rev().zip(values) {
            let param = cfg.blocks[self.exit.0].params.remove(*at);
            at_exit.insert(param, renamed[&value]);
        }

        let domtree = DomTree::build(cfg);
        let mut past_join: Vec<ValueId> = Vec::new();
        for (index, block) in cfg.blocks.iter_mut().enumerate() {
            let block_idx = BlockIdx(index);
            if block_idx == self.header || domtree.dominates(self.body, block_idx) {
                continue;
            }
            if domtree.dominates(self.exit, block_idx) {
                for inst in &mut block.insts {
                    apply_subst(&mut inst.kind, &at_exit);
                }
                apply_subst_terminator(&mut block.terminator, &at_exit);
                continue;
            }
            let read = block
                .insts
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind))
                .chain(inst_info::terminator_uses(&block.terminator))
                .filter(|value| renamed.contains_key(value));
            for value in read {
                let joined = self
                    .own_exit_target
                    .is_some_and(|joins| domtree.dominates(joins, block_idx));
                assert!(
                    joined,
                    "{value:?} is read in {block_idx:?}, which neither the body, the exit nor a \
                     join of the two dominates"
                );
                if !past_join.contains(&value) {
                    past_join.push(value);
                }
            }
        }
        let Some(joins) = self.own_exit_target else {
            return;
        };
        if past_join.is_empty() {
            return;
        }

        let joins_label = cfg.blocks[joins.0].label;
        let mut at_join: FxHashMap<ValueId, ValueId> = FxHashMap::default();
        for value in &past_join {
            let ty = cfg.val_types[value].clone();
            let param = fresh(cfg, &ty);
            cfg.blocks[joins.0].params.push(param);
            at_join.insert(*value, param);
        }
        for (index, block) in cfg.blocks.iter_mut().enumerate() {
            let from_exit = BlockIdx(index) == self.exit;
            for edge in edges_mut(&mut block.terminator) {
                if *edge.to != joins_label {
                    continue;
                }
                edge.args
                    .extend(past_join.iter().map(|value| match from_exit {
                        true => renamed[value],
                        false => *value,
                    }));
            }
        }
        for (index, block) in cfg.blocks.iter_mut().enumerate() {
            if !domtree.dominates(joins, BlockIdx(index)) {
                continue;
            }
            for inst in &mut block.insts {
                apply_subst(&mut inst.kind, &at_join);
            }
            apply_subst_terminator(&mut block.terminator, &at_join);
        }
    }
}

struct ExitEdge {
    label: Label,
    args: Vec<ValueId>,
}

impl OwnExit {
    /// A block that jumps to `exit` with `args`, written after the latch.
    fn write(
        self,
        cfg: &mut CfgBody,
        labels: &mut LabelFactory,
        exit: Label,
        args: Vec<ValueId>,
    ) -> Label {
        let label = labels.fresh();
        let block = Block {
            label,
            params: Vec::new(),
            insts: Vec::new(),
            terminator: Terminator::Jump { label: exit, args },
        };
        let after_latch = cfg.label_to_block[&self.after].0 + 1;
        cfg.blocks.insert(after_latch, block);
        cfg.label_to_block = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| (block.label, BlockIdx(at)))
            .collect();
        label
    }
}

/// RFC-0094 rule 3's trip count, written between the entering block and
/// the header: the entering block branches on whether `b` has passed `n`,
/// one arm computes `(d − 1) / |s| + 1` from the distance `d` read unsigned
/// at the width, the other zero, and the block they join passes the count
/// and the range's start to the header.
struct TripCount {
    header: Label,
    entering: Label,
    width: IntTy,
    init: ValueId,
    bound: ValueId,
    ascending: bool,
    stride: u64,
}

struct Traversal {
    source: ForSource,
    counter: Ty,
}

struct Interval {
    from: ValueId,
    to: ValueId,
}

struct WrittenTrip {
    zero: ValueId,
    count: ValueId,
}

impl TripCount {
    fn write(self, cfg: &mut CfgBody, labels: &mut LabelFactory) -> WrittenTrip {
        let width = Ty::Int(self.width);
        let unsigned_width = unsigned(self.width);
        let unsigned = Ty::Int(unsigned_width);
        let then_label = labels.fresh();
        let else_label = labels.fresh();
        let join = labels.fresh();
        let entering = cfg.label_to_block[&self.entering];

        let before = fresh(cfg, &Ty::Bool);
        let Interval { from, to } = match self.ascending {
            true => Interval {
                from: self.init,
                to: self.bound,
            },
            false => Interval {
                from: self.bound,
                to: self.init,
            },
        };
        push(
            cfg,
            entering,
            InstKind::BinOp {
                dst: before,
                op: BinOp::Lt,
                left: from,
                right: to,
            },
        );
        let Terminator::Jump { label, args } = std::mem::replace(
            &mut cfg.blocks[entering.0].terminator,
            Terminator::Diamond {
                cond: before,
                then_label,
                then_args: Vec::new(),
                else_label,
                else_args: Vec::new(),
                join,
            },
        ) else {
            panic!("the recognizer admits a computed count only where the entry jumps to the header")
        };
        assert_eq!(label, self.header, "the entering block jumps to the header");

        let mut ahead = Vec::new();
        let distance = fresh(cfg, &width);
        ahead.push(InstKind::BinOp {
            dst: distance,
            op: BinOp::Sub(Overflow::Wrap),
            left: to,
            right: from,
        });
        let read = match width == unsigned {
            true => distance,
            false => {
                let read = fresh(cfg, &unsigned);
                ahead.push(InstKind::Cast {
                    dst: read,
                    src: distance,
                    to: CastTy::Int(unsigned_width),
                });
                read
            }
        };
        let one = fresh(cfg, &unsigned);
        ahead.push(InstKind::Const {
            dst: one,
            value: Literal::Int(1),
        });
        let short = fresh(cfg, &unsigned);
        ahead.push(InstKind::BinOp {
            dst: short,
            op: BinOp::Sub(Overflow::Wrap),
            left: read,
            right: one,
        });
        let stride = fresh(cfg, &unsigned);
        ahead.push(InstKind::Const {
            dst: stride,
            value: Literal::Int(i128::from(self.stride)),
        });
        let quotient = fresh(cfg, &unsigned);
        ahead.push(InstKind::BinOp {
            dst: quotient,
            op: BinOp::Div,
            left: short,
            right: stride,
        });
        let count_ahead = fresh(cfg, &unsigned);
        ahead.push(InstKind::BinOp {
            dst: count_ahead,
            op: BinOp::Add(Overflow::Wrap),
            left: quotient,
            right: one,
        });

        let none = fresh(cfg, &unsigned);
        let count = fresh(cfg, &unsigned);
        let zero = fresh(cfg, &unsigned);
        let blocks = [
            Block {
                label: then_label,
                params: Vec::new(),
                insts: ahead.into_iter().map(inst).collect(),
                terminator: Terminator::Jump {
                    label: join,
                    args: vec![count_ahead],
                },
            },
            Block {
                label: else_label,
                params: Vec::new(),
                insts: vec![inst(InstKind::Const {
                    dst: none,
                    value: Literal::Int(0),
                })],
                terminator: Terminator::Jump {
                    label: join,
                    args: vec![none],
                },
            },
            Block {
                label: join,
                params: vec![count],
                insts: vec![inst(InstKind::Const {
                    dst: zero,
                    value: Literal::Int(0),
                })],
                terminator: Terminator::Jump { label, args },
            },
        ];
        let header = cfg.label_to_block[&self.header];
        let tail = cfg.blocks.split_off(header.0);
        cfg.blocks.extend(blocks);
        cfg.blocks.extend(tail);
        cfg.label_to_block = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(at, block)| (block.label, BlockIdx(at)))
            .collect();
        WrittenTrip { zero, count }
    }
}

/// A `Check` of the program's step that defines `next`, written at its
/// place.
fn keep_step_trap(cfg: &mut CfgBody, next: ValueId) {
    let found = cfg.blocks.iter().enumerate().find_map(|(block, held)| {
        held.insts
            .iter()
            .position(|inst| inst_info::defs(&inst.kind).contains(&next))
            .map(|at| (BlockIdx(block), at))
    });
    let Some((block, at)) = found else {
        panic!("{next:?} is the counter's step, which an instruction of the loop defines")
    };
    let step = &cfg.blocks[block.0].insts[at];
    let InstKind::BinOp {
        op, left, right, ..
    } = step.kind
    else {
        panic!("the counter's step is the `BinOp` `analysis::affine` read, not {step:?}")
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

fn unsigned(width: IntTy) -> IntTy {
    match width {
        IntTy::I8 | IntTy::U8 => IntTy::U8,
        IntTy::I16 | IntTy::U16 => IntTy::U16,
        IntTy::I32 | IntTy::U32 => IntTy::U32,
        IntTy::I64 | IntTy::U64 => IntTy::U64,
    }
}

fn inst(kind: InstKind) -> Inst {
    Inst {
        span: acvus_ast::Span::ZERO,
        kind,
    }
}

fn push(cfg: &mut CfgBody, block: BlockIdx, kind: InstKind) {
    cfg.blocks[block.0].insts.push(inst(kind));
}

fn constant(cfg: &mut CfgBody, block: BlockIdx, ty: &Ty, value: i128) -> ValueId {
    let dst = fresh(cfg, ty);
    push(
        cfg,
        block,
        InstKind::Const {
            dst,
            value: Literal::Int(value),
        },
    );
    dst
}

fn substitute_body_params(cfg: &mut CfgBody, body: Label, passed: Vec<ValueId>) {
    let body_block = cfg.label_to_block[&body];
    let params = std::mem::take(&mut cfg.blocks[body_block.0].params);
    let subst: FxHashMap<ValueId, ValueId> = params.into_iter().zip(passed).collect();
    for block in &mut cfg.blocks {
        for inst in &mut block.insts {
            apply_subst(&mut inst.kind, &subst);
        }
        apply_subst_terminator(&mut block.terminator, &subst);
    }
}

struct LabelFactory {
    next: u32,
}

impl LabelFactory {
    fn of(cfg: &CfgBody) -> Self {
        let next = cfg
            .blocks
            .iter()
            .map(|block| block.label)
            .filter(|label| *label != ENTRY_LABEL)
            .map(|label| label.0 + 1)
            .max()
            .unwrap_or(0);
        Self { next }
    }

    fn fresh(&mut self) -> Label {
        let label = Label(self.next);
        self.next += 1;
        label
    }
}

fn const_literals(cfg: &CfgBody) -> FxHashMap<ValueId, Literal> {
    cfg.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.kind {
            InstKind::Const { dst, value } => Some((*dst, value.clone())),
            _ => None,
        })
        .collect()
}

fn fresh(cfg: &mut CfgBody, ty: &Ty) -> ValueId {
    let value = cfg.val_factory.next();
    let previous = cfg.val_types.insert(value, ty.clone());
    assert!(
        previous.is_none(),
        "{value:?} is fresh from the factory and already carried a type"
    );
    cfg.debug.set(value, ValOrigin::Expr);
    value
}
