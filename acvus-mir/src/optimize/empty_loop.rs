//! A `for` whose body does nothing is a jump to its exit (RFC-0088).
//!
//! A range's trip count is `(max(hi, at) as u64) − (at as u64)` and not
//! `max(hi − at, 0)` at the range's width `w`, because the second leaves the
//! width: at `i8`, `-100..100` has `hi − at` at `200`. The `max` compares at
//! `w`'s own signedness, so `max(hi, at) − at` is the count over the
//! integers, which lies in `[0, 2^64)`. An integer `Cast` keeps its operand
//! modulo `2^64`, and the subtraction is this pass's, so it wraps modulo
//! `2^64` (RFC-0037 rule 3): the difference of the two casts is that count
//! exactly, and at `-5..3` the casts' own difference wraps on the way.
//!
//! A slice is declined, not an omission: the MIR has no instruction that
//! reads a slice's length, and one is not added for this pass.
//!
//! A body whose only work is `Check`s of the `+` that advances a value by
//! a fixed step, and the wrapping arithmetic from the counter those checks
//! read, is removed too, and the checks become one `CheckSteps` each at the
//! header (RFC-0088 rule 8). That is where IV canonicalization leaves a
//! variable whose step can overflow (RFC-0066 rule 7). A check reads
//! `v(k) + step`, where `v(k)` is computed wrapping as `v(0) + k·step`
//! modulo `2^w`. `v(0)` is a value of the width; on a run whose checks at
//! iterations below `k` passed, `v(0) + k·step` fits, so `v(k)` is that
//! integer, and the check at `k` traps exactly where `v(0) + (k+1)·step`
//! does not fit. That value is monotone in `k` and `v(0)` fits, so some
//! iteration below the count `n` traps exactly where `v(0) + n·step` does
//! not fit: one `CheckSteps { from: v(0), step, count: n }`. Nothing else in
//! the body can end the run or be observed, so no other trap or effect
//! could precede it inside the loop, and every check traps with `+`'s text,
//! so which of several traps first does not show.

use acvus_ast::{Literal, Span};
use rustc_hash::FxHashMap;

use crate::analysis::affine::for_body;
use crate::analysis::domtree::DomTree;
use crate::analysis::loops::{NaturalLoop, natural_loops_innermost_first};
use crate::cfg::{BlockIdx, CfgBody, Terminator, prune, reachable};
use crate::ir::{
    BinOp, Checked, ExitTrip, ForSource, Inst, InstKind, Label, Overflow, ValOrigin, ValueId,
};
use crate::laws::LawTable;
use crate::optimize::dce;
use crate::ty::{CastTy, IntTy, LenTerm, Ty};

pub fn run(cfg: &mut CfgBody, laws: &LawTable) {
    while let Some(empty) = first_empty(cfg) {
        remove(cfg, empty);
        let alive = reachable(cfg);
        prune(cfg, &alive);
        dce::run(cfg, laws);
    }
}

enum Count {
    Range {
        at: ValueId,
        hi: ValueId,
        width: IntTy,
    },
    Array {
        len: u64,
    },
}

struct Empty {
    header: BlockIdx,
    count: Count,
    exit: Label,
    exit_trip: ExitTrip,
    exit_args: Vec<ValueId>,
    checks: Vec<Stepped>,
    /// The body's instructions but its checks, by the value each defines.
    defs: FxHashMap<ValueId, InstKind>,
}

/// One `Check { op: Add, left, right }` of the body whose `left` advances
/// by `right` on every iteration.
struct Stepped {
    span: Span,
    left: ValueId,
    right: ValueId,
}

fn first_empty(cfg: &CfgBody) -> Option<Empty> {
    let domtree = DomTree::build(cfg);
    natural_loops_innermost_first(cfg, &domtree)
        .iter()
        .find_map(|loop_| Empty::of(cfg, loop_))
}

impl Empty {
    fn of(cfg: &CfgBody, loop_: &NaturalLoop) -> Option<Empty> {
        let header = &cfg.blocks[loop_.header.0];
        let Terminator::For {
            source,
            exit,
            exit_trip,
            exit_args,
            ..
        } = &header.terminator
        else {
            return None;
        };
        let (source, exit, exit_trip) = (*source, *exit, *exit_trip);
        if !header.params.is_empty() || !header.insts.is_empty() {
            return None;
        }
        let only_jumps_within = |block: BlockIdx| {
            let held = &cfg.blocks[block.0];
            let Terminator::Jump { label, .. } = &held.terminator else {
                return false;
            };
            cfg.label_to_block
                .get(label)
                .is_some_and(|to| loop_.contains(*to))
        };
        let body: Vec<BlockIdx> = loop_
            .blocks()
            .filter(|block| *block != loop_.header)
            .collect();
        if !body.iter().all(|block| only_jumps_within(*block)) {
            return None;
        }
        let slopes = Slopes::of(cfg, loop_, source, &body)?;
        let checks = slopes.checks()?;
        let defs = slopes
            .defs
            .iter()
            .map(|(value, kind)| (*value, (*kind).clone()))
            .collect();
        let count = Count::of(cfg, source)?;
        Some(Empty {
            header: loop_.header,
            count,
            exit,
            exit_trip,
            exit_args: exit_args.to_vec(),
            checks,
            defs,
        })
    }
}

/// How a value of the body moves with the counter: it gains `coef`, or
/// `coef·per` where `per` names an invariant value, each time the counter
/// gains one, modulo `2^w` at its width `w`. `coef` is kept as that width
/// reads it, and the value that gains nothing is `FLAT`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Slope {
    coef: i128,
    per: Option<ValueId>,
}

impl Slope {
    const FLAT: Slope = Slope { coef: 0, per: None };

    fn at(width: IntTy, coef: i128, per: Option<ValueId>) -> Slope {
        match width.read(coef as u64) {
            0 => Slope::FLAT,
            coef => Slope { coef, per },
        }
    }

    fn is_flat(self) -> bool {
        self == Slope::FLAT
    }
}

/// The body's instructions, read as how each value moves with the counter.
/// The body qualifies only when it holds nothing but constants, integer
/// casts, wrapping `+`, `-` and `*`, and `Check`s of a `+`: none of them
/// writes, calls, or ends the run but the checks.
struct Slopes<'a> {
    cfg: &'a CfgBody,
    counter: ValueId,
    defs: FxHashMap<ValueId, &'a InstKind>,
    params: Vec<ValueId>,
    checks: Vec<(Span, ValueId, ValueId)>,
}

impl<'a> Slopes<'a> {
    fn of(
        cfg: &'a CfgBody,
        loop_: &NaturalLoop,
        source: ForSource,
        body: &[BlockIdx],
    ) -> Option<Slopes<'a>> {
        let counter = cfg.blocks[for_body(cfg, loop_.header).0].params[source.counter_param()];
        let mut defs = FxHashMap::default();
        let mut params = Vec::new();
        let mut checks = Vec::new();
        for block in body {
            let held = &cfg.blocks[block.0];
            params.extend(held.params.iter().copied());
            for inst in &held.insts {
                match &inst.kind {
                    InstKind::Check {
                        op: Checked::Add,
                        left,
                        right,
                    } => checks.push((inst.span, *left, *right)),
                    InstKind::Const { dst, .. } => {
                        defs.insert(*dst, &inst.kind);
                    }
                    InstKind::Cast {
                        dst,
                        to: CastTy::Int(_),
                        ..
                    } if matches!(cfg.val_types[dst], Ty::Int(_)) => {
                        defs.insert(*dst, &inst.kind);
                    }
                    InstKind::BinOp {
                        dst,
                        op:
                            BinOp::Add(Overflow::Wrap)
                            | BinOp::Sub(Overflow::Wrap)
                            | BinOp::Mul(Overflow::Wrap),
                        ..
                    } if matches!(cfg.val_types[dst], Ty::Int(_)) => {
                        defs.insert(*dst, &inst.kind);
                    }
                    _ => return None,
                }
            }
        }
        Some(Slopes {
            cfg,
            counter,
            defs,
            params,
            checks,
        })
    }

    fn width(&self, value: ValueId) -> IntTy {
        let Ty::Int(width) = self.cfg.val_types[&value] else {
            panic!("{value:?} is a value `Slopes::of` admitted as an integer")
        };
        width
    }

    /// The value of a constant the body or the function defines.
    fn constant(&self, value: ValueId) -> Option<i128> {
        let kind = self.defs.get(&value).copied().or_else(|| {
            self.cfg
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .map(|inst| &inst.kind)
                .find(|kind| matches!(kind, InstKind::Const { dst, .. } if *dst == value))
        })?;
        let InstKind::Const { value: literal, .. } = kind else {
            return None;
        };
        match literal.desugared() {
            Literal::Int(n) => Some(self.width(value).read(n as u64)),
            _ => None,
        }
    }

    /// `None` where the value is not affine in the counter by the rules
    /// below, or is a parameter of a body block other than the counter.
    fn slope(&self, value: ValueId) -> Option<Slope> {
        if value == self.counter {
            return Some(Slope::at(self.width(value), 1, None));
        }
        if self.params.contains(&value) {
            return None;
        }
        let Some(kind) = self.defs.get(&value) else {
            return Some(Slope::FLAT);
        };
        match kind {
            InstKind::Const { .. } => Some(Slope::FLAT),
            InstKind::Cast { src, .. } => {
                if *src == self.counter {
                    // The counter is an integer of its width, so its cast
                    // is that integer modulo `2^w`.
                    return Some(Slope::at(self.width(value), 1, None));
                }
                self.slope(*src)?.is_flat().then_some(Slope::FLAT)
            }
            InstKind::BinOp {
                op, left, right, ..
            } => {
                let width = self.width(value);
                let (l, r) = (self.slope(*left)?, self.slope(*right)?);
                match op {
                    BinOp::Add(_) | BinOp::Sub(_) => {
                        let r = match op {
                            BinOp::Sub(_) => Slope {
                                coef: r.coef.wrapping_neg(),
                                per: r.per,
                            },
                            _ => r,
                        };
                        match (l.is_flat(), r.is_flat()) {
                            (_, true) => Some(l),
                            (true, false) => Some(Slope::at(width, r.coef, r.per)),
                            (false, false) if l.per == r.per => {
                                Some(Slope::at(width, l.coef.wrapping_add(r.coef), l.per))
                            }
                            (false, false) => None,
                        }
                    }
                    BinOp::Mul(_) => {
                        let (moving, factor) = match (l.is_flat(), r.is_flat()) {
                            (true, true) => return Some(Slope::FLAT),
                            (false, true) => (l, *right),
                            (true, false) => (r, *left),
                            (false, false) => return None,
                        };
                        match (self.constant(factor), moving.per) {
                            (Some(c), per) => {
                                Some(Slope::at(width, moving.coef.wrapping_mul(c), per))
                            }
                            (None, None) => Some(Slope::at(width, moving.coef, Some(factor))),
                            (None, Some(_)) => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Every check, where each one's `left` advances by its `right`; `None`
    /// where one does not, and the loop stays.
    fn checks(&self) -> Option<Vec<Stepped>> {
        self.checks
            .iter()
            .map(|&(span, left, right)| {
                let width = self.width(left);
                let advances = self.slope(left)?;
                if !self.slope(right)?.is_flat() {
                    return None;
                }
                let step = match self.constant(right) {
                    Some(c) => Slope::at(width, c, None),
                    None => Slope::at(width, 1, Some(right)),
                };
                (!advances.is_flat() && advances == step).then_some(Stepped { span, left, right })
            })
            .collect()
    }
}

impl Count {
    fn of(cfg: &CfgBody, source: ForSource) -> Option<Count> {
        match source {
            ForSource::Range { at, hi } => {
                let Ty::Int(width) = cfg.val_types[&at] else {
                    panic!("a range's bound {at:?} is an integer (RFC-0057 rule 1)")
                };
                Some(Count::Range { at, hi, width })
            }
            ForSource::Array(array) => {
                let Ty::Array(_, len) = &cfg.val_types[&array] else {
                    panic!("an array source {array:?} is an `Array`")
                };
                let len = match len {
                    LenTerm::Known(len) => *len,
                    LenTerm::Var(never) => match *never {},
                };
                let len = u64::try_from(len).expect("an array's length is a `u64`");
                Some(Count::Array { len })
            }
            ForSource::Slice(_) | ForSource::SliceMut(_) => None,
        }
    }
}

fn remove(cfg: &mut CfgBody, empty: Empty) {
    let Empty {
        header,
        count,
        exit,
        exit_trip,
        exit_args,
        checks,
        defs,
    } = empty;
    let first = first_counter(cfg, header, &count);
    let mut tail = HeaderTail { cfg, header };
    let trip = match (exit_trip, checks.is_empty()) {
        (ExitTrip::Absent, true) => None,
        (ExitTrip::Defined, _) | (ExitTrip::Absent, false) => Some(tail.trip(count)),
    };
    if let Some(count) = trip {
        let counter = tail.counter();
        let mut first_iteration = Clones {
            tail: &mut tail,
            defs: &defs,
            counter,
            first,
            cloned: FxHashMap::default(),
        };
        let steps: Vec<(Span, ValueId, ValueId)> = checks
            .iter()
            .map(|check| {
                (
                    check.span,
                    first_iteration.at_first(check.left),
                    first_iteration.at_first(check.right),
                )
            })
            .collect();
        for (span, from, step) in steps {
            tail.push_at(span, InstKind::CheckSteps { from, step, count });
        }
    }
    let args = match exit_trip {
        ExitTrip::Absent => exit_args,
        ExitTrip::Defined => trip.into_iter().chain(exit_args).collect(),
    };
    cfg.blocks[header.0].terminator = Terminator::Jump { label: exit, args };
}

/// The counter's value on the first iteration: a range's `at`, an array's
/// index `0`.
fn first_counter(cfg: &mut CfgBody, header: BlockIdx, count: &Count) -> ValueId {
    match count {
        Count::Range { at, .. } => *at,
        Count::Array { .. } => {
            let mut tail = HeaderTail { cfg, header };
            let counter = tail.counter();
            let ty = tail.cfg.val_types[&counter].clone();
            let zero = tail.fresh(ty);
            tail.push(InstKind::Const {
                dst: zero,
                value: Literal::Int(0),
            });
            zero
        }
    }
}

/// The body's computation of a value, copied to the header's end with the
/// counter at its first value.
struct Clones<'t, 'a> {
    tail: &'t mut HeaderTail<'a>,
    defs: &'t FxHashMap<ValueId, InstKind>,
    counter: ValueId,
    first: ValueId,
    cloned: FxHashMap<ValueId, ValueId>,
}

impl Clones<'_, '_> {
    fn at_first(&mut self, value: ValueId) -> ValueId {
        if value == self.counter {
            return self.first;
        }
        if let Some(cloned) = self.cloned.get(&value) {
            return *cloned;
        }
        let Some(kind) = self.defs.get(&value).cloned() else {
            return value;
        };
        let dst = self.tail.fresh(self.tail.cfg.val_types[&value].clone());
        let kind = match kind {
            InstKind::Const { value: literal, .. } => InstKind::Const {
                dst,
                value: literal,
            },
            InstKind::Cast { src, to, .. } => InstKind::Cast {
                dst,
                src: self.at_first(src),
                to,
            },
            InstKind::BinOp {
                op, left, right, ..
            } => InstKind::BinOp {
                dst,
                op,
                left: self.at_first(left),
                right: self.at_first(right),
            },
            other => {
                panic!("`Slopes::of` admitted only constants, casts and arithmetic, not {other:?}")
            }
        };
        self.tail.push(kind);
        self.cloned.insert(value, dst);
        dst
    }
}

struct HeaderTail<'a> {
    cfg: &'a mut CfgBody,
    header: BlockIdx,
}

impl HeaderTail<'_> {
    fn fresh(&mut self, ty: Ty) -> ValueId {
        let value = self.cfg.val_factory.next();
        let previous = self.cfg.val_types.insert(value, ty);
        assert!(
            previous.is_none(),
            "{value:?} is fresh from the factory and already carried a type"
        );
        self.cfg.debug.set(value, ValOrigin::Expr);
        value
    }

    fn push(&mut self, kind: InstKind) {
        self.push_at(Span::ZERO, kind);
    }

    fn push_at(&mut self, span: Span, kind: InstKind) {
        self.cfg.blocks[self.header.0]
            .insts
            .push(Inst { span, kind });
    }

    fn counter(&self) -> ValueId {
        let Terminator::For { source, .. } = &self.cfg.blocks[self.header.0].terminator else {
            panic!(
                "block {} is a `for` header until its removal",
                self.header.0
            )
        };
        self.cfg.blocks[for_body(self.cfg, self.header).0].params[source.counter_param()]
    }

    fn as_u64(&mut self, value: ValueId) -> ValueId {
        if self.cfg.val_types[&value] == Ty::U64 {
            return value;
        }
        let dst = self.fresh(Ty::U64);
        self.push(InstKind::Cast {
            dst,
            src: value,
            to: CastTy::Int(IntTy::U64),
        });
        dst
    }

    fn trip(&mut self, count: Count) -> ValueId {
        match count {
            Count::Range { at, hi, width } => {
                let reached = self.fresh(Ty::Int(width));
                self.push(InstKind::BinOp {
                    dst: reached,
                    op: BinOp::Max,
                    left: hi,
                    right: at,
                });
                let (reached, at) = (self.as_u64(reached), self.as_u64(at));
                let trip = self.fresh(Ty::U64);
                self.push(InstKind::BinOp {
                    dst: trip,
                    op: BinOp::Sub(Overflow::Wrap),
                    left: reached,
                    right: at,
                });
                trip
            }
            Count::Array { len } => {
                let trip = self.fresh(Ty::U64);
                self.push(InstKind::Const {
                    dst: trip,
                    value: Literal::Int(i128::from(len)),
                });
                trip
            }
        }
    }
}
