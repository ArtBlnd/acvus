//! A `for` whose body does nothing is a jump to its exit (RFC-0088).
//!
//! A range's trip count is `(max(hi, at) as u64) − (at as u64)` and not
//! `max(hi − at, 0)` at the range's width `w`, because the second wraps: at
//! `i8`, `-100..100` has `hi − at` wrap to `-56`. The `max` compares at `w`'s
//! own signedness, so `max(hi, at) − at` is the count over the integers,
//! which lies in `[0, 2^64)`. An integer `Cast` keeps its operand modulo
//! `2^64` and a `u64` subtraction wraps modulo `2^64` (RFC-0037), so the
//! difference of the two casts is that count exactly.
//!
//! A slice is declined, not an omission: the MIR has no instruction that
//! reads a slice's length, and one is not added for this pass.

use acvus_ast::{Literal, Span};

use crate::analysis::domtree::DomTree;
use crate::analysis::loops::{NaturalLoop, natural_loops_innermost_first};
use crate::cfg::{BlockIdx, CfgBody, Terminator, prune, reachable};
use crate::ir::{
    BinOp, ExitTrip, ForSource, Inst, InstKind, Label, Traversal, ValOrigin, ValueId,
};
use crate::optimize::dce;
use crate::ty::{CastTy, IntTy, LenTerm, Ty};

pub fn run(cfg: &mut CfgBody) {
    while let Some(empty) = first_empty(cfg) {
        remove(cfg, empty);
        let alive = reachable(cfg);
        prune(cfg, &alive);
        dce::run(cfg);
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
        let Traversal {
            source,
            exit,
            exit_trip,
            exit_args,
            ..
        } = header.terminator.traversal()?;
        if !header.params.is_empty() || !header.insts.is_empty() {
            return None;
        }
        let only_jumps_within = |block: BlockIdx| {
            let held = &cfg.blocks[block.0];
            let Terminator::Jump { label, .. } = &held.terminator else {
                return false;
            };
            held.insts.is_empty()
                && cfg
                    .label_to_block
                    .get(label)
                    .is_some_and(|to| loop_.contains(*to))
        };
        let body_does_nothing = loop_
            .blocks()
            .filter(|block| *block != loop_.header)
            .all(only_jumps_within);
        if !body_does_nothing {
            return None;
        }
        let count = Count::of(cfg, source)?;
        Some(Empty {
            header: loop_.header,
            count,
            exit,
            exit_trip,
            exit_args: exit_args.to_vec(),
        })
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
    } = empty;
    let trip = match exit_trip {
        ExitTrip::Absent => None,
        ExitTrip::Defined => Some(HeaderTail { cfg, header }.trip(count)),
    };
    cfg.blocks[header.0].terminator = Terminator::Jump {
        label: exit,
        args: trip.into_iter().chain(exit_args).collect(),
    };
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
        self.cfg.blocks[self.header.0].insts.push(Inst {
            span: Span::ZERO,
            kind,
        });
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

    fn trip(mut self, count: Count) -> ValueId {
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
                    op: BinOp::Sub,
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
