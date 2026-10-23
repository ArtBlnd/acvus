//! The points of a body a walk reaches along its edges, block by block. A
//! point is before an instruction or at a block's end, keyed within its
//! block by the instruction's order number, the end last and the block's
//! start first. A walk reaches, in each block, at most two runs of points:
//! one from where it starts or enters, and one from where it enters again.

use smallvec::SmallVec;

use crate::ir::cfg::body::InstId;
use crate::ir::cfg::{BlockId, Body};

/// A point's place within its block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct PointKey(u64);

impl PointKey {
    /// The block's start, before every point of it.
    const START: PointKey = PointKey(0);
    /// The block's end, after every instruction and before the terminator.
    pub(super) const END: PointKey = PointKey(u64::MAX);

    /// The point just before `inst`.
    pub(super) fn before(body: &Body, inst: InstId) -> PointKey {
        PointKey(body.order_number(inst))
    }

    /// The point just after `inst`: before the instruction that follows it,
    /// or the block's end.
    pub(super) fn after(body: &Body, inst: InstId) -> PointKey {
        match body.next_in_block(inst) {
            Some(next) => PointKey::before(body, next),
            None => PointKey::END,
        }
    }

    fn just_below(self) -> PointKey {
        PointKey(self.0 - 1)
    }

    fn just_above(self) -> PointKey {
        PointKey(self.0 + 1)
    }
}

/// A point of a body: in `block`, before the instruction `before`, or at
/// the block's end where that is none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BlockPoint {
    pub(super) block: BlockId,
    pub(super) before: Option<InstId>,
}

impl BlockPoint {
    /// The point just after `inst`.
    pub(super) fn after(body: &Body, inst: InstId) -> BlockPoint {
        BlockPoint { block: body.block_of(inst), before: body.next_in_block(inst) }
    }

    /// The first point of `block`.
    pub(super) fn first_of(body: &Body, block: BlockId) -> BlockPoint {
        BlockPoint { block, before: body.first_inst(block) }
    }

    pub(super) fn key(self, body: &Body) -> PointKey {
        match self.before {
            Some(inst) => PointKey::before(body, inst),
            None => PointKey::END,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Direction {
    Forward,
    Backward,
}

/// The points `lo..=hi` of one block, and the first instruction whose point
/// lies in them, none where only the block's end does.
#[derive(Debug, Clone, Copy)]
struct Run {
    lo: PointKey,
    hi: PointKey,
    first: Option<InstId>,
}

impl Run {
    fn holds(self, key: PointKey) -> bool {
        self.lo <= key && key <= self.hi
    }
}

/// The points one walk reached, as runs per block.
pub(super) struct Reached {
    runs: Vec<SmallVec<[Run; 2]>>,
}

impl Reached {
    /// The points reachable from `start` in `direction` without stepping
    /// onto `avoided`; `start` itself is reached. Cost: O(blocks + edges).
    pub(super) fn walk(body: &Body, start: BlockPoint, avoided: Option<BlockPoint>, direction: Direction) -> Reached {
        let count = body.block_count();
        let mut runs: Vec<SmallVec<[Run; 2]>> = vec![SmallVec::new(); count];
        let mut entered = vec![false; count];
        let preds = match direction {
            Direction::Forward => None,
            Direction::Backward => Some(body.predecessors()),
        };
        let avoided_in = |block: BlockId| avoided.filter(|point| point.block == block);
        // The run above an avoided point starts at the instruction after it;
        // an avoided end has none above it.
        let above = |point: BlockPoint| -> Option<InstId> { point.before.and_then(|inst| body.next_in_block(inst)) };
        let start_key = start.key(body);
        let (run, leaves) = match direction {
            Direction::Forward => match avoided_in(start.block).filter(|at| at.key(body) > start_key) {
                Some(at) => (Run { lo: start_key, hi: at.key(body).just_below(), first: start.before }, false),
                None => (Run { lo: start_key, hi: PointKey::END, first: start.before }, true),
            },
            Direction::Backward => match avoided_in(start.block).filter(|at| at.key(body) < start_key) {
                Some(at) => (Run { lo: at.key(body).just_above(), hi: start_key, first: above(at) }, false),
                None => (Run { lo: PointKey::START, hi: start_key, first: body.first_inst(start.block) }, true),
            },
        };
        runs[start.block.index()].push(run);
        let mut pending: Vec<BlockId> = Vec::new();
        if leaves {
            pending.push(start.block);
        }
        while let Some(block) = pending.pop() {
            let next: SmallVec<[BlockId; 2]> = match &preds {
                None => body.block(block).terminator().edges().iter().map(|edge| edge.to).collect(),
                Some(preds) => preds[&block].iter().copied().collect(),
            };
            for into in next {
                if entered[into.index()] {
                    continue;
                }
                entered[into.index()] = true;
                let head = body.first_inst(into);
                let (run, leaves) = match (direction, avoided_in(into)) {
                    (Direction::Forward, Some(at)) if at == BlockPoint::first_of(body, into) => continue,
                    (Direction::Forward, Some(at)) => {
                        (Run { lo: PointKey::START, hi: at.key(body).just_below(), first: head }, false)
                    }
                    (Direction::Forward, None) => (Run { lo: PointKey::START, hi: PointKey::END, first: head }, true),
                    (Direction::Backward, Some(at)) if at.before.is_none() => continue,
                    (Direction::Backward, Some(at)) => {
                        (Run { lo: at.key(body).just_above(), hi: PointKey::END, first: above(at) }, false)
                    }
                    (Direction::Backward, None) => (Run { lo: PointKey::START, hi: PointKey::END, first: head }, true),
                };
                runs[into.index()].push(run);
                if leaves {
                    pending.push(into);
                }
            }
        }
        Reached { runs }
    }

    /// Whether the walk reached the point `key` of `block`. Cost: O(1).
    pub(super) fn holds(&self, block: BlockId, key: PointKey) -> bool {
        self.runs[block.index()].iter().any(|run| run.holds(key))
    }
}

/// The instructions of `block` a way passes: the point before each is
/// reached by `forward`, and the point after it by `backward`. Cost: O(the
/// instructions from the start of each forward run to the last passed).
pub(super) fn passed<'r>(body: &'r Body, block: BlockId, forward: &'r Reached, backward: &'r Reached) -> Vec<InstId> {
    let mut passed = Vec::new();
    for ahead in &forward.runs[block.index()] {
        for behind in &backward.runs[block.index()] {
            let mut at = ahead.first;
            while let Some(inst) = at {
                let (before, after) = (PointKey::before(body, inst), PointKey::after(body, inst));
                if before > ahead.hi || after > behind.hi {
                    break;
                }
                if after >= behind.lo {
                    passed.push(inst);
                }
                at = body.next_in_block(inst);
            }
        }
    }
    passed
}
