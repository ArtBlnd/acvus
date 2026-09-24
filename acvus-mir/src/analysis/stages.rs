//! Which blocks each stage of a `For` holds (RFC-0089 rule 1): what the
//! stage's entry reaches inside the loop before the next stage's entry or
//! the header.

use rustc_hash::FxHashSet;

use crate::analysis::loops::NaturalLoop;
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::Stages;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeFault {
    EntryNamesNoBlock,
    BodyParams,
    StageEntryEnteredElsewhere,
    StagesOverlap,
    StageDoesNotReachNext,
    EverythingInAChain,
    CarriedTargetIsNoHeaderParam,
}

impl ShapeFault {
    pub fn shown(self) -> &'static str {
        match self {
            Self::EntryNamesNoBlock => "a stage's entry names no block",
            Self::BodyParams => "its body block's parameters are not the element and the counter",
            Self::StageEntryEnteredElsewhere => {
                "a stage's entry is entered other than from the stage before it"
            }
            Self::StagesOverlap => "a block lies in two stages",
            Self::StageDoesNotReachNext => {
                "a stage does not end in one jump to the next stage's entry"
            }
            Self::EverythingInAChain => {
                "a join over everything the body touches is not its one `InOrder` stage"
            }
            Self::CarriedTargetIsNoHeaderParam => "a carried target is not a header parameter",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageBlocks {
    pub blocks: Vec<BlockIdx>,
    ends_toward_next_or_header: Vec<BlockIdx>,
}

impl StageBlocks {
    pub fn sole_end(&self) -> Option<BlockIdx> {
        match self.ends_toward_next_or_header[..] {
            [end] => Some(end),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMembership {
    stages: Vec<StageBlocks>,
}

impl StageMembership {
    pub fn of(
        cfg: &CfgBody,
        header: BlockIdx,
        stages: &Stages,
        loop_blocks: &[BlockIdx],
    ) -> Result<StageMembership, ShapeFault> {
        let entries: Vec<BlockIdx> = stages
            .entries()
            .map(|entry| {
                cfg.label_to_block
                    .get(&entry)
                    .copied()
                    .ok_or(ShapeFault::EntryNamesNoBlock)
            })
            .collect::<Result<_, _>>()?;
        let preds = cfg.predecessors();
        let inside = |block: BlockIdx| loop_blocks.contains(&block);
        let mut claimed: FxHashSet<BlockIdx> = FxHashSet::default();
        let mut found = Vec::with_capacity(entries.len());
        for (index, &entry) in entries.iter().enumerate() {
            let next = entries.get(index + 1).copied();
            let mut blocks = vec![entry];
            let mut work = vec![entry];
            let mut seen: FxHashSet<BlockIdx> = FxHashSet::from_iter([entry]);
            let mut ends: Vec<BlockIdx> = Vec::new();
            while let Some(block) = work.pop() {
                for succ in cfg.successors(block) {
                    if succ == next.unwrap_or(header) {
                        if !ends.contains(&block) {
                            ends.push(block);
                        }
                        continue;
                    }
                    let within = inside(succ) && succ != header;
                    if !within || entries.contains(&succ) || !seen.insert(succ) {
                        continue;
                    }
                    blocks.push(succ);
                    work.push(succ);
                }
            }
            if let Some(next) = next {
                let [last] = ends[..] else {
                    return Err(ShapeFault::StageDoesNotReachNext);
                };
                let jumps_on = matches!(&cfg.blocks[last.0].terminator,
                    Terminator::Jump { args, .. } if args.is_empty());
                if !jumps_on {
                    return Err(ShapeFault::StageDoesNotReachNext);
                }
                let entered_alone =
                    preds.get(&next).map(|from| from.as_slice()) == Some(&[last][..]);
                if !entered_alone || !cfg.blocks[next.0].params.is_empty() {
                    return Err(ShapeFault::StageEntryEnteredElsewhere);
                }
            }
            for block in &blocks {
                if !claimed.insert(*block) {
                    return Err(ShapeFault::StagesOverlap);
                }
            }
            found.push(StageBlocks {
                blocks,
                ends_toward_next_or_header: ends,
            });
        }
        if preds.get(&entries[0]).map(|from| from.as_slice()) != Some(&[header][..]) {
            return Err(ShapeFault::StageEntryEnteredElsewhere);
        }
        Ok(StageMembership { stages: found })
    }

    pub fn stages(&self) -> &[StageBlocks] {
        &self.stages
    }

    pub fn stage_of(&self, block: BlockIdx) -> Option<usize> {
        self.stages
            .iter()
            .position(|stage| stage.blocks.contains(&block))
    }
}

/// The blocks of the loop `header` heads. A header no back edge reaches is
/// a loop of its own block alone: its body never runs twice.
pub fn loop_blocks_of(loops: &[NaturalLoop], header: BlockIdx) -> Vec<BlockIdx> {
    match loops.iter().find(|loop_| loop_.header == header) {
        Some(loop_) => loop_.blocks().collect(),
        None => vec![header],
    }
}
