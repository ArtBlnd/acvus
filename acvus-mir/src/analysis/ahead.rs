//! Which lowering a loop takes (RFC-0092): in place, or RFC-0103's first
//! shape, which issues the first stage's heavy or io spawns of later
//! iterations ahead and runs the rest of each iteration in place, in index
//! order. The program's MIR is not changed: `prepare` builds the run from
//! the [`Plan`].

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info::{defs, terminator_uses, uses};
use crate::analysis::interval;
use crate::analysis::loop_deps::{
    Control, Head, HeaderDeps, InstAt, LoopDeps, Member, ShapeFault, StageBlocks, has_effect,
};
use crate::analysis::loops::{NaturalLoop, natural_loops_innermost_first};
use crate::analysis::raise::{FunctionSummary, Removal};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, ForSource, InstKind, ValueId};
use crate::laws::LawTable;
use crate::ty::Task;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lowering {
    InPlace(Refused),
    Ahead(Plan),
}

/// The first condition of RFC-0103 rule 1 a loop fails, in the rule's
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    Enclosed { by: BlockIdx },
    Source(RefusedSource),
    Unstaged(ShapeFault),
    FirstStageNotFree,
    CycleCrosses,
    NoAheadSpawn,
    /// A prefix instruction that may trap or may not finish
    /// (`raise::Removal::stays_unused`).
    MayTrap(InstAt),
    ReadsLoopValue { member: Member, value: ValueId },
    EffectBeforeExit(InstAt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusedSource {
    SliceMut,
    Array,
    Pull,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub body: BlockIdx,
    pub laid: HeaderLays,
    /// The instructions and terminators the prefix passes, in run order,
    /// each arm of a region in turn.
    pub prefix: Vec<Member>,
    pub rest: RestStart,
    pub spawns: Vec<AheadSpawn>,
    /// Every register the prefix defines, the storage slots it assigns and
    /// the parameters of the blocks it enters included, and `laid` not:
    /// what `prepare` holds dead at `body` (RFC-0103 rule 4).
    pub written: Vec<ValueId>,
    pub task: Task,
}

/// A range's element is its counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderLays {
    pub element: ValueId,
    pub counter: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestStart {
    Inst(InstAt),
    Terminator(BlockIdx),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AheadSpawn {
    pub at: InstAt,
    pub handle: ValueId,
    pub task: Task,
}

/// Decides each loop of one body. A prefix instruction is asked of
/// `raise` knowing no callee, as run-ahead is (RFC-0089 rule 5): the plan
/// is the body's own.
pub struct Lowerer<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    loops: Vec<NaturalLoop>,
}

impl<'a> Lowerer<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &'a LawTable) -> Self {
        Self {
            cfg,
            laws,
            loops: natural_loops_innermost_first(cfg, &DomTree::build(cfg)),
        }
    }

    pub fn lowering(&self, found: &HeaderDeps) -> Lowering {
        match self.plan(found.header, found.deps.as_ref()) {
            Ok(plan) => Lowering::Ahead(plan),
            Err(refused) => Lowering::InPlace(refused),
        }
    }

    fn plan(&self, header: BlockIdx, deps: Result<&LoopDeps, &ShapeFault>) -> Result<Plan, Refused> {
        let cfg = self.cfg;
        if let Some(enclosing) = self
            .loops
            .iter()
            .find(|loop_| loop_.header != header && loop_.contains(header))
        {
            return Err(Refused::Enclosed {
                by: enclosing.header,
            });
        }
        let Some((head, _)) = Head::of(&cfg.blocks[header.0].terminator) else {
            panic!("block {} heads no `For` and no `While`", header.0)
        };
        let source = match head {
            Head::For(source @ (ForSource::Slice(_) | ForSource::Range { .. })) => source,
            Head::For(ForSource::SliceMut(_)) => {
                return Err(Refused::Source(RefusedSource::SliceMut));
            }
            Head::For(ForSource::Array(_)) => return Err(Refused::Source(RefusedSource::Array)),
            Head::Pull => return Err(Refused::Source(RefusedSource::Pull)),
        };
        let deps = deps.map_err(|fault| Refused::Unstaged(*fault))?;
        let first = deps.membership.body_stages().start;
        if !deps.is_free(first) {
            return Err(Refused::FirstStageNotFree);
        }
        if deps.crossing().next().is_some() {
            return Err(Refused::CycleCrosses);
        }

        let stage = &deps.membership.stages()[first];
        let units = straight_run(cfg, stage);
        let Some(last) = units
            .iter()
            .rposition(|step| step.insts().any(|at| self.ahead_spawn(at).is_some()))
        else {
            return Err(Refused::NoAheadSpawn);
        };
        let prefix = &units[..=last];
        let rest = match &prefix[last] {
            RunStep::Inst(at) if at.at + 1 < cfg.blocks[at.block.0].insts.len() => {
                RestStart::Inst(InstAt {
                    block: at.block,
                    at: at.at + 1,
                })
            }
            RunStep::Inst(at) => RestStart::Terminator(at.block),
            RunStep::Region(Region { join: to, .. }) | RunStep::Edge(Edge { to, .. }) => {
                match cfg.blocks[to.0].insts.is_empty() {
                    true => RestStart::Terminator(*to),
                    false => RestStart::Inst(InstAt { block: *to, at: 0 }),
                }
            }
        };

        let members: Vec<Member> = prefix.iter().flat_map(RunStep::members).collect();
        let insts = || {
            members.iter().filter_map(|member| match member {
                Member::Inst(at) => Some(*at),
                Member::Term(_) => None,
            })
        };
        let kind = |at: InstAt| &cfg.blocks[at.block.0].insts[at.at].kind;

        let spawns: Vec<AheadSpawn> = insts().filter_map(|at| self.ahead_spawn(at)).collect();

        let summary = FunctionSummary::unknown();
        let removal = Removal::of(cfg, self.laws, &summary);
        let may_trap = |at: InstAt| {
            let at = interval::InstAt {
                block: at.block,
                at: at.at,
            };
            removal.stays_unused(at, &cfg.blocks[at.block.0].insts[at.at].kind)
        };
        if let Some(at) = insts()
            .filter(|at| !spawns.iter().any(|spawn| spawn.at == *at))
            .find(|at| may_trap(*at))
        {
            return Err(Refused::MayTrap(at));
        }

        let body = stage.entry_block;
        let params = &cfg.blocks[body.0].params;
        let laid = HeaderLays {
            element: params[0],
            counter: params[source.counter_param()],
        };
        let written: Vec<ValueId> = insts()
            .flat_map(|at| defs(kind(at)))
            .chain(prefix.iter().flat_map(|unit| unit.entered(cfg)))
            .collect();
        let defined_in_loop: FxHashSet<ValueId> = self
            .loops
            .iter()
            .filter(|loop_| loop_.header == header)
            .flat_map(NaturalLoop::blocks)
            .chain([header])
            .flat_map(|block| {
                let block = &cfg.blocks[block.0];
                block
                    .params
                    .iter()
                    .copied()
                    .chain(block.insts.iter().flat_map(|inst| defs(&inst.kind)))
            })
            .collect();
        let own = |value: &ValueId| written.contains(value) || params.contains(value);
        for &member in &members {
            let read = match member {
                Member::Inst(at) => uses(kind(at)),
                Member::Term(block) => terminator_uses(&cfg.blocks[block.0].terminator),
            };
            if let Some(value) = read
                .into_iter()
                .find(|value| defined_in_loop.contains(value) && !own(value))
            {
                return Err(Refused::ReadsLoopValue { member, value });
            }
        }

        if let Control::Chained { .. } = deps.control
            && let Some(at) = insts().find(|at| has_effect(kind(*at)))
        {
            return Err(Refused::EffectBeforeExit(at));
        }

        let task = spawns
            .iter()
            .map(|spawn| spawn.task)
            .fold(Task::Async, Task::join);
        Ok(Plan {
            body,
            laid,
            prefix: members,
            rest,
            spawns,
            written,
            task,
        })
    }

    fn ahead_spawn(&self, at: InstAt) -> Option<AheadSpawn> {
        let InstKind::Spawn {
            dst,
            callee: callee @ Callee::Extern { .. },
            ..
        } = &self.cfg.blocks[at.block.0].insts[at.at].kind
        else {
            return None;
        };
        let task = self.laws.task_of(callee).filter(|task| *task > Task::Sync)?;
        Some(AheadSpawn {
            at,
            handle: *dst,
            task,
        })
    }
}

/// A callee type that is no function type states no task, and is taken
/// to suspend.
fn suspends(kind: &InstKind) -> bool {
    match kind {
        InstKind::Eval { .. } => true,
        InstKind::FunctionCall { callee_ty, .. } => {
            callee_ty.effect().is_none_or(|effect| effect.task > Task::Sync)
        }
        _ => false,
    }
}

#[derive(Debug)]
enum RunStep {
    Inst(InstAt),
    Edge(Edge),
    Region(Region),
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    from: BlockIdx,
    to: BlockIdx,
}

impl Edge {
    fn sole_entry(
        cfg: &CfgBody,
        preds: &FxHashMap<BlockIdx, SmallVec<[BlockIdx; 2]>>,
        stage: &StageBlocks,
        from: BlockIdx,
    ) -> Option<Edge> {
        let Terminator::Jump { label, .. } = &cfg.blocks[from.0].terminator else {
            return None;
        };
        let to = *cfg.label_to_block.get(label)?;
        let entered_alone = preds.get(&to).map(|p| p.as_slice()) == Some(&[from][..]);
        (stage.blocks.contains(&to) && entered_alone).then_some(Edge { from, to })
    }
}

#[derive(Debug)]
struct Region {
    branch: BlockIdx,
    members: Vec<Member>,
    entered_blocks: Vec<BlockIdx>,
    join: BlockIdx,
}

impl RunStep {
    fn members(&self) -> Vec<Member> {
        match self {
            RunStep::Inst(at) => vec![Member::Inst(*at)],
            RunStep::Edge(edge) => vec![Member::Term(edge.from)],
            RunStep::Region(region) => std::iter::once(Member::Term(region.branch))
                .chain(region.members.iter().copied())
                .collect(),
        }
    }

    fn insts(&self) -> impl Iterator<Item = InstAt> {
        self.members()
            .into_iter()
            .filter_map(|member| match member {
                Member::Inst(at) => Some(at),
                Member::Term(_) => None,
            })
    }

    fn entered<'c>(&self, cfg: &'c CfgBody) -> impl Iterator<Item = ValueId> + 'c {
        let blocks: Vec<BlockIdx> = match self {
            RunStep::Inst(_) => Vec::new(),
            RunStep::Edge(edge) => vec![edge.to],
            RunStep::Region(region) => region.entered_blocks.clone(),
        };
        blocks
            .into_iter()
            .flat_map(move |block| cfg.blocks[block.0].params.iter().copied())
    }
}

fn straight_run(cfg: &CfgBody, stage: &StageBlocks) -> Vec<RunStep> {
    let preds = cfg.predecessors();
    let mut units = Vec::new();
    let mut block = stage.entry_block;
    loop {
        for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
            if suspends(&inst.kind) {
                return units;
            }
            units.push(RunStep::Inst(InstAt { block, at }));
        }
        let next = match &cfg.blocks[block.0].terminator {
            Terminator::Diamond { .. } => match Region::rejoining(cfg, stage, block) {
                Some(region) => {
                    let join = region.join;
                    units.push(RunStep::Region(region));
                    join
                }
                None => return units,
            },
            Terminator::Jump { .. } => match Edge::sole_entry(cfg, &preds, stage, block) {
                Some(edge) => {
                    units.push(RunStep::Edge(edge));
                    edge.to
                }
                None => return units,
            },
            _ => return units,
        };
        block = next;
    }
}

impl Region {
    fn rejoining(cfg: &CfgBody, stage: &StageBlocks, branch: BlockIdx) -> Option<Region> {
        acvus_utils::grow(|| Self::rejoining_level(cfg, stage, branch))
    }

    fn rejoining_level(cfg: &CfgBody, stage: &StageBlocks, branch: BlockIdx) -> Option<Region> {
        let Terminator::Diamond {
            then_label,
            else_label,
            join,
            ..
        } = &cfg.blocks[branch.0].terminator
        else {
            return None;
        };
        let join = *cfg.label_to_block.get(join)?;
        if !stage.blocks.contains(&join) {
            return None;
        }
        let mut members = Vec::new();
        let mut entered = Vec::new();
        for arm in [then_label, else_label] {
            let mut block = *cfg.label_to_block.get(arm)?;
            while block != join {
                if !stage.blocks.contains(&block) || entered.contains(&block) {
                    return None;
                }
                entered.push(block);
                for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
                    if suspends(&inst.kind) {
                        return None;
                    }
                    members.push(Member::Inst(InstAt { block, at }));
                }
                block = match &cfg.blocks[block.0].terminator {
                    Terminator::Jump { label, .. } => {
                        members.push(Member::Term(block));
                        *cfg.label_to_block.get(label)?
                    }
                    Terminator::Diamond { .. } => {
                        let inner = Region::rejoining(cfg, stage, block)?;
                        members.push(Member::Term(inner.branch));
                        members.extend(inner.members);
                        entered.extend(inner.entered_blocks);
                        inner.join
                    }
                    _ => return None,
                };
            }
        }
        entered.push(join);
        Some(Region {
            branch,
            members,
            entered_blocks: entered,
            join,
        })
    }
}
