//! What each stage of a `For` is (RFC-0089 rules 1, 2 and 4): the one place
//! that computes, from the tokens the IR holds and the operations'
//! declarations, each stage's blocks, the dependence cycles through a
//! token and where each lies, each cycle's order and law, and the loop's
//! control. The terminator states only where the body is cut.
//!
//! A cycle's members are the instructions and terminators of the body's
//! blocks; the header belongs to no stage. Within one iteration a member
//! runs after the members whose values it reads, the branches that decide
//! whether it runs, and the earlier members that touch a storage slot or a
//! context it touches where one of the two writes it.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::AffineValues;
use crate::analysis::carried::{Carried, CarriedState, MergeOp};
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariants, LoopNest, NaturalLoop, natural_loops_innermost_first};
use crate::analysis::targets::{TargetSlots, Written, effect, slots_lent_mutably, touched_slots};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{BinOp, ForSource, InstKind, Label, Stages, ValueId};
use crate::laws::{FoldLaw, LawTable};
use crate::ty::Ty;

// -- Stage membership (rule 1) ----------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeFault {
    EntryNamesNoBlock,
    BodyParams,
    StageEntryEnteredElsewhere,
    StagesOverlap,
    StageDoesNotReachNext,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageBlocks {
    pub entry: Label,
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

/// Which blocks each stage holds: what the stage's entry reaches inside the
/// loop before the next stage's entry or the header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageMembership {
    stages: Vec<StageBlocks>,
}

impl StageMembership {
    pub fn of(
        cfg: &CfgBody,
        header: BlockIdx,
        source: ForSource,
        stages: &Stages,
        loop_blocks: &[BlockIdx],
    ) -> Result<StageMembership, ShapeFault> {
        let labels: Vec<Label> = stages.entries().collect();
        let entries: Vec<BlockIdx> = labels
            .iter()
            .map(|entry| {
                cfg.label_to_block
                    .get(entry)
                    .copied()
                    .ok_or(ShapeFault::EntryNamesNoBlock)
            })
            .collect::<Result<_, _>>()?;
        if cfg.blocks[entries[0].0].params.len() != source.supplied_params() {
            return Err(ShapeFault::BodyParams);
        }
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
                entry: labels[index],
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

// -- Members and tokens (rule 2) ---------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstAt {
    pub block: BlockIdx,
    pub at: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Member {
    Inst(InstAt),
    Term(BlockIdx),
}

impl Member {
    pub fn block(self) -> BlockIdx {
        match self {
            Self::Inst(InstAt { block, .. }) | Self::Term(block) => block,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Token {
    /// A header parameter of type `Order` (RFC-0007).
    Order(ValueId),
    Carried(ValueId),
    Storage(Storage),
    /// The fact that an iteration exists, in a loop the body can leave.
    Control,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Storage {
    Slot(ValueId),
    Context(QualifiedRef),
    /// The element of a `&mut` source, written at its counter's slot.
    Element,
}

impl Token {
    pub fn shown(self) -> String {
        match self {
            Self::Order(param) => format!("the order {param:?}"),
            Self::Carried(param) => format!("the carried value {param:?}"),
            Self::Storage(Storage::Slot(slot)) => format!("the storage {slot:?}"),
            Self::Storage(Storage::Context(context)) => format!("the context {context:?}"),
            Self::Storage(Storage::Element) => "the element".to_string(),
            Self::Control => "the control token".to_string(),
        }
    }

    pub fn shown_all(tokens: &[Token]) -> String {
        tokens
            .iter()
            .map(|token| token.shown())
            .collect::<Vec<_>>()
            .join(" and ")
    }

    pub fn header_param(self) -> Option<ValueId> {
        match self {
            Self::Order(param) | Self::Carried(param) => Some(param),
            Self::Storage(_) | Self::Control => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    Stage(usize),
    /// Rule 3 refuses this: the cycle's members lie in these stages.
    Crosses(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cycle {
    pub tokens: Vec<Token>,
    pub members: Vec<Member>,
    pub placement: Placement,
}

impl Cycle {
    pub fn stage(&self) -> Option<usize> {
        match self.placement {
            Placement::Stage(stage) => Some(stage),
            Placement::Crosses(_) => None,
        }
    }

    fn earliest_stage(&self) -> usize {
        match &self.placement {
            Placement::Stage(stage) => *stage,
            Placement::Crosses(stages) => stages[0],
        }
    }

    fn lies_in(&self, stage: usize) -> bool {
        match &self.placement {
            Placement::Stage(at) => *at == stage,
            Placement::Crosses(stages) => stages.contains(&stage),
        }
    }
}

/// A cycle rule 3 refuses: its tokens, and the stages its members lie in.
pub struct Crossing<'a> {
    pub tokens: &'a [Token],
    pub stages: &'a [usize],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Control {
    /// Every exit is the header's: every iteration holds the token from
    /// its start.
    Upfront,
    /// The iteration receives the token from the exits of the one before,
    /// which cycle `cycle` of [`LoopDeps::cycles`] holds.
    Chained { cycle: usize },
}

/// A member that reads a header parameter's state in a stage before the
/// one its cycle lies in, so that stage waits on a later stage of the
/// iteration before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EarlyRead {
    pub token: Token,
    pub member: Member,
    pub stage: usize,
    pub cycle_stage: usize,
}

/// An operation with an effect in a free stage. Rule 5 lets it run only
/// once its iteration holds the control token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectWait {
    pub member: Member,
    pub stage: usize,
    pub control: Control,
}

#[derive(Debug, Clone)]
pub struct LoopDeps {
    pub header: BlockIdx,
    pub membership: StageMembership,
    pub cycles: Vec<Cycle>,
    pub control: Control,
    early_reads: Vec<EarlyRead>,
    effects: Vec<EffectWait>,
}

pub struct HeaderDeps {
    pub header: BlockIdx,
    pub deps: Result<LoopDeps, ShapeFault>,
}

pub struct BodyDeps {
    pub loops: Vec<HeaderDeps>,
}

impl BodyDeps {
    pub fn of(cfg: &CfgBody) -> Self {
        let loans = Loans::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        let found = (0..cfg.blocks.len())
            .map(BlockIdx)
            .filter(|at| matches!(cfg.blocks[at.0].terminator, Terminator::For { .. }))
            .map(|header| HeaderDeps {
                header,
                deps: LoopDeps::with(cfg, &loans, header, &loop_blocks_of(&loops, header)),
            })
            .collect();
        Self { loops: found }
    }
}

impl LoopDeps {
    /// # Panics
    /// If `header` does not end in `For`.
    pub fn of(cfg: &CfgBody, header: BlockIdx) -> Result<LoopDeps, ShapeFault> {
        let loans = Loans::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        LoopDeps::with(cfg, &loans, header, &loop_blocks_of(&loops, header))
    }

    fn with(
        cfg: &CfgBody,
        loans: &Loans<'_>,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
    ) -> Result<LoopDeps, ShapeFault> {
        let Terminator::For { source, stages, .. } = &cfg.blocks[header.0].terminator else {
            panic!("block {} heads no `For`", header.0)
        };
        let membership = StageMembership::of(cfg, header, *source, stages, loop_blocks)?;
        let graph = Graph::of(cfg, loans, header, loop_blocks, stages.body());
        let slots = TargetSlots::of(loans, *source, loop_blocks);
        let stage_of = |member: Member| {
            membership
                .stage_of(member.block())
                .expect("a chain whose shape holds puts every block of the body in a stage")
        };
        let mut cycles: Vec<Cycle> = graph
            .cycles(cfg, loans, &slots, header, loop_blocks)
            .into_iter()
            .map(|found| place(found, &stage_of))
            .collect();

        // RFC-0066 rule 10: the control token passes with the exiting
        // stage's tokens, so the cycles lying there are one join with it.
        let exiting = cycles
            .iter()
            .find(|cycle| cycle.tokens.contains(&Token::Control))
            .and_then(Cycle::stage);
        if let Some(stage) = exiting {
            let (joined, kept): (Vec<Cycle>, Vec<Cycle>) = cycles
                .into_iter()
                .partition(|cycle| cycle.stage() == Some(stage));
            let mut one = Found::default();
            for cycle in joined {
                one.tokens.extend(cycle.tokens);
                one.members.extend(cycle.members);
            }
            cycles = kept;
            cycles.push(place(one, &stage_of));
        }
        cycles.sort_by_key(|cycle| (cycle.earliest_stage(), cycle.members[0]));
        let control = match cycles
            .iter()
            .position(|cycle| cycle.tokens.contains(&Token::Control))
        {
            Some(cycle) => Control::Chained { cycle },
            None => Control::Upfront,
        };

        let mut early_reads: Vec<EarlyRead> = Vec::new();
        for cycle in &cycles {
            let cycle_stage = cycle.earliest_stage();
            for &token in &cycle.tokens {
                let Some(param) = token.header_param() else {
                    continue;
                };
                for &member in graph.readers(param) {
                    let stage = stage_of(member);
                    if stage < cycle_stage {
                        early_reads.push(EarlyRead {
                            token,
                            member,
                            stage,
                            cycle_stage,
                        });
                    }
                }
            }
        }

        let effects = graph
            .members
            .iter()
            .copied()
            .filter(|member| {
                let stage = stage_of(*member);
                !cycles.iter().any(|cycle| cycle.lies_in(stage))
            })
            .filter(|member| match member {
                Member::Inst(at) => has_effect(&cfg.blocks[at.block.0].insts[at.at].kind),
                Member::Term(_) => false,
            })
            .map(|member| EffectWait {
                member,
                stage: stage_of(member),
                control,
            })
            .collect();

        Ok(LoopDeps {
            header,
            membership,
            cycles,
            control,
            early_reads,
            effects,
        })
    }

    pub fn cycles_in(&self, stage: usize) -> impl Iterator<Item = &Cycle> + '_ {
        self.cycles
            .iter()
            .filter(move |cycle| cycle.stage() == Some(stage))
    }

    pub fn is_free(&self, stage: usize) -> bool {
        !self.cycles.iter().any(|cycle| cycle.lies_in(stage))
    }

    pub fn crossing(&self) -> impl Iterator<Item = Crossing<'_>> + '_ {
        self.cycles
            .iter()
            .filter_map(|cycle| match &cycle.placement {
                Placement::Crosses(stages) => Some(Crossing {
                    tokens: &cycle.tokens,
                    stages,
                }),
                Placement::Stage(_) => None,
            })
    }

    pub fn early_reads(&self) -> &[EarlyRead] {
        &self.early_reads
    }

    pub fn effects_in_free_stages(&self) -> &[EffectWait] {
        &self.effects
    }

    /// Each cycle's order and law, by the index of [`Self::cycles`].
    pub fn judge(&self, cfg: &CfgBody, laws: &LawTable) -> Vec<Judged> {
        let domtree = DomTree::build(cfg);
        let invariants = Invariants::of(cfg);
        let nest = LoopNest::of(cfg, &domtree, &invariants);
        let Some(id) = nest.by_header(self.header) else {
            // No back edge reaches the header: the body runs at most once,
            // so no iteration hands a next one anything to merge.
            return self.cycles.iter().map(|cycle| judge(cycle, None)).collect();
        };
        let loop_ = nest.get(id);
        let loans = Loans::build(cfg);
        let affine = AffineValues::of(cfg, loop_, &invariants);
        let state = CarriedState::of(&loans, loop_, &affine, laws);
        self.cycles
            .iter()
            .map(|cycle| judge(cycle, Some(&state)))
            .collect()
    }
}

#[derive(Default)]
struct Found {
    tokens: Vec<Token>,
    members: Vec<Member>,
}

fn place(found: Found, stage_of: &impl Fn(Member) -> usize) -> Cycle {
    let Found {
        tokens,
        mut members,
    } = found;
    members.sort_unstable();
    members.dedup();
    let mut unique: Vec<Token> = Vec::new();
    for token in tokens {
        if !unique.contains(&token) {
            unique.push(token);
        }
    }
    let mut stages: Vec<usize> = members.iter().map(|member| stage_of(*member)).collect();
    stages.sort_unstable();
    stages.dedup();
    let placement = match stages[..] {
        [stage] => Placement::Stage(stage),
        _ => Placement::Crosses(stages),
    };
    Cycle {
        tokens: unique,
        members,
        placement,
    }
}

/// An operation whose effect keeps its place in the run: one that carries
/// an `Order` (RFC-0013, RFC-0046), or a commit of a context.
pub fn has_effect(kind: &InstKind) -> bool {
    match kind {
        InstKind::FunctionCall { order, .. } => order.is_some(),
        InstKind::Spawn { order, .. } | InstKind::Eval { order, .. } => order.is_some(),
        InstKind::Commit { .. } => true,
        _ => false,
    }
}

// -- Order and law (rule 4) ----------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Disjoint,
    AnyOrder,
    InOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Judged {
    pub order: Order,
    pub law: Option<Accumulator>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Accumulator {
    pub law: Law,
    pub exact: bool,
    pub commutative: bool,
}

/// `analysis::carried` is the recognizer every law enters through.
#[derive(Debug, Clone, PartialEq)]
pub enum Law {
    Op(LawOp),
    Call(CallLaw),
    Fold(FoldAccumulator),
    /// The `Order` of an `anyorder` region, joined by `Merge`.
    Order,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawOp {
    Add,
    Mul,
}

impl LawOp {
    pub fn bin_op(self) -> BinOp {
        match self {
            Self::Add => BinOp::Add,
            Self::Mul => BinOp::Mul,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallLaw {
    pub callee: QualifiedRef,
    pub instance: usize,
    pub identity: CallIdentity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallIdentity {
    Declared,
    /// The extern declares no identity, so a chunk's monoid is `Option` of
    /// the type with `None` as its identity; the lowerer writes the lifting.
    OptionLifted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldAccumulator {
    pub storage: ValueId,
    pub callee: QualifiedRef,
    pub instance: usize,
    pub fold: FoldLaw,
}

/// A cycle through several tokens has no one update, and is `InOrder`
/// with no law. A float law joined in arrival order changes the rounding,
/// so an inexact law is `InOrder` too.
fn judge(cycle: &Cycle, state: Option<&CarriedState>) -> Judged {
    let law = match (&cycle.tokens[..], state) {
        ([Token::Storage(Storage::Element)], _) => {
            return Judged {
                order: Order::Disjoint,
                law: None,
            };
        }
        ([Token::Carried(param) | Token::Order(param)], Some(state)) => state
            .params
            .iter()
            .find(|carried| carried.param == *param)
            .and_then(|carried| match carried.carried {
                Carried::Merge { op, exact } => Some(accumulator(op, exact)),
                Carried::Iv | Carried::Recurrence => None,
            }),
        ([Token::Storage(Storage::Slot(slot))], Some(state)) => state
            .storage_merges
            .iter()
            .find(|merge| merge.storage == *slot)
            .map(|merge| Accumulator {
                law: Law::Fold(FoldAccumulator {
                    storage: merge.storage,
                    callee: merge.callee,
                    instance: merge.instance,
                    fold: merge.fold,
                }),
                exact: true,
                commutative: merge.fold.commutative,
            }),
        _ => None,
    };
    let order = match &law {
        Some(acc) if acc.exact && acc.commutative => Order::AnyOrder,
        _ => Order::InOrder,
    };
    Judged { order, law }
}

fn accumulator(op: MergeOp, exact: bool) -> Accumulator {
    match op {
        MergeOp::Add => Accumulator {
            law: Law::Op(LawOp::Add),
            exact,
            commutative: true,
        },
        MergeOp::Mul => Accumulator {
            law: Law::Op(LawOp::Mul),
            exact,
            commutative: true,
        },
        MergeOp::Extern(merge) => Accumulator {
            law: Law::Call(CallLaw {
                callee: merge.callee,
                instance: merge.instance,
                identity: merge.identity,
            }),
            exact,
            commutative: merge.commutative,
        },
        MergeOp::Order => Accumulator {
            law: Law::Order,
            exact,
            commutative: true,
        },
    }
}

// -- The dependence graph within one iteration ----------------------------

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum InBlock {
    Inst(usize),
    Terminator,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RunPosition {
    block_rank: usize,
    in_block: InBlock,
}

struct Touch {
    position: RunPosition,
    member: usize,
    writes: bool,
}

struct MemberValues {
    defined: Vec<ValueId>,
    used: Vec<ValueId>,
}

struct MemberStorage {
    touched: Vec<ValueId>,
    written: Vec<ValueId>,
}

struct Overlap {
    keep: usize,
    absorb: usize,
}

struct Held {
    tokens: Vec<Token>,
    members: FxHashSet<usize>,
}

/// One edge of a terminator: the target's parameters before `fills_from`
/// are the terminator's own, and `args` fill the rest.
struct EdgeArgs<'a> {
    target: Label,
    args: &'a [ValueId],
    fills_from: usize,
}

fn edges(term: &Terminator) -> Vec<EdgeArgs<'_>> {
    fn plain(target: Label, args: &[ValueId]) -> EdgeArgs<'_> {
        EdgeArgs {
            target,
            args,
            fills_from: 0,
        }
    }
    match term {
        Terminator::Jump { label, args } => vec![plain(*label, args)],
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
        } => vec![plain(*then_label, then_args), plain(*else_label, else_args)],
        Terminator::Switch { arms, default, .. } => arms
            .iter()
            .map(|(_, label, args)| plain(*label, args))
            .chain(default.iter().map(|(label, args)| plain(*label, args)))
            .collect(),
        Terminator::For {
            source,
            stages,
            exit,
            exit_trip,
            exit_args,
        } => vec![
            EdgeArgs {
                target: stages.body(),
                args: &[],
                fills_from: source.supplied_params(),
            },
            EdgeArgs {
                target: *exit,
                args: exit_args,
                fills_from: exit_trip.supplied_params(),
            },
        ],
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

/// Every value some instruction names as a storage slot: a slot is reached
/// through what touches it, not as a value.
fn slot_values(cfg: &CfgBody) -> FxHashSet<ValueId> {
    let mut slots: FxHashSet<ValueId> = FxHashSet::default();
    for inst in cfg.blocks.iter().flat_map(|block| &block.insts) {
        if let InstKind::Ref { target, .. }
        | InstKind::Take { target, .. }
        | InstKind::Assign { target, .. } = &inst.kind
            && let Some(slot) = inst_info::storage(target)
        {
            slots.insert(slot);
        }
    }
    slots
}

fn written_slots(loans: &Loans<'_>, kind: &InstKind) -> Vec<ValueId> {
    effect(loans, kind)
        .writes
        .into_iter()
        .chain(slots_lent_mutably(loans, kind))
        .collect()
}

fn terminator_touched(loans: &Loans<'_>, term: &Terminator) -> Vec<ValueId> {
    inst_info::terminator_uses(term)
        .into_iter()
        .flat_map(|read| loans.holds(read))
        .filter_map(|loan| loan.storage.slot())
        .collect()
}

struct Graph {
    members: Vec<Member>,
    index: FxHashMap<Member, usize>,
    succs: Vec<FxHashSet<usize>>,
    preds: Vec<FxHashSet<usize>>,
    definers: FxHashMap<ValueId, Vec<usize>>,
    readers: FxHashMap<ValueId, Vec<Member>>,
}

impl Graph {
    fn of(
        cfg: &CfgBody,
        loans: &Loans<'_>,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
        body_label: Label,
    ) -> Graph {
        let body: Vec<BlockIdx> = loop_blocks
            .iter()
            .copied()
            .filter(|block| *block != header)
            .collect();
        let inside = |block: BlockIdx| body.contains(&block);
        let ranks = run_order(cfg, cfg.label_to_block[&body_label], &inside);
        let rank = |block: BlockIdx| {
            *ranks
                .get(&block)
                .expect("the body block reaches every block of the body without the header")
        };

        let mut blocks = body.clone();
        blocks.sort_by_key(|block| rank(*block));
        let mut members: Vec<Member> = Vec::new();
        for &block in &blocks {
            for at in 0..cfg.blocks[block.0].insts.len() {
                members.push(Member::Inst(InstAt { block, at }));
            }
            members.push(Member::Term(block));
        }
        let index: FxHashMap<Member, usize> = members
            .iter()
            .enumerate()
            .map(|(at, member)| (*member, at))
            .collect();
        let mut graph = Graph {
            succs: vec![FxHashSet::default(); members.len()],
            preds: vec![FxHashSet::default(); members.len()],
            members,
            index,
            definers: FxHashMap::default(),
            readers: FxHashMap::default(),
        };

        let slots = slot_values(cfg);
        let mut used_by: Vec<Vec<ValueId>> = Vec::with_capacity(graph.members.len());
        for (at, member) in graph.members.iter().enumerate() {
            let MemberValues { defined, used } = member_values(cfg, *member, &inside);
            for value in defined.into_iter().filter(|value| !slots.contains(value)) {
                graph.definers.entry(value).or_default().push(at);
            }
            used_by.push(
                used.into_iter()
                    .filter(|value| !slots.contains(value))
                    .collect(),
            );
        }
        for (at, used) in used_by.iter().enumerate() {
            for value in used {
                let member = graph.members[at];
                graph.readers.entry(*value).or_default().push(member);
                let definers: Vec<usize> = graph.definers.get(value).cloned().unwrap_or_default();
                for definer in definers.into_iter().filter(|definer| *definer != at) {
                    graph.edge(definer, at);
                }
            }
        }

        for Controlled { block, deciders } in control_dependence(cfg, &body, &inside) {
            let dependents: Vec<usize> = (0..cfg.blocks[block.0].insts.len())
                .map(|at| Member::Inst(InstAt { block, at }))
                .chain([Member::Term(block)])
                .map(|member| graph.index[&member])
                .collect();
            for decider in deciders {
                let decider = graph.index[&Member::Term(decider)];
                for &dependent in dependents.iter().filter(|dependent| **dependent != decider) {
                    graph.edge(decider, dependent);
                }
            }
        }

        let mut by_slot: FxHashMap<ValueId, Vec<Touch>> = FxHashMap::default();
        let mut by_context: FxHashMap<QualifiedRef, Vec<Touch>> = FxHashMap::default();
        for (at, member) in graph.members.iter().enumerate() {
            let position = RunPosition {
                block_rank: rank(member.block()),
                in_block: match *member {
                    Member::Inst(InstAt { at, .. }) => InBlock::Inst(at),
                    Member::Term(_) => InBlock::Terminator,
                },
            };
            if let Member::Inst(InstAt { block, at: inst }) = *member
                && let InstKind::Fetch { context, .. } | InstKind::Commit { context, .. } =
                    &cfg.blocks[block.0].insts[inst].kind
            {
                by_context.entry(*context).or_default().push(Touch {
                    position,
                    member: at,
                    writes: matches!(
                        cfg.blocks[block.0].insts[inst].kind,
                        InstKind::Commit { .. }
                    ),
                });
            }
            let MemberStorage { touched, written } = member_storage(cfg, loans, *member);
            let mut seen: Vec<ValueId> = Vec::new();
            for slot in touched.into_iter().chain(written.iter().copied()) {
                if seen.contains(&slot) {
                    continue;
                }
                seen.push(slot);
                by_slot.entry(slot).or_default().push(Touch {
                    position,
                    member: at,
                    writes: written.contains(&slot),
                });
            }
        }
        for touches in by_slot.into_values().chain(by_context.into_values()) {
            graph.order_touches(touches);
        }
        graph
    }

    fn order_touches(&mut self, mut touches: Vec<Touch>) {
        if !touches.iter().any(|touch| touch.writes) {
            return;
        }
        touches.sort_by_key(|touch| touch.position);
        for (i, before) in touches.iter().enumerate() {
            for after in &touches[i + 1..] {
                if (before.writes || after.writes) && before.member != after.member {
                    self.edge(before.member, after.member);
                }
            }
        }
    }

    fn edge(&mut self, from: usize, to: usize) {
        self.succs[from].insert(to);
        self.preds[to].insert(from);
    }

    fn readers(&self, value: ValueId) -> &[Member] {
        self.readers.get(&value).map_or(&[], Vec::as_slice)
    }

    fn definers(&self, value: ValueId) -> &[usize] {
        self.definers.get(&value).map_or(&[], Vec::as_slice)
    }

    fn reach(from: impl IntoIterator<Item = usize>, next: &[FxHashSet<usize>]) -> FxHashSet<usize> {
        let mut seen: FxHashSet<usize> = FxHashSet::default();
        let mut work: Vec<usize> = from.into_iter().collect();
        while let Some(member) = work.pop() {
            if seen.insert(member) {
                work.extend(next[member].iter().copied());
            }
        }
        seen
    }

    /// Every member on a path from one of `starts` to one of `ends`, both
    /// included.
    fn between(&self, starts: &[usize], ends: &[usize]) -> FxHashSet<usize> {
        let forward = Self::reach(starts.iter().copied(), &self.succs);
        let backward = Self::reach(ends.iter().copied(), &self.preds);
        forward.intersection(&backward).copied().collect()
    }

    fn with(&self, holds: impl Fn(Member) -> bool) -> Vec<usize> {
        (0..self.members.len())
            .filter(|at| holds(self.members[*at]))
            .collect()
    }

    fn cycles(
        &self,
        cfg: &CfgBody,
        loans: &Loans<'_>,
        slots: &TargetSlots,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
    ) -> Vec<Found> {
        let mut found: Vec<Held> = self.carried_cycles(cfg, header, loop_blocks);
        found.extend(self.storage_cycles(cfg, loans, slots));
        let exits = self.with(|member| match member {
            Member::Term(block) => {
                matches!(cfg.blocks[block.0].terminator, Terminator::Return { .. })
                    || cfg
                        .successors(block)
                        .into_iter()
                        .any(|succ| !loop_blocks.contains(&succ))
            }
            Member::Inst(_) => false,
        });
        if !exits.is_empty() {
            let mut members = self.between(&exits, &exits);
            members.extend(exits);
            found.push(Held {
                tokens: vec![Token::Control],
                members,
            });
        }

        loop {
            let mut changed = false;
            if let Some(Overlap { keep, absorb }) = first_overlap(&found) {
                let absorbed = found.remove(absorb);
                found[keep].tokens.extend(absorbed.tokens);
                found[keep].members.extend(absorbed.members);
                changed = true;
            }
            for held in &mut found {
                let within: Vec<usize> = held.members.iter().copied().collect();
                for member in self.between(&within, &within) {
                    changed |= held.members.insert(member);
                }
            }
            if !changed {
                break;
            }
        }
        found
            .into_iter()
            .map(|held| Found {
                tokens: held.tokens,
                members: held
                    .members
                    .into_iter()
                    .map(|at| self.members[at])
                    .collect(),
            })
            .collect()
    }

    /// A header parameter's cycle runs from its readers to the definers of
    /// what the back edges send it. Where no such path exists, it is the
    /// readers, the definers and what runs between them. A parameter no
    /// member reads or advances lies at the back edge, in the stage of the
    /// latches.
    fn carried_cycles(
        &self,
        cfg: &CfgBody,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
    ) -> Vec<Held> {
        let header_label = cfg.blocks[header.0].label;
        let latches: Vec<BlockIdx> = loop_blocks
            .iter()
            .copied()
            .filter(|block| *block != header && cfg.successors(*block).contains(&header))
            .collect();
        let mut found: Vec<Held> = Vec::new();
        for (index, &param) in cfg.blocks[header.0].params.iter().enumerate() {
            let mut sent: Vec<ValueId> = Vec::new();
            for latch in &latches {
                let into_header = edges(&cfg.blocks[latch.0].terminator)
                    .into_iter()
                    .filter(|edge| edge.target == header_label);
                for edge in into_header {
                    let Some(position) = index.checked_sub(edge.fills_from) else {
                        // The terminator fills this parameter itself, from
                        // nothing an iteration computes.
                        continue;
                    };
                    let arg = *edge
                        .args
                        .get(position)
                        .expect("an edge passes an argument for each parameter it fills");
                    let arg = copied_from(cfg, header, arg);
                    if arg != param && !sent.contains(&arg) {
                        sent.push(arg);
                    }
                }
            }
            if sent.is_empty() {
                continue;
            }
            let ty = cfg
                .val_types
                .get(&param)
                .expect("lowering types every header parameter");
            let is_order = *ty == Ty::Order;
            let starts: Vec<usize> = self
                .readers(param)
                .iter()
                .map(|member| self.index[member])
                .collect();
            let ends: Vec<usize> = sent
                .iter()
                .flat_map(|value| self.definers(*value).iter().copied())
                .collect();
            let mut members = self.between(&starts, &ends);
            if is_order {
                self.add_merged(cfg, &mut members);
            }
            if members.is_empty() {
                // The next value reads nothing of the state, so the readers
                // of one iteration wait on the definers of the one before:
                // the cycle closes through the order of the iteration.
                members.extend(self.between(&ends, &starts));
                members.extend(starts.iter().copied());
                members.extend(ends.iter().copied());
            }
            if members.is_empty() {
                members.extend(
                    latches
                        .iter()
                        .map(|latch| self.index[&Member::Term(*latch)]),
                );
            }
            found.push(Held {
                tokens: vec![match is_order {
                    true => Token::Order(param),
                    false => Token::Carried(param),
                }],
                members,
            });
        }
        found
    }

    /// A storage's cycle is every member that touches it and every member
    /// between them; the element's, its writers and the members between
    /// them, since each iteration writes only the element at its counter.
    fn storage_cycles(&self, cfg: &CfgBody, loans: &Loans<'_>, slots: &TargetSlots) -> Vec<Held> {
        let mut storages: Vec<ValueId> = Vec::new();
        let mut element_writers: Vec<usize> = Vec::new();
        let mut contexts: Vec<QualifiedRef> = Vec::new();
        for (at, member) in self.members.iter().enumerate() {
            let Member::Inst(InstAt { block, at: inst }) = *member else {
                continue;
            };
            let kind = &cfg.blocks[block.0].insts[inst].kind;
            for slot in written_slots(loans, kind) {
                match slots.target_of(slot) {
                    Some(Written::Element) if !element_writers.contains(&at) => {
                        element_writers.push(at)
                    }
                    Some(Written::Storage(slot)) if !storages.contains(&slot) => {
                        storages.push(slot)
                    }
                    Some(Written::Element | Written::Storage(_)) | None => {}
                }
            }
            if let InstKind::Commit { context, .. } = kind
                && !contexts.contains(context)
            {
                contexts.push(*context);
            }
        }
        let mut found: Vec<Held> = Vec::new();
        for slot in storages {
            let touchers =
                self.with(|member| member_storage(cfg, loans, member).touched.contains(&slot));
            let writers =
                self.with(|member| member_storage(cfg, loans, member).written.contains(&slot));
            let mut members = self.between(&touchers, &writers);
            members.extend(touchers);
            found.push(Held {
                tokens: vec![Token::Storage(Storage::Slot(slot))],
                members,
            });
        }
        if !element_writers.is_empty() {
            let mut members = self.between(&element_writers, &element_writers);
            members.extend(element_writers);
            found.push(Held {
                tokens: vec![Token::Storage(Storage::Element)],
                members,
            });
        }
        for context in contexts {
            let touchers = self.with(|member| match member {
                Member::Inst(InstAt { block, at }) => matches!(
                    &cfg.blocks[block.0].insts[at].kind,
                    InstKind::Fetch { context: read, .. } | InstKind::Commit { context: read, .. }
                        if *read == context
                ),
                Member::Term(_) => false,
            });
            let mut members = self.between(&touchers, &touchers);
            members.extend(touchers);
            found.push(Held {
                tokens: vec![Token::Storage(Storage::Context(context))],
                members,
            });
        }
        found
    }

    /// RFC-0007 rule 7: a `merge` joins the operations whose `Order`s it
    /// reads, so they belong to the cycle of the `Order` it advances.
    fn add_merged(&self, cfg: &CfgBody, members: &mut FxHashSet<usize>) {
        loop {
            let mut added: Vec<usize> = Vec::new();
            for &at in members.iter() {
                let Member::Inst(InstAt { block, at: inst }) = self.members[at] else {
                    continue;
                };
                let InstKind::Merge { orders, .. } = &cfg.blocks[block.0].insts[inst].kind else {
                    continue;
                };
                for order in orders {
                    added.extend(
                        self.definers(*order)
                            .iter()
                            .copied()
                            .filter(|definer| !members.contains(definer)),
                    );
                }
            }
            if added.is_empty() {
                return;
            }
            members.extend(added);
        }
    }
}

/// The value `value` is a copy of: a parameter of a block other than the
/// header that every edge into it passes one value is that value, and the
/// back edge that sends such a parameter sends what it copies.
fn copied_from(cfg: &CfgBody, header: BlockIdx, value: ValueId) -> ValueId {
    let preds = cfg.predecessors();
    let mut value = value;
    let mut seen: Vec<ValueId> = Vec::new();
    loop {
        seen.push(value);
        let owner = (0..cfg.blocks.len()).map(BlockIdx).find_map(|block| {
            cfg.blocks[block.0]
                .params
                .iter()
                .position(|param| *param == value)
                .map(|at| BlockParam { block, at })
        });
        let Some(BlockParam { block, at }) = owner else {
            return value;
        };
        if block == header {
            return value;
        }
        let label = cfg.blocks[block.0].label;
        let mut sent: Vec<ValueId> = Vec::new();
        for pred in preds.get(&block).into_iter().flatten() {
            for edge in edges(&cfg.blocks[pred.0].terminator) {
                if edge.target != label {
                    continue;
                }
                let Some(position) = at.checked_sub(edge.fills_from) else {
                    // A terminator fills this parameter itself: it copies
                    // nothing.
                    return value;
                };
                let arg = *edge
                    .args
                    .get(position)
                    .expect("an edge passes an argument for each parameter it fills");
                if !sent.contains(&arg) {
                    sent.push(arg);
                }
            }
        }
        match sent[..] {
            [one] if !seen.contains(&one) => value = one,
            _ => return value,
        }
    }
}

struct BlockParam {
    block: BlockIdx,
    at: usize,
}

fn member_values(
    cfg: &CfgBody,
    member: Member,
    inside: &impl Fn(BlockIdx) -> bool,
) -> MemberValues {
    match member {
        Member::Inst(InstAt { block, at }) => {
            let kind = &cfg.blocks[block.0].insts[at].kind;
            MemberValues {
                defined: inst_info::defs(kind).into_iter().collect(),
                used: inst_info::uses(kind).into_iter().collect(),
            }
        }
        Member::Term(block) => MemberValues {
            defined: cfg
                .successors(block)
                .into_iter()
                .filter(|succ| inside(*succ))
                .flat_map(|succ| cfg.blocks[succ.0].params.iter().copied())
                .collect(),
            used: inst_info::terminator_uses(&cfg.blocks[block.0].terminator)
                .into_iter()
                .collect(),
        },
    }
}

fn member_storage(cfg: &CfgBody, loans: &Loans<'_>, member: Member) -> MemberStorage {
    match member {
        Member::Inst(InstAt { block, at }) => {
            let kind = &cfg.blocks[block.0].insts[at].kind;
            MemberStorage {
                touched: touched_slots(loans, kind),
                written: written_slots(loans, kind),
            }
        }
        Member::Term(block) => MemberStorage {
            touched: terminator_touched(loans, &cfg.blocks[block.0].terminator),
            written: Vec::new(),
        },
    }
}

fn first_overlap(found: &[Held]) -> Option<Overlap> {
    (0..found.len()).find_map(|keep| {
        (keep + 1..found.len())
            .find(|absorb| !found[keep].members.is_disjoint(&found[*absorb].members))
            .map(|absorb| Overlap { keep, absorb })
    })
}

/// Reverse post order from the body block over the edges inside the body.
fn run_order(
    cfg: &CfgBody,
    body: BlockIdx,
    inside: &impl Fn(BlockIdx) -> bool,
) -> FxHashMap<BlockIdx, usize> {
    fn visit(
        cfg: &CfgBody,
        block: BlockIdx,
        inside: &impl Fn(BlockIdx) -> bool,
        seen: &mut FxHashSet<BlockIdx>,
        post: &mut Vec<BlockIdx>,
    ) {
        if !seen.insert(block) {
            return;
        }
        for succ in cfg.successors(block) {
            if inside(succ) {
                visit(cfg, succ, inside, seen, post);
            }
        }
        post.push(block);
    }
    let mut post: Vec<BlockIdx> = Vec::new();
    visit(cfg, body, inside, &mut FxHashSet::default(), &mut post);
    post.into_iter()
        .rev()
        .enumerate()
        .map(|(rank, block)| (block, rank))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PostNode {
    Block(BlockIdx),
    Exit,
}

struct Controlled {
    block: BlockIdx,
    deciders: Vec<BlockIdx>,
}

fn control_dependence(
    cfg: &CfgBody,
    body: &[BlockIdx],
    inside: &impl Fn(BlockIdx) -> bool,
) -> Vec<Controlled> {
    let succs: FxHashMap<BlockIdx, Vec<PostNode>> = body
        .iter()
        .map(|&block| {
            let mut out: Vec<PostNode> = Vec::new();
            let mut push = |node: PostNode| {
                if !out.contains(&node) {
                    out.push(node);
                }
            };
            let successors = cfg.successors(block);
            for succ in &successors {
                push(match inside(*succ) {
                    true => PostNode::Block(*succ),
                    false => PostNode::Exit,
                });
            }
            if successors.is_empty() {
                push(PostNode::Exit);
            }
            (block, out)
        })
        .collect();
    let all: FxHashSet<PostNode> = body
        .iter()
        .map(|block| PostNode::Block(*block))
        .chain([PostNode::Exit])
        .collect();
    let mut pdom: FxHashMap<PostNode, FxHashSet<PostNode>> = all
        .iter()
        .map(|node| match node {
            PostNode::Exit => (*node, FxHashSet::from_iter([PostNode::Exit])),
            PostNode::Block(_) => (*node, all.clone()),
        })
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for &block in body {
            let mut next = all.clone();
            for succ in &succs[&block] {
                next.retain(|held| pdom[succ].contains(held));
            }
            next.insert(PostNode::Block(block));
            if next != pdom[&PostNode::Block(block)] {
                pdom.insert(PostNode::Block(block), next);
                changed = true;
            }
        }
    }
    let ipdom = |block: BlockIdx| -> PostNode {
        let node = PostNode::Block(block);
        let strict: Vec<PostNode> = pdom[&node]
            .iter()
            .copied()
            .filter(|other| *other != node)
            .collect();
        strict
            .iter()
            .copied()
            .find(|candidate| pdom[candidate].len() == strict.len())
            .expect("every block of a loop's body reaches the latch, so the exit post-dominates it")
    };
    let mut deciders: FxHashMap<BlockIdx, Vec<BlockIdx>> = FxHashMap::default();
    for &block in body {
        let out = &succs[&block];
        if out.len() < 2 {
            continue;
        }
        let stop = ipdom(block);
        for &succ in out {
            let mut runner = succ;
            while runner != stop {
                let PostNode::Block(held) = runner else {
                    break;
                };
                let entry = deciders.entry(held).or_default();
                if !entry.contains(&block) {
                    entry.push(block);
                }
                runner = ipdom(held);
            }
        }
    }
    let mut found: Vec<Controlled> = deciders
        .into_iter()
        .map(|(block, deciders)| Controlled { block, deciders })
        .collect();
    found.sort_by_key(|controlled| controlled.block);
    found
}
