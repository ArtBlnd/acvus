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

use acvus_ast::Literal;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::AffineValues;
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info::{self, Reads};
use crate::analysis::interval::ConstantBounds;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{
    Invariants, LoopNest, NaturalLoop, Term, natural_loops_innermost_first,
};
use crate::analysis::raise::{self, FunctionSummary};
use crate::analysis::targets::{TargetSlots, Written, effect, slots_lent_mutably, touched_slots};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{
    BinOp, Callee, ForSource, IndexMode, InstKind, Label, PathSeg, RefTarget, Stages, ValueId,
};
use crate::laws::{
    ExternInstance, LawTable, ReachedPlace, Reaches, ResolvedBinary, ResolvedFold,
    ResolvedIdentity, ResolvedLaws,
};
use crate::ty::{Mutability, Ty};

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
    pub entry_block: BlockIdx,
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
                entry_block: entry,
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
    /// The cycles before those of the exiting stage join the control
    /// token's. `optimize::stages` cuts from these: where the body is cut
    /// decides which cycles share the exiting stage.
    pub unjoined: Vec<Cycle>,
    pub control: Control,
    early_reads: Vec<EarlyRead>,
    effects: Vec<EffectWait>,
    ahead_of_exit: Vec<AheadOfExit>,
    disjoint: Vec<ValueId>,
}

pub struct HeaderDeps {
    pub header: BlockIdx,
    pub deps: Result<LoopDeps, ShapeFault>,
}

pub struct BodyDeps {
    pub loops: Vec<HeaderDeps>,
}

impl BodyDeps {
    pub fn of(cfg: &CfgBody, laws: &LawTable) -> Self {
        let loans = Loans::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        let found = (0..cfg.blocks.len())
            .map(BlockIdx)
            .filter(|at| matches!(cfg.blocks[at.0].terminator, Terminator::For { .. }))
            .map(|header| HeaderDeps {
                header,
                deps: LoopDeps::with(cfg, &loans, laws, header, &loop_blocks_of(&loops, header)),
            })
            .collect();
        Self { loops: found }
    }
}

impl LoopDeps {
    /// # Panics
    /// If `header` does not end in `For`.
    pub fn of(cfg: &CfgBody, laws: &LawTable, header: BlockIdx) -> Result<LoopDeps, ShapeFault> {
        let loans = Loans::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        LoopDeps::with(cfg, &loans, laws, header, &loop_blocks_of(&loops, header))
    }

    fn with(
        cfg: &CfgBody,
        loans: &Loans<'_>,
        laws: &LawTable,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
    ) -> Result<LoopDeps, ShapeFault> {
        let Terminator::For { source, stages, .. } = &cfg.blocks[header.0].terminator else {
            panic!("block {} heads no `For`", header.0)
        };
        let membership = StageMembership::of(cfg, header, *source, stages, loop_blocks)?;
        let graph = Graph::of(cfg, loans, header, loop_blocks, stages.body());
        let slots = TargetSlots::of(loans, *source, loop_blocks);
        let disjoint = disjoint_storages(
            loans,
            laws,
            header,
            loop_blocks,
            &graph.written_storages(cfg, loans, &slots),
        );
        let stage_of = |member: Member| {
            membership
                .stage_of(member.block())
                .expect("a chain whose shape holds puts every block of the body in a stage")
        };
        let mut cycles: Vec<Cycle> = graph
            .cycles(cfg, loans, &slots, &disjoint, header, loop_blocks)
            .into_iter()
            .map(|found| place(found, &stage_of))
            .collect();
        let unjoined = cycles.clone();

        // RFC-0092: the control token passes with the exiting
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

        let ahead_of_exit = match control {
            Control::Upfront => Vec::new(),
            Control::Chained { cycle } => {
                let exit_stage = cycles[cycle].earliest_stage();
                let run_ahead = RunAhead::of(cfg, laws, header);
                graph
                    .members
                    .iter()
                    .filter(|member| stage_of(**member) < exit_stage)
                    .filter_map(|&member| {
                        run_ahead.held_back(member).map(|held_back| AheadOfExit {
                            member,
                            stage: stage_of(member),
                            exit_stage,
                            held_back,
                        })
                    })
                    .collect()
            }
        };

        Ok(LoopDeps {
            header,
            membership,
            cycles,
            unjoined,
            control,
            early_reads,
            effects,
            ahead_of_exit,
            disjoint,
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

    pub fn ahead_of_exit(&self) -> &[AheadOfExit] {
        &self.ahead_of_exit
    }

    /// Each cycle's order and law, by the index of [`Self::cycles`].
    pub fn judge(&self, cfg: &CfgBody, laws: &LawTable) -> Vec<Judged> {
        let domtree = DomTree::build(cfg);
        let invariants = Invariants::of(cfg);
        let nest = LoopNest::of(cfg, &domtree, &invariants);
        let Some(id) = nest.by_header(self.header) else {
            // No back edge reaches the header: the body runs at most once,
            // so no iteration hands a next one anything to merge.
            return self
                .cycles
                .iter()
                .map(|cycle| judge(cycle, &self.disjoint, None))
                .collect();
        };
        let loop_ = nest.get(id);
        let loans = Loans::build(cfg);
        let Terminator::For { source, .. } = &cfg.blocks[self.header.0].terminator else {
            panic!("block {} heads the `For` `loop_deps` read", self.header.0)
        };
        let loop_blocks: Vec<BlockIdx> = loop_.natural.blocks().collect();
        let slots = TargetSlots::of(&loans, *source, &loop_blocks);
        let reading_of = |state: State| {
            LawReading::of(&loans, laws, &slots, self.header, &loop_blocks, state)
        };
        let read = |state: State| reading_of(state).law();
        let folds = FoldReading {
            loans: &loans,
            laws,
            reads: Reads::of(cfg, loop_blocks.iter().copied()),
            insts: loop_blocks
                .iter()
                .flat_map(|block| &cfg.blocks[block.0].insts)
                .map(|inst| &inst.kind)
                .collect(),
        };
        let params = &cfg.blocks[self.header.0].params;
        let state_of = |token: Token| match token {
            Token::Carried(param) | Token::Order(param) => {
                let index = params.iter().position(|held| *held == param)?;
                Some(State::Param { param, index })
            }
            Token::Storage(Storage::Slot(slot)) => Some(State::Slot(slot)),
            Token::Storage(Storage::Element | Storage::Context(_)) | Token::Control => None,
        };
        let alone = |token: Token| match token {
            Token::Storage(Storage::Slot(slot)) => {
                folds.law(slot).or_else(|| read(State::Slot(slot)))
            }
            Token::Carried(_) => {
                read(state_of(token)?).or_else(|| first(&[token], &state_of, &reading_of))
            }
            token => read(state_of(token)?),
        };
        let reading = |tokens: &[Token]| match tokens {
            [token] => alone(*token),
            tokens => product(tokens, &state_of, &reading_of)
                .or_else(|| extremum(tokens, &state_of, &reading_of))
                .or_else(|| first(tokens, &state_of, &reading_of)),
        };
        self.cycles
            .iter()
            .map(|cycle| judge(cycle, &self.disjoint, Some(&reading)))
            .collect()
    }
}

/// RFC-0089 rule 4: a cycle of several tokens whose steps read no other
/// token has the product of their laws, each read as if its token were
/// alone. A token's steps read another's where a value its update is
/// computed through reads the other's state.
fn product<'a, 's, 'cfg>(
    tokens: &[Token],
    state_of: &dyn Fn(Token) -> Option<State>,
    reading_of: &dyn Fn(State) -> LawReading<'a, 's, 'cfg>,
) -> Option<Accumulator> {
    let updates: Vec<(Token, Update<'_>)> = tokens
        .iter()
        .map(|&token| Some((token, reading_of(state_of(token)?).update()?)))
        .collect::<Option<_>>()?;
    for (at, (_, update)) in updates.iter().enumerate() {
        for (other, (_, other_update)) in updates.iter().enumerate() {
            if at != other && !update.chain.is_disjoint(&other_update.reading_state) {
                return None;
            }
        }
    }
    let parts: Vec<(Token, Accumulator)> = updates
        .into_iter()
        .map(|(token, update)| (token, update.accumulator()))
        .collect();
    Some(Accumulator {
        exact: parts.iter().all(|(_, acc)| acc.exact),
        commutative: parts.iter().all(|(_, acc)| acc.commutative),
        law: Law::Product(parts),
    })
}

/// RFC-0089 rule 4: a cycle of carried tokens one of which is chosen by a
/// strict compare and select that carries every other with it has the
/// left-biased maximum or minimum.
fn extremum<'a, 's, 'cfg>(
    tokens: &[Token],
    state_of: &dyn Fn(Token) -> Option<State>,
    reading_of: &dyn Fn(State) -> LawReading<'a, 's, 'cfg>,
) -> Option<Accumulator> {
    let carried: Vec<(Token, ValueId, usize)> = tokens
        .iter()
        .map(|&token| match (token, state_of(token)?) {
            (Token::Carried(_), State::Param { param, index }) => Some((token, param, index)),
            _ => None,
        })
        .collect::<Option<_>>()?;
    carried.iter().find_map(|&(over, param, index)| {
        let others: Vec<(ValueId, usize)> = carried
            .iter()
            .filter(|(token, ..)| *token != over)
            .map(|&(_, param, index)| (param, index))
            .collect();
        let op = reading_of(State::Param { param, index }).extremum_carrying(&others)?;
        Some(Accumulator {
            law: Law::Extremum {
                op,
                over,
                carried: tokens.iter().copied().filter(|token| *token != over).collect(),
            },
            exact: true,
            commutative: false,
        })
    })
}

/// RFC-0093 rules 5 and 7: a cycle of carried tokens one of which is a `||`
/// guard an arm sets, the arm taken only where it is unset and sending every
/// other token a value that reads no token, the other arm leaving them all,
/// is `first`: `last` of the others, guarded by that token.
fn first<'a, 's, 'cfg>(
    tokens: &[Token],
    state_of: &dyn Fn(Token) -> Option<State>,
    reading_of: &dyn Fn(State) -> LawReading<'a, 's, 'cfg>,
) -> Option<Accumulator> {
    let carried: Vec<(Token, ValueId, usize)> = tokens
        .iter()
        .map(|&token| match (token, state_of(token)?) {
            (Token::Carried(_), State::Param { param, index }) => Some((token, param, index)),
            _ => None,
        })
        .collect::<Option<_>>()?;
    carried.iter().find_map(|&(guard, param, index)| {
        let others: Vec<(ValueId, usize)> = carried
            .iter()
            .filter(|(token, ..)| *token != guard)
            .map(|&(_, param, index)| (param, index))
            .collect();
        let carried = tokens.iter().copied().filter(|token| *token != guard).collect();
        let guard = match reading_of(State::Param { param, index }).guards_first(&others)? {
            Unset::Flag => Guard::Flag(guard),
            Unset::Sentinel(sentinel) => Guard::Sentinel {
                token: guard,
                sentinel,
            },
        };
        Some(Accumulator {
            law: Law::First { guard, carried },
            exact: true,
            commutative: false,
        })
    })
}

/// RFC-0082 rule 6: a storage every write of which in the loop is a call of
/// one instance of an extern with a `fold` law, lending the storage through
/// its first argument and no other, and which the loop reads only to lend it
/// to those calls, has that instance's `Fold` law.
struct FoldReading<'a, 'cfg> {
    loans: &'a Loans<'cfg>,
    laws: &'a LawTable,
    reads: Reads,
    insts: Vec<&'cfg InstKind>,
}

impl FoldReading<'_, '_> {
    fn law(&self, storage: ValueId) -> Option<Accumulator> {
        let mut found: Option<FoldCall> = None;
        let mut lenders: Vec<ValueId> = Vec::new();
        for kind in &self.insts {
            if !self.loans.storage_effect(kind).writes.contains(&storage) {
                continue;
            }
            let call = self.fold_call(kind, storage)?;
            lenders.push(call.lender);
            match &found {
                Some(held) if held.callee != call.callee => return None,
                Some(_) => {}
                None => found = Some(call),
            }
        }
        let only_lent_to_the_folds = self.insts.iter().all(|kind| {
            let effect = self.loans.storage_effect(kind);
            if !effect.reads.contains(&storage) || effect.writes.contains(&storage) {
                return true;
            }
            match kind {
                InstKind::Ref { dst, .. } => lenders.contains(dst) && self.reads.count(*dst) == 1,
                _ => false,
            }
        });
        let FoldCall { callee, fold, .. } = found.filter(|_| only_lent_to_the_folds)?;
        Some(Accumulator {
            law: Law::Fold(FoldAccumulator {
                storage,
                callee,
                fold,
            }),
            exact: true,
            commutative: fold.commutative,
        })
    }

    fn fold_call(&self, kind: &InstKind, storage: ValueId) -> Option<FoldCall> {
        let InstKind::FunctionCall {
            callee: callee @ Callee::Extern { id, instance, .. },
            args,
            ..
        } = kind
        else {
            return None;
        };
        let ResolvedLaws::Fold(fold) = self.laws.of_callee(callee) else {
            return None;
        };
        let (&lender, rest) = args.split_first()?;
        let lends = |value: ValueId, mutability: Mutability| {
            self.loans
                .holds(value)
                .any(|loan| loan.storage.slot() == Some(storage) && loan.mutability == mutability)
        };
        let lent_by_the_state_alone = lends(lender, Mutability::Mut)
            && !rest
                .iter()
                .any(|&arg| lends(arg, Mutability::Mut) || lends(arg, Mutability::Shared));
        lent_by_the_state_alone.then_some(FoldCall {
            lender,
            callee: ExternInstance {
                id: *id,
                instance: *instance,
            },
            fold: *fold,
        })
    }
}

struct FoldCall {
    lender: ValueId,
    callee: ExternInstance,
    fold: ResolvedFold,
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

/// Why an operation of a loop's body waits for its iteration's control
/// token (RFC-0089 rule 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeldBack {
    Effect,
    MayNotFinish,
}

/// A `for` terminator is never held back: the operations of its body are
/// members of the same work, and each is asked for itself.
pub struct RunAhead<'a> {
    cfg: &'a CfgBody,
    laws: &'a LawTable,
    in_while: FxHashSet<BlockIdx>,
}

impl<'a> RunAhead<'a> {
    pub fn of(cfg: &'a CfgBody, laws: &'a LawTable, header: BlockIdx) -> Self {
        let loops = natural_loops_innermost_first(cfg, &DomTree::build(cfg));
        let ours = loop_blocks_of(&loops, header);
        let in_while = loops
            .iter()
            .filter(|inner| inner.header != header && ours.contains(&inner.header))
            .filter(|inner| inner.is_while(cfg))
            .flat_map(NaturalLoop::blocks)
            .collect();
        Self {
            cfg,
            laws,
            in_while,
        }
    }

    pub fn held_back(&self, member: Member) -> Option<HeldBack> {
        let Member::Inst(at) = member else {
            return self
                .in_while
                .contains(&member.block())
                .then_some(HeldBack::MayNotFinish);
        };
        let kind = &self.cfg.blocks[at.block.0].insts[at.at].kind;
        if has_effect(kind) {
            return Some(HeldBack::Effect);
        }
        let finishes = !self.in_while.contains(&at.block)
            && raise::finishes(kind, self.laws, &FunctionSummary::unknown());
        (!finishes).then_some(HeldBack::MayNotFinish)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AheadOfExit {
    pub member: Member,
    pub stage: usize,
    pub exit_stage: usize,
    pub held_back: HeldBack,
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

#[derive(Debug, Clone, PartialEq)]
pub enum Law {
    Op(LawOp),
    Call(CallLaw),
    Fold(FoldAccumulator),
    /// The `Order` of an `anyorder` region, joined by `Merge`.
    Order,
    /// `a · b = b`, associative since both groupings of three are the last.
    /// It has no identity in the token's type: a chunk's partial is `Option`
    /// of it with `None` as identity, which the lowerer writes.
    Last,
    /// `(a, s) · (b, t)` is `(b, t)` where `b` wins the strict compare `op`
    /// (`Max` or `Min`) against `a`, and `(a, s)` otherwise, so a tie keeps
    /// the earlier. It has no identity in the tokens' types: a chunk's
    /// partial is `Option` of the tuple with `None` as identity.
    Extremum {
        op: LawOp,
        over: Token,
        carried: Vec<Token>,
    },
    /// `None · Some(y)` is `Some(y)` and `Some(b) · Some(y)` is
    /// `Some(b ⊕ y)`, `⊕` the inner law.
    OptionLifted(Box<Law>),
    /// Each token combined by its own law.
    Product(Vec<(Token, Accumulator)>),
    /// The greater (`Max`) or lesser (`Min`) of two values under the total
    /// order the extern `order` compares by (RFC-0082 rule 10). Equal values
    /// are one value under it, so the choice is associative and commutative.
    /// It has no identity in the token's type: a chunk's partial is `Option`
    /// of it with `None` as identity.
    Ordered { op: LawOp, order: ExternInstance },
    /// `last` of `carried`, guarded by `guard`, which its arm sets:
    /// `(f, a) · (g, b)` is `(f ‖ g, b)` where `g` holds and `f` does not,
    /// and `(f ‖ g, a)` otherwise, so the earlier set values stay.
    /// Associative, not commutative.
    First { guard: Guard, carried: Vec<Token> },
    /// The inner law, whose token an arm taken only at the first iteration
    /// sets to a value reading none of it: the run from the entry value
    /// discards that value there, so the join takes no entry value.
    Reset(Box<Law>),
}

/// Whether a `first`'s arm has run (RFC-0093 rules 5 and 7).
#[derive(Debug, Clone, PartialEq)]
pub enum Guard {
    /// A `Bool` token the arm sets to `true`.
    Flag(Token),
    /// The token itself, which the arm sets and which holds the constant
    /// `sentinel` until then: no write in the loop sends that constant. Its
    /// value is guarded with `carried`, and a chunk starts from `sentinel`.
    Sentinel { token: Token, sentinel: ValueId },
}

/// An operation the language owns, with the laws its own definition gives
/// it. Each is associative, and each has an identity at its type:
///
/// - `Add` and `Mul` over an integer wrap (RFC-0037), which is arithmetic
///   modulo `2^w`, a commutative ring: both are associative and commutative,
///   with identities `0` and `1`. Over a float they round, so a regrouping
///   changes the result and the law is inexact (RFC-0066 rule 5).
/// - `Min` and `Max` over an integer of one width choose the lesser or the
///   greater of two values under the width's total order (`BinOp::Min`,
///   `BinOp::Max`). A lesser-of over a total order is associative
///   (`min(min(a, b), c)` and `min(a, min(b, c))` are both the least of the
///   three) and commutative, with the width's `MAX` as identity; `Max` is
///   its mirror, with `MIN`.
/// - `Concat` over `String` is the concatenation of byte sequences
///   (`InstKind::StringConcat`): `(a ++ b) ++ c` and `a ++ (b ++ c)` are the
///   bytes of `a`, then of `b`, then of `c`, so it is associative with the
///   empty string as identity. It does not commute: `"a" ++ "b"` is not
///   `"b" ++ "a"`, and a law over it joins its partials in chunk order.
/// - `Or`, `And` and `Xor` over `Bool` are the language's `||`, `&&` and
///   `!=` (RFC-0089 rule 4). Each is a function of the two truth values its
///   table defines, and the tables are associative and symmetric: `Or` is
///   true where either is, `And` where both are, `Xor` where exactly one
///   is. Their identities are `false`, `true` and `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawOp {
    Add,
    Mul,
    Min,
    Max,
    Concat,
    Or,
    And,
    Xor,
}

impl LawOp {
    pub fn commutes(self) -> bool {
        match self {
            Self::Add | Self::Mul | Self::Min | Self::Max | Self::Or | Self::And | Self::Xor => {
                true
            }
            Self::Concat => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallLaw {
    pub callee: ExternInstance,
    pub identity: CallIdentity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CallIdentity {
    Declared(ResolvedIdentity),
    /// The extern declares no identity, so a chunk's monoid is `Option` of
    /// the type with `None` as its identity; the lowerer writes the lifting.
    OptionLifted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldAccumulator {
    pub storage: ValueId,
    pub callee: ExternInstance,
    pub fold: ResolvedFold,
}

/// A cycle through several tokens has a law only by a reading that names
/// several. A float law joined in arrival order changes the rounding, so an
/// inexact law is `InOrder`, and so is a law that does not commute, whose
/// partials join in chunk order.
fn judge(
    cycle: &Cycle,
    disjoint: &[ValueId],
    reading: Option<&dyn Fn(&[Token]) -> Option<Accumulator>>,
) -> Judged {
    let law = match (&cycle.tokens[..], reading) {
        ([Token::Storage(Storage::Element)], _) => {
            return Judged {
                order: Order::Disjoint,
                law: None,
            };
        }
        ([Token::Storage(Storage::Slot(slot))], _) if disjoint.contains(slot) => {
            return Judged {
                order: Order::Disjoint,
                law: None,
            };
        }
        (tokens, Some(reading)) => reading(tokens),
        (_, None) => None,
    };
    let order = match &law {
        Some(acc) if acc.exact && acc.commutative => Order::AnyOrder,
        _ => Order::InOrder,
    };
    Judged { order, law }
}

// -- A storage reached at one affine index of the counter (rule 4) -------

#[derive(Debug, Clone, PartialEq)]
enum Component {
    Named(PathSeg),
    At(ValueId),
}

#[derive(Debug, Clone)]
struct Place {
    slot: ValueId,
    path: Vec<Component>,
}

impl Place {
    fn below(mut self, steps: impl IntoIterator<Item = Component>) -> Place {
        self.path.extend(steps);
        self
    }
}

enum Reach {
    Places(Vec<Place>),
    Whole,
}

/// The element parameter of a `for` body over a slice is the source's
/// element at the counter parameter beside it.
#[derive(Clone, Copy)]
struct ForElement {
    source: ValueId,
    counter: ValueId,
}

/// The place a value names is read off the instruction that defines it;
/// a value no `Ref`, `AsSlice`, `Index` by reference or `for` element
/// defines names no place, and whatever reaches through it reaches the
/// whole storage.
struct Places<'a, 'cfg> {
    cfg: &'cfg CfgBody,
    loans: &'a Loans<'cfg>,
    laws: &'a LawTable,
    defs: FxHashMap<ValueId, InstAt>,
    elements: FxHashMap<ValueId, ForElement>,
}

impl<'a, 'cfg> Places<'a, 'cfg> {
    fn of(loans: &'a Loans<'cfg>, laws: &'a LawTable) -> Self {
        let cfg = loans.cfg();
        let mut defs: FxHashMap<ValueId, InstAt> = FxHashMap::default();
        let mut elements: FxHashMap<ValueId, ForElement> = FxHashMap::default();
        for (index, block) in cfg.blocks.iter().enumerate() {
            for (at, inst) in block.insts.iter().enumerate() {
                for def in inst_info::defs(&inst.kind) {
                    defs.insert(
                        def,
                        InstAt {
                            block: BlockIdx(index),
                            at,
                        },
                    );
                }
            }
            if let Terminator::For {
                source: ForSource::Slice(source) | ForSource::SliceMut(source),
                stages,
                ..
            } = &block.terminator
                && let Some(body) = cfg.label_to_block.get(&stages.body())
                && let [element, counter] = cfg.blocks[body.0].params[..]
            {
                elements.insert(
                    element,
                    ForElement {
                        source: *source,
                        counter,
                    },
                );
            }
        }
        Self {
            cfg,
            loans,
            laws,
            defs,
            elements,
        }
    }

    fn of_target(&self, target: &RefTarget) -> Option<Place> {
        match target {
            RefTarget::Var(slot) | RefTarget::Param(slot) => Some(Place {
                slot: self.loans.storage_of(*slot).slot()?,
                path: Vec::new(),
            }),
            RefTarget::Through(reference) => self.named_by(*reference),
        }
    }

    fn named_by(&self, value: ValueId) -> Option<Place> {
        if let Some(element) = self.elements.get(&value) {
            return Some(
                self.named_by(element.source)?
                    .below([Component::At(element.counter)]),
            );
        }
        let at = self.defs.get(&value)?;
        match &self.cfg.blocks[at.block.0].insts[at.at].kind {
            InstKind::Ref { target, path, .. } => Some(
                self.of_target(target)?
                    .below(path.iter().cloned().map(Component::Named)),
            ),
            InstKind::AsSlice { container, .. } => self.named_by(*container),
            InstKind::Index {
                slice,
                index,
                mode: IndexMode::Ref,
                ..
            } => Some(self.named_by(*slice)?.below([Component::At(*index)])),
            _ => None,
        }
    }

    fn holds(&self, value: ValueId, slot: ValueId) -> bool {
        self.loans
            .holds(value)
            .any(|loan| loan.storage.slot() == Some(slot))
    }

    fn within(place: Option<Place>, slot: ValueId) -> Reach {
        match place {
            Some(place) if place.slot == slot => Reach::Places(vec![place]),
            Some(_) | None => Reach::Whole,
        }
    }

    /// `None` for an instruction that reaches nothing of `slot`, which
    /// includes a `Ref`, an `AsSlice` and an `Index` by reference: each
    /// names a place, and the instructions that read or write through what
    /// it defines reach it.
    fn reach_of_inst(&self, kind: &InstKind, slot: ValueId) -> Option<Reach> {
        let touches = touched_slots(self.loans, kind).contains(&slot)
            || written_slots(self.loans, kind).contains(&slot)
            || inst_info::uses(kind)
                .into_iter()
                .any(|used| self.holds(used, slot));
        if !touches {
            return None;
        }
        Some(match kind {
            InstKind::Ref { .. }
            | InstKind::AsSlice { .. }
            | InstKind::Index {
                mode: IndexMode::Ref,
                ..
            } => return None,
            InstKind::Index {
                slice,
                index,
                mode: IndexMode::Copy,
                ..
            }
            | InstKind::IndexSet { slice, index, .. } => Self::within(
                self.named_by(*slice)
                    .map(|place| place.below([Component::At(*index)])),
                slot,
            ),
            InstKind::Take { target, path, .. } | InstKind::Assign { target, path, .. } => {
                Self::within(
                    self.of_target(target)
                        .map(|place| place.below(path.iter().cloned().map(Component::Named))),
                    slot,
                )
            }
            InstKind::StringAppend { target, .. } => Self::within(self.named_by(*target), slot),
            InstKind::FunctionCall {
                callee: callee @ Callee::Extern { .. },
                args,
                ..
            } => match self.laws.reaches_of(callee) {
                Reaches::Lent => self.reach_of_args(args, slot),
                Reaches::Places(declared) => self.reach_of_declared(declared, args, slot),
            },
            InstKind::FunctionCall {
                callee: Callee::Direct(_),
                args,
                ..
            } => self.reach_of_args(args, slot),
            _ => Reach::Whole,
        })
    }

    /// A callee reaches, through each argument, the place it names and
    /// everything under it.
    fn reach_of_args(&self, args: &[ValueId], slot: ValueId) -> Reach {
        let mut places: Vec<Place> = Vec::new();
        for &arg in args.iter().filter(|arg| self.holds(**arg, slot)) {
            match Self::within(self.named_by(arg), slot) {
                Reach::Places(found) => places.extend(found),
                Reach::Whole => return Reach::Whole,
            }
        }
        match places.is_empty() {
            true => Reach::Whole,
            false => Reach::Places(places),
        }
    }

    /// A callee whose declaration states its places reaches, through an
    /// argument that lends `slot`, the places stated of that argument's
    /// parameter; an argument lending `slot` at a parameter of which none
    /// is stated reaches the whole storage.
    fn reach_of_declared(
        &self,
        declared: &[ReachedPlace],
        args: &[ValueId],
        slot: ValueId,
    ) -> Reach {
        let mut places: Vec<Place> = Vec::new();
        for (param, &arg) in args.iter().enumerate() {
            if !self.holds(arg, slot) {
                continue;
            }
            let stated: Vec<&ReachedPlace> = declared
                .iter()
                .filter(|place| place.param == param)
                .collect();
            if stated.is_empty() {
                return Reach::Whole;
            }
            let Some(lent) = self.named_by(arg).filter(|place| place.slot == slot) else {
                return Reach::Whole;
            };
            for place in stated {
                places.push(match place.element {
                    Some(index) => lent.clone().below([Component::At(args[index])]),
                    None => lent.clone(),
                });
            }
        }
        match places.is_empty() {
            true => Reach::Whole,
            false => Reach::Places(places),
        }
    }

    fn reach_of_term(&self, term: &Terminator, slot: ValueId) -> Option<Reach> {
        let touches = terminator_touched(self.loans, term).contains(&slot)
            || inst_info::terminator_uses(term)
                .into_iter()
                .any(|used| self.holds(used, slot));
        if !touches {
            return None;
        }
        Some(match term {
            Terminator::For {
                source: ForSource::Slice(source) | ForSource::SliceMut(source),
                ..
            } => Self::within(self.named_by(*source), slot),
            _ => Reach::Whole,
        })
    }
}

/// The storages among `written` whose every place the loop reaches lies
/// under one path component that is `a·k + b` of the counter at every
/// access, one step `a ≠ 0` and bases that differ by constants none a
/// nonzero multiple of `a`. The component's value is exact on every run
/// past it (RFC-0037 rule 3), so no two iterations reach one place.
fn disjoint_storages(
    loans: &Loans<'_>,
    laws: &LawTable,
    header: BlockIdx,
    loop_blocks: &[BlockIdx],
    written: &[ValueId],
) -> Vec<ValueId> {
    if written.is_empty() {
        return Vec::new();
    }
    let cfg = loans.cfg();
    let invariants = Invariants::of(cfg);
    let nest = LoopNest::of(cfg, &DomTree::build(cfg), &invariants);
    let Some(id) = nest.by_header(header) else {
        return Vec::new();
    };
    let affine = AffineValues::of(cfg, nest.get(id), &invariants, laws);
    let places = Places::of(loans, laws);
    let reached = |slot: ValueId| -> Option<Vec<Place>> {
        let mut found: Vec<Place> = Vec::new();
        for &block in loop_blocks {
            let held = &cfg.blocks[block.0];
            let reaches = held
                .insts
                .iter()
                .map(|inst| places.reach_of_inst(&inst.kind, slot))
                .chain([places.reach_of_term(&held.terminator, slot)]);
            for reach in reaches.flatten() {
                match reach {
                    Reach::Places(at) => found.extend(at),
                    Reach::Whole => return None,
                }
            }
        }
        Some(found)
    };
    let by_slot: Vec<(ValueId, Vec<Place>)> = written
        .iter()
        .filter_map(|slot| Some((*slot, reached(*slot)?)))
        .collect();
    let proven = Proven::of(cfg, laws, header, &affine, &by_slot);
    by_slot
        .into_iter()
        .filter(|(_, found)| one_affine_component(found, &affine, &proven))
        .map(|(slot, _)| slot)
        .collect()
}

/// What the interval domain proves, where the loop's header begins, of the
/// values an affine index's terms name: they are invariant in the loop, so
/// it holds at every access.
struct Proven {
    bounds: FxHashMap<ValueId, ConstantBounds>,
}

impl Proven {
    fn of(
        cfg: &CfgBody,
        laws: &LawTable,
        header: BlockIdx,
        affine: &AffineValues,
        reached: &[(ValueId, Vec<Place>)],
    ) -> Self {
        let mut atoms: Vec<ValueId> = Vec::new();
        for place in reached.iter().flat_map(|(_, found)| found) {
            for component in &place.path {
                let Component::At(index) = component else {
                    continue;
                };
                if let Some(found) = affine.get(*index) {
                    value_atoms(&found.step, &mut atoms);
                    value_atoms(&found.base, &mut atoms);
                }
            }
        }
        let bounds = match atoms.is_empty() {
            true => FxHashMap::default(),
            false => atoms
                .iter()
                .copied()
                .zip(crate::analysis::interval::constant_bounds_on_entry(
                    cfg, laws, header, &atoms,
                ))
                .collect(),
        };
        Self { bounds }
    }

    fn constant(&self, value: ValueId) -> Option<i128> {
        match self.bounds.get(&value) {
            Some(ConstantBounds {
                lo: Some(lo),
                hi: Some(hi),
            }) if lo == hi => Some(*lo),
            _ => None,
        }
    }

    /// The least magnitude `value` can have, where its interval excludes 0.
    fn least_magnitude(&self, value: ValueId) -> Option<i128> {
        match self.bounds.get(&value) {
            Some(ConstantBounds { lo: Some(lo), .. }) if *lo > 0 => Some(*lo),
            Some(ConstantBounds { hi: Some(hi), .. }) if *hi < 0 => hi.checked_neg(),
            _ => None,
        }
    }
}

fn value_atoms(term: &Term, into: &mut Vec<ValueId>) {
    match term {
        Term::Value(value) => {
            if !into.contains(value) {
                into.push(*value);
            }
        }
        Term::Const(_) | Term::Len(_) | Term::LenOnEntry(_) => {}
        Term::Add(a, b) | Term::Sub(a, b) | Term::Mul(a, b) | Term::Max(a, b) => {
            value_atoms(a, into);
            value_atoms(b, into);
        }
    }
}

/// An affine index's step `a`, known nonzero (RFC-0089 rule 4).
#[derive(Clone, Copy, PartialEq)]
enum Stride {
    Constant(i128),
    /// `factor` times an invariant value the interval domain proves
    /// nonzero, of magnitude at least `least`.
    Invariant {
        value: ValueId,
        factor: i128,
        least: u128,
    },
}

fn stride(step: &Term, proven: &Proven) -> Option<Stride> {
    if let Some(step) = constant(step, proven) {
        return (step != 0).then_some(Stride::Constant(step));
    }
    let linear = Linear::of(step, proven)?;
    let atoms: Vec<&(&Term, i128)> = linear
        .atoms
        .iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .collect();
    let (&[&(Term::Value(value), factor)], 0) = (&atoms[..], linear.constant) else {
        return None;
    };
    let least = proven
        .least_magnitude(*value)?
        .unsigned_abs()
        .checked_mul(factor.unsigned_abs())?;
    Some(Stride::Invariant {
        value: *value,
        factor,
        least,
    })
}

fn one_affine_component(reached: &[Place], affine: &AffineValues, proven: &Proven) -> bool {
    let Some(shortest) = reached.iter().map(|place| place.path.len()).min() else {
        return false;
    };
    let term_at = |place: &Place, position: usize| match &place.path[position] {
        Component::At(index) => affine
            .get(*index)
            .and_then(|found| Some((&found.base, stride(&found.step, proven)?))),
        Component::Named(_) => None,
    };
    (0..shortest).any(|position| {
        let terms: Option<Vec<(&Term, Stride)>> = reached
            .iter()
            .map(|place| term_at(place, position))
            .collect();
        terms.is_some_and(|terms| never_meet(&terms, proven))
    })
}

/// Terms `base + k·a` of one step `a ≠ 0` whose bases differ pairwise by
/// constants none a nonzero multiple of `a`: `b₁ + k₁·a = b₂ + k₂·a` holds
/// only where `b₁ − b₂ = (k₂ − k₁)·a`, so only at `k₁ = k₂` (RFC-0089 rule
/// 4). Of an invariant step only its least magnitude is known, and a
/// difference below it is no nonzero multiple.
fn never_meet(terms: &[(&Term, Stride)], proven: &Proven) -> bool {
    let Some(&(_, step)) = terms.first() else {
        return false;
    };
    if terms.iter().any(|(_, other)| *other != step) {
        return false;
    }
    let mut bases: Vec<&Term> = Vec::new();
    for (base, _) in terms {
        if !bases.contains(base) {
            bases.push(base);
        }
    }
    let apart = |difference: i128| match step {
        Stride::Constant(step) => {
            difference == 0 || difference.checked_rem(step).is_some_and(|rest| rest != 0)
        }
        Stride::Invariant { least, .. } => difference.unsigned_abs() < least,
    };
    bases.iter().enumerate().all(|(at, first)| {
        bases[at + 1..].iter().all(|second| {
            constant_difference(first, second, proven).is_some_and(apart)
        })
    })
}

/// `a − b` where it is a constant: the two terms as sums of constant
/// multiples of the same atoms, whose every atom cancels.
fn constant_difference(a: &Term, b: &Term, proven: &Proven) -> Option<i128> {
    let difference = Linear::of(a, proven)?.minus(Linear::of(b, proven)?)?;
    difference
        .atoms
        .iter()
        .all(|(_, coefficient)| *coefficient == 0)
        .then_some(difference.constant)
}

/// A term as `constant + Σ coefficient·atom`. An atom is a value, a
/// length, or a product or `max` no constant factors out of, compared as
/// written.
#[derive(Clone)]
struct Linear<'a> {
    constant: i128,
    atoms: Vec<(&'a Term, i128)>,
}

impl<'a> Linear<'a> {
    fn constant(constant: i128) -> Self {
        Self {
            constant,
            atoms: Vec::new(),
        }
    }

    fn atom(term: &'a Term) -> Self {
        Self {
            constant: 0,
            atoms: vec![(term, 1)],
        }
    }

    /// `None` where a coefficient leaves `i128`.
    fn of(term: &'a Term, proven: &Proven) -> Option<Self> {
        if let Some(value) = constant(term, proven) {
            return Some(Self::constant(value));
        }
        match term {
            Term::Const(_)
            | Term::Value(_)
            | Term::Len(_)
            | Term::LenOnEntry(_)
            | Term::Max(..) => Some(Self::atom(term)),
            Term::Add(a, b) => Self::of(a, proven)?.plus(Self::of(b, proven)?),
            Term::Sub(a, b) => Self::of(a, proven)?.minus(Self::of(b, proven)?),
            Term::Mul(a, b) => match (constant(a, proven), constant(b, proven)) {
                (Some(factor), _) => Self::of(b, proven)?.times(factor),
                (_, Some(factor)) => Self::of(a, proven)?.times(factor),
                (None, None) => Some(Self::atom(term)),
            },
        }
    }

    fn plus(mut self, other: Self) -> Option<Self> {
        self.constant = self.constant.checked_add(other.constant)?;
        for (atom, coefficient) in other.atoms {
            match self.atoms.iter_mut().find(|(held, _)| **held == *atom) {
                Some((_, held)) => *held = held.checked_add(coefficient)?,
                None => self.atoms.push((atom, coefficient)),
            }
        }
        Some(self)
    }

    fn minus(self, other: Self) -> Option<Self> {
        self.plus(other.times(-1)?)
    }

    fn times(mut self, factor: i128) -> Option<Self> {
        self.constant = self.constant.checked_mul(factor)?;
        for (_, coefficient) in &mut self.atoms {
            *coefficient = coefficient.checked_mul(factor)?;
        }
        Some(self)
    }
}

/// A term's value where it is a constant: its literals, and each invariant
/// value the interval domain proves one constant.
fn constant(term: &Term, proven: &Proven) -> Option<i128> {
    match term {
        Term::Const(Literal::Int(value)) => Some(*value),
        Term::Const(Literal::IntOf(suffixed)) => Some(suffixed.value),
        Term::Value(value) => proven.constant(*value),
        Term::Const(_) | Term::Len(_) | Term::LenOnEntry(_) => None,
        Term::Add(a, b) => constant(a, proven)?.checked_add(constant(b, proven)?),
        Term::Sub(a, b) => constant(a, proven)?.checked_sub(constant(b, proven)?),
        Term::Mul(a, b) => constant(a, proven)?.checked_mul(constant(b, proven)?),
        Term::Max(a, b) => Some(constant(a, proven)?.max(constant(b, proven)?)),
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
        disjoint: &[ValueId],
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
    ) -> Vec<Found> {
        let mut found: Vec<Held> = self.carried_cycles(cfg, header, loop_blocks);
        found.extend(self.storage_cycles(cfg, loans, slots, disjoint));
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
        let domtree = DomTree::build(cfg);
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
            if sent.is_empty()
                || constant_past_first(cfg, &domtree, header, param, &latches).is_some()
            {
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

    /// The storages live at the header that a member writes.
    fn written_storages(
        &self,
        cfg: &CfgBody,
        loans: &Loans<'_>,
        slots: &TargetSlots,
    ) -> Vec<ValueId> {
        let mut storages: Vec<ValueId> = Vec::new();
        for member in &self.members {
            let Member::Inst(InstAt { block, at }) = *member else {
                continue;
            };
            for slot in written_slots(loans, &cfg.blocks[block.0].insts[at].kind) {
                if let Some(Written::Storage(slot)) = slots.target_of(slot)
                    && !storages.contains(&slot)
                {
                    storages.push(slot);
                }
            }
        }
        storages
    }

    /// A storage's cycle is every member that touches it and every member
    /// between them. One no two iterations reach at one place, as the
    /// element each iteration writes at its counter, holds its writers and
    /// the members between them: a read of it reads what no other
    /// iteration writes.
    fn storage_cycles(
        &self,
        cfg: &CfgBody,
        loans: &Loans<'_>,
        slots: &TargetSlots,
        disjoint: &[ValueId],
    ) -> Vec<Held> {
        let storages = self.written_storages(cfg, loans, slots);
        let mut element_writers: Vec<usize> = Vec::new();
        let mut contexts: Vec<QualifiedRef> = Vec::new();
        for (at, member) in self.members.iter().enumerate() {
            let Member::Inst(InstAt { block, at: inst }) = *member else {
                continue;
            };
            let kind = &cfg.blocks[block.0].insts[inst].kind;
            for slot in written_slots(loans, kind) {
                if let Some(Written::Element) = slots.target_of(slot)
                    && !element_writers.contains(&at)
                {
                    element_writers.push(at);
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
            let from = match disjoint.contains(&slot) {
                true => writers.clone(),
                false => touchers,
            };
            let mut members = self.between(&from, &writers);
            members.extend(from);
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
}

/// RFC-0089 rule 2: the one constant every back edge sends header parameter
/// `param`, which it holds past the first iteration, so that it is no
/// token. An edge sends it the constant `c` where the value it passes is
/// `c`'s `Const`, a block parameter every edge into which sends `c`, or
/// `param` itself on an edge only a branch deciding by `param` being `c`
/// reaches.
fn constant_past_first(
    cfg: &CfgBody,
    domtree: &DomTree,
    header: BlockIdx,
    param: ValueId,
    latches: &[BlockIdx],
) -> Option<Literal> {
    let header_label = cfg.blocks[header.0].label;
    let Some(index) = cfg.blocks[header.0]
        .params
        .iter()
        .position(|held| *held == param)
    else {
        return None;
    };
    let mut constants: Vec<Literal> = Vec::new();
    let mut fixed_by_branch: Vec<(BlockIdx, usize)> = Vec::new();
    for latch in latches {
        for (edge_at, edge) in edges(&cfg.blocks[latch.0].terminator).into_iter().enumerate() {
            if edge.target != header_label {
                continue;
            }
            let Some(&arg) = index
                .checked_sub(edge.fills_from)
                .and_then(|position| edge.args.get(position))
            else {
                return None;
            };
            let mut seen: Vec<ValueId> = Vec::new();
            if !constants_sent(
                cfg,
                header,
                param,
                (*latch, edge_at),
                arg,
                &mut seen,
                &mut constants,
                &mut fixed_by_branch,
            ) {
                return None;
            }
        }
    }
    let [constant] = &constants[..] else {
        return None;
    };
    let fixed = match constant {
        Literal::Bool(held) => fixed_by_branch
            .iter()
            .all(|&(from, edge)| decided_to(cfg, domtree, param, *held, from, edge)),
        _ => fixed_by_branch.is_empty(),
    };
    fixed.then(|| constant.clone())
}

/// The literal a `Const` anywhere in the body defines `value` as.
fn constant_of(cfg: &CfgBody, value: ValueId) -> Option<&Literal> {
    cfg.blocks
        .iter()
        .flat_map(|block| &block.insts)
        .find_map(|inst| match &inst.kind {
            InstKind::Const { dst, value: literal } if *dst == value => Some(literal),
            _ => None,
        })
}

fn int_constant(cfg: &CfgBody, value: ValueId) -> Option<i128> {
    match constant_of(cfg, value) {
        Some(Literal::Int(held)) => Some(*held),
        _ => None,
    }
}

/// Gathers the constants `value`, passed by the edge `via` (a block and
/// the index of its terminator's edge), stands for; each place it is
/// `param` itself is gathered as the edge that passes it.
#[allow(clippy::too_many_arguments)]
fn constants_sent(
    cfg: &CfgBody,
    header: BlockIdx,
    param: ValueId,
    via: (BlockIdx, usize),
    value: ValueId,
    seen: &mut Vec<ValueId>,
    constants: &mut Vec<Literal>,
    fixed_by_branch: &mut Vec<(BlockIdx, usize)>,
) -> bool {
    if value == param {
        fixed_by_branch.push(via);
        return true;
    }
    if seen.contains(&value) {
        return false;
    }
    seen.push(value);
    let constant = cfg.blocks.iter().flat_map(|block| &block.insts).find_map(|inst| {
        match &inst.kind {
            InstKind::Const { dst, value: literal } if *dst == value => Some(literal.clone()),
            _ => None,
        }
    });
    if let Some(constant) = constant {
        if !constants.contains(&constant) {
            constants.push(constant);
        }
        return true;
    }
    let owner = (0..cfg.blocks.len()).map(BlockIdx).find_map(|block| {
        cfg.blocks[block.0]
            .params
            .iter()
            .position(|held| *held == value)
            .map(|at| BlockParam { block, at })
    });
    let Some(BlockParam { block, at }) = owner.filter(|owner| owner.block != header) else {
        return false;
    };
    let label = cfg.blocks[block.0].label;
    let preds = cfg.predecessors();
    for pred in preds.get(&block).into_iter().flatten() {
        for (edge_at, edge) in edges(&cfg.blocks[pred.0].terminator).into_iter().enumerate() {
            if edge.target != label {
                continue;
            }
            let Some(&arg) = at
                .checked_sub(edge.fills_from)
                .and_then(|position| edge.args.get(position))
            else {
                return false;
            };
            let via = (*pred, edge_at);
            if !constants_sent(cfg, header, param, via, arg, seen, constants, fixed_by_branch) {
                return false;
            }
        }
    }
    true
}

/// Whether edge `edge` of `from`'s terminator is taken only where a
/// branch on `param` itself found it `held`: it is that branch's edge on
/// `held`'s side, or `from` lies under that side's block, which only the
/// branch enters.
fn decided_to(
    cfg: &CfgBody,
    domtree: &DomTree,
    param: ValueId,
    held: bool,
    from: BlockIdx,
    edge: usize,
) -> bool {
    let side_edge = match held {
        true => 0,
        false => 1,
    };
    let preds = cfg.predecessors();
    (0..cfg.blocks.len()).map(BlockIdx).any(|branch| {
        let (Terminator::JumpIf {
            cond,
            then_label,
            else_label,
            ..
        }
        | Terminator::Diamond {
            cond,
            then_label,
            else_label,
            ..
        }) = &cfg.blocks[branch.0].terminator
        else {
            return false;
        };
        if *cond != param || then_label == else_label {
            return false;
        }
        if branch == from {
            return edge == side_edge;
        }
        let side = cfg.label_to_block[match held {
            true => then_label,
            false => else_label,
        }];
        let entered_by_the_branch_alone =
            preds.get(&side).map(|from| from.as_slice()) == Some(&[branch][..]);
        entered_by_the_branch_alone && (side == from || domtree.dominates(side, from))
    })
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

// -- A law read from what a cycle computes (rule 4) -----------------------

/// What a value of one iteration is, in terms of a token's state as the
/// iteration received it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Form<'a> {
    /// Reads nothing of the state.
    Free,
    /// The state as the iteration received it.
    State,
    /// The state combined through one law with values that read nothing
    /// of it, the state on the left.
    Combined(Step<'a>),
}

/// One law a cycle combines its state through.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Step<'a> {
    Op {
        op: LawOp,
        exact: bool,
    },
    Call {
        callee: ExternInstance,
        law: &'a ResolvedBinary,
    },
    Order,
    Last,
    OptionLifted(Lifted<'a>),
    Ordered {
        op: LawOp,
        order: ExternInstance,
    },
}

/// A law an `Option` token's payload combines through.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Lifted<'a> {
    Op {
        op: LawOp,
        exact: bool,
    },
    Call {
        callee: ExternInstance,
        law: &'a ResolvedBinary,
    },
}

impl<'a> Lifted<'a> {
    fn step(self) -> Step<'a> {
        match self {
            Self::Op { op, exact } => Step::Op { op, exact },
            Self::Call { callee, law } => Step::Call { callee, law },
        }
    }
}

impl Step<'_> {
    fn accumulator(self) -> Accumulator {
        match self {
            Self::Op { op, exact } => Accumulator {
                law: Law::Op(op),
                exact,
                commutative: op.commutes(),
            },
            Self::Call { callee, law } => Accumulator {
                law: Law::Call(CallLaw {
                    callee,
                    identity: match &law.identity {
                        Some(identity) => CallIdentity::Declared(identity.clone()),
                        None => CallIdentity::OptionLifted,
                    },
                }),
                exact: true,
                commutative: law.commutative,
            },
            Self::Order => Accumulator {
                law: Law::Order,
                exact: true,
                commutative: true,
            },
            Self::Last => Accumulator {
                law: Law::Last,
                exact: true,
                commutative: false,
            },
            Self::OptionLifted(inner) => {
                let Accumulator {
                    law,
                    exact,
                    commutative,
                } = inner.step().accumulator();
                Accumulator {
                    law: Law::OptionLifted(Box::new(law)),
                    exact,
                    commutative,
                }
            }
            Self::Ordered { op, order } => Accumulator {
                law: Law::Ordered { op, order },
                exact: true,
                commutative: true,
            },
        }
    }
}

impl<'a> Form<'a> {
    /// This form combined once more through `step`, the state still on the
    /// left: the state or a combination through the same law.
    fn then(self, step: Step<'a>) -> Option<Form<'a>> {
        match self {
            Self::State => Some(Self::Combined(step)),
            Self::Combined(previous) if previous == step => Some(Self::Combined(step)),
            Self::Combined(_) | Self::Free => None,
        }
    }

    fn reads_state(self) -> bool {
        !matches!(self, Self::Free)
    }
}

/// What a law reading finds: the step, whether an arm taken only at the
/// first iteration resets the state, and the values that read the state.
struct Update<'a> {
    step: Step<'a>,
    resets: bool,
    reading_state: FxHashSet<ValueId>,
    chain: FxHashSet<ValueId>,
}

impl Update<'_> {
    fn accumulator(&self) -> Accumulator {
        let Accumulator {
            law,
            exact,
            commutative,
        } = self.step.accumulator();
        let law = match self.resets {
            true => Law::Reset(Box::new(law)),
            false => law,
        };
        Accumulator {
            law,
            exact,
            commutative,
        }
    }
}

/// The state a cycle's law is read over: a header parameter, or a storage
/// the body loads and stores whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Param { param: ValueId, index: usize },
    Slot(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Def {
    Inst(InstAt),
    Param { block: BlockIdx, index: usize },
}

/// Which of a two-way branch's edges an iteration leaves by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Then,
    Else,
}

/// How a `first` reads its guard unset: a `Bool` flag that is `false`, or
/// an integer that holds the constant `ValueId` (RFC-0093 rules 5 and 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unset {
    Flag,
    Sentinel(ValueId),
}

/// What the edges into a join send one of its parameters.
struct Sent {
    sent: Vec<(BlockIdx, Side, ValueId)>,
    deciders: Vec<BlockIdx>,
}

/// A compare of two values under the total order an extern states, as the
/// sign of its call against `0` spells it: `left op right`.
struct OrderedCompare {
    op: BinOp,
    sign: ValueId,
    order: ExternInstance,
    left: ValueId,
    right: ValueId,
}

/// The side of a compare and select that yields a value of the iteration,
/// not the state.
struct Select {
    branch: BlockIdx,
    chosen: ValueId,
    chosen_on: Side,
}

/// Reads, for one token of one loop, whether every iteration's update of
/// its state is the state combined through one law with values that read
/// nothing of it, and that nothing else of the iteration reads the state or
/// a partial of it.
struct LawReading<'a, 's, 'cfg> {
    cfg: &'cfg CfgBody,
    loans: &'a Loans<'cfg>,
    laws: &'a LawTable,
    slots: &'s TargetSlots,
    header: BlockIdx,
    /// The header and the body: every block an iteration runs.
    blocks: Vec<BlockIdx>,
    defs: FxHashMap<ValueId, Def>,
    deciders: FxHashMap<BlockIdx, Vec<BlockIdx>>,
    domtree: DomTree,
    /// The natural loops of the body the loop lies in, innermost first.
    loops: Vec<NaturalLoop>,
    state: State,
    /// Every value of the iteration that reads the state, and every storage
    /// the iteration defines and drops that holds such a value.
    dependent: FxHashSet<ValueId>,
    /// The values the update reaches the state through, and the compares
    /// that choose between the state and a value of the iteration.
    chain: FxHashSet<ValueId>,
    forms: FxHashMap<ValueId, Option<Form<'a>>>,
    /// Where a storage state is assigned more than once in an iteration,
    /// what it holds where each block begins.
    entry_forms: FxHashMap<BlockIdx, Option<Form<'a>>>,
    assigned_more_than_once: bool,
    /// Whether a join took a value reading none of the state from an arm
    /// taken only at the first iteration (RFC-0089 rule 4's reset).
    resets: bool,
    /// The constant an integer state is unset at while `first` is read with
    /// it as its own guard (RFC-0093 rule 7).
    sentinel: Option<i128>,
}

impl<'a, 's, 'cfg> LawReading<'a, 's, 'cfg> {
    fn of(
        loans: &'a Loans<'cfg>,
        laws: &'a LawTable,
        slots: &'s TargetSlots,
        header: BlockIdx,
        loop_blocks: &[BlockIdx],
        state: State,
    ) -> Self {
        let cfg = loans.cfg();
        let body: Vec<BlockIdx> = loop_blocks
            .iter()
            .copied()
            .filter(|block| *block != header)
            .collect();
        let inside = |block: BlockIdx| body.contains(&block);
        let deciders: FxHashMap<BlockIdx, Vec<BlockIdx>> = control_dependence(cfg, &body, &inside)
            .into_iter()
            .map(|Controlled { block, deciders }| (block, deciders))
            .collect();
        let mut blocks = vec![header];
        blocks.extend(body.iter().copied());
        let mut defs: FxHashMap<ValueId, Def> = FxHashMap::default();
        for &block in &blocks {
            for (at, inst) in cfg.blocks[block.0].insts.iter().enumerate() {
                for def in inst_info::defs(&inst.kind) {
                    defs.insert(def, Def::Inst(InstAt { block, at }));
                }
            }
            if block != header {
                for (index, &param) in cfg.blocks[block.0].params.iter().enumerate() {
                    defs.insert(param, Def::Param { block, index });
                }
            }
        }
        let domtree = DomTree::build(cfg);
        let loops = natural_loops_innermost_first(cfg, &domtree);
        let mut reading = LawReading {
            cfg,
            loans,
            laws,
            slots,
            header,
            blocks,
            defs,
            deciders,
            domtree,
            loops,
            state,
            dependent: FxHashSet::default(),
            chain: FxHashSet::default(),
            forms: FxHashMap::default(),
            entry_forms: FxHashMap::default(),
            assigned_more_than_once: false,
            resets: false,
            sentinel: None,
        };
        reading.dependent = reading.values_reading_state(None);
        reading
    }

    fn body(&self) -> impl Iterator<Item = BlockIdx> + '_ {
        self.blocks
            .iter()
            .copied()
            .filter(|block| *block != self.header)
    }

    fn inst(&self, at: InstAt) -> &'cfg InstKind {
        &self.cfg.blocks[at.block.0].insts[at.at].kind
    }

    fn insts(&self) -> impl Iterator<Item = (InstAt, &'cfg InstKind)> + '_ {
        let cfg = self.cfg;
        self.blocks.iter().flat_map(move |&block| {
            cfg.blocks[block.0]
                .insts
                .iter()
                .enumerate()
                .map(move |(at, inst)| (InstAt { block, at }, &inst.kind))
        })
    }

    fn is_state_slot(&self, target: &RefTarget, path: &[crate::ir::PathSeg]) -> bool {
        matches!(self.state, State::Slot(slot) if inst_info::storage(target) == Some(slot))
            && path.is_empty()
    }

    fn is_local(&self, slot: ValueId) -> bool {
        self.slots.target_of(slot).is_none()
    }

    /// The values a branch decides by.
    fn decision(term: &Terminator) -> Vec<ValueId> {
        match term {
            Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } => vec![*cond],
            Terminator::Switch { tag, .. } => vec![*tag],
            // A jump leaves by its one edge whatever it sends.
            Terminator::Jump { .. } => Vec::new(),
            // A traversal takes its body or its exit by its counter against
            // its source; the values its exit edge passes decide nothing.
            Terminator::For { source, .. } => source.uses().into_iter().collect(),
            other => inst_info::terminator_uses(other).into_iter().collect(),
        }
    }

    /// Every value of the iteration that reads the state: through what it
    /// reads, through a storage the iteration defines and drops that was
    /// given such a value, and through a branch that decides by one other
    /// than the branch ending `excused`.
    fn values_reading_state(&self, excused: Option<BlockIdx>) -> FxHashSet<ValueId> {
        let mut dependent: FxHashSet<ValueId> = FxHashSet::default();
        if let State::Param { param, .. } = self.state {
            dependent.insert(param);
        }
        loop {
            let mut changed = false;
            for (at, kind) in self.insts() {
                let loads_state = match kind {
                    InstKind::Take { target, .. } | InstKind::Ref { target, .. } => matches!(
                        self.state,
                        State::Slot(slot) if inst_info::storage(target) == Some(slot)
                    ),
                    _ => false,
                };
                let reads_local = match kind {
                    InstKind::Take { target, .. } | InstKind::Ref { target, .. } => {
                        inst_info::storage(target).is_some_and(|slot| dependent.contains(&slot))
                    }
                    _ => false,
                };
                let reads = inst_info::uses(kind)
                    .into_iter()
                    .any(|used| dependent.contains(&used));
                let decided = self.decided_by_state(at.block, &dependent, excused);
                if !(loads_state || reads_local || reads || decided) {
                    continue;
                }
                for def in inst_info::defs(kind) {
                    changed |= dependent.insert(def);
                }
                // A storage given a value that reads the state holds it.
                for slot in effect(self.loans, kind)
                    .writes
                    .into_iter()
                    .chain(slots_lent_mutably(self.loans, kind))
                    .chain(match kind {
                        InstKind::Assign { target, .. } => inst_info::storage(target),
                        _ => None,
                    })
                {
                    if Some(slot) != self.state_slot() && (reads || decided) {
                        changed |= dependent.insert(slot);
                    }
                }
            }
            for block in self.body().collect::<Vec<_>>() {
                let term = &self.cfg.blocks[block.0].terminator;
                let decides_by_state = Some(block) != excused
                    && Self::decision(term)
                        .iter()
                        .any(|value| dependent.contains(value));
                for edge in edges(term) {
                    let Some(&target) = self.cfg.label_to_block.get(&edge.target) else {
                        continue;
                    };
                    if target == self.header || !self.blocks.contains(&target) {
                        continue;
                    }
                    let params = &self.cfg.blocks[target.0].params;
                    for (at, &param) in params.iter().enumerate().skip(edge.fills_from) {
                        let sent = edge.args[at - edge.fills_from];
                        let decided =
                            decides_by_state || self.decided_by_state(block, &dependent, excused);
                        if dependent.contains(&sent) || decided {
                            changed |= dependent.insert(param);
                        }
                    }
                }
            }
            if !changed {
                return dependent;
            }
        }
    }

    fn state_slot(&self) -> Option<ValueId> {
        match self.state {
            State::Slot(slot) => Some(slot),
            State::Param { .. } => None,
        }
    }

    /// Whether a branch that decides whether `block` runs reads the state.
    fn decided_by_state(
        &self,
        block: BlockIdx,
        dependent: &FxHashSet<ValueId>,
        excused: Option<BlockIdx>,
    ) -> bool {
        self.deciders
            .get(&block)
            .into_iter()
            .flatten()
            .filter(|decider| Some(**decider) != excused)
            .any(|decider| {
                Self::decision(&self.cfg.blocks[decider.0].terminator)
                    .iter()
                    .any(|value| dependent.contains(value))
            })
    }

    /// The law the state is updated through, where the update reads the
    /// state only through it and nothing else of the iteration reads the
    /// state or a partial of it.
    fn law(self) -> Option<Accumulator> {
        self.update().map(|update| update.accumulator())
    }

    /// The step every iteration combines the state through, and every value
    /// of the iteration that reads the state, each of which the update
    /// accounts for.
    fn update(mut self) -> Option<Update<'a>> {
        let next = match self.state {
            State::Param { index, .. } => {
                let loop_ = self
                    .loops
                    .iter()
                    .find(|loop_| loop_.header == self.header)?;
                let next = loop_.back_arg(self.cfg, index)?;
                self.form(next)?
            }
            State::Slot(slot) => self.stored(slot)?,
        };
        let Form::Combined(step) = next else {
            return None;
        };
        let slots = slot_values(self.cfg);
        // A storage the iteration defines and drops is read only through
        // the values its reads define, which are checked themselves; any
        // other storage given a partial is a second target.
        let partial_read_elsewhere =
            self.dependent
                .iter()
                .any(|value| match slots.contains(value) {
                    true => Some(*value) != self.state_slot() && !self.is_local(*value),
                    false => !self.chain.contains(value),
                });
        let other_header_args_read_state = self.blocks.iter().any(|&block| {
            edges(&self.cfg.blocks[block.0].terminator)
                .into_iter()
                .filter(|edge| edge.target == self.cfg.blocks[self.header.0].label)
                .any(|edge| {
                    edge.args.iter().enumerate().any(|(at, arg)| {
                        let own = matches!(self.state,
                            State::Param { index, .. } if at + edge.fills_from == index);
                        !own && self.dependent.contains(arg)
                    })
                })
        });
        (!partial_read_elsewhere && !other_header_args_read_state).then_some(Update {
            step,
            resets: self.resets,
            reading_state: self.dependent,
            chain: self.chain,
        })
    }

    /// The form of the value the storage holds when the iteration ends. With
    /// one store: that store, loaded before it only as the update's operand,
    /// run where every branch it lies under decides by values that read
    /// nothing of the state, or as the chosen arm of a compare and select.
    /// With several: what the latch holds at its end, each assignment read
    /// over what its block holds before it (RFC-0089 rule 4, "the chain of
    /// all the iteration's assignments").
    fn stored(&mut self, slot: ValueId) -> Option<Form<'a>> {
        let mut stores: Vec<(InstAt, ValueId)> = Vec::new();
        for (at, kind) in self.insts() {
            let touched = touched_slots(self.loans, kind);
            let written = written_slots(self.loans, kind);
            match kind {
                InstKind::Assign {
                    target,
                    path,
                    value,
                    restores: false,
                } if inst_info::storage(target) == Some(slot) => {
                    if !path.is_empty() {
                        return None;
                    }
                    stores.push((at, *value));
                }
                InstKind::Take {
                    target,
                    path,
                    taken_out: false,
                    ..
                }
                | InstKind::Ref {
                    target,
                    path,
                    mutability: crate::ty::Mutability::Shared,
                    ..
                } if inst_info::storage(target) == Some(slot) => {
                    let whole_or_payload =
                        path.is_empty() || matches!(path[..], [crate::ir::PathSeg::Payload]);
                    if !whole_or_payload || written.contains(&slot) {
                        return None;
                    }
                }
                _ if written.contains(&slot) => return None,
                // Read through a loan of the state: the reader is a value of
                // the iteration that reads it, which the chain accounts for.
                _ if touched.contains(&slot) => {
                    let through_chain = inst_info::uses(kind)
                        .into_iter()
                        .any(|used| self.dependent.contains(&used));
                    if !through_chain {
                        return None;
                    }
                }
                _ => {}
            }
        }
        let (at, value) = match stores[..] {
            [] => return None,
            [one] => one,
            _ => {
                if stores
                    .iter()
                    .any(|(at, _)| !self.nested_loops_holding(at.block).is_empty())
                {
                    return None;
                }
                self.assigned_more_than_once = true;
                let [latch] = self.latches()[..] else {
                    return None;
                };
                let end = self.cfg.blocks[latch.0].insts.len();
                return self.held_before(InstAt {
                    block: latch,
                    at: end,
                });
            }
        };
        match self
            .deciders
            .get(&at.block)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            [] => self.form(value),
            deciders if self.decide_freely(deciders) => {
                let form = self.form(value)?;
                form.reads_state().then_some(form)
            }
            [branch] => {
                let side = self.side_reaching(*branch, at.block)?;
                self.select(&Select {
                    branch: *branch,
                    chosen: value,
                    chosen_on: side,
                })
            }
            _ => None,
        }
    }

    /// The blocks of the body whose terminator jumps to the header.
    fn latches(&self) -> Vec<BlockIdx> {
        self.body()
            .filter(|block| self.cfg.successors(*block).contains(&self.header))
            .collect()
    }

    /// What the storage state holds just before `at`: the value of the last
    /// assignment before it in its block, and otherwise what its block holds
    /// where it begins.
    fn held_before(&mut self, at: InstAt) -> Option<Form<'a>> {
        match self.assigned_before(at) {
            Some(value) => self.form(value),
            None => self.held_at_entry(at.block),
        }
    }

    /// The value of the last assignment of the storage state in `at`'s block
    /// before `at`.
    fn assigned_before(&self, at: InstAt) -> Option<ValueId> {
        let State::Slot(slot) = self.state else {
            return None;
        };
        self.cfg.blocks[at.block.0].insts[..at.at]
            .iter()
            .rev()
            .find_map(|inst| match &inst.kind {
                InstKind::Assign { target, value, .. }
                    if inst_info::storage(target) == Some(slot) =>
                {
                    Some(*value)
                }
                _ => None,
            })
    }

    fn body_label(&self) -> Label {
        let Terminator::For { stages, .. } = &self.cfg.blocks[self.header.0].terminator else {
            panic!("block {} heads the `For` a law is read of", self.header.0)
        };
        stages.body()
    }

    /// What the storage state holds where `block` begins: the state itself
    /// at the body's entry, and otherwise what its predecessors hold at
    /// their ends, joined as a join's parameter is. A predecessor `block`
    /// dominates is a nested loop's back edge, and that loop assigns no
    /// state.
    fn held_at_entry(&mut self, block: BlockIdx) -> Option<Form<'a>> {
        if block == self.cfg.label_to_block[&self.body_label()] {
            return Some(Form::State);
        }
        if let Some(known) = self.entry_forms.get(&block) {
            return *known;
        }
        // A path back to `block` holds none of the state's forms.
        self.entry_forms.insert(block, None);
        let preds: Vec<BlockIdx> = self
            .cfg
            .predecessors()
            .get(&block)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|pred| *pred != self.header && self.blocks.contains(pred))
            .filter(|pred| !self.domtree.dominates(block, *pred))
            .collect();
        let form = self.held_at_join(block, &preds);
        self.entry_forms.insert(block, form);
        form
    }

    fn held_at_join(&mut self, block: BlockIdx, preds: &[BlockIdx]) -> Option<Form<'a>> {
        let mut sent: Vec<(Form<'a>, Option<ValueId>, BlockIdx)> = Vec::new();
        for &pred in preds {
            let end = InstAt {
                block: pred,
                at: self.cfg.blocks[pred.0].insts.len(),
            };
            sent.push((self.held_before(end)?, self.assigned_before(end), pred));
        }
        match preds {
            [] => None,
            [_] => Some(sent[0].0),
            _ => {
                let deciders = self.join_deciders(block, preds);
                match self.decide_freely(&deciders) {
                    true => self.joined(&sent),
                    false => None,
                }
            }
        }
    }

    /// The branches that decide which of `preds` reaches the join `block`,
    /// other than those that decide whether `block` runs at all.
    fn join_deciders(&self, block: BlockIdx, preds: &[BlockIdx]) -> Vec<BlockIdx> {
        let own: Vec<BlockIdx> = self.deciders.get(&block).cloned().unwrap_or_default();
        let mut deciders: Vec<BlockIdx> = Vec::new();
        for &from in preds {
            let mut by = self.deciders.get(&from).cloned().unwrap_or_default();
            if self.cfg.successors(from).len() > 1 {
                by.push(from);
            }
            for decider in by {
                if !own.contains(&decider) && !deciders.contains(&decider) {
                    deciders.push(decider);
                }
            }
        }
        deciders
    }

    /// The forms a join receives from the blocks it names, joined where
    /// every branch choosing among them reads nothing of the state: one law
    /// the forms that read the state combine through, which a value reading
    /// none of it stands in for as that law allows (see
    /// [`Self::free_arm_step`]), or which it resets from an arm taken only
    /// at the first iteration (see [`Self::taken_only_first`]).
    fn joined(&mut self, sent: &[(Form<'a>, Option<ValueId>, BlockIdx)]) -> Option<Form<'a>> {
        let mut step: Option<Step<'a>> = None;
        let mut reads_state = false;
        let mut free: Vec<Option<ValueId>> = Vec::new();
        let mut resets = false;
        for &(form, value, from) in sent {
            match form {
                Form::Free if self.taken_only_first(from) => resets = true,
                Form::Free => free.push(value),
                Form::State => reads_state = true,
                Form::Combined(held) => {
                    if step.is_some_and(|step| step != held) {
                        return None;
                    }
                    step = Some(held);
                    reads_state = true;
                }
            }
        }
        if !reads_state {
            return None;
        }
        self.resets |= resets;
        if free.is_empty() {
            return Some(step.map_or(Form::State, Form::Combined));
        }
        let free_step = self.free_arm_step(&free, step)?;
        match step {
            Some(held) if held != free_step => None,
            _ => Some(Form::Combined(free_step)),
        }
    }

    /// Whether `block` runs at the first iteration and at no other: the one
    /// branch deciding it runs in every iteration and decides by a header
    /// parameter other than the state, which enters as a `Bool` and holds its
    /// negation past the first iteration (RFC-0089 rule 2), and `block` lies
    /// on the side the entry value takes.
    fn taken_only_first(&self, block: BlockIdx) -> bool {
        let [branch] = self.deciders.get(&block).map(Vec::as_slice).unwrap_or(&[])[..] else {
            return false;
        };
        if branch == block || self.deciders.get(&branch).is_some_and(|by| !by.is_empty()) {
            return false;
        }
        let (Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. }) =
            &self.cfg.blocks[branch.0].terminator
        else {
            return false;
        };
        let flag = copied_from(self.cfg, self.header, *cond);
        let params = &self.cfg.blocks[self.header.0].params;
        let Some(index) = params.iter().position(|param| *param == flag) else {
            return false;
        };
        if matches!(self.state, State::Param { param, .. } if param == flag) {
            return false;
        }
        let Some(loop_) = self.loops.iter().find(|loop_| loop_.header == self.header) else {
            return false;
        };
        let entered = loop_
            .entry_arg(self.cfg, index)
            .and_then(|entry| constant_of(self.cfg, entry));
        let past_first =
            constant_past_first(self.cfg, &self.domtree, self.header, flag, &self.latches());
        let (Some(Literal::Bool(entered)), Some(Literal::Bool(past_first))) = (entered, past_first)
        else {
            return false;
        };
        let side = match entered {
            true => Side::Then,
            false => Side::Else,
        };
        *entered != past_first && self.side_reaching(branch, block) == Some(side)
    }

    /// The law an arm that sends the token a value reading none of it reads
    /// as, the other arms leaving the token or combining it through `step`:
    /// over `Bool`, an arm that fixes it to `true` is `‖` of that arm's test
    /// and one that fixes it to `false` is `&&` of its negation; otherwise
    /// the arm is `last` (RFC-0089 rule 4).
    fn free_arm_step(
        &self,
        free: &[Option<ValueId>],
        step: Option<Step<'a>>,
    ) -> Option<Step<'a>> {
        let constants: Option<Vec<bool>> = free
            .iter()
            .map(|value| value.and_then(|value| self.bool_constant(value)))
            .collect();
        let fixed = match constants.as_deref() {
            Some([first, rest @ ..]) if rest.iter().all(|other| other == first) => Some(*first),
            _ => None,
        };
        let bool_law = |op: LawOp| Step::Op { op, exact: true };
        match (fixed, step) {
            (Some(true), None | Some(Step::Op { op: LawOp::Or, .. })) => Some(bool_law(LawOp::Or)),
            (Some(false), None | Some(Step::Op { op: LawOp::And, .. })) => {
                Some(bool_law(LawOp::And))
            }
            (_, None | Some(Step::Last)) => Some(Step::Last),
            _ => None,
        }
    }

    /// The `Bool` a value is where a `Const` of the iteration defines it.
    fn bool_constant(&self, value: ValueId) -> Option<bool> {
        let Some(Def::Inst(at)) = self.defs.get(&value) else {
            return None;
        };
        match self.inst(*at) {
            InstKind::Const {
                value: Literal::Bool(held),
                ..
            } => Some(*held),
            _ => None,
        }
    }

    /// Whether each of `deciders` decides by values that read nothing of
    /// the state.
    fn decide_freely(&self, deciders: &[BlockIdx]) -> bool {
        deciders.iter().all(|decider| {
            Self::decision(&self.cfg.blocks[decider.0].terminator)
                .iter()
                .all(|value| !self.dependent.contains(value))
        })
    }

    /// The edge of two-way `branch` whose side reaches `block`.
    fn side_reaching(&self, branch: BlockIdx, block: BlockIdx) -> Option<Side> {
        let (Terminator::JumpIf {
            then_label,
            else_label,
            ..
        }
        | Terminator::Diamond {
            then_label,
            else_label,
            ..
        }) = &self.cfg.blocks[branch.0].terminator
        else {
            return None;
        };
        let reaches = |label: &Label| {
            let first = self.cfg.label_to_block[label];
            first == block || self.domtree.dominates(first, block)
        };
        match (reaches(then_label), reaches(else_label)) {
            (true, false) => Some(Side::Then),
            (false, true) => Some(Side::Else),
            _ => None,
        }
    }

    fn form(&mut self, value: ValueId) -> Option<Form<'a>> {
        if !self.dependent.contains(&value) {
            return Some(Form::Free);
        }
        if let Some(known) = self.forms.get(&value) {
            return *known;
        }
        // A value on the way to itself is no update of one iteration.
        self.forms.insert(value, None);
        let form = self.computed_form(value);
        self.forms.insert(value, form);
        if form.is_some() {
            self.chain.insert(value);
        }
        form
    }

    fn computed_form(&mut self, value: ValueId) -> Option<Form<'a>> {
        if matches!(self.state, State::Param { param, .. } if param == value) {
            return Some(Form::State);
        }
        let def = *self.defs.get(&value)?;
        let block = match def {
            Def::Inst(at) => at.block,
            Def::Param { block, .. } => block,
        };
        if let State::Param { .. } = self.state {
            match self.nested_loops_holding(block)[..] {
                [] => {}
                // A header parameter of a loop nested one level down, read
                // where the nested loop has handed it back.
                [nested] if self.loops[nested].header == block => {
                    let Def::Param { index, .. } = def else {
                        return None;
                    };
                    return self.through_nested_loop(nested, index);
                }
                // A value of a nested loop's iteration is none of this
                // iteration's: the nested loop runs it many times.
                _ => return None,
            }
        }
        match def {
            Def::Inst(at) => self.inst_form(at),
            Def::Param { block, index } => self.param_form(block, index),
        }
    }

    /// The loops nested in this one whose blocks hold `block`.
    fn nested_loops_holding(&self, block: BlockIdx) -> Vec<usize> {
        (0..self.loops.len())
            .filter(|at| {
                let loop_ = &self.loops[*at];
                loop_.header != self.header
                    && self.blocks.contains(&loop_.header)
                    && loop_.contains(block)
            })
            .collect()
    }

    /// RFC-0089 rule 4, through a nested loop: header parameter `index` of
    /// `nested`, a `for`, entered with a value of this iteration and read
    /// where the nested loop holds it or has handed it back. When that
    /// parameter's cycle in the nested loop has law `L` and nothing else
    /// there reads it, the parameter at the start of the nested loop's
    /// `k`-th iteration is the entry value combined through `L` with the
    /// run of its first `k - 1` iterations from `L`'s identity:
    /// `((e ⊕ y₁) ⊕ y₂) … = e ⊕ (y₁ ⊕ y₂ …)` by associativity, and `e`
    /// itself where the run is empty.
    ///
    /// This reading does not ask how the nested loop leaves, and that is a
    /// decision. The identity holds at every iteration, so an exit that hands
    /// on the parameter hands on the entry value combined with a run from the
    /// identity. An exit that hands on some other value passes it into a
    /// block parameter, which is read by its own form, and a `return` hands
    /// nothing back to this loop. The tests in `acvus-mir-test/tests/stages.rs`
    /// that name an inner loop leaving early pin each of these cases.
    fn through_nested_loop(&mut self, nested: usize, index: usize) -> Option<Form<'a>> {
        let cfg = self.cfg;
        let nested = &self.loops[nested];
        let header = nested.header;
        let Terminator::For { source, .. } = &cfg.blocks[header.0].terminator else {
            return None;
        };
        let nested_blocks: Vec<BlockIdx> = nested.blocks().collect();
        let entry = self.form(nested.entry_arg(cfg, index)?)?;
        let param = cfg.blocks[header.0].params[index];
        let slots = TargetSlots::of(self.loans, *source, &nested_blocks);
        let Update {
            step,
            resets,
            reading_state,
            ..
        } = LawReading::of(
            self.loans,
            self.laws,
            &slots,
            header,
            &nested_blocks,
            State::Param { param, index },
        )
        .update()?;
        // A nested loop that resets the token drops the entry value, which
        // this combination keeps.
        if resets {
            return None;
        }
        let combined = entry.then(step)?;
        // What the nested loop computes of the state is its update, which
        // the reading there accounts for; this reading reads it as a whole.
        self.chain.extend(reading_state);
        Some(combined)
    }

    fn inst_form(&mut self, at: InstAt) -> Option<Form<'a>> {
        let kind = self.inst(at);
        match kind {
            InstKind::Take {
                target,
                path,
                taken_out: false,
                ..
            }
            | InstKind::Ref {
                target,
                path,
                mutability: crate::ty::Mutability::Shared,
                ..
            } => match inst_info::storage(target) {
                Some(_) if self.is_state_slot(target, path) => match self.assigned_more_than_once {
                    true => self.held_before(at),
                    false => Some(Form::State),
                },
                Some(slot) if self.is_local(slot) && path.is_empty() => self.held(slot),
                Some(_) => None,
                None => {
                    let RefTarget::Through(reference) = target else {
                        return None;
                    };
                    match path.is_empty() {
                        true => self.form(*reference),
                        false => None,
                    }
                }
            },
            InstKind::BinOp {
                op: op @ (BinOp::Or | BinOp::And | BinOp::Xor | BinOp::Neq),
                left,
                right,
                ..
            } if matches!(self.cfg.val_types.get(left), Some(Ty::Bool)) => {
                let op = match op {
                    BinOp::Or => LawOp::Or,
                    BinOp::And => LawOp::And,
                    _ => LawOp::Xor,
                };
                let step = Step::Op { op, exact: true };
                match (self.form(*left)?, self.form(*right)?) {
                    (state, Form::Free) | (Form::Free, state) => state.then(step),
                    _ => None,
                }
            }
            // `!p` is `p != true`.
            InstKind::UnaryOp {
                op: crate::ir::UnaryOp::Not,
                operand,
                ..
            } => self.form(*operand)?.then(Step::Op {
                op: LawOp::Xor,
                exact: true,
            }),
            InstKind::BinOp {
                op, left, right, ..
            } => {
                let ty = self.cfg.val_types.get(&inst_info::defs(kind)[0])?;
                let exact = match ty {
                    ty if crate::analysis::affine::exact_under_wrapping(ty) => true,
                    Ty::Float => false,
                    _ => return None,
                };
                let (left, right) = (self.form(*left)?, self.form(*right)?);
                // An integer `+` or `*` of either kind is one law: on every
                // run that goes past a trapping one its result is the
                // integer one, and modulo `2^width` that is the wrapping
                // one's (RFC-0037 rule 3). What joins the partials is an
                // operation the lowerer writes, so it wraps.
                let law = match op {
                    BinOp::Add(_) => LawOp::Add,
                    BinOp::Mul(_) => LawOp::Mul,
                    // `a - b` is `a + (-b)`, exactly at every width and in
                    // floats alike.
                    BinOp::Sub(_) => {
                        return match right {
                            Form::Free => left.then(Step::Op {
                                op: LawOp::Add,
                                exact,
                            }),
                            Form::State | Form::Combined(_) => None,
                        };
                    }
                    _ => return None,
                };
                let step = Step::Op { op: law, exact };
                match (left, right) {
                    (state, Form::Free) | (Form::Free, state) => state.then(step),
                    _ => None,
                }
            }
            InstKind::StringConcat { parts, .. } => {
                let (&first, rest) = parts.split_first()?;
                let first = self.form(first)?;
                for part in rest {
                    if self.form(*part)? != Form::Free {
                        return None;
                    }
                }
                first.then(Step::Op {
                    op: LawOp::Concat,
                    exact: true,
                })
            }
            InstKind::Merge { orders, .. } => {
                let mut combined: Option<Form<'a>> = None;
                for order in orders {
                    match self.form(*order)? {
                        Form::Free => {}
                        form if combined.is_none() => combined = Some(form),
                        _ => return None,
                    }
                }
                combined?.then(Step::Order)
            }
            InstKind::StringClone { src, .. } => self.form(*src),
            // A shared view of a container lends the container's whole value.
            InstKind::AsSlice {
                container,
                mutability: Mutability::Shared,
                ..
            } => self.form(*container),
            InstKind::FunctionCall {
                callee: callee @ Callee::Extern { id, instance, .. },
                args,
                ..
            } => {
                // RFC-0082 rule 10: the result is the value the argument lends.
                if let Some(copied) = self.laws.copies_of(callee) {
                    return self.form(*args.get(copied)?);
                }
                let ResolvedLaws::Binary(
                    law @ ResolvedBinary {
                        associative: true,
                        commutative,
                        ..
                    },
                ) = self.laws.of_callee(callee)
                else {
                    return None;
                };
                let &[first, second] = args.as_slice() else {
                    return None;
                };
                let step = Step::Call {
                    callee: ExternInstance {
                        id: *id,
                        instance: *instance,
                    },
                    law,
                };
                match (self.form(first)?, self.form(second)?) {
                    (state, Form::Free) => state.then(step),
                    (Form::Free, state) if *commutative => state.then(step),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// What a storage the iteration defines and drops holds where it is
    /// read: the value of its one store.
    fn held(&mut self, slot: ValueId) -> Option<Form<'a>> {
        let mut stored: Option<ValueId> = None;
        for (_, kind) in self.insts() {
            if let InstKind::Assign { target, value, .. } = kind
                && inst_info::storage(target) == Some(slot)
            {
                if stored.is_some() {
                    return None;
                }
                stored = Some(*value);
            }
        }
        self.form(stored?)
    }

    /// A join's parameter: the values its edges send, chosen by branches
    /// that decide by values that read nothing of the state, or a compare
    /// and select.
    fn param_form(&mut self, block: BlockIdx, index: usize) -> Option<Form<'a>> {
        let Sent { sent, deciders } = self.sent_to(block, index)?;
        if self.decide_freely(&deciders) {
            let mut forms: Vec<(Form<'a>, Option<ValueId>, BlockIdx)> = Vec::new();
            for &(from, _, value) in &sent {
                forms.push((self.form(value)?, Some(value), from));
            }
            return self.joined(&forms);
        }
        let [branch] = deciders[..] else {
            return None;
        };
        if let Terminator::Switch { .. } = &self.cfg.blocks[branch.0].terminator {
            return self.option_lift(branch, &sent);
        }
        let [first, second] = sent[..] else {
            return None;
        };
        let side_of = |(from, side, _): (BlockIdx, Side, ValueId)| match from == branch {
            true => Some(side),
            false => self.side_reaching(branch, from),
        };
        let (first_side, second_side) = (side_of(first)?, side_of(second)?);
        if first_side == second_side {
            return None;
        }
        let (then_value, else_value) = match first_side {
            Side::Then => (first.2, second.2),
            Side::Else => (second.2, first.2),
        };
        if let Some(form) = self.short_circuit(branch, then_value, else_value) {
            return Some(form);
        }
        let (chosen, chosen_on) =
            self.chosen_over_state(branch, [(first_side, first.2), (second_side, second.2)])?;
        self.select(&Select {
            branch,
            chosen,
            chosen_on,
        })
    }

    /// Of the two values a join receives from the two sides of `branch`,
    /// the one that reads nothing of the state but through `branch`, where
    /// the other is the state.
    fn chosen_over_state(
        &mut self,
        branch: BlockIdx,
        sent: [(Side, ValueId); 2],
    ) -> Option<(ValueId, Side)> {
        let excused = self.values_reading_state(Some(branch));
        let mut forms: Vec<Form<'a>> = Vec::new();
        for (_, value) in sent {
            forms.push(match excused.contains(&value) {
                false => Form::Free,
                true => self.form(value)?,
            });
        }
        match forms[..] {
            [Form::Free, Form::State] => Some((sent[0].1, sent[0].0)),
            [Form::State, Form::Free] => Some((sent[1].1, sent[1].0)),
            _ => None,
        }
    }

    /// What the edges into `block` send its parameter `index`, each with the
    /// block it leaves and the edge's side, and the branches that choose
    /// among those edges other than those deciding whether `block` runs.
    fn sent_to(&self, block: BlockIdx, index: usize) -> Option<Sent> {
        let label = self.cfg.blocks[block.0].label;
        let mut sent: Vec<(BlockIdx, Side, ValueId)> = Vec::new();
        let mut from_blocks: Vec<BlockIdx> = Vec::new();
        for from in self.body() {
            let term = &self.cfg.blocks[from.0].terminator;
            for (at, edge) in edges(term).into_iter().enumerate() {
                if edge.target != label {
                    continue;
                }
                let position = index.checked_sub(edge.fills_from)?;
                let side = match at {
                    0 => Side::Then,
                    _ => Side::Else,
                };
                sent.push((from, side, edge.args[position]));
                if !from_blocks.contains(&from) {
                    from_blocks.push(from);
                }
            }
        }
        Some(Sent {
            sent,
            deciders: self.join_deciders(block, &from_blocks),
        })
    }

    /// RFC-0089 rule 4's left-biased extremum: this reading's state `p`, an
    /// integer, is chosen by a compare with a value `y` of the iteration by
    /// a strict order, the chosen arm sending `y` to `p` and to each of
    /// `carried` a value that reads no token, the other arm leaving all of
    /// them; and nothing else of the iteration reads `p` or any of
    /// `carried`. Returns `Max` or `Min`.
    fn extremum_carrying(mut self, carried: &[(ValueId, usize)]) -> Option<LawOp> {
        let State::Param { param, index } = self.state else {
            return None;
        };
        let loop_ = self.loops.iter().find(|loop_| loop_.header == self.header)?;
        let back = loop_.back_arg(self.cfg, index)?;
        let carried_backs: Vec<ValueId> = carried
            .iter()
            .map(|&(_, q_index)| loop_.back_arg(self.cfg, q_index))
            .collect::<Option<_>>()?;
        let (join, at) = self.param_of(back)?;
        let Sent { sent, deciders } = self.sent_to(join, at)?;
        let [branch] = deciders[..] else {
            return None;
        };
        let sides = |this: &Self, sent: &[(BlockIdx, Side, ValueId)]| -> Option<[(Side, ValueId); 2]> {
            let [first, second] = sent[..] else {
                return None;
            };
            let side_of = |(from, side, _): (BlockIdx, Side, ValueId)| match from == branch {
                true => Some(side),
                false => this.side_reaching(branch, from),
            };
            let pair = [(side_of(first)?, first.2), (side_of(second)?, second.2)];
            (pair[0].0 != pair[1].0).then_some(pair)
        };
        let [first, second] = sides(&self, &sent)?;
        let (chosen, chosen_on) = self.chosen_over_state(branch, [first, second])?;
        let (Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. }) =
            &self.cfg.blocks[branch.0].terminator
        else {
            return None;
        };
        let cond = *cond;
        let strict = matches!(self.defs.get(&cond), Some(Def::Inst(at))
            if matches!(self.inst(*at), InstKind::BinOp { op: BinOp::Lt | BinOp::Gt, .. }));
        let Form::Combined(Step::Op {
            op: op @ (LawOp::Min | LawOp::Max),
            ..
        }) = self.select(&Select {
            branch,
            chosen,
            chosen_on,
        })?
        else {
            return None;
        };
        if !strict {
            return None;
        }
        let reading_p = self.values_reading_state(Some(branch));
        let slots = slot_values(self.cfg);
        let only_the_update = reading_p
            .iter()
            .all(|value| !slots.contains(value) && [param, cond, back].contains(value));
        if !only_the_update {
            return None;
        }
        let readings: Vec<FxHashSet<ValueId>> = carried
            .iter()
            .map(|&(param, index)| {
                LawReading::of(
                    self.loans,
                    self.laws,
                    self.slots,
                    self.header,
                    &self.blocks,
                    State::Param { param, index },
                )
                .dependent
            })
            .collect();
        for ((&(q, _), &q_back), reading_q) in carried.iter().zip(&carried_backs).zip(&readings) {
            let (q_join, q_at) = self.param_of(q_back)?;
            if q_join != join {
                return None;
            }
            let Sent { sent, .. } = self.sent_to(join, q_at)?;
            let [(first_side, first), (_, second)] = sides(&self, &sent)?;
            let (sent_chosen, sent_left) = match first_side == chosen_on {
                true => (first, second),
                false => (second, first),
            };
            let reads_a_token = reading_p.contains(&sent_chosen)
                || readings.iter().any(|reading| reading.contains(&sent_chosen));
            let only_handed_on = reading_q
                .iter()
                .all(|value| !slots.contains(value) && [q, q_back].contains(value));
            if sent_left != q || reads_a_token || !only_handed_on {
                return None;
            }
        }
        Some(op)
    }

    /// RFC-0093 rules 5 and 7's `first`: this reading's state `g` is sent,
    /// by one side of a branch whose test holds there only where `g` is
    /// unset, `true` where `g` is a `Bool` (a flag), or a value reading no
    /// token that the interval domain proves is never the constant `c` where
    /// `g` is an integer compared with `c` (a sentinel), and `g` itself by
    /// the other side; the same join receives, for each of `guarded`, a value
    /// reading no token from that side and the token itself from the other.
    /// Nothing reads a guarded token but its hand-on, and what reads `g`
    /// decides only branches, runs no effect and nothing that can raise or
    /// not finish, since a chunk run from the unset guard computes it where
    /// the program skips it, and reaches the loop's next iteration only as
    /// `g` and the guarded tokens.
    fn guards_first(mut self, guarded: &[(ValueId, usize)]) -> Option<Unset> {
        self.guards_first_checked(guarded)
    }

    fn guards_first_checked(&mut self, guarded: &[(ValueId, usize)]) -> Option<Unset> {
        let State::Param { param: guard, index } = self.state else {
            return None;
        };
        let unset = match self.cfg.val_types.get(&guard) {
            Some(Ty::Bool) if !guarded.is_empty() => Unset::Flag,
            Some(Ty::Int(_)) => {
                let sentinel = self.sentinel_compared(guard)?;
                self.sentinel = Some(int_constant(self.cfg, sentinel)?);
                Unset::Sentinel(sentinel)
            }
            _ => return None,
        };
        let loop_ = self.loops.iter().find(|loop_| loop_.header == self.header)?;
        let back = loop_.back_arg(self.cfg, index)?;
        let backs: Vec<ValueId> = guarded
            .iter()
            .map(|&(_, at)| loop_.back_arg(self.cfg, at))
            .collect::<Option<_>>()?;
        let (join, at) = self.param_of(back)?;
        let Sent { sent, deciders } = self.sent_to(join, at)?;
        let [branch] = deciders[..] else {
            return None;
        };
        let (Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. }) =
            &self.cfg.blocks[branch.0].terminator
        else {
            return None;
        };
        let cond = *cond;
        let sides = |sent: &[(BlockIdx, Side, ValueId)]| -> Option<[(Side, ValueId); 2]> {
            let [first, second] = sent[..] else {
                return None;
            };
            let side_of = |(from, side, _): (BlockIdx, Side, ValueId)| match from == branch {
                true => Some(side),
                false => self.side_reaching(branch, from),
            };
            let pair = [(side_of(first)?, first.2), (side_of(second)?, second.2)];
            (pair[0].0 != pair[1].0).then_some(pair)
        };
        let [(first_side, first_sent), (second_side, second_sent)] = sides(&sent)?;
        let (arm, set) = match (first_sent == guard, second_sent == guard) {
            (false, true) => (first_side, first_sent),
            (true, false) => (second_side, second_sent),
            _ => return None,
        };
        let readings: Vec<FxHashSet<ValueId>> = guarded
            .iter()
            .map(|&(param, index)| {
                LawReading::of(
                    self.loans,
                    self.laws,
                    self.slots,
                    self.header,
                    &self.blocks,
                    State::Param { param, index },
                )
                .dependent
            })
            .collect();
        let sets_the_guard = match (unset, self.sentinel) {
            (Unset::Flag, _) => self.bool_constant(set) == Some(true),
            (Unset::Sentinel(_), Some(sentinel)) => {
                let reads_a_token = self.dependent.contains(&set)
                    || readings.iter().any(|reading| reading.contains(&set));
                let &(from, ..) = sent.iter().find(|(_, _, value)| *value == set)?;
                !reads_a_token && self.never_sent(from, set, sentinel)
            }
            (Unset::Sentinel(_), None) => false,
        };
        if !sets_the_guard || !self.unset_where(cond, arm == Side::Then, &mut Vec::new()) {
            return None;
        }
        let slots = slot_values(self.cfg);
        for ((&(q, _), &q_back), reading_q) in guarded.iter().zip(&backs).zip(&readings) {
            let (q_join, q_at) = self.param_of(q_back)?;
            if q_join != join {
                return None;
            }
            let Sent { sent, .. } = self.sent_to(join, q_at)?;
            let [(first_side, first), (_, second)] = sides(&sent)?;
            let (set, kept) = match first_side == arm {
                true => (first, second),
                false => (second, first),
            };
            let reads_a_token = self.dependent.contains(&set)
                || readings.iter().any(|reading| reading.contains(&set));
            let only_handed_on = reading_q
                .iter()
                .all(|value| !slots.contains(value) && [q, q_back].contains(value));
            if kept != q || reads_a_token || !only_handed_on {
                return None;
            }
        }
        // What reads the guard: it decides only branches, and reaches the
        // next iteration only as the guard and the guarded tokens.
        if self.dependent.iter().any(|value| slots.contains(value)) {
            return None;
        }
        let header_label = self.cfg.blocks[self.header.0].label;
        let mut handed_on = vec![back];
        handed_on.extend(&backs);
        for block in self.body().collect::<Vec<_>>() {
            let term = &self.cfg.blocks[block.0].terminator;
            let deciding = matches!(
                term,
                Terminator::JumpIf { .. } | Terminator::Diamond { .. } | Terminator::Jump { .. }
            );
            if !deciding
                && inst_info::terminator_uses(term)
                    .into_iter()
                    .any(|used| self.dependent.contains(&used))
            {
                return None;
            }
            for edge in edges(term) {
                if edge.target != header_label {
                    continue;
                }
                let passes_other = edge.args.iter().enumerate().any(|(at, arg)| {
                    let position = at + edge.fills_from;
                    let own = position == index || guarded.iter().any(|&(_, q)| q == position);
                    !own && self.dependent.contains(arg)
                });
                if passes_other {
                    return None;
                }
            }
        }
        let decided: Vec<BlockIdx> = self
            .body()
            .filter(|&block| self.decided_by_state(block, &self.dependent, None))
            .collect();
        if !self.runs_ahead_harmlessly(&decided) {
            return None;
        }
        let summary = FunctionSummary::unknown();
        let removal = raise::Removal::of(self.cfg, self.laws, &summary);
        let reads_harmlessly = self.insts().all(|(at, kind)| {
            let reads_guard = inst_info::uses(kind)
                .into_iter()
                .any(|used| self.dependent.contains(&used));
            let at = crate::analysis::interval::InstAt {
                block: at.block,
                at: at.at,
            };
            !reads_guard || (!has_effect(kind) && !removal.stays_unused(at, kind))
        });
        reads_harmlessly.then_some(unset)
    }

    /// The constant `c` every compare of `guard` for equality is with, where
    /// there is one and only one (RFC-0093 rule 7).
    fn sentinel_compared(&self, guard: ValueId) -> Option<ValueId> {
        let mut found: Option<(ValueId, i128)> = None;
        for (_, kind) in self.insts() {
            let Some(other) = self.compared_for_equality(kind, guard) else {
                continue;
            };
            let constant = int_constant(self.cfg, other)?;
            match found {
                Some((_, held)) if held != constant => return None,
                Some(_) => {}
                None => found = Some((other, constant)),
            }
        }
        found.map(|(value, _)| value)
    }

    /// The other operand of `kind`, where it is `==` or `!=` with `value` as
    /// one operand.
    fn compared_for_equality(&self, kind: &InstKind, value: ValueId) -> Option<ValueId> {
        let InstKind::BinOp {
            op: BinOp::Eq | BinOp::Neq,
            left,
            right,
            ..
        } = kind
        else {
            return None;
        };
        match (*left == value, *right == value) {
            (true, false) => Some(*right),
            (false, true) => Some(*left),
            _ => None,
        }
    }

    /// Whether the interval domain proves `value` is not `sentinel` where
    /// `from`, the block whose edge sends it, begins.
    fn never_sent(&self, from: BlockIdx, value: ValueId, sentinel: i128) -> bool {
        crate::analysis::interval::constant_bounds_on_entry(self.cfg, self.laws, from, &[value])
            .into_iter()
            .all(|bounds| bounds.excludes(sentinel))
    }

    /// Whether, wherever `value` is `holds`, this reading's `Bool` state is
    /// `false`: `value` is the state and `holds` is `false`, its negation,
    /// an `&&` one operand of which says so where it holds or an `||` one
    /// of which says so where it fails, a constant that is never `holds`, or
    /// a join parameter each edge into which sends such a value or leaves a
    /// block that runs only where the state is `false`.
    fn unset_where(&self, value: ValueId, holds: bool, seen: &mut Vec<(ValueId, bool)>) -> bool {
        let State::Param { param: guard, .. } = self.state else {
            return false;
        };
        if value == guard && self.sentinel.is_none() {
            return !holds;
        }
        if seen.contains(&(value, holds)) {
            return false;
        }
        seen.push((value, holds));
        if let Some(constant) = self.bool_constant(value) {
            return constant != holds;
        }
        if let (Some(sentinel), Some(Def::Inst(at))) = (self.sentinel, self.defs.get(&value)) {
            let kind = self.inst(*at);
            if let Some(other) = self.compared_for_equality(kind, guard)
                && int_constant(self.cfg, other) == Some(sentinel)
            {
                let equal_where_holds = matches!(kind, InstKind::BinOp { op: BinOp::Eq, .. });
                return equal_where_holds == holds;
            }
        }
        match self.defs.get(&value) {
            Some(Def::Inst(at)) => match self.inst(*at) {
                InstKind::UnaryOp {
                    op: crate::ir::UnaryOp::Not,
                    operand,
                    ..
                } => self.unset_where(*operand, !holds, seen),
                InstKind::BinOp {
                    op: BinOp::And,
                    left,
                    right,
                    ..
                } if holds => {
                    self.unset_where(*left, true, seen) || self.unset_where(*right, true, seen)
                }
                InstKind::BinOp {
                    op: BinOp::Or,
                    left,
                    right,
                    ..
                } if !holds => {
                    self.unset_where(*left, false, seen) || self.unset_where(*right, false, seen)
                }
                _ => false,
            },
            Some(Def::Param { block, index }) => {
                let Some(Sent { sent, .. }) = self.sent_to(*block, *index) else {
                    return false;
                };
                !sent.is_empty()
                    && sent.iter().all(|&(from, side, arg)| {
                        self.bool_constant(arg) == Some(!holds)
                            || self.unset_where(arg, holds, seen)
                            || self.runs_only_unset(from, Some(side), seen)
                    })
            }
            None => false,
        }
    }

    /// Whether `block` runs only where this reading's `Bool` state is
    /// `false`: a branch deciding it, or `block` itself leaving by `side`,
    /// tests a value that says so on the side that reaches it.
    fn runs_only_unset(
        &self,
        block: BlockIdx,
        side: Option<Side>,
        seen: &mut Vec<(ValueId, bool)>,
    ) -> bool {
        let tests = |branch: BlockIdx, side: Side, seen: &mut Vec<(ValueId, bool)>| {
            match &self.cfg.blocks[branch.0].terminator {
                Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. } => {
                    self.unset_where(*cond, side == Side::Then, seen)
                }
                _ => false,
            }
        };
        if let Some(side) = side
            && self.cfg.successors(block).len() > 1
            && tests(block, side, seen)
        {
            return true;
        }
        let deciders: Vec<BlockIdx> = self.deciders.get(&block).cloned().unwrap_or_default();
        deciders.into_iter().any(|branch| {
            self.side_reaching(branch, block)
                .is_some_and(|side| tests(branch, side, seen))
        })
    }

    /// The block and position of the block parameter `value` is.
    fn param_of(&self, value: ValueId) -> Option<(BlockIdx, usize)> {
        match self.defs.get(&value) {
            Some(Def::Param { block, index }) => Some((*block, *index)),
            _ => None,
        }
    }

    /// A branch on a `Bool` token `p` itself, whose arm where `p` holds
    /// sends `true` and whose other arm sends `y`, is `p ‖ y`; one whose arm
    /// where `p` fails sends `false` and whose other arm sends `y` is
    /// `p && y` (RFC-0089 rule 4). `y` reads nothing of `p` but through the
    /// branch, and its arm holds no effect, no operation that can raise and
    /// no loop: a chunk combined from the identity computes `y` where the
    /// program's own run skips it.
    fn short_circuit(
        &mut self,
        branch: BlockIdx,
        then_value: ValueId,
        else_value: ValueId,
    ) -> Option<Form<'a>> {
        let (Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. }) =
            &self.cfg.blocks[branch.0].terminator
        else {
            return None;
        };
        let cond = *cond;
        if !matches!(self.cfg.val_types.get(&cond), Some(Ty::Bool)) || self.form(cond)? != Form::State
        {
            return None;
        }
        let (op, y, arm) = match (self.bool_constant(then_value), self.bool_constant(else_value)) {
            (Some(true), _) => (LawOp::Or, else_value, Side::Else),
            (_, Some(false)) => (LawOp::And, then_value, Side::Then),
            _ => return None,
        };
        if self.values_reading_state(Some(branch)).contains(&y) {
            return None;
        }
        let arm_blocks: Vec<BlockIdx> = self
            .body()
            .filter(|block| {
                self.deciders
                    .get(block)
                    .is_some_and(|deciders| deciders.contains(&branch))
                    && self.side_reaching(branch, *block) == Some(arm)
            })
            .collect();
        if !self.runs_ahead_harmlessly(&arm_blocks) {
            return None;
        }
        for &block in &arm_blocks {
            for inst in &self.cfg.blocks[block.0].insts {
                self.chain.extend(inst_info::defs(&inst.kind));
            }
        }
        self.chain.extend([y, cond, then_value, else_value]);
        Some(Form::Combined(Step::Op { op, exact: true }))
    }

    /// Whether `blocks` hold no loop's header, no operation with an effect,
    /// and none that can raise or may not finish.
    fn runs_ahead_harmlessly(&self, blocks: &[BlockIdx]) -> bool {
        let summary = FunctionSummary::unknown();
        let removal = raise::Removal::of(self.cfg, self.laws, &summary);
        let holds_a_loop = self
            .loops
            .iter()
            .any(|loop_| blocks.contains(&loop_.header));
        !holds_a_loop
            && blocks.iter().all(|block| {
                self.cfg.blocks[block.0]
                    .insts
                    .iter()
                    .enumerate()
                    .all(|(at, inst)| {
                        let at = crate::analysis::interval::InstAt { block: *block, at };
                        !has_effect(&inst.kind) && !removal.stays_unused(at, &inst.kind)
                    })
            })
    }

    /// A switch on an `Option` token whose `None` arm sends `Some(y)` and
    /// whose `Some(b)` arm sends `Some(b ⊕ y)`, `y` reading nothing of the
    /// token but through the switch and `⊕` a law, is `⊕` lifted over
    /// `Option` (RFC-0089 rule 4).
    fn option_lift(&mut self, switch: BlockIdx, sent: &[(BlockIdx, Side, ValueId)]) -> Option<Form<'a>> {
        let Terminator::Switch { tag, arms, default } = &self.cfg.blocks[switch.0].terminator
        else {
            return None;
        };
        let tag = *tag;
        if self.form(tag)? != Form::State {
            return None;
        }
        let targets: Vec<(Option<&crate::ir::SwitchKey>, BlockIdx)> = arms
            .iter()
            .map(|(key, label, _)| (Some(key), self.cfg.label_to_block[label]))
            .chain(
                default
                    .iter()
                    .map(|(label, _)| (None, self.cfg.label_to_block[label])),
            )
            .collect();
        if targets.len() != 2 {
            return None;
        }
        // Each value the join receives, with the arm it is sent from.
        let mut from_arms: Vec<(Option<&crate::ir::SwitchKey>, BlockIdx, ValueId)> = Vec::new();
        for &(from, _, value) in sent {
            let [(key, arm)] = targets
                .iter()
                .copied()
                .filter(|(_, arm)| *arm == from || self.domtree.dominates(*arm, from))
                .collect::<Vec<_>>()[..]
            else {
                return None;
            };
            from_arms.push((key, arm, value));
        }
        let [(first_key, first_arm, first), (second_key, second_arm, second)] = from_arms[..]
        else {
            return None;
        };
        if first_arm == second_arm {
            return None;
        }
        let payload_of = |value: ValueId| match self.defs.get(&value) {
            Some(Def::Inst(at)) => match self.inst(*at) {
                InstKind::MakeVariant {
                    tag,
                    payload: Some(payload),
                    ..
                } => Some((*tag, *payload)),
                _ => None,
            },
            _ => None,
        };
        let ((first_tag, first_payload), (second_tag, second_payload)) =
            (payload_of(first)?, payload_of(second)?);
        let is_option = matches!(self.cfg.val_types.get(&first), Some(Ty::Option(_)));
        if first_tag != second_tag || !is_option {
            return None;
        }
        let some = crate::ir::SwitchKey::Tag(first_tag);
        let (y, combined, some_arm) = match (first_key == Some(&some), second_key == Some(&some)) {
            (false, true) => (first_payload, second_payload, second_arm),
            (true, false) => (second_payload, first_payload, first_arm),
            _ => return None,
        };
        let excused = self.values_reading_state(Some(switch));
        if excused.contains(&y) {
            return None;
        }
        let Some(&Def::Inst(at)) = self.defs.get(&combined) else {
            return None;
        };
        let (lifted, left, right) = self.binary_law(self.inst(at))?;
        let is_payload = |value: ValueId| self.payload_read(value, tag, some_arm);
        let (payload, other, commutes_needed) = match (is_payload(left), is_payload(right)) {
            (true, false) => (left, right, false),
            (false, true) => (right, left, true),
            _ => return None,
        };
        let commutes = match lifted {
            Lifted::Op { op, .. } => op.commutes(),
            Lifted::Call { law, .. } => law.commutative,
        };
        if (commutes_needed && !commutes)
            || !self.same_value(y, other)
            || excused.contains(&other)
        {
            return None;
        }
        self.chain
            .extend([tag, y, other, payload, combined, first, second]);
        Some(Form::Combined(Step::OptionLifted(lifted)))
    }

    /// Whether `value` reads the payload of the `Option` token the switch on
    /// `tag` decides by, in that switch's `Some` arm `some_arm`.
    fn payload_read(&self, value: ValueId, tag: ValueId, some_arm: BlockIdx) -> bool {
        let Some(&Def::Inst(at)) = self.defs.get(&value) else {
            return false;
        };
        let in_arm = at.block == some_arm || self.domtree.dominates(some_arm, at.block);
        let reads = match (self.inst(at), self.state) {
            (
                InstKind::Take {
                    target,
                    path,
                    taken_out: false,
                    ..
                },
                State::Slot(slot),
            ) => {
                matches!(path[..], [crate::ir::PathSeg::Payload])
                    && (inst_info::storage(target) == Some(slot)
                        || *target == RefTarget::Through(tag))
            }
            (InstKind::UnwrapVariant { src, .. }, State::Param { param, .. }) => *src == param,
            _ => false,
        };
        in_arm && reads
    }

    /// An associative binary law an instruction combines two values
    /// through, and those values in order.
    fn binary_law(&self, kind: &InstKind) -> Option<(Lifted<'a>, ValueId, ValueId)> {
        match kind {
            InstKind::BinOp {
                op, left, right, ..
            } => {
                let ty = self.cfg.val_types.get(&inst_info::defs(kind)[0])?;
                let exact = match ty {
                    ty if crate::analysis::affine::exact_under_wrapping(ty) => true,
                    Ty::Float => false,
                    _ => return None,
                };
                let op = match op {
                    BinOp::Add(_) => LawOp::Add,
                    BinOp::Mul(_) => LawOp::Mul,
                    BinOp::Min => LawOp::Min,
                    BinOp::Max => LawOp::Max,
                    _ => return None,
                };
                Some((Lifted::Op { op, exact }, *left, *right))
            }
            InstKind::FunctionCall {
                callee: callee @ Callee::Extern { id, instance, .. },
                args,
                ..
            } => {
                let ResolvedLaws::Binary(law @ ResolvedBinary {
                    associative: true, ..
                }) = self.laws.of_callee(callee)
                else {
                    return None;
                };
                let &[left, right] = args.as_slice() else {
                    return None;
                };
                Some((
                    Lifted::Call {
                        callee: ExternInstance {
                            id: *id,
                            instance: *instance,
                        },
                        law,
                    },
                    left,
                    right,
                ))
            }
            _ => None,
        }
    }

    /// `if y < s { s = y }` and its mirror images over an integer: the
    /// branch compares the state with a value of the iteration and chooses
    /// that value on one side and the state on the other, which is `min`
    /// or `max` of the two (`LawOp::Min`, `LawOp::Max`).
    fn select(&mut self, select: &Select) -> Option<Form<'a>> {
        let (Terminator::JumpIf { cond, .. } | Terminator::Diamond { cond, .. }) =
            &self.cfg.blocks[select.branch.0].terminator
        else {
            return None;
        };
        let Some(&Def::Inst(at)) = self.defs.get(cond) else {
            return None;
        };
        let InstKind::BinOp {
            op, left, right, ..
        } = self.inst(at)
        else {
            return None;
        };
        let (op, left, right) = (*op, *left, *right);
        if let Some(ordered) = self.ordered_compare(op, left, right) {
            return self.ordered_select(select, *cond, ordered);
        }
        if !matches!(self.cfg.val_types.get(&left), Some(Ty::Int(_))) {
            return None;
        }
        // The compare as `y ? s`, the value of the iteration on the left.
        let y_first = match (self.form(left)?, self.form(right)?) {
            (Form::Free, Form::State) => true,
            (Form::State, Form::Free) => false,
            _ => return None,
        };
        let compared = if y_first { left } else { right };
        if !self.same_value(compared, select.chosen) {
            return None;
        }
        // Whether `y` is chosen when `y < s` (a minimum) or when `y > s`.
        let y_less_when_true = match (op, y_first) {
            (BinOp::Lt | BinOp::Lte, true) | (BinOp::Gt | BinOp::Gte, false) => true,
            (BinOp::Gt | BinOp::Gte, true) | (BinOp::Lt | BinOp::Lte, false) => false,
            _ => return None,
        };
        let minimum = match select.chosen_on {
            Side::Then => y_less_when_true,
            Side::Else => !y_less_when_true,
        };
        self.chain.insert(*cond);
        self.chain.insert(select.chosen);
        Some(Form::Combined(Step::Op {
            op: if minimum { LawOp::Min } else { LawOp::Max },
            exact: true,
        }))
    }

    /// A compare of the sign of a call of an instance that states
    /// `law(total_order)` (RFC-0082 rule 10) with `0`, read as the same
    /// compare of the call's two arguments under that order: `f(a, b) > 0`
    /// is `a > b`, and `0 > f(a, b)` is `b > a`.
    fn ordered_compare(&self, op: BinOp, left: ValueId, right: ValueId) -> Option<OrderedCompare> {
        let call_of = |value: ValueId| {
            let Some(Def::Inst(at)) = self.defs.get(&value) else {
                return None;
            };
            let InstKind::FunctionCall {
                callee: callee @ Callee::Extern { id, instance, .. },
                args,
                ..
            } = self.inst(*at)
            else {
                return None;
            };
            let ResolvedLaws::TotalOrder = self.laws.of_callee(callee) else {
                return None;
            };
            let &[a, b] = args.as_slice() else {
                return None;
            };
            let order = ExternInstance {
                id: *id,
                instance: *instance,
            };
            Some((value, order, a, b))
        };
        let (sign, order, left, right) = match (call_of(left), call_of(right)) {
            (Some((sign, order, a, b)), None) if int_constant(self.cfg, right) == Some(0) => {
                (sign, order, a, b)
            }
            (None, Some((sign, order, a, b))) if int_constant(self.cfg, left) == Some(0) => {
                (sign, order, b, a)
            }
            _ => return None,
        };
        Some(OrderedCompare {
            op,
            sign,
            order,
            left,
            right,
        })
    }

    /// `if f(y, s) > 0 { s = y' }` and its mirror images, `f` stating a total
    /// order and `y'` a value equal to what `y` lends: the greater or the
    /// lesser of the two under that order (RFC-0082 rule 10).
    fn ordered_select(
        &mut self,
        select: &Select,
        cond: ValueId,
        compare: OrderedCompare,
    ) -> Option<Form<'a>> {
        let OrderedCompare {
            op,
            sign,
            order,
            left,
            right,
        } = compare;
        let y_first = match (self.form(left)?, self.form(right)?) {
            (Form::Free, Form::State) => true,
            (Form::State, Form::Free) => false,
            _ => return None,
        };
        let compared = if y_first { left } else { right };
        if self.copy_of(select.chosen) != Some(self.lent(compared)) {
            return None;
        }
        let y_less_when_true = match (op, y_first) {
            (BinOp::Lt | BinOp::Lte, true) | (BinOp::Gt | BinOp::Gte, false) => true,
            (BinOp::Gt | BinOp::Gte, true) | (BinOp::Lt | BinOp::Lte, false) => false,
            _ => return None,
        };
        let minimum = match select.chosen_on {
            Side::Then => y_less_when_true,
            Side::Else => !y_less_when_true,
        };
        self.chain.extend([cond, sign, select.chosen]);
        Some(Form::Combined(Step::Ordered {
            op: if minimum { LawOp::Min } else { LawOp::Max },
            order,
        }))
    }

    /// The reference whose lent value `reference` lends: itself, or the
    /// container a shared view is taken of.
    fn lent(&self, reference: ValueId) -> ValueId {
        match self.defs.get(&reference) {
            Some(Def::Inst(at)) => match self.inst(*at) {
                InstKind::AsSlice {
                    container,
                    mutability: Mutability::Shared,
                    ..
                } => self.lent(*container),
                _ => reference,
            },
            _ => reference,
        }
    }

    /// The reference whose lent value `value` equals: a read of the whole
    /// value through it, the language's clone of the `String` it lends, or a
    /// call of an instance stating `copies(x)` (RFC-0082 rule 10) with it as
    /// `x`.
    fn copy_of(&self, value: ValueId) -> Option<ValueId> {
        let Some(Def::Inst(at)) = self.defs.get(&value) else {
            return None;
        };
        match self.inst(*at) {
            InstKind::Take {
                target: RefTarget::Through(reference),
                path,
                taken_out: false,
                ..
            } if path.is_empty() => Some(self.lent(*reference)),
            InstKind::StringClone { src, .. } => Some(self.lent(*src)),
            InstKind::FunctionCall { callee, args, .. } => {
                let copied = self.laws.copies_of(callee)?;
                Some(self.lent(*args.get(copied)?))
            }
            _ => None,
        }
    }

    /// Whether two values of the iteration are one value: the same, or two
    /// copies of one word read through one shared reference, which nothing
    /// writes while the reference lives (RFC-0018, RFC-0029).
    fn same_value(&self, a: ValueId, b: ValueId) -> bool {
        if a == b {
            return true;
        }
        // A copy of an element read through a shared slice at one index is
        // the element's place, as a take through a reference is.
        let read = |value: ValueId| match self.defs.get(&value) {
            Some(Def::Inst(at)) => match self.inst(*at) {
                InstKind::Take {
                    target: RefTarget::Through(reference),
                    path,
                    taken_out: false,
                    ..
                } => Some((*reference, path.iter().cloned().map(Component::Named).collect())),
                InstKind::Index {
                    slice,
                    index,
                    mode: IndexMode::Copy,
                    ..
                } => Some((*slice, vec![Component::At(*index)])),
                _ => None,
            },
            _ => None,
        };
        let (Some((ra, pa)), Some((rb, pb))) = (read(a), read(b)) else {
            return false;
        };
        let shared = matches!(
            self.cfg.val_types.get(&ra),
            Some(Ty::Ref(crate::ty::Mutability::Shared, _))
        );
        ra == rb && pa == pb && shared
    }
}
