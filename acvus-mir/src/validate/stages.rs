//! RFC-0089 rules 1, 3, 4 and 5, asked of every `For` from its stage
//! membership and `analysis::loans`, so no loop runs apart on a fact that
//! stopped holding.
//!
//! Rule 3 is also asked of the induction variables: a pure stage reads no
//! header parameter `analysis::affine` derives as carried. IV
//! canonicalization computes every such variable that anything besides its
//! step reads from the counter (RFC-0066 rule 7), so one a pure stage reads
//! is one it missed, and the stage would wait on the iteration before. A
//! target's own rule does not refuse it: the reader is on no path to the
//! variable's next value, so it is outside the variable's update cycle.
//!
//! It is asked of the module the pipeline hands on, after every pass and
//! after the drops, which is the body the machine runs.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::affine::{AffineValues, Derivation};
use crate::analysis::carried::carries_order;
use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Invariants, LoopNest, natural_loops_innermost_first};
use crate::analysis::stages::{ShapeFault, StageMembership, loop_blocks_of};
use crate::analysis::targets::{TargetSlots, effect, slots_lent_mutably};
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::ir::{
    Accumulator, Callee, FoldAccumulator, ForSource, InstKind, Label, Law, MirBody, MirModule,
    Order, Stage, Stages, Target, Targets, ValueId,
};
use crate::ty::{Mutability, Ty};
use crate::validate::{ValidationError, ValidationErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PureEffect {
    WritesTarget(ValueId),
    LendsTargetMutably(ValueId),
    ChangesCarried(ValueId),
    OrderedEffect,
    CommitsContext,
    /// A carried induction variable, which IV canonicalization computes
    /// from the counter wherever anything besides its step reads it
    /// (RFC-0066 rule 7), so a pure stage has none to read.
    ReadsCarriedIv(ValueId),
}

impl PureEffect {
    pub fn shown(self) -> String {
        match self {
            Self::WritesTarget(slot) => format!("writes target {slot:?}"),
            Self::LendsTargetMutably(slot) => format!("lends target {slot:?} `&mut`"),
            Self::ChangesCarried(param) => {
                format!("holds the update of carried value {param:?}")
            }
            Self::OrderedEffect => "holds an order-carrying instruction".to_string(),
            Self::CommitsContext => "commits a context".to_string(),
            Self::ReadsCarriedIv(param) => {
                format!("reads carried induction variable {param:?}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderFault {
    /// A `Disjoint` join changes something other than the element of a
    /// `&mut` source.
    DisjointBeyondElement,
    /// An `AnyOrder` join has no law, or its law is inexact or does not
    /// commute.
    AnyOrderWithoutCommutingLaw,
    /// An `AnyOrder` join changes something its law does not join.
    AnyOrderBeyondLaw,
    /// The join's law is not what its target's update does.
    LawIsNotTheUpdate,
    /// The law's exactness is not its operation's at the target's type.
    Exactness,
}

impl OrderFault {
    pub fn shown(self) -> String {
        match self {
            Self::DisjointBeyondElement => {
                "is `Disjoint` and changes more than the element at its counter".to_string()
            }
            Self::AnyOrderWithoutCommutingLaw => {
                "is `AnyOrder` without an exact, commutative law".to_string()
            }
            Self::AnyOrderBeyondLaw => {
                "is `AnyOrder` and changes what its law does not join".to_string()
            }
            Self::LawIsNotTheUpdate => "states a law its target's update is not".to_string(),
            Self::Exactness => "states a law whose exactness is not its operation's".to_string(),
        }
    }
}

pub fn check(module: &MirModule) -> Vec<ValidationError> {
    let mut errors = check_body("main", &module.main);
    for (label, closure) in &module.closures {
        errors.extend(check_body(&format!("closure({label:?})"), closure));
    }
    errors
}

fn check_body(scope: &str, body: &MirBody) -> Vec<ValidationError> {
    let mut stated_at: FxHashMap<Label, usize> = FxHashMap::default();
    let mut block = crate::cfg::ENTRY_LABEL;
    for (at, inst) in body.insts.iter().enumerate() {
        match inst.kind {
            InstKind::BlockLabel { label, .. } => block = label,
            InstKind::For { .. } => {
                stated_at.insert(block, at);
            }
            _ => {}
        }
    }
    if stated_at.is_empty() {
        return Vec::new();
    }
    let cfg = promote(body.clone());
    let loans = Loans::build(&cfg);
    let domtree = DomTree::build(&cfg);
    let loops = natural_loops_innermost_first(&cfg, &domtree);
    let invariants = Invariants::of(&cfg);
    let nest = LoopNest::of(&cfg, &domtree, &invariants);
    let mut errors: Vec<ValidationError> = Vec::new();
    for (at, block) in cfg.blocks.iter().enumerate() {
        let Terminator::For { source, stages, .. } = &block.terminator else {
            continue;
        };
        let header = BlockIdx(at);
        let loop_blocks = loop_blocks_of(&loops, header);
        let inst_index = stated_at[&block.label];
        let carried_ivs = match nest.by_header(header) {
            Some(id) => {
                let affine = AffineValues::of(&cfg, nest.get(id), &invariants);
                block
                    .params
                    .iter()
                    .copied()
                    .filter(|param| {
                        matches!(
                            affine.get(*param).map(|found| &found.derivation),
                            Some(Derivation::Carried { .. })
                        )
                    })
                    .collect()
            }
            None => Vec::new(),
        };
        let chain = Chain {
            cfg: &cfg,
            loans: &loans,
            header,
            source: *source,
            stages,
            loop_blocks,
            carried_ivs,
        };
        errors.extend(chain.check().into_iter().map(|kind| ValidationError {
            scope: scope.to_string(),
            inst_index,
            span: body.insts[inst_index].span,
            kind,
        }));
    }
    errors
}

/// One edge of a terminator: its target, the arguments it passes, and how
/// many of the target's leading parameters the terminator fills itself.
struct Edge<'a> {
    target: Label,
    args: &'a [ValueId],
    first: usize,
}

fn edges<'a>(term: &'a Terminator) -> Vec<Edge<'a>> {
    let plain = |target: Label, args: &'a [ValueId]| Edge {
        target,
        args,
        first: 0,
    };
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
        } => {
            vec![
                Edge {
                    target: stages.body(),
                    args: &[],
                    first: source.supplied_params(),
                },
                Edge {
                    target: *exit,
                    args: exit_args,
                    first: exit_trip.supplied_params(),
                },
            ]
        }
        Terminator::Return { .. } | Terminator::Diverge | Terminator::Fallthrough => Vec::new(),
    }
}

struct ParamArg {
    param: ValueId,
    arg: ValueId,
}

/// One `For` as its terminator states it.
struct Chain<'a> {
    cfg: &'a CfgBody,
    loans: &'a Loans<'a>,
    header: BlockIdx,
    source: ForSource,
    stages: &'a Stages,
    loop_blocks: Vec<BlockIdx>,
    /// The header parameters `analysis::affine` derives as carried.
    carried_ivs: Vec<ValueId>,
}

/// The stage chain once its shape holds.
struct Form<'a> {
    chain: &'a Chain<'a>,
    membership: StageMembership,
    slots: TargetSlots,
    back_edges: Vec<Vec<ValueId>>,
    cycles: Vec<CarriedCycle>,
}

/// The instructions a header parameter's next value depends on and that
/// depend on its current one: its update, which a pure stage never holds.
struct CarriedCycle {
    param: ValueId,
    insts: FxHashSet<InstAt>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct InstAt {
    block: BlockIdx,
    at: usize,
}

impl<'a> Chain<'a> {
    fn check(&self) -> Vec<ValidationErrorKind> {
        let header_label = self.cfg.blocks[self.header.0].label;
        let shape = |fault| ValidationErrorKind::StageShape {
            header: header_label,
            fault,
        };
        if let Some(fault) = self.targets_fault() {
            return vec![shape(fault)];
        }
        let Some(&body) = self.cfg.label_to_block.get(&self.stages.body()) else {
            return vec![shape(ShapeFault::EntryNamesNoBlock)];
        };
        if self.cfg.blocks[body.0].params.len() != self.source.supplied_params() {
            return vec![shape(ShapeFault::BodyParams)];
        }
        let membership =
            match StageMembership::of(self.cfg, self.header, self.stages, &self.loop_blocks) {
                Ok(membership) => membership,
                Err(fault) => return vec![shape(fault)],
            };

        let back_edges: Vec<Vec<ValueId>> = self
            .loop_blocks
            .iter()
            .flat_map(|block| edges(&self.cfg.blocks[block.0].terminator))
            .filter(|edge| edge.target == header_label)
            .map(|edge| edge.args.to_vec())
            .collect();
        let form = Form {
            chain: self,
            slots: TargetSlots::of(self.loans, self.source, &self.loop_blocks),
            cycles: self.carried_cycles(&back_edges),
            back_edges,
            membership,
        };
        let mut refusals = form.leaves();
        refusals.extend(form.pure_effects());
        refusals.extend(form.orders());
        refusals
    }

    fn targets_fault(&self) -> Option<ShapeFault> {
        let header_params = &self.cfg.blocks[self.header.0].params;
        for (index, stage) in self.stages.iter().enumerate() {
            let Stage::Join { targets, order, .. } = stage else {
                continue;
            };
            match targets {
                Targets::Everything => {
                    if self.stages.len() != 1 || index != 0 || *order != Order::InOrder {
                        return Some(ShapeFault::EverythingInAChain);
                    }
                }
                Targets::Listed(listed) => {
                    let stray = listed.iter().any(|target| {
                        matches!(target, Target::Carried(param) if !header_params.contains(param))
                    });
                    if stray {
                        return Some(ShapeFault::CarriedTargetIsNoHeaderParam);
                    }
                }
            }
        }
        None
    }

    fn carried_cycles(&self, back_edges: &[Vec<ValueId>]) -> Vec<CarriedCycle> {
        let header_params = &self.cfg.blocks[self.header.0].params;
        header_params
            .iter()
            .enumerate()
            .map(|(index, &param)| {
                let sent: Vec<ValueId> = back_edges
                    .iter()
                    .filter_map(|args| args.get(index).copied())
                    .filter(|value| *value != param)
                    .collect();
                CarriedCycle {
                    param,
                    insts: self.cycle(param, &sent),
                }
            })
            .collect()
    }

    /// The instructions of the loop on a def-use path from `from` to one
    /// of `to`.
    fn cycle(&self, from: ValueId, to: &[ValueId]) -> FxHashSet<InstAt> {
        let insts: Vec<InstAt> = self
            .loop_blocks
            .iter()
            .filter(|block| **block != self.header)
            .flat_map(|&block| {
                (0..self.cfg.blocks[block.0].insts.len()).map(move |at| InstAt { block, at })
            })
            .collect();
        let kind = |inst: &InstAt| &self.cfg.blocks[inst.block.0].insts[inst.at].kind;
        let mut forward: FxHashSet<ValueId> = FxHashSet::from_iter([from]);
        let mut reached: FxHashSet<InstAt> = FxHashSet::default();
        let mut changed = true;
        while changed {
            changed = false;
            for inst in &insts {
                let reads = inst_info::uses(kind(inst))
                    .iter()
                    .any(|used| forward.contains(used));
                if reads && reached.insert(*inst) {
                    forward.extend(inst_info::defs(kind(inst)));
                    changed = true;
                }
            }
            for &block in &self.loop_blocks {
                for ParamArg { param, arg } in self.block_param_args(block) {
                    if forward.contains(&arg) && forward.insert(param) {
                        changed = true;
                    }
                }
            }
        }
        let mut backward: FxHashSet<ValueId> = to.iter().copied().collect();
        let mut feeding: FxHashSet<InstAt> = FxHashSet::default();
        let mut changed = true;
        while changed {
            changed = false;
            for inst in &insts {
                let defines = inst_info::defs(kind(inst))
                    .iter()
                    .any(|def| backward.contains(def));
                if defines && feeding.insert(*inst) {
                    backward.extend(inst_info::uses(kind(inst)));
                    changed = true;
                }
            }
            for &block in &self.loop_blocks {
                for ParamArg { param, arg } in self.block_param_args(block) {
                    if backward.contains(&param) && backward.insert(arg) {
                        changed = true;
                    }
                }
            }
        }
        reached.intersection(&feeding).copied().collect()
    }

    /// Each argument an edge from block `from` passes a block parameter,
    /// the header's aside, which a back edge fills for the next iteration.
    fn block_param_args(&self, from: BlockIdx) -> Vec<ParamArg> {
        let header_label = self.cfg.blocks[self.header.0].label;
        let mut pairs = Vec::new();
        for edge in edges(&self.cfg.blocks[from.0].terminator) {
            if edge.target == header_label {
                continue;
            }
            let Some(&target) = self.cfg.label_to_block.get(&edge.target) else {
                continue;
            };
            let params = self.cfg.blocks[target.0]
                .params
                .get(edge.first..)
                .unwrap_or(&[]);
            pairs.extend(
                params
                    .iter()
                    .zip(edge.args)
                    .map(|(&param, &arg)| ParamArg { param, arg }),
            );
        }
        pairs
    }
}

impl Form<'_> {
    fn insts_of(&self, stage: usize) -> impl Iterator<Item = (InstAt, &InstKind)> + '_ {
        let cfg = self.chain.cfg;
        self.membership.stages()[stage]
            .blocks
            .iter()
            .flat_map(move |&block| {
                cfg.blocks[block.0]
                    .insts
                    .iter()
                    .enumerate()
                    .map(move |(at, inst)| (InstAt { block, at }, &inst.kind))
            })
    }

    fn all_insts(&self) -> impl Iterator<Item = &InstKind> + '_ {
        (0..self.membership.stages().len())
            .flat_map(|stage| self.insts_of(stage).map(|(_, kind)| kind))
    }

    fn header_label(&self) -> Label {
        self.chain.cfg.blocks[self.chain.header.0].label
    }

    /// Rule 7: the loop is left only from the header or an `InOrder` join.
    fn leaves(&self) -> Vec<ValidationErrorKind> {
        let cfg = self.chain.cfg;
        let mut refusals = Vec::new();
        for (index, stage) in self.chain.stages.iter().enumerate() {
            if matches!(
                stage,
                Stage::Join {
                    order: Order::InOrder,
                    ..
                }
            ) {
                continue;
            }
            for &block in &self.membership.stages()[index].blocks {
                let returns = matches!(cfg.blocks[block.0].terminator, Terminator::Return { .. });
                let escapes = cfg.successors(block).into_iter().any(|succ| {
                    succ != self.chain.header && !self.chain.loop_blocks.contains(&succ)
                });
                if returns || escapes {
                    refusals.push(ValidationErrorKind::StageLeaves {
                        header: self.header_label(),
                        stage: index,
                        from: cfg.blocks[block.0].label,
                    });
                }
            }
        }
        refusals
    }

    fn in_cycle(&self, inst: InstAt) -> Option<ValueId> {
        self.cycles
            .iter()
            .find(|cycle| cycle.insts.contains(&inst))
            .map(|cycle| cycle.param)
    }

    /// Rule 3: a pure stage changes no target.
    fn pure_effects(&self) -> Vec<ValidationErrorKind> {
        let loans = self.chain.loans;
        let mut refusals = Vec::new();
        for (index, stage) in self.chain.stages.iter().enumerate() {
            let Stage::Pure { .. } = stage else {
                continue;
            };
            let mut found: Vec<PureEffect> = Vec::new();
            let mut note = |effect: PureEffect| {
                if !found.contains(&effect) {
                    found.push(effect);
                }
            };
            for (inst, kind) in self.insts_of(index) {
                if carries_order(kind) {
                    note(PureEffect::OrderedEffect);
                }
                if let InstKind::Commit { .. } = kind {
                    note(PureEffect::CommitsContext);
                }
                for slot in effect(loans, kind).writes {
                    if self.slots.target_of(slot).is_some() {
                        note(PureEffect::WritesTarget(slot));
                    }
                }
                for slot in slots_lent_mutably(loans, kind) {
                    if self.slots.target_of(slot).is_some() {
                        note(PureEffect::LendsTargetMutably(slot));
                    }
                }
                if let Some(param) = self.in_cycle(inst) {
                    note(PureEffect::ChangesCarried(param));
                }
                for used in inst_info::uses(kind) {
                    if self.chain.carried_ivs.contains(&used) {
                        note(PureEffect::ReadsCarriedIv(used));
                    }
                }
            }
            for &block in &self.membership.stages()[index].blocks {
                let terminator = &self.chain.cfg.blocks[block.0].terminator;
                for used in inst_info::terminator_uses(terminator) {
                    if self.chain.carried_ivs.contains(&used) {
                        note(PureEffect::ReadsCarriedIv(used));
                    }
                }
            }
            refusals.extend(
                found
                    .into_iter()
                    .map(|effect| ValidationErrorKind::PureStageEffect {
                        header: self.header_label(),
                        stage: index,
                        effect,
                    }),
            );
        }
        refusals
    }

    /// Rule 5 and rule 6: each join's order is one its operations admit,
    /// and its law is what its target's update does.
    fn orders(&self) -> Vec<ValidationErrorKind> {
        let mut refusals = Vec::new();
        for (index, stage) in self.chain.stages.iter().enumerate() {
            let Stage::Join {
                targets,
                order,
                law,
                ..
            } = stage
            else {
                continue;
            };
            let listed: &[Target] = match targets {
                Targets::Everything => &[],
                Targets::Listed(listed) => listed,
            };
            let mut faults: Vec<OrderFault> = Vec::new();
            if let Some(law) = law {
                faults.extend(self.law_fault(index, listed, law));
            }
            match order {
                Order::InOrder => {}
                Order::Disjoint => {
                    if !self.only_the_element(index) {
                        faults.push(OrderFault::DisjointBeyondElement);
                    }
                }
                Order::AnyOrder => match law {
                    Some(acc) if acc.exact && acc.commutative => {
                        if !self.only_the_law(index, listed, acc) {
                            faults.push(OrderFault::AnyOrderBeyondLaw);
                        }
                    }
                    _ => faults.push(OrderFault::AnyOrderWithoutCommutingLaw),
                },
            }
            faults.dedup();
            refusals.extend(
                faults
                    .into_iter()
                    .map(|fault| ValidationErrorKind::JoinOrder {
                        header: self.header_label(),
                        stage: index,
                        fault,
                    }),
            );
        }
        refusals
    }

    /// Whether join `stage` changes only the element of a `&mut` source:
    /// every write lands in the source's storage, and it holds no carried
    /// value's update, no ordered effect and no commit.
    fn only_the_element(&self, stage: usize) -> bool {
        if !matches!(self.chain.source, ForSource::SliceMut(_)) {
            return false;
        }
        let loans = self.chain.loans;
        self.insts_of(stage).all(|(inst, kind)| {
            let writes = effect(loans, kind)
                .writes
                .into_iter()
                .chain(slots_lent_mutably(loans, kind));
            let element_only = writes
                .into_iter()
                .all(|slot| matches!(self.slots.target_of(slot), None | Some(Target::Element)));
            element_only
                && !carries_order(kind)
                && !matches!(kind, InstKind::Commit { .. })
                && self.in_cycle(inst).is_none()
        })
    }

    /// Whether `AnyOrder` join `stage` changes only its law's target: every
    /// write is its fold's storage, every carried update is its
    /// accumulator's, and every ordered effect is one its `Order` law joins.
    fn only_the_law(&self, stage: usize, listed: &[Target], acc: &Accumulator) -> bool {
        let loans = self.chain.loans;
        let [target] = listed else {
            return false;
        };
        let fold = match &acc.law {
            Law::Fold(fold) => Some(fold.storage),
            Law::Op(_) | Law::Call(_) | Law::Order => None,
        };
        let carried = match target {
            Target::Carried(param) => Some(*param),
            Target::Storage(_) | Target::Context(_) | Target::Element => None,
        };
        let excused = match (&acc.law, carried) {
            (Law::Order, Some(param)) => self.unordered_calls(stage, param),
            _ => Vec::new(),
        };
        self.insts_of(stage).all(|(inst, kind)| {
            let writes_only_fold = effect(loans, kind)
                .writes
                .into_iter()
                .chain(slots_lent_mutably(loans, kind))
                .all(|slot| self.slots.target_of(slot).is_none() || Some(slot) == fold);
            let is_excused = inst_info::defs(kind)
                .iter()
                .any(|dst| excused.contains(dst));
            let own_update = self
                .in_cycle(inst)
                .is_none_or(|param| Some(param) == carried);
            writes_only_fold
                && own_update
                && (!carries_order(kind) || is_excused)
                && !matches!(kind, InstKind::Commit { .. })
        })
    }

    fn law_fault(&self, stage: usize, listed: &[Target], acc: &Accumulator) -> Option<OrderFault> {
        let [target] = listed else {
            return Some(OrderFault::LawIsNotTheUpdate);
        };
        match (&acc.law, target) {
            (Law::Fold(fold), Target::Storage(slot)) if fold.storage == *slot => {
                if !acc.exact {
                    return Some(OrderFault::Exactness);
                }
                (!self.folds_only(stage, fold)).then_some(OrderFault::LawIsNotTheUpdate)
            }
            (Law::Op(_) | Law::Call(_) | Law::Order, Target::Carried(param)) => {
                let Some(ty) = self.chain.cfg.val_types.get(param) else {
                    return Some(OrderFault::Exactness);
                };
                let exact = !matches!(acc.law, Law::Op(_)) || *ty != Ty::Float;
                if acc.exact != exact {
                    return Some(OrderFault::Exactness);
                }
                let header_params = &self.chain.cfg.blocks[self.chain.header.0].params;
                let Some(at) = header_params.iter().position(|held| held == param) else {
                    return Some(OrderFault::LawIsNotTheUpdate);
                };
                (!self.read_as_operand(acc, *param, at)).then_some(OrderFault::LawIsNotTheUpdate)
            }
            _ => Some(OrderFault::LawIsNotTheUpdate),
        }
    }

    /// The one back edge's arguments, where the loop has one back edge.
    fn latch_args(&self) -> Option<&[ValueId]> {
        match &self.back_edges[..] {
            [only] => Some(only),
            _ => None,
        }
    }

    fn readers(&self, value: ValueId) -> Vec<&InstKind> {
        self.all_insts()
            .filter(|kind| inst_info::uses(kind).contains(&value))
            .collect()
    }

    /// How many terminators other than the back edge's read `value`.
    fn read_by_terminators(&self, value: ValueId) -> usize {
        let header_label = self.header_label();
        let cfg = self.chain.cfg;
        self.membership
            .stages()
            .iter()
            .flat_map(|stage| &stage.blocks)
            .map(|block| &cfg.blocks[block.0].terminator)
            .filter(|term| !edges(term).iter().any(|edge| edge.target == header_label))
            .map(|term| {
                inst_info::terminator_uses(term)
                    .iter()
                    .filter(|used| **used == value)
                    .count()
            })
            .sum()
    }

    /// Whether `next` is what the back edge sends carried value `at`, and
    /// nothing else in the body reads it.
    fn sent_back_alone(&self, next: ValueId, at: usize) -> bool {
        let Some(latch_args) = self.latch_args() else {
            return false;
        };
        latch_args.get(at) == Some(&next)
            && latch_args.iter().filter(|arg| **arg == next).count() == 1
            && self.readers(next).is_empty()
            && self.read_by_terminators(next) == 0
    }

    fn read_as_operand(&self, acc: &Accumulator, param: ValueId, at: usize) -> bool {
        let Some(latch_args) = self.latch_args() else {
            return false;
        };
        if self.read_by_terminators(param) != 0 || latch_args.contains(&param) {
            return false;
        }
        let [reader] = self.readers(param)[..] else {
            return false;
        };
        match (&acc.law, reader) {
            (
                Law::Op(op),
                InstKind::BinOp {
                    dst,
                    op: read_op,
                    left,
                    right,
                },
            ) => {
                *read_op == op.bin_op()
                    && ((*left == param) != (*right == param))
                    && self.sent_back_alone(*dst, at)
            }
            (
                Law::Call(call),
                InstKind::FunctionCall {
                    dst, callee, args, ..
                },
            ) => {
                let named = matches!(callee, Callee::Extern { id, instance, .. }
                    if *id == call.callee && *instance == call.instance);
                let operand = match args[..] {
                    [a, b] if a == param && b != param => true,
                    [a, b] if b == param && a != param => acc.commutative,
                    _ => false,
                };
                named && operand && self.sent_back_alone(*dst, at)
            }
            (Law::Order, InstKind::Merge { dst, orders }) => {
                orders.iter().filter(|order| **order == param).count() == 1
                    && self.merged_on_to_latch(*dst, at)
            }
            _ => false,
        }
    }

    /// Whether `order`, a `Merge` of the accumulator, reaches the back edge
    /// through `Merge`s that nothing else reads.
    fn merged_on_to_latch(&self, order: ValueId, at: usize) -> bool {
        if self.sent_back_alone(order, at) {
            return true;
        }
        let sent_back = self.latch_args().is_some_and(|args| args.contains(&order));
        if self.read_by_terminators(order) != 0 || sent_back {
            return false;
        }
        match self.readers(order)[..] {
            [InstKind::Merge { dst, orders }] => {
                orders.iter().filter(|o| **o == order).count() == 1
                    && self.merged_on_to_latch(*dst, at)
            }
            _ => false,
        }
    }

    /// Whether storage `slot` is written in join `stage` only by calls of
    /// the fold's extern whose first argument alone lends it, and read only
    /// by a `Ref` that makes such a lender and nothing else reads: the
    /// storage lent only to the fold's calls, as `analysis::carried`
    /// recognizes it.
    fn folds_only(&self, stage: usize, fold: &FoldAccumulator) -> bool {
        let loans = self.chain.loans;
        let (slot, callee, instance) = (fold.storage, fold.callee, fold.instance);
        let lends = |value: ValueId, mutability: Option<Mutability>| {
            loans.holds(value).any(|loan| {
                loan.storage.slot() == Some(slot)
                    && mutability.is_none_or(|wanted| loan.mutability == wanted)
            })
        };
        let mut lenders: Vec<ValueId> = Vec::new();
        for (_, kind) in self.insts_of(stage) {
            if !loans.storage_effect(kind).writes.contains(&slot) {
                continue;
            }
            let InstKind::FunctionCall {
                callee: Callee::Extern {
                    id, instance: at, ..
                },
                args,
                ..
            } = kind
            else {
                return false;
            };
            let Some((&lender, rest)) = args.split_first() else {
                return false;
            };
            let folds = *id == callee
                && *at == instance
                && lends(lender, Some(Mutability::Mut))
                && !rest.iter().any(|arg| lends(*arg, None));
            if !folds {
                return false;
            }
            lenders.push(lender);
        }
        self.insts_of(stage).all(|(_, kind)| {
            let effect = loans.storage_effect(kind);
            if !effect.reads.contains(&slot) || effect.writes.contains(&slot) {
                return true;
            }
            match kind {
                InstKind::Ref { dst, .. } => {
                    lenders.contains(dst)
                        && self.readers(*dst).len() == 1
                        && self.read_by_terminators(*dst) == 0
                }
                _ => false,
            }
        })
    }

    /// Rule 6's calls in join `stage`, by the `dst` of each instruction: a
    /// call whose order input is the `Order` accumulator's entry value and
    /// whose order output is merged into that accumulator alone.
    fn unordered_calls(&self, stage: usize, param: ValueId) -> Vec<ValueId> {
        let Some(init) = self.entry_value(param) else {
            return Vec::new();
        };
        let merged = |order: ValueId| {
            matches!(self.readers(order)[..], [InstKind::Merge { .. }])
                && self.read_by_terminators(order) == 0
        };
        let mut unordered = Vec::new();
        for (_, kind) in self.insts_of(stage) {
            match kind {
                InstKind::FunctionCall {
                    dst,
                    order: Some(edge),
                    ..
                } if edge.before == init && merged(edge.after) => unordered.push(*dst),
                InstKind::Eval {
                    dst,
                    src,
                    order: Some(after),
                } if merged(*after) => {
                    let spawned = self.insts_of(stage).find_map(|(_, kind)| match kind {
                        InstKind::Spawn {
                            dst: handle,
                            order: Some(before),
                            ..
                        } if handle == src && *before == init => Some(*handle),
                        _ => None,
                    });
                    if let Some(handle) = spawned
                        && self.readers(handle).len() == 1
                    {
                        unordered.push(handle);
                        unordered.push(*dst);
                    }
                }
                _ => {}
            }
        }
        unordered
    }

    /// The one value every edge into the header from outside the loop
    /// sends header parameter `param`.
    fn entry_value(&self, param: ValueId) -> Option<ValueId> {
        let cfg = self.chain.cfg;
        let header = self.chain.header;
        let at = cfg.blocks[header.0]
            .params
            .iter()
            .position(|p| *p == param)?;
        let preds = cfg.predecessors();
        let mut sent: Option<ValueId> = None;
        for &pred in preds.get(&header)? {
            if self.chain.loop_blocks.contains(&pred) {
                continue;
            }
            let into_header: Vec<Edge<'_>> = edges(&cfg.blocks[pred.0].terminator)
                .into_iter()
                .filter(|edge| edge.target == self.header_label())
                .collect();
            let [edge] = &into_header[..] else {
                return None;
            };
            let arg = *edge.args.get(at)?;
            match sent {
                Some(seen) if seen != arg => return None,
                _ => sent = Some(arg),
            }
        }
        sent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::QualifiedRef;
    use crate::ir::{DebugInfo, ExitTrip, Inst, LawOp, OrderEdge, RefTarget};
    use crate::ty::{Task, TypeArg};
    use acvus_ast::Literal;
    use acvus_utils::{Interner, LocalFactory, LocalIdOps};

    fn v(n: usize) -> ValueId {
        ValueId::from_raw(n)
    }

    const AT: usize = 0;
    const HI: usize = 1;
    const S_INIT: usize = 2;
    const P_INIT: usize = 3;
    const H_S: usize = 4;
    const H_P: usize = 5;
    const I: usize = 6;
    const T: usize = 7;
    const COND: usize = 8;
    const S_NEXT: usize = 9;
    const P_NEXT: usize = 10;
    const SUM: usize = 11;
    const OUT: usize = 12;
    const SPARE: usize = 13;

    const HEADER: Label = Label(0);
    const PURE: Label = Label(1);
    const EXIT: Label = Label(2);
    const S_JOIN: Label = Label(3);
    const P_JOIN: Label = Label(4);
    const LEAVE: Label = Label(5);
    const STAY: Label = Label(6);

    fn exact_op(law: LawOp) -> Option<Accumulator> {
        Some(Accumulator {
            law: Law::Op(law),
            exact: true,
            commutative: true,
        })
    }

    fn join_over(entry: Label, param: usize, law: LawOp) -> Stage {
        Stage::Join {
            entry,
            targets: Targets::Listed(vec![Target::Carried(v(param))]),
            order: Order::AnyOrder,
            law: exact_op(law),
        }
    }

    /// `for i in 0..10 { let t = i + i; s = s + t; p = p * i }` as rule 1's
    /// chain: a pure stage for `t`, then an `AnyOrder` join over `s` and one
    /// over `p`. A slot `out` is written above the loop, so it is live at the
    /// header. `edit` changes the body, then `restate` its stages, before it
    /// is checked.
    fn body(
        edit: impl FnOnce(&mut Vec<InstKind>, &mut FxHashMap<ValueId, Ty>),
        restate: impl FnOnce(&mut Vec<Stage>),
    ) -> MirModule {
        let binop = |dst, left, right, op| InstKind::BinOp {
            dst: v(dst),
            op,
            left: v(left),
            right: v(right),
        };
        let mut stages = vec![
            Stage::Pure { entry: PURE },
            join_over(S_JOIN, H_S, LawOp::Add),
            join_over(P_JOIN, H_P, LawOp::Mul),
        ];
        restate(&mut stages);
        let first = stages.remove(0);
        let mut insts = vec![
            InstKind::Const {
                dst: v(AT),
                value: Literal::Int(0),
            },
            InstKind::Const {
                dst: v(HI),
                value: Literal::Int(10),
            },
            InstKind::Const {
                dst: v(S_INIT),
                value: Literal::Int(0),
            },
            InstKind::Const {
                dst: v(P_INIT),
                value: Literal::Int(1),
            },
            InstKind::Assign {
                target: RefTarget::Var(v(OUT)),
                path: vec![],
                value: v(AT),
                restores: false,
            },
            InstKind::Jump {
                label: HEADER,
                args: vec![v(S_INIT), v(P_INIT)],
            },
            InstKind::BlockLabel {
                label: HEADER,
                params: vec![v(H_S), v(H_P)],
            },
            InstKind::For {
                source: ForSource::Range {
                    at: v(AT),
                    hi: v(HI),
                },
                stages: Stages::new(first, stages),
                exit: EXIT,
                exit_trip: ExitTrip::Absent,
                exit_args: vec![],
            },
            InstKind::BlockLabel {
                label: PURE,
                params: vec![v(I)],
            },
            binop(T, I, I, crate::ir::BinOp::Add),
            InstKind::Jump {
                label: S_JOIN,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: S_JOIN,
                params: vec![],
            },
            binop(S_NEXT, H_S, T, crate::ir::BinOp::Add),
            InstKind::Jump {
                label: P_JOIN,
                args: vec![],
            },
            InstKind::BlockLabel {
                label: P_JOIN,
                params: vec![],
            },
            binop(P_NEXT, H_P, I, crate::ir::BinOp::Mul),
            InstKind::Jump {
                label: HEADER,
                args: vec![v(S_NEXT), v(P_NEXT)],
            },
            InstKind::BlockLabel {
                label: EXIT,
                params: vec![],
            },
            binop(SUM, H_S, H_P, crate::ir::BinOp::Add),
            InstKind::Return {
                value: v(SUM),
                order: None,
            },
        ];
        let mut val_types: FxHashMap<ValueId, Ty> = (0..=SPARE).map(|n| (v(n), Ty::I64)).collect();
        val_types.insert(v(COND), Ty::Bool);
        edit(&mut insts, &mut val_types);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=SPARE + 4 {
            factory.next();
        }
        MirModule {
            declared_params: 0,
            main: MirBody {
                demoted_diamonds: Default::default(),
                insts: insts
                    .into_iter()
                    .map(|kind| Inst {
                        span: acvus_ast::Span::ZERO,
                        kind,
                    })
                    .collect(),
                val_types,
                params: Vec::new(),
                captures: Vec::new(),
                order_param: None,
                task: Task::Sync,
                debug: DebugInfo::new(),
                val_factory: factory,
                label_count: 8,
            },
            closures: FxHashMap::default(),
            ret: Ty::I64,
            flows: crate::ty::Flows::Every,
        }
    }

    fn position(insts: &[InstKind], wanted: impl Fn(&InstKind) -> bool) -> usize {
        insts
            .iter()
            .position(wanted)
            .expect("the baseline body holds the instruction")
    }

    fn defines(dst: usize) -> impl Fn(&InstKind) -> bool {
        move |kind| inst_info::defs(kind).contains(&v(dst))
    }

    fn in_the_pure_stage(
        inserted: InstKind,
    ) -> impl FnOnce(&mut Vec<InstKind>, &mut FxHashMap<ValueId, Ty>) {
        move |insts, _| {
            let at = position(insts, defines(T));
            insts.insert(at + 1, inserted);
        }
    }

    fn leaving_from(from: Label) -> impl FnOnce(&mut Vec<InstKind>, &mut FxHashMap<ValueId, Ty>) {
        move |insts, _| {
            let label = position(
                insts,
                |kind| matches!(kind, InstKind::BlockLabel { label, .. } if *label == from),
            );
            insts.splice(
                label + 1..label + 1,
                [
                    InstKind::Const {
                        dst: v(COND),
                        value: Literal::Bool(true),
                    },
                    InstKind::JumpIf {
                        cond: v(COND),
                        then_label: LEAVE,
                        then_args: vec![],
                        else_label: STAY,
                        else_args: vec![],
                    },
                    InstKind::BlockLabel {
                        label: LEAVE,
                        params: vec![],
                    },
                    InstKind::Jump {
                        label: EXIT,
                        args: vec![],
                    },
                    InstKind::BlockLabel {
                        label: STAY,
                        params: vec![],
                    },
                ],
            );
        }
    }

    fn refusals(module: &MirModule) -> Vec<ValidationErrorKind> {
        check(module).into_iter().map(|error| error.kind).collect()
    }

    fn keep(_: &mut Vec<InstKind>, _: &mut FxHashMap<ValueId, Ty>) {}

    fn as_stated(_: &mut Vec<Stage>) {}

    #[test]
    fn the_chain_as_rule_1_states_it_is_admitted() {
        let module = body(keep, as_stated);
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
    }

    // -- Rule 1 -----------------------------------------------------------

    #[test]
    fn a_body_block_parameter_after_the_counter_is_refused() {
        let module = body(
            |insts, _| {
                let at = position(
                    insts,
                    |kind| matches!(kind, InstKind::BlockLabel { label, .. } if *label == PURE),
                );
                if let InstKind::BlockLabel { params, .. } = &mut insts[at] {
                    params.push(v(SPARE));
                }
            },
            as_stated,
        );
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::StageShape {
                fault: ShapeFault::BodyParams,
                ..
            }]
        ));
    }

    #[test]
    fn a_stage_that_skips_the_next_is_refused() {
        let module = body(
            |insts, _| {
                let at = position(
                    insts,
                    |kind| matches!(kind, InstKind::Jump { label, .. } if *label == S_JOIN),
                );
                insts.splice(
                    at..=at,
                    [
                        InstKind::Const {
                            dst: v(COND),
                            value: Literal::Bool(true),
                        },
                        InstKind::JumpIf {
                            cond: v(COND),
                            then_label: S_JOIN,
                            then_args: vec![],
                            else_label: P_JOIN,
                            else_args: vec![],
                        },
                    ],
                );
            },
            as_stated,
        );
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::StageShape {
                    fault: ShapeFault::StageDoesNotReachNext,
                    ..
                }]
            ),
            "{found:?}"
        );
    }

    #[test]
    fn a_join_over_everything_in_a_chain_is_refused() {
        let module = body(keep, |stages| {
            stages[2] = Stage::Join {
                entry: P_JOIN,
                targets: Targets::Everything,
                order: Order::InOrder,
                law: None,
            };
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::StageShape {
                fault: ShapeFault::EverythingInAChain,
                ..
            }]
        ));
    }

    // -- Rule 3 -----------------------------------------------------------

    #[test]
    fn a_pure_stage_holding_a_carried_update_is_refused() {
        let module = body(keep, |stages| stages[1] = Stage::Pure { entry: S_JOIN });
        let found = refusals(&module);
        assert!(
            found.iter().any(|refusal| matches!(refusal,
                ValidationErrorKind::PureStageEffect {
                    stage: 1,
                    effect: PureEffect::ChangesCarried(param),
                    ..
                } if *param == v(H_S))),
            "{found:?}"
        );
    }

    #[test]
    fn a_pure_stage_writing_a_target_is_refused() {
        let module = body(
            in_the_pure_stage(InstKind::Assign {
                target: RefTarget::Var(v(OUT)),
                path: vec![],
                value: v(T),
                restores: false,
            }),
            as_stated,
        );
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::PureStageEffect {
                    stage: 0,
                    effect: PureEffect::WritesTarget(slot),
                    ..
                }] if slot == v(OUT)
            ),
            "{found:?}"
        );
    }

    #[test]
    fn a_pure_stage_lending_a_target_mutably_is_refused() {
        let module = body(
            |insts, val_types| {
                val_types.insert(
                    v(SPARE),
                    Ty::Ref(Mutability::Mut, Box::new(TypeArg::uniform(Ty::I64))),
                );
                in_the_pure_stage(InstKind::Ref {
                    dst: v(SPARE),
                    target: RefTarget::Var(v(OUT)),
                    path: vec![],
                    mutability: Mutability::Mut,
                })(insts, val_types);
            },
            as_stated,
        );
        let found = refusals(&module);
        assert!(
            found.iter().any(|refusal| matches!(refusal,
                ValidationErrorKind::PureStageEffect {
                    stage: 0,
                    effect: PureEffect::LendsTargetMutably(slot),
                    ..
                } if *slot == v(OUT))),
            "{found:?}"
        );
    }

    #[test]
    fn a_pure_stage_holding_an_ordered_call_is_refused() {
        let interner = Interner::new();
        let callee = QualifiedRef::root(interner.intern("effect"));
        let module = body(
            |insts, val_types| {
                let (before, after) = (v(SPARE), v(SPARE + 1));
                val_types.insert(before, Ty::Order);
                val_types.insert(after, Ty::Order);
                val_types.insert(v(SPARE + 2), Ty::Unit);
                insts.insert(0, InstKind::Undef { dst: before });
                in_the_pure_stage(InstKind::FunctionCall {
                    dst: v(SPARE + 2),
                    callee: Callee::Extern {
                        id: callee,
                        instance: 0,
                        required: Vec::new(),
                    },
                    callee_ty: Ty::Unit,
                    args: vec![],
                    order: Some(OrderEdge { before, after }),
                })(insts, val_types);
            },
            as_stated,
        );
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::PureStageEffect {
                stage: 0,
                effect: PureEffect::OrderedEffect,
                ..
            }]
        ));
    }

    // -- Rule 5 -----------------------------------------------------------

    #[test]
    fn a_disjoint_join_over_a_carried_value_is_refused() {
        let module = body(keep, |stages| {
            stages[1] = Stage::Join {
                entry: S_JOIN,
                targets: Targets::Listed(vec![Target::Carried(v(H_S))]),
                order: Order::Disjoint,
                law: None,
            };
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::JoinOrder {
                stage: 1,
                fault: OrderFault::DisjointBeyondElement,
                ..
            }]
        ));
    }

    #[test]
    fn an_any_order_join_without_a_law_is_refused() {
        let module = body(keep, |stages| {
            stages[1] = Stage::Join {
                entry: S_JOIN,
                targets: Targets::Listed(vec![Target::Carried(v(H_S))]),
                order: Order::AnyOrder,
                law: None,
            };
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::JoinOrder {
                stage: 1,
                fault: OrderFault::AnyOrderWithoutCommutingLaw,
                ..
            }]
        ));
    }

    /// A float `+` joined in arrival order changes the rounding, so its law
    /// is inexact and its join `InOrder` (RFC-0089 rule 5).
    #[test]
    fn a_float_sum_marked_any_order_is_refused() {
        let module = body(
            |_, val_types| {
                for float in [S_INIT, H_S, T, S_NEXT, I] {
                    val_types.insert(v(float), Ty::Float);
                }
            },
            |stages| {
                stages[1] = Stage::Join {
                    entry: S_JOIN,
                    targets: Targets::Listed(vec![Target::Carried(v(H_S))]),
                    order: Order::AnyOrder,
                    law: Some(Accumulator {
                        law: Law::Op(LawOp::Add),
                        exact: false,
                        commutative: true,
                    }),
                };
            },
        );
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::JoinOrder {
                stage: 1,
                fault: OrderFault::AnyOrderWithoutCommutingLaw,
                ..
            }]
        ));
    }

    #[test]
    fn a_float_law_stated_exact_is_refused() {
        let module = body(
            |_, val_types| {
                for float in [S_INIT, H_S, T, S_NEXT, I] {
                    val_types.insert(v(float), Ty::Float);
                }
            },
            as_stated,
        );
        let found = refusals(&module);
        assert!(
            found.iter().any(|refusal| matches!(
                refusal,
                ValidationErrorKind::JoinOrder {
                    stage: 1,
                    fault: OrderFault::Exactness,
                    ..
                }
            )),
            "{found:?}"
        );
    }

    #[test]
    fn a_law_its_update_is_not_is_refused() {
        let module = body(
            |insts, _| {
                let at = position(insts, defines(S_NEXT));
                insts[at] = InstKind::BinOp {
                    dst: v(S_NEXT),
                    op: crate::ir::BinOp::Mul,
                    left: v(H_S),
                    right: v(T),
                };
            },
            as_stated,
        );
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::JoinOrder {
                stage: 1,
                fault: OrderFault::LawIsNotTheUpdate,
                ..
            }]
        ));
    }

    // -- Rule 7 -----------------------------------------------------------

    #[test]
    fn a_pure_stage_that_leaves_the_loop_is_refused() {
        let module = body(leaving_from(PURE), as_stated);
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::StageLeaves { stage: 0, from, .. }] if from == PURE
            ),
            "{found:?}"
        );
    }

    #[test]
    fn an_any_order_join_that_leaves_the_loop_is_refused() {
        let module = body(leaving_from(S_JOIN), as_stated);
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::StageLeaves { stage: 1, from, .. }] if from == S_JOIN
            ),
            "{found:?}"
        );
    }

    #[test]
    fn an_in_order_join_may_leave_the_loop() {
        let module = body(leaving_from(S_JOIN), |stages| {
            stages[1] = Stage::Join {
                entry: S_JOIN,
                targets: Targets::Listed(vec![Target::Carried(v(H_S))]),
                order: Order::InOrder,
                law: None,
            };
        });
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
    }
}
