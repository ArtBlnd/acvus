//! RFC-0089 rule 7: a `ForParts` is refused where its form no longer holds,
//! so no loop runs apart on a fact that stopped holding.
//!
//! It is asked of the module the pipeline hands on, after every pass and
//! after the drops, which is the body the machine runs.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::carried::carries_order;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::cfg::{BlockIdx, CfgBody, Terminator, promote};
use crate::graph::QualifiedRef;
use crate::ir::{
    Accumulator, Callee, ForSource, InstKind, Label, Law, MirBody, MirModule, Part, PartKind,
    ValueId,
};
use crate::ty::{Mutability, Ty};
use crate::validate::{ValidationError, ValidationErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeFault {
    NoParts,
    EntryNamesNoBlock,
    HeaderHoldsInstructions,
    BodyIsNotFirstEntry,
    CarriedAreNotHeaderParams,
    BodyParams,
    PartEntryEnteredElsewhere,
    PartsOverlap,
    PartDoesNotReachNext,
    LatchArgs,
    AccumulatorLayout,
    Exactness,
}

impl ShapeFault {
    pub fn shown(self) -> &'static str {
        match self {
            Self::NoParts => "it has no part",
            Self::EntryNamesNoBlock => "a part's entry names no block",
            Self::HeaderHoldsInstructions => "its header holds an instruction",
            Self::BodyIsNotFirstEntry => "its body block is not its first part's entry",
            Self::CarriedAreNotHeaderParams => {
                "its parts do not carry each of the header's parameters exactly once"
            }
            Self::BodyParams => "its body block's parameters are not the element and the counter",
            Self::PartEntryEnteredElsewhere => {
                "a part's entry is entered other than from the part before it"
            }
            Self::PartsOverlap => "a block lies in two parts",
            Self::PartDoesNotReachNext => {
                "a part does not end in one jump to the next part's entry"
            }
            Self::LatchArgs => "its last part does not jump to the header with every carried value",
            Self::AccumulatorLayout => {
                "a law part's accumulators are not one per carried value and then its folds"
            }
            Self::Exactness => "an accumulator's exactness is not its law's",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossing {
    Value(ValueId),
    Storage(ValueId),
    Context(QualifiedRef),
}

impl Crossing {
    pub fn shown(self) -> String {
        match self {
            Self::Value(value) => format!("reads {value:?}"),
            Self::Storage(slot) => format!("touches storage {slot:?}"),
            Self::Context(_) => "touches a context".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawEffect {
    OrderedCall,
    StorageWrite(ValueId),
    ContextWrite,
}

impl LawEffect {
    pub fn shown(self) -> String {
        match self {
            Self::OrderedCall => {
                "holds an order-carrying instruction that is not an `anyorder` call".to_string()
            }
            Self::StorageWrite(slot) => format!("writes storage {slot:?}"),
            Self::ContextWrite => "commits a context".to_string(),
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
            InstKind::ForParts { .. } => {
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
    let mut errors: Vec<ValidationError> = Vec::new();
    for (at, block) in cfg.blocks.iter().enumerate() {
        let Terminator::ForParts {
            source,
            body: body_label,
            parts,
            exit,
            ..
        } = &block.terminator
        else {
            continue;
        };
        let inst_index = stated_at[&block.label];
        let stated = Stated {
            header: BlockIdx(at),
            source: *source,
            body: *body_label,
            exit: *exit,
            parts,
        };
        errors.extend(
            Form::check(&cfg, &loans, stated)
                .into_iter()
                .map(|kind| ValidationError {
                    scope: scope.to_string(),
                    inst_index,
                    span: body.insts[inst_index].span,
                    kind,
                }),
        );
    }
    errors
}

/// What a `ForParts` terminator states.
struct Stated<'a> {
    header: BlockIdx,
    source: ForSource,
    body: Label,
    exit: Label,
    parts: &'a [Part],
}

/// One `ForParts` as its terminator states it, with the blocks of each
/// part.
struct Form<'a> {
    cfg: &'a CfgBody,
    loans: &'a Loans<'a>,
    header: BlockIdx,
    header_label: Label,
    source: ForSource,
    parts: &'a [Part],
    regions: Vec<Vec<BlockIdx>>,
    latch: BlockIdx,
    carried_owner: FxHashMap<ValueId, usize>,
    carried_part: Vec<usize>,
    supplied: Vec<ValueId>,
    latch_args: Vec<ValueId>,
}

impl<'a> Form<'a> {
    fn check(
        cfg: &'a CfgBody,
        loans: &'a Loans<'a>,
        stated: Stated<'a>,
    ) -> Vec<ValidationErrorKind> {
        let Stated {
            header,
            source,
            body,
            exit,
            parts,
        } = stated;
        let body = &body;
        let header_label = cfg.blocks[header.0].label;
        let shape = |fault| ValidationErrorKind::ForPartsShape {
            header: header_label,
            fault,
        };
        let Some(first) = parts.first() else {
            return vec![shape(ShapeFault::NoParts)];
        };
        if !cfg.blocks[header.0].insts.is_empty() {
            return vec![shape(ShapeFault::HeaderHoldsInstructions)];
        }
        if first.entry != *body {
            return vec![shape(ShapeFault::BodyIsNotFirstEntry)];
        }
        let header_params = &cfg.blocks[header.0].params;
        let mut carried_owner: FxHashMap<ValueId, usize> = FxHashMap::default();
        for (index, part) in parts.iter().enumerate() {
            for value in &part.carried {
                let named_once = carried_owner.insert(*value, index).is_none();
                if !named_once || !header_params.contains(value) {
                    return vec![shape(ShapeFault::CarriedAreNotHeaderParams)];
                }
            }
        }
        if carried_owner.len() != header_params.len() {
            return vec![shape(ShapeFault::CarriedAreNotHeaderParams)];
        }
        let Some(&body_block) = cfg.label_to_block.get(body) else {
            return vec![shape(ShapeFault::BodyParams)];
        };
        let body_params = &cfg.blocks[body_block.0].params;
        if body_params.len() != source.supplied_params() {
            return vec![shape(ShapeFault::BodyParams)];
        }
        let regions = match Self::regions(cfg, header, exit, parts) {
            Ok(regions) => regions,
            Err(fault) => return vec![shape(fault)],
        };
        let Some(&latch) = regions.last().and_then(|region| region.last()) else {
            return vec![shape(ShapeFault::PartDoesNotReachNext)];
        };
        let latch_args = match &cfg.blocks[latch.0].terminator {
            Terminator::Jump { label, args }
                if *label == header_label && args.len() == header_params.len() =>
            {
                args.clone()
            }
            _ => return vec![shape(ShapeFault::LatchArgs)],
        };
        let carried_part: Vec<usize> = header_params
            .iter()
            .map(|param| carried_owner[param])
            .collect();
        let form = Form {
            cfg,
            loans,
            header,
            header_label,
            source,
            parts,
            regions,
            latch,
            carried_owner,
            carried_part,
            supplied: body_params.clone(),
            latch_args,
        };
        if let Some(fault) = form.law_layout() {
            return vec![shape(fault)];
        }
        let mut refusals = form.leaves();
        refusals.extend(form.crossings());
        refusals.extend(form.accumulator_reads());
        refusals.extend(form.law_effects());
        refusals
    }

    /// Each part's blocks, entry first and its last block last: what its
    /// entry reaches before the next part's entry, and the header for the
    /// last part, short of the loop's exit.
    fn regions(
        cfg: &CfgBody,
        header: BlockIdx,
        exit: Label,
        parts: &[Part],
    ) -> Result<Vec<Vec<BlockIdx>>, ShapeFault> {
        let exit = cfg.label_to_block.get(&exit).copied();
        let entries: Vec<BlockIdx> = parts
            .iter()
            .map(|part| {
                cfg.label_to_block
                    .get(&part.entry)
                    .copied()
                    .ok_or(ShapeFault::EntryNamesNoBlock)
            })
            .collect::<Result<_, _>>()?;
        let preds = cfg.predecessors();
        let mut claimed: FxHashSet<BlockIdx> = FxHashSet::default();
        let mut regions = Vec::with_capacity(parts.len());
        for (index, &entry) in entries.iter().enumerate() {
            let next = entries.get(index + 1).copied().unwrap_or(header);
            let mut region = vec![entry];
            let mut work = vec![entry];
            let mut seen: FxHashSet<BlockIdx> = FxHashSet::from_iter([entry]);
            let mut reaching_next: Vec<BlockIdx> = Vec::new();
            while let Some(block) = work.pop() {
                for succ in cfg.successors(block) {
                    if succ == next {
                        reaching_next.push(block);
                        continue;
                    }
                    let outside = succ == header || Some(succ) == exit;
                    if outside || entries.contains(&succ) || !seen.insert(succ) {
                        continue;
                    }
                    region.push(succ);
                    work.push(succ);
                }
            }
            let [last] = reaching_next[..] else {
                return Err(ShapeFault::PartDoesNotReachNext);
            };
            if index + 1 < entries.len() {
                let jumps_on = matches!(&cfg.blocks[last.0].terminator,
                    Terminator::Jump { args, .. } if args.is_empty());
                if !jumps_on || preds.get(&next).map(|from| from.as_slice()) != Some(&[last][..]) {
                    return Err(ShapeFault::PartEntryEnteredElsewhere);
                }
                if !cfg.blocks[next.0].params.is_empty() {
                    return Err(ShapeFault::PartEntryEnteredElsewhere);
                }
            }
            region.retain(|block| *block != last);
            region.push(last);
            for block in &region {
                if !claimed.insert(*block) {
                    return Err(ShapeFault::PartsOverlap);
                }
            }
            regions.push(region);
        }
        if preds.get(&entries[0]).map(|from| from.as_slice()) != Some(&[header][..]) {
            return Err(ShapeFault::PartEntryEnteredElsewhere);
        }
        Ok(regions)
    }

    fn law_layout(&self) -> Option<ShapeFault> {
        for part in self.parts {
            let PartKind::Law(accs) = &part.kind else {
                continue;
            };
            let width = part.carried.len();
            if accs.len() < width {
                return Some(ShapeFault::AccumulatorLayout);
            }
            let (carried, folds) = accs.split_at(width);
            let carried_are_values = carried.iter().all(|acc| !matches!(acc.law, Law::Fold(_)));
            let rest_are_folds = folds.iter().all(|acc| matches!(acc.law, Law::Fold(_)));
            if !carried_are_values || !rest_are_folds {
                return Some(ShapeFault::AccumulatorLayout);
            }
            for (acc, value) in carried.iter().zip(&part.carried) {
                let float = matches!(self.cfg.val_types.get(value), Some(Ty::Float));
                let exact = !matches!(acc.law, Law::Op(_)) || !float;
                if acc.exact != exact {
                    return Some(ShapeFault::Exactness);
                }
            }
        }
        None
    }

    fn part_of_block(&self, block: BlockIdx) -> Option<usize> {
        self.regions
            .iter()
            .position(|region| region.contains(&block))
    }

    fn leaves(&self) -> Vec<ValidationErrorKind> {
        let mut refusals = Vec::new();
        for &block in self.regions.iter().flatten() {
            let returns = matches!(
                self.cfg.blocks[block.0].terminator,
                Terminator::Return { .. }
            );
            let escapes = self
                .cfg
                .successors(block)
                .into_iter()
                .any(|succ| succ != self.header && self.part_of_block(succ).is_none());
            if returns || escapes {
                refusals.push(ValidationErrorKind::ForPartsLeaves {
                    header: self.header_label,
                    from: self.cfg.blocks[block.0].label,
                });
            }
        }
        refusals
    }

    /// The part each value a part may read belongs to: a header parameter
    /// to the part that carries it, and a value the body defines to the part
    /// whose block defines it. The element and the counter are every part's.
    fn owners(&self) -> FxHashMap<ValueId, usize> {
        let mut owners: FxHashMap<ValueId, usize> = self.carried_owner.clone();
        for (index, region) in self.regions.iter().enumerate() {
            for &block in region {
                let block = &self.cfg.blocks[block.0];
                for param in &block.params {
                    owners.insert(*param, index);
                }
                for inst in &block.insts {
                    for def in inst_info::defs(&inst.kind) {
                        if !is_slot(self.cfg, def) {
                            owners.insert(def, index);
                        }
                    }
                }
            }
        }
        for supplied in &self.supplied {
            owners.remove(supplied);
        }
        owners
    }

    fn crossings(&self) -> Vec<ValidationErrorKind> {
        let owners = self.owners();
        let mut refusals = Vec::new();
        let mut refuse = |part: usize, crossing: Crossing| {
            let refusal = ValidationErrorKind::ForPartsCrossing {
                header: self.header_label,
                part,
                crossing,
            };
            if !refusals.iter().any(|seen: &ValidationErrorKind| {
                matches!(seen, ValidationErrorKind::ForPartsCrossing { part: p, crossing: c, .. }
                    if *p == part && *c == crossing)
            }) {
                refusals.push(refusal);
            }
        };
        let latch = self.latch;
        let mut touched: FxHashMap<Touched, Vec<Touch>> = FxHashMap::default();
        for (index, region) in self.regions.iter().enumerate() {
            for &block in region {
                let held = &self.cfg.blocks[block.0];
                let mut read: Vec<ValueId> = held
                    .insts
                    .iter()
                    .flat_map(|inst| inst_info::uses(&inst.kind))
                    .collect();
                if block != latch {
                    read.extend(inst_info::terminator_uses(&held.terminator));
                }
                for value in read {
                    if owners.get(&value).is_some_and(|owner| *owner != index) {
                        refuse(index, Crossing::Value(value));
                    }
                }
                for inst in &held.insts {
                    let mut touch = |on: Touched, writes: bool| {
                        touched.entry(on).or_default().push(Touch {
                            part: index,
                            writes,
                        });
                    };
                    let effect = self.loans.storage_effect(&inst.kind);
                    for slot in &effect.reads {
                        touch(Touched::Storage(*slot), false);
                    }
                    for slot in &effect.writes {
                        touch(Touched::Storage(*slot), true);
                    }
                    match &inst.kind {
                        InstKind::Commit { context, .. } => touch(Touched::Context(*context), true),
                        InstKind::Fetch { context, .. } => touch(Touched::Context(*context), false),
                        InstKind::Drop { src } if is_slot(self.cfg, *src) => {
                            touch(Touched::Storage(*src), true)
                        }
                        _ => {}
                    }
                }
            }
        }
        for (at, arg) in self.latch_args.iter().enumerate() {
            let owner = self.carried_part[at];
            if owners
                .get(arg)
                .is_some_and(|defined_by| *defined_by != owner)
            {
                refuse(owner, Crossing::Value(*arg));
            }
        }
        for (on, touches) in touched {
            let writers: FxHashSet<usize> = touches
                .iter()
                .filter(|touch| touch.writes)
                .map(|touch| touch.part)
                .collect();
            for touch in &touches {
                if writers.iter().any(|writer| *writer != touch.part) {
                    refuse(touch.part, on.crossing());
                }
            }
        }
        refusals
    }

    fn insts_of(&self, part: usize) -> impl Iterator<Item = &'a InstKind> + '_ {
        let cfg = self.cfg;
        self.regions[part]
            .iter()
            .flat_map(move |block| &cfg.blocks[block.0].insts)
            .map(|inst| &inst.kind)
    }

    fn all_insts(&self) -> impl Iterator<Item = &'a InstKind> + '_ {
        (0..self.regions.len()).flat_map(|part| self.insts_of(part))
    }

    fn readers(&self, value: ValueId) -> Vec<&'a InstKind> {
        self.all_insts()
            .filter(|kind| inst_info::uses(kind).contains(&value))
            .collect()
    }

    fn read_by_terminators(&self, value: ValueId) -> usize {
        let latch = self.latch;
        self.regions
            .iter()
            .flatten()
            .filter(|block| **block != latch)
            .map(|block| {
                inst_info::terminator_uses(&self.cfg.blocks[block.0].terminator)
                    .iter()
                    .filter(|used| **used == value)
                    .count()
            })
            .sum()
    }

    /// Whether `next` is what the latch sends carried value `at`, and
    /// nothing else in the body reads it.
    fn sent_back_alone(&self, next: ValueId, at: usize) -> bool {
        self.latch_args.get(at) == Some(&next)
            && self.latch_args.iter().filter(|arg| **arg == next).count() == 1
            && self.readers(next).is_empty()
            && self.read_by_terminators(next) == 0
    }

    fn accumulator_reads(&self) -> Vec<ValidationErrorKind> {
        let header_params = &self.cfg.blocks[self.header.0].params;
        let mut refusals = Vec::new();
        for (index, part) in self.parts.iter().enumerate() {
            let PartKind::Law(accs) = &part.kind else {
                continue;
            };
            for (acc_at, acc) in accs.iter().enumerate() {
                let lawful = match &acc.law {
                    Law::Fold(fold) => {
                        self.folds_only(index, fold.storage, fold.callee, fold.instance)
                    }
                    Law::Op(_) | Law::Call(_) | Law::Order => {
                        let param = part.carried.get(acc_at).copied();
                        let at = param
                            .and_then(|param| header_params.iter().position(|held| *held == param));
                        param
                            .zip(at)
                            .is_some_and(|(param, at)| self.read_as_operand(acc, param, at))
                    }
                };
                if !lawful {
                    refusals.push(ValidationErrorKind::ForPartsAccumulatorRead {
                        header: self.header_label,
                        part: index,
                        acc: acc_at,
                    });
                }
            }
        }
        refusals
    }

    fn read_as_operand(&self, acc: &Accumulator, param: ValueId, at: usize) -> bool {
        if self.read_by_terminators(param) != 0 || self.latch_args.contains(&param) {
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

    /// Whether `order`, a `Merge` of the accumulator, reaches the latch
    /// through `Merge`s that nothing else reads.
    fn merged_on_to_latch(&self, order: ValueId, at: usize) -> bool {
        if self.sent_back_alone(order, at) {
            return true;
        }
        if self.read_by_terminators(order) != 0 || self.latch_args.contains(&order) {
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

    /// Whether storage `slot` is written in part `part` only by calls of
    /// the fold's extern whose first argument alone lends it, and read only
    /// by a `Ref` that makes such a lender and nothing else reads: the
    /// storage lent only to the fold's calls, as `analysis::carried`
    /// recognizes it.
    fn folds_only(
        &self,
        part: usize,
        slot: ValueId,
        callee: QualifiedRef,
        instance: usize,
    ) -> bool {
        let lends = |value: ValueId, mutability: Option<Mutability>| {
            self.loans.holds(value).any(|loan| {
                loan.storage.slot() == Some(slot)
                    && mutability.is_none_or(|wanted| loan.mutability == wanted)
            })
        };
        let mut lenders: Vec<ValueId> = Vec::new();
        for kind in self.insts_of(part) {
            if !self.loans.storage_effect(kind).writes.contains(&slot) {
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
        self.insts_of(part).all(|kind| {
            let effect = self.loans.storage_effect(kind);
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

    fn law_effects(&self) -> Vec<ValidationErrorKind> {
        let lent = self.lent_mutably_by_source();
        let mut refusals = Vec::new();
        for (index, part) in self.parts.iter().enumerate() {
            let PartKind::Law(accs) = &part.kind else {
                continue;
            };
            let folds: Vec<ValueId> = accs
                .iter()
                .filter_map(|acc| match &acc.law {
                    Law::Fold(fold) => Some(fold.storage),
                    Law::Op(_) | Law::Call(_) | Law::Order => None,
                })
                .collect();
            let excused = self.unordered_calls(index, part, accs);
            let mut refuse = |effect: LawEffect| {
                refusals.push(ValidationErrorKind::ForPartsLawEffect {
                    header: self.header_label,
                    part: index,
                    effect,
                })
            };
            for kind in self.insts_of(index) {
                let is_excused = inst_info::defs(kind)
                    .iter()
                    .any(|dst| excused.contains(dst));
                if carries_order(kind) && !is_excused {
                    refuse(LawEffect::OrderedCall);
                }
                if let InstKind::Commit { .. } = kind {
                    refuse(LawEffect::ContextWrite);
                }
                for slot in self.loans.storage_effect(kind).writes {
                    if !lent.contains(&slot) && !folds.contains(&slot) {
                        refuse(LawEffect::StorageWrite(slot));
                    }
                }
            }
        }
        refusals
    }

    fn lent_mutably_by_source(&self) -> Vec<ValueId> {
        let ForSource::SliceMut(slice) = self.source else {
            return Vec::new();
        };
        self.loans
            .names(slice)
            .iter()
            .filter(|loan| loan.mutability == Mutability::Mut)
            .filter_map(|loan| loan.storage.slot())
            .collect()
    }

    /// Rule 6's calls in part `index`, by the `dst` of each instruction: a
    /// call whose order input is an `Order` accumulator's entry value and
    /// whose order output is merged into that accumulator alone.
    fn unordered_calls(&self, index: usize, part: &Part, accs: &[Accumulator]) -> Vec<ValueId> {
        let inits: Vec<ValueId> = part
            .carried
            .iter()
            .zip(accs)
            .filter(|(_, acc)| acc.law == Law::Order)
            .filter_map(|(param, _)| self.entry_value(*param))
            .collect();
        let merged = |order: ValueId| {
            matches!(self.readers(order)[..], [InstKind::Merge { .. }])
                && self.read_by_terminators(order) == 0
        };
        let mut unordered = Vec::new();
        for kind in self.insts_of(index) {
            match kind {
                InstKind::FunctionCall {
                    dst,
                    order: Some(edge),
                    ..
                } if inits.contains(&edge.before) && merged(edge.after) => unordered.push(*dst),
                InstKind::Eval {
                    dst,
                    src,
                    order: Some(after),
                } if merged(*after) => {
                    let spawned = self.insts_of(index).find_map(|kind| match kind {
                        InstKind::Spawn {
                            dst: handle,
                            order: Some(before),
                            ..
                        } if handle == src && inits.contains(before) => Some(*handle),
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
        let at = self.cfg.blocks[self.header.0]
            .params
            .iter()
            .position(|p| *p == param)?;
        let preds = self.cfg.predecessors();
        let mut sent: Option<ValueId> = None;
        for &pred in preds.get(&self.header)? {
            if self.part_of_block(pred).is_some() {
                continue;
            }
            let arg = entering_arg(&self.cfg.blocks[pred.0].terminator, self.header_label, at)?;
            match sent {
                Some(seen) if seen != arg => return None,
                _ => sent = Some(arg),
            }
        }
        sent
    }
}

/// The argument an edge of `term` to `label` passes parameter `at`, where
/// exactly one edge of it goes there and passes the whole list.
fn entering_arg(term: &Terminator, label: Label, at: usize) -> Option<ValueId> {
    let args = crate::analysis::loops::edge_args(term, label)?;
    args.get(at).copied()
}

struct Touch {
    part: usize,
    writes: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Touched {
    Storage(ValueId),
    Context(QualifiedRef),
}

impl Touched {
    fn crossing(self) -> Crossing {
        match self {
            Self::Storage(slot) => Crossing::Storage(slot),
            Self::Context(context) => Crossing::Context(context),
        }
    }
}

fn is_slot(cfg: &CfgBody, value: ValueId) -> bool {
    cfg.blocks.iter().flat_map(|block| &block.insts).any(|inst| {
        matches!(&inst.kind,
            InstKind::Ref { target, .. } | InstKind::Take { target, .. } | InstKind::Assign { target, .. }
            if inst_info::storage(target) == Some(value))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Accumulator, DebugInfo, ExitTrip, Inst, LawOp, OrderEdge, RefTarget};
    use crate::ty::Task;
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
    const S_NEXT: usize = 9;
    const P_NEXT: usize = 10;
    const SUM: usize = 11;
    const SPARE: usize = 12;

    fn op(law: LawOp) -> Accumulator {
        Accumulator {
            law: Law::Op(law),
            exact: true,
            commutative: true,
        }
    }

    /// `for i in 0..10 { s = s + i; p = p * i }` as rule 1's chain: part
    /// L1 carries `s`, part L3 carries `p`, and `edit` changes it before it
    /// is checked.
    fn body(edit: impl FnOnce(&mut Vec<InstKind>, &mut FxHashMap<ValueId, Ty>)) -> MirModule {
        let add = |dst, left, right, op| InstKind::BinOp {
            dst: v(dst),
            op,
            left: v(left),
            right: v(right),
        };
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
            InstKind::Jump {
                label: Label(0),
                args: vec![v(S_INIT), v(P_INIT)],
            },
            InstKind::BlockLabel {
                label: Label(0),
                params: vec![v(H_S), v(H_P)],
            },
            InstKind::ForParts {
                source: ForSource::Range {
                    at: v(AT),
                    hi: v(HI),
                },
                body: Label(1),
                parts: vec![
                    Part {
                        entry: Label(1),
                        carried: vec![v(H_S)],
                        kind: PartKind::Law(vec![op(LawOp::Add)]),
                    },
                    Part {
                        entry: Label(3),
                        carried: vec![v(H_P)],
                        kind: PartKind::Law(vec![op(LawOp::Mul)]),
                    },
                ],
                exit: Label(2),
                exit_trip: ExitTrip::Absent,
                exit_args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(1),
                params: vec![v(I)],
            },
            add(S_NEXT, H_S, I, crate::ir::BinOp::Add),
            InstKind::Jump {
                label: Label(3),
                args: vec![],
            },
            InstKind::BlockLabel {
                label: Label(3),
                params: vec![],
            },
            add(P_NEXT, H_P, I, crate::ir::BinOp::Mul),
            InstKind::Jump {
                label: Label(0),
                args: vec![v(S_NEXT), v(P_NEXT)],
            },
            InstKind::BlockLabel {
                label: Label(2),
                params: vec![],
            },
            add(SUM, H_S, H_P, crate::ir::BinOp::Add),
            InstKind::Return {
                value: v(SUM),
                order: None,
            },
        ];
        let mut val_types: FxHashMap<ValueId, Ty> = (0..=SPARE).map(|n| (v(n), Ty::I64)).collect();
        edit(&mut insts, &mut val_types);
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..=SPARE + 4 {
            factory.next();
        }
        MirModule {
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

    fn refusals(module: &MirModule) -> Vec<ValidationErrorKind> {
        check(module).into_iter().map(|error| error.kind).collect()
    }

    #[test]
    fn the_chain_as_rule_1_states_it_is_admitted() {
        let module = body(|_, _| {});
        assert!(refusals(&module).is_empty(), "{:?}", refusals(&module));
    }

    #[test]
    fn a_header_parameter_two_parts_carry_is_refused() {
        let module = body(|insts, _| {
            let at = position(insts, |kind| matches!(kind, InstKind::ForParts { .. }));
            if let InstKind::ForParts { parts, .. } = &mut insts[at] {
                parts[1].carried = vec![v(H_S)];
            }
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::ForPartsShape {
                fault: ShapeFault::CarriedAreNotHeaderParams,
                ..
            }]
        ));
    }

    #[test]
    fn a_body_block_parameter_after_the_counter_is_refused() {
        let module = body(|insts, _| {
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::BlockLabel { label, .. } if *label == Label(1)),
            );
            if let InstKind::BlockLabel { params, .. } = &mut insts[at] {
                params.push(v(SPARE));
            }
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::ForPartsShape {
                fault: ShapeFault::BodyParams,
                ..
            }]
        ));
    }

    #[test]
    fn a_part_reading_another_parts_value_is_refused() {
        let module = body(|insts, _| {
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::BinOp { dst, .. } if *dst == v(P_NEXT)),
            );
            insts[at] = InstKind::BinOp {
                dst: v(P_NEXT),
                op: crate::ir::BinOp::Mul,
                left: v(H_P),
                right: v(S_NEXT),
            };
        });
        let found = refusals(&module);
        assert!(
            found.iter().any(|refusal| matches!(refusal,
                ValidationErrorKind::ForPartsCrossing {
                    part: 1,
                    crossing: Crossing::Value(value),
                    ..
                } if *value == v(S_NEXT))),
            "{found:?}"
        );
    }

    #[test]
    fn an_accumulator_read_other_than_by_its_law_is_refused() {
        let module = body(|insts, _| {
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::BinOp { dst, .. } if *dst == v(S_NEXT)),
            );
            insts[at] = InstKind::BinOp {
                dst: v(S_NEXT),
                op: crate::ir::BinOp::Mul,
                left: v(H_S),
                right: v(I),
            };
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::ForPartsAccumulatorRead {
                part: 0,
                acc: 0,
                ..
            }]
        ));
    }

    #[test]
    fn a_law_part_holding_an_ordered_call_is_refused() {
        let interner = Interner::new();
        let callee = QualifiedRef::root(interner.intern("effect"));
        let module = body(|insts, val_types| {
            let (before, after) = (v(SPARE), v(SPARE + 1));
            val_types.insert(before, Ty::Order);
            val_types.insert(after, Ty::Order);
            val_types.insert(v(SPARE + 2), Ty::Unit);
            insts.insert(0, InstKind::Undef { dst: before });
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::BinOp { dst, .. } if *dst == v(S_NEXT)),
            );
            insts.insert(
                at,
                InstKind::FunctionCall {
                    dst: v(SPARE + 2),
                    callee: Callee::Extern {
                        id: callee,
                        instance: 0,
                        required: Vec::new(),
                    },
                    callee_ty: Ty::Unit,
                    args: vec![],
                    order: Some(OrderEdge { before, after }),
                },
            );
        });
        assert!(matches!(
            refusals(&module)[..],
            [ValidationErrorKind::ForPartsLawEffect {
                part: 0,
                effect: LawEffect::OrderedCall,
                ..
            }]
        ));
    }

    #[test]
    fn a_part_that_leaves_the_body_is_refused() {
        let module = body(|insts, val_types| {
            val_types.insert(v(SPARE), Ty::Bool);
            let at = position(
                insts,
                |kind| matches!(kind, InstKind::Jump { label, .. } if *label == Label(3)),
            );
            insts.splice(
                at..=at,
                [
                    InstKind::Const {
                        dst: v(SPARE),
                        value: Literal::Bool(true),
                    },
                    InstKind::JumpIf {
                        cond: v(SPARE),
                        then_label: Label(5),
                        then_args: vec![],
                        else_label: Label(6),
                        else_args: vec![],
                    },
                    InstKind::BlockLabel {
                        label: Label(5),
                        params: vec![],
                    },
                    InstKind::Jump {
                        label: Label(2),
                        args: vec![],
                    },
                    InstKind::BlockLabel {
                        label: Label(6),
                        params: vec![],
                    },
                    InstKind::Jump {
                        label: Label(3),
                        args: vec![],
                    },
                ],
            );
        });
        let found = refusals(&module);
        assert!(
            matches!(
                found[..],
                [ValidationErrorKind::ForPartsLeaves { from, .. }] if from == Label(5)
            ),
            "{found:?}"
        );
    }

    /// Rule 2 after the drops: a storage one part writes, dropped in the
    /// part before it, is a crossing.
    #[test]
    fn a_drop_of_a_storage_another_part_writes_is_refused() {
        let module = body(|insts, _| {
            let slot = v(SPARE);
            let written = position(
                insts,
                |kind| matches!(kind, InstKind::BinOp { dst, .. } if *dst == v(P_NEXT)),
            );
            insts.insert(
                written + 1,
                InstKind::Assign {
                    target: RefTarget::Var(slot),
                    path: vec![],
                    value: v(I),
                    restores: false,
                },
            );
            let first = position(
                insts,
                |kind| matches!(kind, InstKind::BinOp { dst, .. } if *dst == v(S_NEXT)),
            );
            insts.insert(first, InstKind::Drop { src: slot });
        });
        let found = refusals(&module);
        assert!(
            found.iter().any(|refusal| matches!(refusal,
                ValidationErrorKind::ForPartsCrossing {
                    part: 0,
                    crossing: Crossing::Storage(slot),
                    ..
                } if *slot == v(SPARE))),
            "{found:?}"
        );
    }
}
