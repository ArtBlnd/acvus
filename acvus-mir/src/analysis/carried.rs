//! What one iteration of a loop hands the next, and whether that orders
//! the iterations.
//!
//! Every header parameter is carried state, and each is exactly one of:
//!
//! - [`Carried::Iv`]: affine in the loop (`analysis::affine`).
//! - [`Carried::Merge`]: every back edge sends `p ⊕ x`, and inside the loop
//!   `p` is read only as that `⊕`'s operand and `p ⊕ x` only by the back
//!   edges. The iterations then contribute their `x`s and nothing reads a
//!   partial result, so the state is a reduction.
//! - [`Carried::Recurrence`]: anything else.
//!
//! A merge is exact when `⊕` is associative and commutative at its type:
//! integer `+` and `*`, which wrap (RFC-0037). Float `+` and `*` round, so a
//! float merge is inexact: regrouping it gives a different number, and
//! whether to regroup is the lowerer's, by its reassociation policy
//! (RFC-0066). Integer and float `+` and `*` reach MIR as `InstKind::BinOp`,
//! the instruction an operator on a language-owned type is (RFC-0020), and
//! that is how they are recognized. A call of an extern that declares
//! itself associative (RFC-0082 rule 2), such as `num::min`, is an exact
//! merge on the extern's word: the analysis reads the declaration and does
//! not discover the law. `&&` and `||` reach MIR as the lowering's
//! short-circuit `Diamond`, and recognizing that form is not settled; until
//! it is, a loop that merges through one carries a recurrence.
//!
//! A loop is weak when every carried parameter is an `Iv` or a `Merge`, no
//! instruction of the body carries an `Order`, and every storage the body
//! writes is one its `SliceMut` source lends. Otherwise it is strong, and
//! [`CarriedState::dependences`] names each reason. The kind of the state
//! decides, not its count: how many merges a loop carries is a cost, and
//! cost is the lowerer's.
//!
//! An `Order` is how MIR marks an effect that keeps its place in the run:
//! a call whose effect is not Pure carries one and a Pure call carries
//! none (RFC-0013, RFC-0046), so an ordered effect is an instruction that
//! carries one. A write is what `analysis::loans` says an instruction
//! writes, and a `Commit` of a context. The source's own storage is the
//! element write RFC-0057 rule 3 admits: rule 5 holds the container
//! exclusively for the loop, so the element is the only path to it.
//!
//! A merge through storage, `v.push(x)` in a loop, is a write of `v`, and
//! what kind of merge it is, ordered or not, is the extern's to declare
//! (RFC-0066 rule 6). Where every write of a storage in the loop is a call
//! of one extern that declares a `fold` law (RFC-0082 rule 3), and the loop
//! reads that storage only to lend it to those calls, the storage is a
//! [`StorageMerge`] and not a dependence. Every other write is strong.
//!
//! A loop left from anywhere but its header, by a `break` or a `return`,
//! is ordered: the iterations after the one that leaves never run.
//!
//! Inside `anyorder` the lowering gives every effectful call the region's
//! entry order and merges the order the call yields into the region's
//! accumulator, a header parameter of type `Order` whose back edges send a
//! chain of `Merge`s over it: a [`MergeOp::Order`]. A call whose order input
//! is that accumulator's entry value and whose order output reaches only the
//! chain is unordered by the author's declaration (RFC-0089 rule 6), and it
//! is not an ordered effect here. After `spawn_split` such a call is a
//! `Spawn` that takes the order and the `Eval` that yields it, and the pair
//! is excused as the one call it was.

use crate::ir::BinOp;
use rustc_hash::FxHashMap;

use crate::analysis::affine::{AffineValues, Arithmetic, Derivation, exact_under_wrapping};
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loops::{Loop, LoopKind, passed_into_body};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::graph::QualifiedRef;
use crate::ir::{CallIdentity, Callee, ForSource, InstKind, ValueId};
use crate::laws::{BinaryLaws, FoldLaw, LawTable, Laws};
use crate::ty::{Mutability, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeOp {
    Add,
    Mul,
    Extern(ExternMerge),
    Order,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternMerge {
    pub callee: QualifiedRef,
    pub instance: usize,
    pub commutative: bool,
    pub identity: CallIdentity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageMerge {
    pub storage: ValueId,
    pub callee: QualifiedRef,
    pub instance: usize,
    pub fold: FoldLaw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carried {
    Iv,
    Merge { op: MergeOp, exact: bool },
    Recurrence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarriedParam {
    pub param: ValueId,
    pub carried: Carried,
}

/// One reason a loop is strong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dependence {
    Recurrence(ValueId),
    OrderedEffect {
        block: BlockIdx,
    },
    StorageWrite(ValueId),
    ContextWrite(QualifiedRef),
    /// A `break` or `return` leaves from `block`, not from the header: the
    /// iterations after the one that leaves never run, so which runs first
    /// decides what the loop does.
    EarlyExit {
        block: BlockIdx,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strength {
    /// The iterations may run in any order.
    Weak,
    Strong,
}

pub struct CarriedState {
    pub params: Vec<CarriedParam>,
    pub storage_merges: Vec<StorageMerge>,
    pub dependences: Vec<Dependence>,
    /// By `dst`: RFC-0089 rule 6's order-carrying instructions.
    pub unordered: Vec<ValueId>,
}

impl CarriedState {
    pub fn of(loans: &Loans<'_>, loop_: &Loop, affine: &AffineValues, laws: &LawTable) -> Self {
        let cfg = loans.cfg();
        let natural = &loop_.natural;
        let into_body = passed_into_body(cfg, natural.header);
        let body = Body {
            loans,
            loop_,
            laws,
            reads: Reads::in_loop(cfg, loop_, &into_body),
            arithmetic: Arithmetic::in_loop(cfg, loop_),
            into_body,
        };
        let params: Vec<CarriedParam> = cfg.blocks[natural.header.0]
            .params
            .iter()
            .enumerate()
            .map(|(index, &param)| {
                let carried = match affine.get(param).map(|a| &a.derivation) {
                    Some(Derivation::Carried { .. }) => Carried::Iv,
                    _ => match body.merge(index, param) {
                        Some(merge) => merge,
                        None => Carried::Recurrence,
                    },
                };
                CarriedParam { param, carried }
            })
            .collect();

        let mut dependences: Vec<Dependence> = params
            .iter()
            .filter(|p| p.carried == Carried::Recurrence)
            .map(|p| Dependence::Recurrence(p.param))
            .collect();
        let unordered = body.unordered_calls(&params);
        let lent = lent_by_source(loop_, loans);
        let storage_merges = body.storage_merges(&lent);
        for block in natural.blocks().filter(|&block| block != natural.header) {
            let leaves = matches!(cfg.blocks[block.0].terminator, Terminator::Return { .. })
                || cfg
                    .successors(block)
                    .iter()
                    .any(|&succ| !natural.contains(succ));
            if leaves {
                dependences.push(Dependence::EarlyExit { block });
            }
        }
        for block in natural.blocks() {
            for inst in &cfg.blocks[block.0].insts {
                let mut found: Vec<Dependence> = loans
                    .storage_effect(&inst.kind)
                    .writes
                    .into_iter()
                    .filter(|storage| !lent.contains(storage))
                    .filter(|storage| !storage_merges.iter().any(|m| m.storage == *storage))
                    .map(Dependence::StorageWrite)
                    .collect();
                if let InstKind::Commit { context, .. } = &inst.kind {
                    found.push(Dependence::ContextWrite(*context));
                }
                let excused = inst_info::defs(&inst.kind)
                    .iter()
                    .any(|dst| unordered.contains(dst));
                if carries_order(&inst.kind) && !excused {
                    found.push(Dependence::OrderedEffect { block });
                }
                for dependence in found {
                    if !dependences.contains(&dependence) {
                        dependences.push(dependence);
                    }
                }
            }
        }
        Self {
            params,
            storage_merges,
            dependences,
            unordered,
        }
    }

    pub fn strength(&self) -> Strength {
        match self.dependences.is_empty() {
            true => Strength::Weak,
            false => Strength::Strong,
        }
    }

    /// RFC-0057 rule 3's independence: a weak loop that carries nothing.
    pub fn runs_apart(&self) -> bool {
        self.params.is_empty()
            && self.storage_merges.is_empty()
            && self.strength() == Strength::Weak
    }
}

struct Body<'a, 'cfg> {
    loans: &'a Loans<'cfg>,
    loop_: &'a Loop,
    laws: &'a LawTable,
    reads: Reads,
    arithmetic: Arithmetic,
    into_body: FxHashMap<ValueId, ValueId>,
}

impl<'cfg> Body<'_, 'cfg> {
    fn cfg(&self) -> &'cfg CfgBody {
        self.loans.cfg()
    }

    /// Whether `value` is header parameter `param`: the parameter itself,
    /// or the body parameter its traversal's body edge passes it to.
    fn is_param(&self, value: ValueId, param: ValueId) -> bool {
        value == param || self.into_body.get(&param) == Some(&value)
    }

    /// `Carried::Merge` when header parameter `param`, at `index`, is one.
    fn merge(&self, index: usize, param: ValueId) -> Option<Carried> {
        let natural = &self.loop_.natural;
        let next = natural.back_arg(self.cfg(), index)?;
        let merge = match self.arithmetic.get(next) {
            Some(operation) => {
                let taken = self.into_body.get(&param).copied();
                let reads_param = operation.one_operand_is(param)
                    || taken.is_some_and(|taken| operation.one_operand_is(taken));
                if !reads_param {
                    return None;
                }
                let op = match operation.op {
                    BinOp::Add => MergeOp::Add,
                    BinOp::Mul => MergeOp::Mul,
                    _ => return None,
                };
                let exact = match &self.cfg().val_types[&next] {
                    ty if exact_under_wrapping(ty) => true,
                    Ty::Float => false,
                    _ => return None,
                };
                Carried::Merge { op, exact }
            }
            None if self.order_chain(next, param).is_some() => Carried::Merge {
                op: MergeOp::Order,
                exact: true,
            },
            None => Carried::Merge {
                op: MergeOp::Extern(self.extern_merge(next, param)?),
                exact: true,
            },
        };
        let read_only_as_operand = self.reads.count(param) == 1;
        let read_only_by_back_edges = self.reads.count(next) == natural.latches.len();
        (read_only_as_operand && read_only_by_back_edges).then_some(merge)
    }

    /// The `Merge`s from `next` down to `param`, an `Order`, where each link
    /// is read by the next one alone and `param` is an operand of the last.
    fn order_chain(&self, next: ValueId, param: ValueId) -> Option<Vec<ValueId>> {
        if self.cfg().val_types.get(&param) != Some(&Ty::Order) {
            return None;
        }
        let mut chain = vec![next];
        let mut at = next;
        loop {
            let InstKind::Merge { orders, .. } = self.defining(at)? else {
                return None;
            };
            if orders.iter().any(|order| self.is_param(*order, param)) {
                return Some(chain);
            }
            let linked: Vec<ValueId> = orders
                .iter()
                .copied()
                .filter(|order| self.reaches_through_merges(*order, param))
                .collect();
            let [link] = linked[..] else {
                return None;
            };
            if self.reads.count(link) != 1 {
                return None;
            }
            chain.push(link);
            at = link;
        }
    }

    fn reaches_through_merges(&self, value: ValueId, param: ValueId) -> bool {
        match self.defining(value) {
            Some(InstKind::Merge { orders, .. }) => orders
                .iter()
                .any(|order| self.is_param(*order, param) || self.reaches_through_merges(*order, param)),
            _ => false,
        }
    }

    /// RFC-0089 rule 6's calls, by the `dst` of each instruction.
    fn unordered_calls(&self, params: &[CarriedParam]) -> Vec<ValueId> {
        let natural = &self.loop_.natural;
        let mut unordered = Vec::new();
        for (index, carried) in params.iter().enumerate() {
            let Carried::Merge {
                op: MergeOp::Order, ..
            } = carried.carried
            else {
                continue;
            };
            let (Some(next), Some(init)) = (
                natural.back_arg(self.cfg(), index),
                natural.entry_arg(self.cfg(), index),
            ) else {
                continue;
            };
            let Some(chain) = self.order_chain(next, carried.param) else {
                continue;
            };
            let merged_into_chain = |order: ValueId| {
                self.reads.count(order) == 1
                    && chain.iter().any(|link| {
                        matches!(self.defining(*link),
                            Some(InstKind::Merge { orders, .. }) if orders.contains(&order))
                    })
            };
            for kind in self.body_insts() {
                match kind {
                    InstKind::FunctionCall {
                        dst,
                        order: Some(edge),
                        ..
                    } if edge.before == init && merged_into_chain(edge.after) => {
                        unordered.push(*dst);
                    }
                    InstKind::Eval {
                        dst,
                        src,
                        order: Some(after),
                    } if merged_into_chain(*after) && self.reads.count(*src) == 1 => {
                        let Some(InstKind::Spawn {
                            dst: handle,
                            order: Some(before),
                            ..
                        }) = self.defining(*src)
                        else {
                            continue;
                        };
                        if *before == init {
                            unordered.push(*handle);
                            unordered.push(*dst);
                        }
                    }
                    _ => {}
                }
            }
        }
        unordered
    }

    fn body_insts(&self) -> impl Iterator<Item = &'cfg InstKind> + '_ {
        let cfg = self.cfg();
        self.loop_
            .natural
            .blocks()
            .flat_map(move |block| &cfg.blocks[block.0].insts)
            .map(|inst| &inst.kind)
    }

    fn extern_merge(&self, next: ValueId, param: ValueId) -> Option<ExternMerge> {
        let InstKind::FunctionCall { callee, args, .. } = self.defining(next)? else {
            return None;
        };
        let Callee::Extern { id, instance, .. } = callee else {
            return None;
        };
        let Laws::Binary(BinaryLaws {
            associative: true,
            commutative,
            identity,
        }) = self.laws.of_callee(callee)
        else {
            return None;
        };
        let &[first, second] = args.as_slice() else {
            return None;
        };
        let param_is_first = self.is_param(first, param) && !self.is_param(second, param);
        let param_is_second = self.is_param(second, param) && !self.is_param(first, param);
        (param_is_first || (*commutative && param_is_second)).then_some(ExternMerge {
            callee: *id,
            instance: *instance,
            commutative: *commutative,
            identity: match identity {
                Some(_) => CallIdentity::Declared,
                None => CallIdentity::OptionLifted,
            },
        })
    }

    fn defining(&self, value: ValueId) -> Option<&InstKind> {
        self.loop_
            .natural
            .blocks()
            .flat_map(|block| &self.cfg().blocks[block.0].insts)
            .map(|inst| &inst.kind)
            .find(|kind| inst_info::defs(kind).contains(&value))
    }

    fn storage_merges(&self, lent: &[ValueId]) -> Vec<StorageMerge> {
        let loans = self.loans;
        let insts: Vec<&InstKind> = self
            .loop_
            .natural
            .blocks()
            .flat_map(|block| &self.cfg().blocks[block.0].insts)
            .map(|inst| &inst.kind)
            .collect();
        let mut written: Vec<ValueId> = Vec::new();
        for kind in &insts {
            for storage in loans.storage_effect(kind).writes {
                if !lent.contains(&storage) && !written.contains(&storage) {
                    written.push(storage);
                }
            }
        }
        written
            .into_iter()
            .filter_map(|storage| self.storage_merge(storage, &insts))
            .collect()
    }

    fn storage_merge(&self, storage: ValueId, insts: &[&InstKind]) -> Option<StorageMerge> {
        let loans = self.loans;
        let mut merge: Option<StorageMerge> = None;
        let mut lenders: Vec<ValueId> = Vec::new();
        for kind in insts {
            let effect = loans.storage_effect(kind);
            if !effect.writes.contains(&storage) {
                continue;
            }
            let found = self.fold_call(kind, storage)?;
            lenders.push(found.lender);
            match merge {
                Some(merged) if merged != found.merge => return None,
                Some(_) => {}
                None => merge = Some(found.merge),
            }
        }
        let only_lent_to_the_folds = insts.iter().all(|kind| {
            let effect = loans.storage_effect(kind);
            if !effect.reads.contains(&storage) || effect.writes.contains(&storage) {
                return true;
            }
            match kind {
                InstKind::Ref { dst, .. } => lenders.contains(dst) && self.reads.count(*dst) == 1,
                _ => false,
            }
        });
        merge.filter(|_| only_lent_to_the_folds)
    }

    fn fold_call(&self, kind: &InstKind, storage: ValueId) -> Option<FoldCall> {
        let loans = self.loans;
        let InstKind::FunctionCall { callee, args, .. } = kind else {
            return None;
        };
        let Callee::Extern { id, instance, .. } = callee else {
            return None;
        };
        let Laws::Fold(fold) = self.laws.of_callee(callee) else {
            return None;
        };
        let (&lender, rest) = args.split_first()?;
        let lends = |value: ValueId, mutability: Mutability| {
            loans
                .holds(value)
                .any(|loan| loan.storage.slot() == Some(storage) && loan.mutability == mutability)
        };
        let lent_by_the_state_alone = lends(lender, Mutability::Mut)
            && !rest
                .iter()
                .any(|&arg| lends(arg, Mutability::Mut) || lends(arg, Mutability::Shared));
        lent_by_the_state_alone.then_some(FoldCall {
            lender,
            merge: StorageMerge {
                storage,
                callee: *id,
                instance: *instance,
                fold: *fold,
            },
        })
    }
}

struct FoldCall {
    lender: ValueId,
    merge: StorageMerge,
}

/// How many times the loop's instructions and terminators read each value.
struct Reads {
    by_value: FxHashMap<ValueId, usize>,
}

impl Reads {
    /// A body parameter a header parameter is passed to is read as that
    /// header parameter, and the pass itself is no read.
    fn in_loop(cfg: &CfgBody, loop_: &Loop, into_body: &FxHashMap<ValueId, ValueId>) -> Self {
        let as_header: FxHashMap<ValueId, ValueId> = into_body
            .iter()
            .map(|(header, body)| (*body, *header))
            .collect();
        let header = loop_.natural.header;
        let mut by_value: FxHashMap<ValueId, usize> = FxHashMap::default();
        for at in loop_.natural.blocks() {
            let block = &cfg.blocks[at.0];
            let insts = block
                .insts
                .iter()
                .flat_map(|inst| inst_info::uses(&inst.kind));
            let mut terminator = inst_info::terminator_uses(&block.terminator);
            if at == header
                && let Some(traversal) = block.terminator.traversal()
            {
                for passed in traversal.body_args.iter() {
                    if into_body.contains_key(passed)
                        && let Some(at) = terminator.iter().position(|used| used == passed)
                    {
                        terminator.remove(at);
                    }
                }
            }
            for value in insts.chain(terminator) {
                let value = as_header.get(&value).copied().unwrap_or(value);
                *by_value.entry(value).or_default() += 1;
            }
        }
        Self { by_value }
    }

    fn count(&self, value: ValueId) -> usize {
        match self.by_value.get(&value) {
            Some(count) => *count,
            None => 0,
        }
    }
}

/// The storages a `SliceMut` source lends the loop mutably.
fn lent_by_source(loop_: &Loop, loans: &Loans<'_>) -> Vec<ValueId> {
    let LoopKind::For {
        source: ForSource::SliceMut(slice),
    } = loop_.kind
    else {
        return Vec::new();
    };
    loans
        .names(slice)
        .iter()
        .filter(|loan| loan.mutability == Mutability::Mut)
        .filter_map(|loan| loan.storage.slot())
        .collect()
}

pub fn carries_order(kind: &InstKind) -> bool {
    match kind {
        InstKind::FunctionCall { order, .. } => order.is_some(),
        InstKind::Spawn { order, .. } | InstKind::Eval { order, .. } => order.is_some(),
        _ => false,
    }
}
