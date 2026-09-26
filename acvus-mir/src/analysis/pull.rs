//! RFC-0089 rule 1: a pull loop's count is at most what the call that made
//! its iterator states as `len(ret)` (RFC-0082 rule 4), where nothing but
//! the header touches the iterator. Each pull yields at most one of the
//! elements the call stated, so the loop runs at most that many iterations.
//! And RFC-0089 rule 5: a pull runs ahead of a body exit where its instance
//! states `returns` and no effect and nothing reads the iterator after the
//! loop.

use crate::analysis::domtree::DomTree;
use crate::analysis::inst_info;
use crate::analysis::loans::Loans;
use crate::analysis::loop_deps::InstAt;
use crate::analysis::loops::{NaturalLoop, Term};
use crate::cfg::{BlockIdx, CfgBody, Terminator};
use crate::ir::{Callee, InstKind, ValueId};
use crate::laws::{LawTable, PostTerm, Postcondition, Relation, Subject};
use crate::ty::{Mutability, Ty};

/// The call that made a pull loop's iterator, and the most iterations the
/// loop runs: a term over that call's arguments as they stood when it ran.
#[derive(Debug, Clone, PartialEq)]
pub struct PullBound {
    pub made_at: InstAt,
    pub count: Term,
}

struct HeaderLend {
    at: InstAt,
    lent: ValueId,
    iterator: ValueId,
}

struct WholeStore {
    at: InstAt,
    value: ValueId,
}

struct MakingCall<'c> {
    at: InstAt,
    callee: &'c Callee,
    args: &'c [ValueId],
}

/// `None` where `natural` is no pull loop, or where its count is unknown
/// on entry.
pub fn bound(
    cfg: &CfgBody,
    laws: &LawTable,
    loans: &Loans<'_>,
    domtree: &DomTree,
    natural: &NaturalLoop,
) -> Option<PullBound> {
    let Terminator::While { .. } = &cfg.blocks[natural.header.0].terminator else {
        return None;
    };
    let lend = header_lend(cfg, natural.header)?;
    let store = only_store_beside_the_pull(cfg, loans, &lend)?;
    if !domtree.dominates(store.at.block, natural.header) || natural.contains(store.at.block) {
        return None;
    }
    let made = making_call(cfg, store.value)?;
    let count = laws
        .postconditions_of(made.callee)
        .iter()
        .find_map(|stated| match stated {
            Postcondition {
                left: PostTerm::Len(Subject::Ret),
                relation: Relation::Eq | Relation::Le | Relation::Lt,
                right,
            } => over_args(cfg, right, made.args),
            _ => None,
        })?;
    Some(PullBound {
        made_at: made.at,
        count,
    })
}

/// Whether the header's pull runs ahead of an exit from the loop's body:
/// every pull returns, changes nothing but the iterator, and no path from
/// the loop's exits reads the iterator before a store replaces it whole.
pub fn runs_ahead(
    cfg: &CfgBody,
    laws: &LawTable,
    loans: &Loans<'_>,
    header: BlockIdx,
    loop_blocks: &[BlockIdx],
) -> bool {
    let Some(lend) = header_lend(cfg, header) else {
        return false;
    };
    let Some(pull) = cfg.blocks[header.0]
        .insts
        .iter()
        .map(|inst| &inst.kind)
        .find(|kind| is_pull_of(kind, lend.lent))
    else {
        return false;
    };
    let InstKind::FunctionCall {
        callee,
        callee_ty: Ty::Fn { effect, .. },
        order: None,
        ..
    } = pull
    else {
        return false;
    };
    laws.returns_of(callee).returns_or_traps()
        && effect.get().is_pure()
        && !read_after_the_loop(cfg, loans, lend.iterator, loop_blocks)
}

/// A walk from each edge leaving the loop, which a whole store of the
/// iterator ends: any access of it on the way but a release reads it.
fn read_after_the_loop(
    cfg: &CfgBody,
    loans: &Loans<'_>,
    iterator: ValueId,
    loop_blocks: &[BlockIdx],
) -> bool {
    let reaches_iterator = |value: ValueId| {
        value == iterator || loans.holds(value).any(|loan| loan.storage.slot() == Some(iterator))
    };
    let mut work: Vec<BlockIdx> = loop_blocks
        .iter()
        .flat_map(|block| cfg.successors(*block))
        .filter(|succ| !loop_blocks.contains(succ))
        .collect();
    let mut seen: rustc_hash::FxHashSet<BlockIdx> = rustc_hash::FxHashSet::default();
    'blocks: while let Some(block) = work.pop() {
        if !seen.insert(block) {
            continue;
        }
        let held = &cfg.blocks[block.0];
        for inst in &held.insts {
            match &inst.kind {
                InstKind::Drop { .. } => {}
                InstKind::Assign { target, path, .. }
                    if inst_info::storage(target) == Some(iterator) && path.is_empty() =>
                {
                    continue 'blocks;
                }
                InstKind::Ref { target, .. }
                | InstKind::Take { target, .. }
                | InstKind::Assign { target, .. }
                    if inst_info::storage(target) == Some(iterator) =>
                {
                    return true;
                }
                kind if inst_info::uses(kind).into_iter().any(reaches_iterator) => return true,
                _ => {}
            }
        }
        if inst_info::terminator_uses(&held.terminator)
            .into_iter()
            .any(reaches_iterator)
        {
            return true;
        }
        work.extend(cfg.successors(block));
    }
    false
}

fn header_lend(cfg: &CfgBody, header: BlockIdx) -> Option<HeaderLend> {
    let insts = &cfg.blocks[header.0].insts;
    let lend = insts.iter().enumerate().find_map(|(at, inst)| match &inst.kind {
        InstKind::Ref {
            dst,
            target,
            mutability: Mutability::Mut,
            ..
        } => Some(HeaderLend {
            at: InstAt { block: header, at },
            lent: *dst,
            iterator: inst_info::storage(target)?,
        }),
        _ => None,
    })?;
    insts
        .iter()
        .any(|inst| is_pull_of(&inst.kind, lend.lent))
        .then_some(lend)
}

fn is_pull_of(kind: &InstKind, lent: ValueId) -> bool {
    matches!(kind, InstKind::FunctionCall { callee: Callee::Extern { .. }, args, .. }
        if args.as_slice() == [lent])
}

fn only_store_beside_the_pull(
    cfg: &CfgBody,
    loans: &Loans<'_>,
    lend: &HeaderLend,
) -> Option<WholeStore> {
    let iterator = lend.iterator;
    let reaches_iterator = |value: ValueId| {
        value == iterator || loans.holds(value).any(|loan| loan.storage.slot() == Some(iterator))
    };
    let mut store: Option<WholeStore> = None;
    for (b, block) in cfg.blocks.iter().enumerate() {
        for (at, inst) in block.insts.iter().enumerate() {
            let here = InstAt {
                block: BlockIdx(b),
                at,
            };
            let kind = &inst.kind;
            let the_pull = here.block == lend.at.block && is_pull_of(kind, lend.lent);
            if here == lend.at || the_pull || matches!(kind, InstKind::Drop { .. }) {
                continue;
            }
            match kind {
                InstKind::Assign {
                    target,
                    path,
                    value,
                    ..
                } if inst_info::storage(target) == Some(iterator) => {
                    if !path.is_empty() || store.is_some() {
                        return None;
                    }
                    store = Some(WholeStore {
                        at: here,
                        value: *value,
                    });
                }
                InstKind::Ref { target, .. } | InstKind::Take { target, .. }
                    if inst_info::storage(target) == Some(iterator) =>
                {
                    return None;
                }
                _ if inst_info::uses(kind).into_iter().any(reaches_iterator) => return None,
                _ => {}
            }
        }
        if inst_info::terminator_uses(&block.terminator)
            .into_iter()
            .any(reaches_iterator)
        {
            return None;
        }
    }
    store
}

fn making_call(cfg: &CfgBody, value: ValueId) -> Option<MakingCall<'_>> {
    cfg.blocks.iter().enumerate().find_map(|(b, block)| {
        block.insts.iter().enumerate().find_map(|(at, inst)| match &inst.kind {
            InstKind::FunctionCall {
                dst,
                callee: callee @ Callee::Extern { .. },
                args,
                ..
            } if *dst == value => Some(MakingCall {
                at: InstAt {
                    block: BlockIdx(b),
                    at,
                },
                callee,
                args,
            }),
            _ => None,
        })
    })
}

fn over_args(cfg: &CfgBody, term: &PostTerm, args: &[ValueId]) -> Option<Term> {
    let over = |term: &PostTerm| over_args(cfg, term, args);
    Some(match term {
        PostTerm::Const(value) => Term::int(*value),
        PostTerm::Param(at) => {
            let arg = *args.get(*at)?;
            match cfg.val_types.get(&arg)? {
                Ty::Int(_) => Term::Value(arg),
                _ => return None,
            }
        }
        PostTerm::Len(Subject::Param(at)) => Term::Len(*args.get(*at)?),
        PostTerm::Add(a, b) => over(a)?.add(over(b)?),
        PostTerm::Sub(a, b) => over(a)?.sub(over(b)?),
        PostTerm::Mul(a, b) => Term::Mul(Box::new(over(a)?), Box::new(over(b)?)),
        PostTerm::Max(a, b) => over(a)?.max(over(b)?),
        PostTerm::Ret | PostTerm::Len(Subject::Ret) | PostTerm::Old(_) => return None,
    })
}
