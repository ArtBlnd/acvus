//! Transforms (passes-structure §1, §4; passes-contracts.md). Each reads
//! facts through the analysis manager and changes IR only through
//! `ir::cfg::edit::Editor`, with `EditFacts` as its needs. Each states the
//! analysis-domain pairs it reads, the form it requires and keeps, its
//! standard algorithm and cost, why it keeps the sequential meaning
//! (compilation §1), and that it keeps extern-defined type identity
//! (compilation §3). A transform never calls another; composition is
//! `pipeline`'s.

mod bce;
mod commute;
mod dce;
mod dse;
mod expand;
mod gvn;
mod inline;
mod iv_canon;
mod licm;
mod loop_deletion;
mod mem2reg;
mod reborrow;
mod schedule;
mod sccp;
mod simplify_cfg;
mod sink;
mod split_calls;
mod sroa;
mod while_to_for;
#[cfg(test)]
pub(in crate::passes) mod tests;

pub use bce::Bce;
pub use commute::Commute;
pub use dce::Dce;
pub use dse::Dse;
pub use gvn::Gvn;
pub use inline::Inline;
pub use iv_canon::IvCanon;
pub use licm::Licm;
pub use loop_deletion::LoopDeletion;
pub use mem2reg::Mem2Reg;
pub use reborrow::Reborrow;
pub use schedule::Schedule;
pub use sccp::Sccp;
pub use simplify_cfg::SimplifyCfg;
pub use sink::Sink;
pub use split_calls::SplitCalls;
pub use sroa::Sroa;
pub use while_to_for::WhileToFor;

use crate::ir::cfg::{BlockId, Body, BodyId, Changes, Module};
use crate::passes::analysis::{AnalysisManager, Dominators, LoopId, LoopTree};

/// A transform over one body (passes-pipeline §1).
pub trait BodyTransform {
    const NAME: &'static str;
    fn run(module: &mut Module, body: BodyId, am: &mut AnalysisManager<'_>) -> Changes;
}

/// A transform that reads one body while editing another: `inline` only.
pub trait ModuleTransform {
    const NAME: &'static str;
    fn run(module: &mut Module, am: &mut AnalysisManager<'_>) -> Vec<(BodyId, Changes)>;
}

/// What a transform that edits through several editors in turn changed:
/// every aspect any of them recorded. Cost: O(aspects) per editor.
pub(in crate::passes) fn union_of<I>(changes: I) -> Changes
where
    I: IntoIterator<Item = Changes>,
{
    let mut union = Changes::default();
    for changed in changes {
        for aspect in changed.aspects() {
            union.record(aspect);
        }
    }
    union
}

/// The blocks the entry reaches, in a preorder of the dominator tree, so
/// that every block comes after its dominators. Cost: O(blocks).
fn dominator_preorder(body: &Body, dominators: &Dominators) -> Vec<BlockId> {
    let mut children = vec![Vec::new(); body.blocks().count()];
    for (block, _) in body.blocks() {
        if let Some(parent) = dominators.idom(block) {
            children[parent.index()].push(block);
        }
    }
    let mut order = Vec::new();
    let mut pending = vec![body.entry()];
    while let Some(block) = pending.pop() {
        order.push(block);
        pending.extend(children[block.index()].iter().rev().copied());
    }
    order
}

/// The nearest block that dominates both `a` and `b`; none where the
/// entry reaches only one of them, so that no block dominates both. Cost:
/// O(the depth of `a` in the dominator tree).
fn nearest_common_dominator(dominators: &Dominators, a: BlockId, b: BlockId) -> Option<BlockId> {
    if !dominators.reachable(a) || !dominators.reachable(b) {
        return None;
    }
    let mut common = a;
    while !dominators.dominates(common, b) {
        common = dominators.idom(common).expect("the entry dominates every block it reaches");
    }
    Some(common)
}

/// Every loop, the deepest first.
fn inner_first(loops: &LoopTree) -> Vec<LoopId> {
    let depth = |of: LoopId| std::iter::successors(Some(of), |at| loops.parent(*at)).count();
    let mut ordered: Vec<LoopId> = loops.loops().collect();
    ordered.sort_by_key(|of| std::cmp::Reverse(depth(*of)));
    ordered
}
