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
mod gvn;
mod inline;
mod licm;
mod mem2reg;
mod reborrow;
mod schedule;
mod sccp;
mod simplify_cfg;
mod sink;
mod split_calls;
mod sroa;

pub use bce::Bce;
pub use commute::Commute;
pub use dce::Dce;
pub use dse::Dse;
pub use gvn::Gvn;
pub use inline::Inline;
pub use licm::Licm;
pub use mem2reg::Mem2Reg;
pub use reborrow::Reborrow;
pub use schedule::Schedule;
pub use sccp::Sccp;
pub use simplify_cfg::SimplifyCfg;
pub use sink::Sink;
pub use split_calls::SplitCalls;
pub use sroa::Sroa;

use crate::ir::cfg::{BodyId, Changes, Module};
use crate::passes::analysis::AnalysisManager;

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
