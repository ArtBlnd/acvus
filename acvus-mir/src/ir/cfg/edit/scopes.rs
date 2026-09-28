//! Scopes: an editor's kind as a type. Capabilities are sealed, so no pass
//! defines a scope; scopes are added here by the coordinator, and none may do
//! everything.

use super::keeps::{DefDominatesUses, Invariants, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree};
use super::needs::{DominanceFacts, Facts, LoanFacts, NoFacts, OrderFacts};
use crate::ir::cfg::Aspect;

mod sealed {
    pub trait Sealed {}
}

pub trait Scope: sealed::Sealed {
    /// What the editor may change; a recorded change outside this is the
    /// editor's defect (checked in debug and test builds).
    const CHANGES: &'static [Aspect];
    /// The facts its legality checks read.
    type Needs: Facts + ?Sized;
    /// The invariants every edit keeps, one checker each.
    type Keeps: Invariants;
}

pub trait ReplacesValues: Scope {}
pub trait RemovesInsts: Scope {}
pub trait EditsCfg: Scope {}
pub trait MovesInsts: Scope {}
pub trait EditsOrder: Scope {}
pub trait PromotesStorage: Scope {}
pub trait EditsLoops: Scope {}

/// Values: replace uses, insert pure instructions (GVN, reborrow).
pub struct Value;
/// Removal of unused instructions (DCE, DSE).
pub struct Remove;
/// CFG edits keeping instructions and structure (CFG simplification).
pub struct Cfg;
/// Values and CFG together (SCCP).
pub struct Sccp;
/// Instruction motion (LICM, sinking, scheduling).
pub struct Motion;
/// Effect order and call splitting (commute, split).
pub struct Effect;
/// Storage to values (SSA construction, SROA).
pub struct Storage;
/// Loop structure (compilation §4); its operations come with the loop design.
pub struct Loop;

/// Every scope keeps these.
type Base = (TypesAgree, DefDominatesUses, OrderChainWhole, StructureKept);

macro_rules! scope {
    ($name:ident, [$($aspect:ident),*], $needs:ty, $keeps:ty, [$($cap:ident),*]) => {
        impl sealed::Sealed for $name {}
        impl Scope for $name {
            const CHANGES: &'static [Aspect] = &[$(Aspect::$aspect),*];
            type Needs = $needs;
            type Keeps = $keeps;
        }
        $(impl $cap for $name {})*
    };
}

scope!(Value, [Instructions, Storage], dyn DominanceFacts, Base, [ReplacesValues]);
scope!(Remove, [Instructions, Storage, OrderTokens], NoFacts, Base, [RemovesInsts]);
scope!(Cfg, [CfgShape, Instructions, Storage, OrderTokens], NoFacts, Base, [EditsCfg]);
scope!(Sccp, [Instructions, CfgShape], dyn DominanceFacts, Base, [ReplacesValues, EditsCfg]);
scope!(Motion, [Instructions, OrderTokens, Storage], dyn MotionFacts, (Base, SplitTakenOnce), [MovesInsts]);
scope!(Effect, [Instructions, OrderTokens], dyn OrderFacts, (Base, SplitTakenOnce), [EditsOrder]);
scope!(Storage, [Instructions, Storage], dyn LoanFacts, Base, [PromotesStorage]);
scope!(Loop, [CfgShape, Instructions], NoFacts, Base, [EditsLoops]);

/// What motion reads: dominance and loans. Effect order is not motion's:
/// an effect's token keeps it on its chain, so dominance alone keeps it in
/// order (D54).
pub trait MotionFacts: DominanceFacts + LoanFacts {}
impl<T> MotionFacts for T where T: DominanceFacts + LoanFacts + ?Sized {}
