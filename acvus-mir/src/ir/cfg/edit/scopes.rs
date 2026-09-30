//! Scopes: an editor's kind as a type. Capabilities are sealed, so no pass
//! defines a scope; scopes are added here by the coordinator, and none may do
//! everything.

use super::keeps::{DefDominatesUses, Invariants, NeverDiverges, OrderChainWhole, SplitTakenOnce, StructureKept, TypesAgree};
use super::needs::{ContentFacts, DischargeFacts, DominanceFacts, Facts, LoanFacts, LoopFacts, NoFacts, OrderFacts};
use crate::ir::cfg::Aspect;

mod sealed {
    pub trait Sealed {}
}

/// A scope is a type-level name and holds no borrow, so it is `'static`;
/// that is what lets an editor borrow its needs for exactly its own life.
pub trait Scope: sealed::Sealed + 'static {
    /// What the editor may change; a recorded change outside this is the
    /// editor's defect (checked in debug and test builds).
    const CHANGES: &'static [Aspect];
    /// The facts its legality checks read, borrowed for `'n`.
    type Needs<'n>: Facts + ?Sized
    where
        Self: 'n;
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
/// Removal of what the core keeps, on a proven fact (G1).
pub trait Discharges: Scope {}
/// Inlining a call's callee body into the caller (S2).
pub trait Inlines: Scope {}
/// Drop insertion, the required step of ownership §6 only (S3).
pub trait InsertsDrops: Scope {}
/// A `String` read that is not its last use becoming a clone, the required
/// step of ownership §3 only (S4).
pub trait ClonesReads: Scope {}

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
/// Loop structure (compilation §4): its heads, joins and exits, with the
/// value and CFG edits a loop edit is composed of.
pub struct Loop;
/// Removal on a proven fact: dse and bce (G1).
pub struct Discharge;
/// Inlining (S2).
pub struct Inline;
/// Drop insertion, used by the required step only (S3); a take out of a
/// volatile context stands on the order chain (context §3).
pub struct Drops;
/// Clones of `String` reads, used by the required step only (S4); a lend
/// of a volatile context stands on the order chain (context §3).
pub struct Clones;

/// Every scope keeps these.
type Base = ((TypesAgree, DefDominatesUses, OrderChainWhole, StructureKept), NeverDiverges);

macro_rules! scope {
    ($name:ident, [$($aspect:ident),*], $lt:lifetime => $needs:ty, $keeps:ty, [$($cap:ident),*]) => {
        impl sealed::Sealed for $name {}
        impl Scope for $name {
            const CHANGES: &'static [Aspect] = &[$(Aspect::$aspect),*];
            type Needs<$lt> = $needs where Self: $lt;
            type Keeps = $keeps;
        }
        $(impl $cap for $name {})*
    };
}

scope!(Value, [Instructions, Storage, OrderTokens], 'n => dyn DominanceFacts + 'n, Base, [ReplacesValues]);
scope!(Remove, [Instructions, Storage, OrderTokens], 'n => NoFacts, Base, [RemovesInsts]);
scope!(Cfg, [CfgShape, Instructions, Storage, OrderTokens], 'n => NoFacts, Base, [EditsCfg]);
scope!(Sccp, [CfgShape, Instructions, Storage, OrderTokens], 'n => dyn DominanceFacts + 'n, Base, [ReplacesValues, EditsCfg]);
scope!(Motion, [Instructions, OrderTokens, Storage], 'n => dyn MotionFacts + 'n, (Base, SplitTakenOnce), [MovesInsts]);
scope!(Effect, [Instructions, OrderTokens], 'n => dyn OrderFacts + 'n, (Base, SplitTakenOnce), [EditsOrder]);
scope!(Storage, [Instructions, Storage], 'n => dyn LoanFacts + 'n, Base, [PromotesStorage]);
scope!(Loop, [CfgShape, Instructions, Storage, OrderTokens], 'n => dyn LoopEditFacts + 'n, Base, [ReplacesValues, EditsCfg, EditsLoops]);
scope!(Discharge, [Instructions, Storage, OrderTokens], 'n => dyn DischargeFacts + 'n, Base, [Discharges]);
scope!(Inline, [CfgShape, Instructions, Storage, OrderTokens, Types], 'n => NoFacts, (Base, SplitTakenOnce), [Inlines]);
scope!(Drops, [Instructions, Storage, OrderTokens], 'n => dyn DropFacts + 'n, Base, [InsertsDrops]);
scope!(Clones, [Instructions, Storage, OrderTokens, Types], 'n => NoFacts, Base, [ClonesReads]);

/// What motion reads: dominance and loans. Effect order is not motion's:
/// an effect's token keeps it on its chain, so dominance alone keeps it in
/// order (D54).
pub trait MotionFacts: DominanceFacts + LoanFacts {}

/// What the loop edits read: dominance, and each loop's preheader (F2).
pub trait LoopEditFacts: DominanceFacts + LoopFacts {}

/// What drop insertion reads: dominance for a value's drop, and whether a
/// storage's content is read after a position for a content's drop.
pub trait DropFacts: DominanceFacts + ContentFacts {}
impl<T> DropFacts for T where T: DominanceFacts + ContentFacts + ?Sized {}
impl<T> MotionFacts for T where T: DominanceFacts + LoanFacts + ?Sized {}
impl<T> LoopEditFacts for T where T: DominanceFacts + LoopFacts + ?Sized {}
