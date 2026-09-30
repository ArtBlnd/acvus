use std::collections::BTreeSet;

use acvus_utils::QualifiedRef;

use super::Phase;

/// Whether a call may be issued again. The derived order is the chain
/// `Pure < Idempotent < Opaque` (extern-fn §2).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Reissue {
    Pure,
    Idempotent,
    Opaque,
}

/// The contexts a call may touch.
pub type Contexts = BTreeSet<QualifiedRef>;

/// The backend's cost of a split call (effects §6). The compiler carries it
/// and never reads it: it compares for equality and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Cost(u64);

impl Cost {
    /// A cost as a backend states it.
    pub fn stated(cost: u64) -> Cost {
        Cost(cost)
    }

    /// The cost back to the backend that stated it.
    pub fn for_backend(self) -> u64 {
        self.0
    }
}

/// Whether a call may be split, and at what cost (effects §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SplitCost {
    Unavailable,
    WithCost(Cost),
}

impl SplitCost {
    /// A joined function splits only where both sides do, at one cost
    /// (types §2).
    pub fn join(self, other: SplitCost) -> SplitCost {
        match (self, other) {
            (SplitCost::WithCost(a), SplitCost::WithCost(b)) if a == b => SplitCost::WithCost(a),
            _ => SplitCost::Unavailable,
        }
    }

    /// Whether `self` is no more than `other`: equal, or `other` does not split.
    pub fn at_most(self, other: SplitCost) -> bool {
        self == other || other == SplitCost::Unavailable
    }
}

/// The effect of a call (effects, extern-fn §2). A pure call that writes
/// nothing commutes, and a call that writes a context never commutes; every
/// constructor keeps both.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    reissue: Reissue,
    commutes: bool,
    reads: Contexts,
    writes: Contexts,
    split: SplitCost,
}

impl Effect {
    pub fn new(reissue: Reissue, commutes: bool, reads: Contexts, writes: Contexts, split: SplitCost) -> Effect {
        Effect {
            reissue,
            commutes: (commutes || reissue == Reissue::Pure) && writes.is_empty(),
            reads,
            writes,
            split,
        }
    }

    /// The effect that is always sound (effects).
    pub fn opaque() -> Effect {
        Effect::new(Reissue::Opaque, false, Contexts::new(), Contexts::new(), SplitCost::Unavailable)
    }

    pub fn pure() -> Effect {
        Effect::new(Reissue::Pure, true, Contexts::new(), Contexts::new(), SplitCost::Unavailable)
    }

    pub fn reissue(&self) -> Reissue {
        self.reissue
    }

    pub fn commutes(&self) -> bool {
        self.commutes
    }

    pub fn reads(&self) -> &Contexts {
        &self.reads
    }

    pub fn writes(&self) -> &Contexts {
        &self.writes
    }

    pub fn split(&self) -> SplitCost {
        self.split
    }

    /// The least effect above both (types §2).
    pub fn join(&self, other: &Effect) -> Effect {
        Effect::new(
            self.reissue.max(other.reissue),
            self.commutes && other.commutes,
            self.reads.union(&other.reads).copied().collect(),
            self.writes.union(&other.writes).copied().collect(),
            self.split.join(other.split),
        )
    }

    /// Whether `self` is no more than `other` on the chain, commutes where
    /// `other` does, and splits no differently. The context sets are not
    /// part of this order: a bound on an effect bounds its level, and the
    /// sets only accumulate through `join`; `touches_at_most` orders them.
    pub fn at_most(&self, other: &Effect) -> bool {
        self.reissue <= other.reissue
            && (self.commutes || !other.commutes)
            && self.split.at_most(other.split)
    }

    /// Whether `self` touches no context `other` does not.
    pub fn touches_at_most(&self, other: &Effect) -> bool {
        self.reads.is_subset(&other.reads) && self.writes.is_subset(&other.writes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EffectTerm<V: Phase> {
    Known(Effect),
    Var(V::EffectVar),
}
