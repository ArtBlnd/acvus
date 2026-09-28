//! The relations A speaks in: `eq`, `join`, `enter` (typeck-structure
//! section 2). Each either records what it learned or refuses in B's terms.

use acvus_utils::{Astr, QualifiedRef};

use super::{Infer, Term};
use crate::decl::{CastTree, VerifiedExternCast};
use crate::ty::{EffectTerm, Infer as InferPhase, TyVarId};

/// Why two terms do not relate: B's refusals (refusals).
#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch {
    /// Two heads that are not one head.
    Heads { a: Term, b: Term },
    /// A variable would contain itself (occurs check).
    Occurs { var: TyVarId, in_term: Term },
    /// A field an opaque object lacks, which a transparent one would add.
    OpaqueLacksField { object: Term, field: Astr },
    /// A variant a closed enum lacks (patterns §2).
    OpaqueLacksVariant { r#enum: Term, tag: Astr },
    /// Two enums of two names (types §2).
    TwoEnums { a: QualifiedRef, b: QualifiedRef },
    /// A variable's bound does not admit the term.
    Bound { var: TyVarId, term: Term },
    /// `#` on one side only (types §8).
    Mark { a: Term, b: Term },
}

/// What an `enter` did (types §3, §7).
#[derive(Debug, Clone, PartialEq)]
pub enum Entered {
    /// The value's head is not closed yet; ask again.
    NotYet,
    /// The slot took the value's type, or folded it up the tree.
    Folded,
    /// The value converts to the slot through one declared conversion.
    Converted(VerifiedExternCast),
}

impl Infer {
    /// Unification with the occurs check; every constructor invariant
    /// (types §4).
    pub fn eq(&mut self, a: Term, b: Term) -> Result<(), Mismatch> {
        todo!("infer::eq")
    }

    /// The join of two terms (types §2): a transparent side grows by union
    /// toward the other; an opaque side never grows; afterwards the two are
    /// `eq`. Function types join their effects, flows, calls and split.
    pub fn join(&mut self, a: Term, b: Term) -> Result<(), Mismatch> {
        todo!("infer::join")
    }

    /// A value entering a slot (types §3, §7): once the value's head is
    /// closed, the slot folds it by ⊔ along the cast tree, or the value
    /// converts by exactly one declared conversion.
    pub fn enter(&mut self, casts: &CastTree, value: Term, slot: Term) -> Result<Entered, Mismatch> {
        todo!("infer::enter")
    }

    /// The join of two effects (types §2).
    pub fn join_effects(
        &mut self,
        a: &EffectTerm<InferPhase>,
        b: &EffectTerm<InferPhase>,
    ) -> EffectTerm<InferPhase> {
        todo!("infer::join_effects")
    }
}
