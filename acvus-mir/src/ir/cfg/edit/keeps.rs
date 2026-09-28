//! Invariants an editor keeps, one checker each. They run after every edit
//! in debug and test builds; `validate` reuses them after each transform.

use crate::ir::cfg::Body;
use crate::ty::{Concrete, Types};

/// A breach of an invariant: a compiler defect, never a refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub invariant: &'static str,
    pub detail: String,
}

pub trait Invariant {
    const NAME: &'static str;
    fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation>;
}

/// A set of invariants, as a tuple.
pub trait Invariants {
    fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation>;
}

impl Invariants for () {
    fn check(_: &Body, _: &Types<Concrete>) -> Result<(), Violation> {
        Ok(())
    }
}

macro_rules! tuple_invariants {
    ($($name:ident),+) => {
        impl<$($name),+> Invariants for ($($name,)+)
        where
            $($name: Invariants,)+
        {
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                $($name::check(body, types)?;)+
                Ok(())
            }
        }
    };
}
tuple_invariants!(A, B);
tuple_invariants!(A, B, C, D);

macro_rules! invariant {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        pub struct $name;
        impl Invariant for $name {
            const NAME: &'static str = stringify!($name);
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                todo!(concat!("ir::cfg::edit::keeps::", stringify!($name)))
            }
        }
        impl Invariants for $name {
            fn check(body: &Body, types: &Types<Concrete>) -> Result<(), Violation> {
                <$name as Invariant>::check(body, types)
            }
        }
    };
}

invariant!(TypesAgree, "Every operand's type is what its instruction takes, and every edge's arguments agree with its target's parameters.");
invariant!(DefDominatesUses, "Every value's definition dominates each of its uses.");
invariant!(OrderChainWhole, "Every effectful instruction consumes exactly one live token, and the chain reaches the body's return (effects §2-3).");
invariant!(StructureKept, "Every `Diamond`'s join and every loop's stages and exit are blocks of the body (D45).");
invariant!(SplitTakenOnce, "Every start of a split call is taken exactly once on every path from it (effects §6).");
