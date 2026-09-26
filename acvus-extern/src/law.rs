//! The law a requirement names at its type (RFC-0070 rule 6): the checker
//! resolves an `InstanceOf<S, I, Rt, T, L>` only by an instance stating
//! `L`'s law, so a handler holding the instance holds the fact.

use acvus_mir::laws::NamedLaw;

mod sealed {
    pub trait Sealed {}
}

pub trait Law: sealed::Sealed + Send + Sync + 'static {
    const NAMED: Option<NamedLaw>;
}

pub enum Unnamed {}

/// A `core::eq` instance that is reflexive, symmetric and transitive, whose
/// type's `hash` instance hashes the values it calls equal alike (RFC-0082
/// rule 11).
pub enum Equivalence {}

impl sealed::Sealed for Unnamed {}
impl sealed::Sealed for Equivalence {}

impl Law for Unnamed {
    const NAMED: Option<NamedLaw> = None;
}

impl Law for Equivalence {
    const NAMED: Option<NamedLaw> = Some(NamedLaw::Equivalence);
}

/// `#[extern_fn]` bounds the signature of an instance stating
/// `law(equivalence)` by this trait, through `stated_over`.
#[diagnostic::on_unimplemented(
    message = "`law(equivalence)` is stated on a `core::eq` instance, and this is an instance of `{Self}`",
    label = "not `core::eq`",
    note = "an equivalence is a law of `eq` and the `hash` beside it (RFC-0082 rule 11)"
)]
pub trait EquivalenceOver: sealed::Sealed {}

impl<T, Rt> sealed::Sealed for crate::core::eq<T, Rt> {}
impl<T, Rt> EquivalenceOver for crate::core::eq<T, Rt> {}

#[doc(hidden)]
pub const fn stated_over<S>()
where
    S: EquivalenceOver,
{
}
