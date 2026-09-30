//! The facts no type can carry (runtime contract §4, safety §2).

/// S1: storage a shared guard lends keeps its address, and is written only
/// under an exclusive loan, until the checker's loan ends, after the guard
/// drops as well.
///
/// # Safety
/// The implementing runtime keeps that storage so.
pub unsafe trait SignedS1 {}

/// S2: a value made as a reference to a target names that target, or its
/// written-back copy, until the checker's loan ends.
///
/// # Safety
/// The implementing runtime keeps every reference value so.
pub unsafe trait SignedS2 {}

/// S3: storage lent in place to another run, and a kept value, are released
/// only after the last work that may read them has ended or been dropped.
///
/// # Safety
/// The implementing runtime releases no such storage earlier.
pub unsafe trait SignedS3 {}

/// The layout identity a type has with a related Rust type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutFact {
    /// The type is its own canonical form.
    OwnCanonical,
    /// The type's layout is its canonical form's.
    CanonicalOf,
}

/// A generic parameter the checker's flows track, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenericParam {
    Lifetime,
    Type,
}

/// A type's structural facts (safety §2): its layout identity (L), that its
/// layout and carriers reach its type parameters only by holding them (P),
/// that no part a lifetime or type parameter reaches has interior mutability
/// (M), and its generic parameters (G).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuralFacts {
    pub layout: LayoutFact,
    pub generics: &'static [GenericParam],
}

/// A type's structural facts, signed at the type by the derive or by this
/// crate for the types it defines.
///
/// # Safety
/// `Canon` is `Self` with every uniform index fixed and every lifetime
/// `'static`, and has `Self`'s size, alignment, field offsets and release;
/// `Self` reaches its type parameters only by holding them, with no interior
/// mutability over a lifetime or type parameter; `FACTS` states it.
pub unsafe trait Structural {
    type Canon: Structural<Canon = Self::Canon> + 'static;
    const FACTS: StructuralFacts;
}
