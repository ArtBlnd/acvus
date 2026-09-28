use acvus_utils::QualifiedRef;

use super::casts::Head;
use super::{Declarations, RawDeclarations};

/// Why a set of declarations is not taken in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// A declaration carries an array type (extern-fn §1).
    ArrayType { declaration: QualifiedRef },
    /// Two declared steps up from one head (types §3).
    TwoStepsUp { head: Head, via: [QualifiedRef; 2] },
    /// The steps up from `head` return to it (types §3).
    CastCycle { head: Head },
    /// A declared cast leaves an identity-carrying head for another head
    /// before reaching its form without identity (types §3).
    IdentityNotFirst { via: QualifiedRef },
    /// A required instance's types are not smaller than the declaration's
    /// own (extern-fn §6).
    RequirementNotSmaller { declaration: QualifiedRef, signature: QualifiedRef },
    /// Two types under one name.
    DuplicateType { id: QualifiedRef },
    /// A type with more than one identity parameter. Not stated in
    /// rfc_compiler; carried from the registry's rule until it is.
    TwoIdentities { id: QualifiedRef },
}

/// Build the verified set, or every refusal the set earns.
pub(super) fn verify(raw: RawDeclarations) -> Result<Declarations, Vec<Refused>> {
    todo!("decl::verify: {} functions", raw.functions.len())
}
