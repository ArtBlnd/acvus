use acvus_utils::QualifiedRef;

use crate::ty::{Poly, TyId};

/// A declared conversion (extern-fn §7): `from` converts to `to` through
/// `via`, the two patterns sharing their placeholders. It is no step of the
/// cast tree, and a type may have several (types §7).
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionRule {
    pub from: TyId<Poly>,
    pub to: TyId<Poly>,
    pub via: QualifiedRef,
}

/// A conversion `verify` admitted. Only `verify` builds one.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedConversion(pub(super) ConversionRule);

impl VerifiedConversion {
    pub fn rule(&self) -> &ConversionRule {
        &self.0
    }
}
