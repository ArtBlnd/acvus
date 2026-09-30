use acvus_utils::QualifiedRef;

use super::TyVarBound;

/// An extern-defined type (extern-fn §8), opaque (rfc_extern types §2):
/// the single source of its parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserTypeDecl {
    pub id: QualifiedRef,
    /// One bound per type parameter.
    pub type_params: Box<[TyVarBound]>,
    /// Whether it carries an effect (extern-fn §8): an upper bound on the
    /// work its values hold, which runs later.
    pub carries_effect: bool,
    /// Whether it carries an identity, at most one (types §3).
    pub carries_identity: bool,
    /// The lifetimes it declares, each a position of its values (ownership §5).
    pub lifetimes: usize,
}
