use acvus_utils::QualifiedRef;

use super::TyVarBound;

/// An extern-defined type (extern-fn §8): the single source of its
/// parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserTypeDecl {
    pub id: QualifiedRef,
    /// One bound per type parameter.
    pub type_params: Box<[TyVarBound]>,
    /// Per type parameter, whether it may carry `#` (types §8).
    pub specializes: Box<[bool]>,
    pub effect_params: usize,
    /// At most one (`Refused::TwoIdentities`): a value of a type with one is
    /// a distinct source (types §3).
    pub identity_params: usize,
    /// The lifetimes it declares, each a position of its values (ownership §5).
    pub lifetimes: usize,
}
