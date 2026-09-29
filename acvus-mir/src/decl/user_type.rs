use acvus_utils::{Astr, QualifiedRef};

use crate::ty::{Poly, TyId};

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
    /// Whether it carries an effect (extern-fn §8): an upper bound on the
    /// work its values hold, which runs later.
    pub carries_effect: bool,
    /// Whether it carries an identity, at most one (types §3).
    pub carries_identity: bool,
    /// The lifetimes it declares, each a position of its values (ownership §5).
    pub lifetimes: usize,
    /// How much of it a program sees (rfc_extern types).
    pub kind: UserTypeKind,
}

/// How much of an extern-defined type a program sees (rfc_extern types).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserTypeKind {
    /// A derived struct: its declared fields are read, and written at
    /// their declared types, which widens nothing (rfc_extern types §1).
    Derived { fields: Box<[(Astr, TyId<Poly>)]> },
    /// A closed enum: a script path naming one of its variants builds a
    /// value of it, exactly as declared (rfc_extern types §2).
    Enum { variants: Box<[(Astr, Option<TyId<Poly>>)]> },
    /// Known only by its name and the extern fns that take or give it
    /// (rfc_extern types §3).
    Opaque,
}
