use super::{Phase, TyId};

/// A type argument, with or without `#` (types §8). The mark stands only at
/// the argument's root, so an argument is one of these and nothing below it
/// carries a mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeArg<V: Phase> {
    /// Never specialized.
    Uniform(TyId<V>),
    /// `#τ`: always specializable.
    Specialized(TyId<V>),
    /// The mark not decided yet.
    Open(V::ReprVar, TyId<V>),
}

impl<V: Phase> TypeArg<V> {
    pub fn ty(&self) -> TyId<V> {
        match self {
            TypeArg::Uniform(ty) | TypeArg::Specialized(ty) | TypeArg::Open(_, ty) => *ty,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenTerm<V: Phase> {
    Known(usize),
    Var(V::LenVar),
}

acvus_utils::declare_local_id!(pub IdentityId);

/// The source a value of a type with an identity came from (types §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityTerm<V: Phase> {
    Known(IdentityId),
    Var(V::IdentityVar),
}
