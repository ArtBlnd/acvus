use super::Phase;

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
