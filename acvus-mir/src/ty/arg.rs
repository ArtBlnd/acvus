use super::Phase;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LenTerm<V: Phase> {
    Known(usize),
    Var(V::LenVar),
}

/// The mark `#` of an extern-defined type (types §8), ordered marked
/// below unmarked: two marks that meet join at the higher, as effects do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mark {
    Marked,
    Unmarked,
}

impl Mark {
    pub fn join(&self, other: &Mark) -> Mark {
        (*self).max(*other)
    }
}

/// A mark as a term: known, or, while solving, its variable, a member of
/// its type known once nothing can meet in it (types §6, §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkTerm<V: Phase> {
    Known(Mark),
    Var(V::MarkVar),
}

acvus_utils::declare_local_id!(pub IdentityId);

/// The source a value of a type with an identity came from (types §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityTerm<V: Phase> {
    Known(IdentityId),
    Var(V::IdentityVar),
}
