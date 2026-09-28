use crate::ty::{IntTy, Poly, TyId};

/// The one bound a declaration's type parameter carries (extern-fn §1).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyVarBound {
    /// Any type.
    Any,
    /// One of these shapes.
    OneOf(Box<[TyId<Poly>]>),
    /// An integer type among these widths, signed only where `signed`.
    Integer { signed: bool, among: Box<[IntTy]> },
    /// Any type the use must settle; never taken as `!` (types §5).
    Settled,
}
