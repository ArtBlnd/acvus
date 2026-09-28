use crate::ty::IntTy;

/// The one bound a declaration's type parameter carries (extern-fn §1).
/// Which shapes a function takes is said by its impls (extern-fn §6),
/// not by a bound.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TyVarBound {
    /// Any type.
    Any,
    /// An integer type among these widths, signed only where `signed`.
    Integer { signed: bool, among: Box<[IntTy]> },
    /// Any type the use must settle; never taken as `!` (types §5).
    Settled,
}
