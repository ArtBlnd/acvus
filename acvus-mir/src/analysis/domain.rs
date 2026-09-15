//! The fixpoint algebra for dataflow analysis.

/// Join-semilattice with bottom. The algebra that dataflow fixpoints require.
///
/// Laws:
/// - `join(x, bottom) = x`  (bottom is identity)
/// - `join(x, x) = x`       (idempotent)
/// - `join(x, y) = join(y, x)` (commutative)
/// - `join(x, join(y, z)) = join(join(x, y), z)` (associative)
pub trait SemiLattice: Clone + PartialEq {
    fn bottom() -> Self;

    /// Least upper bound. Mutates self to `join(self, other)`.
    /// Returns true if self changed.
    fn join_mut(&mut self, other: &Self) -> bool;
}
