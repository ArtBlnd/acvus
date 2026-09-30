//! The value algebra: polynomials over symbolic atoms with integer
//! coefficients, `ℤ[a₁, …, aₙ]`. Every symbolic value the analyses name is an
//! element of this ring; the affine combinations are its elements of degree at
//! most one, and the scalar evolution builds its recurrences over it.
//!
//! The ring answers two questions only: which element an arithmetic expression
//! is, and whether two elements are equal. Each element has one normal form,
//! so equality is equality of normal forms. Sign and order are not the ring's;
//! a reader asks the interval domain for them.
//!
//! Integer arithmetic is exact here: the language makes overflow undefined, so
//! a value the program computes is its exact integer, and the ring computes
//! with that integer. A coefficient that leaves `i128`, or an element past the
//! size bound, is no element: the operation returns `None`, and the reader
//! takes the value as unknown. The bound keeps every operation polynomial in
//! the size of the program (compilation §2); without it a repeated product
//! grows exponentially.

use std::collections::BTreeMap;

/// The most terms an element holds, and the highest degree of any of its
/// monomials. An operation whose result exceeds either returns `None`.
pub const MAX_TERMS: usize = 64;
pub const MAX_DEGREE: u32 = 4;

/// A product of atoms, each raised to a positive power, sorted by atom with
/// each atom once. The empty product is the unit monomial `1`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Monomial<A>
where
    A: Ord,
{
    factors: Box<[(A, u32)]>,
}

impl<A> Monomial<A>
where
    A: Ord + Clone,
{
    pub fn unit() -> Self {
        todo!("the empty product")
    }

    pub fn atom(atom: A) -> Self {
        todo!("atom^1")
    }

    /// The sum of the powers; the unit has degree zero.
    pub fn degree(&self) -> u32 {
        todo!()
    }

    /// Each atom with its power, sorted by atom, no power zero.
    pub fn factors(&self) -> impl Iterator<Item = (&A, u32)> {
        self.factors.iter().map(|(atom, power)| (atom, *power))
    }

    /// `self · other`: powers of a shared atom add.
    pub fn mul(&self, other: &Self) -> Self {
        todo!("merge the sorted factors")
    }
}

/// `Σ cᵢ·mᵢ` over distinct monomials, no coefficient zero, ordered by
/// monomial: the normal form, so equal polynomials are equal values of this
/// type. The zero polynomial has no term.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Polynomial<A>
where
    A: Ord,
{
    terms: BTreeMap<Monomial<A>, i128>,
}

impl<A> Polynomial<A>
where
    A: Ord + Clone,
{
    pub fn zero() -> Self {
        todo!()
    }

    pub fn constant(constant: i128) -> Self {
        todo!("zero where the constant is zero")
    }

    pub fn atom(atom: A) -> Self {
        todo!()
    }

    /// The constant alone, where the polynomial has no other term.
    pub fn as_constant(&self) -> Option<i128> {
        todo!()
    }

    /// The highest degree of its monomials; the zero polynomial has degree 0.
    pub fn degree(&self) -> u32 {
        todo!()
    }

    /// Each monomial with its coefficient, in normal order, none zero.
    pub fn terms(&self) -> impl Iterator<Item = (&Monomial<A>, i128)> {
        self.terms.iter().map(|(monomial, coefficient)| (monomial, *coefficient))
    }

    /// A monomial absent from the normal form has coefficient zero.
    pub fn coefficient(&self, monomial: &Monomial<A>) -> i128 {
        self.terms.get(monomial).copied().unwrap_or(0)
    }

    /// Whether an atom occurs in any term.
    pub fn mentions(&self, atom: &A) -> bool {
        todo!()
    }

    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        todo!("None past i128 or the size bound")
    }

    pub fn checked_neg(&self) -> Option<Self> {
        todo!()
    }

    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        todo!()
    }

    pub fn checked_scale(&self, factor: i128) -> Option<Self> {
        todo!()
    }

    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        todo!("None past i128, MAX_TERMS or MAX_DEGREE")
    }

    /// The polynomial with each atom replaced by the polynomial `by` gives
    /// it; `None` where `by` gives none for an atom, or the result is no
    /// element.
    pub fn substitute<B, F>(&self, by: F) -> Option<Polynomial<B>>
    where
        B: Ord + Clone,
        F: FnMut(&A) -> Option<Polynomial<B>>,
    {
        todo!()
    }

    /// The polynomial as `base + slope·atom`, where `atom` occurs in it at
    /// most to the first power. `None` where `atom` occurs to a higher power.
    /// This is the view the scalar evolution reads a recurrence's step
    /// through.
    pub fn linear_in(&self, atom: &A) -> Option<Linear<A>> {
        todo!()
    }
}

/// A polynomial seen as `base + slope·x` in one atom `x`: neither part
/// mentions `x`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear<A>
where
    A: Ord,
{
    pub base: Polynomial<A>,
    pub slope: Polynomial<A>,
}
