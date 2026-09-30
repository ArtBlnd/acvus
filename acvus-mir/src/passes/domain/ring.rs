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
use std::collections::btree_map::Entry;

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
        Monomial { factors: Box::new([]) }
    }

    pub fn atom(atom: A) -> Self {
        Monomial { factors: Box::new([(atom, 1)]) }
    }

    /// The sum of the powers; the unit has degree zero.
    ///
    /// # Panics
    /// The sum leaves `u32`, as `mul` panics.
    pub fn degree(&self) -> u32 {
        self.factors
            .iter()
            .try_fold(0u32, |sum, (_, power)| sum.checked_add(*power))
            .expect("a monomial's degree leaves u32")
    }

    /// Each atom with its power, sorted by atom, no power zero.
    pub fn factors(&self) -> impl Iterator<Item = (&A, u32)> {
        self.factors.iter().map(|(atom, power)| (atom, *power))
    }

    /// `self · other`: powers of a shared atom add.
    ///
    /// # Panics
    /// A power leaves `u32`. A polynomial never asks for such a product,
    /// since its monomials stay within `MAX_DEGREE`.
    pub fn mul(&self, other: &Self) -> Self {
        let mut powers: BTreeMap<&A, u32> = self.factors().collect();
        for (atom, power) in other.factors() {
            let sum = powers.entry(atom).or_insert(0);
            *sum = sum.checked_add(power).expect("a monomial's power leaves u32");
        }
        Monomial { factors: powers.into_iter().map(|(atom, power)| (atom.clone(), power)).collect() }
    }

    fn power_of(&self, atom: &A) -> u32 {
        self.factors
            .binary_search_by(|(factor, _)| factor.cmp(atom))
            .map_or(0, |at| self.factors[at].1)
    }

    /// The monomial with `atom` removed, whatever its power.
    fn without(&self, atom: &A) -> Self {
        Monomial { factors: self.factors.iter().filter(|(factor, _)| factor != atom).cloned().collect() }
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
        Polynomial { terms: BTreeMap::new() }
    }

    pub fn constant(constant: i128) -> Self {
        Polynomial::from_terms([(Monomial::unit(), constant)])
    }

    pub fn atom(atom: A) -> Self {
        Polynomial::from_terms([(Monomial::atom(atom), 1)])
    }

    /// The constant alone, where the polynomial has no other term.
    pub fn as_constant(&self) -> Option<i128> {
        match self.terms.len() {
            0 => Some(0),
            1 => self.terms.get(&Monomial::unit()).copied(),
            _ => None,
        }
    }

    /// The highest degree of its monomials; the zero polynomial has degree 0.
    pub fn degree(&self) -> u32 {
        self.terms.keys().map(Monomial::degree).max().unwrap_or(0)
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
        self.terms.keys().any(|monomial| monomial.power_of(atom) > 0)
    }

    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        self.merge(other, i128::checked_add)
    }

    pub fn checked_neg(&self) -> Option<Self> {
        self.checked_scale(-1)
    }

    pub fn checked_sub(&self, other: &Self) -> Option<Self> {
        self.merge(other, i128::checked_sub)
    }

    pub fn checked_scale(&self, factor: i128) -> Option<Self> {
        let terms = self
            .terms
            .iter()
            .map(|(monomial, coefficient)| Some((monomial.clone(), coefficient.checked_mul(factor)?)))
            .collect::<Option<Vec<_>>>()?;
        Some(Polynomial::from_terms(terms))
    }

    /// Over the integers the degree of a product is the sum of its factors'
    /// degrees, so a product past `MAX_DEGREE` is refused before it is built.
    /// Each coefficient of the product sums its positive and its negative
    /// contributions apart, so whether it leaves `i128` does not depend on
    /// the order of the factors.
    pub fn checked_mul(&self, other: &Self) -> Option<Self> {
        if self.terms.is_empty() || other.terms.is_empty() {
            return Some(Polynomial::zero());
        }
        if self.degree() + other.degree() > MAX_DEGREE {
            return None;
        }
        let mut sums: BTreeMap<Monomial<A>, (i128, i128)> = BTreeMap::new();
        for (left, left_coefficient) in &self.terms {
            for (right, right_coefficient) in &other.terms {
                let product = left_coefficient.checked_mul(*right_coefficient)?;
                let (positive, negative) = sums.entry(left.mul(right)).or_insert((0, 0));
                if product > 0 {
                    *positive = positive.checked_add(product)?;
                } else {
                    *negative = negative.checked_add(product)?;
                }
            }
        }
        let terms = sums.into_iter().map(|(monomial, (positive, negative))| (monomial, positive + negative));
        Polynomial::bounded(Polynomial::from_terms(terms))
    }

    /// The polynomial with each atom replaced by the polynomial `by` gives
    /// it; `None` where `by` gives none for an atom, or the result is no
    /// element.
    pub fn substitute<B, F>(&self, mut by: F) -> Option<Polynomial<B>>
    where
        B: Ord + Clone,
        F: FnMut(&A) -> Option<Polynomial<B>>,
    {
        let mut images: BTreeMap<&A, Polynomial<B>> = BTreeMap::new();
        let mut out = Polynomial::zero();
        for (monomial, coefficient) in self.terms() {
            let mut term = Polynomial::constant(coefficient);
            for (atom, power) in monomial.factors() {
                let image = match images.entry(atom) {
                    Entry::Occupied(known) => known.into_mut(),
                    Entry::Vacant(unknown) => unknown.insert(by(atom)?),
                };
                for _ in 0..power {
                    term = term.checked_mul(image)?;
                }
            }
            out = out.checked_add(&term)?;
        }
        Some(out)
    }

    /// The polynomial as `base + slope·atom`, where `atom` occurs in it at
    /// most to the first power. `None` where `atom` occurs to a higher power.
    /// This is the view the scalar evolution reads a recurrence's step
    /// through.
    pub fn linear_in(&self, atom: &A) -> Option<Linear<A>> {
        let mut base = BTreeMap::new();
        let mut slope = BTreeMap::new();
        for (monomial, coefficient) in self.terms() {
            match monomial.power_of(atom) {
                0 => base.insert(monomial.clone(), coefficient),
                1 => slope.insert(monomial.without(atom), coefficient),
                _ => return None,
            };
        }
        Some(Linear { base: Polynomial { terms: base }, slope: Polynomial { terms: slope } })
    }

    /// The normal form of `Σ cᵢ·mᵢ` over distinct monomials: zero
    /// coefficients dropped.
    fn from_terms<I>(terms: I) -> Self
    where
        I: IntoIterator<Item = (Monomial<A>, i128)>,
    {
        Polynomial { terms: terms.into_iter().filter(|(_, coefficient)| *coefficient != 0).collect() }
    }

    /// The polynomial, where it is within `MAX_TERMS` and `MAX_DEGREE`.
    fn bounded(self) -> Option<Self> {
        (self.terms.len() <= MAX_TERMS && self.degree() <= MAX_DEGREE).then_some(self)
    }

    /// Each monomial's coefficient `combine(mine, theirs)`, absent ones
    /// zero; `None` where a coefficient leaves `i128` or the result is past
    /// the size bound.
    fn merge<F>(&self, other: &Self, combine: F) -> Option<Self>
    where
        F: Fn(i128, i128) -> Option<i128>,
    {
        let mut terms = self.terms.clone();
        for (monomial, theirs) in other.terms() {
            let mine = self.coefficient(monomial);
            terms.insert(monomial.clone(), combine(mine, theirs)?);
        }
        Polynomial::bounded(Polynomial::from_terms(terms))
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

#[cfg(test)]
mod tests {
    use super::*;

    type P = Polynomial<u8>;

    fn x(atom: u8) -> P {
        Polynomial::atom(atom)
    }

    fn k(constant: i128) -> P {
        Polynomial::constant(constant)
    }

    fn add(a: &P, b: &P) -> P {
        a.checked_add(b).unwrap()
    }

    fn mul(a: &P, b: &P) -> P {
        a.checked_mul(b).unwrap()
    }

    /// Elements over the atoms 0, 1 and 2 of degree at most two, so any
    /// product of two stays within `MAX_DEGREE`.
    fn samples() -> Vec<P> {
        vec![
            P::zero(),
            k(1),
            k(-3),
            x(0),
            x(1),
            add(&x(0), &x(1)),
            add(&mul(&k(2), &mul(&x(0), &x(1))), &k(-5)),
            add(&mul(&x(0), &x(0)), &x(2)),
            x(0).checked_sub(&mul(&k(4), &x(2))).unwrap(),
        ]
    }

    /// The integer `p` is where atom `i` is `env[i]`.
    fn evaluate(p: &P, env: &[i128; 3]) -> i128 {
        p.terms()
            .map(|(monomial, coefficient)| {
                monomial.factors().fold(coefficient, |product, (atom, power)| product * env[usize::from(*atom)].pow(power))
            })
            .sum()
    }

    fn environments() -> Vec<[i128; 3]> {
        const VALUES: [i128; 4] = [-2, 0, 1, 3];
        VALUES.into_iter().flat_map(|a| VALUES.into_iter().flat_map(move |b| VALUES.into_iter().map(move |c| [a, b, c]))).collect()
    }

    /// The normal form's invariant, seen through the public interface: no
    /// zero coefficient, monomials strictly ascending, factors strictly
    /// ascending with positive powers.
    fn assert_normal(p: &P) {
        let monomials: Vec<&Monomial<u8>> = p.terms().map(|(monomial, coefficient)| {
            assert_ne!(coefficient, 0, "{p:?}");
            monomial
        }).collect();
        assert!(monomials.windows(2).all(|pair| pair[0] < pair[1]), "{p:?}");
        for monomial in monomials {
            let factors: Vec<(&u8, u32)> = monomial.factors().collect();
            assert!(factors.windows(2).all(|pair| pair[0].0 < pair[1].0), "{monomial:?}");
            assert!(factors.iter().all(|(_, power)| *power > 0), "{monomial:?}");
        }
    }

    #[test]
    fn addition_is_associative_and_commutative() {
        let all = samples();
        for a in &all {
            for b in &all {
                assert_eq!(a.checked_add(b), b.checked_add(a));
                for c in &all {
                    assert_eq!(add(&add(a, b), c), add(a, &add(b, c)));
                }
            }
        }
    }

    #[test]
    fn multiplication_is_associative_and_commutative() {
        let all: Vec<P> = samples().into_iter().filter(|p| p.degree() <= 1).collect();
        for a in &all {
            for b in &all {
                assert_eq!(a.checked_mul(b), b.checked_mul(a));
                for c in &all {
                    assert_eq!(mul(&mul(a, b), c), mul(a, &mul(b, c)));
                }
            }
        }
    }

    #[test]
    fn multiplication_distributes_over_addition() {
        let all = samples();
        for a in &all {
            for b in &all {
                for c in &all {
                    assert_eq!(mul(a, &add(b, c)), add(&mul(a, b), &mul(a, c)));
                }
            }
        }
    }

    #[test]
    fn zero_and_one_are_the_units() {
        for a in &samples() {
            assert_eq!(add(a, &P::zero()), *a);
            assert_eq!(mul(a, &k(1)), *a);
            assert_eq!(mul(a, &P::zero()), P::zero());
            assert_eq!(a.checked_scale(1).unwrap(), *a);
            assert_eq!(a.checked_scale(0).unwrap(), P::zero());
        }
        assert_eq!(k(0), P::zero());
        assert_eq!(P::zero().as_constant(), Some(0));
    }

    #[test]
    fn an_element_minus_itself_is_zero_and_negation_is_an_involution() {
        for a in &samples() {
            assert_eq!(a.checked_sub(a).unwrap(), P::zero());
            assert_eq!(a.checked_neg().unwrap().checked_neg().unwrap(), *a);
            assert_eq!(add(a, &a.checked_neg().unwrap()), P::zero());
        }
    }

    /// The ring's operations are the integers' at every choice of the atoms,
    /// and every result is in normal form.
    #[test]
    fn every_operation_agrees_with_evaluation() {
        let all = samples();
        let envs = environments();
        for a in &all {
            let negated = a.checked_neg().unwrap();
            let scaled = a.checked_scale(-7).unwrap();
            assert_normal(&negated);
            assert_normal(&scaled);
            for env in &envs {
                assert_eq!(evaluate(&negated, env), -evaluate(a, env));
                assert_eq!(evaluate(&scaled, env), -7 * evaluate(a, env));
            }
            for b in &all {
                let (sum, difference, product) = (add(a, b), a.checked_sub(b).unwrap(), mul(a, b));
                for p in [&sum, &difference, &product] {
                    assert_normal(p);
                }
                for env in &envs {
                    let (va, vb) = (evaluate(a, env), evaluate(b, env));
                    assert_eq!(evaluate(&sum, env), va + vb);
                    assert_eq!(evaluate(&difference, env), va - vb);
                    assert_eq!(evaluate(&product, env), va * vb, "{a:?} · {b:?}");
                }
            }
        }
    }

    #[test]
    fn equal_polynomials_built_in_any_order_are_equal_values() {
        let (a, b) = (x(0), x(1));
        let squared = mul(&add(&a, &b), &add(&a, &b));
        let expanded = add(&add(&mul(&a, &a), &mul(&k(2), &mul(&a, &b))), &mul(&b, &b));
        let reordered = add(&add(&add(&mul(&b, &b), &mul(&b, &a)), &mul(&a, &b)), &mul(&a, &a));
        let cancelled = add(&add(&squared, &x(2)), &x(2).checked_neg().unwrap());
        assert_eq!(squared, expanded);
        assert_eq!(squared, reordered);
        assert_eq!(squared, cancelled);
        assert_eq!(Monomial::atom(1u8).mul(&Monomial::atom(0)), Monomial::atom(0).mul(&Monomial::atom(1)));
        assert_eq!(mul(&k(3), &k(4)).as_constant(), Some(12));
        assert_eq!(add(&a, &k(1)).as_constant(), None);
        assert_eq!(squared.degree(), 2);
        assert_eq!(squared.coefficient(&Monomial::atom(0).mul(&Monomial::atom(1))), 2);
        assert!(squared.mentions(&1) && !squared.mentions(&2));
    }

    #[test]
    fn a_product_past_max_degree_is_no_element() {
        let at_bound = (1..MAX_DEGREE).fold(x(0), |p, _| mul(&p, &x(0)));
        assert_eq!(at_bound.degree(), MAX_DEGREE);
        assert_eq!(at_bound.checked_mul(&x(1)), None);
        assert_eq!(at_bound.checked_mul(&add(&x(1), &k(1))), None);
        assert_eq!(at_bound.checked_mul(&k(2)).map(|p| p.degree()), Some(MAX_DEGREE));
        assert_eq!(at_bound.substitute(|_| Some(add(&x(0), &x(1)))).map(|p| p.degree()), Some(MAX_DEGREE));
        assert_eq!(at_bound.substitute(|_| Some(mul(&x(0), &x(1)))), None);
    }

    #[test]
    fn a_result_past_max_terms_is_no_element() {
        let atoms: Vec<Polynomial<u32>> = (0..).take(MAX_TERMS + 1).map(Polynomial::atom).collect();
        let at_bound = atoms[..MAX_TERMS].iter().fold(Polynomial::zero(), |p, a| p.checked_add(a).unwrap());
        assert_eq!(at_bound.terms().count(), MAX_TERMS);
        assert_eq!(at_bound.checked_add(&atoms[MAX_TERMS]), None);
        assert!(at_bound.checked_add(&atoms[0]).is_some(), "a shared monomial adds no term");
        let sum = atoms[..9].iter().fold(Polynomial::zero(), |p, a| p.checked_add(a).unwrap());
        let square = sum.checked_mul(&sum).unwrap();
        assert_eq!(square.terms().count(), 45);
        assert_eq!(square.checked_mul(&sum), None, "the cube has 165 monomials");
    }

    #[test]
    fn a_coefficient_leaving_i128_is_no_element() {
        let big = k(i128::MAX);
        assert_eq!(big.checked_add(&k(1)), None);
        assert_eq!(k(i128::MIN).checked_neg(), None);
        assert_eq!(k(i128::MIN).checked_sub(&k(1)), None);
        assert_eq!(big.checked_scale(2), None);
        assert_eq!(add(&x(0), &big).checked_mul(&k(2)), None);
        assert_eq!(k(i128::MIN).checked_sub(&k(-1)), Some(k(i128::MIN + 1)), "a difference that fits is exact");
        let half = k(1 << 100);
        assert_eq!(half.checked_mul(&half), None);
        assert_eq!(x(0).substitute(|_| Some(big.clone())).unwrap(), big);
        assert_eq!(mul(&k(2), &x(0)).substitute(|_| Some(big.clone())), None);
    }

    /// The monomial `a₀a₁a₂` of the product gathers `MAX`, `MAX` and
    /// `-MAX`: summed in one order it passes `i128` on the way, in the other
    /// it does not.
    #[test]
    fn whether_a_product_leaves_i128_does_not_depend_on_the_order_of_its_factors() {
        let p = add(&add(&x(0), &x(1)), &x(2));
        let q = [(1, 2, i128::MAX), (0, 2, i128::MAX), (0, 1, -i128::MAX)]
            .into_iter()
            .fold(P::zero(), |q, (a, b, c)| add(&q, &mul(&k(c), &mul(&x(a), &x(b)))));
        assert_eq!(p.checked_mul(&q), q.checked_mul(&p));
    }

    #[test]
    fn substituting_then_evaluating_is_evaluating_the_substitution() {
        let all = samples();
        let images: Vec<P> = all.iter().filter(|p| p.degree() <= 1).cloned().collect();
        for p in all.iter().filter(|p| p.degree() <= 2) {
            for (at, first) in images.iter().enumerate() {
                let second = &images[(at + 3) % images.len()];
                let third = &images[(at + 5) % images.len()];
                let by = [first, second, third];
                let substituted = p.substitute(|atom| Some(by[usize::from(*atom)].clone())).unwrap();
                assert_normal(&substituted);
                for env in &environments() {
                    let inner = [evaluate(first, env), evaluate(second, env), evaluate(third, env)];
                    assert_eq!(evaluate(&substituted, env), evaluate(p, &inner), "{p:?} at {by:?}");
                }
            }
        }
    }

    #[test]
    fn substitute_gives_none_where_by_gives_none() {
        let p = add(&x(0), &x(1));
        assert_eq!(p.substitute(|atom| (*atom == 0).then(|| k(1))), None);
        assert_eq!(k(4).substitute(|_: &u8| -> Option<Polynomial<u8>> { None }), Some(k(4)));
    }

    #[test]
    fn linear_in_rebuilds_the_polynomial() {
        for p in &samples() {
            for atom in 0..3u8 {
                let Some(Linear { base, slope }) = p.linear_in(&atom) else {
                    continue;
                };
                assert!(!base.mentions(&atom) && !slope.mentions(&atom), "{p:?} in {atom}");
                assert_eq!(add(&base, &mul(&slope, &x(atom))), *p);
            }
        }
        let p = add(&mul(&k(2), &mul(&x(0), &x(1))), &add(&x(1), &k(-5)));
        let Linear { base, slope } = p.linear_in(&1).unwrap();
        assert_eq!(base, k(-5));
        assert_eq!(slope, add(&mul(&k(2), &x(0)), &k(1)));
    }

    #[test]
    fn linear_in_is_none_where_the_atom_occurs_squared() {
        let p = add(&mul(&x(0), &x(0)), &x(2));
        assert_eq!(p.linear_in(&0), None);
        assert_eq!(p.linear_in(&2), Some(Linear { base: mul(&x(0), &x(0)), slope: k(1) }));
        assert_eq!(mul(&x(1), &mul(&x(0), &x(1))).linear_in(&1), None);
    }
}
